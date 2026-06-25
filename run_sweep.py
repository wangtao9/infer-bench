"""渐进并发扫描脚本。

对 vLLM、SGLang、Transformers 三个引擎执行渐进并发测试，
从 start → stop 按倍数递增并发级别，采集各级别的 TTFT/TPS/VRAM。
"""

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.client import concurrent_stream_requests, wait_for_server
from common.config import load_config
from common.gpu import GPUMonitor
from common.metrics import BenchmarkResult, compute_percentile_stats, make_run_id, results_to_csv
from common.prompts import generate_batch_prompts
from run_transformers import batch_generate

logger = logging.getLogger("run_sweep")

# dtype 字符串 → torch dtype 映射
DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


# ============================================================
# 并发级别生成
# ============================================================


def generate_sweep_concurrency(start: int, stop: int, multiplier: int) -> list[int]:
    """生成渐进并发级别列表。

    从 start 开始，每次乘以 multiplier，直到达到或超过 stop。

    Args:
        start: 起始并发数。
        stop: 最大并发数。
        multiplier: 乘数因子。

    Returns:
        并发级别列表，例如 [1, 2, 4, 8, 16, 32]。
    """
    levels = []
    current = start
    while current <= stop:
        levels.append(current)
        current *= multiplier
    # 确保 stop 包含在内
    if not levels or levels[-1] < stop:
        levels.append(stop)
    return levels


# ============================================================
# 服务器启停（通用）
# ============================================================


def start_server(engine: str, cfg) -> subprocess.Popen:
    """启动 vLLM 或 SGLang 服务器。

    Args:
        engine: 引擎名称，"vllm" 或 "sglang"。
        cfg: BenchmarkConfig 实例。

    Returns:
        subprocess.Popen 对象。

    Raises:
        ValueError: 不支持的引擎名称。
    """
    model = cfg.model.path or cfg.model.name
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cfg.gpu.device

    if engine == "vllm":
        port = cfg.engines.vllm.port
        extra_args = cfg.engines.vllm.extra_args
        cmd = (
            f"vllm serve {model} "
            f"--port {port} "
            f"--dtype auto "
            f"{extra_args}"
        )
    elif engine == "sglang":
        port = cfg.engines.sglang.port
        extra_args = cfg.engines.sglang.extra_args
        cmd = (
            f"sglang serve "
            f"--model-path {model} "
            f"--port {port} "
            f"--enable-metrics "
            f"{extra_args}"
        )
    else:
        raise ValueError(f"Unsupported engine for server start: {engine}. Must be 'vllm' or 'sglang'.")

    logger.info("Starting %s server: %s", engine, cmd)
    proc = subprocess.Popen(
        cmd,
        shell=True,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        start_new_session=True,  # 新进程组，方便 kill 整个进程树
    )
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    """停止服务器进程及其所有子进程。"""
    import signal

    logger.info("Stopping server (PID %d)...", proc.pid)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        logger.warning("Permission denied when killing process group, falling back to terminate")
        proc.terminate()

    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        logger.warning("Server did not terminate in 15s, force killing...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait(timeout=5)
    logger.info("Server stopped.")


# ============================================================
# HTTP 引擎扫描
# ============================================================


async def sweep_http_engine(engine: str, cfg, run_id: str) -> list[BenchmarkResult]:
    """对 HTTP 引擎（vLLM/SGLang）执行渐进并发扫描。

    启动服务器 → 等待就绪 → 逐级提升并发 → 停止服务器。

    Args:
        engine: 引擎名称，"vllm" 或 "sglang"。
        cfg: BenchmarkConfig 实例。
        run_id: 运行标识符。

    Returns:
        list[BenchmarkResult]: 各并发级别的测试结果。
    """
    model = cfg.model.path or cfg.model.name
    sw_cfg = cfg.test.sweep
    port = cfg.engines.vllm.port if engine == "vllm" else cfg.engines.sglang.port
    base_url = f"http://localhost:{port}"

    # Start GPU monitor before server — baseline = idle GPU memory.
    gpu_monitor = GPUMonitor(
        device_index=int(cfg.gpu.device.split(",")[0]),
        interval_ms=cfg.gpu.monitor_interval_ms,
    )
    gpu_monitor.start(reset_baseline=True)

    results = []
    proc = start_server(engine, cfg)

    try:
        # Wait for server
        ready = await wait_for_server(base_url, timeout=300)
        if not ready:
            logger.error("%s server failed to start, aborting sweep.", engine)
            return results

        # Stabilization
        logger.info("%s server ready. Waiting 10s for stabilization...", engine)
        await asyncio.sleep(10)

        # Load tokenizer
        logger.info("Loading tokenizer: %s", model)
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        # Sweep concurrency levels
        concurrency_levels = generate_sweep_concurrency(sw_cfg.start, sw_cfg.stop, sw_cfg.multiplier)
        logger.info("%s sweep concurrency levels: %s", engine, concurrency_levels)

        for concurrency in concurrency_levels:
            logger.info(
                "[%s sweep] concurrency=%d, prompt_length=%d, max_new_tokens=%d",
                engine,
                concurrency,
                sw_cfg.prompt_length,
                sw_cfg.max_new_tokens,
            )

            try:
                prompts = generate_batch_prompts(
                    concurrency, sw_cfg.prompt_length, tokenizer=tokenizer
                )

                gpu_monitor.start(reset_baseline=False)
                res = await concurrent_stream_requests(
                    base_url, prompts, model, max_tokens=sw_cfg.max_new_tokens,
                )
                gpu_monitor.stop()

                # 逐请求数据计算百分位统计（修复之前 mean-of-means 问题）
                ttft_stats = compute_percentile_stats(res["all_ttfts_ms"])
                itl_stats = compute_percentile_stats(res["all_itls_ms"])
                tpot_stats = compute_percentile_stats(res["all_tpots_ms"])

                results.append(
                    BenchmarkResult(
                        engine=engine,
                        test_type="sweep",
                        num_requests=concurrency,
                        request_rate=float("inf"),
                        prompt_tokens=sw_cfg.prompt_length,
                        max_new_tokens=sw_cfg.max_new_tokens,
                        ttft_ms=round(ttft_stats["mean"], 2),
                        median_ttft_ms=round(ttft_stats["median"], 2),
                        p99_ttft_ms=round(ttft_stats["p99"], 2),
                        mean_tps=round(res["concurrent_tps"], 2),
                        mean_itl_ms=round(itl_stats["mean"], 2),
                        median_itl_ms=round(itl_stats["median"], 2),
                        p99_itl_ms=round(itl_stats["p99"], 2),
                        mean_tpot_ms=round(tpot_stats["mean"], 2),
                        median_tpot_ms=round(tpot_stats["median"], 2),
                        p99_tpot_ms=round(tpot_stats["p99"], 2),
                        peak_vram_mb=round(gpu_monitor.peak_vram_mb, 1),
                        peak_vram_abs_mb=round(gpu_monitor.peak_vram_abs_mb, 1),
                        run_id=run_id,
                        timestamp=datetime.now().isoformat(),
                    )
                )
                logger.info(
                    "[%s sweep] concurrency=%d => ttft=%.2f ms (p99=%.2f), tps=%.2f tok/s, itl=%.2f ms (p99=%.2f), tpot=%.2f ms",
                    engine,
                    concurrency,
                    ttft_stats["mean"], ttft_stats["p99"],
                    res["concurrent_tps"],
                    itl_stats["mean"], itl_stats["p99"],
                    tpot_stats["mean"],
                )

            except Exception as e:
                logger.error(
                    "[%s sweep] concurrency=%d failed: %s", engine, concurrency, e
                )
                gpu_monitor.stop()
                results.append(
                    BenchmarkResult(
                        engine=engine,
                        test_type="sweep",
                        num_requests=concurrency,
                        request_rate=float("inf"),
                        prompt_tokens=sw_cfg.prompt_length,
                        max_new_tokens=sw_cfg.max_new_tokens,
                        ttft_ms=-1,
                        median_ttft_ms=-1.0,
                        p99_ttft_ms=-1.0,
                        mean_tps=-1,
                        mean_itl_ms=-1.0,
                        median_itl_ms=-1.0,
                        p99_itl_ms=-1.0,
                        mean_tpot_ms=-1.0,
                        median_tpot_ms=-1.0,
                        p99_tpot_ms=-1.0,
                        peak_vram_mb=-1,
                        peak_vram_abs_mb=-1,
                        run_id=run_id,
                        timestamp=datetime.now().isoformat(),
                    )
                )

            # Brief pause between levels
            await asyncio.sleep(3)

    finally:
        stop_server(proc)

    return results


# ============================================================
# Transformers 引擎扫描
# ============================================================


def sweep_transformers(cfg, run_id: str) -> list[BenchmarkResult]:
    """对 Transformers 引擎执行渐进并发扫描。

    在进程内加载模型，逐级提升批量大小进行批量生成。
    在高并发级别可能因 OOM 失败，此时记录 -1 值并继续。

    Args:
        cfg: BenchmarkConfig 实例。
        run_id: 运行标识符。

    Returns:
        list[BenchmarkResult]: 各并发级别的测试结果。
    """
    model_name = cfg.model.path or cfg.model.name
    sw_cfg = cfg.test.sweep

    # Start GPU monitor BEFORE loading model — baseline = idle GPU memory.
    gpu_monitor = GPUMonitor(
        device_index=int(cfg.gpu.device.split(",")[0]),
        interval_ms=cfg.gpu.monitor_interval_ms,
    )
    gpu_monitor.start(reset_baseline=True)

    # Load model (this allocates most GPU memory)
    dtype_str = cfg.engines.transformers.dtype
    device_map = cfg.engines.transformers.device_map
    torch_dtype = DTYPE_MAP.get(dtype_str)
    if torch_dtype is None:
        raise ValueError(
            f"Unsupported dtype: {dtype_str}. Must be one of {list(DTYPE_MAP.keys())}"
        )

    logger.info("Loading model: %s (dtype=%s, device_map=%s)", model_name, dtype_str, device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded successfully.")

    # Stop initial sampling; baseline is now locked.
    gpu_monitor.stop()

    # Sweep concurrency levels
    concurrency_levels = generate_sweep_concurrency(sw_cfg.start, sw_cfg.stop, sw_cfg.multiplier)
    logger.info("transformers sweep concurrency levels: %s", concurrency_levels)

    results = []
    for concurrency in concurrency_levels:
        logger.info(
            "[transformers sweep] concurrency=%d, prompt_length=%d, max_new_tokens=%d",
            concurrency,
            sw_cfg.prompt_length,
            sw_cfg.max_new_tokens,
        )

        try:
            prompts = generate_batch_prompts(
                concurrency, sw_cfg.prompt_length, tokenizer=tokenizer
            )

            batch_size = cfg.engines.transformers.batch_size
            gpu_monitor.start(reset_baseline=False)
            res = batch_generate(model, tokenizer, prompts, sw_cfg.max_new_tokens, batch_size=batch_size)
            gpu_monitor.stop()

            peak_vram = gpu_monitor.peak_vram_mb
            peak_vram_abs = gpu_monitor.peak_vram_abs_mb

            # Transformers 批量模式：单次运行，百分位即自身
            ttft_stats = compute_percentile_stats([res["mean_ttft_ms"]])
            itl_stats = compute_percentile_stats([res["mean_itl_ms"]])   # → 全 -1.0
            tpot_stats = compute_percentile_stats([res["mean_tpot_ms"]])

            results.append(
                BenchmarkResult(
                    engine="transformers",
                    test_type="sweep",
                    num_requests=concurrency,
                    request_rate=float("inf"),
                    prompt_tokens=sw_cfg.prompt_length,
                    max_new_tokens=sw_cfg.max_new_tokens,
                    ttft_ms=round(ttft_stats["mean"], 2),
                    median_ttft_ms=round(ttft_stats["median"], 2),
                    p99_ttft_ms=round(ttft_stats["p99"], 2),
                    mean_tps=round(res["concurrent_tps"], 2),
                    mean_itl_ms=round(itl_stats["mean"], 2),
                    median_itl_ms=round(itl_stats["median"], 2),
                    p99_itl_ms=round(itl_stats["p99"], 2),
                    mean_tpot_ms=round(tpot_stats["mean"], 2),
                    median_tpot_ms=round(tpot_stats["median"], 2),
                    p99_tpot_ms=round(tpot_stats["p99"], 2),
                    peak_vram_mb=round(peak_vram, 1),
                    peak_vram_abs_mb=round(peak_vram_abs, 1),
                    run_id=run_id,
                    timestamp=datetime.now().isoformat(),
                )
            )
            logger.info(
                "[transformers sweep] concurrency=%d => ttft=%.2f ms, tps=%.2f tok/s, itl=N/A, tpot=%.2f ms",
                concurrency,
                res["mean_ttft_ms"],
                res["concurrent_tps"],
                res["mean_tpot_ms"],
            )

        except Exception as e:
            logger.error(
                "[transformers sweep] concurrency=%d failed (likely OOM): %s",
                concurrency,
                e,
            )
            gpu_monitor.stop()
            results.append(
                    BenchmarkResult(
                        engine="transformers",
                        test_type="sweep",
                        num_requests=concurrency,
                        request_rate=float("inf"),
                        prompt_tokens=sw_cfg.prompt_length,
                        max_new_tokens=sw_cfg.max_new_tokens,
                        ttft_ms=-1,
                        median_ttft_ms=-1.0,
                        p99_ttft_ms=-1.0,
                        mean_tps=-1,
                        mean_itl_ms=-1.0,
                        median_itl_ms=-1.0,
                        p99_itl_ms=-1.0,
                        mean_tpot_ms=-1.0,
                        median_tpot_ms=-1.0,
                        p99_tpot_ms=-1.0,
                        peak_vram_mb=-1,
                        peak_vram_abs_mb=-1,
                        run_id=run_id,
                        timestamp=datetime.now().isoformat(),
                    )
                )

        # Brief pause between levels
        time.sleep(3)

    # Clean up
    del model
    torch.cuda.empty_cache()
    logger.info("Model unloaded, GPU cache cleared.")

    return results


# ============================================================
# 主入口
# ============================================================


async def main():
    """渐进并发扫描主入口。"""
    parser = argparse.ArgumentParser(description="Progressive concurrency sweep")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["vllm", "sglang", "transformers", "all"],
        default="all",
        help="Engine(s) to sweep (default: all)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg = load_config(args.config)
    run_id = make_run_id(cfg.output.timestamp_format)

    engines = (
        ["vllm", "sglang", "transformers"]
        if args.engine == "all"
        else [args.engine]
    )

    logger.info("=== Sweep === run_id=%s engines=%s", run_id, engines)

    all_results = []

    for i, engine in enumerate(engines):
        logger.info("=== Sweeping engine: %s ===", engine)

        if engine in ("vllm", "sglang"):
            results = await sweep_http_engine(engine, cfg, run_id)
        else:
            results = sweep_transformers(cfg, run_id)

        all_results.extend(results)

        # Wait between engines for GPU memory cleanup (skip after last engine)
        if i < len(engines) - 1:
            logger.info("Waiting 30s for GPU memory cleanup before next engine...")
            await asyncio.sleep(30)

    # Output CSV
    csv_path = f"{cfg.output.results_dir}/sweep_{run_id}.csv"
    results_to_csv(all_results, csv_path)
    logger.info("Sweep results saved to %s", csv_path)


if __name__ == "__main__":
    asyncio.run(main())
