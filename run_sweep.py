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
from common.engine_metrics import fetch_engine_metrics
from common.gpu import GPUMonitor
from common.metrics import BenchmarkResult, make_run_id, results_to_csv
from common.prompts import generate_batch_prompts

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
            f"python -m vllm.entrypoints.openai.api_server "
            f"--model {model} "
            f"--port {port} "
            f"--dtype auto "
            f"{extra_args}"
        )
    elif engine == "sglang":
        port = cfg.engines.sglang.port
        extra_args = cfg.engines.sglang.extra_args
        cmd = (
            f"python -m sglang.launch_server "
            f"--model-path {model} "
            f"--port {port} "
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
    )
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    """停止服务器进程。"""
    logger.info("Stopping server (PID %d)...", proc.pid)
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        logger.warning("Server did not terminate, killing...")
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

        for batch_size in concurrency_levels:
            logger.info(
                "[%s sweep] concurrency=%d, prompt_length=%d, max_new_tokens=%d",
                engine,
                batch_size,
                sw_cfg.prompt_length,
                sw_cfg.max_new_tokens,
            )

            try:
                prompts = generate_batch_prompts(
                    batch_size, sw_cfg.prompt_length, tokenizer=tokenizer
                )

                gpu_monitor.start(reset_baseline=False)
                res = await concurrent_stream_requests(
                    base_url, prompts, model, max_tokens=sw_cfg.max_new_tokens
                )
                gpu_monitor.stop()

                results.append(
                    BenchmarkResult(
                        engine=engine,
                        test_type="sweep",
                        batch_size=batch_size,
                        prompt_tokens=sw_cfg.prompt_length,
                        max_new_tokens=sw_cfg.max_new_tokens,
                        ttft_ms=round(res["mean_ttft_ms"], 2),
                        mean_tps=round(res["concurrent_tps"], 2),
                        peak_vram_mb=round(gpu_monitor.peak_vram_mb, 1),
                        peak_vram_abs_mb=round(gpu_monitor.peak_vram_abs_mb, 1),
                        kv_cache_usage=-1,
                        num_running_reqs=-1,
                        num_waiting_reqs=-1,
                        run_id=run_id,
                        timestamp=datetime.now().isoformat(),
                    )
                )
                # Fetch engine internal metrics from /metrics endpoint
                engine_m = await fetch_engine_metrics(base_url, engine)
                kv_pct = engine_m.get("kv_cache_usage", -1) * 100 if engine_m.get("kv_cache_usage", -1) >= 0 else -1
                # Update the last result with engine metrics
                if engine_m:
                    last = results[-1]
                    results[-1] = BenchmarkResult(
                        engine=last.engine,
                        test_type=last.test_type,
                        batch_size=last.batch_size,
                        prompt_tokens=last.prompt_tokens,
                        max_new_tokens=last.max_new_tokens,
                        ttft_ms=last.ttft_ms,
                        mean_tps=last.mean_tps,
                        peak_vram_mb=last.peak_vram_mb,
                        peak_vram_abs_mb=last.peak_vram_abs_mb,
                        kv_cache_usage=engine_m.get("kv_cache_usage", -1),
                        num_running_reqs=int(engine_m.get("num_running_reqs", -1)),
                        num_waiting_reqs=int(engine_m.get("num_waiting_reqs", -1)),
                        run_id=last.run_id,
                        timestamp=last.timestamp,
                    )
                logger.info(
                    "[%s sweep] concurrency=%d => ttft=%.2f ms, tps=%.2f tok/s, kv=%.1f%%, running=%d, waiting=%d",
                    engine,
                    batch_size,
                    res["mean_ttft_ms"],
                    res["concurrent_tps"],
                    kv_pct,
                    int(engine_m.get("num_running_reqs", -1)),
                    int(engine_m.get("num_waiting_reqs", -1)),
                )

            except Exception as e:
                logger.error(
                    "[%s sweep] concurrency=%d failed: %s", engine, batch_size, e
                )
                gpu_monitor.stop()
                results.append(
                    BenchmarkResult(
                        engine=engine,
                        test_type="sweep",
                        batch_size=batch_size,
                        prompt_tokens=sw_cfg.prompt_length,
                        max_new_tokens=sw_cfg.max_new_tokens,
                        ttft_ms=-1,
                        mean_tps=-1,
                        peak_vram_mb=-1,
                        peak_vram_abs_mb=-1,
                        kv_cache_usage=-1,
                        num_running_reqs=-1,
                        num_waiting_reqs=-1,
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
    for batch_size in concurrency_levels:
        logger.info(
            "[transformers sweep] batch_size=%d, prompt_length=%d, max_new_tokens=%d",
            batch_size,
            sw_cfg.prompt_length,
            sw_cfg.max_new_tokens,
        )

        try:
            prompts = generate_batch_prompts(
                batch_size, sw_cfg.prompt_length, tokenizer=tokenizer
            )

            all_inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True,
            ).to(model.device)

            gpu_monitor.start(reset_baseline=False)
            start_time = time.monotonic()

            with torch.no_grad():
                outputs = model.generate(
                    **all_inputs,
                    max_new_tokens=sw_cfg.max_new_tokens,
                )

            end_time = time.monotonic()
            gpu_monitor.stop()

            total_time_s = end_time - start_time
            input_lengths = all_inputs["attention_mask"].sum(dim=1).tolist()
            total_tokens = sum(len(outputs[i]) - input_lengths[i] for i in range(len(prompts)))
            concurrent_tps = total_tokens / total_time_s if total_time_s > 0 else 0.0
            mean_ttft_ms = total_time_s * 1000.0 / len(prompts)  # estimate
            peak_vram = gpu_monitor.peak_vram_mb
            peak_vram_abs = gpu_monitor.peak_vram_abs_mb

            results.append(
                BenchmarkResult(
                    engine="transformers",
                    test_type="sweep",
                    batch_size=batch_size,
                    prompt_tokens=sw_cfg.prompt_length,
                    max_new_tokens=sw_cfg.max_new_tokens,
                    ttft_ms=round(mean_ttft_ms, 2),
                    mean_tps=round(concurrent_tps, 2),
                    peak_vram_mb=round(peak_vram, 1),
                    peak_vram_abs_mb=round(peak_vram_abs, 1),
                    kv_cache_usage=-1,  # Not applicable for Transformers
                    num_running_reqs=-1,
                    num_waiting_reqs=-1,
                    run_id=run_id,
                    timestamp=datetime.now().isoformat(),
                )
            )
            logger.info(
                "[transformers sweep] batch_size=%d => ttft=%.2f ms, tps=%.2f tok/s, vram=%.1f MB, abs=%.1f MB",
                batch_size,
                mean_ttft_ms,
                concurrent_tps,
                peak_vram, peak_vram_abs,
            )

        except Exception as e:
            logger.error(
                "[transformers sweep] batch_size=%d failed (likely OOM): %s",
                batch_size,
                e,
            )
            gpu_monitor.stop()
            results.append(
                BenchmarkResult(
                    engine="transformers",
                    test_type="sweep",
                    batch_size=batch_size,
                    prompt_tokens=sw_cfg.prompt_length,
                    max_new_tokens=sw_cfg.max_new_tokens,
                    ttft_ms=-1,
                    mean_tps=-1,
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