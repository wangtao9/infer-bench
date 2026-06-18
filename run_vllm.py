"""vLLM 推理基准测试脚本。

自动启动/停止 vLLM HTTP 服务器，通过 OpenAI-compatible API
执行单请求和并发测试，采集 TTFT/TPS/VRAM 指标。
"""

import asyncio
import logging
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime

from common.client import concurrent_stream_requests, stream_request, wait_for_server
from common.config import load_config
from common.engine_metrics import fetch_engine_metrics
from common.gpu import GPUMonitor
from common.metrics import BenchmarkResult, make_run_id, results_to_csv
from common.prompts import generate_batch_prompts, generate_prompt
from transformers import AutoTokenizer

logger = logging.getLogger("run_vllm")


# ============================================================
# 服务器启停
# ============================================================


def start_vllm_server(cfg) -> subprocess.Popen:
    """启动 vLLM 服务器，返回 Popen 对象。

    使用 start_new_session=True 将服务器进程放入独立进程组，
    以便 stop_vllm_server 能 kill 整个进程树。
    """
    model = cfg.model.path or cfg.model.name
    port = cfg.engines.vllm.port
    extra_args = cfg.engines.vllm.extra_args

    cmd = (
        f"vllm serve {model} "
        f"--port {port} "
        f"--dtype auto "
        f"{extra_args}"
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cfg.gpu.device

    logger.info("Starting vLLM server: %s", cmd)
    proc = subprocess.Popen(
        cmd,
        shell=True,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        start_new_session=True,  # 新进程组，方便 kill 整个进程树
    )
    return proc


def stop_vllm_server(proc: subprocess.Popen) -> None:
    """停止 vLLM 服务器进程及其所有子进程。

    vLLM 通过 shell=True 启动，proc.terminate() 只会终止 shell 进程，
    不会传递给 vLLM 子进程。使用 os.killpg() kill 整个进程组。
    """
    import signal

    logger.info("Stopping vLLM server (PID %d)...", proc.pid)
    try:
        # kill 整个进程组（shell + vLLM 子进程）
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass  # 进程已退出
    except PermissionError:
        logger.warning("Permission denied when killing vLLM process group, falling back to terminate")
        proc.terminate()

    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        logger.warning("vLLM server did not terminate in 15s, force killing...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait(timeout=5)
    logger.info("vLLM server stopped.")


# ============================================================
# 单请求测试
# ============================================================


async def run_single_request_tests(
    base_url: str,
    model: str,
    cfg,
    tokenizer,
    run_id: str,
    gpu_monitor: GPUMonitor,
) -> list[BenchmarkResult]:
    """对每个 prompt 长度执行单请求测试。"""
    results = []
    sr_cfg = cfg.test.single_request

    for prompt_len in sr_cfg.prompt_lengths:
        prompt = generate_prompt(prompt_len, tokenizer=tokenizer)
        logger.info(
            "[single] prompt_length=%d, max_new_tokens=%d",
            prompt_len,
            sr_cfg.max_new_tokens,
        )

        # Warmup
        for _ in range(sr_cfg.num_warmup):
            await stream_request(
                base_url, prompt, model, max_tokens=sr_cfg.max_new_tokens
            )

        # Benchmark runs
        ttfts = []
        tps_list = []
        gpu_monitor.start(reset_baseline=False)

        for _ in range(sr_cfg.num_runs):
            res = await stream_request(
                base_url, prompt, model, max_tokens=sr_cfg.max_new_tokens
            )
            ttfts.append(res["ttft_ms"])
            tps_list.append(res["tps"])

        gpu_monitor.stop()

        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
        avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0
        peak_vram = gpu_monitor.peak_vram_mb
        peak_vram_abs = gpu_monitor.peak_vram_abs_mb

        # 从 /metrics 端点抓取引擎内部指标
        engine_metrics = await fetch_engine_metrics(base_url, "vllm")
        kv_usage = engine_metrics.get("kv_cache_usage", -1)
        num_running = int(engine_metrics.get("num_running_reqs", -1))
        num_waiting = int(engine_metrics.get("num_waiting_reqs", -1))

        results.append(
            BenchmarkResult(
                engine="vllm",
                test_type="single",
                batch_size=1,
                prompt_tokens=prompt_len,
                max_new_tokens=sr_cfg.max_new_tokens,
                ttft_ms=round(avg_ttft, 2),
                mean_tps=round(avg_tps, 2),
                peak_vram_mb=round(peak_vram, 1),
                peak_vram_abs_mb=round(peak_vram_abs, 1),
                kv_cache_usage=kv_usage,
                num_running_reqs=num_running,
                num_waiting_reqs=num_waiting,
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
            )
        )
        logger.info(
            "[single] prompt_length=%d => avg_ttft=%.2f ms, avg_tps=%.2f tok/s, kv=%.1f%%, running=%d, waiting=%d",
            prompt_len,
            avg_ttft,
            avg_tps,
            kv_usage * 100 if kv_usage >= 0 else -1,
            num_running,
            num_waiting,
        )

    return results


# ============================================================
# 并发测试
# ============================================================


async def run_concurrent_tests(
    base_url: str,
    model: str,
    cfg,
    tokenizer,
    run_id: str,
    gpu_monitor: GPUMonitor,
) -> list[BenchmarkResult]:
    """对每个 batch_size 执行并发测试。"""
    results = []
    cc_cfg = cfg.test.concurrent

    for batch_size in cc_cfg.batch_sizes:
        prompts = generate_batch_prompts(
            batch_size, cc_cfg.prompt_length, tokenizer=tokenizer
        )
        logger.info(
            "[concurrent] batch_size=%d, prompt_length=%d, max_new_tokens=%d",
            batch_size,
            cc_cfg.prompt_length,
            cc_cfg.max_new_tokens,
        )

        # Warmup
        for _ in range(cc_cfg.num_warmup):
            await concurrent_stream_requests(
                base_url, prompts, model, max_tokens=cc_cfg.max_new_tokens
            )

        # Benchmark runs
        ttfts = []
        tps_list = []
        gpu_monitor.start(reset_baseline=False)

        for _ in range(cc_cfg.num_runs):
            res = await concurrent_stream_requests(
                base_url, prompts, model, max_tokens=cc_cfg.max_new_tokens
            )
            ttfts.append(res["mean_ttft_ms"])
            tps_list.append(res["concurrent_tps"])

        gpu_monitor.stop()

        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
        avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0
        peak_vram = gpu_monitor.peak_vram_mb
        peak_vram_abs = gpu_monitor.peak_vram_abs_mb

        # 从 /metrics 端点抓取引擎内部指标
        engine_metrics = await fetch_engine_metrics(base_url, "vllm")
        kv_usage = engine_metrics.get("kv_cache_usage", -1)
        num_running = int(engine_metrics.get("num_running_reqs", -1))
        num_waiting = int(engine_metrics.get("num_waiting_reqs", -1))

        results.append(
            BenchmarkResult(
                engine="vllm",
                test_type="concurrent",
                batch_size=batch_size,
                prompt_tokens=cc_cfg.prompt_length,
                max_new_tokens=cc_cfg.max_new_tokens,
                ttft_ms=round(avg_ttft, 2),
                mean_tps=round(avg_tps, 2),
                peak_vram_mb=round(peak_vram, 1),
                peak_vram_abs_mb=round(peak_vram_abs, 1),
                kv_cache_usage=kv_usage,
                num_running_reqs=num_running,
                num_waiting_reqs=num_waiting,
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
            )
        )
        logger.info(
            "[concurrent] batch_size=%d => avg_ttft=%.2f ms, avg_tps=%.2f tok/s, kv=%.1f%%, running=%d, waiting=%d",
            batch_size,
            avg_ttft,
            avg_tps,
            kv_usage * 100 if kv_usage >= 0 else -1,
            num_running,
            num_waiting,
        )

    return results


# ============================================================
# 主入口
# ============================================================


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="vLLM inference benchmark")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config YAML"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg = load_config(args.config)
    model = cfg.model.path or cfg.model.name
    port = cfg.engines.vllm.port
    base_url = f"http://localhost:{port}"
    run_id = make_run_id(cfg.output.timestamp_format)

    logger.info("=== vLLM Benchmark === run_id=%s model=%s", run_id, model)

    # Initialize GPU monitor BEFORE starting server so baseline is
    # captured before model loading (otherwise peak - baseline ≈ 0).
    gpu_monitor = GPUMonitor(
        device_index=int(cfg.gpu.device.split(",")[0]),
        interval_ms=cfg.gpu.monitor_interval_ms,
    )

    # Start GPU monitor before server — baseline = idle GPU memory.
    # Subsequent start(reset_baseline=False) calls preserve this baseline
    # so peak_vram_mb reflects the full model + inference delta.
    gpu_monitor.start(reset_baseline=True)

    server_proc = start_vllm_server(cfg)

    try:
        # Wait for server to be ready
        ready = await wait_for_server(base_url, timeout=300)
        if not ready:
            logger.error("Server failed to start, aborting.")
            return

        # Stabilization warmup
        logger.info("Server ready. Waiting 10s for warmup stabilization...")
        await asyncio.sleep(10)

        # Load tokenizer
        logger.info("Loading tokenizer: %s", model)
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        # Run tests
        all_results = []

        logger.info("--- Single Request Tests ---")
        sr_results = await run_single_request_tests(
            base_url, model, cfg, tokenizer, run_id, gpu_monitor
        )
        all_results.extend(sr_results)

        logger.info("--- Concurrent Tests ---")
        cc_results = await run_concurrent_tests(
            base_url, model, cfg, tokenizer, run_id, gpu_monitor
        )
        all_results.extend(cc_results)

        # Output CSV
        csv_path = f"{cfg.output.results_dir}/vllm_{run_id}.csv"
        results_to_csv(all_results, csv_path)
        logger.info("Results saved to %s", csv_path)

    finally:
        stop_vllm_server(server_proc)


if __name__ == "__main__":
    asyncio.run(main())