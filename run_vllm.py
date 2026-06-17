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
from common.gpu import GPUMonitor
from common.metrics import BenchmarkResult, make_run_id, results_to_csv
from common.prompts import generate_batch_prompts, generate_prompt
from transformers import AutoTokenizer

logger = logging.getLogger("run_vllm")


# ============================================================
# 服务器启停
# ============================================================


def start_vllm_server(cfg) -> subprocess.Popen:
    """启动 vLLM 服务器，返回 Popen 对象。"""
    model = cfg.model.path or cfg.model.name
    port = cfg.engines.vllm.port
    extra_args = cfg.engines.vllm.extra_args

    cmd = (
        f"python -m vllm.entrypoints.openai.api_server "
        f"--model {model} "
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
    )
    return proc


def stop_vllm_server(proc: subprocess.Popen) -> None:
    """停止 vLLM 服务器进程。"""
    logger.info("Stopping vLLM server (PID %d)...", proc.pid)
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        logger.warning("vLLM server did not terminate, killing...")
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
        gpu_monitor.start()

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
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
            )
        )
        logger.info(
            "[single] prompt_length=%d => avg_ttft=%.2f ms, avg_tps=%.2f tok/s, peak_vram=%.1f MB",
            prompt_len,
            avg_ttft,
            avg_tps,
            peak_vram,
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
        gpu_monitor.start()

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
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
            )
        )
        logger.info(
            "[concurrent] batch_size=%d => avg_ttft=%.2f ms, avg_tps=%.2f tok/s, peak_vram=%.1f MB",
            batch_size,
            avg_ttft,
            avg_tps,
            peak_vram,
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

        # Initialize GPU monitor
        gpu_monitor = GPUMonitor(
            device_index=int(cfg.gpu.device.split(",")[0]),
            interval_ms=cfg.gpu.monitor_interval_ms,
        )

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