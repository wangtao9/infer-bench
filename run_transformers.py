"""Transformers 推理基准测试脚本。

在进程内使用 AutoModelForCausalLM 执行单请求和批量测试，
采集 TTFT/TPS/VRAM 指标。与 vLLM/SGLang 不同，Transformers
不启动 HTTP 服务器，所有推理均在进程内完成。
"""

import argparse
import logging
import threading
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from common.config import load_config
from common.gpu import GPUMonitor
from common.metrics import BenchmarkResult, make_run_id, results_to_csv
from common.prompts import generate_batch_prompts, generate_prompt

logger = logging.getLogger("run_transformers")

# dtype 字符串 → torch dtype 映射
DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


# ============================================================
# 模型加载
# ============================================================


def load_model_and_tokenizer(cfg):
    """加载模型和 tokenizer。

    Args:
        cfg: BenchmarkConfig 实例。

    Returns:
        (model, tokenizer) 元组。
    """
    model_name = cfg.model.path or cfg.model.name
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
    return model, tokenizer


# ============================================================
# 单请求生成（流式）
# ============================================================


def single_generate(model, tokenizer, prompt, max_new_tokens, temperature=0.0):
    """单请求流式生成，测量 TTFT 和 TPS。

    使用 TextIteratorStreamer 在后台线程中运行 model.generate()，
    记录首 token 时间和总时间。

    Args:
        model: 已加载的模型。
        tokenizer: 已加载的 tokenizer。
        prompt: 输入 prompt 文本。
        max_new_tokens: 最大生成 token 数。
        temperature: 采样温度，0.0 表示贪心解码。

    Returns:
        dict: ttft_ms, total_tokens, total_time_s, tps, text
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature if temperature > 0 else None,
        "do_sample": temperature > 0,
        "streamer": streamer,
    }

    start_time = time.monotonic()
    first_token_time = None
    output_text = ""

    # 在后台线程中运行 generate
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    for token_text in streamer:
        if first_token_time is None:
            first_token_time = time.monotonic()
        output_text += token_text

    thread.join()
    end_time = time.monotonic()

    ttft_ms = (first_token_time - start_time) * 1000.0 if first_token_time else 0.0
    total_time_s = end_time - start_time
    total_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))
    tps = total_tokens / total_time_s if total_time_s > 0 else 0.0

    return {
        "ttft_ms": ttft_ms,
        "total_tokens": total_tokens,
        "total_time_s": total_time_s,
        "tps": tps,
        "text": output_text,
    }


# ============================================================
# 批量生成（无流式）
# ============================================================


def batch_generate(model, tokenizer, prompts, max_new_tokens, temperature=0.0):
    """批量生成（无流式），测量并发吞吐量。

    NOTE: 批量生成无法精确测量每个请求的 TTFT，因此使用
    total_time_s / batch_size 作为 TTFT 的估计值。

    Args:
        model: 已加载的模型。
        tokenizer: 已加载的 tokenizer。
        prompts: 输入 prompt 文本列表。
        max_new_tokens: 最大生成 token 数。
        temperature: 采样温度，0.0 表示贪心解码。

    Returns:
        dict: mean_ttft_ms, total_tokens, concurrent_tps, total_time_s
    """
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature if temperature > 0 else None,
        "do_sample": temperature > 0,
    }

    start_time = time.monotonic()

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    end_time = time.monotonic()

    total_time_s = end_time - start_time
    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    total_tokens = sum(
        len(outputs[i]) - input_lengths[i] for i in range(len(prompts))
    )
    concurrent_tps = total_tokens / total_time_s if total_time_s > 0 else 0.0
    mean_ttft_ms = total_time_s * 1000.0 / len(prompts)  # estimate

    return {
        "mean_ttft_ms": mean_ttft_ms,
        "total_tokens": total_tokens,
        "concurrent_tps": concurrent_tps,
        "total_time_s": total_time_s,
    }


# ============================================================
# 单请求测试
# ============================================================


def run_single_request_tests(model, tokenizer, cfg, run_id, gpu_monitor):
    """对每个 prompt 长度执行单请求测试。

    与 vLLM/SGLang 脚本结构相同，但使用 single_generate() 代替
    stream_request()，不需要 HTTP 服务器。

    Args:
        model: 已加载的模型。
        tokenizer: 已加载的 tokenizer。
        cfg: BenchmarkConfig 实例。
        run_id: 运行标识符。
        gpu_monitor: GPUMonitor 实例。

    Returns:
        list[BenchmarkResult]: 测试结果列表。
    """
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
            single_generate(model, tokenizer, prompt, sr_cfg.max_new_tokens)

        # Benchmark runs
        ttfts = []
        tps_list = []
        gpu_monitor.start(reset_baseline=False)

        for _ in range(sr_cfg.num_runs):
            res = single_generate(model, tokenizer, prompt, sr_cfg.max_new_tokens)
            ttfts.append(res["ttft_ms"])
            tps_list.append(res["tps"])

        gpu_monitor.stop()

        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
        avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0
        peak_vram = gpu_monitor.peak_vram_mb

        results.append(
            BenchmarkResult(
                engine="transformers",
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


def run_concurrent_tests(model, tokenizer, cfg, run_id, gpu_monitor):
    """对每个 batch_size 执行批量测试。

    NOTE: 使用 batch_generate() 代替 vLLM/SGLang 的
    concurrent_stream_requests()。批量生成是同步的，
    语义上与 HTTP 并发请求不同——所有 prompt 在同一
    forward pass 中处理，而非独立并发请求。

    Args:
        model: 已加载的模型。
        tokenizer: 已加载的 tokenizer。
        cfg: BenchmarkConfig 实例。
        run_id: 运行标识符。
        gpu_monitor: GPUMonitor 实例。

    Returns:
        list[BenchmarkResult]: 测试结果列表。
    """
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
            batch_generate(model, tokenizer, prompts, cc_cfg.max_new_tokens)

        # Benchmark runs
        ttfts = []
        tps_list = []
        gpu_monitor.start(reset_baseline=False)

        for _ in range(cc_cfg.num_runs):
            res = batch_generate(model, tokenizer, prompts, cc_cfg.max_new_tokens)
            ttfts.append(res["mean_ttft_ms"])
            tps_list.append(res["concurrent_tps"])

        gpu_monitor.stop()

        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
        avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0
        peak_vram = gpu_monitor.peak_vram_mb

        results.append(
            BenchmarkResult(
                engine="transformers",
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


def main():
    """主入口（同步，非 async）。"""
    parser = argparse.ArgumentParser(description="Transformers inference benchmark")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config YAML"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg = load_config(args.config)
    model_name = cfg.model.path or cfg.model.name
    run_id = make_run_id(cfg.output.timestamp_format)

    logger.info("=== Transformers Benchmark === run_id=%s model=%s", run_id, model_name)

    # Start GPU monitor BEFORE loading model — baseline = idle GPU memory.
    gpu_monitor = GPUMonitor(
        device_index=int(cfg.gpu.device.split(",")[0]),
        interval_ms=cfg.gpu.monitor_interval_ms,
    )
    gpu_monitor.start(reset_baseline=True)

    # Load model and tokenizer (this is where most GPU memory is allocated)
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Stop the initial sampling; baseline is now locked.
    gpu_monitor.stop()

    # Run tests
    all_results = []

    logger.info("--- Single Request Tests ---")
    sr_results = run_single_request_tests(model, tokenizer, cfg, run_id, gpu_monitor)
    all_results.extend(sr_results)

    logger.info("--- Concurrent Tests ---")
    cc_results = run_concurrent_tests(model, tokenizer, cfg, run_id, gpu_monitor)
    all_results.extend(cc_results)

    # Output CSV
    csv_path = f"{cfg.output.results_dir}/transformers_{run_id}.csv"
    results_to_csv(all_results, csv_path)
    logger.info("Results saved to %s", csv_path)

    # Clean up
    del model
    torch.cuda.empty_cache()
    logger.info("Model unloaded, GPU cache cleared.")


if __name__ == "__main__":
    main()