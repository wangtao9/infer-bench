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
from common.metrics import BenchmarkResult, compute_percentile_stats, make_run_id, results_to_csv
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
    """单请求流式生成，测量 TTFT / ITL / TPOT / E2EL / TPS。

    使用 TextIteratorStreamer 在后台线程中运行 model.generate()，
    逐 token 记录时间戳，计算 TTFT、ITL、TPOT、E2EL。

    Args:
        model: 已加载的模型。
        tokenizer: 已加载的 tokenizer。
        prompt: 输入 prompt 文本。
        max_new_tokens: 最大生成 token 数。
        temperature: 采样温度，0.0 表示贪心解码。

    Returns:
        dict: ttft_ms, itl_ms (list), tpot_ms, e2el_ms,
              total_tokens, total_time_s, tps, text
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
    most_recent_timestamp = start_time
    itl_ms: list[float] = []
    output_text = ""

    # 在后台线程中运行 generate
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    for token_text in streamer:
        timestamp = time.monotonic()
        if first_token_time is None:
            first_token_time = timestamp
        else:
            itl_ms.append((timestamp - most_recent_timestamp) * 1000.0)
        most_recent_timestamp = timestamp
        output_text += token_text

    thread.join()
    end_time = time.monotonic()

    ttft_ms = (first_token_time - start_time) * 1000.0 if first_token_time else 0.0
    total_time_s = end_time - start_time
    e2el_ms = total_time_s * 1000.0
    total_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))

    # TPOT = (E2EL - TTFT) / (output_tokens - 1)
    if total_tokens > 1 and first_token_time:
        tpot_ms = (end_time - first_token_time) * 1000.0 / (total_tokens - 1)
    else:
        tpot_ms = e2el_ms if total_tokens == 1 else 0.0

    tps = total_tokens / total_time_s if total_time_s > 0 else 0.0

    return {
        "ttft_ms": ttft_ms,
        "itl_ms": itl_ms,
        "tpot_ms": tpot_ms,
        "e2el_ms": e2el_ms,
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
    total_time_s / len(prompts) 作为 TTFT 的估计值。
    由于无流式，无法测量逐 token ITL，返回 mean_itl_ms=-1。
    TPOT 从总时间估算。

    Args:
        model: 已加载的模型。
        tokenizer: 已加载的 tokenizer。
        prompts: 输入 prompt 文本列表。
        max_new_tokens: 最大生成 token 数。
        temperature: 采样温度，0.0 表示贪心解码。

    Returns:
        dict: mean_ttft_ms, mean_itl_ms, mean_tpot_ms, e2el_ms,
              total_tokens, concurrent_tps, total_time_s
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
    e2el_ms = total_time_s * 1000.0
    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    total_tokens = sum(
        len(outputs[i]) - input_lengths[i] for i in range(len(prompts))
    )
    concurrent_tps = total_tokens / total_time_s if total_time_s > 0 else 0.0
    mean_ttft_ms = total_time_s * 1000.0 / len(prompts)  # estimate

    # ITL: 无法测量（无流式）
    mean_itl_ms = -1.0

    # TPOT: 估算 = (total_time - estimated_ttft_per_req) / (tokens_per_req - 1)
    # 简化：用 overall TPOT = e2el / total_tokens (粗估)
    mean_tpot_ms = e2el_ms / total_tokens if total_tokens > 0 else 0.0

    return {
        "mean_ttft_ms": mean_ttft_ms,
        "mean_itl_ms": mean_itl_ms,
        "mean_tpot_ms": mean_tpot_ms,
        "e2el_ms": e2el_ms,
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

        # Benchmark runs — 采集 TTFT / ITL / TPOT / E2EL / TPS
        ttfts = []
        tps_list = []
        itl_list = []
        tpot_list = []
        e2el_list = []
        gpu_monitor.start(reset_baseline=False)

        for _ in range(sr_cfg.num_requests):
            res = single_generate(model, tokenizer, prompt, sr_cfg.max_new_tokens)
            ttfts.append(res["ttft_ms"])
            tps_list.append(res["tps"])
            itl_list.append(res["itl_ms"])
            tpot_list.append(res["tpot_ms"])
            e2el_list.append(res["e2el_ms"])

        gpu_monitor.stop()

        # ITL: 所有 run 中所有 token 间隔合并后再统计
        all_itls = [itl for itls in itl_list for itl in itls]
        ttft_stats = compute_percentile_stats(ttfts)
        itl_stats = compute_percentile_stats(all_itls)
        tpot_stats = compute_percentile_stats(tpot_list)
        e2el_stats = compute_percentile_stats(e2el_list)
        avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0
        peak_vram = gpu_monitor.peak_vram_mb
        peak_vram_abs = gpu_monitor.peak_vram_abs_mb

        results.append(
            BenchmarkResult(
                engine="transformers",
                test_type="single",
                num_requests=sr_cfg.num_requests,
                request_rate=float("inf"),
                prompt_tokens=prompt_len,
                max_new_tokens=sr_cfg.max_new_tokens,
                ttft_ms=round(ttft_stats["mean"], 2),
                median_ttft_ms=round(ttft_stats["median"], 2),
                p90_ttft_ms=round(ttft_stats["p90"], 2),
                p99_ttft_ms=round(ttft_stats["p99"], 2),
                mean_tps=round(avg_tps, 2),
                mean_itl_ms=round(itl_stats["mean"], 2),
                median_itl_ms=round(itl_stats["median"], 2),
                p90_itl_ms=round(itl_stats["p90"], 2),
                p99_itl_ms=round(itl_stats["p99"], 2),
                mean_tpot_ms=round(tpot_stats["mean"], 2),
                median_tpot_ms=round(tpot_stats["median"], 2),
                p90_tpot_ms=round(tpot_stats["p90"], 2),
                p99_tpot_ms=round(tpot_stats["p99"], 2),
                e2el_ms=round(e2el_stats["mean"], 2),
                median_e2el_ms=round(e2el_stats["median"], 2),
                p90_e2el_ms=round(e2el_stats["p90"], 2),
                p99_e2el_ms=round(e2el_stats["p99"], 2),
                peak_vram_mb=round(peak_vram, 1),
                peak_vram_abs_mb=round(peak_vram_abs, 1),
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
            )
        )
        logger.info(
            "[single] prompt_length=%d => ttft=%.2f ms (p99=%.2f), tps=%.2f tok/s, itl=%.2f ms (p99=%.2f), tpot=%.2f ms, e2el=%.2f ms",
            prompt_len,
            ttft_stats["mean"], ttft_stats["p99"],
            avg_tps,
            itl_stats["mean"], itl_stats["p99"],
            tpot_stats["mean"],
            e2el_stats["mean"],
        )

    return results


# ============================================================
# 并发测试
# ============================================================


def run_concurrent_tests(model, tokenizer, cfg, run_id, gpu_monitor):
    """执行批量并发测试。

    使用 batch_generate() 代替 vLLM/SGLang 的
    concurrent_stream_requests()。批量生成是同步的，
    语义上与 HTTP 并发请求不同——所有 prompt 在同一
    forward pass 中处理，而非独立并发请求。

    NOTE: Transformers 为同步批量处理，仅支持 request_rate=inf（batch），
    忽略有限 request_rate 值（Poisson 调度不适用于同步批量推理）。

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
    cc_cfg.validate()

    num_requests = cc_cfg.num_requests
    prompts = generate_batch_prompts(
        num_requests, cc_cfg.prompt_length, tokenizer=tokenizer
    )

    # Warmup：使用不同的 prompt 避免正式测量时命中 KV cache
    warmup_prompts = generate_batch_prompts(
        num_requests, cc_cfg.prompt_length, tokenizer=tokenizer,
        variant_offset=100,
    )
    for _ in range(cc_cfg.num_warmup):
        batch_generate(model, tokenizer, warmup_prompts, cc_cfg.max_new_tokens)

    # 仅处理 request_rate=inf 的轮次，跳过有限值（Poisson 不适用于同步批量）
    for rate in cc_cfg.request_rate:
        if rate != float("inf"):
            logger.warning(
                "Transformers 忽略 request_rate=%s（同步批量处理仅支持 batch 模式）",
                rate,
            )
            continue

        logger.info(
            "[concurrent] num_requests=%d, prompt_length=%d, max_new_tokens=%d",
            num_requests,
            cc_cfg.prompt_length,
            cc_cfg.max_new_tokens,
        )

        # 单次运行
        gpu_monitor.start(reset_baseline=False)
        res = batch_generate(model, tokenizer, prompts, cc_cfg.max_new_tokens)
        gpu_monitor.stop()

        ttft_stats = compute_percentile_stats([res["mean_ttft_ms"]])
        itl_stats = compute_percentile_stats([res["mean_itl_ms"]])   # → 全 -1.0
        tpot_stats = compute_percentile_stats([res["mean_tpot_ms"]])
        e2el_stats = compute_percentile_stats([res["e2el_ms"]])
        peak_vram = gpu_monitor.peak_vram_mb
        peak_vram_abs = gpu_monitor.peak_vram_abs_mb

        results.append(
            BenchmarkResult(
                engine="transformers",
                test_type="concurrent",
                num_requests=num_requests,
                prompt_tokens=cc_cfg.prompt_length,
                max_new_tokens=cc_cfg.max_new_tokens,
                ttft_ms=round(ttft_stats["mean"], 2),
                median_ttft_ms=round(ttft_stats["median"], 2),
                p90_ttft_ms=round(ttft_stats["p90"], 2),
                p99_ttft_ms=round(ttft_stats["p99"], 2),
                mean_tps=round(res["concurrent_tps"], 2),
                mean_itl_ms=round(itl_stats["mean"], 2),
                median_itl_ms=round(itl_stats["median"], 2),
                p90_itl_ms=round(itl_stats["p90"], 2),
                p99_itl_ms=round(itl_stats["p99"], 2),
                mean_tpot_ms=round(tpot_stats["mean"], 2),
                median_tpot_ms=round(tpot_stats["median"], 2),
                p90_tpot_ms=round(tpot_stats["p90"], 2),
                p99_tpot_ms=round(tpot_stats["p99"], 2),
                e2el_ms=round(e2el_stats["mean"], 2),
                median_e2el_ms=round(e2el_stats["median"], 2),
                p90_e2el_ms=round(e2el_stats["p90"], 2),
                p99_e2el_ms=round(e2el_stats["p99"], 2),
                peak_vram_mb=round(peak_vram, 1),
                peak_vram_abs_mb=round(peak_vram_abs, 1),
                request_rate=float("inf"),
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
            )
        )
        logger.info(
            "[concurrent] num_requests=%d => ttft=%.2f ms, tps=%.2f tok/s, itl=N/A, tpot=%.2f ms, e2el=%.2f ms",
            num_requests,
            res["mean_ttft_ms"],
            res["concurrent_tps"],
            res["mean_tpot_ms"],
            res["e2el_ms"],
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