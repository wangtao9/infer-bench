"""指标采集工具：TTFT/TPS/ITL/TPOT/E2EL 计算和数据记录。

参照 vLLM bench serve 的指标体系：
- TTFT: 首 Token 延迟 (ms) — 客户端时间戳
- ITL: Inter-Token Latency (ms) — 逐 chunk 时间戳差值
- TPOT: Time Per Output Token (ms) — (E2EL - TTFT) / (output_tokens - 1)
- E2EL: 端到端延迟 (ms) — 请求发出到最后一个 chunk
- TPS: 吞吐 (tokens/s) — 客户端统计

百分位统计（参照 vLLM bench serve calculate_metrics）：
- median / P90 / P99: 使用 numpy 计算，覆盖 TTFT / ITL / TPOT / E2EL

GPU 显存指标：
- peak_vram_mb: 峰值增量 (MB) — pynvml（Transformers 有意义，预分配引擎为 0）
- peak_vram_abs_mb: 峰值绝对占用量 (MB) — pynvml（始终有意义）
"""

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass
class BenchmarkResult:
    """单次测试结果。"""
    engine: str
    test_type: str          # "single" / "concurrent" / "sweep"
    batch_size: int
    prompt_tokens: int
    max_new_tokens: int
    ttft_ms: float          # 首 Token 延迟 mean (ms)
    median_ttft_ms: float = 0.0
    p90_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    mean_tps: float = 0.0         # 平均吞吐 (tokens/s)
    mean_itl_ms: float = 0.0      # 平均 Inter-Token Latency (ms)
    median_itl_ms: float = 0.0
    p90_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0
    mean_tpot_ms: float = 0.0     # 平均 Time Per Output Token (ms)
    median_tpot_ms: float = 0.0
    p90_tpot_ms: float = 0.0
    p99_tpot_ms: float = 0.0
    e2el_ms: float = 0.0          # 端到端延迟 mean (ms)
    median_e2el_ms: float = 0.0
    p90_e2el_ms: float = 0.0
    p99_e2el_ms: float = 0.0
    peak_vram_mb: float = 0.0     # 峰值显存增量 (MB) — pynvml
    peak_vram_abs_mb: float = 0.0 # 峰值显存绝对占用量 (MB) — pynvml
    run_id: str = ""
    timestamp: str = ""


def compute_percentile_stats(
    values: list[float],
    percentiles: list[float] | None = None,
) -> dict[str, float]:
    """计算一组值的均值、中位数和百分位数。

    参照 vLLM bench serve 的 calculate_metrics()，使用 numpy 计算。
    使用 -1.0 作为不可测量的哨兵值（Transformers 批量模式）。

    Args:
        values: 指标值列表（如所有 TTFT）。
        percentiles: 百分位级别列表，默认 [90, 99]。

    Returns:
        dict with keys: mean, median, p90, p99（及自定义百分位）。
        - 空列表 → 全部 0.0
        - 全部 -1.0 → 全部 -1.0（哨兵传播）
    """
    if percentiles is None:
        percentiles = [90, 99]

    sentinel = -1.0
    result_keys = ["mean", "median"] + [f"p{int(p)}" for p in percentiles]
    default_value = 0.0

    if not values:
        return {k: default_value for k in result_keys}

    # 哨兵传播：如果全部是 -1.0，返回全部 -1.0
    if all(v == sentinel for v in values):
        return {k: sentinel for k in result_keys}

    # 过滤掉 -1.0 哨兵值后计算
    filtered = [v for v in values if v != sentinel]

    if not filtered:
        return {k: default_value for k in result_keys}

    arr = np.array(filtered)
    stats = {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
    }
    for p in percentiles:
        stats[f"p{int(p)}"] = float(np.percentile(arr, p))

    return stats


def compute_ttft_ms(start_time: float, first_token_time: float) -> float:
    """计算首 Token 延迟。"""
    return (first_token_time - start_time) * 1000.0


def compute_tps(total_tokens: int, total_time_s: float) -> float:
    """计算吞吐量。"""
    if total_time_s <= 0:
        return 0.0
    return total_tokens / total_time_s


def compute_tpot_ms(e2el_s: float, ttft_s: float, output_tokens: int) -> float:
    """计算 Time Per Output Token。

    TPOT = (E2EL - TTFT) / (output_tokens - 1)
    参照 vLLM bench serve 的计算方式。
    """
    if output_tokens <= 1:
        return 0.0
    return (e2el_s - ttft_s) * 1000.0 / (output_tokens - 1)


def compute_concurrent_tps(
    total_tokens: int,
    earliest_start: float,
    latest_end: float,
) -> float:
    """计算并发吞吐量。"""
    elapsed = latest_end - earliest_start
    if elapsed <= 0:
        return 0.0
    return total_tokens / elapsed


def results_to_csv(
    results: list[BenchmarkResult],
    filepath: str | Path,
) -> str:
    """将结果列表写入 CSV 文件。"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "engine", "test_type", "batch_size", "prompt_tokens",
        "max_new_tokens",
        "ttft_ms", "median_ttft_ms", "p90_ttft_ms", "p99_ttft_ms",
        "mean_tps",
        "mean_itl_ms", "median_itl_ms", "p90_itl_ms", "p99_itl_ms",
        "mean_tpot_ms", "median_tpot_ms", "p90_tpot_ms", "p99_tpot_ms",
        "e2el_ms", "median_e2el_ms", "p90_e2el_ms", "p99_e2el_ms",
        "peak_vram_mb", "peak_vram_abs_mb",
        "run_id", "timestamp",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "engine": r.engine,
                "test_type": r.test_type,
                "batch_size": r.batch_size,
                "prompt_tokens": r.prompt_tokens,
                "max_new_tokens": r.max_new_tokens,
                "ttft_ms": f"{r.ttft_ms:.2f}",
                "median_ttft_ms": f"{r.median_ttft_ms:.2f}",
                "p90_ttft_ms": f"{r.p90_ttft_ms:.2f}",
                "p99_ttft_ms": f"{r.p99_ttft_ms:.2f}",
                "mean_tps": f"{r.mean_tps:.2f}",
                "mean_itl_ms": f"{r.mean_itl_ms:.2f}",
                "median_itl_ms": f"{r.median_itl_ms:.2f}",
                "p90_itl_ms": f"{r.p90_itl_ms:.2f}",
                "p99_itl_ms": f"{r.p99_itl_ms:.2f}",
                "mean_tpot_ms": f"{r.mean_tpot_ms:.2f}",
                "median_tpot_ms": f"{r.median_tpot_ms:.2f}",
                "p90_tpot_ms": f"{r.p90_tpot_ms:.2f}",
                "p99_tpot_ms": f"{r.p99_tpot_ms:.2f}",
                "e2el_ms": f"{r.e2el_ms:.2f}",
                "median_e2el_ms": f"{r.median_e2el_ms:.2f}",
                "p90_e2el_ms": f"{r.p90_e2el_ms:.2f}",
                "p99_e2el_ms": f"{r.p99_e2el_ms:.2f}",
                "peak_vram_mb": f"{r.peak_vram_mb:.1f}",
                "peak_vram_abs_mb": f"{r.peak_vram_abs_mb:.1f}",
                "run_id": r.run_id,
                "timestamp": r.timestamp,
            })

    return str(filepath)


def make_run_id(timestamp_format: str = "%Y%m%d_%H%M%S") -> str:
    """生成运行 ID。"""
    return f"run_{datetime.now().strftime(timestamp_format)}"