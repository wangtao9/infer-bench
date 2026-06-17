"""指标采集工具：TTFT/TPS 计算和数据记录。"""

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class BenchmarkResult:
    """单次测试结果。"""
    engine: str
    test_type: str          # "single" / "concurrent" / "sweep"
    batch_size: int
    prompt_tokens: int
    max_new_tokens: int
    ttft_ms: float          # 首 Token 延迟 (ms)
    mean_tps: float         # 平均吞吐 (tokens/s)
    peak_vram_mb: float     # 峰值显存增量 (MB)
    run_id: str
    timestamp: str


def compute_ttft_ms(start_time: float, first_token_time: float) -> float:
    """计算首 Token 延迟。"""
    return (first_token_time - start_time) * 1000.0


def compute_tps(total_tokens: int, total_time_s: float) -> float:
    """计算吞吐量。"""
    if total_time_s <= 0:
        return 0.0
    return total_tokens / total_time_s


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
        "max_new_tokens", "ttft_ms", "mean_tps", "peak_vram_mb",
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
                "mean_tps": f"{r.mean_tps:.2f}",
                "peak_vram_mb": f"{r.peak_vram_mb:.1f}",
                "run_id": r.run_id,
                "timestamp": r.timestamp,
            })

    return str(filepath)


def make_run_id(timestamp_format: str = "%Y%m%d_%H%M%S") -> str:
    """生成运行 ID。"""
    return f"run_{datetime.now().strftime(timestamp_format)}"