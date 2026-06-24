#!/usr/bin/env python3
"""LLM 推理引擎性能对比报告生成脚本。

读取 results/ 目录下的 CSV 文件，生成包含 7 张图表的 Markdown 报告。

Usage:
    python generate_report.py [--results-dir results] [--output-dir reports]
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── 全局样式配置 ──────────────────────────────────────────────
ENGINE_COLORS = {"vllm": "#1f77b4", "sglang": "#ff7f0e", "transformers": "#2ca02c"}
ENGINE_LABELS = {"vllm": "vLLM", "sglang": "SGLang", "transformers": "Transformers"}

plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "WenQuanYi Micro Hei", "SimHei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

sns.set_theme(style="whitegrid", font_scale=1.1)

# ── 数据加载 ──────────────────────────────────────────────────

def load_results(results_dir: str) -> pd.DataFrame:
    """加载 results 目录下所有 CSV 文件并合并为 DataFrame。

    过滤掉包含 -1 值的行（表示失败的测试），并打印加载的文件列表。

    Args:
        results_dir: 结果目录路径。

    Returns:
        合并后的 DataFrame。
    """
    csv_pattern = os.path.join(results_dir, "*.csv")
    csv_files = sorted(glob.glob(csv_pattern))

    if not csv_files:
        print(f"[WARNING] No CSV files found in {results_dir}")
        return pd.DataFrame()

    dfs = []
    for f in csv_files:
        print(f"  loaded: {f}")
        dfs.append(pd.read_csv(f))

    df = pd.concat(dfs, ignore_index=True)

    # ── 向后兼容：旧 CSV 的 batch_size 列 → num_requests ──
    if "batch_size" in df.columns and "num_requests" not in df.columns:
        df = df.rename(columns={"batch_size": "num_requests"})
    # ── 旧 CSV 无 request_rate 列，补全为 inf ──
    if "request_rate" not in df.columns:
        df["request_rate"] = float("inf")
    # ── 归一化 request_rate：空值/None → inf ──
    if "request_rate" in df.columns:
        df["request_rate"] = df["request_rate"].replace("", float("inf"))
        df["request_rate"] = df["request_rate"].fillna(float("inf"))
        # 字符串 "inf" → float inf
        df["request_rate"] = df["request_rate"].apply(
            lambda x: float("inf") if str(x).strip().lower() == "inf" else x
        )
        df["request_rate"] = pd.to_numeric(df["request_rate"], errors="coerce").fillna(float("inf"))

    # 过滤掉 -1 值（失败的测试）
    numeric_cols = ["ttft_ms", "mean_tps", "peak_vram_mb", "peak_vram_abs_mb",
                    "median_itl_ms", "median_e2el_ms"]
    for col in numeric_cols:
        if col in df.columns:
            df = df[df[col] != -1]

    df.reset_index(drop=True, inplace=True)
    print(f"  Total rows after filtering: {len(df)}")
    return df


# ── 图表 1: TTFT vs 输入长度（单请求）─────────────────────────

def plot_single_request_ttft(df: pd.DataFrame, output_dir: str) -> None:
    """绘图表 1: 单请求 TTFT vs 输入长度（mean 实线 + P99 虚线）。

    Args:
        df: 测试结果 DataFrame。
        output_dir: 图表输出目录。
    """
    subset = df[df["test_type"] == "single"].copy()
    if subset.empty:
        print("[SKIP] No single-request data for chart 1")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    has_p99 = "p99_ttft_ms" in subset.columns

    for engine in ["vllm", "sglang", "transformers"]:
        eng_data = subset[subset["engine"] == engine].sort_values("prompt_tokens")
        if eng_data.empty:
            continue
        grouped = eng_data.groupby("prompt_tokens")["ttft_ms"].mean().reset_index()
        ax.plot(
            grouped["prompt_tokens"],
            grouped["ttft_ms"],
            marker="o",
            color=ENGINE_COLORS[engine],
            label=ENGINE_LABELS[engine],
            linewidth=2,
        )
        # P99 虚线
        if has_p99:
            grouped_p99 = eng_data.groupby("prompt_tokens")["p99_ttft_ms"].mean().reset_index()
            ax.plot(
                grouped_p99["prompt_tokens"],
                grouped_p99["p99_ttft_ms"],
                marker="o",
                color=ENGINE_COLORS[engine],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
            )

    ax.set_xlabel("输入长度 (tokens)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("TTFT vs 输入长度（单请求）— 实线=mean, 虚线=P99")
    ax.legend(title="引擎")
    fig.tight_layout()

    path = os.path.join(output_dir, "1_ttft_vs_input_length.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


# ── 图表 2: TPS vs 输入长度（单请求）─────────────────────────

def plot_single_request_tps(df: pd.DataFrame, output_dir: str) -> None:
    """绘图表 2: 单请求 TPS vs 输入长度（3 条线）。

    Args:
        df: 测试结果 DataFrame。
        output_dir: 图表输出目录。
    """
    subset = df[df["test_type"] == "single"].copy()
    if subset.empty:
        print("[SKIP] No single-request data for chart 2")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for engine in ["vllm", "sglang", "transformers"]:
        eng_data = subset[subset["engine"] == engine].sort_values("prompt_tokens")
        if eng_data.empty:
            continue
        grouped = eng_data.groupby("prompt_tokens")["mean_tps"].mean().reset_index()
        ax.plot(
            grouped["prompt_tokens"],
            grouped["mean_tps"],
            marker="o",
            color=ENGINE_COLORS[engine],
            label=ENGINE_LABELS[engine],
            linewidth=2,
        )

    ax.set_xlabel("输入长度 (tokens)")
    ax.set_ylabel("吞吐量 (tokens/s)")
    ax.set_title("TPS vs 输入长度（单请求）")
    ax.legend(title="引擎")
    fig.tight_layout()

    path = os.path.join(output_dir, "2_tps_vs_input_length.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


# ── 图表 3: TPS vs 并发数（concurrent）───────────────────────

def plot_concurrent_tps(df: pd.DataFrame, output_dir: str) -> None:
    """绘图表 3: 并发 TPS vs 并发数（3 条线，x 轴 log2 刻度）。

    Args:
        df: 测试结果 DataFrame。
        output_dir: 图表输出目录。
    """
    subset = df[df["test_type"] == "concurrent"].copy()
    if subset.empty:
        print("[SKIP] No concurrent data for chart 3")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for engine in ["vllm", "sglang", "transformers"]:
        eng_data = subset[subset["engine"] == engine].sort_values("num_requests")
        if eng_data.empty:
            continue
        grouped = eng_data.groupby("num_requests")["mean_tps"].mean().reset_index()
        ax.plot(
            grouped["num_requests"],
            grouped["mean_tps"],
            marker="o",
            color=ENGINE_COLORS[engine],
            label=ENGINE_LABELS[engine],
            linewidth=2,
        )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("并发数")
    ax.set_ylabel("吞吐量 (tokens/s)")
    ax.set_title("TPS vs 并发数（并发测试）")
    ax.legend(title="引擎")
    fig.tight_layout()

    path = os.path.join(output_dir, "3_tps_vs_concurrency.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


# ── 图表 4: TTFT vs 并发数（concurrent）──────────────────────

def plot_concurrent_ttft(df: pd.DataFrame, output_dir: str) -> None:
    """绘图表 4: 并发 TTFT vs 并发数（mean 实线 + P99 虚线）。

    Args:
        df: 测试结果 DataFrame。
        output_dir: 图表输出目录。
    """
    subset = df[df["test_type"] == "concurrent"].copy()
    if subset.empty:
        print("[SKIP] No concurrent data for chart 4")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    has_p99 = "p99_ttft_ms" in subset.columns

    for engine in ["vllm", "sglang", "transformers"]:
        eng_data = subset[subset["engine"] == engine].sort_values("num_requests")
        if eng_data.empty:
            continue
        grouped = eng_data.groupby("num_requests")["ttft_ms"].mean().reset_index()
        ax.plot(
            grouped["num_requests"],
            grouped["ttft_ms"],
            marker="o",
            color=ENGINE_COLORS[engine],
            label=ENGINE_LABELS[engine],
            linewidth=2,
        )
        # P99 虚线
        if has_p99:
            grouped_p99 = eng_data.groupby("num_requests")["p99_ttft_ms"].mean().reset_index()
            ax.plot(
                grouped_p99["num_requests"],
                grouped_p99["p99_ttft_ms"],
                marker="o",
                color=ENGINE_COLORS[engine],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
            )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("并发数")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("TTFT vs 并发数（并发测试）— 实线=mean, 虚线=P99")
    ax.legend(title="引擎")
    fig.tight_layout()

    path = os.path.join(output_dir, "4_ttft_vs_concurrency.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


# ── 图表 5: TPS & TTFT vs 并发数双 Y 轴图（sweep）────────────

def plot_sweep_dual_axis(df: pd.DataFrame, output_dir: str) -> None:
    """绘图表 5: sweep 数据双 Y 轴图（左轴 TPS 实线，右轴 TTFT 虚线）。

    Args:
        df: 测试结果 DataFrame。
        output_dir: 图表输出目录。
    """
    subset = df[df["test_type"] == "sweep"].copy()
    if subset.empty:
        print("[SKIP] No sweep data for chart 5")
        return

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax2 = ax1.twinx()

    handles = []
    labels = []

    for engine in ["vllm", "sglang", "transformers"]:
        eng_data = subset[subset["engine"] == engine].sort_values("num_requests")
        if eng_data.empty:
            continue
        grouped = eng_data.groupby("num_requests").agg(
            mean_tps=("mean_tps", "mean"),
            ttft_ms=("ttft_ms", "mean"),
        ).reset_index()

        color = ENGINE_COLORS[engine]
        lbl = ENGINE_LABELS[engine]

        # TPS — 左轴实线
        h1, = ax1.plot(
            grouped["num_requests"],
            grouped["mean_tps"],
            marker="o",
            color=color,
            linestyle="-",
            linewidth=2,
            label=f"{lbl} TPS",
        )
        handles.append(h1)
        labels.append(f"{lbl} TPS")

        # TTFT — 右轴虚线
        h2, = ax2.plot(
            grouped["num_requests"],
            grouped["ttft_ms"],
            marker="s",
            color=color,
            linestyle="--",
            linewidth=2,
            label=f"{lbl} TTFT",
        )
        handles.append(h2)
        labels.append(f"{lbl} TTFT")

    ax1.set_xscale("log", base=2)
    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax1.set_xlabel("并发数")
    ax1.set_ylabel("吞吐量 TPS (tokens/s)")
    ax2.set_ylabel("TTFT (ms)")
    ax1.set_title("TPS & TTFT vs 并发数（渐进扫描）")

    ax1.legend(handles, labels, title="指标", loc="best", fontsize=9)
    fig.tight_layout()

    path = os.path.join(output_dir, "5_sweep_tps_ttft.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


# ── 图表 6: 峰值显存 vs 并发数分组柱状图 ───────────────────────

def plot_vram_comparison(df: pd.DataFrame, output_dir: str) -> None:
    """绘图表 6: 峰值显存 vs 并发数分组柱状图（concurrent + sweep 数据）。

    Args:
        df: 测试结果 DataFrame。
        output_dir: 图表输出目录。
    """
    subset = df[df["test_type"].isin(["concurrent", "sweep"])].copy()
    if subset.empty:
        print("[SKIP] No concurrent/sweep data for chart 6")
        return

    # 按 engine + num_requests 聚合平均 VRAM
    # 优先使用 peak_vram_abs_mb（绝对占用量），vLLM/SGLang 预分配引擎下
    # peak_vram_mb（增量）始终为 0，abs 才有意义
    vram_col = "peak_vram_abs_mb" if "peak_vram_abs_mb" in subset.columns else "peak_vram_mb"
    grouped = (
        subset.groupby(["engine", "num_requests"])[vram_col]
        .mean()
        .reset_index()
    )

    engines_present = grouped["engine"].unique().tolist()
    num_requests_list = sorted(grouped["num_requests"].unique())

    x = np.arange(len(num_requests_list))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, engine in enumerate(["vllm", "sglang", "transformers"]):
        if engine not in engines_present:
            continue
        eng_data = grouped[grouped["engine"] == engine]
        vram_vals = []
        for nr in num_requests_list:
            row = eng_data[eng_data["num_requests"] == nr]
            vram_vals.append(row[vram_col].values[0] if not row.empty else 0)

        bars = ax.bar(
            x + i * width,
            vram_vals,
            width,
            color=ENGINE_COLORS[engine],
            label=ENGINE_LABELS[engine],
        )

        # 在柱子上方标注数值
        for bar, val in zip(bars, vram_vals):
            if val > 0:
                ax.annotate(
                    f"{val:.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x + width)
    ax.set_xticklabels([str(nr) for nr in num_requests_list])
    ax.set_xlabel("并发数")
    ax.set_ylabel("峰值显存 (MB)")
    ax.set_title("峰值显存 vs 并发数（分组柱状图）")
    ax.legend(title="引擎")
    fig.tight_layout()

    path = os.path.join(output_dir, "6_vram_vs_concurrency.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


# ── 图表 8: ITL P99 vs 输入长度（单请求）─────────────────────


def plot_single_request_itl(df: pd.DataFrame, output_dir: str) -> None:
    """绘图表 8: 单请求 ITL P99 vs 输入长度。"""
    subset = df[df["test_type"] == "single"].copy()
    if subset.empty or "p99_itl_ms" not in subset.columns:
        print("[SKIP] No single-request ITL data for chart 8")
        return
    # 过滤掉 ITL 为 -1 的行（Transformers 不可测）
    subset = subset[subset["p99_itl_ms"] > 0]
    if subset.empty:
        print("[SKIP] No valid ITL data (all -1 or missing)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for engine in ["vllm", "sglang", "transformers"]:
        eng_data = subset[subset["engine"] == engine].sort_values("prompt_tokens")
        if eng_data.empty:
            continue
        grouped = eng_data.groupby("prompt_tokens")["p99_itl_ms"].mean().reset_index()
        ax.plot(
            grouped["prompt_tokens"],
            grouped["p99_itl_ms"],
            marker="o",
            color=ENGINE_COLORS[engine],
            label=ENGINE_LABELS[engine],
            linewidth=2,
        )

    ax.set_xlabel("输入长度 (tokens)")
    ax.set_ylabel("ITL P99 (ms)")
    ax.set_title("ITL P99 vs 输入长度（单请求）")
    ax.legend(title="引擎")
    fig.tight_layout()

    path = os.path.join(output_dir, "8_itl_vs_input_length.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


# ── 图表 9: TPOT P99 vs 并发数 ────────────────────────────────


def plot_concurrent_tpot(df: pd.DataFrame, output_dir: str) -> None:
    """绘图表 9: TPOT P99 vs 并发数（concurrent + sweep 数据）。"""
    subset = df[df["test_type"].isin(["concurrent", "sweep"])].copy()
    if subset.empty or "p99_tpot_ms" not in subset.columns:
        print("[SKIP] No TPOT data for chart 9")
        return
    subset = subset[subset["p99_tpot_ms"] > 0]
    if subset.empty:
        print("[SKIP] No valid TPOT data (all -1 or missing)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for engine in ["vllm", "sglang", "transformers"]:
        eng_data = subset[subset["engine"] == engine].sort_values("num_requests")
        if eng_data.empty:
            continue
        grouped = eng_data.groupby("num_requests")["p99_tpot_ms"].mean().reset_index()
        ax.plot(
            grouped["num_requests"],
            grouped["p99_tpot_ms"],
            marker="o",
            color=ENGINE_COLORS[engine],
            label=ENGINE_LABELS[engine],
            linewidth=2,
        )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("并发数")
    ax.set_ylabel("TPOT P99 (ms)")
    ax.set_title("TPOT P99 vs 并发数")
    ax.legend(title="引擎")
    fig.tight_layout()

    path = os.path.join(output_dir, "9_tpot_vs_concurrency.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


# ── 图表 10: E2EL P99 vs 并发数 ───────────────────────────────


def plot_concurrent_e2el(df: pd.DataFrame, output_dir: str) -> None:
    """绘图表 10: E2EL P99 vs 并发数（concurrent + sweep 数据）。"""
    subset = df[df["test_type"].isin(["concurrent", "sweep"])].copy()
    if subset.empty or "p99_e2el_ms" not in subset.columns:
        print("[SKIP] No E2EL data for chart 10")
        return
    subset = subset[subset["p99_e2el_ms"] > 0]
    if subset.empty:
        print("[SKIP] No valid E2EL data (all -1 or missing)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for engine in ["vllm", "sglang", "transformers"]:
        eng_data = subset[subset["engine"] == engine].sort_values("num_requests")
        if eng_data.empty:
            continue
        grouped = eng_data.groupby("num_requests")["p99_e2el_ms"].mean().reset_index()
        ax.plot(
            grouped["num_requests"],
            grouped["p99_e2el_ms"],
            marker="o",
            color=ENGINE_COLORS[engine],
            label=ENGINE_LABELS[engine],
            linewidth=2,
        )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("并发数")
    ax.set_ylabel("E2EL P99 (ms)")
    ax.set_title("E2EL P99 vs 并发数")
    ax.legend(title="引擎")
    fig.tight_layout()

    path = os.path.join(output_dir, "10_e2el_vs_concurrency.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


# ── 图表 11: TPS vs Request Rate（Poisson 模式）─────────────────


def plot_poisson_tps(df: pd.DataFrame, output_dir: str) -> None:
    """绘图表 11: TPS vs Request Rate（Poisson 调度并发测试）。

    仅绘制 request_rate 非空的数据行，x 轴为 request_rate (req/s)。

    Args:
        df: 测试结果 DataFrame。
        output_dir: 图表输出目录。
    """
    subset = df[df["test_type"] == "concurrent"].copy()
    if subset.empty or "request_rate" not in subset.columns:
        print("[SKIP] No Poisson data for chart 11")
        return
    # 过滤出 request_rate 为有限值（Poisson 模式）的行
    subset = subset[subset["request_rate"] != float("inf")]
    if subset.empty:
        print("[SKIP] No Poisson data for chart 11")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for engine in ["vllm", "sglang", "transformers"]:
        eng_data = subset[subset["engine"] == engine].sort_values("request_rate")
        if eng_data.empty:
            continue
        grouped = eng_data.groupby("request_rate")["mean_tps"].mean().reset_index()
        ax.plot(
            grouped["request_rate"],
            grouped["mean_tps"],
            marker="o",
            color=ENGINE_COLORS[engine],
            label=ENGINE_LABELS[engine],
            linewidth=2,
        )

    ax.set_xlabel("Request Rate (req/s)")
    ax.set_ylabel("吞吐量 (tokens/s)")
    ax.set_title("TPS vs Request Rate（Poisson 调度）")
    ax.legend(title="引擎")
    fig.tight_layout()

    path = os.path.join(output_dir, "11_tps_vs_request_rate.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


# ── 图表 12: TTFT vs Request Rate（Poisson 模式）────────────────


def plot_poisson_ttft(df: pd.DataFrame, output_dir: str) -> None:
    """绘图表 12: TTFT vs Request Rate（Poisson 调度并发测试）。

    仅绘制 request_rate 非空的数据行，mean 实线 + P99 虚线。

    Args:
        df: 测试结果 DataFrame。
        output_dir: 图表输出目录。
    """
    subset = df[df["test_type"] == "concurrent"].copy()
    if subset.empty or "request_rate" not in subset.columns:
        print("[SKIP] No Poisson data for chart 12")
        return
    # 过滤出 request_rate 为有限值（Poisson 模式）的行
    subset = subset[subset["request_rate"] != float("inf")]
    if subset.empty:
        print("[SKIP] No Poisson data for chart 12")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    has_p99 = "p99_ttft_ms" in subset.columns

    for engine in ["vllm", "sglang", "transformers"]:
        eng_data = subset[subset["engine"] == engine].sort_values("request_rate")
        if eng_data.empty:
            continue
        grouped = eng_data.groupby("request_rate")["ttft_ms"].mean().reset_index()
        ax.plot(
            grouped["request_rate"],
            grouped["ttft_ms"],
            marker="o",
            color=ENGINE_COLORS[engine],
            label=ENGINE_LABELS[engine],
            linewidth=2,
        )
        if has_p99:
            grouped_p99 = eng_data.groupby("request_rate")["p99_ttft_ms"].mean().reset_index()
            ax.plot(
                grouped_p99["request_rate"],
                grouped_p99["p99_ttft_ms"],
                marker="o",
                color=ENGINE_COLORS[engine],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
            )

    ax.set_xlabel("Request Rate (req/s)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("TTFT vs Request Rate（Poisson 调度）— 实线=mean, 虚线=P99")
    ax.legend(title="引擎")
    fig.tight_layout()

    path = os.path.join(output_dir, "12_ttft_vs_request_rate.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


# ── 图表 7: 综合 Radar 图 ─────────────────────────────────────

def plot_radar(df: pd.DataFrame, output_dir: str) -> None:
    """绘图表 7: 综合雷达图（4 维度: TTFT↓, ITL↓, TPS↑, 显存效率↑）。

    归一化方法:
    - TTFT: 取倒数后归一化到 0-1（越小越好 → 越大越好）
    - ITL P99: 取倒数后归一化到 0-1（越小越好 → 越大越好）
    - TPS: 直接归一化到 0-1（越大越好）
    - 显存效率 = TPS / VRAM: 归一化到 0-1（越大越好）

    Args:
        df: 测试结果 DataFrame。
        output_dir: 图表输出目录。
    """
    if df.empty:
        print("[SKIP] No data for chart 7")
        return

    # 计算每个引擎的平均指标
    # 优先使用 peak_vram_abs_mb（绝对占用量），vLLM/SGLang 预分配引擎下增量无意义
    vram_col = "peak_vram_abs_mb" if "peak_vram_abs_mb" in df.columns else "peak_vram_mb"
    agg_dict = {
        "avg_ttft": ("ttft_ms", "mean"),
        "avg_tps": ("mean_tps", "mean"),
        "avg_vram": (vram_col, "mean"),
    }
    # ITL P99: 如果有数据，取均值（过滤 -1）
    if "p99_itl_ms" in df.columns:
        # 临时替换 -1 为 NaN 以便 mean() 忽略
        df_temp = df.copy()
        df_temp.loc[df_temp["p99_itl_ms"] < 0, "p99_itl_ms"] = float("nan")
        avg_itl = df_temp.groupby("engine")["p99_itl_ms"].mean()
        has_itl = True
    else:
        has_itl = False

    engine_stats = df.groupby("engine").agg(**agg_dict).reset_index()

    if len(engine_stats) == 0:
        print("[SKIP] Not enough data for radar chart")
        return

    # 计算分数
    scores = {}
    for _, row in engine_stats.iterrows():
        eng = row["engine"]
        scores[eng] = {
            "ttft_inv": 1.0 / row["avg_ttft"] if row["avg_ttft"] > 0 else 0,
            "tps": row["avg_tps"],
            "vram_eff": row["avg_tps"] / row["avg_vram"] if row["avg_vram"] > 0 else 0,
        }
        if has_itl and eng in avg_itl.index and not np.isnan(avg_itl[eng]) and avg_itl[eng] > 0:
            scores[eng]["itl_inv"] = 1.0 / avg_itl[eng]
        else:
            scores[eng]["itl_inv"] = 0

    # 归一化到 0-1
    if has_itl:
        categories = ["TTFT↓", "ITL↓", "TPS↑", "显存效率↑"]
        raw_keys = ["ttft_inv", "itl_inv", "tps", "vram_eff"]
    else:
        categories = ["TTFT↓", "TPS↑", "显存效率↑"]
        raw_keys = ["ttft_inv", "tps", "vram_eff"]

    for key in raw_keys:
        vals = [scores[e][key] for e in scores]
        min_v, max_v = min(vals), max(vals)
        rng = max_v - min_v if max_v != min_v else 1.0
        for eng in scores:
            scores[eng][key] = (scores[eng][key] - min_v) / rng

    # 雷达图绘制
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    engines_plot = [e for e in ["vllm", "sglang", "transformers"] if e in scores]

    for engine in engines_plot:
        values = [scores[engine][k] for k in raw_keys]
        values += values[:1]  # 闭合
        ax.plot(angles, values, color=ENGINE_COLORS[engine], linewidth=2,
                label=ENGINE_LABELS[engine])
        ax.fill(angles, values, color=ENGINE_COLORS[engine], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8)
    ax.set_title("综合性能雷达图", y=1.08, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()

    path = os.path.join(output_dir, "7_radar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


# ── Markdown 报告生成 ─────────────────────────────────────────

def generate_markdown_report(
    df: pd.DataFrame,
    output_dir: str,
    report_path: str,
) -> None:
    """生成 Markdown 格式的性能对比报告。

    Args:
        df: 测试结果 DataFrame。
        output_dir: 图表所在目录。
        report_path: Markdown 报告输出路径。
    """
    lines: list[str] = []

    lines.append("# LLM 推理引擎性能对比报告\n")

    # ── 测试环境 ──
    lines.append("## 测试环境\n")
    run_ids = df["run_id"].unique().tolist() if not df.empty else []
    engines = df["engine"].unique().tolist() if not df.empty else []
    lines.append(f"- **GPU**: (见配置)")
    lines.append(f"- **模型**: (见配置)")
    lines.append(f"- **引擎**: {', '.join(ENGINE_LABELS.get(e, e) for e in engines)}")
    lines.append(f"- **运行 ID**: {', '.join(run_ids)}")
    lines.append("")

    # ── Transformers 批处理语义说明 ──
    lines.append("## 重要说明\n")
    lines.append(
        "> **Transformers 引擎的批处理语义与 vLLM/SGLang 不同**: "
        "Transformers 使用同步批量推理（所有请求作为一个 batch 同时处理），"
        "而 vLLM/SGLang 使用连续批处理（continuous batching，请求可以动态加入/离开）。"
        "因此，并发测试中 Transformers 的 TTFT 和 TPS 指标与传统 HTTP 服务端引擎"
        "不完全可比，需结合具体部署场景理解。\n"
    )

    # ── Section 1: 单请求延迟 ──
    lines.append("## 1. 单请求延迟\n")
    single = df[df["test_type"] == "single"] if not df.empty else pd.DataFrame()
    has_p99 = "p99_ttft_ms" in df.columns if not df.empty else False
    has_itl = "p99_itl_ms" in df.columns if not df.empty else False
    if not single.empty:
        # 表头
        cols = "| 引擎 | 输入长度 (tokens) | TTFT mean (ms)"
        sep = "|------|-------------------|---------------"
        row_fmt = "f'| {{eng}} | {{prompt}} | {{ttft:.2f}}"
        if has_p99:
            cols += " | TTFT P99 (ms)"
            sep += "|--------------"
            row_fmt += " | {{p99_ttft:.2f}}"
        if has_itl:
            cols += " | ITL P99 (ms)"
            sep += "|-------------"
            row_fmt += " | {{p99_itl:.2f}}" if has_itl else ""
        cols += " | TPS (tokens/s) |"
        sep += "|----------------|"
        row_fmt += " | {{tps:.2f}} |'"
        lines.append(cols)
        lines.append(sep)
        for _, row in single.sort_values(["engine", "prompt_tokens"]).iterrows():
            eng = ENGINE_LABELS.get(row['engine'], row['engine'])
            vals = {
                "eng": eng,
                "prompt": int(row['prompt_tokens']),
                "ttft": row['ttft_ms'],
                "tps": row['mean_tps'],
            }
            if has_p99:
                vals["p99_ttft"] = row['p99_ttft_ms']
            if has_itl:
                itl_val = row['p99_itl_ms']
                vals["p99_itl"] = "N/A" if itl_val < 0 else f"{itl_val:.2f}"
            # 手动组装行以处理 N/A
            parts = [eng, str(int(row['prompt_tokens'])), f"{row['ttft_ms']:.2f}"]
            if has_p99:
                parts.append(f"{row['p99_ttft_ms']:.2f}")
            if has_itl:
                itl_val = row['p99_itl_ms']
                parts.append("N/A" if itl_val < 0 else f"{itl_val:.2f}")
            parts.append(f"{row['mean_tps']:.2f}")
            lines.append("| " + " | ".join(parts) + " |")
        lines.append("")
    else:
        lines.append("*无单请求数据*\n")

    lines.append("### TTFT vs 输入长度\n")
    lines.append("![TTFT vs 输入长度](1_ttft_vs_input_length.png)\n")
    lines.append("### TPS vs 输入长度\n")
    lines.append("![TPS vs 输入长度](2_tps_vs_input_length.png)\n")
    lines.append("### ITL P99 vs 输入长度\n")
    lines.append("![ITL P99 vs 输入长度](8_itl_vs_input_length.png)\n")

    # ── Section 2: 并发吞吐 ──
    lines.append("## 2. 并发吞吐\n")
    concurrent = df[df["test_type"] == "concurrent"] if not df.empty else pd.DataFrame()
    if not concurrent.empty:
        has_p99 = "p99_ttft_ms" in concurrent.columns
        has_e2el = "p99_e2el_ms" in concurrent.columns
        has_rate = "request_rate" in concurrent.columns
        cols = "| 引擎 | 请求数 | Request Rate | TTFT mean (ms)"
        sep = "|------|--------|--------------|---------------"
        if has_p99:
            cols += " | TTFT P99 (ms)"
            sep += "|--------------"
        if has_e2el:
            cols += " | E2EL P99 (ms)"
            sep += "|--------------"
        cols += " | TPS (tokens/s) | 峰值显存 (MB) |"
        sep += "|----------------|---------------|"
        lines.append(cols)
        lines.append(sep)
        for _, row in concurrent.sort_values(["engine", "num_requests"]).iterrows():
            eng = ENGINE_LABELS.get(row['engine'], row['engine'])
            vram = row.get("peak_vram_abs_mb", row.get("peak_vram_mb", 0))
            rate_val = row.get("request_rate", float("inf"))
            try:
                rate_f = float(rate_val)
            except (ValueError, TypeError):
                rate_f = float("inf")
            rate_str = "batch (inf)" if rate_f == float("inf") else f"{rate_f:.1f}"
            parts = [eng, str(int(row['num_requests'])), rate_str, f"{row['ttft_ms']:.2f}"]
            if has_p99:
                parts.append(f"{row['p99_ttft_ms']:.2f}")
            if has_e2el:
                parts.append(f"{row['p99_e2el_ms']:.2f}")
            parts.append(f"{row['mean_tps']:.2f}")
            parts.append(f"{vram:.0f}")
            lines.append("| " + " | ".join(parts) + " |")
        lines.append("")
    else:
        lines.append("*无并发数据*\n")

    lines.append("### TPS vs 并发数\n")
    lines.append("![TPS vs 并发数](3_tps_vs_concurrency.png)\n")
    lines.append("### TTFT vs 并发数\n")
    lines.append("![TTFT vs 并发数](4_ttft_vs_concurrency.png)\n")
    lines.append("### TPOT P99 vs 并发数\n")
    lines.append("![TPOT P99 vs 并发数](9_tpot_vs_concurrency.png)\n")
    lines.append("### E2EL P99 vs 并发数\n")
    lines.append("![E2EL P99 vs 并发数](10_e2el_vs_concurrency.png)\n")

    # ── Poisson 模式图表（仅在有数据时展示）──
    if not concurrent.empty and "request_rate" in concurrent.columns:
        poisson_data = concurrent[concurrent["request_rate"] != float("inf")]
        if not poisson_data.empty:
            lines.append("### TPS vs Request Rate（Poisson 调度）\n")
            lines.append("![TPS vs Request Rate](11_tps_vs_request_rate.png)\n")
            lines.append("### TTFT vs Request Rate（Poisson 调度）\n")
            lines.append("![TTFT vs Request Rate](12_ttft_vs_request_rate.png)\n")

    # ── Section 3: 渐进并发扫描 ──
    lines.append("## 3. 渐进并发扫描\n")
    sweep = df[df["test_type"] == "sweep"] if not df.empty else pd.DataFrame()
    if not sweep.empty:
        has_p99 = "p99_ttft_ms" in sweep.columns
        has_e2el = "p99_e2el_ms" in sweep.columns
        cols = "| 引擎 | 并发数 | TTFT mean (ms)"
        sep = "|------|--------|---------------"
        if has_p99:
            cols += " | TTFT P99 (ms)"
            sep += "|--------------"
        if has_e2el:
            cols += " | E2EL P99 (ms)"
            sep += "|--------------"
        cols += " | TPS (tokens/s) | 峰值显存 (MB) |"
        sep += "|----------------|---------------|"
        lines.append(cols)
        lines.append(sep)
        for _, row in sweep.sort_values(["engine", "num_requests"]).iterrows():
            eng = ENGINE_LABELS.get(row['engine'], row['engine'])
            vram = row.get("peak_vram_abs_mb", row.get("peak_vram_mb", 0))
            parts = [eng, str(int(row['num_requests'])), f"{row['ttft_ms']:.2f}"]
            if has_p99:
                parts.append(f"{row['p99_ttft_ms']:.2f}")
            if has_e2el:
                parts.append(f"{row['p99_e2el_ms']:.2f}")
            parts.append(f"{row['mean_tps']:.2f}")
            parts.append(f"{vram:.0f}")
            lines.append("| " + " | ".join(parts) + " |")
        lines.append("")
    else:
        lines.append("*无扫描数据*\n")

    lines.append("### TPS & TTFT vs 并发数（双 Y 轴）\n")
    lines.append("![TPS & TTFT vs 并发数](5_sweep_tps_ttft.png)\n")

    # ── Section 4: 显存占用对比 ──
    lines.append("## 4. 显存占用对比\n")
    lines.append("![峰值显存 vs 并发数](6_vram_vs_concurrency.png)\n")

    # ── Section 5: 综合评价 ──
    lines.append("## 5. 综合评价\n")
    lines.append("![综合雷达图](7_radar.png)\n")
    lines.append("### 维度说明\n")
    lines.append("| 维度 | 含义 | 方向 |")
    lines.append("|------|------|------|")
    lines.append("| TTFT↓ | 首 Token 延迟的倒数（越低越好） | 归一化前取倒数 → 值越大越好 |")
    if has_itl:
        lines.append("| ITL↓ | Inter-Token Latency P99 的倒数（越低越好） | 归一化前取倒数 → 值越大越好 |")
    lines.append("| TPS↑ | 平均吞吐量 | 越高越好 |")
    lines.append("| 显存效率↑ | TPS / 峰值显存 | 单位显存产出的吞吐，越高越好 |")
    lines.append("")
    lines.append(
        "三个维度均归一化到 [0, 1] 区间，其中 1 表示该维度最优，0 表示最差。"
        "雷达图面积越大，综合性能越优。\n"
    )

    # 写入文件
    report_text = "\n".join(lines)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  saved: {report_path}")


# ── 主入口 ────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LLM inference benchmark report with charts"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing CSV result files (default: results)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory for output charts and report (default: reports)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir

    # 加载数据
    print(f"Loading results from {results_dir}/ ...")
    df = load_results(results_dir)

    if df.empty:
        print("[ERROR] No valid data found. Exiting.")
        return

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 生成图表
    print("\nGenerating charts...")
    plot_single_request_ttft(df, output_dir)
    plot_single_request_tps(df, output_dir)
    plot_concurrent_tps(df, output_dir)
    plot_concurrent_ttft(df, output_dir)
    plot_sweep_dual_axis(df, output_dir)
    plot_vram_comparison(df, output_dir)
    plot_radar(df, output_dir)
    plot_single_request_itl(df, output_dir)
    plot_concurrent_tpot(df, output_dir)
    plot_concurrent_e2el(df, output_dir)
    plot_poisson_tps(df, output_dir)
    plot_poisson_ttft(df, output_dir)

    # 生成 Markdown 报告
    print("\nGenerating markdown report...")
    report_path = os.path.join(output_dir, "benchmark_report.md")
    generate_markdown_report(df, output_dir, report_path)

    print("\nDone!")


if __name__ == "__main__":
    main()