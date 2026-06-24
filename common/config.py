"""配置加载：yaml → dataclass。"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-7B-Instruct"
    path: Optional[str] = None


@dataclass
class GpuConfig:
    device: str = "0"
    monitor_interval_ms: int = 100


@dataclass
class SingleRequestConfig:
    prompt_lengths: list[int] = field(default_factory=lambda: [128, 512, 1024])
    max_new_tokens: int = 256
    num_warmup: int = 3
    num_runs: int = 5


@dataclass
class ConcurrentConfig:
    num_requests: int = 64                    # 总请求数（与 vLLM --num-prompts 对齐）
    prompt_length: int = 512
    max_new_tokens: int = 256
    num_warmup: int = 2
    request_rate: list[float] = field(default_factory=lambda: [float("inf")])
    # 请求速率列表，每值一轮测试。inf=batch（同时发出），有限值=Poisson 调度。
    # 与 vLLM bench serve --request-rate 语义对齐。

    def validate(self) -> None:
        """校验配置语义。"""
        if self.num_requests <= 0:
            raise ValueError("num_requests 必须大于 0")
        if not self.request_rate:
            raise ValueError("request_rate 列表不能为空")
        for r in self.request_rate:
            # inf 是合法值（batch 模式）
            if isinstance(r, float) and r == float("inf"):
                continue
            if r <= 0:
                raise ValueError(f"request_rate 中的值必须大于 0，收到 {r}")


@dataclass
class SweepConfig:
    start: int = 1
    stop: int = 32
    multiplier: int = 2
    prompt_length: int = 512
    max_new_tokens: int = 256
    # Sweep 始终 batch 模式，不支持 Poisson 调度


@dataclass
class TestConfig:
    single_request: SingleRequestConfig = field(default_factory=SingleRequestConfig)
    concurrent: ConcurrentConfig = field(default_factory=ConcurrentConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)


@dataclass
class EngineVllmConfig:
    port: int = 8000
    extra_args: str = "--gpu-memory-utilization 0.9"


@dataclass
class EngineSglangConfig:
    port: int = 8001
    extra_args: str = "--mem-fraction-static 0.9"


@dataclass
class EngineTransformersConfig:
    dtype: str = "float16"
    device_map: str = "auto"


@dataclass
class EnginesConfig:
    vllm: EngineVllmConfig = field(default_factory=EngineVllmConfig)
    sglang: EngineSglangConfig = field(default_factory=EngineSglangConfig)
    transformers: EngineTransformersConfig = field(default_factory=EngineTransformersConfig)


@dataclass
class OutputConfig:
    results_dir: str = "results"
    reports_dir: str = "reports"
    timestamp_format: str = "%Y%m%d_%H%M%S"


@dataclass
class BenchmarkConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    gpu: GpuConfig = field(default_factory=GpuConfig)
    test: TestConfig = field(default_factory=TestConfig)
    engines: EnginesConfig = field(default_factory=EnginesConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(path: str | Path = "config.yaml") -> BenchmarkConfig:
    """从 YAML 文件加载配置，返回 BenchmarkConfig 实例。"""
    path = Path(path)
    if not path.exists():
        return BenchmarkConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    model_raw = raw.get("model", {})
    gpu_raw = raw.get("gpu", {})
    test_raw = raw.get("test", {})
    engines_raw = raw.get("engines", {})
    output_raw = raw.get("output", {})

    sr_raw = test_raw.get("single_request", {})
    cc_raw = test_raw.get("concurrent", {})
    sw_raw = test_raw.get("sweep", {})

    # ── 向后兼容：batch_sizes → num_requests + request_rate=[inf] ──
    if "batch_sizes" in cc_raw and "num_requests" not in cc_raw:
        warnings.warn(
            "'batch_sizes' 已废弃，请使用 'num_requests' + 'request_rate' 代替",
            DeprecationWarning,
            stacklevel=2,
        )
        bs = cc_raw.pop("batch_sizes")
        cc_raw["num_requests"] = bs[-1] if isinstance(bs, list) and bs else 16
        cc_raw["request_rate"] = [float("inf")]

    # ── 向后兼容：num_requests 为列表 → 取最大值 + request_rate=[inf] ──
    if "num_requests" in cc_raw and isinstance(cc_raw["num_requests"], list):
        warnings.warn(
            "num_requests 不再支持列表，已取最大值并设定 request_rate=[inf]",
            DeprecationWarning,
            stacklevel=2,
        )
        cc_raw["num_requests"] = max(cc_raw["num_requests"])
        cc_raw.setdefault("request_rate", [float("inf")])

    # ── 向后兼容：num_runs 已移除 ──
    if "num_runs" in cc_raw:
        cc_raw.pop("num_runs")

    # ── 归一化：request_rate 单个 float/int → [float]，字符串 "inf" → inf ──
    if "request_rate" in cc_raw:
        rr = cc_raw["request_rate"]
        if rr is None:
            cc_raw["request_rate"] = [float("inf")]
        elif isinstance(rr, (int, float)):
            cc_raw["request_rate"] = [float(rr)]
        elif isinstance(rr, list):
            normalized = []
            for v in rr:
                if isinstance(v, str) and v.strip().lower() == "inf":
                    normalized.append(float("inf"))
                elif isinstance(v, (int, float)):
                    normalized.append(float(v))
                else:
                    normalized.append(v)
            cc_raw["request_rate"] = normalized
    else:
        cc_raw.setdefault("request_rate", [float("inf")])

    # ── SweepConfig 忽略已废弃的 request_rate 字段 ──
    sw_raw.pop("request_rate", None)

    return BenchmarkConfig(
        model=ModelConfig(**model_raw),
        gpu=GpuConfig(**gpu_raw),
        test=TestConfig(
            single_request=SingleRequestConfig(**sr_raw),
            concurrent=ConcurrentConfig(**cc_raw),
            sweep=SweepConfig(**sw_raw),
        ),
        engines=EnginesConfig(
            vllm=EngineVllmConfig(**engines_raw.get("vllm", {})),
            sglang=EngineSglangConfig(**engines_raw.get("sglang", {})),
            transformers=EngineTransformersConfig(**engines_raw.get("transformers", {})),
        ),
        output=OutputConfig(**output_raw),
    )