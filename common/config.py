"""配置加载：yaml → dataclass。"""

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
    batch_sizes: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    prompt_length: int = 512
    max_new_tokens: int = 256
    num_warmup: int = 2
    num_runs: int = 3


@dataclass
class SweepConfig:
    start: int = 1
    stop: int = 32
    multiplier: int = 2
    prompt_length: int = 512
    max_new_tokens: int = 256


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