# LLM 推理引擎基准测试 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建可重复运行的推理引擎基准测试项目，对比 vLLM、SGLang、原生 Transformers 的 TTFT/TPS/显存性能，输出 Markdown 报告 + 7 张图表。

**Architecture:** 每个引擎独立运行测试脚本，共享 `common/` 工具模块（配置、指标、客户端、Prompt、GPU 监控）。vLLM/SGLang 走 OpenAI-compatible HTTP API，Transformers 进程内调用 generate()。结果输出 CSV，最终由汇总脚本生成报告。

**Tech Stack:** Python 3.10+, vLLM, SGLang, transformers, torch, pynvml, aiohttp, pyyaml, matplotlib, seaborn, pandas

---

## File Structure

```
infer-bench/
├── config.yaml                  # 全局配置
├── requirements.txt             # 依赖
├── common/
│   ├── __init__.py              # 包初始化
│   ├── config.py                # 配置加载（yaml → dataclass）
│   ├── prompts.py               # 种子 prompt + 长度扩展 + tokenize
│   ├── gpu.py                   # GPU 显存后台监控
│   ├── metrics.py               # TTFT/TPS 计算
│   └── client.py                # OpenAI-compatible HTTP 客户端
├── run_vllm.py                  # vLLM 测试脚本
├── run_sglang.py                # SGLang 测试脚本
├── run_transformers.py          # Transformers 测试脚本
├── run_sweep.py                 # 渐进并发扫描
├── generate_report.py           # CSV → Markdown 报告 + 图表
├── results/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
└── README.md
```

---

### Task 1: 项目骨架 + 配置模块

**Files:**
- Create: `config.yaml`
- Create: `requirements.txt`
- Create: `common/__init__.py`
- Create: `common/config.py`
- Create: `results/.gitkeep`
- Create: `reports/.gitkeep`

- [ ] **Step 1: 创建项目目录结构和 .gitkeep**

```bash
mkdir -p common results reports
touch results/.gitkeep reports/.gitkeep
```

- [ ] **Step 2: 创建 config.yaml**

```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  path: null  # 本地路径，null 则从 HuggingFace 下载

gpu:
  device: "0"
  monitor_interval_ms: 100

test:
  single_request:
    prompt_lengths: [128, 512, 1024]
    max_new_tokens: 256
    num_warmup: 3
    num_runs: 5

  concurrent:
    batch_sizes: [1, 2, 4, 8, 16]
    prompt_length: 512
    max_new_tokens: 256
    num_warmup: 2
    num_runs: 3

  sweep:
    start: 1
    stop: 32
    multiplier: 2  # 1, 2, 4, 8, 16, 32
    prompt_length: 512
    max_new_tokens: 256

engines:
  vllm:
    port: 8000
    extra_args: "--gpu-memory-utilization 0.9"
  sglang:
    port: 8001
    extra_args: "--mem-fraction-static 0.9"
  transformers:
    dtype: "float16"
    device_map: "auto"

output:
  results_dir: "results"
  reports_dir: "reports"
  timestamp_format: "%Y%m%d_%H%M%S"
```

- [ ] **Step 3: 创建 requirements.txt**

```
vllm
sglang[all]
torch
transformers
accelerate
pynvml
aiohttp
pyyaml
matplotlib
seaborn
pandas
```

- [ ] **Step 4: 创建 common/__init__.py**

```python
"""infer-bench 共享工具模块。"""
```

- [ ] **Step 5: 创建 common/config.py**

```python
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

    # 逐层解析，缺失字段使用 dataclass 默认值
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
```

- [ ] **Step 6: 验证配置加载**

```bash
cd /Users/wt/share/python/infer-bench1
python -c "from common.config import load_config; cfg = load_config(); print(cfg.model.name, cfg.test.single_request.prompt_lengths, cfg.engines.vllm.port)"
```

Expected: `Qwen/Qwen2.5-7B-Instruct [128, 512, 1024] 8000`

- [ ] **Step 7: 提交**

```bash
git add config.yaml requirements.txt common/__init__.py common/config.py results/.gitkeep reports/.gitkeep
git commit -m "feat: add project skeleton and config module"
```

---

### Task 2: Prompt 生成模块

**Files:**
- Create: `common/prompts.py`

- [ ] **Step 1: 创建 common/prompts.py**

```python
"""种子 prompt 定义 + 长度扩展 + tokenize 截断。

提供 generate_prompt() 和 generate_batch_prompts() 两个公共接口，
为不同测试场景生成指定 token 数的 prompt。

设计要点：
- 3 个种子 prompt 覆盖 QA/摘要/代码场景
- 长度扩展通过追加多轮历史对话实现（不是随机填充）
- 使用模型 tokenizer 精确 tokenize 后截断到目标 token 数
- 并发场景用不同 seed 避免完全 KV cache 命中
"""

# ============================================================
# 种子 prompt 定义
# ============================================================

SEEDS = {
    "seed_qa": {
        "system": "你是一个专业的技术顾问，请详细回答以下问题。",
        "question": "请解释 Transformer 模型中自注意力机制（Self-Attention）的工作原理，包括 Q、K、V 矩阵的作用、缩放点积注意力的计算过程，以及多头注意力如何增强模型的表达能力。",
    },
    "seed_summary": {
        "system": "你是一个专业的文档分析助手，请对以下内容生成摘要。",
        "question": "近年来，大语言模型（LLM）在自然语言处理领域取得了突破性进展。从 GPT 系列到 LLaMA、Mistral 等开源模型，参数规模从数十亿增长到数千亿，模型能力也随之大幅提升。然而，大模型的推理成本和部署难度也成为产业落地的核心挑战。推理优化技术如量化（Quantization）、蒸馏（Distillation）、剪枝（Pruning）和注意力优化（如 Flash Attention、Paged Attention）成为研究热点。vLLM 和 SGLang 等推理框架通过连续批处理（Continuous Batching）和前缀缓存（Prefix Caching）显著提升了推理吞吐。与此同时，模型压缩技术使得在消费级 GPU 上部署大模型成为可能，推动了 AI 应用的普及化。请对以上内容生成一段 200 字以内的中文摘要。",
    },
    "seed_code": {
        "system": "你是一个专业的编程助手，请根据需求生成代码。",
        "question": "请用 Python 实现一个线程安全的 LRU 缓存类，要求：1) 支持 get 和 put 操作；2) 容量满时淘汰最近最少使用的条目；3) 所有操作时间复杂度为 O(1)；4) 使用 threading.Lock 保证线程安全；5) 提供 __len__ 和 __contains__ 魔术方法。请附上简要的使用示例。",
    },
}

# 用于扩展 prompt 的历史轮次模板
HISTORY_TURNS = [
    {"role": "user", "content": "什么是梯度消失问题？它在深层网络中如何表现？"},
    {"role": "assistant", "content": "梯度消失是指在网络反向传播过程中，梯度值逐层指数级衰减，导致靠近输入层的参数几乎无法更新的现象。在深层网络中表现为：浅层参数更新极慢甚至停滞，模型难以学习低层特征，训练损失下降缓慢或过早收敛。常见于使用 Sigmoid/Tanh 激活函数的网络。"},
    {"role": "user", "content": "残差连接是如何缓解梯度消失的？"},
    {"role": "assistant", "content": "残差连接通过引入跳跃连接（skip connection），使梯度可以绕过中间层直接传播到浅层。前向传播时输出为 F(x)+x，反向传播时梯度至少有一条路径可以无损传递。这使网络即使加深也不会导致梯度消失，是 ResNet 的核心创新。"},
    {"role": "user", "content": "LayerNorm 和 BatchNorm 有什么区别？为什么 Transformer 用 LayerNorm？"},
    {"role": "assistant", "content": "BatchNorm 沿 batch 维度归一化，依赖 batch 统计量，在 batch 较小或序列长度不一时不稳定。LayerNorm 沿特征维度归一化，独立于 batch 大小，对每个样本单独计算，更适合变长序列和自回归生成。Transformer 选择 LayerNorm 正是因为其与序列模型的兼容性更好。"},
    {"role": "user", "content": "什么是位置编码？Transformer 为什么需要它？"},
    {"role": "assistant", "content": "位置编码为输入 token 注入位置信息，因为自注意力机制本身是排列不变的——打乱输入顺序不影响输出。Transformer 通过正弦/余弦位置编码或可学习位置嵌入，让模型区分不同位置的 token，从而理解序列的顺序关系。"},
    {"role": "user", "content": "KV Cache 的作用是什么？它如何加速推理？"},
    {"role": "assistant", "content": "KV Cache 在自回归生成中缓存已计算的 Key 和 Value 矩阵。每步生成新 token 时，只需计算当前 token 的 Q/K/V，将 K/V 追加到缓存中，避免对前面所有 token 重复计算注意力。这将每步计算量从 O(n²) 降为 O(n)，显著加速推理。"},
    {"role": "user", "content": "Flash Attention 的优化原理是什么？"},
    {"role": "assistant", "content": "Flash Attention 利用 GPU 内存层次结构（SRAM >> HBM），将注意力计算分块（tiling），每块在 SRAM 中完成 Q·K^T 和 softmax 运算后再写回 HBM。这大幅减少了 HBM 读写次数，同时通过在线 softmax 技巧保证数值等价性，实现 2-4 倍加速且不损失精度。"},
    {"role": "user", "content": "什么是混合专家模型（MoE）？它如何提升模型效率？"},
    {"role": "assistant", "content": "MoE 通过路由门控（Router）机制，每个 token 只激活部分专家子网络（如 8 个专家中激活 2 个），总参数量很大但每次推理只用一小部分。这样既保持了模型容量，又控制了计算成本，是 GPT-4 等大模型的关键架构选择。"},
]


def _build_messages(seed_key: str, num_history_turns: int = 0) -> list[dict]:
    """构建聊天消息列表，可选追加历史轮次。

    Args:
        seed_key: 种子名称，必须是 SEEDS 中的 key
        num_history_turns: 追加的历史轮次数（1 轮 = 1 user + 1 assistant）

    Returns:
        消息列表，格式为 [{"role": ..., "content": ...}, ...]
    """
    seed = SEEDS[seed_key]
    messages = [{"role": "system", "content": seed["system"]}]

    # 追加历史对话
    for i in range(min(num_history_turns, len(HISTORY_TURNS))):
        messages.append(HISTORY_TURNS[i])

    # 追加当前问题
    messages.append({"role": "user", "content": seed["question"]})
    return messages


def _tokenize_and_truncate(
    messages: list[dict],
    target_tokens: int,
    tokenizer,
) -> str:
    """用 tokenizer 将消息 tokenize 后截断到目标 token 数，返回解码后的字符串。

    Args:
        messages: 聊天消息列表
        target_tokens: 目标 token 数
        tokenizer: HuggingFace tokenizer 实例

    Returns:
        截断后的 prompt 字符串
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def generate_prompt(
    target_tokens: int,
    seed: str = "seed_qa",
    tokenizer=None,
) -> str:
    """生成指定 token 数的 prompt。

    短 prompt（~128 tokens）直接使用种子 prompt + 少量历史。
    中 prompt（~512 tokens）追加几轮历史对话。
    长 prompt（~1024 tokens）追加更多轮历史对话。

    Args:
        target_tokens: 目标 token 数
        seed: 种子名称，可选 seed_qa / seed_summary / seed_code
        tokenizer: HuggingFace tokenizer 实例（必需）

    Returns:
        截断到目标 token 数的 prompt 字符串

    Raises:
        ValueError: seed 不在 SEEDS 中
    """
    if seed not in SEEDS:
        raise ValueError(f"Unknown seed: {seed}. Must be one of {list(SEEDS.keys())}")
    if tokenizer is None:
        raise ValueError("tokenizer is required for prompt generation")

    # 逐步增加历史轮次，直到 token 数接近目标
    best_messages = _build_messages(seed, 0)
    best_len = len(tokenizer.encode(
        tokenizer.apply_chat_template(best_messages, tokenize=False, add_generation_prompt=True),
        add_special_tokens=False,
    ))

    for n_turns in range(1, len(HISTORY_TURNS) + 1):
        messages = _build_messages(seed, n_turns)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if n_tokens >= target_tokens:
            # 如果刚好超过或等于，用这个
            return _tokenize_and_truncate(messages, target_tokens, tokenizer)
        best_messages = messages
        best_len = n_tokens

    # 所有历史都加上了仍不够长，用最长的版本截断（实际上就是返回全部内容）
    return _tokenize_and_truncate(best_messages, target_tokens, tokenizer)


def generate_batch_prompts(
    batch_size: int,
    target_tokens: int,
    tokenizer=None,
) -> list[str]:
    """生成一批不同内容但相同目标长度的 prompt。

    通过轮换种子和历史偏移，确保并发请求的 prompt 不完全相同，
    避免不公平的 KV cache 命中。

    Args:
        batch_size: 需要的 prompt 数量
        target_tokens: 每个 prompt 的目标 token 数
        tokenizer: HuggingFace tokenizer 实例（必需）

    Returns:
        prompt 字符串列表
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required for prompt generation")

    seed_keys = list(SEEDS.keys())
    prompts = []
    for i in range(batch_size):
        # 轮换种子
        seed_key = seed_keys[i % len(seed_keys)]
        prompt = generate_prompt(target_tokens, seed=seed_key, tokenizer=tokenizer)
        prompts.append(prompt)
    return prompts
```

- [ ] **Step 2: 验证 prompt 生成（离线验证，无需 GPU）**

```bash
cd /Users/wt/share/python/infer-bench1
python -c "
from common.prompts import SEEDS, HISTORY_TURNS
print('Seeds:', list(SEEDS.keys()))
print('History turns:', len(HISTORY_TURNS))
print('seed_qa question preview:', SEEDS['seed_qa']['question'][:50])
"
```

Expected: 输出种子名称列表、历史轮次数、种子问题预览

- [ ] **Step 3: 提交**

```bash
git add common/prompts.py
git commit -m "feat: add prompt generation module with seed-based expansion"
```

---

### Task 3: GPU 显存监控模块

**Files:**
- Create: `common/gpu.py`

- [ ] **Step 1: 创建 common/gpu.py**

```python
"""GPU 显存后台监控（pynvml）。

使用 pynvml 每 N 毫秒采样一次 GPU 已用显存，
测试前记录基线，测试结束后报告峰值增量。
"""

import logging
import threading
import time

import pynvml

logger = logging.getLogger(__name__)


class GPUMonitor:
    """后台线程采样 GPU 显存占用。

    用法:
        monitor = GPUMonitor(device_index=0, interval_ms=100)
        monitor.start()          # 开始后台采样
        # ... 运行测试 ...
        monitor.stop()           # 停止采样
        peak_vram_mb = monitor.peak_vram_mb  # 获取峰值显存增量 (MB)
    """

    def __init__(self, device_index: int = 0, interval_ms: int = 100):
        self._device_index = device_index
        self._interval_s = interval_ms / 1000.0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._baseline_mb: float = 0.0
        self._samples: list[float] = []
        self._running = False

        # 初始化 pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        logger.info("GPU monitor initialized: device %d", device_index)

    def _read_vram_mb(self) -> float:
        """读取当前 GPU 已用显存 (MB)。"""
        info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        return info.used / (1024 * 1024)

    def _sampling_loop(self):
        """后台采样循环。"""
        while not self._stop_event.is_set():
            try:
                vram = self._read_vram_mb()
                self._samples.append(vram)
            except Exception as e:
                logger.warning("GPU sample failed: %s", e)
            self._stop_event.wait(self._interval_s)

    def start(self):
        """启动后台显存采样。记录基线值并开始采集。"""
        self._samples = []
        self._baseline_mb = self._read_vram_mb()
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._thread.start()
        logger.info("GPU monitoring started (baseline: %.0f MB)", self._baseline_mb)

    def stop(self):
        """停止后台显存采样。"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._running = False
        logger.info("GPU monitoring stopped (%d samples)", len(self._samples))

    @property
    def peak_vram_mb(self) -> float:
        """返回峰值显存增量 (MB) = 采样峰值 − 基线。"""
        if not self._samples:
            return 0.0
        return max(self._samples) - self._baseline_mb

    @property
    def baseline_mb(self) -> float:
        """返回基线显存 (MB)。"""
        return self._baseline_mb

    @property
    def is_running(self) -> bool:
        return self._running

    def __del__(self):
        if self._running:
            self.stop()
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
```

- [ ] **Step 2: 提交**

```bash
git add common/gpu.py
git commit -m "feat: add GPU VRAM monitoring module (pynvml background thread)"
```

---

### Task 4: 指标采集模块

**Files:**
- Create: `common/metrics.py`

- [ ] **Step 1: 创建 common/metrics.py**

```python
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
    """计算首 Token 延迟。

    Args:
        start_time: 请求发出时间（time.monotonic()）
        first_token_time: 收到第一个 token 的时间（time.monotonic()）

    Returns:
        TTFT (毫秒)
    """
    return (first_token_time - start_time) * 1000.0


def compute_tps(total_tokens: int, total_time_s: float) -> float:
    """计算吞吐量。

    Args:
        total_tokens: 总输出 token 数
        total_time_s: 总耗时（秒）

    Returns:
        吞吐量 (tokens/s)
    """
    if total_time_s <= 0:
        return 0.0
    return total_tokens / total_time_s


def compute_concurrent_tps(
    total_tokens: int,
    earliest_start: float,
    latest_end: float,
) -> float:
    """计算并发吞吐量。

    Args:
        total_tokens: 所有请求的总输出 token 数
        earliest_start: 最早请求发出时间（time.monotonic()）
        latest_end: 最晚请求完成时间（time.monotonic()）

    Returns:
        并发吞吐量 (tokens/s)
    """
    elapsed = latest_end - earliest_start
    if elapsed <= 0:
        return 0.0
    return total_tokens / elapsed


def results_to_csv(
    results: list[BenchmarkResult],
    filepath: str | Path,
) -> str:
    """将结果列表写入 CSV 文件。

    Args:
        results: BenchmarkResult 列表
        filepath: 输出文件路径

    Returns:
        写入的文件路径（字符串）
    """
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
```

- [ ] **Step 2: 提交**

```bash
git add common/metrics.py
git commit -m "feat: add metrics module (TTFT/TPS computation and CSV output)"
```

---

### Task 5: OpenAI-compatible API 客户端

**Files:**
- Create: `common/client.py`

- [ ] **Step 1: 创建 common/client.py**

```python
"""OpenAI-compatible API 客户端（vLLM/SGLang 共用）。

使用 aiohttp 发送流式请求，采集 TTFT 和 TPS。
"""

import asyncio
import logging
import time

import aiohttp

logger = logging.getLogger(__name__)

# 健康检查最大等待时间（秒）
HEALTH_CHECK_TIMEOUT = 300
HEALTH_CHECK_INTERVAL = 5


async def wait_for_server(base_url: str, timeout: int = HEALTH_CHECK_TIMEOUT) -> bool:
    """轮询服务健康检查，等待引擎就绪。

    Args:
        base_url: 服务基础 URL，如 http://localhost:8000
        timeout: 最大等待秒数

    Returns:
        True 如果服务就绪，False 如果超时
    """
    health_url = f"{base_url}/health"
    start = time.monotonic()
    async with aiohttp.ClientSession() as session:
        while time.monotonic() - start < timeout:
            try:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status == 200:
                        logger.info("Server ready: %s", base_url)
                        return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
    logger.error("Server not ready after %ds: %s", timeout, base_url)
    return False


async def stream_request(
    base_url: str,
    prompt: str,
    model: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> dict:
    """发送单个流式请求，采集 TTFT 和 TPS。

    Args:
        base_url: 服务基础 URL
        prompt: 输入 prompt 字符串
        model: 模型名称
        max_tokens: 最大输出 token 数
        temperature: 采样温度

    Returns:
        dict 包含:
            - ttft_ms: 首 Token 延迟 (ms)
            - total_tokens: 输出 token 数
            - total_time_s: 总耗时 (s)
            - tps: 吞吐 (tokens/s)
            - text: 完整输出文本
    """
    url = f"{base_url}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    ttft_ms = None
    first_token_time = None
    start_time = time.monotonic()
    output_text = ""
    total_tokens = 0

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]  # 去掉 "data: "
                if data_str == "[DONE]":
                    break

                import json
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # 采集首 token 时间
                choices = data.get("choices", [])
                if choices and first_token_time is None:
                    content = choices[0].get("text", "")
                    if content:
                        first_token_time = time.monotonic()
                        ttft_ms = (first_token_time - start_time) * 1000.0

                # 累积输出
                if choices:
                    output_text += choices[0].get("text", "")

                # 采集 usage
                usage = data.get("usage")
                if usage:
                    total_tokens = usage.get("completion_tokens", total_tokens)

    end_time = time.monotonic()
    total_time_s = end_time - start_time

    # 如果流中没有 usage 信息，用简单估算
    if total_tokens == 0 and output_text:
        # 粗略估算：非空输出至少 1 token
        total_tokens = max(1, len(output_text.split()))

    tps = total_tokens / total_time_s if total_time_s > 0 else 0.0

    return {
        "ttft_ms": ttft_ms if ttft_ms is not None else 0.0,
        "total_tokens": total_tokens,
        "total_time_s": total_time_s,
        "tps": tps,
        "text": output_text,
    }


async def concurrent_stream_requests(
    base_url: str,
    prompts: list[str],
    model: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> dict:
    """并发发送多个流式请求，采集并发 TTFT 和 TPS。

    Args:
        base_url: 服务基础 URL
        prompts: prompt 列表
        model: 模型名称
        max_tokens: 最大输出 token 数
        temperature: 采样温度

    Returns:
        dict 包含:
            - mean_ttft_ms: 平均首 Token 延迟 (ms)
            - total_tokens: 所有请求的输出 token 总数
            - concurrent_tps: 并发吞吐 (tokens/s)
            - total_time_s: 从首个请求发出到最后一个完成的总时间 (s)
            - results: 每个请求的单独结果列表
    """
    start_time = time.monotonic()

    tasks = [
        stream_request(base_url, prompt, model, max_tokens, temperature)
        for prompt in prompts
    ]
    results = await asyncio.gather(*tasks)

    end_time = time.monotonic()
    total_time_s = end_time - start_time

    ttfts = [r["ttft_ms"] for r in results]
    total_tokens = sum(r["total_tokens"] for r in results)
    mean_ttft_ms = sum(ttfts) / len(ttfts) if ttfts else 0.0
    concurrent_tps = total_tokens / total_time_s if total_time_s > 0 else 0.0

    return {
        "mean_ttft_ms": mean_ttft_ms,
        "total_tokens": total_tokens,
        "concurrent_tps": concurrent_tps,
        "total_time_s": total_time_s,
        "results": results,
    }
```

- [ ] **Step 2: 提交**

```bash
git add common/client.py
git commit -m "feat: add OpenAI-compatible streaming API client (aiohttp)"
```

---

### Task 6: vLLM 基准测试脚本

**Files:**
- Create: `run_vllm.py`

- [ ] **Step 1: 创建 run_vllm.py**

```python
"""vLLM 基准测试脚本。

自动启动 vLLM 服务，运行单请求延迟和并发吞吐测试，
输出 CSV 到 results/ 目录。

用法:
    python run_vllm.py [--config config.yaml]
"""

import argparse
import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

from transformers import AutoTokenizer

from common.client import concurrent_stream_requests, stream_request, wait_for_server
from common.config import load_config
from common.gpu import GPUMonitor
from common.metrics import BenchmarkResult, compute_concurrent_tps, make_run_id, results_to_csv
from common.prompts import generate_batch_prompts, generate_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_vllm")


def start_vllm_server(cfg) -> subprocess.Popen:
    """启动 vLLM 服务进程。"""
    model_path = cfg.model.path or cfg.model.name
    port = cfg.engines.vllm.port
    extra_args = cfg.engines.vllm.extra_args

    cmd = (
        f"python -m vllm.entrypoints.openai.api_server "
        f"--model {model_path} "
        f"--port {port} "
        f"--dtype auto "
        f"{extra_args}"
    )
    logger.info("Starting vLLM server: %s", cmd)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cfg.gpu.device

    proc = subprocess.Popen(
        cmd,
        shell=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def stop_vllm_server(proc: subprocess.Popen):
    """停止 vLLM 服务进程。"""
    logger.info("Stopping vLLM server (PID: %d)", proc.pid)
    try:
        proc.terminate()
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    logger.info("vLLM server stopped")


def count_output_tokens(text: str, tokenizer) -> int:
    """用 tokenizer 计算输出文本的 token 数。"""
    return len(tokenizer.encode(text, add_special_tokens=False))


async def run_single_request_tests(
    base_url: str,
    model: str,
    cfg,
    tokenizer,
    run_id: str,
    gpu_monitor: GPUMonitor,
) -> list[BenchmarkResult]:
    """运行单请求延迟测试。"""
    results = []
    sr_cfg = cfg.test.single_request

    for prompt_len in sr_cfg.prompt_lengths:
        prompt = generate_prompt(prompt_len, seed="seed_qa", tokenizer=tokenizer)
        actual_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        logger.info("Single request: prompt_len=%d (actual=%d)", prompt_len, actual_tokens)

        # 预热
        for _ in range(sr_cfg.num_warmup):
            await stream_request(base_url, prompt, model, max_tokens=sr_cfg.max_new_tokens)
            await asyncio.sleep(0.5)

        # 正式测试
        ttfts = []
        tpss = []
        for i in range(sr_cfg.num_runs):
            gpu_monitor.start()
            resp = await stream_request(base_url, prompt, model, max_tokens=sr_cfg.max_new_tokens)
            gpu_monitor.stop()

            ttfts.append(resp["ttft_ms"])
            tpss.append(resp["tps"])
            logger.info(
                "  Run %d/%d: TTFT=%.1fms, TPS=%.1f",
                i + 1, sr_cfg.num_runs, resp["ttft_ms"], resp["tps"],
            )
            await asyncio.sleep(1)

        avg_ttft = sum(ttfts) / len(ttfts)
        avg_tps = sum(tpss) / len(tpss)
        peak_vram = gpu_monitor.peak_vram_mb

        results.append(BenchmarkResult(
            engine="vllm",
            test_type="single",
            batch_size=1,
            prompt_tokens=actual_tokens,
            max_new_tokens=sr_cfg.max_new_tokens,
            ttft_ms=round(avg_ttft, 2),
            mean_tps=round(avg_tps, 2),
            peak_vram_mb=round(peak_vram, 1),
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
        ))

    return results


async def run_concurrent_tests(
    base_url: str,
    model: str,
    cfg,
    tokenizer,
    run_id: str,
    gpu_monitor: GPUMonitor,
) -> list[BenchmarkResult]:
    """运行并发吞吐测试。"""
    results = []
    cc_cfg = cfg.test.concurrent

    for batch_size in cc_cfg.batch_sizes:
        prompts = generate_batch_prompts(
            batch_size,
            cc_cfg.prompt_length,
            tokenizer=tokenizer,
        )
        logger.info("Concurrent: batch_size=%d, prompt_len=%d", batch_size, cc_cfg.prompt_length)

        # 预热
        for _ in range(cc_cfg.num_warmup):
            warmup_prompts = generate_batch_prompts(batch_size, cc_cfg.prompt_length, tokenizer=tokenizer)
            await concurrent_stream_requests(
                base_url, warmup_prompts, model, max_tokens=cc_cfg.max_new_tokens,
            )
            await asyncio.sleep(1)

        # 正式测试
        ttfts = []
        tpss = []
        for i in range(cc_cfg.num_runs):
            gpu_monitor.start()
            resp = await concurrent_stream_requests(
                base_url, prompts, model, max_tokens=cc_cfg.max_new_tokens,
            )
            gpu_monitor.stop()

            ttfts.append(resp["mean_ttft_ms"])
            tpss.append(resp["concurrent_tps"])
            logger.info(
                "  Run %d/%d: mean_TTFT=%.1fms, TPS=%.1f",
                i + 1, cc_cfg.num_runs, resp["mean_ttft_ms"], resp["concurrent_tps"],
            )
            await asyncio.sleep(2)

        avg_ttft = sum(ttfts) / len(ttfts)
        avg_tps = sum(tpss) / len(tpss)
        peak_vram = gpu_monitor.peak_vram_mb

        results.append(BenchmarkResult(
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
        ))

    return results


async def main():
    parser = argparse.ArgumentParser(description="vLLM benchmark")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_id = make_run_id(cfg.output.timestamp_format)
    model = cfg.model.path or cfg.model.name
    port = cfg.engines.vllm.port
    base_url = f"http://localhost:{port}"

    # 启动 vLLM 服务
    proc = start_vllm_server(cfg)

    try:
        # 等待服务就绪
        ready = await wait_for_server(base_url, timeout=300)
        if not ready:
            logger.error("vLLM server failed to start")
            sys.exit(1)

        # 加载 tokenizer
        logger.info("Loading tokenizer: %s", model)
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        # 初始化 GPU 监控
        gpu_monitor = GPUMonitor(
            device_index=int(cfg.gpu.device),
            interval_ms=cfg.gpu.monitor_interval_ms,
        )

        # 等一会让服务稳定
        await asyncio.sleep(10)

        # 运行测试
        all_results = []

        logger.info("=" * 60)
        logger.info("Running single request tests...")
        logger.info("=" * 60)
        single_results = await run_single_request_tests(
            base_url, model, cfg, tokenizer, run_id, gpu_monitor,
        )
        all_results.extend(single_results)

        logger.info("=" * 60)
        logger.info("Running concurrent tests...")
        logger.info("=" * 60)
        concurrent_results = await run_concurrent_tests(
            base_url, model, cfg, tokenizer, run_id, gpu_monitor,
        )
        all_results.extend(concurrent_results)

        # 输出 CSV
        output_path = results_to_csv(
            all_results,
            f"{cfg.output.results_dir}/vllm_{run_id}.csv",
        )
        logger.info("Results saved to %s", output_path)

    finally:
        stop_vllm_server(proc)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: 提交**

```bash
git add run_vllm.py
git commit -m "feat: add vLLM benchmark script (auto start/stop server)"
```

---

### Task 7: SGLang 基准测试脚本

**Files:**
- Create: `run_sglang.py`

- [ ] **Step 1: 创建 run_sglang.py**

```python
"""SGLang 基准测试脚本。

自动启动 SGLang 服务，运行单请求延迟和并发吞吐测试，
输出 CSV 到 results/ 目录。

用法:
    python run_sglang.py [--config config.yaml]
"""

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import time
from datetime import datetime

from transformers import AutoTokenizer

from common.client import concurrent_stream_requests, stream_request, wait_for_server
from common.config import load_config
from common.gpu import GPUMonitor
from common.metrics import BenchmarkResult, make_run_id, results_to_csv
from common.prompts import generate_batch_prompts, generate_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_sglang")


def start_sglang_server(cfg) -> subprocess.Popen:
    """启动 SGLang 服务进程。"""
    model_path = cfg.model.path or cfg.model.name
    port = cfg.engines.sglang.port
    extra_args = cfg.engines.sglang.extra_args

    cmd = (
        f"python -m sglang.launch_server "
        f"--model-path {model_path} "
        f"--port {port} "
        f"{extra_args}"
    )
    logger.info("Starting SGLang server: %s", cmd)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cfg.gpu.device

    proc = subprocess.Popen(
        cmd,
        shell=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def stop_sglang_server(proc: subprocess.Popen):
    """停止 SGLang 服务进程。"""
    logger.info("Stopping SGLang server (PID: %d)", proc.pid)
    try:
        proc.terminate()
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    logger.info("SGLang server stopped")


async def run_single_request_tests(
    base_url: str,
    model: str,
    cfg,
    tokenizer,
    run_id: str,
    gpu_monitor: GPUMonitor,
) -> list[BenchmarkResult]:
    """运行单请求延迟测试。"""
    results = []
    sr_cfg = cfg.test.single_request

    for prompt_len in sr_cfg.prompt_lengths:
        prompt = generate_prompt(prompt_len, seed="seed_qa", tokenizer=tokenizer)
        actual_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        logger.info("Single request: prompt_len=%d (actual=%d)", prompt_len, actual_tokens)

        # 预热
        for _ in range(sr_cfg.num_warmup):
            await stream_request(base_url, prompt, model, max_tokens=sr_cfg.max_new_tokens)
            await asyncio.sleep(0.5)

        # 正式测试
        ttfts = []
        tpss = []
        for i in range(sr_cfg.num_runs):
            gpu_monitor.start()
            resp = await stream_request(base_url, prompt, model, max_tokens=sr_cfg.max_new_tokens)
            gpu_monitor.stop()

            ttfts.append(resp["ttft_ms"])
            tpss.append(resp["tps"])
            logger.info(
                "  Run %d/%d: TTFT=%.1fms, TPS=%.1f",
                i + 1, sr_cfg.num_runs, resp["ttft_ms"], resp["tps"],
            )
            await asyncio.sleep(1)

        avg_ttft = sum(ttfts) / len(ttfts)
        avg_tps = sum(tpss) / len(tpss)
        peak_vram = gpu_monitor.peak_vram_mb

        results.append(BenchmarkResult(
            engine="sglang",
            test_type="single",
            batch_size=1,
            prompt_tokens=actual_tokens,
            max_new_tokens=sr_cfg.max_new_tokens,
            ttft_ms=round(avg_ttft, 2),
            mean_tps=round(avg_tps, 2),
            peak_vram_mb=round(peak_vram, 1),
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
        ))

    return results


async def run_concurrent_tests(
    base_url: str,
    model: str,
    cfg,
    tokenizer,
    run_id: str,
    gpu_monitor: GPUMonitor,
) -> list[BenchmarkResult]:
    """运行并发吞吐测试。"""
    results = []
    cc_cfg = cfg.test.concurrent

    for batch_size in cc_cfg.batch_sizes:
        prompts = generate_batch_prompts(
            batch_size,
            cc_cfg.prompt_length,
            tokenizer=tokenizer,
        )
        logger.info("Concurrent: batch_size=%d, prompt_len=%d", batch_size, cc_cfg.prompt_length)

        # 预热
        for _ in range(cc_cfg.num_warmup):
            warmup_prompts = generate_batch_prompts(batch_size, cc_cfg.prompt_length, tokenizer=tokenizer)
            await concurrent_stream_requests(
                base_url, warmup_prompts, model, max_tokens=cc_cfg.max_new_tokens,
            )
            await asyncio.sleep(1)

        # 正式测试
        ttfts = []
        tpss = []
        for i in range(cc_cfg.num_runs):
            gpu_monitor.start()
            resp = await concurrent_stream_requests(
                base_url, prompts, model, max_tokens=cc_cfg.max_new_tokens,
            )
            gpu_monitor.stop()

            ttfts.append(resp["mean_ttft_ms"])
            tpss.append(resp["concurrent_tps"])
            logger.info(
                "  Run %d/%d: mean_TTFT=%.1fms, TPS=%.1f",
                i + 1, cc_cfg.num_runs, resp["mean_ttft_ms"], resp["concurrent_tps"],
            )
            await asyncio.sleep(2)

        avg_ttft = sum(ttfts) / len(ttfts)
        avg_tps = sum(tpss) / len(tpss)
        peak_vram = gpu_monitor.peak_vram_mb

        results.append(BenchmarkResult(
            engine="sglang",
            test_type="concurrent",
            batch_size=batch_size,
            prompt_tokens=cc_cfg.prompt_length,
            max_new_tokens=cc_cfg.max_new_tokens,
            ttft_ms=round(avg_ttft, 2),
            mean_tps=round(avg_tps, 2),
            peak_vram_mb=round(peak_vram, 1),
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
        ))

    return results


async def main():
    parser = argparse.ArgumentParser(description="SGLang benchmark")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_id = make_run_id(cfg.output.timestamp_format)
    model = cfg.model.path or cfg.model.name
    port = cfg.engines.sglang.port
    base_url = f"http://localhost:{port}"

    # 启动 SGLang 服务
    proc = start_sglang_server(cfg)

    try:
        # 等待服务就绪
        ready = await wait_for_server(base_url, timeout=300)
        if not ready:
            logger.error("SGLang server failed to start")
            sys.exit(1)

        # 加载 tokenizer
        logger.info("Loading tokenizer: %s", model)
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        # 初始化 GPU 监控
        gpu_monitor = GPUMonitor(
            device_index=int(cfg.gpu.device),
            interval_ms=cfg.gpu.monitor_interval_ms,
        )

        # 等一会让服务稳定
        await asyncio.sleep(10)

        # 运行测试
        all_results = []

        logger.info("=" * 60)
        logger.info("Running single request tests...")
        logger.info("=" * 60)
        single_results = await run_single_request_tests(
            base_url, model, cfg, tokenizer, run_id, gpu_monitor,
        )
        all_results.extend(single_results)

        logger.info("=" * 60)
        logger.info("Running concurrent tests...")
        logger.info("=" * 60)
        concurrent_results = await run_concurrent_tests(
            base_url, model, cfg, tokenizer, run_id, gpu_monitor,
        )
        all_results.extend(concurrent_results)

        # 输出 CSV
        output_path = results_to_csv(
            all_results,
            f"{cfg.output.results_dir}/sglang_{run_id}.csv",
        )
        logger.info("Results saved to %s", output_path)

    finally:
        stop_sglang_server(proc)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: 提交**

```bash
git add run_sglang.py
git commit -m "feat: add SGLang benchmark script (auto start/stop server)"
```

---

### Task 8: Transformers 基准测试脚本

**Files:**
- Create: `run_transformers.py`

- [ ] **Step 1: 创建 run_transformers.py**

```python
"""原生 Transformers 基准测试脚本。

进程内加载模型，使用 TextIteratorStreamer 流式输出，
运行单请求延迟和 batch generate 并发测试，
输出 CSV 到 results/ 目录。

注意：Transformers 的并发测试使用 batch generate（非 HTTP 级并发），
与 vLLM/SGLang 的 continuous batching 语义不同，报告中需标注。

用法:
    python run_transformers.py [--config config.yaml]
"""

import argparse
import logging
import threading
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from common.config import load_config
from common.gpu import GPUMonitor
from common.metrics import BenchmarkResult, make_run_id, results_to_csv
from common.prompts import generate_batch_prompts, generate_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_transformers")


def load_model_and_tokenizer(cfg):
    """加载模型和 tokenizer。"""
    model_name = cfg.model.path or cfg.model.name
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map.get(cfg.engines.transformers.dtype, torch.float16)

    logger.info("Loading model: %s (dtype=%s, device_map=%s)", model_name, dtype, cfg.engines.transformers.device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=cfg.engines.transformers.device_map,
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded successfully")
    return model, tokenizer


def single_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.0,
) -> dict:
    """单请求流式生成，采集 TTFT 和 TPS。

    Returns:
        dict: ttft_ms, total_tokens, total_time_s, tps, text
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature if temperature > 0 else None,
        "do_sample": temperature > 0,
        "streamer": streamer,
    }

    # 在后台线程运行 generate
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    # 采集 TTFT
    first_token_time = None
    start_time = time.monotonic()
    output_text = ""

    for token_text in streamer:
        if first_token_time is None and token_text:
            first_token_time = time.monotonic()
        output_text += token_text

    thread.join()
    end_time = time.monotonic()

    ttft_ms = (first_token_time - start_time) * 1000.0 if first_token_time else 0.0
    total_time_s = end_time - start_time

    # 计算输出 token 数
    output_ids = tokenizer.encode(output_text, add_special_tokens=False)
    total_tokens = len(output_ids)

    tps = total_tokens / total_time_s if total_time_s > 0 else 0.0

    return {
        "ttft_ms": ttft_ms,
        "total_tokens": total_tokens,
        "total_time_s": total_time_s,
        "tps": tps,
        "text": output_text,
    }


def batch_generate(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float = 0.0,
) -> dict:
    """Batch generate 模拟并发。

    注意：这是 batch 推理而非 HTTP 级并发，与 vLLM/SGLang 的
    continuous batching 语义不同。

    Returns:
        dict: mean_ttft_ms, total_tokens, concurrent_tps, total_time_s
    """
    # 对 batch 做 padding
    all_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    start_time = time.monotonic()

    with torch.no_grad():
        outputs = model.generate(
            **all_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
        )

    end_time = time.monotonic()
    total_time_s = end_time - start_time

    # 计算输出 token 数（排除 input）
    input_lengths = all_inputs["attention_mask"].sum(dim=1).tolist()
    total_tokens = 0
    ttfts = []

    for i, output in enumerate(outputs):
        new_tokens = output[input_lengths[i]:]
        total_tokens += len(new_tokens)
        # Batch generate 无法精确测量每个请求的 TTFT，用总时间 / batch_size 估算
        ttfts.append(total_time_s * 1000.0 / len(prompts))

    mean_ttft_ms = sum(ttfts) / len(ttfts) if ttfts else 0.0
    concurrent_tps = total_tokens / total_time_s if total_time_s > 0 else 0.0

    return {
        "mean_ttft_ms": mean_ttft_ms,
        "total_tokens": total_tokens,
        "concurrent_tps": concurrent_tps,
        "total_time_s": total_time_s,
    }


def run_single_request_tests(
    model,
    tokenizer,
    cfg,
    run_id: str,
    gpu_monitor: GPUMonitor,
) -> list[BenchmarkResult]:
    """运行单请求延迟测试。"""
    results = []
    sr_cfg = cfg.test.single_request

    for prompt_len in sr_cfg.prompt_lengths:
        prompt = generate_prompt(prompt_len, seed="seed_qa", tokenizer=tokenizer)
        actual_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        logger.info("Single request: prompt_len=%d (actual=%d)", prompt_len, actual_tokens)

        # 预热
        for _ in range(sr_cfg.num_warmup):
            single_generate(model, tokenizer, prompt, max_new_tokens=sr_cfg.max_new_tokens)
            time.sleep(0.5)

        # 正式测试
        ttfts = []
        tpss = []
        for i in range(sr_cfg.num_runs):
            gpu_monitor.start()
            resp = single_generate(model, tokenizer, prompt, max_new_tokens=sr_cfg.max_new_tokens)
            gpu_monitor.stop()

            ttfts.append(resp["ttft_ms"])
            tpss.append(resp["tps"])
            logger.info(
                "  Run %d/%d: TTFT=%.1fms, TPS=%.1f",
                i + 1, sr_cfg.num_runs, resp["ttft_ms"], resp["tps"],
            )
            time.sleep(1)

        avg_ttft = sum(ttfts) / len(ttfts)
        avg_tps = sum(tpss) / len(tpss)
        peak_vram = gpu_monitor.peak_vram_mb

        results.append(BenchmarkResult(
            engine="transformers",
            test_type="single",
            batch_size=1,
            prompt_tokens=actual_tokens,
            max_new_tokens=sr_cfg.max_new_tokens,
            ttft_ms=round(avg_ttft, 2),
            mean_tps=round(avg_tps, 2),
            peak_vram_mb=round(peak_vram, 1),
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
        ))

    return results


def run_concurrent_tests(
    model,
    tokenizer,
    cfg,
    run_id: str,
    gpu_monitor: GPUMonitor,
) -> list[BenchmarkResult]:
    """运行 batch generate 并发测试。

    注意：使用 batch generate 而非 HTTP 并发，语义不同。
    """
    results = []
    cc_cfg = cfg.test.concurrent

    for batch_size in cc_cfg.batch_sizes:
        prompts = generate_batch_prompts(
            batch_size,
            cc_cfg.prompt_length,
            tokenizer=tokenizer,
        )
        logger.info(
            "Concurrent (batch): batch_size=%d, prompt_len=%d",
            batch_size, cc_cfg.prompt_length,
        )

        # 预热
        for _ in range(cc_cfg.num_warmup):
            warmup_prompts = generate_batch_prompts(batch_size, cc_cfg.prompt_length, tokenizer=tokenizer)
            batch_generate(model, tokenizer, warmup_prompts, max_new_tokens=cc_cfg.max_new_tokens)
            time.sleep(1)

        # 正式测试
        ttfts = []
        tpss = []
        for i in range(cc_cfg.num_runs):
            gpu_monitor.start()
            resp = batch_generate(model, tokenizer, prompts, max_new_tokens=cc_cfg.max_new_tokens)
            gpu_monitor.stop()

            ttfts.append(resp["mean_ttft_ms"])
            tpss.append(resp["concurrent_tps"])
            logger.info(
                "  Run %d/%d: mean_TTFT=%.1fms, TPS=%.1f",
                i + 1, cc_cfg.num_runs, resp["mean_ttft_ms"], resp["concurrent_tps"],
            )
            time.sleep(2)

        avg_ttft = sum(ttfts) / len(ttfts)
        avg_tps = sum(tpss) / len(tpss)
        peak_vram = gpu_monitor.peak_vram_mb

        results.append(BenchmarkResult(
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
        ))

    return results


def main():
    parser = argparse.ArgumentParser(description="Transformers benchmark")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_id = make_run_id(cfg.output.timestamp_format)

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(cfg)

    # 初始化 GPU 监控
    gpu_monitor = GPUMonitor(
        device_index=int(cfg.gpu.device),
        interval_ms=cfg.gpu.monitor_interval_ms,
    )

    all_results = []

    # 单请求测试
    logger.info("=" * 60)
    logger.info("Running single request tests...")
    logger.info("=" * 60)
    single_results = run_single_request_tests(model, tokenizer, cfg, run_id, gpu_monitor)
    all_results.extend(single_results)

    # 并发测试
    logger.info("=" * 60)
    logger.info("Running concurrent (batch generate) tests...")
    logger.info("=" * 60)
    concurrent_results = run_concurrent_tests(model, tokenizer, cfg, run_id, gpu_monitor)
    all_results.extend(concurrent_results)

    # 输出 CSV
    output_path = results_to_csv(
        all_results,
        f"{cfg.output.results_dir}/transformers_{run_id}.csv",
    )
    logger.info("Results saved to %s", output_path)

    # 释放显存
    del model
    torch.cuda.empty_cache()
    logger.info("GPU memory released")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 提交**

```bash
git add run_transformers.py
git commit -m "feat: add Transformers benchmark script (in-process generate)"
```

---

### Task 9: 渐进并发扫描脚本

**Files:**
- Create: `run_sweep.py`

- [ ] **Step 1: 创建 run_sweep.py**

```python
"""渐进并发扫描脚本。

对三个引擎循环测试，并发数按倍数增长（1, 2, 4, 8, 16, 32），
输出 TPS + TTFT 随并发数的变化趋势数据。

用法:
    python run_sweep.py [--config config.yaml] [--engine vllm|sglang|transformers]
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
from common.gpu import GPUMonitor
from common.metrics import BenchmarkResult, make_run_id, results_to_csv
from common.prompts import generate_batch_prompts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_sweep")


def generate_sweep_concurrency(start: int, stop: int, multiplier: int) -> list[int]:
    """生成并发级别列表：start, start*multiplier, start*multiplier^2, ..., <=stop。"""
    levels = []
    c = start
    while c <= stop:
        levels.append(c)
        c *= multiplier
    return levels


# ============================================================
# vLLM / SGLang sweep（HTTP API）
# ============================================================

def start_server(engine: str, cfg) -> subprocess.Popen:
    """启动 vLLM 或 SGLang 服务。"""
    model_path = cfg.model.path or cfg.model.name
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cfg.gpu.device

    if engine == "vllm":
        port = cfg.engines.vllm.port
        extra = cfg.engines.vllm.extra_args
        cmd = (
            f"python -m vllm.entrypoints.openai.api_server "
            f"--model {model_path} --port {port} --dtype auto {extra}"
        )
    elif engine == "sglang":
        port = cfg.engines.sglang.port
        extra = cfg.engines.sglang.extra_args
        cmd = (
            f"python -m sglang.launch_server "
            f"--model-path {model_path} --port {port} {extra}"
        )
    else:
        raise ValueError(f"Unsupported engine for server mode: {engine}")

    logger.info("Starting %s server: %s", engine, cmd)
    return subprocess.Popen(cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def stop_server(proc: subprocess.Popen):
    """停止服务进程。"""
    logger.info("Stopping server (PID: %d)", proc.pid)
    try:
        proc.terminate()
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


async def sweep_http_engine(
    engine: str,
    cfg,
    run_id: str,
) -> list[BenchmarkResult]:
    """对 vLLM 或 SGLang 执行渐进并发扫描。"""
    results = []
    sw_cfg = cfg.test.sweep

    if engine == "vllm":
        port = cfg.engines.vllm.port
    else:
        port = cfg.engines.sglang.port
    base_url = f"http://localhost:{port}"
    model = cfg.model.path or cfg.model.name

    # 启动服务
    proc = start_server(engine, cfg)

    try:
        ready = await wait_for_server(base_url, timeout=300)
        if not ready:
            logger.error("%s server failed to start", engine)
            return results

        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        gpu_monitor = GPUMonitor(
            device_index=int(cfg.gpu.device),
            interval_ms=cfg.gpu.monitor_interval_ms,
        )

        await asyncio.sleep(10)

        concurrency_levels = generate_sweep_concurrency(
            sw_cfg.start, sw_cfg.stop, sw_cfg.multiplier,
        )
        logger.info("Sweep concurrency levels: %s", concurrency_levels)

        for concurrency in concurrency_levels:
            logger.info("--- Sweep: %s, concurrency=%d ---", engine, concurrency)

            try:
                prompts = generate_batch_prompts(
                    concurrency, sw_cfg.prompt_length, tokenizer=tokenizer,
                )

                gpu_monitor.start()
                resp = await concurrent_stream_requests(
                    base_url, prompts, model, max_tokens=sw_cfg.max_new_tokens,
                )
                gpu_monitor.stop()

                peak_vram = gpu_monitor.peak_vram_mb

                results.append(BenchmarkResult(
                    engine=engine,
                    test_type="sweep",
                    batch_size=concurrency,
                    prompt_tokens=sw_cfg.prompt_length,
                    max_new_tokens=sw_cfg.max_new_tokens,
                    ttft_ms=round(resp["mean_ttft_ms"], 2),
                    mean_tps=round(resp["concurrent_tps"], 2),
                    peak_vram_mb=round(peak_vram, 1),
                    run_id=run_id,
                    timestamp=datetime.now().isoformat(),
                ))

                logger.info(
                    "  concurrency=%d: TTFT=%.1fms, TPS=%.1f, VRAM=%.0fMB",
                    concurrency, resp["mean_ttft_ms"], resp["concurrent_tps"], peak_vram,
                )

            except Exception as e:
                logger.error("  Sweep %s concurrency=%d failed: %s", engine, concurrency, e)
                results.append(BenchmarkResult(
                    engine=engine,
                    test_type="sweep",
                    batch_size=concurrency,
                    prompt_tokens=sw_cfg.prompt_length,
                    max_new_tokens=sw_cfg.max_new_tokens,
                    ttft_ms=-1,  # 标记失败
                    mean_tps=-1,
                    peak_vram_mb=-1,
                    run_id=run_id,
                    timestamp=datetime.now().isoformat(),
                ))

            await asyncio.sleep(3)

    finally:
        stop_server(proc)

    return results


# ============================================================
# Transformers sweep（batch generate）
# ============================================================

def sweep_transformers(
    cfg,
    run_id: str,
) -> list[BenchmarkResult]:
    """对 Transformers 执行渐进并发扫描。"""
    results = []
    sw_cfg = cfg.test.sweep

    model_name = cfg.model.path or cfg.model.name
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map.get(cfg.engines.transformers.dtype, torch.float16)

    logger.info("Loading model for Transformers sweep: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype,
        device_map=cfg.engines.transformers.device_map,
        trust_remote_code=True,
    )
    model.eval()

    gpu_monitor = GPUMonitor(
        device_index=int(cfg.gpu.device),
        interval_ms=cfg.gpu.monitor_interval_ms,
    )

    concurrency_levels = generate_sweep_concurrency(
        sw_cfg.start, sw_cfg.stop, sw_cfg.multiplier,
    )
    logger.info("Sweep concurrency levels: %s", concurrency_levels)

    for concurrency in concurrency_levels:
        logger.info("--- Sweep: transformers, concurrency=%d ---", concurrency)

        try:
            prompts = generate_batch_prompts(
                concurrency, sw_cfg.prompt_length, tokenizer=tokenizer,
            )

            all_inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True,
            ).to(model.device)

            gpu_monitor.start()
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
            # Batch generate 无法精确 TTFT，用估算
            mean_ttft_ms = total_time_s * 1000.0 / len(prompts)
            peak_vram = gpu_monitor.peak_vram_mb

            results.append(BenchmarkResult(
                engine="transformers",
                test_type="sweep",
                batch_size=concurrency,
                prompt_tokens=sw_cfg.prompt_length,
                max_new_tokens=sw_cfg.max_new_tokens,
                ttft_ms=round(mean_ttft_ms, 2),
                mean_tps=round(concurrent_tps, 2),
                peak_vram_mb=round(peak_vram, 1),
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
            ))

            logger.info(
                "  concurrency=%d: TTFT=%.1fms, TPS=%.1f, VRAM=%.0fMB",
                concurrency, mean_ttft_ms, concurrent_tps, peak_vram,
            )

        except Exception as e:
            logger.error("  Sweep transformers concurrency=%d failed: %s", concurrency, e)
            results.append(BenchmarkResult(
                engine="transformers",
                test_type="sweep",
                batch_size=concurrency,
                prompt_tokens=sw_cfg.prompt_length,
                max_new_tokens=sw_cfg.max_new_tokens,
                ttft_ms=-1,
                mean_tps=-1,
                peak_vram_mb=-1,
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
            ))

        time.sleep(3)

    # 释放显存
    del model
    torch.cuda.empty_cache()
    logger.info("Transformers GPU memory released")

    return results


# ============================================================
# 主入口
# ============================================================

async def main():
    parser = argparse.ArgumentParser(description="Progressive concurrency sweep")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument(
        "--engine",
        choices=["vllm", "sglang", "transformers", "all"],
        default="all",
        help="要测试的引擎（默认 all）",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_id = make_run_id(cfg.output.timestamp_format)
    all_results = []

    engines_to_test = (
        ["vllm", "sglang", "transformers"] if args.engine == "all" else [args.engine]
    )

    for engine in engines_to_test:
        logger.info("=" * 60)
        logger.info("Sweep: %s", engine)
        logger.info("=" * 60)

        if engine in ("vllm", "sglang"):
            engine_results = await sweep_http_engine(engine, cfg, run_id)
        else:
            engine_results = sweep_transformers(cfg, run_id)

        all_results.extend(engine_results)

        # 引擎间等待，让 GPU 显存释放
        logger.info("Waiting 30s for GPU memory cleanup...")
        await asyncio.sleep(30)

    if all_results:
        output_path = results_to_csv(
            all_results,
            f"{cfg.output.results_dir}/sweep_{run_id}.csv",
        )
        logger.info("Sweep results saved to %s", output_path)
    else:
        logger.warning("No sweep results collected")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: 提交**

```bash
git add run_sweep.py
git commit -m "feat: add progressive concurrency sweep script"
```

---

### Task 10: 报告生成脚本

**Files:**
- Create: `generate_report.py`

- [ ] **Step 1: 创建 generate_report.py**

```python
"""汇总 CSV → Markdown 报告 + PNG 图表。

读取 results/ 目录下所有 CSV，生成包含 7 张图表的对比报告。

用法:
    python generate_report.py [--results-dir results] [--output-dir reports]
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 中文字体支持
plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",  # macOS
    "WenQuanYi Micro Hei",  # Linux
    "SimHei",  # Windows
    "DejaVu Sans",  # fallback
]
plt.rcParams["axes.unicode_minus"] = False

# 统一配色
ENGINE_COLORS = {
    "vllm": "#1f77b4",
    "sglang": "#ff7f0e",
    "transformers": "#2ca02c",
}

ENGINE_LABELS = {
    "vllm": "vLLM",
    "sglang": "SGLang",
    "transformers": "Transformers",
}


def load_results(results_dir: str) -> pd.DataFrame:
    """加载 results/ 下所有 CSV 文件。"""
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    csv_files = sorted(results_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"  Loaded: {f.name} ({len(df)} rows)")

    combined = pd.concat(dfs, ignore_index=True)

    # 过滤掉失败的测试（值为 -1）
    combined = combined[
        (combined["ttft_ms"] >= 0) & (combined["mean_tps"] >= 0) & (combined["peak_vram_mb"] >= 0)
    ]

    return combined


def plot_single_request_ttft(df: pd.DataFrame, output_dir: Path):
    """图表 1: TTFT vs 输入长度（单请求，三条折线）。"""
    single = df[df["test_type"] == "single"].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    for engine in ["vllm", "sglang", "transformers"]:
        data = single[single["engine"] == engine].sort_values("prompt_tokens")
        if data.empty:
            continue
        ax.plot(
            data["prompt_tokens"], data["ttft_ms"],
            marker="o", linewidth=2, markersize=8,
            color=ENGINE_COLORS[engine],
            label=ENGINE_LABELS[engine],
        )

    ax.set_xlabel("输入长度 (tokens)")
    ax.set_ylabel("首 Token 延迟 TTFT (ms)")
    ax.set_title("单请求：TTFT vs 输入长度")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "1_ttft_vs_input_length.png", dpi=150)
    plt.close(fig)
    print("  Chart 1: TTFT vs 输入长度")


def plot_single_request_tps(df: pd.DataFrame, output_dir: Path):
    """图表 2: TPS vs 输入长度（单请求，三条折线）。"""
    single = df[df["test_type"] == "single"].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    for engine in ["vllm", "sglang", "transformers"]:
        data = single[single["engine"] == engine].sort_values("prompt_tokens")
        if data.empty:
            continue
        ax.plot(
            data["prompt_tokens"], data["mean_tps"],
            marker="o", linewidth=2, markersize=8,
            color=ENGINE_COLORS[engine],
            label=ENGINE_LABELS[engine],
        )

    ax.set_xlabel("输入长度 (tokens)")
    ax.set_ylabel("吞吐量 TPS (tokens/s)")
    ax.set_title("单请求：TPS vs 输入长度")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "2_tps_vs_input_length.png", dpi=150)
    plt.close(fig)
    print("  Chart 2: TPS vs 输入长度")


def plot_concurrent_tps(df: pd.DataFrame, output_dir: Path):
    """图表 3: TPS vs 并发数（三条折线）。"""
    concurrent = df[df["test_type"] == "concurrent"].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    for engine in ["vllm", "sglang", "transformers"]:
        data = concurrent[concurrent["engine"] == engine].sort_values("batch_size")
        if data.empty:
            continue
        ax.plot(
            data["batch_size"], data["mean_tps"],
            marker="o", linewidth=2, markersize=8,
            color=ENGINE_COLORS[engine],
            label=ENGINE_LABELS[engine],
        )

    ax.set_xlabel("并发数")
    ax.set_ylabel("吞吐量 TPS (tokens/s)")
    ax.set_title("并发吞吐：TPS vs 并发数")
    ax.set_xscale("log", base=2)
    ax.set_xticks(concurrent["batch_size"].unique())
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "3_tps_vs_concurrency.png", dpi=150)
    plt.close(fig)
    print("  Chart 3: TPS vs 并发数")


def plot_concurrent_ttft(df: pd.DataFrame, output_dir: Path):
    """图表 4: TTFT vs 并发数（三条折线）。"""
    concurrent = df[df["test_type"] == "concurrent"].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    for engine in ["vllm", "sglang", "transformers"]:
        data = concurrent[concurrent["engine"] == engine].sort_values("batch_size")
        if data.empty:
            continue
        ax.plot(
            data["batch_size"], data["ttft_ms"],
            marker="o", linewidth=2, markersize=8,
            color=ENGINE_COLORS[engine],
            label=ENGINE_LABELS[engine],
        )

    ax.set_xlabel("并发数")
    ax.set_ylabel("首 Token 延迟 TTFT (ms)")
    ax.set_title("并发吞吐：TTFT vs 并发数")
    ax.set_xscale("log", base=2)
    ax.set_xticks(concurrent["batch_size"].unique())
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "4_ttft_vs_concurrency.png", dpi=150)
    plt.close(fig)
    print("  Chart 4: TTFT vs 并发数")


def plot_sweep_dual_axis(df: pd.DataFrame, output_dir: Path):
    """图表 5: TPS & TTFT vs 并发数双Y轴图（渐进扫描）。"""
    sweep = df[df["test_type"] == "sweep"].copy()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    for engine in ["vllm", "sglang", "transformers"]:
        data = sweep[sweep["engine"] == engine].sort_values("batch_size")
        if data.empty:
            continue
        ax1.plot(
            data["batch_size"], data["mean_tps"],
            marker="o", linewidth=2, markersize=6,
            color=ENGINE_COLORS[engine],
            linestyle="-",
            label=f"{ENGINE_LABELS[engine]} TPS",
        )

    ax1.set_xlabel("并发数")
    ax1.set_ylabel("吞吐量 TPS (tokens/s)", color="black")
    ax1.set_xscale("log", base=2)
    if not sweep.empty:
        ax1.set_xticks(sweep["batch_size"].unique())
        ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.tick_params(axis="y")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    for engine in ["vllm", "sglang", "transformers"]:
        data = sweep[sweep["engine"] == engine].sort_values("batch_size")
        if data.empty:
            continue
        ax2.plot(
            data["batch_size"], data["ttft_ms"],
            marker="s", linewidth=2, markersize=6,
            color=ENGINE_COLORS[engine],
            linestyle="--",
            label=f"{ENGINE_LABELS[engine]} TTFT",
        )

    ax2.set_ylabel("首 Token 延迟 TTFT (ms)", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    ax1.set_title("渐进并发扫描：TPS & TTFT vs 并发数")
    fig.tight_layout()
    fig.savefig(output_dir / "5_sweep_tps_ttft.png", dpi=150)
    plt.close(fig)
    print("  Chart 5: 渐进并发扫描 TPS & TTFT")


def plot_vram_comparison(df: pd.DataFrame, output_dir: Path):
    """图表 6: 峰值显存 vs 并发数分组柱状图。"""
    # 合并 concurrent 和 sweep 数据
    concurrent = df[df["test_type"].isin(["concurrent", "sweep"])].copy()

    if concurrent.empty:
        print("  Chart 6: 跳过（无显存数据）")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    engines = ["vllm", "sglang", "transformers"]
    batch_sizes = sorted(concurrent["batch_size"].unique())
    x = np.arange(len(batch_sizes))
    width = 0.25

    for i, engine in enumerate(engines):
        data = concurrent[concurrent["engine"] == engine].sort_values("batch_size")
        # 对每个 batch_size 取平均值
        vram_values = []
        for bs in batch_sizes:
            bs_data = data[data["batch_size"] == bs]["peak_vram_mb"]
            vram_values.append(bs_data.mean() if not bs_data.empty else 0)

        bars = ax.bar(
            x + i * width, vram_values, width,
            color=ENGINE_COLORS[engine],
            label=ENGINE_LABELS[engine],
        )
        # 在柱子上标注数值
        for bar, val in zip(bars, vram_values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=7,
                )

    ax.set_xlabel("并发数")
    ax.set_ylabel("峰值显存增量 (MB)")
    ax.set_title("显存占用对比：峰值显存 vs 并发数")
    ax.set_xticks(x + width)
    ax.set_xticklabels(batch_sizes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "6_vram_vs_concurrency.png", dpi=150)
    plt.close(fig)
    print("  Chart 6: 峰值显存 vs 并发数")


def plot_radar(df: pd.DataFrame, output_dir: Path):
    """图表 7: 综合 Radar 图（TTFT/TPS/显存效率三维度归一化得分）。"""
    engines = ["vllm", "sglang", "transformers"]

    # 收集各引擎的代表性数据
    scores = {}
    for engine in engines:
        engine_data = df[df["engine"] == engine]
        if engine_data.empty:
            continue

        # TTFT 得分：越低越好，取单请求平均 TTFT 的倒数
        single = engine_data[engine_data["test_type"] == "single"]
        avg_ttft = single["ttft_ms"].mean() if not single.empty else float("inf")

        # TPS 得分：越高越好，取并发最大 TPS
        concurrent = engine_data[engine_data["test_type"].isin(["concurrent", "sweep"])]
        max_tps = concurrent["mean_tps"].max() if not concurrent.empty else 0

        # 显存效率得分：吞吐/显存，越高越好
        avg_vram = concurrent["peak_vram_mb"].mean() if not concurrent.empty else float("inf")
        vram_efficiency = max_tps / avg_vram if avg_vram > 0 else 0

        scores[engine] = {
            "TTFT (↓)": avg_ttft,
            "TPS (↑)": max_tps,
            "显存效率 (↑)": vram_efficiency,
        }

    if len(scores) < 2:
        print("  Chart 7: 跳过（数据不足）")
        return

    # 归一化：TTFT 越低越好，取倒数后归一化；TPS 和显存效率越高越好，直接归一化
    categories = ["TTFT (↓)", "TPS (↑)", "显存效率 (↑)"]
    normalized = {}

    # TTFT: 取倒数后归一化（越大越好）
    ttft_values = {e: 1.0 / max(scores[e]["TTFT (↓)"], 1) for e in scores}
    max_ttft_inv = max(ttft_values.values()) or 1

    # TPS: 直接归一化
    tps_values = {e: scores[e]["TPS (↑)"] for e in scores}
    max_tps_val = max(tps_values.values()) or 1

    # 显存效率: 直接归一化
    eff_values = {e: scores[e]["显存效率 (↑)"] for e in scores}
    max_eff_val = max(eff_values.values()) or 1

    for engine in scores:
        normalized[engine] = [
            ttft_values[engine] / max_ttft_inv,
            tps_values[engine] / max_tps_val,
            eff_values[engine] / max_eff_val,
        ]

    # 绘制 Radar 图
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for engine in scores:
        values = normalized[engine] + normalized[engine][:1]  # 闭合
        ax.plot(angles, values, "o-", linewidth=2,
                color=ENGINE_COLORS[engine], label=ENGINE_LABELS[engine])
        ax.fill(angles, values, alpha=0.1, color=ENGINE_COLORS[engine])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.1)
    ax.set_title("综合性能评价（归一化得分）", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(output_dir / "7_radar.png", dpi=150)
    plt.close(fig)
    print("  Chart 7: 综合 Radar 图")


def generate_markdown_report(df: pd.DataFrame, output_dir: Path, report_path: Path):
    """生成 Markdown 报告。"""
    engines_in_data = df["engine"].unique().tolist()
    run_ids = df["run_id"].unique().tolist()

    lines = []
    lines.append("# LLM 推理引擎性能对比报告\n")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## 测试环境\n")
    lines.append("- **GPU**: NVIDIA A10 24GB")
    lines.append("- **模型**: Qwen/Qwen2.5-7B-Instruct")
    lines.append(f"- **测试引擎**: {', '.join(ENGINE_LABELS.get(e, e) for e in engines_in_data)}")
    lines.append(f"- **运行 ID**: {', '.join(run_ids)}")
    lines.append("")
    lines.append("> **注意**: Transformers 的并发测试使用 batch generate（非 HTTP 级并发），")
    lines.append("> 与 vLLM/SGLang 的 continuous batching 语义不同，TTFT 为估算值。")
    lines.append("")

    # 单请求延迟
    lines.append("## 1. 单请求延迟\n")
    single = df[df["test_type"] == "single"].copy()
    if not single.empty:
        lines.append("| 引擎 | 输入长度 (tokens) | TTFT (ms) | TPS (tokens/s) | 峰值显存 (MB) |")
        lines.append("|------|-------------------|-----------|----------------|---------------|")
        for _, row in single.sort_values(["engine", "prompt_tokens"]).iterrows():
            lines.append(
                f"| {ENGINE_LABELS.get(row['engine'], row['engine'])} "
                f"| {int(row['prompt_tokens'])} "
                f"| {row['ttft_ms']:.2f} "
                f"| {row['mean_tps']:.2f} "
                f"| {row['peak_vram_mb']:.0f} |"
            )
        lines.append("")
        lines.append("![TTFT vs 输入长度](1_ttft_vs_input_length.png)")
        lines.append("")
        lines.append("![TPS vs 输入长度](2_tps_vs_input_length.png)")
    lines.append("")

    # 并发吞吐
    lines.append("## 2. 并发吞吐\n")
    concurrent = df[df["test_type"] == "concurrent"].copy()
    if not concurrent.empty:
        lines.append("| 引擎 | 并发数 | TTFT (ms) | TPS (tokens/s) | 峰值显存 (MB) |")
        lines.append("|------|--------|-----------|----------------|---------------|")
        for _, row in concurrent.sort_values(["engine", "batch_size"]).iterrows():
            lines.append(
                f"| {ENGINE_LABELS.get(row['engine'], row['engine'])} "
                f"| {int(row['batch_size'])} "
                f"| {row['ttft_ms']:.2f} "
                f"| {row['mean_tps']:.2f} "
                f"| {row['peak_vram_mb']:.0f} |"
            )
        lines.append("")
        lines.append("![TPS vs 并发数](3_tps_vs_concurrency.png)")
        lines.append("")
        lines.append("![TTFT vs 并发数](4_ttft_vs_concurrency.png)")
    lines.append("")

    # 渐进并发扫描
    lines.append("## 3. 渐进并发扫描\n")
    sweep = df[df["test_type"] == "sweep"].copy()
    if not sweep.empty:
        lines.append("| 引擎 | 并发数 | TTFT (ms) | TPS (tokens/s) | 峰值显存 (MB) |")
        lines.append("|------|--------|-----------|----------------|---------------|")
        for _, row in sweep.sort_values(["engine", "batch_size"]).iterrows():
            lines.append(
                f"| {ENGINE_LABELS.get(row['engine'], row['engine'])} "
                f"| {int(row['batch_size'])} "
                f"| {row['ttft_ms']:.2f} "
                f"| {row['mean_tps']:.2f} "
                f"| {row['peak_vram_mb']:.0f} |"
            )
        lines.append("")
        lines.append("![渐进并发扫描 TPS & TTFT](5_sweep_tps_ttft.png)")
    lines.append("")

    # 显存对比
    lines.append("## 4. 显存占用对比\n")
    lines.append("![峰值显存 vs 并发数](6_vram_vs_concurrency.png)")
    lines.append("")

    # 综合评价
    lines.append("## 5. 综合评价\n")
    lines.append("![综合 Radar 图](7_radar.png)")
    lines.append("")
    lines.append("### 维度说明\n")
    lines.append("- **TTFT (↓)**: 首 Token 延迟，越低越好（取倒数后归一化）")
    lines.append("- **TPS (↑)**: 吞吐量，越高越好（并发最大 TPS 归一化）")
    lines.append("- **显存效率 (↑)**: 吞吐/显存比值，越高越好（归一化）")
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("--results-dir", default="results", help="CSV 结果目录")
    parser.add_argument("--output-dir", default="reports", help="报告输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    df = load_results(args.results_dir)
    print(f"  Total: {len(df)} rows, engines: {df['engine'].unique().tolist()}")

    print("\nGenerating charts...")
    plot_single_request_ttft(df, output_dir)
    plot_single_request_tps(df, output_dir)
    plot_concurrent_tps(df, output_dir)
    plot_concurrent_ttft(df, output_dir)
    plot_sweep_dual_axis(df, output_dir)
    plot_vram_comparison(df, output_dir)
    plot_radar(df, output_dir)

    print("\nGenerating markdown report...")
    report_path = output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    generate_markdown_report(df, output_dir, report_path)
    print(f"  Report saved to: {report_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 提交**

```bash
git add generate_report.py
git commit -m "feat: add report generation script with 7 charts and markdown report"
```

---

### Task 11: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: 创建 README.md**

```markdown
# infer-bench

LLM 推理引擎性能基准测试工具。在相同 GPU 上对比 vLLM、SGLang、原生 Transformers 的推理性能。

## 测试指标

| 指标 | 说明 |
|------|------|
| TTFT | 首 Token 延迟 (ms) |
| TPS | 吞吐量 (tokens/s) |
| Peak VRAM | 峰值显存增量 (MB) |

## 测试场景

1. **单请求延迟** — 不同 prompt 长度 (128/512/1024 tokens) 下的延迟和吞吐
2. **并发吞吐** — 不同并发数 (1/2/4/8/16) 下的吞吐和延迟
3. **渐进并发扫描** — 并发 1→32 倍增，观察 TPS/TTFT 趋势

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 修改配置（可选）

编辑 `config.yaml` 调整模型路径、GPU 设备、测试参数等。

### 3. 运行测试

```bash
# 运行单个引擎
python run_vllm.py
python run_sglang.py
python run_transformers.py

# 渐进并发扫描（所有引擎）
python run_sweep.py

# 只扫描特定引擎
python run_sweep.py --engine vllm
```

### 4. 生成报告

```bash
python generate_report.py
```

报告和图表保存在 `reports/` 目录。

## 注意事项

- **GPU**: 需要 NVIDIA GPU，已针对 A10 24GB 配置；其他 GPU 需调整 `config.yaml`
- **显存**: 大并发可能 OOM，程序会记录失败并继续测试
- **端口**: vLLM 使用 8000，SGLang 使用 8001，请确保端口空闲
- **并发语义**: Transformers 使用 batch generate 而非 HTTP 级并发，与 vLLM/SGLang 的 continuous batching 语义不同
- **运行时间**: 完整测试（3 引擎 + sweep）预计需要 1-2 小时

## 项目结构

```
infer-bench/
├── config.yaml              # 全局配置
├── common/                  # 共享模块
│   ├── config.py            # 配置加载
│   ├── prompts.py           # Prompt 生成
│   ├── gpu.py               # GPU 显存监控
│   ├── metrics.py           # 指标计算和 CSV 输出
│   └── client.py            # OpenAI API 客户端
├── run_vllm.py              # vLLM 测试
├── run_sglang.py            # SGLang 测试
├── run_transformers.py      # Transformers 测试
├── run_sweep.py             # 渐进并发扫描
├── generate_report.py       # 报告生成
├── results/                 # CSV 结果
└── reports/                 # Markdown 报告 + 图表
```
```

- [ ] **Step 2: 提交**

```bash
git add README.md
git commit -m "docs: add README with quick start guide"
```

---

### Task 12: 端到端验证

**Files:** 无新文件

- [ ] **Step 1: 验证所有模块可导入**

```bash
cd /Users/wt/share/python/infer-bench1
python -c "
from common.config import load_config
from common.prompts import SEEDS, HISTORY_TURNS, generate_prompt, generate_batch_prompts
from common.gpu import GPUMonitor
from common.metrics import BenchmarkResult, compute_ttft_ms, compute_tps, results_to_csv, make_run_id
from common.client import wait_for_server, stream_request, concurrent_stream_requests
print('All modules imported successfully')
cfg = load_config()
print(f'Config loaded: model={cfg.model.name}, gpu={cfg.gpu.device}')
print(f'Seeds: {list(SEEDS.keys())}, History: {len(HISTORY_TURNS)} turns')
result = BenchmarkResult('test', 'single', 1, 512, 256, 45.2, 85.3, 14200.0, 'run_test', '2026-01-01')
print(f'BenchmarkResult: engine={result.engine}, ttft={result.ttft_ms}, tps={result.mean_tps}')
"
```

Expected: 所有模块导入成功，配置加载正常

- [ ] **Step 2: 验证 CSV 输出**

```bash
cd /Users/wt/share/python/infer-bench1
python -c "
from common.metrics import BenchmarkResult, results_to_csv, make_run_id
import tempfile, os

run_id = make_run_id()
results = [
    BenchmarkResult('vllm', 'single', 1, 128, 256, 30.5, 120.0, 8000.0, run_id, '2026-01-01T00:00:00'),
    BenchmarkResult('vllm', 'single', 1, 512, 256, 45.2, 85.3, 8500.0, run_id, '2026-01-01T00:00:01'),
]
path = results_to_csv(results, 'results/test_verify.csv')
print(f'CSV written to: {path}')
with open(path) as f:
    print(f.read())
os.remove(path)
"
```

Expected: CSV 文件创建成功，包含 header 和数据行

- [ ] **Step 3: 验证报告生成（使用 mock 数据）**

```bash
cd /Users/wt/share/python/infer-bench1
python -c "
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, '.')
from generate_report import (
    plot_single_request_ttft, plot_single_request_tps,
    plot_concurrent_tps, plot_concurrent_ttft,
    plot_sweep_dual_axis, plot_vram_comparison, plot_radar,
    generate_markdown_report,
)

# 创建 mock 数据
data = {
    'engine': ['vllm']*6 + ['sglang']*6 + ['transformers']*6,
    'test_type': ['single']*3 + ['concurrent']*3 + ['single']*3 + ['concurrent']*3 + ['single']*3 + ['concurrent']*3,
    'batch_size': [1,1,1,1,4,16, 1,1,1,1,4,16, 1,1,1,1,4,16],
    'prompt_tokens': [128,512,1024,512,512,512, 128,512,1024,512,512,512, 128,512,1024,512,512,512],
    'max_new_tokens': [256]*18,
    'ttft_ms': [30,45,80, 50,120,300, 25,40,70, 45,110,280, 100,200,400, 150,400,1200],
    'mean_tps': [120,85,60, 80,200,350, 130,95,70, 90,220,380, 50,40,30, 45,60,80],
    'peak_vram_mb': [8000,8500,9000, 8500,11000,16000, 7500,8000,8500, 8000,10500,15500, 7000,7000,7000, 7000,9000,14000],
    'run_id': ['test_run']*18,
    'timestamp': ['2026-01-01']*18,
}
df = pd.DataFrame(data)

output_dir = Path('reports')
output_dir.mkdir(exist_ok=True)

plot_single_request_ttft(df, output_dir)
plot_single_request_tps(df, output_dir)
plot_concurrent_tps(df, output_dir)
plot_concurrent_ttft(df, output_dir)

# 添加 sweep 数据
sweep_data = {
    'engine': ['vllm']*5 + ['sglang']*5 + ['transformers']*5,
    'test_type': ['sweep']*15,
    'batch_size': [1,2,4,8,16]*3,
    'prompt_tokens': [512]*15,
    'max_new_tokens': [256]*15,
    'ttft_ms': [30,60,120,250,400, 25,55,110,230,380, 100,200,400,800,1200],
    'mean_tps': [80,150,250,350,380, 90,170,280,400,420, 45,55,60,65,80],
    'peak_vram_mb': [8000,9000,11000,14000,16000, 7500,8500,10500,13500,15500, 7000,7500,9000,11000,14000],
    'run_id': ['test_run']*15,
    'timestamp': ['2026-01-01']*15,
}
sweep_df = pd.DataFrame(sweep_data)
full_df = pd.concat([df, sweep_df], ignore_index=True)

plot_sweep_dual_axis(full_df, output_dir)
plot_vram_comparison(full_df, output_dir)
plot_radar(full_df, output_dir)

report_path = output_dir / 'benchmark_report_test.md'
generate_markdown_report(full_df, output_dir, report_path)
print('Report and charts generated successfully!')
print('Charts:', sorted(p.name for p in output_dir.glob('*.png')))
"
```

Expected: 7 张 PNG 图表 + 1 个 Markdown 报告生成成功

- [ ] **Step 4: 清理 mock 数据并提交**

```bash
rm -f results/test_verify.csv reports/benchmark_report_test.md reports/*.png
git add -A
git commit -m "chore: end-to-end module verification complete"
```

---

## Self-Review

### Spec Coverage Check

| 设计规格要求 | 对应 Task |
|---|---|
| config.yaml 配置 | Task 1 |
| 共享模块 common/ | Task 1-5 |
| Prompt 生成 (种子+扩展) | Task 2 |
| GPU 显存监控 (pynvml) | Task 3 |
| TTFT/TPS 指标 + CSV | Task 4 |
| OpenAI API 客户端 | Task 5 |
| vLLM 测试脚本 | Task 6 |
| SGLang 测试脚本 | Task 7 |
| Transformers 测试脚本 | Task 8 |
| 渐进并发扫描 | Task 9 |
| 报告生成 + 7 张图表 | Task 10 |
| README | Task 11 |
| 端到端验证 | Task 12 |

### Placeholder Scan

- ✅ 无 TBD/TODO/待定内容
- ✅ 所有步骤包含完整代码
- ✅ 所有命令包含预期输出

### Type Consistency

- ✅ `BenchmarkResult` dataclass 在 Task 4 定义，Task 6-9 一致使用
- ✅ `GPUMonitor` 在 Task 3 定义，Task 6-9 一致使用 `.start()/.stop()/.peak_vram_mb`
- ✅ `generate_prompt()` / `generate_batch_prompts()` 签名在 Task 2 定义，后续调用一致
- ✅ `load_config()` 返回 `BenchmarkConfig`，所有脚本一致使用
- ✅ CSV fieldnames 在 Task 4 定义，与 `BenchmarkResult` 字段名一致
- ✅ config.yaml 中 `sweep.multiplier: 2` 与 Task 9 的 `generate_sweep_concurrency()` 乘法逻辑一致