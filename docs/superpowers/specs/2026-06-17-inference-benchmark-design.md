# LLM 推理引擎基准测试 — 设计文档

## 概述

构建 `infer-bench` 项目，在相同 GPU（A10 24GB）上对比 vLLM、SGLang、原生 Transformers 三个推理引擎的性能，输出包含六张图表的 Markdown 对比报告。

## 测试目标

- **模型**: Qwen/Qwen2.5-7B-Instruct
- **GPU**: NVIDIA A10 24GB（device 0）
- **引擎**: vLLM（server 模式）、SGLang（server 模式）、Transformers（进程内 generate()）

## 核心指标

| 指标 | 缩写 | 采集方式 | 精度 |
|------|------|----------|------|
| 首 Token 延迟 | TTFT | stream 首 chunk 时间戳 − 请求发出时间 | ms |
| 吞吐量 | TPS | 总输出 token 数 / 总耗时（含并发） | tokens/s |
| 峰值显存占用 | Peak VRAM | pynvml 后台线程每 100ms 采样，取峰值增量 | MB |

## 架构：直连 API 模式

采用方案 A——每个引擎独立运行：

- vLLM 和 SGLang 通过 OpenAI-compatible HTTP API 提供服务，用统一客户端发送请求并采集时间戳
- Transformers 通过 `TextIteratorStreamer` 流式输出，进程内直接调用 `model.generate()`
- GPU 显存通过 `pynvml` 后台线程独立采样，不依赖引擎 API
- 各脚本独立运行，结果输出到 CSV，最后由汇总脚本生成报告

## 项目结构

```
infer-bench/
├── config.yaml              # 全局配置：模型、GPU、prompt、并发级别
├── run_vllm.py              # vLLM 基准测试脚本
├── run_sglang.py            # SGLang 基准测试脚本
├── run_transformers.py      # 原生 Transformers 基准测试脚本
├── run_sweep.py             # 渐进并发扫描（对三个引擎循环测试）
├── generate_report.py       # 汇总 CSV → Markdown 报告 + PNG 图表
├── common/                  # 共享工具
│   ├── __init__.py
│   ├── config.py            # 配置加载（yaml → dataclass）
│   ├── metrics.py           # 指标采集工具（TTFT/TPS 计算）
│   ├── client.py            # OpenAI-compatible API 客户端（vLLM/SGLang 共用）
│   ├── prompts.py           # 种子 prompt 定义 + 长度扩展 + tokenize 截断
│   └── gpu.py               # GPU 显存监控（pynvml 后台线程）
├── results/                 # 测试结果输出目录（CSV）
│   └── .gitkeep
├── reports/                 # 报告输出目录（Markdown + PNG）
│   └── .gitkeep
├── requirements.txt         # 依赖
└── README.md                # 使用说明
```

## 配置文件（config.yaml）

```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  path: null  # 本地路径，null 则从 HuggingFace 下载

gpu:
  device: "0"
  monitor_interval_ms: 100  # GPU 显存采样间隔

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
    step: 2        # 1, 2, 4, 8, 16, 32
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

## 指标采集方案

### vLLM / SGLang（OpenAI API 模式）

- 启用 `stream=True`，收到第一个 chunk 的时间戳减去请求发出时间 = TTFT
- 累计所有输出 token 数 / (最后一个 chunk 时间 − 请求发出时间) = 单请求 TPS
- 并发场景：所有请求的总输出 token 数 / (最晚完成时间 − 最早开始时间) = 并发 TPS
- 使用 `asyncio` + `aiohttp` 发送并发请求

### Transformers（进程内模式）

- 使用 `TextIteratorStreamer` 实现流式输出
- TTFT = 第一个 token 到达时间 − generate() 调用开始时间
- TPS = 总输出 token 数 / generate() 总耗时
- 并发场景使用 `ThreadPoolExecutor` 模拟并发请求（受 GIL 限制，但模拟的是真实的 batch generate 场景）

### GPU 显存监控

- `common/gpu.py` 使用 `pynvml` 每 100ms 采样 GPU 已用显存
- 测试前记录基线值，测试期间后台线程持续采样
- 测试结束后报告峰值增量（peak − baseline）

## Prompt 生成策略

采用**种子对话 + 上下文扩展**方案，确保不同长度 prompt 的变量唯一（只有长度），且语义连贯、KV cache 访问模式接近真实多轮对话。

### 种子 Prompt 设计

预设 3 个高质量种子 prompt，覆盖典型场景：

| 种子 | 场景 | 内容 |
|------|------|------|
| seed_qa | 多轮问答 | 一个关于技术概念的问答对话（如解释注意力机制） |
| seed_summary | 文档摘要 | 一段长文本要求模型生成摘要 |
| seed_code | 代码生成 | 给出需求描述，要求生成代码 |

### 长度扩展机制

1. 用模型 tokenizer 对种子 prompt 进行 tokenize，获取 token 数
2. 短 prompt（~128 tokens）：直接使用种子 prompt，可能需截断到目标长度
3. 中 prompt（~512 tokens）：在种子 prompt 前追加 N 轮历史对话（预设好的多轮 QA）
4. 长 prompt（~1024 tokens）：追加更多轮历史对话

关键约束：
- **所有长度使用同一组种子内容**，只是历史轮次不同 → 控制变量
- 扩展内容是一段完整的多轮对话历史，不是随机填充 → KV cache 访问模式真实
- 使用模型的 tokenizer 精确 tokenize 后截断到目标 token 数 → 长度准确
- 种子 prompt 内容硬编码在 `common/prompts.py`，不依赖外部数据集

### 并发场景的 Prompt 池

并发测试中，每个并发请求使用同一模板但不同 seed（如不同问题）生成的 prompt，避免 KV cache 完全命中带来的不公平优化。具体做法：为每个并发请求分配不同的 history seed index，从预设的多个变体中选取。

### 新增文件

```
common/
├── prompts.py    # 种子 prompt 定义 + 长度扩展 + tokenize 截断
```

`common/prompts.py` 提供：
- `generate_prompt(target_tokens: int, seed: str = "seed_qa") -> str`：生成指定 token 数的 prompt
- `generate_batch_prompts(batch_size: int, target_tokens: int) -> list[str]`：生成一批不同内容但相同长度的 prompt

## 测试场景

### 场景 1：单请求延迟

- 不同 prompt 长度（128, 512, 1024 tokens）
- 每组预热 3 次，测试 5 次取平均
- 输出指标：TTFT、TPS、峰值显存

### 场景 2：并发吞吐

- 不同 batch_size（1, 2, 4, 8, 16）
- 固定 prompt 长度 512 tokens
- 每组预热 2 次，测试 3 次取平均
- 输出指标：TTFT（平均）、TPS（总吞吐）、峰值显存

### 场景 3：渐进并发扫描

- 并发数从 1 到 32 按 2 倍增长（1, 2, 4, 8, 16, 32）
- 固定 prompt 512 tokens，max_new_tokens 256
- 每级测试 1 次（扫描侧重趋势而非精确值）
- 输出 TPS + TTFT 随并发数的变化曲线

## CSV 输出格式

每个测试脚本输出 CSV 到 `results/`：

```csv
engine,test_type,batch_size,prompt_tokens,max_new_tokens,ttft_ms,mean_tps,peak_vram_mb,run_id,timestamp
vllm,single,1,512,256,45.2,85.3,14200,run_20260617_143000,2026-06-17T14:30:00
```

文件命名：`{engine}_{test_type}_{run_id}.csv`

## 报告结构

```markdown
# LLM 推理引擎性能对比报告

## 测试环境
- GPU: NVIDIA A10 24GB
- 模型: Qwen/Qwen2.5-7B-Instruct
- 日期
- 引擎版本: vLLM, SGLang, Transformers

## 1. 单请求延迟
- 数据表格
- 图表 1: TTFT vs 输入长度折线图
- 图表 2: TPS vs 输入长度折线图

## 2. 并发吞吐
- 数据表格
- 图表 3: TPS vs 并发数折线图
- 图表 4: TTFT vs 并发数折线图

## 3. 渐进并发扫描
- 图表 5: TPS & TTFT vs 并发数双Y轴图

## 4. 显存占用对比
- 图表 6: 峰值显存 vs 并发数分组柱状图

## 5. 综合评价
- 各引擎优势场景
- Radar 图（TTFT/TPS/显存效率三维度归一化得分）
- 建议
```

## 图表清单

共 6 张 PNG + 1 张 Radar：

1. TTFT vs 输入长度（单请求，三条折线）
2. TPS vs 输入长度（单请求，三条折线）
3. TPS vs 并发数（三条折线）
4. TTFT vs 并发数（三条折线）
5. TPS & TTFT vs 并发数双Y轴图（渐进扫描）
6. 峰值显存 vs 并发数分组柱状图
7. 综合 Radar 图（三维度归一化得分）

使用 `matplotlib` + `seaborn` 绘图，支持中文字体。

## 脚本运行流程

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行各引擎测试（按需运行，可单独运行）
python run_vllm.py              # 测试 vLLM（自动启动/停止服务）
python run_sglang.py            # 测试 SGLang（自动启动/停止服务）
python run_transformers.py      # 测试 Transformers（进程内）

# 3. 渐进并发扫描
python run_sweep.py              # 对三个引擎循环测试

# 4. 生成报告
python generate_report.py        # 读取 results/ 下所有 CSV，生成报告
```

每个 `run_*.py` 脚本负责：
1. 加载 config.yaml
2. 启动引擎服务（Transformers 除外）
3. 等待服务就绪（健康检查）
4. 运行测试场景，采集指标
5. 停止引擎服务
6. 输出 CSV 到 results/

## 依赖

```
vllm
sglang
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

## 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| A10 24GB 显存不够跑大并发 | sweep 的 stop=32，如果 OOM 则记录为失败继续测试其他级别 |
| vLLM/SGLang 端口冲突 | 各引擎使用不同端口（8000/8001），且按顺序运行 |
| 模型下载慢 | 支持本地路径配置，首次运行后模型会缓存 |
| 引擎启动超时 | 设定 300s 启动超时，健康检查轮询 |
| GPU 显存未被释放 | 每个引擎测试后强制 kill 进程，torch.cuda.empty_cache() |