# infer-bench

LLM 推理引擎性能基准测试工具。在相同 GPU 上对比 vLLM、SGLang、原生 Transformers 的推理性能。

## 测试指标

| 指标 | 缩写 | 说明 |
|------|------|------|
| Time to First Token | TTFT | 首 Token 延迟 (ms) |
| Throughput | TPS | 输出吞吐量 (tokens/s) |
| Inter-Token Latency | ITL | 逐 Token 间隔延迟 (ms)，仅流式引擎可测 |
| Time Per Output Token | TPOT | 每个 Output Token 的平均时间 (ms) = (E2EL - TTFT) / (output_tokens - 1) |
| Peak VRAM (Δ) | Peak VRAM | 峰值显存增量 (MB)，相对于基线的增长量 |
| Peak VRAM (abs) | Peak VRAM abs | 峰值显存绝对占用量 (MB)，始终有意义 |

> **百分位统计**：TTFT / ITL / TPOT 均计算 mean、median、P99。
> **显存说明**：vLLM/SGLang 启动时预分配显存，Peak VRAM (Δ) ≈ 0，应关注 Peak VRAM (abs)；Transformers 两个指标均有意义。

## 测试场景

1. **单请求延迟** — 不同 prompt 长度（默认 128/512/1024 tokens）下的延迟和吞吐，含 warmup
2. **并发吞吐** — 固定总请求数（默认 64）下，按不同 request rate 发出请求；支持 batch 模式（inf，同时发出）和 **Poisson 调度**（有限值，按指数分布间隔发出，模拟真实负载，对标 vLLM `bench serve --request-rate`）
3. **渐进并发扫描** — 并发从 start 到 stop 按倍数递增（默认 1→2→4→8→16→32），观察 TPS/TTFT 趋势

## 快速开始

### 1. 安装依赖

```bash
# 按需创建环境（vLLM 和 SGLang 不可装在同一环境，需分别安装）
conda create -n vllm python=3.12
conda create -n sglang python=3.12
conda create -n transformers python=3.12

# 各环境分别安装
conda activate vllm          && pip install -r requirements-vllm.txt
conda activate sglang        && pip install -r requirements-sglang.txt
conda activate transformers  && pip install -r requirements-transformers.txt
```

### 2. 修改配置（可选）

编辑 `config.yaml` 调整模型路径、GPU 设备、测试参数等。主要配置项：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `model.name` | 模型名称（HuggingFace ID） | `Qwen/Qwen2.5-7B-Instruct` |
| `model.path` | 本地模型路径，`null` 则从 HuggingFace 下载 | `/home/wt/models/Qwen2.5-7B-Instruct` |
| `gpu.device` | CUDA 可见设备 | `"0"` |
| `test.single_request.prompt_lengths` | 单请求测试的 prompt 长度列表 | `[128, 512, 1024]` |
| `test.single_request.max_new_tokens` | 单请求最大生成 token 数 | `256` |
| `test.single_request.num_warmup` | 单请求 warmup 次数 | `3` |
| `test.single_request.num_requests` | 单请求正式测量次数 | `10` |
| `test.concurrent.num_requests` | 并发测试总请求数 | `64` |
| `test.concurrent.prompt_length` | 并发测试 prompt 长度 | `512` |
| `test.concurrent.request_rate` | 请求速率列表；`inf`=batch，有限值=Poisson | `[4, 8, inf]` |
| `test.sweep.start / stop / multiplier` | 扫描并发范围和乘数 | `1 / 32 / 2` |
| `engines.vllm.port` | vLLM 服务端口 | `8000` |
| `engines.vllm.extra_args` | vLLM 启动额外参数 | `--gpu-memory-utilization 0.9 --max-model-len 4096` |
| `engines.sglang.port` | SGLang 服务端口 | `8001` |
| `engines.sglang.extra_args` | SGLang 启动额外参数 | `--mem-fraction-static 0.9 --context-length 4096` |
| `engines.transformers.dtype` | Transformers 模型精度 | `float16` |
| `output.results_dir` | CSV 结果输出目录 | `results` |
| `output.reports_dir` | 报告输出目录 | `reports` |

### 3. 运行测试

```bash
# 运行单个引擎（自动启停服务器）
python run_vllm.py
python run_sglang.py
python run_transformers.py

# 渐进并发扫描（所有引擎）
python run_sweep.py

# 只扫描特定引擎
python run_sweep.py --engine vllm
```

所有脚本均支持 `--config` 指定配置文件路径（默认 `config.yaml`）。

### 4. 生成报告

```bash
python generate_report.py
# 或指定路径
python generate_report.py --results-dir results --output-dir reports
```

报告和图表保存在 `reports/` 目录。生成的图表包括：

| # | 图表 | 说明 |
|---|------|------|
| 1 | TTFT vs 输入长度 | 单请求场景，实线=mean，虚线=P99 |
| 2 | TPS vs 输入长度 | 单请求场景 |
| 3 | TPS vs 并发数 | 并发测试 |
| 4 | TTFT vs 并发数 | 并发测试，实线=mean，虚线=P99 |
| 5 | TPS & TTFT vs 并发数 | 渐进扫描，双 Y 轴 |
| 6 | 峰值显存 vs 并发数 | 分组柱状图 |
| 7 | 综合雷达图 | TTFT↓ / ITL↓ / TPS↑ / 显存效率↑ |
| 8 | ITL P99 vs 输入长度 | 单请求场景（Transformers 不可测，自动跳过） |
| 9 | TPOT P99 vs 并发数 | 并发 + 扫描数据 |
| 11 | TPS vs Request Rate | Poisson 调度模式（仅有限 request_rate 时生成） |
| 12 | TTFT vs Request Rate | Poisson 调度模式，实线=mean，虚线=P99 |

## 输出文件

- **CSV 结果**：`results/{engine}_{run_id}.csv` 或 `results/sweep_{run_id}.csv`
- **Markdown 报告**：`reports/benchmark_report.md`
- **图表**：`reports/*.png`

`run_id` 格式为 `run_YYYYMMDD_HHMMSS`。

## 注意事项

- **GPU**：需要 NVIDIA GPU，已针对 A10 24GB 配置；其他 GPU 需调整 `config.yaml` 中的 `max-model-len` 等参数
- **显存**：大并发可能 OOM，程序会记录失败（指标值为 -1）并继续测试
- **端口**：vLLM 使用 8000，SGLang 使用 8001，请确保端口空闲
- **并发语义**：Transformers 使用同步 batch generate 而非 HTTP 级并发，与 vLLM/SGLang 的 continuous batching 语义不同，TTFT 和 TPS 不完全可比
- **ITL 不可测**：Transformers 批量模式下 ITL 无法测量，CSV 中记为 -1.0，报告中显示 N/A
- **运行时间**：完整测试（3 引擎 + sweep）预计需要 1-2 小时
- **依赖冲突**：vLLM 和 SGLang 不可装在同一环境，需用 `requirements-vllm.txt` / `requirements-sglang.txt` 分别安装

## 项目结构

```
infer-bench/
├── config.yaml                   # 全局配置
├── common/                       # 共享模块
│   ├── config.py                 # 配置加载（YAML → dataclass）
│   ├── prompts.py                # Prompt 生成（3 种子 × 3 历史池 × variant 组合）
│   ├── client.py                 # OpenAI API 客户端（SSE 流式 + Poisson 调度）
│   ├── gpu.py                    # GPU 显存后台监控（pynvml）
│   └── metrics.py                # 指标计算（TTFT/TPS/ITL/TPOT 百分位）和 CSV 输出
├── run_vllm.py                   # vLLM 基准测试
├── run_sglang.py                 # SGLang 基准测试
├── run_transformers.py           # Transformers 基准测试
├── run_sweep.py                  # 渐进并发扫描
├── generate_report.py            # 报告和图表生成
├── requirements-common.txt       # 共用依赖
├── requirements-vllm.txt         # vLLM 环境依赖
├── requirements-sglang.txt       # SGLang 环境依赖
├── requirements-transformers.txt # Transformers 环境依赖
├── results/                      # CSV 结果输出
└── reports/                      # Markdown 报告 + 图表输出
```