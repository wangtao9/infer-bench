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