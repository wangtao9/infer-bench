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
    """构建聊天消息列表，可选追加历史轮次。"""
    seed = SEEDS[seed_key]
    messages = [{"role": "system", "content": seed["system"]}]

    for i in range(min(num_history_turns, len(HISTORY_TURNS))):
        messages.append(HISTORY_TURNS[i])

    messages.append({"role": "user", "content": seed["question"]})
    return messages


def _tokenize_and_truncate(
    messages: list[dict],
    target_tokens: int,
    tokenizer,
) -> str:
    """用 tokenizer 将消息 tokenize 后截断到目标 token 数，返回解码后的字符串。"""
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
    """生成指定 token 数的 prompt。"""
    if seed not in SEEDS:
        raise ValueError(f"Unknown seed: {seed}. Must be one of {list(SEEDS.keys())}")
    if tokenizer is None:
        raise ValueError("tokenizer is required for prompt generation")

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
            return _tokenize_and_truncate(messages, target_tokens, tokenizer)
        best_messages = messages
        best_len = n_tokens

    return _tokenize_and_truncate(best_messages, target_tokens, tokenizer)


def generate_batch_prompts(
    batch_size: int,
    target_tokens: int,
    tokenizer=None,
) -> list[str]:
    """生成一批不同内容但相同目标长度的 prompt。"""
    if tokenizer is None:
        raise ValueError("tokenizer is required for prompt generation")

    seed_keys = list(SEEDS.keys())
    prompts = []
    for i in range(batch_size):
        seed_key = seed_keys[i % len(seed_keys)]
        prompt = generate_prompt(target_tokens, seed=seed_key, tokenizer=tokenizer)
        prompts.append(prompt)
    return prompts