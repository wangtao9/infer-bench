"""种子 prompt 定义 + 长度扩展 + tokenize 截断。

提供 generate_prompt() 和 generate_batch_prompts() 两个公共接口，
为不同测试场景生成指定 token 数的 prompt。

设计要点：
- 3 个种子分类（QA/摘要/代码），每类多个不同 question
- 多组不同的历史轮次池，不同 variant 选不同池，最大限度减少内容重复
- 长度扩展通过追加多轮历史对话实现（不是随机填充）
- 使用模型 tokenizer 精确 tokenize 后截断到目标 token 数
- 通过 seed × question × history_pool 组合产生足够多的不同 prompt
"""

# ============================================================
# 种子 prompt 定义：每类多个 question
# ============================================================

SEEDS = {
    "seed_qa": {
        "system": "你是一个专业的技术顾问，请详细回答以下问题。",
        "questions": [
            "请解释 Transformer 模型中自注意力机制（Self-Attention）的工作原理，包括 Q、K、V 矩阵的作用、缩放点积注意力的计算过程，以及多头注意力如何增强模型的表达能力。",
            "请详细说明 Beam Search 和 Greedy Decoding 在序列生成任务中的区别，各自的优缺点，以及为什么 beam search 通常能产生更高质量的输出。",
            "请解释 BERT 和 GPT 在预训练目标上的根本差异，以及这种差异如何影响它们各自适用的下游任务类型。",
            "请分析 Gradient Accumulation 的工作原理，说明它如何在显存受限的情况下模拟大 batch 训练，以及与真正的大 batch 训练有何区别。",
            "请解释 Dropout 正则化的原理，说明训练和推理阶段行为不同的原因，以及 Dropout rate 的选择对模型性能的影响。",
            "请详细说明 LSTM 中门控机制（遗忘门、输入门、输出门）的计算过程，以及它是如何缓解 RNN 中的梯度消失问题的。",
        ],
    },
    "seed_summary": {
        "system": "你是一个专业的文档分析助手，请对以下内容生成摘要。",
        "questions": [
            "近年来，大语言模型（LLM）在自然语言处理领域取得了突破性进展。从 GPT 系列到 LLaMA、Mistral 等开源模型，参数规模从数十亿增长到数千亿，模型能力也随之大幅提升。然而，大模型的推理成本和部署难度也成为产业落地的核心挑战。推理优化技术如量化（Quantization）、蒸馏（Distillation）、剪枝（Pruning）和注意力优化（如 Flash Attention、Paged Attention）成为研究热点。vLLM 和 SGLang 等推理框架通过连续批处理（Continuous Batching）和前缀缓存（Prefix Caching）显著提升了推理吞吐。与此同时，模型压缩技术使得在消费级 GPU 上部署大模型成为可能，推动了 AI 应用的普及化。请对以上内容生成一段 200 字以内的中文摘要。",
            "多模态大模型正在重新定义人工智能的应用边界。GPT-4V、Gemini 等模型展示了同时理解文本、图像和音频的能力，使得文档理解、视觉问答、代码生成等任务取得质的飞跃。然而，多模态融合面临对齐难题——不同模态的表征空间差异巨大，如何有效对齐成为核心挑战。CLIP 通过对比学习实现了图文对齐，而 Flamingo 则采用交叉注意力机制在语言模型中注入视觉特征。训练数据方面，LAION 等大规模数据集为多模态预训练提供了基础，但数据质量和偏见问题仍需关注。请对以上内容生成一段 200 字以内的中文摘要。",
            "强化学习从人类反馈（RLHF）已成为大语言模型对齐的关键技术。InstructGPT 和 ChatGPT 的成功证明了 RLHF 能显著提升模型输出的人类偏好匹配度。RLHF 通常包含三个阶段：监督微调（SFT）、奖励模型训练和强化学习优化。然而，RLHF 也面临奖励模型过拟合、训练不稳定和样本效率低等挑战。Direct Preference Optimization（DPO）等替代方案通过直接从偏好数据学习策略，绕过奖励模型训练，提供了更稳定的训练范式。请对以上内容生成一段 200 字以内的中文摘要。",
            "检索增强生成（RAG）技术通过将外部知识库与语言模型结合，有效缓解了大模型的幻觉问题和知识过时问题。RAG 系统通常包含检索和生成两个核心模块：检索模块从知识库中找到相关文档片段，生成模块基于检索结果产生最终答案。向量数据库如 Milvus、Pinecone 为语义检索提供了高效基础设施，而稠密检索模型如 E5、BGE 显著提升了检索精度。当前 RAG 面临的主要挑战包括检索噪声、多跳推理和长文档处理。请对以上内容生成一段 200 字以内的中文摘要。",
            "模型服务化部署正在成为 MLOps 的核心环节。TensorFlow Serving、Triton Inference Server 和 vLLM 等框架提供了高性能的模型推理服务能力。动态批处理（Dynamic Batching）和请求调度优化可以显著提升 GPU 利用率，而模型并行和张量并行则解决了单卡显存不足的问题。自动扩缩容（Autoscaling）机制根据负载动态调整服务实例数量，实现成本与服务质量的平衡。可观测性方面，Prometheus 和 Grafana 构成了监控体系的基础。请对以上内容生成一段 200 字以内的中文摘要。",
        ],
    },
    "seed_code": {
        "system": "你是一个专业的编程助手，请根据需求生成代码。",
        "questions": [
            "请用 Python 实现一个线程安全的 LRU 缓存类，要求：1) 支持 get 和 put 操作；2) 容量满时淘汰最近最少使用的条目；3) 所有操作时间复杂度为 O(1)；4) 使用 threading.Lock 保证线程安全；5) 提供 __len__ 和 __contains__ 魔术方法。请附上简要的使用示例。",
            "请用 Python 实现一个异步任务调度器，要求：1) 支持定时任务和周期任务；2) 使用 asyncio 实现；3) 支持任务优先级；4) 提供取消和查看任务状态的功能；5) 内置简单的重试机制（最多重试 3 次）。请附上使用示例。",
            "请用 Python 实现一个简单的发布-订阅消息系统，要求：1) 支持多个 topic；2) 支持通配符订阅（如 'news.*' 匹配 'news.sports'）；3) 消息持久化到内存队列；4) 支持慢消费者检测和处理；5) 提供 subscribe、publish 和 unsubscribe 三个核心方法。请附上使用示例。",
            "请用 Python 实现一个有限状态机（FSM）框架，要求：1) 支持 state 和 transition 的声明式定义；2) transition 可配置 guard 条件和 action 回调；3) 自动检测无效 transition（不允许的状态转换）；4) 支持嵌套状态（子状态机）；5) 提供 current_state 和 available_transitions 查询接口。请附上使用示例。",
            "请用 Python 实现一个简单的连接池管理器，要求：1) 支持最小和最大连接数配置；2) 空闲连接超时自动回收；3) 连接健康检查；4) 等待超时机制（获取连接超过指定时间则抛异常）；5) 提供 acquire 和 release 上下文管理器接口。请附上使用示例。",
        ],
    },
}

# ============================================================
# 多组历史轮次池 — 不同 variant 选不同池
# ============================================================

HISTORY_POOLS = [
    # Pool 0: 深度学习基础
    [
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
    ],
    # Pool 1: 机器学习实践
    [
        {"role": "user", "content": "什么是交叉熵损失函数？为什么分类任务常用它？"},
        {"role": "assistant", "content": "交叉熵损失衡量模型预测的概率分布与真实标签分布之间的差异。在分类任务中常用，是因为它对错误预测的惩罚随置信度非线性增长——高置信度的错误预测会产生极大的损失值，从而有力地驱动模型修正。相比 MSE，交叉熵与 softmax 配合更自然，梯度也更稳定。"},
        {"role": "user", "content": "正则化有哪些常见方法？各有什么优缺点？"},
        {"role": "assistant", "content": "常见正则化方法包括 L1 正则化（产生稀疏解，可用于特征选择，但对相关特征选择不稳定）、L2 正则化（权重衰减，防止任何单一权重过大，解更平滑）、Dropout（训练时随机屏蔽神经元，近似模型集成）、Early Stopping（验证集性能不再提升时停止训练，简单有效但需留出验证集）和 Data Augmentation（数据增强，增加训练多样性但不改变模型结构）。"},
        {"role": "user", "content": "学习率衰减策略有哪些？如何选择？"},
        {"role": "assistant", "content": "常见策略包括：Step Decay（按固定步长衰减，简单但不够灵活）、Cosine Annealing（余弦退火，平滑衰减到最后学习率，训练后期仍有探索能力）、Warmup + Decay（先线性增后衰减，避免训练初期梯度不稳定）、ReduceLROnPlateau（验证指标停滞时衰减，自适应但可能过晚衰减）。选择原则：大 batch 训练推荐 Warmup + Cosine，小 batch 可试 Step Decay。"},
        {"role": "user", "content": "优化器 Adam 和 SGD 的区别是什么？该如何选择？"},
        {"role": "assistant", "content": "SGD 使用固定学习率或手动调度，收敛路径更稳定，泛化性能通常更优，但对超参数敏感且需要精细的学习率调度。Adam 为每个参数自适应调整学习率（基于一阶和二阶矩估计），收敛更快且对超参数鲁棒，但泛化性能有时不如精心调参的 SGD。实践中 NLP 任务常用 Adam，CV 任务 SGD 仍主流。"},
        {"role": "user", "content": "什么是梯度裁剪？为什么 RNN 中经常使用？"},
        {"role": "assistant", "content": "梯度裁剪（Gradient Clipping）是在反向传播后将梯度向量的范数限制在阈值以内。RNN 中频繁使用是因为长序列的反向传播容易导致梯度爆炸——梯度的指数增长使参数更新步长过大，训练崩溃。裁剪后的梯度方向不变，仅缩放幅度，既防止了数值不稳定，又保留了梯度指引的方向信息。"},
        {"role": "user", "content": "Batch Normalization 为什么能加速训练？"},
        {"role": "assistant", "content": "Batch Normalization 加速训练的核心原因是减少内部协变量偏移（Internal Covariate Shift）——每层的输入分布在前层参数更新后会发生变化，BN 通过归一化使每层输入分布在训练过程中保持稳定。这带来三个好处：允许更大的学习率、减少对初始化的敏感度、起到轻微正则化效果（因为 batch 统计量引入了噪声）。"},
        {"role": "user", "content": "什么是知识蒸馏？它和模型剪枝有什么区别？"},
        {"role": "assistant", "content": "知识蒸馏是用大模型（教师）的软标签（softmax 输出的概率分布）训练小模型（学生），使学生不仅学到硬标签还学到类别间的相似关系。模型剪枝则是直接删除大模型中不重要的参数（权重置零），使模型变稀疏。两者都是模型压缩方法，但蒸馏是训练一个新小模型，剪枝是在原模型上做结构精简。"},
    ],
    # Pool 2: 系统与工程
    [
        {"role": "user", "content": "什么是内存泄漏？在 Python 中如何检测？"},
        {"role": "assistant", "content": "内存泄漏是指程序不再使用的内存因引用未释放而无法被回收。Python 中常见原因是循环引用（尤其涉及 __del__ 方法的对象）、全局变量或缓存无限增长、以及 C 扩展中的原生泄漏。检测方法包括：使用 tracemalloc 追踪内存分配来源、objgraph 查看对象引用图、memory_profiler 逐行分析，以及用弱引用替代强引用打破循环。"},
        {"role": "user", "content": "异步编程中 async/await 的工作原理是什么？"},
        {"role": "assistant", "content": "async/await 基于 coroutine（协程）机制：async def 定义协程函数，调用时返回协程对象而非立即执行；await 挂起当前协程并将控制权交还事件循环（Event Loop），直到被 await 的操作完成后再恢复执行。这避免了线程切换开销，在 I/O 密集场景中用单线程实现高并发——当一个协程等待 I/O 时，其他协程可以运行。"},
        {"role": "user", "content": "GIL 对 Python 多线程有什么影响？"},
        {"role": "assistant", "content": "GIL（全局解释器锁）确保同一时刻只有一个线程执行 Python 字节码。对 CPU 密集任务，多线程无法真正并行，性能甚至不如单线程（因线程切换开销）。对 I/O 密集任务，影响较小——线程在等待 I/O 时会释放 GIL。因此 CPU 密集场景应使用 multiprocessing 或 C 扩展（如 NumPy 在计算时释放 GIL），I/O 密集场景可用 asyncio 或多线程。"},
        {"role": "user", "content": "什么是 CAP 定理？分布式系统如何取舍？"},
        {"role": "assistant", "content": "CAP 定理指出分布式系统最多同时满足一致性（C）、可用性（A）和分区容错性（P）中的两项。由于网络分区不可避免，实际选择是 CP（保证一致性和分区容错，如 ZooKeeper，分区时拒绝部分请求）或 AP（保证可用性和分区容错，如 Cassandra，分区时接受写入但可能不一致）。现代系统常采用最终一致性，在 AP 基础上异步修复不一致。"},
        {"role": "user", "content": "数据库索引的原理是什么？为什么能加速查询？"},
        {"role": "assistant", "content": "数据库索引通常基于 B+树实现：将列值按有序结构存储，查询时通过树形结构以 O(log n) 的复杂度定位到目标行，而非全表扫描的 O(n)。B+树的叶子节点通过链表串联，支持高效范围查询。但索引也有代价：占用额外存储、降低写入速度（每次 INSERT/UPDATE 需同步维护索引结构）、查询优化器可能选错索引。"},
        {"role": "user", "content": "微服务架构有什么优缺点？"},
        {"role": "assistant", "content": "微服务优点：服务独立部署和扩展（按需分配资源）、技术栈灵活（不同服务可用不同语言）、故障隔离（单服务崩溃不影响全局）、团队自治（按服务划分团队）。缺点：分布式系统的复杂性（网络延迟、数据一致性、分布式事务）、运维成本高（服务发现、监控、链路追踪）、测试难度大（需要集成测试环境）和服务间通信开销。适合大型团队和快速迭代场景。"},
        {"role": "user", "content": "Docker 容器和虚拟机的区别是什么？"},
        {"role": "assistant", "content": "虚拟机在硬件层虚拟化，每个 VM 运行完整的操作系统内核，通过 Hypervisor 共享物理硬件，隔离性强但开销大（启动慢、内存占用高）。Docker 容器在操作系统层虚拟化，所有容器共享宿主机内核，通过 namespace 和 cgroup 实现隔离，启动秒级、资源开销小，但隔离性弱于 VM（内核漏洞影响所有容器）。容器适合微服务和 CI/CD，VM 适合强隔离需求。"},
    ],
]


def _build_messages(
    seed_key: str,
    question_index: int = 0,
    num_history_turns: int = 0,
    history_pool: list[dict] | None = None,
) -> list[dict]:
    """构建聊天消息列表。

    Args:
        seed_key: 种子名称，如 "seed_qa"。
        question_index: 该 seed 下的 question 索引。
        num_history_turns: 追加的历史轮次数。
        history_pool: 可选的自定义历史轮次列表。
    """
    seed = SEEDS[seed_key]
    questions = seed["questions"]
    question = questions[question_index % len(questions)]
    messages = [{"role": "system", "content": seed["system"]}]

    turns = history_pool if history_pool is not None else HISTORY_POOLS[0]
    for i in range(min(num_history_turns, len(turns))):
        messages.append(turns[i])

    messages.append({"role": "user", "content": question})
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


def _count_tokens(messages: list[dict], tokenizer) -> int:
    """计算消息列表的 token 数。"""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return len(tokenizer.encode(text, add_special_tokens=False))


def generate_prompt(
    target_tokens: int,
    seed: str = "seed_qa",
    variant: int = 0,
    tokenizer=None,
) -> str:
    """生成指定 token 数的 prompt。

    variant 的组合维度：seed × question_index × history_pool_index，
    确保不同 variant 产生内容差异大的 prompt，最大限度减少 prefix cache 命中。

    Args:
        target_tokens: 目标 prompt token 数。
        seed: 种子名称，决定 system prompt 和 question 分类。
        variant: 变体编号。越大产生越不同的 prompt。
        tokenizer: 模型 tokenizer。
    """
    if seed not in SEEDS:
        raise ValueError(f"Unknown seed: {seed}. Must be one of {list(SEEDS.keys())}")
    if tokenizer is None:
        raise ValueError("tokenizer is required for prompt generation")

    # 用 variant 推导 question 索引和历史轮次池
    questions = SEEDS[seed]["questions"]
    n_questions = len(questions)
    n_pools = len(HISTORY_POOLS)

    # 组合：question 循环最快，history_pool 循环其次
    question_index = variant % n_questions
    pool_index = (variant // n_questions) % n_pools
    history_pool = HISTORY_POOLS[pool_index]

    # 寻找合适的历史轮次数
    best_messages = _build_messages(seed, question_index, 0, history_pool)

    for n_turns in range(1, len(history_pool) + 1):
        messages = _build_messages(seed, question_index, n_turns, history_pool)
        n_tokens = _count_tokens(messages, tokenizer)
        if n_tokens >= target_tokens:
            return _tokenize_and_truncate(messages, target_tokens, tokenizer)
        best_messages = messages

    return _tokenize_and_truncate(best_messages, target_tokens, tokenizer)


def generate_batch_prompts(
    num_requests: int,
    target_tokens: int,
    tokenizer=None,
) -> list[str]:
    """生成一批内容各异的 prompt，最大化多样性以减少 KV cache 命中。

    分配策略：三个维度（seed, pool, question）每次请求都尽量变化，
    确保相邻请求在 seed 和 pool 两个维度都不同，最大限度避免 prefix
    cache 命中。使用 (pool, seed) 交叉轮询，question 按计数器递增。

    总组合数 = 3 seeds × 3 pools × (5~6 questions) = 48 种。

    Args:
        num_requests: 请求总数。
        target_tokens: 每 prompt 的目标 token 数。
        tokenizer: 模型 tokenizer。
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required for prompt generation")

    seed_keys = list(SEEDS.keys())
    n_seeds = len(seed_keys)
    n_pools = len(HISTORY_POOLS)

    # 交叉轮询 (seed, pool) — seed 顺序循环，pool 按 seed+seed_idx 偏移
    # 确保相邻请求 seed 不同且 pool 也不同
    pool_seed_cycle = []
    for sk_idx, sk in enumerate(seed_keys):
        for p in range(n_pools):
            pool_seed_cycle.append((sk, (p + sk_idx) % n_pools))
    # 去重（理论上不需要，但以防万一）
    seen = set()
    unique_cycle = []
    for item in pool_seed_cycle:
        if item not in seen:
            seen.add(item)
            unique_cycle.append(item)
    pool_seed_cycle = unique_cycle
    n_ps = len(pool_seed_cycle)

    # 每个 (pool, seed) 组合维护独立的 question 计数器
    q_counters = {f"{p}_{sk}": 0 for p in range(n_pools) for sk in seed_keys}

    prompts = []
    for i in range(num_requests):
        sk, pool_idx = pool_seed_cycle[i % n_ps]
        key = f"{pool_idx}_{sk}"
        qi = q_counters[key]
        q_counters[key] = (qi + 1) % len(SEEDS[sk]["questions"])
        variant = qi + pool_idx * len(SEEDS[sk]["questions"])
        prompt = generate_prompt(
            target_tokens, seed=sk, variant=variant, tokenizer=tokenizer
        )
        prompts.append(prompt)
    return prompts
