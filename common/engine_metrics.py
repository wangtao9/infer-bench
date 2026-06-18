"""引擎内部指标采集（vLLM / SGLang /metrics 端点）。

从 Prometheus /metrics 端点解析引擎内部指标：
- KV cache 利用率
- 运行/等待请求数
- Prefix cache 命中率

这些指标比 pynvml 的 GPU 显存采样更有意义，因为 vLLM/SGLang
在启动时预分配全部显存，GPU 占用不变，而 KV cache 利用率
会随并发度动态变化。
"""

import logging
import re

import aiohttp

logger = logging.getLogger(__name__)


# ============================================================
# Prometheus 文本格式解析
# ============================================================


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    """解析 Prometheus /metrics 文本输出，提取 vllm: / sglang: 前缀的指标。

    Args:
        text: /metrics 端点返回的文本

    Returns:
        dict: metric_name → value（不含标签的简化版）
    """
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # 匹配形如: vllm:kv_cache_usage_perc 0.35
        # 或带标签: vllm:num_requests_waiting{reason="capacity"} 5
        match = re.match(r'^((?:vllam|sglang|vllm):\S+?)(?:\{[^}]*\})?\s+([\d.e+-]+|nan|inf)', line)
        if not match:
            continue
        name = match.group(1)
        value_str = match.group(2)
        try:
            value = float(value_str)
        except ValueError:
            continue
        result[name] = value
    return result


# ============================================================
# vLLM 指标键名映射
# ============================================================


VLLM_METRIC_KEYS = {
    "kv_cache_usage": "vllm:kv_cache_usage_perc",
    # 旧版 V0 架构使用 gp_cache_usage_perc
    "kv_cache_usage_v0": "vllm:gpu_cache_usage_perc",
    "num_running_reqs": "vllm:num_requests_running",
    "num_waiting_reqs": "vllm:num_requests_waiting",
    "prefix_cache_hits": "vllm:prefix_cache_hits",
    "prefix_cache_queries": "vllm:prefix_cache_queries",
    "num_preemptions": "vllm:num_preemptions",
}


# ============================================================
# SGLang 指标键名映射
# ============================================================


SGLANG_METRIC_KEYS = {
    "kv_cache_usage": "sglang:full_token_usage",
    "num_running_reqs": "sglang:num_running_reqs",
    "num_waiting_reqs": "sglang:num_queue_reqs",
    "cache_hit_rate": "sglang:cache_hit_rate",
    "gen_throughput": "sglang:gen_throughput",
}


# ============================================================
# 采集函数
# ============================================================


async def fetch_engine_metrics(
    base_url: str,
    engine: str,
) -> dict[str, float]:
    """从引擎 /metrics 端点抓取内部指标。

    Args:
        base_url: 引擎服务基础 URL（如 http://localhost:8000）
        engine: 引擎名称，"vllm" 或 "sglang"

    Returns:
        dict: 统一键名的指标值，如:
            {
                "kv_cache_usage": 0.35,
                "num_running_reqs": 4,
                "num_waiting_reqs": 0,
                ...
            }
    """
    metrics_url = f"{base_url}/metrics"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                metrics_url,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    logger.warning(
                        "/metrics endpoint returned %d for %s",
                        resp.status, engine,
                    )
                    return {}
                text = await resp.text()
    except (aiohttp.ClientError, Exception) as e:
        logger.warning("Failed to fetch /metrics from %s: %s", engine, e)
        return {}

    parsed = parse_prometheus_metrics(text)

    # 选择对应的指标键名映射
    if engine == "vllm":
        key_map = VLLM_METRIC_KEYS
    elif engine == "sglang":
        key_map = SGLANG_METRIC_KEYS
    else:
        return {}

    result = {}
    for friendly_name, prom_key in key_map.items():
        if prom_key in parsed:
            result[friendly_name] = parsed[prom_key]

    # vLLM: 如果 V1 指标没有，尝试 V0 的旧名称
    if engine == "vllm" and "kv_cache_usage" not in result:
        v0_key = VLLM_METRIC_KEYS.get("kv_cache_usage_v0", "")
        if v0_key and v0_key in parsed:
            result["kv_cache_usage"] = parsed[v0_key]

    # vLLM: 计算前缀缓存命中率
    if engine == "vllm" and "prefix_cache_hit_rate" not in result:
        hits = result.get("prefix_cache_hits", 0)
        queries = result.get("prefix_cache_queries", 0)
        if queries > 0:
            result["prefix_cache_hit_rate"] = hits / queries

    return result