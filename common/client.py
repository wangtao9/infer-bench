"""OpenAI-compatible API 客户端（vLLM/SGLang 共用）。

参照 vLLM bench serve 的实现：
- 所有延迟指标从客户端 time.monotonic() 时间戳计算
- 逐 SSE chunk 记录时间戳，计算 TTFT / ITL / TPOT / E2EL
- 使用 StreamedResponseHandler 正确处理分块 SSE 流
- 不依赖 /metrics 端点采集延迟/吞吐指标
"""

import asyncio
import codecs
import json
import logging
import time

import aiohttp

logger = logging.getLogger(__name__)

HEALTH_CHECK_TIMEOUT = 300
HEALTH_CHECK_INTERVAL = 5


# ============================================================
# SSE 流解析器（参照 vLLM StreamedResponseHandler）
# ============================================================


class StreamedResponseHandler:
    """正确处理分块 TCP 字节流中的 SSE 消息。

    TCP chunk 可能将一个 SSE 消息拆到多个 chunk，也可能将
    多个消息合并到一个 chunk。此处理器缓冲输入，按 \\n\\n
    分割完整的 SSE 消息。
    """

    def __init__(self):
        self.buffer = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")()

    def add_chunk(self, chunk_bytes: bytes) -> list[str]:
        """将收到的字节块追加到缓冲区，返回已完成的 SSE 消息列表。"""
        chunk_str = self._decoder.decode(chunk_bytes)
        self.buffer += chunk_str
        messages = []

        # 按 \n\n 分割（SSE 标准消息分隔符）
        while "\n\n" in self.buffer:
            message, self.buffer = self.buffer.split("\n\n", 1)
            message = message.strip()
            if message:
                messages.append(message)

        # 处理缓冲区中可能残留的完整消息（无尾随 \n\n）
        if self.buffer.startswith("data: "):
            content = self.buffer.removeprefix("data: ").strip()
            if content == "[DONE]":
                messages.append(self.buffer.strip())
                self.buffer = ""
            elif content:
                try:
                    json.loads(content)
                    messages.append(self.buffer.strip())
                    self.buffer = ""
                except json.JSONDecodeError:
                    pass  # 不完整的 JSON，等后续 chunk

        return messages


# ============================================================
# 健康检查
# ============================================================


async def wait_for_server(base_url: str, timeout: int = HEALTH_CHECK_TIMEOUT) -> bool:
    """轮询服务健康检查，等待引擎就绪。"""
    health_url = f"{base_url}/health"
    start = time.monotonic()
    async with aiohttp.ClientSession() as session:
        while time.monotonic() - start < timeout:
            try:
                async with session.get(
                    health_url, timeout=aiohttp.ClientTimeout(total=3),
                ) as resp:
                    if resp.status == 200:
                        logger.info("Server ready: %s", base_url)
                        return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
    logger.error("Server not ready after %ds: %s", timeout, base_url)
    return False


# ============================================================
# 单请求流式采集（参照 vLLM bench serve 的时间戳方法）
# ============================================================


async def stream_request(
    base_url: str,
    prompt: str,
    model: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> dict:
    """发送单个流式请求，逐 chunk 记录时间戳。

    采集指标（全部从客户端时间戳计算，参照 vLLM bench serve）：
    - TTFT: 首 token 延迟 (ms)
    - ITL: 逐 token 间隔延迟列表 (ms)
    - TPOT: 每个 output token 的平均时间 (ms) = (E2EL - TTFT) / (output_tokens - 1)
    - E2EL: 端到端延迟 (ms) = 最后一个 chunk 时间 − 请求发出时间
    - TPS: 吞吐 (tokens/s) = total_tokens / E2EL_s

    Returns:
        dict with:
            ttft_ms, itl_ms (list), tpot_ms, e2el_ms,
            total_tokens, total_time_s, tps, text
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

    first_token_time = None
    start_time = time.monotonic()
    output_text = ""
    total_tokens = 0
    itl_ms: list[float] = []           # inter-token latency 列表
    most_recent_timestamp = start_time  # 上一个 chunk 的时间戳

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            resp.raise_for_status()
            handler = StreamedResponseHandler()

            async for chunk_bytes in resp.content.iter_any():
                messages = handler.add_chunk(chunk_bytes)

                for message in messages:
                    # 跳过 SSE 注释行
                    if message.startswith(":"):
                        continue

                    if not message.startswith("data: "):
                        continue

                    data_str = message.removeprefix("data: ").strip()
                    if data_str == "[DONE]":
                        continue

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    timestamp = time.monotonic()
                    choices = data.get("choices", [])

                    if choices:
                        content = choices[0].get("text", "")
                        if content:
                            if first_token_time is None:
                                # 首 token → TTFT
                                first_token_time = timestamp
                            else:
                                # 后续 token → ITL
                                itl_ms.append((timestamp - most_recent_timestamp) * 1000.0)
                            most_recent_timestamp = timestamp
                            output_text += content

                    # 采集 usage（含 completion_tokens）
                    usage = data.get("usage")
                    if usage:
                        total_tokens = usage.get("completion_tokens", total_tokens)

    end_time = time.monotonic()
    e2el_s = end_time - start_time

    ttft_ms = (first_token_time - start_time) * 1000.0 if first_token_time else 0.0
    e2el_ms = e2el_s * 1000.0

    # TPOT = (E2EL - TTFT) / (output_tokens - 1)，参照 vLLM bench serve
    if total_tokens > 1 and first_token_time:
        tpot_ms = (end_time - first_token_time) * 1000.0 / (total_tokens - 1)
    else:
        tpot_ms = e2el_ms if total_tokens == 1 else 0.0

    # fallback: 如果流中没有 usage 信息，粗估 token 数
    if total_tokens == 0 and output_text:
        total_tokens = max(1, len(output_text.split()))

    tps = total_tokens / e2el_s if e2el_s > 0 else 0.0

    return {
        "ttft_ms": ttft_ms,
        "itl_ms": itl_ms,
        "tpot_ms": tpot_ms,
        "e2el_ms": e2el_ms,
        "total_tokens": total_tokens,
        "total_time_s": e2el_s,
        "tps": tps,
        "text": output_text,
    }


# ============================================================
# 并发流式采集
# ============================================================


async def concurrent_stream_requests(
    base_url: str,
    prompts: list[str],
    model: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> dict:
    """并发发送多个流式请求，采集并发指标。

    Returns:
        dict with:
            mean_ttft_ms, mean_tpot_ms, mean_itl_ms,
            total_tokens, concurrent_tps, total_time_s, results
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
    tpots = [r["tpot_ms"] for r in results]
    all_itls = [itl for r in results for itl in r["itl_ms"]]
    total_tokens = sum(r["total_tokens"] for r in results)

    mean_ttft_ms = sum(ttfts) / len(ttfts) if ttfts else 0.0
    mean_tpot_ms = sum(tpots) / len(tpots) if tpots else 0.0
    mean_itl_ms = sum(all_itls) / len(all_itls) if all_itls else 0.0
    concurrent_tps = total_tokens / total_time_s if total_time_s > 0 else 0.0

    return {
        "mean_ttft_ms": mean_ttft_ms,
        "mean_tpot_ms": mean_tpot_ms,
        "mean_itl_ms": mean_itl_ms,
        "all_ttfts_ms": ttfts,
        "all_itls_ms": all_itls,
        "all_tpots_ms": tpots,
        "all_e2els_ms": [r["e2el_ms"] for r in results],
        "total_tokens": total_tokens,
        "concurrent_tps": concurrent_tps,
        "total_time_s": total_time_s,
        "results": results,
    }