"""OpenAI-compatible API 客户端（vLLM/SGLang 共用）。

使用 aiohttp 发送流式请求，采集 TTFT 和 TPS。
"""

import asyncio
import json
import logging
import time

import aiohttp

logger = logging.getLogger(__name__)

HEALTH_CHECK_TIMEOUT = 300
HEALTH_CHECK_INTERVAL = 5


async def wait_for_server(base_url: str, timeout: int = HEALTH_CHECK_TIMEOUT) -> bool:
    """轮询服务健康检查，等待引擎就绪。"""
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

    Returns:
        dict with: ttft_ms, total_tokens, total_time_s, tps, text
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
                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = data.get("choices", [])
                if choices and first_token_time is None:
                    content = choices[0].get("text", "")
                    if content:
                        first_token_time = time.monotonic()
                        ttft_ms = (first_token_time - start_time) * 1000.0

                if choices:
                    output_text += choices[0].get("text", "")

                usage = data.get("usage")
                if usage:
                    total_tokens = usage.get("completion_tokens", total_tokens)

    end_time = time.monotonic()
    total_time_s = end_time - start_time

    if total_tokens == 0 and output_text:
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

    Returns:
        dict with: mean_ttft_ms, total_tokens, concurrent_tps, total_time_s, results
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