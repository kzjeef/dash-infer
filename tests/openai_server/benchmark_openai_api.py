#!/usr/bin/env python3
"""
Simple OpenAI API benchmark for DashInfer server.

Reference style: vLLM/OpenAI-compatible concurrent request benchmarking.
"""

import argparse
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from openai import OpenAI


def _single_request(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    stream: bool,
) -> Tuple[float, int]:
    start = time.time()
    output_chars = 0
    if stream:
        chunks = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            stream=True,
        )
        for chunk in chunks:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                output_chars += len(text)
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            stream=False,
        )
        text = resp.choices[0].message.content or ""
        output_chars = len(text)
    latency = time.time() - start
    return latency, output_chars


def main() -> int:
    parser = argparse.ArgumentParser(description="Concurrent benchmark for OpenAI-compatible server")
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--api-key", default="EMPTY", type=str)
    parser.add_argument("--requests", default=64, type=int, help="Total number of requests")
    parser.add_argument("--concurrency", default=8, type=int, help="Concurrent request workers")
    parser.add_argument("--max-tokens", default=128, type=int)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=f"http://{args.host}:{args.port}/v1", timeout=300)
    prompt = (
        "You are a helpful assistant. Summarize the impact of efficient LLM serving in two paragraphs."
    )

    lock = threading.Lock()
    latencies: List[float] = []
    out_chars: List[int] = []

    begin = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(_single_request, client, args.model, prompt, args.max_tokens, args.stream)
            for _ in range(args.requests)
        ]
        for f in as_completed(futures):
            latency, chars = f.result()
            with lock:
                latencies.append(latency)
                out_chars.append(chars)

    elapsed = time.time() - begin
    qps = args.requests / elapsed if elapsed > 0 else 0.0
    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = statistics.quantiles(latencies, n=100)[94] if len(latencies) >= 20 else max(latencies, default=0.0)
    avg_chars = (sum(out_chars) / len(out_chars)) if out_chars else 0.0

    print("=== Benchmark Result ===")
    print(f"requests        : {args.requests}")
    print(f"concurrency     : {args.concurrency}")
    print(f"stream          : {args.stream}")
    print(f"elapsed_sec     : {elapsed:.3f}")
    print(f"qps             : {qps:.3f}")
    print(f"latency_p50_sec : {p50:.3f}")
    print(f"latency_p95_sec : {p95:.3f}")
    print(f"avg_out_chars   : {avg_chars:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
