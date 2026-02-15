#!/usr/bin/env python3
"""
Basic OpenAI-compatible API tests for DashInfer server.
"""

import argparse
import json
import sys
import time
from typing import List
from urllib import request as urlrequest
from urllib.error import HTTPError

from openai import OpenAI


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_list_models(client: OpenAI, expected_model: str) -> None:
    models = client.models.list()
    _assert(len(models.data) > 0, "models.list returned empty data")
    model_ids = [m.id for m in models.data]
    _assert(expected_model in model_ids, f"expected model '{expected_model}' not found, got: {model_ids}")


def test_chat_non_stream(client: OpenAI, model: str) -> None:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        temperature=0.0,
        max_tokens=64,
        stream=False,
    )
    _assert(len(resp.choices) > 0, "non-stream response has no choices")
    choice = resp.choices[0]
    _assert(choice.message is not None, "choice.message is None")
    _assert(choice.message.content is not None, "message.content is None")
    _assert(len(choice.message.content.strip()) > 0, "message.content is empty")
    _assert(resp.id.startswith("chatcmpl-"), "response id format invalid")
    _assert(resp.object == "chat.completion", "response object should be chat.completion")
    _assert(resp.model == model, f"response model mismatch: {resp.model} vs {model}")
    _assert(resp.usage is not None, "usage missing")
    _assert(resp.usage.prompt_tokens is not None, "usage.prompt_tokens missing")
    _assert(resp.usage.completion_tokens is not None, "usage.completion_tokens missing")
    _assert(resp.usage.total_tokens is not None, "usage.total_tokens missing")
    _assert(
        resp.usage.total_tokens == (resp.usage.prompt_tokens + resp.usage.completion_tokens),
        "usage.total_tokens != prompt_tokens + completion_tokens",
    )


def test_chat_stream(client: OpenAI, model: str) -> None:
    chunks = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Count: one, two, three."}],
        temperature=0.0,
        max_tokens=64,
        stream=True,
    )
    collected: List[str] = []
    finish_reasons: List[str] = []
    for chunk in chunks:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        text = getattr(delta, "content", None)
        if text:
            collected.append(text)
        if choice.finish_reason:
            finish_reasons.append(choice.finish_reason)
    merged = "".join(collected).strip()
    _assert(len(merged) > 0, "streaming response text is empty")
    _assert(len(finish_reasons) > 0, "streaming response missing finish_reason")


def test_stop_string_and_list(client: OpenAI, model: str) -> None:
    # stop as string: compatibility test (request accepted + valid response fields)
    resp_1 = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply in one line and include token END at the end."}],
        temperature=0.0,
        max_tokens=64,
        stop="END",
        stream=False,
    )
    text_1 = (resp_1.choices[0].message.content or "")
    _assert(len(text_1.strip()) > 0, "stop string response is empty")
    _assert(resp_1.choices[0].finish_reason in ("stop", "length"), "unexpected finish_reason for stop string")
    _assert(resp_1.usage is not None, "usage missing for stop string")

    # stop as list: compatibility test (request accepted + valid response fields)
    resp_2 = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Output with a marker STOPME soon."}],
        temperature=0.0,
        max_tokens=64,
        stop=["STOPME", "<|im_end|>"],
        stream=False,
    )
    text_2 = (resp_2.choices[0].message.content or "")
    _assert(len(text_2.strip()) > 0, "stop list response is empty")
    _assert(resp_2.choices[0].finish_reason in ("stop", "length"), "unexpected finish_reason for stop list")
    _assert(resp_2.usage is not None, "usage missing for stop list")


def test_stream_with_stop(client: OpenAI, model: str) -> None:
    chunks = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Output a short line and include marker ENDMARK."}],
        temperature=0.0,
        max_tokens=64,
        stop=["ENDMARK", "<|im_end|>"],
        stream=True,
    )
    collected: List[str] = []
    finish_reasons: List[str] = []
    for chunk in chunks:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        text = getattr(delta, "content", None)
        if text:
            collected.append(text)
        if choice.finish_reason:
            finish_reasons.append(choice.finish_reason)
    merged = "".join(collected).strip()
    _assert(len(merged) > 0, "stream+stop output is empty")
    _assert(len(finish_reasons) > 0, "stream+stop missing finish_reason")


def test_tool_call(client: OpenAI, model: str, strict_tool_call: bool) -> None:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Query weather for Beijing and use tool call."}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather by city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        tool_choice="auto",
        temperature=0.0,
        max_tokens=128,
    )
    choice = resp.choices[0]
    if strict_tool_call:
        _assert(choice.finish_reason == "tool_calls", f"strict mode expects tool_calls, got: {choice.finish_reason}")
        _assert(choice.message.tool_calls is not None and len(choice.message.tool_calls) > 0, "tool_calls missing")
    else:
        # Non-strict mode: permit model fallback to plain text.
        if choice.finish_reason == "tool_calls":
            _assert(choice.message.tool_calls is not None and len(choice.message.tool_calls) > 0, "tool_calls missing")


def test_legacy_function_call(client: OpenAI, model: str) -> None:
    # Legacy compatibility path (functions/function_call fields).
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Call function get_time with timezone Asia/Shanghai."}],
        functions=[
            {
                "name": "get_time",
                "description": "Get time by timezone",
                "parameters": {
                    "type": "object",
                    "properties": {"timezone": {"type": "string"}},
                    "required": ["timezone"],
                },
            }
        ],
        function_call={"name": "get_time"},
        temperature=0.0,
        max_tokens=128,
    )
    _assert(len(resp.choices) > 0, "legacy function call response has no choices")


def test_stream_tool_call(client: OpenAI, model: str, strict_tool_call: bool) -> None:
    chunks = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Use tool call to query weather in Beijing."}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather by city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        tool_choice="auto",
        temperature=0.0,
        max_tokens=128,
        stream=True,
    )
    seen_tool_calls = False
    finish_reasons: List[str] = []
    for chunk in chunks:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        tool_calls = getattr(delta, "tool_calls", None)
        if tool_calls:
            seen_tool_calls = True
        if choice.finish_reason:
            finish_reasons.append(choice.finish_reason)
    _assert(len(finish_reasons) > 0, "stream tool-call missing finish_reason")
    if strict_tool_call:
        _assert(seen_tool_calls, "strict mode expects stream tool_calls delta")
        _assert("tool_calls" in finish_reasons, f"strict mode expects tool_calls finish_reason, got {finish_reasons}")


def _post_raw(host: str, port: int, payload: dict, api_key: str = "EMPTY"):
    url = f"http://{host}:{port}/v1/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urlrequest.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _post_sse_lines(host: str, port: int, payload: dict, api_key: str = "EMPTY") -> List[str]:
    url = f"http://{host}:{port}/v1/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {api_key}",
        },
    )
    events: List[str] = []
    with urlrequest.urlopen(req, timeout=60) as resp:
        _assert(resp.status == 200, f"SSE request failed with status {resp.status}")
        _assert("text/event-stream" in resp.headers.get("Content-Type", ""), "unexpected SSE content-type")
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            events.append(data)
            if data == "[DONE]":
                break
    return events


def test_raw_sse_text(host: str, port: int, model: str, api_key: str) -> None:
    events = _post_sse_lines(
        host,
        port,
        {
            "model": model,
            "stream": True,
            "messages": [{"role": "user", "content": "Say hi briefly."}],
            "max_tokens": 64,
            "temperature": 0.0,
        },
        api_key=api_key,
    )
    _assert(len(events) >= 2, f"too few SSE events: {len(events)}")
    _assert(events[-1] == "[DONE]", "SSE stream missing [DONE]")

    json_events = [json.loads(x) for x in events if x != "[DONE]"]
    _assert(len(json_events) > 0, "SSE stream has no JSON events")
    first = json_events[0]
    _assert(first.get("object") == "chat.completion.chunk", "SSE chunk object mismatch")
    _assert("choices" in first and len(first["choices"]) > 0, "SSE chunk choices missing")

    finish_reasons = []
    saw_content_delta = False
    for ev in json_events:
        choice = ev["choices"][0]
        delta = choice.get("delta", {})
        if isinstance(delta, dict) and delta.get("content"):
            saw_content_delta = True
        if choice.get("finish_reason"):
            finish_reasons.append(choice["finish_reason"])

    _assert(saw_content_delta, "SSE text stream missing content delta")
    _assert(len(finish_reasons) > 0, "SSE text stream missing finish_reason")


def test_raw_sse_tool_call(host: str, port: int, model: str, api_key: str, strict_tool_call: bool) -> None:
    events = _post_sse_lines(
        host,
        port,
        {
            "model": model,
            "stream": True,
            "messages": [{"role": "user", "content": "Use tool to get weather for Beijing."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather by city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "max_tokens": 128,
            "temperature": 0.0,
        },
        api_key=api_key,
    )
    _assert(len(events) >= 2, f"too few SSE tool events: {len(events)}")
    _assert(events[-1] == "[DONE]", "SSE tool stream missing [DONE]")

    json_events = [json.loads(x) for x in events if x != "[DONE]"]
    saw_tool_delta = False
    finish_reasons = []
    for ev in json_events:
        choices = ev.get("choices", [])
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta", {})
        if isinstance(delta, dict) and delta.get("tool_calls"):
            saw_tool_delta = True
        if choice.get("finish_reason"):
            finish_reasons.append(choice["finish_reason"])

    _assert(len(finish_reasons) > 0, "SSE tool stream missing finish_reason")
    if strict_tool_call:
        _assert(saw_tool_delta, "strict mode expects SSE tool_calls delta")
        _assert("tool_calls" in finish_reasons, f"strict mode expects tool_calls finish_reason, got {finish_reasons}")


def test_error_shape(host: str, port: int, model: str) -> None:
    # case1: invalid n
    status_1, body_1 = _post_raw(
        host,
        port,
        {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "n": 2,
        },
    )
    _assert(status_1 == 400, f"invalid n should return 400, got {status_1}")
    _assert("error" in body_1, "error field missing in invalid n response")
    _assert("message" in body_1["error"], "error.message missing")
    _assert("type" in body_1["error"], "error.type missing")
    _assert("param" in body_1["error"], "error.param missing")
    _assert("code" in body_1["error"], "error.code missing")

    # case2: validation error (missing messages)
    status_2, body_2 = _post_raw(
        host,
        port,
        {
            "model": model,
        },
    )
    _assert(status_2 == 400, f"validation error should return 400, got {status_2}")
    _assert("error" in body_2, "error field missing in validation error response")
    _assert("message" in body_2["error"], "validation error.message missing")
    _assert("type" in body_2["error"], "validation error.type missing")
    _assert("param" in body_2["error"], "validation error.param missing")
    _assert("code" in body_2["error"], "validation error.code missing")


def main() -> int:
    parser = argparse.ArgumentParser(description="DashInfer OpenAI server smoke tests")
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--api-key", default="EMPTY", type=str)
    parser.add_argument("--strict-tool-call", action="store_true")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}/v1"
    client = OpenAI(api_key=args.api_key, base_url=base_url, timeout=120)

    start = time.time()
    test_list_models(client, args.model)
    print("[PASS] list_models")
    test_chat_non_stream(client, args.model)
    print("[PASS] chat_non_stream")
    test_stop_string_and_list(client, args.model)
    print("[PASS] stop_string_and_list")
    test_chat_stream(client, args.model)
    print("[PASS] chat_stream")
    test_stream_with_stop(client, args.model)
    print("[PASS] stream_with_stop")
    test_tool_call(client, args.model, args.strict_tool_call)
    print("[PASS] tool_call")
    test_stream_tool_call(client, args.model, args.strict_tool_call)
    print("[PASS] stream_tool_call")
    test_raw_sse_text(args.host, args.port, args.model, args.api_key)
    print("[PASS] raw_sse_text")
    test_raw_sse_tool_call(args.host, args.port, args.model, args.api_key, args.strict_tool_call)
    print("[PASS] raw_sse_tool_call")
    test_legacy_function_call(client, args.model)
    print("[PASS] legacy_function_call")
    test_error_shape(args.host, args.port, args.model)
    print("[PASS] error_shape")
    print(f"[DONE] all tests passed in {time.time() - start:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise
