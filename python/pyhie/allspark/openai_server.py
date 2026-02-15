'''
Copyright (c) Alibaba, Inc. and its affiliates.
 Copyright (c) 2025-2026 DashInfer Team.
@file    openai_server.py
'''
from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple, Union

try:
    from fastapi import Depends, FastAPI, HTTPException
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import StreamingResponse, JSONResponse
    from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError as e:
    raise ImportError(
        "OpenAI server requires optional dependencies. "
        "Please install server extra: pip install 'dashinfer[server]' "
        "(or pip install fastapi uvicorn pydantic)"
    ) from e

from dashinfer import allspark
from dashinfer.allspark.engine import TargetDevice
from dashinfer.allspark._allspark import GenerateRequestStatus


class FunctionSpec(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ToolSpec(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionSpec


class ToolChoiceFunction(BaseModel):
    name: str


class ToolChoiceObject(BaseModel):
    type: Literal["function"] = "function"
    function: ToolChoiceFunction


ToolChoice = Union[Literal["none", "auto"], ToolChoiceObject]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 256
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    n: Optional[int] = 1
    tools: Optional[List[ToolSpec]] = None
    tool_choice: Optional[ToolChoice] = "auto"
    # Legacy OpenAI fields (function calling before tools)
    functions: Optional[List[FunctionSpec]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None
    user: Optional[str] = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "dashinfer"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]


def _pydantic_dump(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


@dataclass
class _ToolCall:
    name: str
    arguments: str


class _ServerRuntime:
    def __init__(
        self,
        model_path: str,
        served_model_name: str,
        output_dir: str,
        device: str,
        device_list: List[int],
        max_batch: int,
        max_length: int,
        data_type: str,
        enable_quant: bool,
        weight_only_quant: bool,
        api_keys: Optional[List[str]],
    ) -> None:
        self.model_path = model_path
        self.served_model_name = served_model_name
        self.output_dir = output_dir
        self.device = device
        self.device_list = device_list
        self.max_batch = max_batch
        self.max_length = max_length
        self.data_type = data_type
        self.enable_quant = enable_quant
        self.weight_only_quant = weight_only_quant
        self.api_keys = api_keys or []

        self.engine = allspark.Engine()
        self.model_loader = allspark.HuggingFaceModel(
            model_path,
            served_model_name,
            in_memory_serialize=False,
            user_set_data_type=data_type,
            trust_remote_code=True,
        )
        self.runtime_cfg = None
        self._tokenizer = None

    def init(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        self.model_loader.load_model().serialize_to_path(
            self.engine,
            self.output_dir,
            enable_quant=self.enable_quant,
            weight_only_quant=self.weight_only_quant,
            skip_if_exists=True,
        ).free_model()

        target_device = TargetDevice.CUDA if self.device == "cuda" else TargetDevice.CPU
        ids = self.device_list if self.device == "cuda" else []
        runtime_cfg_builder = self.model_loader.create_reference_runtime_config_builder(
            self.served_model_name,
            target_device,
            ids,
            max_batch=self.max_batch,
        )
        runtime_cfg_builder.max_length(self.max_length)
        self.runtime_cfg = runtime_cfg_builder.build()
        self.engine.install_model(self.runtime_cfg)
        self.engine.start_model(self.served_model_name)
        self._tokenizer = self.model_loader.init_tokenizer().get_tokenizer()

    def tokenizer(self):
        return self._tokenizer


def _stringify_content(content: Optional[Union[str, List[Dict[str, Any]]]]) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    chunks: List[str] = []
    for item in content:
        item_type = item.get("type")
        if item_type == "text":
            chunks.append(str(item.get("text", "")))
        else:
            chunks.append(json.dumps(item, ensure_ascii=False))
    return "\n".join(chunks)


def _normalize_tools(req: ChatCompletionRequest) -> Tuple[List[ToolSpec], Optional[str]]:
    tools: List[ToolSpec] = list(req.tools or [])
    if not tools and req.functions:
        tools = [ToolSpec(function=f) for f in req.functions]

    forced_name: Optional[str] = None
    if isinstance(req.tool_choice, ToolChoiceObject):
        forced_name = req.tool_choice.function.name
    if isinstance(req.function_call, dict):
        forced_name = str(req.function_call.get("name") or "")
    elif req.function_call == "none":
        return [], None

    if req.tool_choice == "none":
        return [], None
    return tools, forced_name or None


def _build_chatml_prompt(messages: List[ChatMessage], tools: List[ToolSpec], forced_tool: Optional[str]) -> str:
    prompt_chunks: List[str] = []

    system_prefix = "You are a helpful assistant."
    if tools:
        tool_lines: List[str] = []
        for t in tools:
            tool_lines.append(
                json.dumps(
                    {
                        "name": t.function.name,
                        "description": t.function.description,
                        "parameters": t.function.parameters,
                    },
                    ensure_ascii=False,
                )
            )
        instruction = (
            "You can call tools. Available tools (JSON schema):\n"
            + "\n".join(tool_lines)
            + "\n\nIf a tool call is needed, output STRICT JSON only in one of these formats:\n"
            + '{"tool_calls":[{"name":"<tool_name>","arguments":{...}}]}\n'
            + '{"name":"<tool_name>","arguments":{...}}\n'
        )
        if forced_tool:
            instruction += f"\nYou must call tool: {forced_tool}\n"
        system_prefix = f"{system_prefix}\n\n{instruction}"

    has_system = any(msg.role == "system" for msg in messages)
    if not has_system:
        prompt_chunks.append(f"<|im_start|>system\n{system_prefix}<|im_end|>\n")

    for msg in messages:
        content = _stringify_content(msg.content)
        if msg.role == "system":
            merged = f"{system_prefix}\n{content}" if tools else content
            prompt_chunks.append(f"<|im_start|>system\n{merged}<|im_end|>\n")
        elif msg.role == "user":
            prompt_chunks.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        elif msg.role == "assistant":
            if msg.tool_calls:
                content = json.dumps({"tool_calls": msg.tool_calls}, ensure_ascii=False)
            elif msg.function_call:
                content = json.dumps(msg.function_call, ensure_ascii=False)
            prompt_chunks.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
        elif msg.role == "tool":
            tool_name = msg.name or "tool"
            tool_id = msg.tool_call_id or ""
            prompt_chunks.append(
                f"<|im_start|>tool\nname={tool_name};tool_call_id={tool_id}\n{content}<|im_end|>\n"
            )

    prompt_chunks.append("<|im_start|>assistant\n")
    return "".join(prompt_chunks)


def _extract_json_candidate(text: str) -> Optional[str]:
    candidate = text.strip()
    if not candidate:
        return None
    if candidate.startswith("{") and candidate.endswith("}"):
        return candidate
    match = re.search(r"\{[\s\S]*\}", candidate)
    if match:
        return match.group(0)
    return None


def _parse_tool_calls(
    output_text: str,
    tools: List[ToolSpec],
    forced_tool_name: Optional[str],
) -> List[_ToolCall]:
    if not tools:
        return []
    allowed_names = {t.function.name for t in tools}
    json_blob = _extract_json_candidate(output_text)
    if not json_blob:
        return []
    try:
        payload = json.loads(json_blob)
    except json.JSONDecodeError:
        return []

    calls: List[_ToolCall] = []
    if isinstance(payload, dict) and isinstance(payload.get("tool_calls"), list):
        for item in payload["tool_calls"]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            args = item.get("arguments", {})
            if isinstance(args, str):
                arguments = args
            else:
                arguments = json.dumps(args, ensure_ascii=False)
            calls.append(_ToolCall(name=name, arguments=arguments))
    elif isinstance(payload, dict) and "name" in payload and "arguments" in payload:
        args = payload.get("arguments", {})
        arguments = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        calls.append(_ToolCall(name=str(payload.get("name", "")), arguments=arguments))
    elif isinstance(payload, dict) and isinstance(payload.get("function_call"), dict):
        fc = payload["function_call"]
        args = fc.get("arguments", {})
        arguments = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        calls.append(_ToolCall(name=str(fc.get("name", "")), arguments=arguments))

    validated: List[_ToolCall] = []
    for c in calls:
        if not c.name:
            continue
        if c.name not in allowed_names:
            continue
        validated.append(c)

    if forced_tool_name:
        validated = [c for c in validated if c.name == forced_tool_name]
    return validated


def _make_usage(prompt_tokens: int, completion_tokens: int) -> Dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def create_openai_server_app(runtime: _ServerRuntime) -> FastAPI:
    app = FastAPI(title="DashInfer OpenAI-Compatible Server")
    runtime.init()

    bearer = HTTPBearer(auto_error=False)

    def check_api_key(auth: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> Optional[str]:
        if not runtime.api_keys:
            return None
        token = auth.credentials if auth else None
        if token not in runtime.api_keys:
            raise HTTPException(status_code=401, detail="invalid api key")
        return token

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_, exc: HTTPException):
        message = exc.detail
        if isinstance(message, dict):
            # keep backward compatibility for callers that already pass dict detail
            message = json.dumps(message, ensure_ascii=False)
        if not isinstance(message, str):
            message = str(message)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_, exc: RequestValidationError):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "validation_error",
                }
            },
        )

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models", dependencies=[Depends(check_api_key)])
    def list_models() -> Dict[str, Any]:
        return _pydantic_dump(ModelList(data=[ModelCard(id=runtime.served_model_name)]))

    @app.post("/v1/chat/completions", dependencies=[Depends(check_api_key)])
    def chat_completions(req: ChatCompletionRequest):
        if req.n not in (None, 1):
            raise HTTPException(status_code=400, detail={"error": {"message": "only n=1 is supported"}})

        tools, forced_tool_name = _normalize_tools(req)
        prompt = _build_chatml_prompt(req.messages, tools, forced_tool_name)
        tokenizer = runtime.tokenizer()
        prompt_token_ids = tokenizer.encode(prompt)
        prompt_tokens = len(prompt_token_ids)

        gen_cfg = runtime.model_loader.create_reference_generation_config_builder(runtime.runtime_cfg)
        max_tokens = int(req.max_tokens or 256)
        max_total_len = min(runtime.max_length, prompt_tokens + max_tokens)
        gen_cfg.update(
            {
                "temperature": req.temperature if req.temperature is not None else 0.7,
                "top_p": req.top_p if req.top_p is not None else 1.0,
                "max_length": max_total_len,
            }
        )
        if req.stop:
            stop_list = [req.stop] if isinstance(req.stop, str) else req.stop
            stop_ids = []
            for s in stop_list:
                ids = tokenizer.encode(s, add_special_tokens=False)
                if ids:
                    stop_ids.append(ids)
            if stop_ids:
                gen_cfg.update({"stop_words_ids": stop_ids})

        status, handle, queue = runtime.engine.start_request_text(
            runtime.served_model_name, runtime.model_loader, prompt, gen_cfg
        )
        if hasattr(status, "name") and status.name != "ALLSPARK_SUCCESS":
            raise HTTPException(status_code=500, detail={"error": {"message": f"start request failed: {status}"}})

        if req.stream:
            def stream_generator() -> Generator[str, None, None]:
                request_id = f"chatcmpl-{uuid.uuid4().hex}"
                created = int(time.time())
                completion_ids: List[int] = []
                # If tool definitions are present, buffer first and classify as
                # normal text vs tool call at the end for OpenAI-compatible shape.
                emit_content_incrementally = not tools
                first_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": runtime.served_model_name,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"
                try:
                    gen_status = queue.GenerateStatus()
                    while gen_status in [
                        GenerateRequestStatus.Init,
                        GenerateRequestStatus.Generating,
                        GenerateRequestStatus.ContextFinished,
                    ]:
                        elements = queue.Get()
                        if elements is not None and elements.ids_from_generate:
                            ids = list(elements.ids_from_generate)
                            completion_ids.extend(ids)
                            if emit_content_incrementally:
                                text = tokenizer.decode(
                                    ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                                )
                                if not text:
                                    gen_status = queue.GenerateStatus()
                                    continue
                                chunk = {
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": runtime.served_model_name,
                                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                                }
                                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        gen_status = queue.GenerateStatus()
                        if gen_status in [
                            GenerateRequestStatus.GenerateFinished,
                            GenerateRequestStatus.GenerateInterrupted,
                        ]:
                            break

                    text_all = tokenizer.decode(
                        completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    tool_calls = _parse_tool_calls(text_all, tools, forced_tool_name)
                    if tool_calls:
                        tc_payload = []
                        for tc in tool_calls:
                            tc_payload.append(
                                {
                                    "id": f"call_{uuid.uuid4().hex[:12]}",
                                    "type": "function",
                                    "function": {"name": tc.name, "arguments": tc.arguments},
                                }
                            )
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": runtime.served_model_name,
                            "choices": [{"index": 0, "delta": {"tool_calls": tc_payload}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        final_reason = "tool_calls"
                    else:
                        if not emit_content_incrementally and text_all:
                            chunk = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": runtime.served_model_name,
                                "choices": [{"index": 0, "delta": {"content": text_all}, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        final_reason = "stop"

                    final_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": runtime.served_model_name,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": final_reason}],
                    }
                    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    runtime.engine.release_request(runtime.served_model_name, handle)

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        completion_ids: List[int] = []
        try:
            gen_status = queue.GenerateStatus()
            while gen_status in [
                GenerateRequestStatus.Init,
                GenerateRequestStatus.Generating,
                GenerateRequestStatus.ContextFinished,
            ]:
                elements = queue.Get()
                if elements is not None and elements.ids_from_generate:
                    completion_ids.extend(elements.ids_from_generate)
                    if len(completion_ids) >= max_tokens:
                        break
                gen_status = queue.GenerateStatus()
                if gen_status in [
                    GenerateRequestStatus.GenerateFinished,
                    GenerateRequestStatus.GenerateInterrupted,
                ]:
                    break
        finally:
            runtime.engine.release_request(runtime.served_model_name, handle)

        output_text = tokenizer.decode(completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        tool_calls = _parse_tool_calls(output_text, tools, forced_tool_name)

        if tool_calls:
            tc_payload = []
            for tc in tool_calls:
                tc_payload.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                )
            message: Dict[str, Any] = {"role": "assistant", "content": None, "tool_calls": tc_payload}
            # Keep legacy field for old function-calling clients.
            message["function_call"] = tc_payload[0]["function"]
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": output_text}
            finish_reason = "stop"

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": runtime.served_model_name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": _make_usage(prompt_tokens=prompt_tokens, completion_tokens=len(completion_ids)),
        }
        return response

    return app


def _parse_device_list(value: str) -> List[int]:
    value = value.strip()
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DashInfer OpenAI-compatible API server")
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--model-path", required=True, type=str, help="Local model path or ModelScope model id")
    parser.add_argument("--served-model-name", default=None, type=str, help="Model id exposed by /v1/models")
    parser.add_argument("--use-modelscope", action="store_true", help="Download model from ModelScope")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--device-list", default="0", type=str, help="Comma-separated CUDA device ids")
    parser.add_argument("--max-batch", default=8, type=int)
    parser.add_argument("--max-length", default=4096, type=int)
    parser.add_argument("--data-type", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--enable-quant", action="store_true")
    parser.add_argument("--weight-only-quant", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=os.path.expanduser("~/.cache/dashinfer_server_models"),
        type=str,
        help="Directory for serialized .asgraph/.asparam files",
    )
    parser.add_argument(
        "--api-keys",
        default="",
        type=str,
        help="Comma-separated API keys. Empty means no authentication.",
    )
    return parser


def _resolve_model_path(model_path: str, use_modelscope: bool) -> str:
    if use_modelscope:
        import modelscope

        return modelscope.snapshot_download(model_path)
    return model_path


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    resolved_path = _resolve_model_path(args.model_path, args.use_modelscope)
    if not os.path.exists(resolved_path):
        raise ValueError(f"model path does not exist: {resolved_path}")

    served_name = args.served_model_name
    if not served_name:
        served_name = os.path.basename(resolved_path.rstrip("/")) or "dashinfer-model"
        served_name = served_name.replace("/", "_")

    runtime = _ServerRuntime(
        model_path=resolved_path,
        served_model_name=served_name,
        output_dir=args.output_dir,
        device=args.device,
        device_list=_parse_device_list(args.device_list) if args.device == "cuda" else [],
        max_batch=args.max_batch,
        max_length=args.max_length,
        data_type=args.data_type,
        enable_quant=args.enable_quant,
        weight_only_quant=args.weight_only_quant,
        api_keys=[k.strip() for k in args.api_keys.split(",") if k.strip()],
    )
    app = create_openai_server_app(runtime)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
