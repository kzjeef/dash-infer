'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    server.py

 Lightweight OpenAI-compatible API server for DashInfer.

 Usage:
     python -m dashinfer.serving --model <hf_model_path> [--port 8000]
     dashinfer_serve --model <hf_model_path> [--port 8000]
'''
import asyncio
import logging
import time
from typing import Optional, List

import fastapi
from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer

from .protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    ErrorResponse,
    ModelCard,
    ModelList,
    UsageInfo,
)
from .engine_handler import BaseEngineHandler

logger = logging.getLogger("dashinfer.serving")


# ──────────────────────── Global State ─────────────────────────

_engine_handler: Optional[BaseEngineHandler] = None
_api_keys: Optional[List[str]] = None
_serving_mode: str = "llm"  # "llm" or "vlm"

app = fastapi.FastAPI(
    title="DashInfer API Server",
    description="OpenAI-compatible API server powered by DashInfer",
    version="1.0.0",
)

get_bearer_token = HTTPBearer(auto_error=False)


def set_engine_handler(handler: BaseEngineHandler, mode: str = "llm"):
    global _engine_handler, _serving_mode
    _engine_handler = handler
    _serving_mode = mode


def set_api_keys(keys: Optional[List[str]]):
    global _api_keys
    _api_keys = keys


# ──────────────────────── Auth ─────────────────────────────────

async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
) -> Optional[str]:
    if _api_keys:
        if auth is None or auth.credentials not in _api_keys:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "Invalid API key",
                        "type": "invalid_request_error",
                        "code": "invalid_api_key",
                    }
                },
            )
        return auth.credentials
    return None


# ──────────────────────── Routes ───────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "mode": _serving_mode}


@app.get("/v1/models", dependencies=[Depends(check_api_key)])
async def list_models():
    return ModelList(data=[
        ModelCard(id=_engine_handler.model_name)
    ])


@app.post("/v1/chat/completions", dependencies=[Depends(check_api_key)])
async def create_chat_completion(request: ChatCompletionRequest):
    error = _validate_request(request)
    if error:
        return error

    if request.stream:
        return StreamingResponse(
            _stream_chat_completion(request),
            media_type="text/event-stream",
        )
    else:
        return await asyncio.to_thread(_sync_chat_completion, request)


# ──────────────────────── Validation ───────────────────────────

def _validate_request(request: ChatCompletionRequest) -> Optional[JSONResponse]:
    if request.max_tokens is not None and request.max_tokens <= 0:
        return _error_response(400, f"max_tokens must be > 0, got {request.max_tokens}")
    if request.temperature is not None and not (0 <= request.temperature <= 2):
        return _error_response(400, f"temperature must be in [0, 2], got {request.temperature}")
    if request.top_p is not None and not (0 <= request.top_p <= 1):
        return _error_response(400, f"top_p must be in [0, 1], got {request.top_p}")
    return None


def _error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, code=status_code).model_dump(),
        status_code=status_code,
    )


# ──────────────────────── Generation ───────────────────────────

def _build_gen_kwargs(request: ChatCompletionRequest, input_ids: list) -> dict:
    """Extract generation kwargs from the request."""
    max_tokens = request.max_tokens or 2048
    stop_words_ids = None
    if request.stop:
        tokenizer = _engine_handler.tokenizer
        stops = [request.stop] if isinstance(request.stop, str) else request.stop
        stop_words_ids = []
        for s in stops:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if ids:
                stop_words_ids.append(ids)

    return dict(
        max_tokens=max_tokens,
        temperature=request.temperature or 0.7,
        top_p=request.top_p or 1.0,
        top_k=request.top_k or 0,
        repetition_penalty=request.repetition_penalty or 1.0,
        presence_penalty=request.presence_penalty or 0.0,
        frequency_penalty=request.frequency_penalty or 0.0,
        stop_words_ids=stop_words_ids,
        seed=request.seed,
    )


def _prepare_generate(request: ChatCompletionRequest):
    """
    Tokenize and prepare generation args.
    For VLM, also builds the VLRequest with image data.
    Returns (input_ids, gen_kwargs) where gen_kwargs may include _vl_request for VLM.
    """
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    input_ids = _engine_handler.tokenize_chat(messages)
    gen_kwargs = _build_gen_kwargs(request, input_ids)

    # For VLM mode, build the VLRequest and pass it through gen_kwargs
    if _serving_mode == "vlm":
        from .vlm_engine_handler import VLMEngineHandler
        if isinstance(_engine_handler, VLMEngineHandler):
            vl_request = _engine_handler.build_vl_gen_config(
                input_ids,
                max_tokens=gen_kwargs["max_tokens"],
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                top_k=gen_kwargs["top_k"],
                presence_penalty=gen_kwargs["presence_penalty"],
                frequency_penalty=gen_kwargs["frequency_penalty"],
                seed=gen_kwargs["seed"],
            )
            gen_kwargs["_vl_request"] = vl_request

    return input_ids, gen_kwargs


def _sync_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Non-streaming chat completion (runs in a thread)."""
    input_ids, gen_kwargs = _prepare_generate(request)

    all_generated_ids = []
    finish_reason = "stop"

    for new_ids, is_finished in _engine_handler.generate(input_ids, **gen_kwargs):
        all_generated_ids.extend(new_ids)

    # Check if we hit the max token limit
    max_tokens = gen_kwargs["max_tokens"]
    if len(all_generated_ids) >= max_tokens:
        finish_reason = "length"

    output_text = _engine_handler.tokenizer.decode(
        all_generated_ids,
        skip_special_tokens=True,
    )

    usage = UsageInfo(
        prompt_tokens=len(input_ids),
        completion_tokens=len(all_generated_ids),
        total_tokens=len(input_ids) + len(all_generated_ids),
    )

    return ChatCompletionResponse(
        model=_engine_handler.model_name,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=output_text),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


async def _stream_chat_completion(request: ChatCompletionRequest):
    """Streaming chat completion using SSE."""
    input_ids, gen_kwargs = _prepare_generate(request)

    response_id = f"chatcmpl-{int(time.time())}"
    tokenizer = _engine_handler.tokenizer

    # Send initial role chunk
    initial_chunk = ChatCompletionStreamResponse(
        id=response_id,
        model=_engine_handler.model_name,
        choices=[
            ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(role="assistant"),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {initial_chunk.model_dump_json(exclude_unset=True)}\n\n"

    # Stream generation in a thread
    loop = asyncio.get_event_loop()

    def _generate_sync():
        results = []
        for new_ids, is_finished in _engine_handler.generate(input_ids, **gen_kwargs):
            results.append((new_ids, is_finished))
        return results

    gen_results = await loop.run_in_executor(None, _generate_sync)

    all_generated_ids = []
    for new_ids, is_finished in gen_results:
        all_generated_ids.extend(new_ids)

        text = tokenizer.decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not text:
            continue

        finish = None
        if is_finished:
            max_tokens = gen_kwargs["max_tokens"]
            finish = "length" if len(all_generated_ids) >= max_tokens else "stop"

        chunk = ChatCompletionStreamResponse(
            id=response_id,
            model=_engine_handler.model_name,
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(content=text),
                    finish_reason=finish,
                )
            ],
        )
        yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

    # Send final [DONE]
    yield "data: [DONE]\n\n"
