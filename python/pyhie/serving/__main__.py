'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    __main__.py

 Unified entry point for DashInfer serving (LLM & VLM).

 Usage:
     python -m dashinfer.serving --model <hf_model_path>
     dashinfer_serve --model <hf_model_path>
'''
import argparse
import json
import logging
import os
import sys

import uvicorn

from .engine_handler import EngineConfig, LLMEngineHandler, detect_model_mode
from .server import app, set_engine_handler, set_api_keys


def parse_args():
    parser = argparse.ArgumentParser(
        description="DashInfer OpenAI-compatible API Server (LLM & VLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Serve a LLM (auto-detected)
  python -m dashinfer.serving --model Qwen/Qwen2.5-7B-Instruct

  # Serve a VLM (auto-detected, requires dashinfer[serving,vlm])
  python -m dashinfer.serving --model Qwen/Qwen2-VL-7B-Instruct

  # Force a specific mode
  python -m dashinfer.serving --model /path/to/model --mode llm
  python -m dashinfer.serving --model /path/to/model --mode vlm

  # Multi-GPU with tensor parallelism
  python -m dashinfer.serving \\
      --model Qwen/Qwen2.5-72B-Instruct \\
      --tensor-parallel 4 --max-batch 64

  # With API key authentication
  python -m dashinfer.serving \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --api-keys key1,key2
""",
    )

    # ── Mode ──
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "llm", "vlm"],
                        help="Serving mode. 'auto' detects from HF config (default: auto)")

    # ── Model ──
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model path or model ID")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Display name for the model (default: derived from model path)")
    parser.add_argument("--data-type", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Weight data type (default: bfloat16)")
    parser.add_argument("--trust-remote-code", action="store_true", default=True,
                        help="Trust remote code from HuggingFace (default: True)")

    # ── Device ──
    parser.add_argument("--device-type", type=str, default="CUDA",
                        choices=["CUDA", "CPU"],
                        help="Device type (default: CUDA)")
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="Number of GPUs for tensor parallelism (default: 1)")
    parser.add_argument("--device-ids", type=str, default=None,
                        help="Comma-separated device IDs (default: 0..tensor_parallel-1)")

    # ── Engine ──
    parser.add_argument("--max-batch", type=int, default=32,
                        help="Maximum batch size (default: 32)")
    parser.add_argument("--max-length", type=int, default=8192,
                        help="Maximum sequence length (default: 8192)")
    parser.add_argument("--enable-prefix-cache", action="store_true", default=True,
                        help="Enable prefix caching (default: True)")
    parser.add_argument("--no-prefix-cache", action="store_true", default=False,
                        help="Disable prefix caching")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Directory for converted model cache")

    # ── VLM-specific (only used when mode=vlm) ──
    parser.add_argument("--vision-engine", type=str, default="tensorrt",
                        choices=["tensorrt", "transformers"],
                        help="[VLM] Vision encoder backend (default: tensorrt)")
    parser.add_argument("--min-pixels", type=int, default=4 * 28 * 28,
                        help="[VLM] Minimum image pixels (default: 3136)")
    parser.add_argument("--max-pixels", type=int, default=16384 * 28 * 28,
                        help="[VLM] Maximum image pixels (default: 12845056)")
    parser.add_argument("--quant-type", type=str, default=None,
                        choices=["gptq", "gptq_weight_only", "a8w8", "a16w4", "a16w8", "fp8"],
                        help="[VLM] Quantization type")

    # ── Server ──
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind (default: 8000)")
    parser.add_argument("--api-keys", type=str, default=None,
                        help="Comma-separated API keys for authentication")

    # ── CORS ──
    parser.add_argument("--allowed-origins", type=json.loads, default=["*"],
                        help="Allowed CORS origins (default: [\"*\"])")
    parser.add_argument("--allowed-methods", type=json.loads, default=["*"],
                        help="Allowed CORS methods (default: [\"*\"])")
    parser.add_argument("--allowed-headers", type=json.loads, default=["*"],
                        help="Allowed CORS headers (default: [\"*\"])")

    # ── SSL ──
    parser.add_argument("--ssl-keyfile", type=str, default=None,
                        help="SSL key file path")
    parser.add_argument("--ssl-certfile", type=str, default=None,
                        help="SSL certificate file path")

    # ── Logging ──
    parser.add_argument("--log-level", type=str, default="info",
                        choices=["debug", "info", "warning", "error"],
                        help="Log level (default: info)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("dashinfer.serving")

    # ── Resolve mode ──
    if args.mode == "auto":
        log.info(f"Auto-detecting model type for: {args.model}")
        mode = detect_model_mode(args.model, trust_remote_code=args.trust_remote_code)
        log.info(f"Detected mode: {mode}")
    else:
        mode = args.mode

    # ── Parse device IDs ──
    if args.device_ids:
        device_ids = [int(x) for x in args.device_ids.split(",")]
    else:
        device_ids = list(range(args.tensor_parallel))

    # ── Build engine config ──
    engine_config = EngineConfig(
        model=args.model,
        model_name=args.model_name,
        data_type=args.data_type,
        device_type=args.device_type,
        device_ids=device_ids,
        engine_max_batch=args.max_batch,
        engine_max_length=args.max_length,
        enable_prefix_cache=args.enable_prefix_cache and not args.no_prefix_cache,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        # VLM-specific
        vision_engine=args.vision_engine,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        quant_type=args.quant_type,
    )

    # ── Create handler based on mode ──
    if mode == "vlm":
        try:
            from .vlm_engine_handler import VLMEngineHandler
        except ImportError:
            log.error(
                "VLM mode requires the dashinfer-vlm package.\n"
                "Install it with: pip install 'dashinfer[serving,vlm]'"
            )
            sys.exit(1)
        handler = VLMEngineHandler(engine_config)
    else:
        handler = LLMEngineHandler(engine_config)

    # ── Start engine ──
    handler.start()
    set_engine_handler(handler, mode=mode)

    # ── Set API keys ──
    if args.api_keys:
        set_api_keys(args.api_keys.split(","))

    # ── Add CORS middleware ──
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=True,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    # ── Print server info ──
    log.info(f"Mode: {mode.upper()}")
    log.info(f"Model: {handler.model_name}")
    log.info(f"Device: {args.device_type}, IDs: {device_ids}")
    log.info(f"Max batch: {args.max_batch}, Max length: {args.max_length}")
    log.info(f"Server: http://{args.host}:{args.port}")
    log.info(f"API docs: http://{args.host}:{args.port}/docs")

    # ── Run server ──
    uvicorn_kwargs = dict(
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    if args.ssl_keyfile and args.ssl_certfile:
        uvicorn_kwargs["ssl_keyfile"] = args.ssl_keyfile
        uvicorn_kwargs["ssl_certfile"] = args.ssl_certfile

    try:
        uvicorn.run(app, **uvicorn_kwargs)
    finally:
        handler.stop()


if __name__ == "__main__":
    main()
