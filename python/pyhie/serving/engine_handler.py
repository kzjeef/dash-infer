'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    engine_handler.py

 Wraps the DashInfer (AllSpark) engine for HTTP serving.
'''
import os
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Iterator, Tuple

import torch
import torch.utils.dlpack as dlpack

logger = logging.getLogger("dashinfer.serving")


@dataclass
class EngineConfig:
    """Configuration for the DashInfer serving engine."""
    model: str                          # HuggingFace model path or model ID
    model_name: Optional[str] = None    # Display name (defaults to last segment of model path)
    data_type: str = "bfloat16"         # Weight data type: float16, bfloat16, float32
    device_type: str = "CUDA"           # CUDA or CPU
    device_ids: List[int] = field(default_factory=lambda: [0])
    engine_max_batch: int = 32
    engine_max_length: int = 8192
    enable_prefix_cache: bool = True
    trust_remote_code: bool = True
    cache_dir: Optional[str] = None     # Output dir for converted model; default: ~/.cache/dashinfer/
    # VLM-specific options (ignored by LLM handler)
    vision_engine: str = "tensorrt"     # tensorrt or transformers
    min_pixels: int = 4 * 28 * 28
    max_pixels: int = 16384 * 28 * 28
    quant_type: Optional[str] = None    # gptq, a8w8, a16w4, a16w8, fp8, etc.


# ──────────────────────── Abstract Base ────────────────────────

class BaseEngineHandler(ABC):
    """
    Abstract base for engine handlers.
    Both LLM and VLM handlers implement this interface so
    the server layer can treat them uniformly.
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self._model_name = config.model_name or os.path.basename(config.model.rstrip("/"))

    @abstractmethod
    def start(self):
        """Load model, convert, install in engine, and start serving."""
        ...

    @abstractmethod
    def stop(self):
        """Stop the engine and release resources."""
        ...

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    @abstractmethod
    def tokenizer(self):
        ...

    @abstractmethod
    def generate(
        self,
        input_ids: list,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        stop_words_ids: Optional[List[List[int]]] = None,
        seed: Optional[int] = None,
    ) -> Iterator[Tuple[List[int], bool]]:
        """
        Generate tokens from input_ids, yielding (new_token_ids, is_finished) tuples.
        This is a synchronous generator -- run in a thread for async usage.
        """
        ...

    @abstractmethod
    def tokenize_chat(self, messages: list) -> list:
        """
        Apply chat template and tokenize messages.
        Args:
            messages: List of {"role": ..., "content": ...} dicts.
        Returns:
            List of token IDs.
        """
        ...


# ──────────────────────── Auto-Detection ───────────────────────

# Known VLM model_type values from HuggingFace configs
_VLM_MODEL_TYPES = {"qwen2_vl", "qwen_vl", "qwen2_audio"}


def detect_model_mode(model_path: str, trust_remote_code: bool = True) -> str:
    """
    Auto-detect whether a HuggingFace model is LLM or VLM
    by inspecting its config.json model_type field.

    Returns "vlm" or "llm".
    """
    from transformers import AutoConfig
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        model_type = getattr(config, "model_type", "").lower()
        if model_type in _VLM_MODEL_TYPES:
            return "vlm"
    except Exception as e:
        logger.warning(f"Could not auto-detect model type: {e}. Defaulting to LLM.")
    return "llm"


# ──────────────────────── LLM Handler ──────────────────────────

class LLMEngineHandler(BaseEngineHandler):
    """
    LLM engine handler.
    Manages the DashInfer engine lifecycle for text-only models:
      1. Load HuggingFace model
      2. Convert to AllSpark format
      3. Install & start the engine
      4. Serve generation requests
    """

    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self._engine = None
        self._tokenizer = None
        self._model_loader = None
        self._gen_config_builder = None

    # ─────────────────────── Lifecycle ───────────────────────

    def start(self):
        """Load model, convert, install in engine, and start serving."""
        from dashinfer import allspark

        cfg = self.config
        logger.info(f"[LLM] Loading model: {cfg.model}")

        # 1. Load and convert HuggingFace model
        home_dir = os.environ.get("HOME", "/root")
        cache_dir = cfg.cache_dir or os.path.join(home_dir, ".cache", "dashinfer", self._model_name)
        os.makedirs(cache_dir, exist_ok=True)

        internal_name = "model"
        self._model_loader = allspark.HuggingFaceModel(
            cfg.model,
            internal_name,
            in_memory_serialize=False,
            user_set_data_type=cfg.data_type,
            trust_remote_code=cfg.trust_remote_code,
        )
        self._model_loader.load_model(direct_load=False, load_format="auto")
        self._model_loader.serialize(model_output_dir=cache_dir)
        self._model_loader.free_model()

        as_graph_path = os.path.join(cache_dir, internal_name + ".asgraph")
        as_weight_path = os.path.join(cache_dir, internal_name + ".asparam")

        # 2. Init tokenizer
        self._model_loader.init_tokenizer()
        self._tokenizer = self._model_loader.get_tokenizer()

        # 3. Create engine
        self._engine = allspark.Engine()

        cuda_devices = ",".join(str(d) for d in cfg.device_ids)
        if cfg.device_type.upper() == "CUDA":
            compute_unit = f"CUDA:{cuda_devices}"
        else:
            compute_unit = f"CPU:{cuda_devices}"

        as_model_config = allspark.AsModelConfig(
            model_name=internal_name,
            model_path=as_graph_path,
            weights_path=as_weight_path,
            engine_max_length=cfg.engine_max_length,
            engine_max_batch=cfg.engine_max_batch,
            compute_unit=compute_unit,
            enable_prefix_cache=cfg.enable_prefix_cache,
        )

        # 4. Install and start
        self._engine.install_model(as_model_config)
        self._engine.start_model(internal_name)

        # 5. Build reference generation config
        self._gen_config_builder = self._model_loader.create_reference_generation_config_builder(as_model_config)

        logger.info(f"[LLM] Engine started. Model: {self._model_name}, "
                     f"Max batch: {cfg.engine_max_batch}, Max length: {cfg.engine_max_length}")

    def stop(self):
        """Stop the engine and release resources."""
        if self._engine:
            internal_name = "model"
            try:
                self._engine.stop_model(internal_name)
                self._engine.release_model(internal_name)
            except Exception as e:
                logger.warning(f"Error stopping engine: {e}")
            self._engine = None

    @property
    def tokenizer(self):
        return self._tokenizer

    # ─────────────────────── Generation ──────────────────────

    def generate(
        self,
        input_ids: list,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        stop_words_ids: Optional[List[List[int]]] = None,
        seed: Optional[int] = None,
    ) -> Iterator[Tuple[List[int], bool]]:
        from dashinfer.allspark import GenerateRequestStatus

        internal_name = "model"

        # Build generation config from reference, then override with request params
        gen_cfg = self._gen_config_builder.build()
        gen_cfg["max_length"] = len(input_ids) + max_tokens
        gen_cfg["temperature"] = temperature
        gen_cfg["top_p"] = top_p
        gen_cfg["top_k"] = top_k
        gen_cfg["repetition_penalty"] = repetition_penalty
        gen_cfg["presence_penalty"] = presence_penalty
        gen_cfg["frequency_penalty"] = frequency_penalty
        gen_cfg["do_sample"] = True
        gen_cfg["early_stopping"] = True
        if stop_words_ids:
            gen_cfg["stop_words_ids"] = stop_words_ids
        if seed is not None:
            gen_cfg["seed"] = seed
        else:
            gen_cfg["seed"] = random.randint(1, 0x7FFFFFFF)

        # Prepare input tensor
        input_tensor = torch.LongTensor([input_ids]).cpu()
        input_dict = {
            "input_ids": dlpack.to_dlpack(input_tensor),
        }

        # Start request
        status, handle, queue = self._engine.start_request(internal_name, input_dict, gen_cfg)

        try:
            while True:
                gen_elem = queue.Get()
                if gen_elem is None:
                    break

                req_status = queue.GenerateStatus()
                new_ids = list(gen_elem.ids_from_generate)
                is_finished = (req_status == GenerateRequestStatus.GenerateFinished)

                yield new_ids, is_finished

                if is_finished:
                    break
        finally:
            self._engine.stop_request(internal_name, handle)
            self._engine.release_request(internal_name, handle)

    def tokenize_chat(self, messages: list) -> list:
        try:
            input_ids = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            )
        except Exception:
            # Fallback: concatenate messages manually
            text = ""
            for msg in messages:
                text += f"{msg['role']}: {msg['content']}\n"
            text += "assistant: "
            input_ids = self._tokenizer.encode(text)
        return input_ids


# Backward compatibility alias
EngineHandler = LLMEngineHandler
