'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    vlm_engine_handler.py

 VLM engine handler -- wraps the dashinfer_vlm multimodal pipeline
 behind the same BaseEngineHandler interface used by the LLM path.

 This module is imported *conditionally* (only when --mode vlm is active).
 It requires the `dashinfer-vlm` package to be installed:
     pip install "dashinfer[serving,vlm]"
'''
import os
import logging
import random
from typing import Optional, List, Iterator, Tuple

import torch

from .engine_handler import BaseEngineHandler, EngineConfig

logger = logging.getLogger("dashinfer.serving")


class VLMEngineHandler(BaseEngineHandler):
    """
    VLM engine handler.
    Wraps the dashinfer_vlm multimodal pipeline (ViT + AllSpark LLM)
    behind the unified BaseEngineHandler interface.

    The heavy lifting is delegated to the existing dashinfer_vlm package:
      - HuggingFaceVLModel for model loading
      - QwenVl for the combined ViT + LLM forward pass
      - Image/video preprocessing via get_image_preprocessor
    """

    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self._tokenizer = None
        self._vl_engine = None      # QwenVl instance
        self._preprocessor = None
        self._model_type = None
        self._eos_token_id = None

    # ─────────────────────── Lifecycle ───────────────────────

    def start(self):
        """Load VLM model, set up ViT engine, install AllSpark engine."""
        # Conditional imports -- fail fast with a clear message
        try:
            from dashinfer_vlm.vl_inference.utils.model_loader import HuggingFaceVLModel
            from dashinfer_vlm.vl_inference.utils.hie.vit_preprocess import get_image_preprocessor
            from dashinfer_vlm.vl_inference.runtime.qwen_vl import QwenVl
            from dashinfer_vlm.vl_inference.utils.config import VitConfig, CacheConfig
            from dashinfer_vlm.vl_inference.utils.env import setenv, getenv
        except ImportError:
            raise ImportError(
                "VLM mode requires the dashinfer-vlm package. "
                "Install it with: pip install 'dashinfer[serving,vlm]'"
            )

        from dashinfer import allspark
        from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig

        cfg = self.config
        logger.info(f"[VLM] Loading model: {cfg.model}")

        setenv()

        # 1. Load and convert model
        home_dir = os.environ.get("HOME", "/root")
        cache_dir = cfg.cache_dir or os.path.join(home_dir, ".cache", "dashinfer", self._model_name)
        os.makedirs(cache_dir, exist_ok=True)
        internal_name = "model"

        model_loader = HuggingFaceVLModel(
            cfg.model,
            internal_name,
            in_memory_serialize=False,
            user_set_data_type=cfg.data_type,
            trust_remote_code=True,
            vision_engine=cfg.vision_engine,
            quant_type=cfg.quant_type,
        )
        (
            model_loader.load_model(direct_load=False, load_format="auto")
            .serialize(model_output_dir=cache_dir)
            .free_model()
        )

        as_graph_path = os.path.join(cache_dir, internal_name + ".asgraph")
        as_weight_path = os.path.join(cache_dir, internal_name + ".asparam")
        vit_model_path = model_loader.vision_model_path

        # 2. Read model config and tokenizer
        hf_config = Qwen2VLConfig.from_pretrained(
            model_loader.hf_model_path, trust_remote_code=True,
        )
        self._model_type = hf_config.model_type.upper().replace("_", "-")
        self._tokenizer = model_loader.tokenizer
        self._eos_token_id = self._tokenizer.eos_token_id

        # 3. Build AllSpark engine config
        cuda_devices = ",".join(str(d) for d in cfg.device_ids)
        compute_unit = f"CUDA:{cuda_devices}"

        as_model_config = allspark.AsModelConfig(
            model_name=internal_name,
            model_path=as_graph_path,
            weights_path=as_weight_path,
            engine_max_length=cfg.engine_max_length,
            engine_max_batch=cfg.engine_max_batch,
            compute_unit=compute_unit,
            enable_prefix_cache=cfg.enable_prefix_cache,
        )

        # 4. Build ViT config
        vit_config = VitConfig(
            model_path=vit_model_path,
            precision="fp16",
            workers=1,
            backend=cfg.vision_engine,
        )

        # 5. Build cache config
        cache_config = CacheConfig(
            url=getenv("VL_REDIS_URL", "127.0.0.1"),
            port=getenv("VL_REDIS_PORT", 6379),
            passwd=getenv("VL_REDIS_PASSWD", "1234"),
            valid_cache_time=getenv("VL_VALID_CACHE_TIME", 300000),
        )

        # 6. Create VL engine
        self._vl_engine = QwenVl(
            as_config=as_model_config,
            vit_config=vit_config,
            cache_config=cache_config,
            trt_vit_config=model_loader.vit_config,
        )

        # 7. Create image preprocessor
        self._preprocessor = get_image_preprocessor(
            workers=vit_config.workers,
            vl_version=2 if self._model_type == "QWEN2-VL" else 1,
            dtype=torch.float16,
        )

        # Store config values for request building
        self._min_pixels = cfg.min_pixels
        self._max_pixels = cfg.max_pixels
        self._model_loader = model_loader

        logger.info(f"[VLM] Engine started. Model: {self._model_name}, "
                     f"Type: {self._model_type}, "
                     f"Max batch: {cfg.engine_max_batch}, Max length: {cfg.engine_max_length}")

    def stop(self):
        """Stop the VLM engine."""
        self._vl_engine = None
        logger.info("[VLM] Engine stopped.")

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
        # VLM-specific: the pre-built VLRequest object
        _vl_request=None,
    ) -> Iterator[Tuple[List[int], bool]]:
        """
        Generate tokens via the VLM pipeline.

        When called from the server, `_vl_request` carries the full
        multimodal request (images + text).  The plain `input_ids` arg
        is only used as a fallback for pure-text requests.
        """
        from dashinfer import allspark

        if _vl_request is None:
            raise ValueError("VLM generate requires a _vl_request object. "
                             "Use tokenize_chat() to build one first.")

        for gen_result in self._vl_engine.forward(_vl_request):
            if len(gen_result) == 5:
                vl_status, req_id, as_status, output, prompt_nums = gen_result
            else:
                vl_status, req_id, as_status, output = gen_result

            is_finished = (as_status == allspark.GenerateRequestStatus.GenerateFinished)
            yield list(output), is_finished

            if is_finished:
                break

    def tokenize_chat(self, messages: list) -> list:
        """
        Build a VLRequest from OpenAI-format messages.

        For VLM, this handles the full multimodal conversation parsing:
        text + image URLs + video.  The returned ``input_ids`` are the
        text token portion; the full VLRequest (with image data) is
        stored on ``self._last_vl_request`` so that ``generate()`` can
        pick it up.

        Returns:
            List of token IDs (text portion only, for usage counting).
        """
        from dashinfer_vlm.vl_inference.runtime.qwen_vl import VLRequest
        from dashinfer_vlm.api_server.conversation import Conversation, SeparatorStyle
        import shortuuid

        conv = Conversation(
            name="qwen2-vl-chatml",
            system_template="<|im_start|>system\n{system_message}",
            system_message="",
            messages=[],
            roles=("<|im_start|>user", "<|im_start|>assistant"),
            sep_style=SeparatorStyle.CHATML,
            sep="<|im_end|>",
            stop_str="<|im_end|>",
        )

        vl_request = VLRequest(
            shortuuid.random(),
            self._tokenizer,
            "CHATML",
            self._model_type,
            self._preprocessor,
            self._min_pixels,
            self._max_pixels,
        )

        for message in messages:
            msg_role = message.get("role", "")
            content = message.get("content", "")
            if msg_role == "system":
                conv.set_system_message(content)
            elif msg_role in ("user", "assistant"):
                if isinstance(content, list):
                    text, image_list = conv.get_content(content)
                    role = conv.roles[0] if msg_role == "user" else conv.roles[1]
                    conv.append_message(role, (text, image_list))
                elif isinstance(content, str):
                    role = conv.roles[0] if msg_role == "user" else conv.roles[1]
                    conv.append_message(role, (content, []))

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = conv.get_input_tokens(prompt, self._tokenizer)

        preprocess_req = conv.get_preprocess_req(
            min_pixels=self._min_pixels, max_pixels=self._max_pixels,
        )
        vl_request.preprocess_req = preprocess_req
        vl_request.context_length = len(input_ids)
        vl_request.input_tokens = [input_ids]
        vl_request.truncate_lengths = conv.get_truncate_length()

        # Store for generate() to pick up
        self._last_vl_request = vl_request
        return input_ids

    def build_vl_gen_config(
        self,
        input_ids: list,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        seed: Optional[int] = None,
    ):
        """Attach generation config to the last VLRequest and return it."""
        vl_request = self._last_vl_request
        vl_request.gen_cfg = {
            "num_beams": 1,
            "num_return_sequences": 1,
            "temperature": temperature,
            "do_sample": True,
            "early_stopping": True,
            "top_k": top_k,
            "top_p": top_p,
            "max_length": max_tokens + len(input_ids),
            "min_length": 5,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "length_penalty": 1,
            "stop_words_ids": [[151643], [151644], [151645]],
            "eos_token_id": self._eos_token_id,
            "seed": seed or random.randint(1, 0x7FFFFFFF),
        }
        return vl_request
