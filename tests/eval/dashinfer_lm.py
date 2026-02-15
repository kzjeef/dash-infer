"""
DashInfer adapter for EleutherAI's lm-evaluation-harness.

This module implements the LM interface required by lm-evaluation-harness,
enabling standard LLM benchmarks (MMLU, GSM8K, HellaSwag, WikiText, etc.)
to run against the DashInfer inference engine.

Usage:
    # Python API (recommended)
    from dashinfer_lm import DashInferLM
    import lm_eval

    lm = DashInferLM(
        pretrained="/path/to/model",
        device="cpu",             # "cpu" or "cuda:0"
        data_type="float32",      # "float32", "bfloat16", "float16"
        max_length=4096,
        max_batch=1,
    )
    results = lm_eval.simple_evaluate(model=lm, tasks=["gsm8k"])

    # CLI (requires lm_eval to discover this module)
    # lm_eval --model dashinfer --model_args pretrained=/path/to/model ...

Copyright (c) Alibaba, Inc. and its affiliates.
"""

import os
import sys
import copy
import logging
import tempfile
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.dlpack as dlpack

from lm_eval.api.model import LM
from lm_eval.api.instance import Instance

try:
    from lm_eval.api.registry import register_model
except ImportError:
    # Fallback if registry is not available in the installed version
    def register_model(*args, **kwargs):
        def decorator(cls):
            return cls
        return decorator

logger = logging.getLogger(__name__)


def _find_token_logprob(
    log_probs_list: List[List[Tuple[int, float]]],
    token_idx: int,
    target_token_id: int,
    fallback_logprob: float = -100.0,
) -> Tuple[float, bool]:
    """
    Find the logprob of a specific token from the top-K logprobs at a given position.

    Args:
        log_probs_list: The log_probs_list from GeneratedElements.
            Shape: [num_generated_tokens][top_logprobs] where each entry is (token_id, logprob).
        token_idx: Which generated token position to look at.
        target_token_id: The token ID whose logprob we want.
        fallback_logprob: Value to return if the target token is not in top-K.

    Returns:
        (logprob, is_greedy): logprob of the target token, and whether it's the
        top-ranked (greedy) token at this position.
    """
    if token_idx >= len(log_probs_list):
        return fallback_logprob, False

    position_logprobs = log_probs_list[token_idx]
    if not position_logprobs:
        return fallback_logprob, False

    # The first entry in the sorted top-K list is the greedy token
    greedy_token_id = position_logprobs[0][0]
    is_greedy = (greedy_token_id == target_token_id)

    for tid, lp in position_logprobs:
        if tid == target_token_id:
            return lp, is_greedy

    # Target token not in top-K; use fallback
    return fallback_logprob, False


@register_model("dashinfer")
class DashInferLM(LM):
    """
    lm-evaluation-harness adapter for the DashInfer (AllSpark) inference engine.

    Supports both CPU and CUDA backends. Implements all three evaluation methods:
    - loglikelihood: scoring of (context, continuation) pairs
    - loglikelihood_rolling: full-sequence perplexity (for WikiText, etc.)
    - generate_until: free-form text generation (for GSM8K, etc.)

    Note on loglikelihood implementation:
        DashInfer does not natively expose prompt-level logprobs (logprobs during
        prefill). Instead, this adapter computes loglikelihood by making one
        generation call per continuation token position, feeding
        context + continuation[:i] as input and extracting the logprob of
        continuation[i] from the top-K generated logprobs (K=10). This approach:
        - Works perfectly for single-token continuations (e.g., MMLU: A/B/C/D)
        - Works for multi-token continuations if each token appears in top-10
        - Returns a conservative fallback (-100) if a token is outside top-10
        DashInfer's prefix caching accelerates repeated calls with shared prefixes.
    """

    def __init__(
        self,
        pretrained: str,
        device: str = "cpu",
        data_type: str = "float32",
        max_length: int = 4096,
        max_batch: int = 1,
        max_gen_toks: int = 256,
        weight_only_quant: bool = True,
        enable_quant: bool = False,
        model_output_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        top_logprobs: int = 10,
        batch_size: Union[int, str] = 1,
        **kwargs,
    ):
        """
        Initialize the DashInfer LM adapter.

        Args:
            pretrained: Path to the HuggingFace model (local or model ID).
            device: Target device - "cpu", "cuda:0", "cuda:0,1", etc.
            data_type: Weight data type - "float32", "bfloat16", "float16".
            max_length: Maximum sequence length (input + output).
            max_batch: Maximum concurrent batch size for the engine.
            max_gen_toks: Default maximum tokens to generate per request.
            weight_only_quant: Enable weight-only quantization during serialization.
            enable_quant: Enable full quantization during serialization.
            model_output_dir: Directory to store serialized model files.
                If None, uses a temp directory.
            trust_remote_code: Trust remote code in HuggingFace models.
            top_logprobs: Number of top logprobs to request (max 10).
            batch_size: Batch size for evaluation (currently only 1 supported).
        """
        super().__init__()

        from dashinfer import allspark
        from dashinfer.allspark.engine import TargetDevice
        from dashinfer.allspark._allspark import GenerateRequestStatus

        self._pretrained = pretrained
        self._max_length = max_length
        self._max_gen_toks = max_gen_toks
        self._top_logprobs = min(top_logprobs, 10)  # Engine max is 10
        self._batch_size_val = int(batch_size) if batch_size != "auto" else 1
        self._GenerateRequestStatus = GenerateRequestStatus

        # Parse device specification
        if device.startswith("cuda"):
            parts = device.split(":")
            if len(parts) > 1:
                device_ids = [int(x) for x in parts[1].split(",")]
            else:
                device_ids = [0]
            target_device = TargetDevice.CUDA
        else:
            device_ids = [0]
            target_device = TargetDevice.CPU

        self._device_str = device
        self._target_device = target_device

        # Model name (safe for filesystem)
        safe_model_name = os.path.basename(pretrained.rstrip("/")).replace("/", "_")
        if not safe_model_name:
            safe_model_name = "dashinfer_model"
        self._model_name = safe_model_name

        # Output directory for serialized model
        if model_output_dir is None:
            self._tmp_dir_obj = tempfile.TemporaryDirectory(prefix="dashinfer_eval_")
            self._model_output_dir = self._tmp_dir_obj.name
        else:
            self._tmp_dir_obj = None
            self._model_output_dir = model_output_dir
            os.makedirs(model_output_dir, exist_ok=True)

        logger.info(f"Initializing DashInfer with model: {pretrained}")
        logger.info(f"Device: {device}, Data type: {data_type}, Max length: {max_length}")

        # Initialize engine and model loader
        self._engine = allspark.Engine()
        self._model_loader = allspark.HuggingFaceModel(
            pretrained,
            safe_model_name,
            user_set_data_type=data_type,
            in_memory_serialize=False,
            trust_remote_code=trust_remote_code,
        )

        # Load, serialize, and configure model
        (
            self._model_loader.load_model()
            .read_model_config()
            .serialize_to_path(
                self._engine,
                self._model_output_dir,
                enable_quant=enable_quant,
                weight_only_quant=weight_only_quant,
                skip_if_exists=True,
            )
            .free_model()
        )

        # Build runtime config
        runtime_cfg_builder = self._model_loader.create_reference_runtime_config_builder(
            safe_model_name,
            target_device,
            device_ids,
            max_batch=max_batch,
        )
        runtime_cfg_builder.max_length(max_length)
        self._runtime_cfg = runtime_cfg_builder.build()

        # Install and start model
        self._engine.install_model(self._runtime_cfg)
        self._engine.start_model(safe_model_name)

        # Initialize tokenizer
        self._tokenizer = self._model_loader.init_tokenizer().get_tokenizer()

        logger.info("DashInfer engine initialized and model started.")

    def __del__(self):
        """Clean up engine resources."""
        try:
            self._engine.stop_model(self._model_name)
            self._engine.release_model(self._model_name)
        except Exception:
            pass
        if self._tmp_dir_obj is not None:
            try:
                self._tmp_dir_obj.cleanup()
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────────────
    # Properties required by lm-evaluation-harness
    # ──────────────────────────────────────────────────────────────────────

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return self._batch_size_val

    @property
    def device(self):
        return self._device_str

    @property
    def tokenizer(self):
        return self._tokenizer

    # ──────────────────────────────────────────────────────────────────────
    # Core helpers
    # ──────────────────────────────────────────────────────────────────────

    def tok_encode(self, string: str, add_special_tokens: bool = False) -> List[int]:
        """Tokenize a string into token IDs."""
        return self._tokenizer.encode(string, add_special_tokens=add_special_tokens)

    def tok_decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to string."""
        return self._tokenizer.decode(tokens, skip_special_tokens=False)

    def _make_gen_config(
        self,
        max_new_tokens: int = 1,
        temperature: float = 0.0,
        top_k: int = 1,
        logprobs: bool = True,
        stop_words_ids: Optional[List[List[int]]] = None,
    ) -> dict:
        """
        Create a generation config dict for DashInfer.

        Args:
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_k: Top-K for sampling.
            logprobs: Whether to return logprobs.
            stop_words_ids: Stop sequences as token ID lists.

        Returns:
            Generation config dict.
        """
        gen_cfg = self._model_loader.create_reference_generation_config_builder(
            self._runtime_cfg
        )
        gen_cfg.update({
            "top_k": top_k,
            "repetition_penalty": 1.0,
        })

        if temperature > 0:
            gen_cfg.update({"temperature": temperature})

        if logprobs:
            gen_cfg.log_probs(True, self._top_logprobs)

        # max_length in DashInfer is total (input + output)
        # We'll set it generously; actual stopping is done via stop words or EOS
        gen_cfg.update({"max_length": self._max_length})

        if stop_words_ids:
            gen_cfg.update({"stop_words_ids": stop_words_ids})

        return gen_cfg

    def _generate_tokens(
        self,
        input_ids: List[int],
        max_new_tokens: int = 1,
        logprobs: bool = True,
        stop_words_ids: Optional[List[List[int]]] = None,
    ) -> Tuple[List[int], List[List[Tuple[int, float]]], List[float]]:
        """
        Run a single generation request through DashInfer.

        Args:
            input_ids: Input token IDs.
            max_new_tokens: Maximum tokens to generate.
            logprobs: Whether to collect logprobs.
            stop_words_ids: Stop sequences.

        Returns:
            Tuple of:
            - generated_ids: List of generated token IDs
            - log_probs_list: Top-K logprobs per token
            - token_logprobs_list: Logprob of the selected token at each position
        """
        gen_cfg = self._make_gen_config(
            max_new_tokens=max_new_tokens,
            logprobs=logprobs,
            stop_words_ids=stop_words_ids,
        )

        # Truncate input if too long (leave room for generation)
        max_input_len = self._max_length - max_new_tokens
        if len(input_ids) > max_input_len:
            input_ids = input_ids[-max_input_len:]

        status, handle, queue = self._engine.start_request_ids(
            self._model_name,
            self._model_loader,
            input_ids,
            gen_cfg,
        )

        generated_ids = []
        all_log_probs = []
        all_token_logprobs = []

        # Collect results
        status = queue.GenerateStatus()
        while status in [
            self._GenerateRequestStatus.Init,
            self._GenerateRequestStatus.Generating,
            self._GenerateRequestStatus.ContextFinished,
        ]:
            elements = queue.Get()
            if elements is not None:
                generated_ids.extend(elements.ids_from_generate)
                if logprobs and hasattr(elements, "log_probs_list"):
                    all_log_probs.extend(elements.log_probs_list)
                if logprobs and hasattr(elements, "token_logprobs_list"):
                    all_token_logprobs.extend(elements.token_logprobs_list)

                # Stop if we have enough tokens
                if len(generated_ids) >= max_new_tokens:
                    break

            status = queue.GenerateStatus()
            if status in [
                self._GenerateRequestStatus.GenerateFinished,
                self._GenerateRequestStatus.GenerateInterrupted,
            ]:
                break

        self._engine.release_request(self._model_name, handle)

        return generated_ids[:max_new_tokens], all_log_probs, all_token_logprobs

    # ──────────────────────────────────────────────────────────────────────
    # lm-evaluation-harness interface methods
    # ──────────────────────────────────────────────────────────────────────

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """
        Compute log-probability of continuation conditioned on context.

        For each (context, continuation) pair:
        1. Tokenize context and continuation separately
        2. For each token position i in the continuation:
           - Feed context + continuation[:i] as input
           - Generate 1 token with logprobs
           - Extract the logprob of continuation[i] from top-K
        3. Sum logprobs, check if all tokens were greedy

        This approach leverages DashInfer's prefix caching for efficiency
        when multiple continuations share the same context prefix.

        Args:
            requests: List of Instance objects. Each instance.args is
                (context: str, continuation: str).

        Returns:
            List of (log_probability, is_greedy) tuples.
        """
        results = []
        for req in requests:
            context, continuation = req.args

            # Tokenize
            if context:
                ctx_ids = self.tok_encode(context)
            else:
                ctx_ids = [self.eot_token_id]

            # Tokenize continuation: ensure proper word-boundary handling
            # by encoding context+continuation together and taking the suffix
            full_ids = self.tok_encode(context + continuation)
            cont_ids = full_ids[len(ctx_ids):]

            if not cont_ids:
                # If tokenization produces nothing for continuation, try direct
                cont_ids = self.tok_encode(continuation)

            if not cont_ids:
                results.append((-float("inf"), False))
                self.cache_hook.add_partial("loglikelihood", req.args, results[-1])
                continue

            total_logprob = 0.0
            all_greedy = True

            for i, target_token_id in enumerate(cont_ids):
                # Input = context tokens + continuation tokens up to (but not including) position i
                input_ids = ctx_ids + cont_ids[:i]

                gen_ids, log_probs, token_logprobs = self._generate_tokens(
                    input_ids, max_new_tokens=1, logprobs=True
                )

                # Extract logprob of the target token from top-K
                logprob, is_greedy_pos = _find_token_logprob(
                    log_probs, token_idx=0, target_token_id=target_token_id
                )

                total_logprob += logprob
                if not is_greedy_pos:
                    all_greedy = False

            result = (total_logprob, all_greedy)
            results.append(result)
            self.cache_hook.add_partial("loglikelihood", req.args, result)

        return results

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        """
        Compute rolling log-likelihood of a string (for perplexity measurement).

        WARNING: This method is very slow with DashInfer because the engine does not
        support prompt-level logprobs (logprobs during prefill). Each token position
        requires a separate generation call, making this O(N) calls for N tokens.
        For a 500-token document this means ~500 sequential generation calls.

        For precision regression testing, prefer using:
        - GSM8K (generate_until) — fast, sensitive to accumulated FP errors
        - MMLU/tinyMMLU (loglikelihood with single-token continuations) — fast

        If you need perplexity as a metric, consider adding prompt logprobs support
        at the C++ engine level (in the prefill path of generate_op).

        Implementation:
        For each string:
        1. Tokenize the full text
        2. For each token position i (starting from 1):
           - Feed tokens[:i] as input
           - Generate 1 token with logprobs
           - Extract the logprob of tokens[i] from top-K
        3. Sum all logprobs

        DashInfer's prefix caching helps somewhat by reusing computation from
        shared prefixes across sequential calls.

        Args:
            requests: List of Instance objects. Each instance.args is (string,).

        Returns:
            List of total log-probabilities (one per string).
        """
        if requests:
            logger.warning(
                "loglikelihood_rolling is very slow with DashInfer (no prompt logprobs). "
                "Processing %d requests with O(N) generation calls per token. "
                "Consider using generate_until-based benchmarks (e.g., GSM8K) instead.",
                len(requests),
            )

        results = []
        for req_idx, req in enumerate(requests):
            (string,) = req.args

            token_ids = self.tok_encode(string)
            if not token_ids:
                results.append(-float("inf"))
                self.cache_hook.add_partial("loglikelihood_rolling", req.args, -float("inf"))
                continue

            total_logprob = 0.0
            num_tokens = len(token_ids)

            if num_tokens > 100:
                logger.info(
                    "loglikelihood_rolling: request %d/%d has %d tokens, "
                    "this will take ~%d generation calls",
                    req_idx + 1, len(requests), num_tokens, num_tokens - 1,
                )

            max_ctx = self._max_length - 1  # leave room for 1 generated token

            # Score positions 1..len(token_ids)-1
            # At each position i, context is token_ids[max(0, i-max_ctx):i]
            for i in range(1, num_tokens):
                start = max(0, i - max_ctx)
                input_ids = token_ids[start:i]
                target_token_id = token_ids[i]

                gen_ids, log_probs, token_logprobs = self._generate_tokens(
                    input_ids, max_new_tokens=1, logprobs=True
                )

                logprob, _ = _find_token_logprob(
                    log_probs, token_idx=0, target_token_id=target_token_id
                )
                total_logprob += logprob

            results.append(total_logprob)
            self.cache_hook.add_partial("loglikelihood_rolling", req.args, total_logprob)

        return results

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Generate text until stop conditions are met.

        For each request:
        1. Tokenize the context
        2. Generate tokens with DashInfer until a stop sequence or max_gen_toks
        3. Decode and truncate at the first stop sequence

        Args:
            requests: List of Instance objects. Each instance.args is
                (context: str, gen_kwargs: dict).
                gen_kwargs may contain:
                - "until": List[str] - stop sequences
                - "max_gen_toks": int - maximum tokens to generate
                - "temperature": float - sampling temperature
                - "do_sample": bool - whether to sample

        Returns:
            List of generated text strings (not including the context).
        """
        results = []
        for req in requests:
            context, gen_kwargs = req.args

            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            max_gen = gen_kwargs.get("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.get("temperature", 0.0)
            do_sample = gen_kwargs.get("do_sample", False)

            # Convert stop strings to token IDs for early stopping
            stop_words_ids = []
            for stop_str in until:
                stop_ids = self.tok_encode(stop_str)
                if stop_ids:
                    stop_words_ids.append(stop_ids)

            # Tokenize context
            ctx_ids = self.tok_encode(context)

            # Set top_k based on sampling mode
            top_k = 0 if (do_sample and temperature > 0) else 1
            temp = temperature if temperature > 0 else 0.0

            gen_cfg = self._make_gen_config(
                max_new_tokens=max_gen,
                temperature=temp,
                top_k=top_k,
                logprobs=False,
                stop_words_ids=stop_words_ids if stop_words_ids else None,
            )

            # Truncate context if needed
            max_ctx_len = self._max_length - max_gen
            if len(ctx_ids) > max_ctx_len:
                ctx_ids = ctx_ids[-max_ctx_len:]

            status, handle, queue = self._engine.start_request_ids(
                self._model_name,
                self._model_loader,
                ctx_ids,
                gen_cfg,
            )

            generated_ids = []
            status = queue.GenerateStatus()
            while status in [
                self._GenerateRequestStatus.Init,
                self._GenerateRequestStatus.Generating,
                self._GenerateRequestStatus.ContextFinished,
            ]:
                elements = queue.Get()
                if elements is not None:
                    generated_ids.extend(elements.ids_from_generate)
                    if len(generated_ids) >= max_gen:
                        break

                status = queue.GenerateStatus()
                if status in [
                    self._GenerateRequestStatus.GenerateFinished,
                    self._GenerateRequestStatus.GenerateInterrupted,
                ]:
                    break

            self._engine.release_request(self._model_name, handle)

            # Decode generated tokens
            generated_text = self.tok_decode(generated_ids[:max_gen])

            # Truncate at first stop sequence
            for stop_str in until:
                idx = generated_text.find(stop_str)
                if idx != -1:
                    generated_text = generated_text[:idx]

            results.append(generated_text)
            self.cache_hook.add_partial("generate_until", req.args, generated_text)

        return results
