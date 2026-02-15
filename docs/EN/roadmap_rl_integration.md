# Roadmap: Reinforcement Learning Integration for DashInfer

> **Status**: Draft  
> **Created**: 2026-02-14  
> **Last Updated**: 2026-02-14

## 1. Motivation

Reinforcement Learning from Human Feedback (RLHF) and its variants (DPO, GRPO, REINFORCE++)
have become the standard post-training methodology for LLMs. The training loop critically depends
on an inference engine for two tasks:

1. **Rollout**: Generate responses from the current policy model at high throughput.
2. **Scoring**: Compute token-level log probabilities for generated sequences under
   reference/reward models.

Leading inference engines (vLLM, SGLang, TensorRT-LLM) have built dedicated features for these
workflows and are integrated with training frameworks such as OpenRLHF, veRL (ByteDance), and
TRL (HuggingFace). DashInfer currently lacks these features, preventing its adoption in RL
training pipelines.

This document outlines the features required for DashInfer to serve as the inference backend in
RL training, prioritized by necessity and impact.

## 2. RL Training Loop Overview

A typical PPO/GRPO training iteration looks like:

```
┌───────────────────────────────────────────────────────────────┐
│                      RL Training Loop                         │
│                                                               │
│  Step 1: Rollout (Inference Engine)                           │
│    ├─ Actor model generates responses for a batch of prompts  │
│    ├─ Collect per-token logprobs of generated sequences       │
│    └─ (Optional) Multi-turn dialogue with environment         │
│                                                               │
│  Step 2: Scoring (Inference Engine)                           │
│    ├─ Reference model computes logprobs of generated sequences│
│    └─ Reward model scores the responses                       │
│                                                               │
│  Step 3: Training (Training Framework, e.g., DeepSpeed)       │
│    ├─ Compute PPO/GRPO/DPO loss using collected logprobs      │
│    └─ Update actor model weights via gradient descent         │
│                                                               │
│  Step 4: Weight Sync                                          │
│    └─ Push updated weights to inference engine (no restart)   │
│                                                               │
│  └─── Repeat from Step 1                                      │
└───────────────────────────────────────────────────────────────┘
```

Each step places specific requirements on the inference engine, detailed below.

## 3. Feature Roadmap

### 3.1 Phase 1: Minimum Viable RL Backend

These features are **prerequisites** — without them, DashInfer cannot participate in an
RL training loop at all.

#### 3.1.1 Prompt Logprobs (Prefill-Stage Log Probabilities)

| | |
|---|---|
| **Priority** | P0 — Blocker |
| **Effort** | Medium |
| **Depends on** | None |

**What**: Return per-token log probabilities for all input (prompt) tokens in a single
forward pass, not just for generated (decode) tokens.

**Why**: In RL training, the reference model must score an entire sequence
(prompt + generated response) by computing `log P(token_i | token_{<i})` for every
position. This is also required for:
- KL divergence computation between policy and reference models (PPO, GRPO)
- DPO loss on chosen/rejected response pairs
- Perplexity-based benchmarks (WikiText, HellaSwag)
- Best-of-N rejection sampling
- Data quality filtering by perplexity
- Knowledge distillation (teacher model logits)

**Current state**: DashInfer computes logprobs only during the decode (sampling) phase
in `GenerateOp::RunSample()`. The underlying `logprobs_cpu()` / `logprobs_gpu()` functions
(log_softmax + top-K) already exist but are not invoked during prefill.

**Proposed API**:
```python
gen_cfg.update({"prompt_logprobs": 5})  # Return top-5 logprobs per prompt token

status, handle, queue = engine.start_request_ids(model_name, model, input_ids, gen_cfg)
# ...
elements = queue.Get()
# elements.prompt_log_probs_list: List[List[Tuple[int, float]]]
#   → per input position, top-K (token_id, logprob) pairs
# elements.prompt_token_logprobs_list: List[float]
#   → logprob of the actual input token at each position
```

**Implementation sketch**:
1. Add `prompt_logprobs` field to `GenerateConfig` (proto + Python).
2. In `GenerateOp::RunContext()` (the prefill path), after the final logits are computed:
   - Call `logprobs_launcher()` on the logits tensor for each position.
   - Note: prefill logits shape is `[1, seq_len, vocab_size]` vs decode's
     `[batch_size, 1, vocab_size]` — requires reshaping or a batched loop.
3. Store results in `Request::prompt_log_probs_list` and expose via `GeneratedElements`.

**Reference implementations**:
- vLLM: `SamplingParams(prompt_logprobs=N)` → `output.prompt_logprobs`
- SGLang: `return_logprob=True, logprob_start_len=0` → `input_token_logprobs`
- TensorRT-LLM: `--gather_context_logits` build flag → `context_logits`

---

#### 3.1.2 In-Place Weight Update (Hot Reload)

| | |
|---|---|
| **Priority** | P0 — Blocker |
| **Effort** | High |
| **Depends on** | None |

**What**: Update model weights in the running engine without restarting, re-serializing,
or re-allocating KV cache and other internal data structures.

**Why**: RL training updates actor weights every iteration (every few seconds to minutes).
The current DashInfer flow requires `stop_model → release_model → serialize → install_model
→ start_model`, which takes minutes and is completely impractical for a training loop.

**Proposed API**:
```python
# Option A: Update from state dict (CPU tensors)
engine.update_weights(model_name, state_dict: Dict[str, torch.Tensor])

# Option B: Update from GPU tensors via NCCL
engine.update_weights_nccl(model_name, nccl_comm, src_rank=0)

# Option C: Update from shared GPU memory (zero-copy)
engine.update_weights_from_ipc(model_name, ipc_handles: Dict[str, IpcHandle])
```

**Implementation considerations**:
- Weight tensors in the engine must be updatable without rebuilding the computation graph.
- The engine must skip re-serialization (`.asparam` files) — work directly on in-memory tensors.
- KV cache, operator state, and runtime config should remain untouched.
- Must handle tensor parallelism: each rank updates only its shard.

**Reference implementations**:
- vLLM: `llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights()`
  combined with sleep/wake for memory management.
- SGLang: Checkpoint engine (`ckpt-engine`) with async pipelined weight transfer.

---

### 3.2 Phase 2: Training Framework Integration

These features enable DashInfer to be adopted by training frameworks (OpenRLHF, veRL, TRL)
with reasonable performance.

#### 3.2.1 Sleep/Wake Mode (GPU Memory Yield)

| | |
|---|---|
| **Priority** | P1 — Important |
| **Effort** | High |
| **Depends on** | 3.1.2 (Weight Update) |

**What**: Allow the engine to temporarily release most GPU memory (weights, KV cache)
so that training can use the same GPUs, then quickly restore when inference is needed.

**Why**: In colocated setups, training and inference alternate on the same GPU group.
Without sleep/wake, you need 2x GPUs (one set for training, one for inference).

**Proposed API**:
```python
engine.sleep(model_name, level=1)
# Level 1: offload weights to CPU, discard KV cache  (~90% VRAM freed)
# Level 2: discard weights entirely                   (~95% VRAM freed)

engine.wake_up(model_name, tags=["weights"])    # restore weights only
# ... update weights here (no KV cache allocated, avoids OOM) ...
engine.wake_up(model_name, tags=["kv_cache"])   # then restore KV cache
```

**Reference**: vLLM Sleep Mode (v0.8+) — Level 1/2 sleep, fine-grained wake_up with tags.

---

#### 3.2.2 Training-Inference Colocation

| | |
|---|---|
| **Priority** | P1 — Important |
| **Effort** | High |
| **Depends on** | 3.2.1 (Sleep/Wake) |

**What**: Run training (PyTorch/DeepSpeed) and inference (DashInfer) in the same process,
sharing the same CUDA context and GPU memory pool. They take turns using the GPU.

**Why**: Eliminates the need for separate GPU allocations for training vs inference. In
the TRL + vLLM colocation model, this halves GPU requirements.

**Key requirements**:
- DashInfer must be able to initialize within an existing CUDA context (not create its own).
- Weight tensors should be directly passable from PyTorch (via DLPack or shared CUDA memory)
  without serialization.
- The engine must tolerate other CUDA allocations coexisting on the same GPU.

**Reference**: TRL v0.18+ colocated vLLM — training and inference share the same distributed
process group and devices.

---

#### 3.2.3 Distributed Orchestration (Ray / RPC Integration)

| | |
|---|---|
| **Priority** | P1 — Important |
| **Effort** | Medium |
| **Depends on** | 3.1.1, 3.1.2 |

**What**: Provide a Ray Actor wrapper or gRPC/RPC interface so that RLHF orchestrators
can schedule DashInfer as a remote service.

**Why**: RLHF pipelines have multiple roles (Actor, Critic, Reference, Reward) that must
be orchestrated across nodes. OpenRLHF and veRL use Ray for this. Without Ray integration,
DashInfer cannot be used in these frameworks.

**Proposed approach**:
```python
import ray

@ray.remote(num_gpus=1)
class DashInferRolloutWorker:
    def __init__(self, model_path, ...):
        self.engine = allspark.Engine()
        # ... setup ...

    def generate(self, prompts, gen_config) -> List[RolloutResult]:
        # batch generation with logprobs
        ...

    def compute_logprobs(self, sequences) -> List[TokenLogprobs]:
        # prompt logprobs for scoring
        ...

    def update_weights(self, state_dict):
        # in-place weight update
        ...

    def sleep(self):
        self.engine.sleep(self.model_name)

    def wake_up(self):
        self.engine.wake_up(self.model_name)
```

**Reference**: OpenRLHF's `vLLMRayActor`, veRL's `SGLangRolloutWorker`.

---

#### 3.2.4 Async Weight Synchronization

| | |
|---|---|
| **Priority** | P1 — Nice to have |
| **Effort** | Medium |
| **Depends on** | 3.1.2 (Weight Update) |

**What**: Overlap weight transfer with ongoing rollout generation, so that weight
sync does not block the pipeline.

**Why**: Weight sync (CPU→GPU or GPU→GPU via NCCL) can take seconds for large models.
If done synchronously, this directly adds to each training iteration's wall time.

**Approach**: Background worker thread that streams weight tensors to GPU while the
engine continues generating with the previous weights. Once transfer completes, swap
atomically.

**Reference**: SGLang checkpoint engine — asynchronous pipelined data transfer with
background workers.

---

### 3.3 Phase 3: Quality and Performance

These features improve training quality and efficiency but are not strict blockers.

#### 3.3.1 Deterministic Inference

| | |
|---|---|
| **Priority** | P2 |
| **Effort** | Medium |
| **Depends on** | None |

**What**: Given the same input, sampling parameters, and random seed, produce
bit-identical output across runs.

**Why**: Reproducible RL training is essential for debugging and scientific rigor.
SGLang achieved 100% reproducible RL training in 2025, validated on Qwen3-8B with
identical training curves across independent runs.

**Key challenges**:
- CUDA floating-point non-determinism (atomics, reduction order)
- Non-deterministic thread scheduling in CPU operators
- Prefix cache eviction order affecting results

---

#### 3.3.2 Partial Rollout (Pause/Resume Generation)

| | |
|---|---|
| **Priority** | P2 |
| **Effort** | Medium |
| **Depends on** | None |

**What**: Generate tokens in segments, returning control to the caller between
segments while preserving KV cache state.

**Why**: Multi-turn agentic RL requires interleaving generation with environment
interaction:
```
Generate(prompt) → action tokens
Environment(action) → observation
Generate(prompt + action + observation) → next action tokens
...
```

Current DashInfer supports request interruption (`GenerateInterrupted`), but resuming
generation with preserved KV cache requires explicit support.

---

#### 3.3.3 Speculative Decoding

| | |
|---|---|
| **Priority** | P2 |
| **Effort** | High |
| **Depends on** | 3.1.1 (Prompt Logprobs) |

**What**: Use a small draft model to propose multiple tokens, then verify them in
parallel with the target model.

**Why**: During rollout, generation speed directly impacts training throughput. With
chain-of-thought responses potentially being thousands of tokens long, speculative
decoding can provide 2-3x speedup. The verification step is essentially computing
prompt logprobs for the draft tokens.

---

## 4. Competitive Landscape

| Feature | vLLM | SGLang | TRT-LLM | DashInfer |
|---------|------|--------|---------|-----------|
| Prompt logprobs | Yes | Yes | Yes (build flag) | **No** |
| Weight hot reload | Yes (sleep/wake) | Yes (ckpt-engine) | Partial | **No** |
| Sleep/Wake | Yes (L1/L2) | No | No | **No** |
| Colocation | Yes (TRL) | Partial | No | **No** |
| Ray integration | Yes (native) | Yes (veRL) | No | **No** |
| Async weight sync | Partial | Yes | No | **No** |
| Deterministic inference | Partial | Yes | No | Unknown |
| Partial rollout | Yes | Yes | No | Partial |
| Speculative decoding | Yes | Yes | Yes | **No** |
| **RL framework adoption** | OpenRLHF, TRL, veRL | veRL | NeMo-Aligner | **None** |

## 5. Proposed Timeline

```
Phase 1 (Minimum Viable RL Backend)
├── 3.1.1 Prompt Logprobs
│     ├── C++ engine changes (prefill logits extraction)
│     ├── Python API (prompt_logprobs config + GeneratedElements fields)
│     └── Unit tests + lm-eval integration
└── 3.1.2 In-Place Weight Update
      ├── Weight tensor management refactor
      ├── Python API (update_weights)
      └── Integration test with PyTorch state_dict

Phase 2 (Training Framework Integration)
├── 3.2.1 Sleep/Wake Mode
├── 3.2.2 Colocation Support
├── 3.2.3 Ray Actor wrapper
└── 3.2.4 Async Weight Sync

Phase 3 (Quality and Performance)
├── 3.3.1 Deterministic Inference
├── 3.3.2 Partial Rollout
└── 3.3.3 Speculative Decoding
```

## 6. Success Criteria

Phase 1 is complete when:
- DashInfer can serve as a drop-in replacement for vLLM in a simple GRPO training script
  (e.g., OpenRLHF's `train_grpo.py`).
- An end-to-end test demonstrates: rollout → scoring → weight update → rollout with
  updated weights, all without engine restart.

Phase 2 is complete when:
- DashInfer is integrated as an optional backend in at least one major RL framework
  (OpenRLHF or veRL).
- Colocated training achieves comparable throughput to the vLLM backend on the same hardware.

## 7. Related Work

- [vLLM RLHF Documentation](https://docs.vllm.ai/en/latest/training/rlhf.html)
- [OpenRLHF + vLLM Best Practices](https://vllm-project.github.io/2025/04/23/openrlhf-vllm.html)
- [SGLang Deterministic Inference for RL](https://lmsys.org/blog/2025-09-22-sglang-deterministic/)
- [veRL SGLang Worker](https://verl.readthedocs.io/en/latest/workers/sglang_worker.html)
- [TRL + vLLM Colocation](https://huggingface.co/blog/vllm-colocate)
- [OpenRLHF Paper (arXiv:2405.11143)](https://arxiv.org/abs/2405.11143)

## 8. Appendix: DashInfer Existing Capabilities

Features that already support RL workloads (no changes needed):

- **Continuous batching**: Efficient generation for rollout with varying sequence lengths.
- **Paged attention (SpanAttention)**: Memory-efficient KV cache for long CoT sequences.
- **Prefix caching**: Reuse KV cache across requests sharing the same prompt prefix
  (useful when scoring multiple responses for the same prompt).
- **Multi-GPU via NCCL**: Tensor parallelism for large models.
- **Logprobs during decode**: Top-K logprobs for generated tokens (already implemented).
- **Python bindings (pybind11)**: Direct integration without HTTP overhead.
