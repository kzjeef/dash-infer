# Roadmap: Performance Optimization

> **Status**: Draft  
> **Created**: 2026-02-14  
> **Last Updated**: 2026-02-14

## 1. Goals

| Target | Model | Hardware | Baseline | Metric |
|--------|-------|----------|----------|--------|
| **Target 1** | Dense 72B (Qwen2.5-72B) | 8×H100/H800 | vLLM / SGLang | Throughput + latency parity |
| **Target 2** | DeepSeek V3.2 (671B MoE+NSA) | 8×B200 | SGLang | Throughput + latency parity |

## 2. Key Feature Gaps

### Summary

| # | Feature | Target | Priority | Current State |
|---|---------|--------|----------|---------------|
| 1 | [CUDA Graph Full (decode)](#31-cuda-graph-full-decode) | T1 T2 | P0 | Piecewise only — attention runs eager |
| 2 | [Chunked Prefill + Unified Scheduler](#32-chunked-prefill--unified-scheduler) | T1 T2 | P0 | Explicitly disabled |
| 3 | [DP Attention](#33-dp-attention) | T2 | P0 | Not implemented |
| 4 | [Speculative Decoding (EAGLE)](#34-speculative-decoding-eagle) | T1 T2 | P0 | Not implemented |
| 5 | [FP4 MoE Fused Kernel](#35-fp4-moe-fused-kernel) | T2 | P1 | No FP4; MoE kernel exists but no FP4 fusion |
| 6 | [NSA Kernel Fusion (DeepSeek V3.2)](#36-nsa-kernel-fusion-deepseek-v32) | T2 | P1 | No NSA support |
| 7 | [Multi-Token Prediction (MTP)](#37-multi-token-prediction-mtp) | T2 | P0 | Not implemented |
| 8 | [DeepSeek Architecture Hardening](#38-deepseek-architecture-hardening) | T2 | P0 | Partial support; no dedicated DeepSeek hardening suite |
| 9 | [PD Separation + Mooncake Unified KV-Cache](#39-pd-separation--mooncake-unified-kv-cache) | T2 | P2 | Single-node PD split exists; no unified distributed KV-cache layer |

---

## 3. Feature Details

### 3.1 CUDA Graph Full (Decode)

| | |
|---|---|
| **Priority** | P0 |
| **Target** | T1 (dense 72B) + T2 (DeepSeek) |
| **Effort** | High |
| **Impact** | 15-30% decode latency reduction |

#### Problem

DashInfer currently uses **piecewise** CUDA graph capture. The following operators are
marked `IsGraphUnsafe()` and run eagerly between captured graph segments:

| Operator | Why graph-unsafe | Can it be fixed? |
|----------|-----------------|------------------|
| `SpanAttnOp` (paged attention) | Dynamic KV cache pointer updates | Yes — with pre-registered addresses |
| `MLAAttnOp` (MLA attention) | Same as above | Yes |
| `AllReduceOp` / `AllGatherOp` (NCCL) | NCCL semantics | Partially — via graph-safe NCCL or custom kernels |
| `EmbeddingT5Op` / `RichEmbeddingOp` | Stack-allocated host buffer for H2D | Yes — use persistent buffer |
| `DecOptEmbeddingOp` | Same as above | Yes |

Each eager segment break introduces a `cudaGraphLaunch` → eager kernel → `cudaGraphLaunch`
transition with several microseconds of overhead. For a typical decode step with ~3 eager
breaks, this adds ~10-30μs per step. At high batch sizes / short generation, this overhead
accumulates into a measurable throughput loss.

#### Goal: Full Graph Capture for Uniform Decode Batches

Capture the **entire** decode forward pass (all decoder layers + sampling) as a single
CUDA graph, eliminating all kernel launch overhead during decode.

#### Key Technical Challenges

**1. Attention with dynamic KV cache pointers**

Paged attention reads from KV cache blocks whose addresses change between steps (as new
blocks are allocated). In a captured CUDA graph, pointer values are baked in at capture time.

**Solutions** (pick one):
- **Indirection buffer**: Allocate a persistent GPU buffer holding KV block pointers.
  At each step, update the pointer buffer (a single small H2D memcpy) rather than the
  graph. The attention kernel reads from the indirection buffer. This is how FlashInfer
  enables graph capture for paged attention.
- **cudaGraphExecUpdate**: Use CUDA's graph update API to patch pointer parameters
  in-place. Requires CUDA 12.x and careful handling of parameter nodes.
- **Pre-allocated virtual address pool**: Reserve a contiguous virtual address space
  for all possible KV cache blocks. Map physical memory into this space dynamically.
  Graph captures fixed virtual addresses; physical backing changes. (CUDA VMM API)

**2. NCCL collectives**

NCCL operations cannot be captured in standard CUDA graphs. Options:
- **NCCL graph support**: CUDA 12.4+ added `ncclCommRegister` for graph-safe NCCL.
  Requires NCCL 2.19+ and explicit registration of communication buffers.
- **Custom AllReduce kernels**: For small tensor parallel groups (TP2/TP4/TP8 within
  NVLink), custom AllReduce kernels using NVLink direct access can be graph-captured.
  vLLM uses this approach for TP ≤ 8 with NVLink.
- **Piecewise fallback for NCCL**: Keep NCCL eager but capture everything else in
  one graph. This is the `FULL_AND_PIECEWISE` approach in vLLM — full graph for
  single-GPU, piecewise for multi-GPU. Still a significant improvement.

**3. Embedding ops with host memory**

`EmbeddingT5Op` uses a stack-allocated host buffer for H2D transfer. Fix by using a
persistent pinned-memory buffer allocated once during model initialization.

#### Implementation Plan

1. Fix embedding ops to use persistent buffers → remove `IsGraphUnsafe()`.
2. Implement KV cache indirection buffer for SpanAttn and MLA decode kernels.
3. Make attention ops graph-safe with the indirection buffer → remove `IsGraphUnsafe()`.
4. (Optional) Integrate graph-safe NCCL or custom AllReduce for TP ≤ 8.
5. Add `FULL_DECODE_ONLY` mode: full graph when batch is uniform decode, piecewise otherwise.
6. Benchmark: compare piecewise vs full on Qwen2.5-72B decode latency.

#### Reference

- vLLM CUDA Graph modes: `FULL`, `FULL_DECODE_ONLY`, `FULL_AND_PIECEWISE`
- FlashInfer: graph-safe paged attention via indirection buffer
- SGLang: piecewise CUDA graph with FlashInfer backend

---

### 3.2 Chunked Prefill + Unified Scheduler

| | |
|---|---|
| **Priority** | P0 |
| **Target** | T1 (dense 72B) + T2 (DeepSeek) |
| **Effort** | High |
| **Impact** | 30-50% TTFT improvement, 10-20% throughput improvement |

#### Problem

DashInfer explicitly disables chunked prefill (`as_engine.cpp:440`: `"Current version DO
NOT support chunk prefill"`). The scheduler uses separate `PrefillThread()` and
`DecodeThread()` with a `PrefillDecodeSharedData` bridge.

Without chunked prefill:
- A 32K-token prefill monopolizes the GPU for hundreds of ms.
- All concurrent decode requests stall, causing inter-token latency spikes.
- GPU utilization is suboptimal under mixed workloads.

#### Goal

Split long prefills into chunks (e.g., 8192 tokens) co-scheduled with decode batches
in a **unified token-budget scheduler**.

```
Each iteration:
  budget = max_num_batched_tokens  (e.g., 8192)
  1. Allocate prefill chunk tokens (from pending prefills, up to budget)
  2. Fill remaining budget with decode tokens
  3. Execute mixed batch as one forward pass
```

#### Implementation Plan

1. Merge prefill/decode into a single scheduling loop with a token budget.
2. Implement prefill chunking: split context into `ceil(seq_len / chunk_size)` chunks.
3. Preserve KV cache across chunks (already supported by SpanAttention).
4. Handle attention mask correctly across chunk boundaries.
5. The final chunk's last position transitions to decode phase.
6. Benchmark: ShareGPT workload with mixed short/long requests.

---

### 3.3 DP Attention

| | |
|---|---|
| **Priority** | P0 |
| **Target** | T2 (DeepSeek on B200) |
| **Effort** | High |
| **Impact** | 30-50% decode throughput for MoE + MLA models |

#### Problem

In standard tensor parallelism, every GPU executes attention on the **full batch**, then
all-reduces. For DeepSeek's MLA (small per-head dimension, memory-bandwidth-bound), each
GPU wastes bandwidth loading the full KV cache for the full batch.

#### Goal: Data-Parallel Attention + Expert-Parallel MoE

Each GPU handles a **fraction of the batch** for attention, but **all tokens for its
assigned experts** in MoE layers. An all-to-all exchange redistributes tokens between
attention and MoE phases.

```
GPU Layout (8 GPUs, batch=B, experts=256):

Attention phase (DP):
  GPU 0: attention on batch[0 : B/8]
  GPU 1: attention on batch[B/8 : 2B/8]
  ...
  GPU 7: attention on batch[7B/8 : B]

  → All-to-all: redistribute tokens by expert assignment

MoE phase (EP):
  GPU 0: experts[0:32] on all routed tokens
  GPU 1: experts[32:64] on all routed tokens
  ...
  GPU 7: experts[224:256] on all routed tokens

  → All-to-all: redistribute results back to original GPUs
```

#### Why This Matters

- Each GPU's KV cache is `1/N` the size → supports `N×` larger batch → higher throughput.
- Memory-bandwidth pressure for decode attention is reduced by `N×`.
- MoE layers naturally use EP — no change needed.
- This is SGLang's primary architecture for DeepSeek serving.

#### Implementation Plan

1. Implement request-to-GPU assignment for DP attention (partition batch across GPUs).
2. Each GPU maintains KV cache only for its assigned requests.
3. Add all-to-all communication between attention and MoE phases:
   - After attention: dispatch tokens to expert-owning GPUs (can use DeepEP or NCCL).
   - After MoE: combine results back to attention-owning GPUs.
4. Scheduling: ensure balanced assignment across GPUs (account for varying sequence lengths).
5. Benchmark: DeepSeek V3 decode throughput at batch sizes [8, 16, 32, 64, 128].

#### Interaction with Other Features

- Requires efficient all-to-all communication (DeepEP or custom kernels).
- Benefits strongly from CUDA graph full (the attention phase and MoE phase per GPU
  are uniform and graph-capturable).
- Chunked prefill must also support DP layout.

---

### 3.4 Speculative Decoding (EAGLE)

| | |
|---|---|
| **Priority** | P0 |
| **Target** | T1 (dense 72B) + T2 (DeepSeek) |
| **Effort** | High |
| **Impact** | 2-3× single-request latency reduction |

#### Background

Speculative decoding uses a small **draft model** to propose `K` tokens ahead, then the
**target model** verifies them in a single forward pass. If the draft is correct (often
60-80% acceptance rate), the effective generation speed multiplies.

EAGLE is currently the best-performing approach:
- **EAGLE-2**: Used by SGLang, 2-3× speedup at low batch sizes.
- **EAGLE-3** (2025): Up to 6.5× speedup ratio, 1.4× over EAGLE-2, works at batch size 64.
- DeepSeek V3 includes a built-in `NextN` draft head (similar to EAGLE) for speculative
  decoding without a separate draft model.

#### Key Technical Components

**1. Draft Model Execution**

The draft model (small FFN head on top of the target model's hidden states) generates `K`
candidate token sequences as a token tree.

**Requirements**:
- Run draft model efficiently alongside the target model.
- For DeepSeek's built-in `NextN` head, the draft is nearly free (reuses hidden states).
- Draft model weights must be loaded alongside target model weights.

**2. Tree-Based Verification**

The target model verifies multiple draft sequences (token tree) in a single forward pass
using tree attention masking.

**Requirements**:
- Support tree attention mask (non-causal, non-sequential token dependencies).
- This is essentially a prefill-like operation with custom attention mask.
- Extract per-position logprobs for acceptance/rejection (relates to prompt logprobs).

**3. Acceptance/Rejection Sampling**

After verification, apply token-level acceptance criteria:
- Greedy: accept if draft token matches target model's greedy output.
- Stochastic: accept with probability `min(1, p_target / p_draft)` (standard spec sampling).

**Requirements**:
- Per-position logprobs from the verification forward pass (prompt logprobs).
- Efficient KV cache management: accepted tokens stay in cache, rejected tokens
  are discarded, new tokens from the target model are appended.

**4. KV Cache Management**

After accepting `k ≤ K` tokens from the draft:
- Extend the KV cache by `k` positions (the accepted tokens).
- Insert the target model's correction token at position `k+1`.
- Roll back draft model state to position `k+1`.

**Requirements**:
- Efficient KV cache append/rollback operations.
- SpanAttention already supports dynamic allocation; need rollback support.

#### Implementation Plan

1. Implement prompt logprobs (prerequisite — see [RL Roadmap 3.1.1](roadmap_rl_integration.md#311-prompt-logprobs-prefill-stage-log-probabilities)).
2. Implement tree attention mask support in the prefill path.
3. Add draft model loading and execution framework (EAGLE head + lightweight FFN).
4. Implement DeepSeek `NextN` draft head support (model-specific).
5. Implement acceptance/rejection sampling with KV cache rollback.
6. Optimize: overlap draft generation with target model's post-processing.
7. Benchmark: Qwen2.5-72B and DeepSeek V3 single-request latency with EAGLE-2/3.

#### Dependency

- **Prompt logprobs** is a prerequisite for the verification step.
- **CUDA graph full** improves both draft and target model execution.
- **Tree attention** can share prefill infrastructure with chunked prefill.

---

### 3.5 FP4 MoE Fused Kernel

| | |
|---|---|
| **Priority** | P1 |
| **Target** | T2 (DeepSeek on B200) |
| **Effort** | High |
| **Impact** | 1.6× memory reduction, 2-3× MoE compute speedup on B200 |

#### Background

NVIDIA Blackwell (B200) natively supports FP4 (nvfp4) compute via UMMA instructions.
For MoE models like DeepSeek V3 (256 experts), FP4 provides:
- **1.6× memory reduction** vs FP8 → fit more experts per GPU or larger batch.
- **~2× compute throughput** vs FP8 on B200 Tensor Cores.

SGLang achieves **1262 TFLOPS** for FP4 MoE on B200 (3.54× over BF16) through:
1. Kernel fusion: reduce memory passes from 7 to 5.
2. Adaptive grid sizing: maximize SM occupancy at small batch sizes.
3. Blackwell-specific CUTLASS schedules with FP4 warp specialization.

#### Current State

DashInfer has:
- ✅ MoE operator with Expert Parallelism and custom WGMMA kernels (SM90+)
- ✅ FP8 A8W8 MoE kernels (`moe_a8w8_perc_kernel.cu`)
- ❌ No FP4 support
- ❌ No fused MoE kernel (routing + GEMM + activation + GEMM in one kernel)

#### Implementation Plan

1. Add nvfp4 data type support in the type system and weight loading pipeline.
2. Implement FP4 grouped GEMM kernel for B200:
   - Use CUTLASS 3.x with SM100 FP4 schedule.
   - Fuse: dequant → GEMM → activation → GEMM (gate + up + SiLU + down in one kernel).
3. Integrate with MoE operator: fuse token routing dispatch with grouped GEMM.
4. Auto-tune tile sizes and pipeline depth per (GPU type, expert shape).
5. Benchmark: DeepSeek V3 MoE layer throughput vs SGLang on B200.

---

### 3.6 NSA Kernel Fusion (DeepSeek V3.2)

| | |
|---|---|
| **Priority** | P1 |
| **Target** | T2 (DeepSeek V3.2 on B200) |
| **Effort** | High |
| **Impact** | 2-5× attention speedup for long contexts |

#### Background

DeepSeek V3.2 introduced **Native Sparse Attention (NSA)** — a fine-grained sparse
attention mechanism that dynamically selects which attention weights to compute based on
learned importance patterns. Unlike fixed sparse patterns (sliding window, strided), NSA
learns token-level sparsity during training.

NSA consists of three parallel attention branches:

```
Input tokens
  ├── Compressed tokens (block-level pooling → coarse attention)
  ├── Selected tokens (top-k important tokens → fine-grained attention)
  └── Sliding window tokens (local context → exact attention)

→ Gated combination of three branches → Output
```

DeepSeek released optimized NSA kernels via FlashMLA (Sept 2025):
- **Prefill**: up to 640 TFlops on H800
- **Decode**: up to 410 TFlops on H800
- **FP8 KV cache** support
- **SM90 and SM100** support

#### Current State

DashInfer has:
- ✅ Standard MLA attention for DeepSeek V3
- ❌ No NSA support
- ❌ No sparse attention infrastructure

#### Key Technical Components

**1. Block-level compression branch**

Pool consecutive KV tokens into compressed block representations. Requires:
- Pooling kernel (average or learned compression within each block)
- Attention over compressed blocks (standard but with shorter sequence)

**2. Token selection branch**

Select top-k important tokens based on learned scoring. Requires:
- Scoring kernel: compute importance scores for all tokens
- Top-k selection kernel: gather the most important tokens
- Sparse attention over selected tokens

**3. Sliding window branch**

Standard sliding window attention (already common; similar to Mistral's SWA).

**4. Gated combination**

Learned gating network combines outputs from all three branches.

**5. Kernel fusion**

The performance gain comes from **fusing** these branches to minimize memory traffic:
- Single kernel reads Q, K, V once and computes all three branches.
- Intermediate results stay in shared memory / registers.
- Gated combination is fused into the output write.

#### Implementation Plan

1. Add NSA model graph definition in Python (`python/pyhie/allspark/model/`).
2. Implement NSA operator that orchestrates the three branches.
3. Integrate DeepSeek's open-source FlashMLA NSA kernels (SM90/SM100, FP8 KV cache).
4. Implement fused NSA decode kernel for single-token generation.
5. Ensure compatibility with CUDA graph full capture.
6. Benchmark: DeepSeek V3.2 long-context attention throughput vs SGLang.

---

### 3.7 Multi-Token Prediction (MTP)

| | |
|---|---|
| **Priority** | P0 |
| **Target** | T2 (DeepSeek), optional for T1 |
| **Effort** | High |
| **Impact** | 1.5-3× decode throughput / TPOT improvement in low-batch serving |

#### Background

MTP reduces decode loop depth by proposing multiple next tokens per iteration.
For DeepSeek-family models, MTP can be implemented with model-native draft heads
(`NextN`-like path) or through a generic speculative framework.

#### Goal

Deliver production-ready MTP for DeepSeek serving with explicit runtime controls:

- `mtp=1` fallback (disabled)
- `mtp=2/3/4` configurable
- stable acceptance/rollback logic
- quality guardrails (no unacceptable accuracy regression)

#### Implementation Plan

1. Add runtime config options for MTP depth and acceptance policy.
2. Implement MTP decode loop in scheduler/runtime (draft -> verify -> commit).
3. Integrate DeepSeek model-specific draft path (NextN head) first.
4. Add per-request metrics: accepted tokens/step, acceptance rate, rollback count.
5. Benchmark across batch sizes [1, 2, 4, 8, 16] with long-context workloads.

---

### 3.8 DeepSeek Architecture Hardening

| | |
|---|---|
| **Priority** | P0 |
| **Target** | T2 (DeepSeek V3 / V3.2) |
| **Effort** | Medium-High |
| **Impact** | Stronger stability, compatibility, and release confidence for DeepSeek serving |

#### Scope

Turn DeepSeek support from "feature available" to "first-class production path":

- MLA decode path stability under long contexts
- MoE routing/expert-parallel correctness and balance
- NextN/MTP compatibility with scheduler and KV cache lifecycle
- long-CoT generation stability (8K-32K output)
- CUDA graph compatibility checks for DeepSeek-specific operators

#### Implementation Plan

1. Build a dedicated DeepSeek regression matrix (functional + accuracy + perf smoke).
2. Add DeepSeek-focused CI release gates (multi-model coverage and long-context cases).
3. Add operator-level sanity checks for MLA/MoE/NSA interactions.
4. Add strict fallback policy: if MTP/spec path fails, auto-fallback to standard decode.
5. Publish a DeepSeek support profile (recommended runtime knobs and limits).

---

### 3.9 PD Separation + Mooncake Unified KV-Cache

| | |
|---|---|
| **Priority** | P2 |
| **Target** | T2 (distributed DeepSeek serving) |
| **Effort** | Very High |
| **Impact** | Enables clean distributed PD architecture and cross-node KV-cache sharing |

#### Background

DashInfer already has single-node prefill/decode (PD) separation. This is the correct
foundation for future distributed PD split. However, a unified KV-cache layer across PD
stages and nodes is not available yet.

Given current system complexity, this work is intentionally lower priority than:

- single-node performance and model-architecture optimization (MTP/EAGLE/DP Attention/NSA/FP4)
- RL integration blockers (especially prompt logprobs and in-place weight update)

#### Goal

Build a clean path from single-node PD split to distributed PD split with a unified KV-cache
abstraction (Mooncake-based), without destabilizing current single-node serving.

#### Execution Strategy (Phased)

1. **Single-node first (now)**:
   - Keep improving token-level scheduling on top of existing PD split.
   - Avoid aggressive prefill/decode co-running if it hurts latency/throughput.
   - Stabilize KV lifecycle, rollback, and observability in single-node mode.
2. **Distributed foundation (later)**:
   - Introduce Mooncake-backed unified KV-cache interface.
   - Define ownership/lease/eviction semantics across prefill and decode stages.
   - Add transport/metadata protocol for cross-node KV lookup and fetch.
3. **Distributed PD rollout (after validation)**:
   - Enable remote prefill -> decode handoff.
   - Add fault handling and fallback to local-only path.
   - Gate by staged canary and strict regression tests.

#### Acceptance Criteria

- Single-node PD split remains stable and regression-free.
- Distributed mode can be disabled with zero impact on single-node path.
- KV-cache correctness is validated under failover and long-context workloads.

---

## 4. Dependencies and Execution Order

```
                    ┌─────────────────────┐
                    │  Chunked Prefill +   │
                    │  Unified Scheduler   │──────────────────┐
                    └──────────┬──────────┘                  │
                               │                              │
                    ┌──────────▼──────────┐                  │
                    │  CUDA Graph Full     │                  │
                    │  (decode)            │                  │
                    └──────────┬──────────┘                  │
                               │                              │
              ┌────────────────┼────────────────┐            │
              ▼                ▼                 ▼            ▼
    ┌─────────────────┐ ┌──────────────┐ ┌──────────────────────┐
    │ Speculative      │ │ DP Attention │ │ FP4 MoE Fused Kernel │
    │ Decoding (EAGLE) │ │              │ │                      │
    └────────┬────────┘ └──────┬───────┘ └──────────┬───────────┘
             │                 │                     │
             │      ┌─────────▼─────────┐           │
             │      │ MTP (NextN)        │           │
             │      └─────────┬─────────┘           │
             │                │                     │
             │      ┌─────────▼─────────┐           │
             │      │ NSA Kernel Fusion  │           │
             │      │ (DeepSeek V3.2)   │           │
             │      └───────────────────┘           │
             │                                       │
             └───────────────┬───────────────────────┘
                             ▼
                    ┌─────────────────┐
                    │  Target Parity   │
                    └─────────────────┘
```

**Critical path**:
1. Chunked Prefill + Unified Scheduler (unblocks everything)
2. CUDA Graph Full (maximizes decode efficiency)
3. Speculative Decoding / MTP / DP Attention / FP4 MoE (parallel workstreams)
4. NSA Kernel Fusion + DeepSeek hardening (depends on DP Attention for DeepSeek V3.2 serving)
5. PD Separation + Mooncake unified KV-cache (after single-node path and RL P0 blockers are stable)

Note: Speculative Decoding also depends on **Prompt Logprobs** (see
[RL Integration Roadmap](roadmap_rl_integration.md#311-prompt-logprobs-prefill-stage-log-probabilities)).

## 5. Implementation Phases

### Phase 1: Foundation (Target 1 — Dense 72B)

```
┌─ 3.2 Chunked Prefill + Unified Scheduler    [P0, ~6-8 weeks]
│    The single most impactful change for real-world serving.
│    Unblocks mixed prefill/decode workloads.
│
└─ 3.1 CUDA Graph Full (decode)                [P0, ~4-6 weeks]
      Eliminate kernel launch overhead for decode.
      Steps: fix embedding ops → KV indirection buffer → attention graph-safe
```

**Milestone**: Match vLLM/SGLang throughput on Qwen2.5-72B 8×H100 ShareGPT benchmark.

### Phase 2: Acceleration (Target 1+2)

```
┌─ 3.4 Speculative Decoding (EAGLE)            [P0, ~6-8 weeks]
│    2-3× single-request latency.
│    Prerequisite: prompt logprobs (from RL roadmap).
│
├─ 3.7 Multi-Token Prediction (MTP)             [P0, ~4-6 weeks]
│    DeepSeek-first rollout with NextN path.
│    Target: stable acceptance and measurable TPOT gain.
│
└─ 3.3 DP Attention                             [P0, ~6-8 weeks]
      Critical for DeepSeek MoE+MLA decode throughput.
      Can be developed in parallel with spec/MTP workstreams.
```

**Milestone**: Match SGLang latency on Qwen2.5-72B (with EAGLE); DeepSeek V3 decode
throughput within 80% of SGLang on 8×H100/B200 with MTP enabled.

### Phase 3: Blackwell + DeepSeek V3.2 (Target 2)

```
┌─ 3.5 FP4 MoE Fused Kernel                    [P1, ~4-6 weeks]
│    Blackwell-specific. 1.6× memory, 2-3× MoE compute.
│
├─ 3.6 NSA Kernel Fusion                        [P1, ~6-8 weeks]
│     DeepSeek V3.2 specific. Integrate FlashMLA NSA kernels.
│
└─ 3.8 DeepSeek Architecture Hardening          [P0, ~4-6 weeks]
      DeepSeek-specific release gate and stability hardening.
```

**Milestone**: Match SGLang throughput on DeepSeek V3.2 on 8×B200.

### Phase 4: Distributed PD Foundation (Target 2+)

```
┌─ 3.9 PD Separation + Mooncake Unified KV-Cache [P2, ~8-12+ weeks]
│    Lower-priority due to system complexity.
│    Start only after single-node PD split path and RL P0 blockers are stable.
│
└─ Distributed PD rollout with strict fallback and canary gating.
```

**Milestone**: Distributed PD prototype with unified KV-cache and no regression to single-node mode.

## 6. Benchmarking Plan

### Dense 72B Benchmark (Target 1)

```
Model:     Qwen2.5-72B-Instruct (BF16 / FP8)
Hardware:  8×H100-80GB (NVLink)
Workload:  ShareGPT dataset, QPS sweep [1, 5, 10, 20, 50]
           Fixed: input [128, 512, 2048, 8192] × output [128, 512]
Metrics:   Throughput (tok/s), TTFT (p50/p99), ITL (p50/p99)
Compare:   vLLM (latest) / SGLang (latest) / DashInfer

Track after each feature lands:
  Phase 1 (chunked prefill + CUDA graph full): expect ~80-90% parity
  Phase 2 (+ spec decoding):                   expect ~100% parity
```

### DeepSeek V3.2 Benchmark (Target 2)

```
Model:     DeepSeek-V3.2 (FP8 / FP4)
Hardware:  8×B200-192GB (NVLink)
Workload:  ShareGPT + long-CoT (8K-32K output)
Metrics:   Same as above
Compare:   SGLang (latest) / TensorRT-LLM / DashInfer

Track after each feature lands:
  Phase 2 (DP attention):        expect ~60-70% of SGLang
  Phase 2 (+ MTP):               expect notable TPOT reduction at low batch
  Phase 3 (+ FP4 MoE + NSA):    expect ~90-100% parity
  Phase 3 (+ hardening suite):  release-grade DeepSeek stability and compatibility
```

## 7. References

- [vLLM CUDA Graphs Design](https://docs.vllm.ai/en/stable/design/cuda_graphs/)
- [vLLM V1 Architecture](https://vllm-project.github.io/2025/01/27/v1-alpha-release.html)
- [SGLang DeepSeek Optimization](https://datacrunch.io/blog/deepseek-v3-sglang-inference-optimization)
- [EAGLE-3: Scaling up Inference Acceleration](https://arxiv.org/abs/2503.01840)
- [DeepSeek NSA: Native Sparse Attention](https://github.com/deepseek-ai/FlashMLA)
- [FP4 MoE Kernel Engineering on Blackwell](https://huggingface.co/blog/apsys/blackwell-nvfp4-comparison)
- [NVIDIA NVFP4 Blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference)
- [SGLang Expert Parallelism](https://docs.sglang.io/advanced_features/expert_parallelism.html)
- [SGLang Chunked Pipeline Parallelism](https://lmsys.org/blog/2026-01-15-chunked-pipeline/)
