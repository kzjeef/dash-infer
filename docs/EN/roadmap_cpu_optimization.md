# DashInfer CPU Performance Optimization Plan

## Current Status

**Platform**: Intel Xeon Platinum 8480+ (Sapphire Rapids, 2x56 cores, AMX/AVX-512 BF16)
**Model**: Qwen2.5-7B-Instruct, FP32, 28 layers
**Baseline Performance** (batch=1):
- Prefill: ~7.2 tokens/s (26 tokens)
- Decode: ~2.86 tokens/s (TPOT ~350ms)
- Weight loading: 29GB FP32 in ~37s (785 MB/s)

## Issue Summary

### P0: CPU Correctness Bug (Blocking)

**Symptom**: CPU inference produces complete garbage output (random multilingual tokens).
HuggingFace Transformers reference produces correct output with the same model and FP32.

**Root cause**: Under active investigation with per-operator tensor dumps.

Key findings from layer-by-layer comparison (DashInfer CPU vs HuggingFace FP32):
- **Embedding**: Values look reasonable (mean~0.01, no NaN/Inf) ✓
- **LayerNorm (RMSNorm)**: Output in correct range ✓
- **GEMM (QKV projection)**: Statistics similar to HF ✓
- **Rotary (RoPE)**: Output changes only K/V portions as expected ✓
- **DecOptMQA (GQA attention)**: Output in reasonable range ✓
- **FFN MLP pipeline**: Gate/Up/Mul/Down all produce finite values ✓
- **Layer 27 output**: abs_mean=4.47, min=-170, max=143 (growing magnitudes) ⚠️
- **First generated token**: Token 124262 (garbage) vs HF expected token 17 ("2") ✗

The divergence is **subtle and accumulative** across 28 layers:
- Each layer's output is finite and in a plausible range
- But small errors (likely in attention or residual connections) compound
- By the last layer, the hidden state has drifted enough to produce wrong logits

**Detailed per-operator tensor dump results** (layer 0, prefill 254 tokens):

| Operator | abs_mean | min | max | Status |
|----------|----------|-----|-----|--------|
| Embedding | 0.012 | -0.146 | 0.118 | ✓ reasonable |
| LayerNorm (L0.attn) | 0.214 | -2.58 | 2.93 | ✓ normalized |
| Gemm (QKV, L0) | 1.939 | -163 | 171 | ✓ large but valid |
| Rotary (L0) | 1.940 | -163 | 171 | ✓ similar to QKV |
| DecOptMQA (L0.attn) | 0.150 | -0.82 | 1.28 | ✓ attention output |
| Gemm (attn_out, L0) | 0.110 | -2.14 | 1.64 | ✓ with residual ADD |
| Gemm (gate, L0, SiLU fused) | 0.114 | -0.278 | 4.21 | ✓ post-SiLU |
| Binary (ffn.mul, L0) | 0.027 | -18.6 | 1.41 | ✓ gate*up |
| Gemm (ffn_out, L0, +residual) | 0.355 | -5.83 | 3.57 | ✓ |

After 28 layers (layer 27):
| Operator | abs_mean | min | max |
|----------|----------|-----|-----|
| Gemm (attn_out, L27) | 4.47 | -170 | 143 |
| Gemm (ffn_out, L27) | TBD | | |

- No NaN or Inf detected at any layer
- Values are finite but magnitudes grow across layers (abs_mean: 0.36 → 4.47)
- First generated token: **124262** (garbage) vs HF expected **17** ("2")

**Most likely root causes** (in priority order):
1. **Cumulative numerical drift** across 28 layers from small per-layer errors in
   attention, RoPE, or GEMM that compound to produce wrong final logits
2. **GQA attention (DecOptMQA)**: KV cache or score computation may have subtle
   bug for GQA (4 KV heads vs 28 Q heads) that doesn't produce NaN but shifts
   attention weights slightly each layer
3. **Fused GEMM+ADD residual path**: oneDNN post-op binary ADD might have
   numerical differences from naive add
4. All CPU C++ test cases are disabled (`#if 0`), indicating long-term neglect

**ROOT CAUSE IDENTIFIED**: The **EmbeddingT5 operator** on CPU produces wrong output.

Definitive proof (same token 151644 `<|im_start|>`):
- HF embedding[151644,:5] = `[0.000124, -0.000091, 0.000121, -0.000141, -0.000041]`
- DI embedding output first5 = `[-0.0155, -0.0041, 0.0148, 0.0015, 0.0226]`
- HF embedding[**0**,:5] = `[-0.0155, -0.0041, 0.0148, 0.0015, 0.0226]` ← EXACT MATCH

DashInfer's embedding operator is **reading from row 0** instead of the actual token ID!
This means all tokens get wrong embeddings, producing garbage through all 28 layers.

**UPDATED**: Embedding is CORRECT. The true divergence point is:
- **Layer0 RMSNorm**: EXACT match with HuggingFace ✓
- **QKV GEMM**: ~10% divergence from HF (first point of error)
- **GQA Attention output**: Completely wrong (softmax amplifies the QKV error)

The ~10% QKV GEMM error is likely caused by **GROUP_VSPLIT weight splitting** when
`multigpu_mode=1` (always enabled). For single-rank CPU, the weight splitting may
reorganize the QKV weight layout in a way that doesn't match the attention kernel's
expected format.

**UPDATED: multigpu_mode is NOT the root cause** — tested with multigpu_mode=0, still garbage output.

The true root cause is in the **oneDNN GEMM operator** (`GemmOpCPU` / `GemmOpSpr`).
The QKV GEMM (first GEMM in layer 0) produces ~10% different values from PyTorch:
- DI: matmul_only = [0.1347, 0.0126, -0.0099, 0.0811, 0.0137]
- HF: matmul_only = [0.0849, 0.1156, 0.0516, 0.0748, 0.0706]
These differences are too large for numerical rounding and suggest the weight matrix
is being read with incorrect strides/transposition by oneDNN, OR the weight reorder
during oneDNN Reshape corrupts the data.

**Recommended investigation**:
1. Check `GemmOpCPU::Reshape()` — the oneDNN weight reorder may corrupt weight layout
2. Check `GemmOpSpr::InitV2()` — for FLOAT32 path, it delegates to GemmOpCPU
3. Verify the weight tensor dimensions (k_, n_, lda_, ldb_, transB_) match expected
4. Compare raw weight bytes from asparam with HF model weights
5. Try disabling oneDNN weight reorder to see if values match

**Compilation fixes already applied** (6 issues):
1. `generate_op.h`: `cudaStream_t` missing CUDA guard
2. `model.cpp`: `Tracer` → `TracerLog` (CUDA-only class used on CPU)
3. `generate_op.cpp`: `EnqueueSampleD2H`/`CompleteSampleD2H` need CUDA guard
4. `mla_attn_op.cpp`: `virtual_k_cache` needs `ENABLE_SPAN_ATTENTION` guard
5. `as_engine.cpp:314`: CPU inference hardcoded-disabled, re-enabled
6. `allgather_op.cpp`/`allreduce_op.cpp`: Single-rank CPU lacks fallback

### P1: Hotspot Analysis (Per-layer, Prefill, 26 tokens)

| Priority | Operator | Avg Time | % Total | Issue |
|----------|----------|----------|---------|-------|
| **H1** | Binary (SiLU*gate) | 81.9ms | **31.8%** | Element-wise op extremely slow; no SIMD optimization |
| **H2** | LayerNormNoBeta | 37.9ms | **29.4%** | RMSNorm lacks fused kernel; excessive memory traffic |
| **H3** | Gemm (DNNL/MKL) | 12.6ms avg | **23.3%** | FP32 GEMM; not using AMX BF16 |
| H4 | Rotary (RoPE) | 20.9ms | 8.1% | First-layer spike (75ms); freq table recomputed |
| H5 | DecOptMQA | 17.7ms | 6.9% | GQA attention; batched GEMM overhead |

## Optimization Plan

### Phase 1: Fix Correctness (P0)

- [ ] **1.1** Enable and fix CPU C++ unit tests (`tests/cpp/operator/cpu/`)
- [ ] **1.2** Add per-operator output validation (compare with HuggingFace layer-by-layer)
- [ ] **1.3** Investigate GQA attention (DecOptMQA) KV cache indexing
- [ ] **1.4** Verify RoPE frequency computation matches HuggingFace
- [ ] **1.5** Verify weight loading/serialization produces correct FP32 values

### Phase 2: Quick Wins (~2-4x speedup expected)

- [ ] **2.1 AMX BF16 GEMM** (`matmul_precision=medium_bf16`)
  - Use `GemmOpSpr` BF16 path with oneDNN AMX backend
  - Expected: GEMM 2-4x faster (Sapphire Rapids AMX_BF16 throughput)
  - Risk: slight accuracy drop (BF16 mantissa = 7 bits vs FP32 = 23 bits)
  - Files: `csrc/core/operator/general/gemm/gemm_op_x86_spr.cpp`

- [ ] **2.2 Binary Op SIMD Optimization**
  - Current: naive element-wise loop, 81.9ms per call
  - Target: AVX-512 vectorized SiLU * gate_proj fusion
  - Expected: 5-10x faster for element-wise ops
  - Files: `csrc/core/kernel/cpu/`, `csrc/core/operator/general/binary/`

- [ ] **2.3 LayerNorm SIMD Optimization**
  - Current: 37.9ms per RMSNorm call
  - Target: AVX-512 vectorized RMSNorm with fused residual add
  - Expected: 3-5x faster
  - Files: `csrc/core/kernel/cpu/layernorm.cpp`

### Phase 3: Architecture-Level Optimizations

- [ ] **3.1 Fused Attention Kernel**
  - Fuse Q*K^T + Softmax + AV into single tiled kernel (FlashAttention-style for CPU)
  - Already partially implemented: `cpu_ctx_single_famha` / `cpu_ctx_single_famqa`
  - Enable via AVX-512 flash attention path for both prefill and decode

- [ ] **3.2 RoPE Frequency Table Caching**
  - Pre-compute and cache cos/sin tables instead of recomputing per layer
  - Fix first-layer 75ms spike (subsequent layers are 2.9ms)
  - Files: `csrc/core/operator/general/rotary/rotary_op.cpp`

- [ ] **3.3 Operator Fusion**
  - Fuse LayerNorm + GEMM (reduce memory bandwidth)
  - Fuse GEMM + SiLU + elementwise multiply (MLP fusion)
  - Files: Model graph definition in Python + new fused operators

- [ ] **3.4 NUMA-Aware Thread Management**
  - Current: 224 threads without NUMA awareness
  - Target: Bind threads to NUMA nodes, use local memory
  - Use `ENABLE_MULTINUMA=ON` for multi-socket parallelism

### Phase 4: Weight Optimization

- [ ] **4.1 Weight-Only INT8 Quantization**
  - Reduce model size from 29GB to ~7.5GB
  - Faster weight loading (4x less I/O)
  - INT8 dequantize-on-the-fly during GEMM

- [ ] **4.2 Memory-Mapped Weight Loading**
  - Use mmap instead of fread for weight loading
  - Set `AS_WEIGHT_LOAD_FROM_MMAP=on`
  - Reduces startup memory peak

## Performance Targets

| Metric | Current | Phase 2 Target | Phase 3 Target |
|--------|---------|----------------|----------------|
| Prefill TPS (26 tok) | 7.2 | ~25 | ~50 |
| Decode TPS (batch=1) | 2.86 | ~8 | ~15 |
| TPOT (ms) | 350 | ~125 | ~67 |
| Weight Load (s) | 37 | 37 | 10 (mmap) |

## References

- Hotspot data from: operator-level timing via `[CPU-DBG]` logging
- System: Intel Xeon 8480+ (SPR), 2 sockets, 112 cores, 2TB RAM
- Build: `AS_PLATFORM=x86`, `AS_CXX11_ABI=ON`, `ENABLE_DNNL=ON`, `ENABLE_SPAN_ATTENTION=OFF`
