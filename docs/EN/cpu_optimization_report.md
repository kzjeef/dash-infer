# CPU Inference Optimization Report

## Summary

DashInfer CPU inference on x86 (Xeon 8480+ Sapphire Rapids, 112 physical cores) was optimized from **0.51 T/s to 3.34 T/s** (6.5x improvement) through a series of targeted fixes and optimizations.

## Hardware

- **CPU**: 2x Intel Xeon Platinum 8480+ (Sapphire Rapids)
- **Cores**: 112 physical / 224 logical (2 sockets, 56 cores each)
- **ISA**: AVX-512, AMX-BF16, AMX-INT8, AVX-512 FP16
- **Memory**: DDR5, ~150 GB/s aggregate bandwidth
- **Model**: Qwen2.5-7B-Instruct (FP32 weights, 28GB)

## Benchmark Configuration

- **Task**: Long-form generation (79 tokens output)
- **Prompt**: "Write a detailed explanation of how photosynthesis works..."
- **Sampling**: top_k=1 (greedy)
- **Batch size**: 1

## Final Performance

| Mode | Tokens | Time (s) | Generate TPS | vs Baseline |
|------|--------|----------|-------------|-------------|
| Baseline (before optimization) | - | - | 0.51 | 1.0x |
| FP32 (MKL sgemm, 112 threads) | 79 | 35.4 | 2.23 | 4.4x |
| **BF16 (MKL bf16 gemm, 112 threads)** | 79 | 23.7 | **3.34** | **6.5x** |

### Comparison with llama.cpp

| Engine | Precision | Threads | Prefill TPS | Generate TPS |
|--------|-----------|---------|-------------|-------------|
| **DashInfer** | FP32 | 112 | ~53 | 2.23 |
| **DashInfer** | BF16 | 112 | ~53 | 3.34 |
| llama.cpp | BF16 | 56 | 170 | 12.3 |

Gap: ~3.5x for both prefill and generate. The gap is primarily due to llama.cpp's hand-written AVX-512/AMX kernels and optimized memory access patterns in ggml.

---

## Optimization Details

### 1. Fix CPU Correctness (Commits `0f50913`)

**Problem**: CPU inference produced garbled output due to two bugs.

**Bug 1: AllGather operator crash**
- `allgather_op.cpp` set `nranks_=1` but `kernel_launcher` still pointed to `mpi_allgather_launcher` which throws without `ENABLE_MULTINUMA`.
- **Fix**: Simple memcpy lambda for single-rank CPU mode.

**Bug 2: oneDNN `tag::any` weight reorder corruption**
- `GemmOpCPU::Reshape()` used `tag::any` which triggered oneDNN internal weight reorder, producing incorrect results for certain matrix shapes.
- Verified via manual dot product: QKV GEMM matched but output.dense (21% error) and FFN (58% error) diverged.
- **Fix**: Use explicit strides matching InitV2 layout.

**Additional**: `#ifdef ENABLE_CUDA` guards for `generate_op.h/cpp`, `TracerLog` replacement in `model.cpp`, SpanAttention guards in `mla_attn_op.cpp`, removed hardcoded "CPU not supported" error.

### 2. Replace oneDNN with MKL (Commit `7fae7d4`)

**Problem**: oneDNN matmul had correctness bugs and high overhead for element-wise ops.

**Changes**:

| Component | Before (oneDNN) | After (MKL/parallel_for) | Code Change |
|-----------|-----------------|--------------------------|-------------|
| GEMM (Linear) | `dnnl::matmul` primitive | `cblas_sgemm` / `cblas_gemm_bf16bf16f32` | -140 lines |
| Binary (SiLU*gate) | `dnnl::binary` + `dnnl::eltwise` | `cpu::parallel_for` + simple loop | -106 lines |
| Unary (SiLU/ReLU/GELU) | `dnnl::eltwise_forward` | `cpu::parallel_for` + simple loop | -14 lines |
| **Total** | 651 lines | 345 lines | **-306 lines** |

**Impact**: First-request latency improved 4.3x (42.6s → 10s) due to elimination of oneDNN primitive JIT and weight reorder.

### 3. BF16 GEMM Support (Commits `dc6f803`)

**Implementation**: MKL `cblas_gemm_bf16bf16f32` for BF16 input × BF16 weights → FP32 output.

**Key optimizations**:
- Weight conversion (FP32→BF16) done once in `InitV2()`, cached in `w_bf16_cache_`
- Input conversion buffer pre-allocated in `Reshape()`, reused in `Forward()`
- Auto-detection of CPU BF16 support (`avx512_bf16` / `amx_bf16`)

**Before optimization**: BF16 was 10% *slower* than FP32 (0.78 vs 0.87 T/s) due to per-call weight conversion overhead.
**After optimization**: BF16 is 1.50x faster than FP32 (3.34 vs 2.23 T/s).

### 4. Auto Thread Detection (Commit `6480011`)

**Problem**: DashInfer defaulted to OMP's default thread count (often 1), leaving 111 of 112 cores idle.

**Fix**: Set `OMP_NUM_THREADS` to physical core count (logical CPUs / 2) at Python package import time, before any OpenMP runtime is initialized.

```python
# In __init__.py, before any C++ imports
import os
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() // 2)
```

Also added C++ fallback in `as_engine.cpp` using `std::thread::hardware_concurrency()`.

**Impact**: 4.6x improvement (0.90 → 4.18 T/s with manual OMP_NUM_THREADS=112).

### 5. Skip CPU Warmup (Commit `6480011`)

**Problem**: DashInfer warmup sends a 254-token fake request through all 28 layers, taking ~60s on CPU. This is designed for GPU memory pre-allocation and is unnecessary on CPU.

**Fix**: Skip warmup when `DeviceType::CPU`. Override with `ALLSPARK_FORCE_WARMUP=1`.

**Impact**: Startup time reduced from ~100s to ~37s (weight loading only).

### 6. MKL Upgrade 2022 → 2025.3.1 (Commit `1d24cfb`)

**Problem**: MKL 2022.0.2 predates Sapphire Rapids optimizations.

**Fix**: Replaced `third_party/mkl_2022.0.2.tar.gz` with MKL 2025.3.1 (from pip `mkl-static`). Updated `cmake/mkl.cmake` to handle newer MKL without bundled CMake config files.

**Impact**: +28% BF16 generate throughput (2.54 → 3.25 T/s).

### 7. GEMV vs GEMM Analysis (Commit `161d319`)

**Hypothesis**: `cblas_sgemv` should be faster than `cblas_sgemm` for decode (m=1).

**Micro-benchmark results (m=1, k=3584, n=4608, 112 threads)**:

| Function | Time (us) | GFLOPS | Notes |
|----------|----------|--------|-------|
| `cblas_sgemm` FP32 | 105 | 316 | Parallelizes over N dimension |
| `cblas_hgemm` FP16 | 163 | 203 | AVX-512 FP16 |
| `cblas_gemm_bf16bf16f32` | 186 | 177 | AMX BF16 |
| `cblas_sgemv` FP32 | 678 | 49 | Poor multi-thread reduction |

**Conclusion**: `sgemm` is 6.5x faster than `sgemv` for m=1 with 112 threads. MKL 2025's JIT already optimizes sgemm for small m. The sgemv "optimization" was reverted.

---

## Bottleneck Analysis

### Decode Phase (m=1)

Per-token time breakdown (TPOT = 225ms, 28 layers):

```
forward_time: 222ms (98.7%)
gen_forward:  2.3ms (1.0%)
reshape:      0.01ms
alloc:        0.07ms
```

The 222ms forward is dominated by **memory bandwidth** (reading ~9.4 GB of BF16 weights per token):

```
Theoretical minimum: 9.4 GB / 150 GB/s = 63ms
Actual: 222ms → 28% bandwidth utilization
llama.cpp: ~81ms → ~77% bandwidth utilization
```

### Remaining Gap vs llama.cpp

| Factor | Impact | Solution |
|--------|--------|----------|
| Memory access pattern | 2-3x | llama.cpp uses contiguous weight layout optimized for streaming |
| GEMM kernel overhead | 1.2-1.5x | libxsmm JIT or hand-written AVX-512 kernels |
| Weight quantization | 2x | INT8/INT4 reduces memory bandwidth requirement |
| NUMA optimization | 1.1-1.2x | Weight placement aware of NUMA topology |

---

## Files Modified

| File | Change |
|------|--------|
| `csrc/common/as_engine.cpp` | CPU warmup skip, auto thread detection |
| `csrc/core/operator/general/gemm/gemm_op_cpu.cpp` | MKL sgemm/bf16 GEMM, weight caching |
| `csrc/core/operator/general/gemm/gemm_op_cpu.h` | BF16 cache members |
| `csrc/core/operator/general/gemm/gemm_op_x86_spr.cpp` | Simplified BF16 path via GemmOpCPU |
| `csrc/core/operator/general/binary/binary_op.cpp` | parallel_for (SWIGLU/ADD/MUL) |
| `csrc/core/operator/general/unary/unary_op.cpp` | parallel_for (SiLU/ReLU/GELU/TANH) |
| `csrc/core/operator/nccl/allgather/allgather_op.cpp` | Single-rank CPU memcpy fix |
| `csrc/core/operator/nccl/allreduce/allreduce_op.cpp` | Single-rank CPU copy fix |
| `csrc/core/operator/generate_opt/generate/generate_op.h/cpp` | ENABLE_CUDA guards |
| `csrc/core/operator/generate_opt/mla_attn/mla_attn_op.cpp` | ENABLE_SPAN_ATTENTION guard |
| `csrc/core/model/model.cpp` | TracerLog for CPU compatibility |
| `cmake/mkl.cmake` | MKL 2025 support without CMake config |
| `third_party/mkl_2022.0.2.tar.gz` | Upgraded to MKL 2025.3.1 |
| `third_party/dnnl.tar.bz2` | Upgraded to oneDNN v3.7 |
| `python/pyhie/allspark/__init__.py` | Auto OMP_NUM_THREADS |
| `python/pyhie/allspark/runtime_config.py` | BF16 auto-detection, CPU thread setup |
| `python/pyhie/allspark/model_loader.py` | BF16 auto-enable for CPU |
| `docs/EN/cpu_correctness_bug_report.md` | Bug report documentation |
