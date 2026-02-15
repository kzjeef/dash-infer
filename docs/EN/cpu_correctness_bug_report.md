# CPU Inference Correctness Bug: Root Cause Analysis and Fix

## Summary

CPU inference on x86 platform produced completely garbled output (e.g., `טיולбереж Geological Sandwich炼 Guardian` instead of coherent text). Two independent bugs were identified and fixed. The core correctness issue was traced to **oneDNN's `tag::any` weight reorder producing incorrect results for certain matrix shapes** in the GEMM operator.

## Environment

- Model: Qwen2.5-7B-Instruct (FP32)
- Platform: x86_64 Linux, `AS_PLATFORM="x86"`
- oneDNN: v3.2.0 (bundled), later upgraded to v3.7.0 — both exhibited the same bug
- Build: `AS_CXX11_ABI="ON"`, `ENABLE_SPAN_ATTENTION=OFF`, `ENABLE_MULTINUMA=OFF`

## Bug 1: AllGather Operator Crash (Blocker)

### Symptom

Engine stuck at `Running: 0 Pending: 0` during warmup. The prefill threw `std::exception` with garbled `what()` message, indicating memory corruption from an unhandled throw:

```
E model.cpp:866] runDecoderContext() Failed: ��Hw7, request_id = warmup_request_0
E allgather_op.cpp:88] Multi-NUMA codes are not compiled
```

### Root Cause

In `allgather_op.cpp`, the `Init()` function correctly set `nranks_ = 1` and `rank_id_ = 0` for single-rank CPU mode (when `ENABLE_MULTINUMA` is not defined). However, it still assigned `kernel_launcher = mpi_allgather_launcher`, which unconditionally throws when `ENABLE_MULTINUMA` is off:

```cpp
// allgather_op.cpp line 87-89 (BEFORE fix)
#else
  LOG(ERROR) << "Multi-NUMA codes are not compiled" << std::endl;
  AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);  // <-- crash here
#endif
```

### Fix

Replace `mpi_allgather_launcher` with a simple memcpy lambda for single-rank CPU mode:

```cpp
case DeviceType::CPU: {
#ifdef ENABLE_MULTINUMA
      kernel_launcher = mpi_allgather_launcher;
      // ... multinuma setup ...
#else
      // Single-rank CPU: use simple memcpy instead of MPI launcher
      kernel_launcher = [](DataType dtype, void* out, void* in,
                           void* /*tmp_data*/, int count, int /*batch_size*/,
                           int /*hidden_size*/, int /*nranks*/,
                           const DeviceContext* /*ctx*/) {
        if (in != out) {
          memcpy(out, in, count * SizeofType(dtype));
        }
      };
      nranks_ = 1;
      rank_id_ = 0;
#endif
      break;
    }
```

A similar defensive fix was applied to `allreduce_op.cpp` `Forward()` to ensure `memcpy(out, in, ...)` when `nranks_ == 1` and `in != out`.

### Files Changed

- `csrc/core/operator/nccl/allgather/allgather_op.cpp`
- `csrc/core/operator/nccl/allreduce/allreduce_op.cpp`

---

## Bug 2: oneDNN Weight Reorder Corruption (Core Correctness Issue)

### Symptom

After fixing Bug 1, inference ran to completion but produced completely garbled output:

```
Q: What is 2+3?
A: טיולбереж Geological Sandwich炼 Guardian-it倾向àng Ownership Open...
```

### Debugging Process

**Step 1: Layer-by-layer comparison with HuggingFace reference**

Confirmed that:
- ✅ Embedding output: exact match with HuggingFace
- ✅ LayerNorm output: exact match with HuggingFace
- ❌ First QKV GEMM output: ~10% divergence

This narrowed the problem to the GEMM operator.

**Step 2: Weight verification**

Dumped weight values in `GemmOpCPU::InitV2` before any oneDNN processing. Values matched HuggingFace exactly — the weights were loaded correctly.

**Step 3: Manual dot product verification**

Added code to `GemmOpCPU::Forward` that computed the first output element manually using the original (pre-reorder) weights, and compared with oneDNN's output:

```
[GEMM-VERIFY] op=decoder.layer.0.attention.self (QKV)
  oneDNN out[0]=0.482902  manual out[0]=0.482902  ← MATCH ✓

[GEMM-VERIFY] op=decoder.layer.0.attention.output.dense
  oneDNN out[0]=-0.0899   manual out[0]=-0.0744   ← MISMATCH ✗ (21% error)

[GEMM-VERIFY] op=decoder.layer.0.ffn.intermediate.dense
  oneDNN out[0]=-0.1428   manual out[0]=-0.3442   ← MISMATCH ✗ (58% error)
```

This proved that:
1. The original weight data was correct
2. oneDNN's weight reorder produced corrupted weights for certain matrix shapes
3. The first GEMM (k=3584, n=4608) was unaffected, but the second (k=3584, n=3584) and third (k=3584, n=18944) were corrupted

### Root Cause

In `gemm_op_cpu.cpp` `Reshape()`, the weight memory descriptor used `tag::any`:

```cpp
// BEFORE fix — Reshape()
memory::desc w_desc({k_, n_}, dt, tag::any);
```

`tag::any` tells oneDNN to choose its preferred internal blocked format for weights, then reorder from the user's row/column-major format to that internal format. This reorder was producing incorrect weight data for certain matrix dimensions.

Meanwhile, `InitV2()` correctly described the weight layout with explicit strides:

```cpp
// InitV2() — always correct
memory::dims w_stride = transB_ ? memory::dims{1, ldb_} : memory::dims{ldb_, 1};
memory::desc w_desc({k_, n_}, dt, w_stride);
```

The mismatch between the `tag::any` in Reshape and the explicit strides in InitV2 triggered oneDNN's internal weight reorder, which was buggy for certain shapes.

**Note:** This bug was reproduced with both oneDNN v3.2.0 (original) and v3.7.0 (upgraded), confirming it is not version-specific.

### Fix

Use explicit strides in `Reshape()` matching those in `InitV2()`, preventing the weight reorder entirely:

```cpp
// AFTER fix — Reshape()
memory::dims w_stride =
    transB_ ? memory::dims{1, ldb_} : memory::dims{ldb_, 1};
memory::desc w_desc({k_, n_}, dt, w_stride);
```

### Files Changed

- `csrc/core/operator/general/gemm/gemm_op_cpu.cpp`

---

## Additional Fixes (Compilation)

Several `#ifdef` guards were added to enable CPU compilation without CUDA/SpanAttention:

| File | Issue |
|------|-------|
| `csrc/core/operator/generate_opt/generate/generate_op.h` | `cudaStream_t` used without `ENABLE_CUDA` guard |
| `csrc/core/operator/generate_opt/generate/generate_op.cpp` | Same as above |
| `csrc/core/model/model.cpp` | `Tracer` class is CUDA-only; replaced with `TracerLog` |
| `csrc/core/operator/generate_opt/mla_attn/mla_attn_op.cpp` | `virtual_k_cache` etc. require `ENABLE_SPAN_ATTENTION` |
| `csrc/common/as_engine.cpp` | Hardcoded `CPU inference is NOT support` error removed |

---

## Verification

After all fixes, CPU inference produces correct results:

```
Q: What is 2+3?
A: 2 + 3 equals 5.  (9 tokens, 4.2s)

Q: Capital of France?
A: The capital of France is Paris.  (8 tokens, 3.4s)

Q: What is the chemical formula of water?
A: The chemical formula of water is H2O.  (11 tokens, 5.0s)
```

Performance: ~2.3 tokens/sec for Qwen2.5-7B FP32 on CPU (single-threaded prefill, expected for unoptimized CPU path).

## Recommendations

1. **Add CPU CI tests** — A simple 3-question correctness test should be added to prevent regressions.
2. **Investigate oneDNN reorder bug** — File an upstream issue with oneDNN reproducer for the weight reorder corruption with specific matrix shapes.
3. **Enable BF16 matmul** — The current test machine (Xeon 8480+, Sapphire Rapids) supports `avx512_bf16` and `amx_bf16`. The `GemmOpSpr` path already supports BF16 weights via oneDNN or Intel intrinsics (`ig_bgemm_f32bf16f32`). Setting `matmul_precision="medium_bf16"` would halve weight memory and improve GEMM throughput 2-4x via AMX acceleration. Note: the explicit-strides fix in `GemmOpCPU::Reshape` also applies to the oneDNN BF16 path (which calls `GemmOpCPU::Reshape`), so the fix is automatically inherited.
4. **Performance optimization** — Current CPU path is naive single-core. OMP threading, NUMA-awareness, and KV cache optimization would significantly improve performance.
