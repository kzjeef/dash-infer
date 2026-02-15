# CUDA Graph Implementation Status

> Date: 2026-02-15
> Branch: `eval_auto_reg`
> Machine: 8× NVIDIA H200 (SM90, 143GB HBM each)

---

## 1. What Was Implemented

### 1.1 `AsExecuteMode` Enum

Replaced boolean `enable_cuda_graph` with a proper enum, extensible for future modes.

```cpp
// csrc/interface/allspark.h
enum class AsExecuteMode {
  Eager = 0,       // No CUDA graph, all ops run eagerly
  CudaGraph = 1,   // Piecewise CUDA graph with lazy capture
};
```

- **Default**: `CudaGraph` for CUDA platform
- **Env override**: `ALLSPARK_EXECUTE_MODE=eager|graph`
- **Python API**: `cfg.execute_mode = AsExecuteMode.CudaGraph`
- **Builder**: `cfg_builder.execute_mode(AsExecuteMode.CudaGraph)`

**Files**:
- `csrc/interface/allspark.h` — enum + `AsModelConfig.execute_mode` + `AsModelConfig.cuda_graph_batch_sizes`
- `csrc/core/model/model.h` — `execute_mode_` member replacing `cuda_graph_enabled_`
- `csrc/core/model/model.cpp` — constructor reads env var `ALLSPARK_EXECUTE_MODE`
- `python/allspark_binding.cpp` — `py::enum_<AsExecuteMode>` + `def_readwrite("execute_mode")`
- `python/dashinfer/allspark/runtime_config.py` — `execute_mode()` builder method

### 1.2 Batch Size Bucketing

Graph plans are cached by **power-of-2 bucket** (1, 2, 4, 8, 16, ...) instead of exact batch size. This prevents graph invalidation when batch size changes as requests arrive/finish.

```
batch_size=3 → bucket=4 → reuse graph captured for 4
batch_size=5 → bucket=8 → reuse graph captured for 8
```

**Key change**: `StopRequest` no longer calls `CudaGraphClear()`. Graphs persist across batch size changes.

**Files**: `csrc/core/model/model.cpp` — decode path, `CudaGraphRunPiecewise`, gen_graph lookup all use `CudaGraphBatchBucket(batch_size)`.

### 1.3 User-Configurable Capture Batch Sizes

```python
cfg_builder.cuda_graph_batch_sizes([1, 2, 4, 8, 16, 32])
```

Empty list = auto (power-of-2 up to `engine_max_batch`). Used by `CudaGraphPreCapture()`.

**Files**: `csrc/interface/allspark.h`, `csrc/common/as_engine.h`, `csrc/common/as_engine.cpp`, `python/allspark_binding.cpp`, `python/dashinfer/allspark/runtime_config.py`

### 1.4 Warmup Graph Clearing

Graphs captured during engine warmup have stale buffer state (zeros input, warmup KV cache). They produce garbage if replayed with real inference data.

**Fix**: After `WarmupModelInternal_`, all warmup-captured graphs are cleared. First real decode step triggers fresh lazy capture.

```cpp
// csrc/common/as_engine.cpp — WarmupModel()
for (int i = 0; i < nranks_; ++i) {
    workers_decode_[i]->CudaGraphClear();
}
```

### 1.5 NCCL Synchronize Removal

Removed `cudaStreamSynchronize` from NCCL AllReduce and AllGather ops. This was unnecessary (same-stream ops are already ordered by CUDA) and blocked CUDA Graph capture.

```cpp
// BEFORE (allreduce_op.cpp):
ncclAllReduce(..., stream);
ctx_->Synchronize();  // cudaStreamSynchronize — blocks host, breaks graph capture

// AFTER:
ncclAllReduce(..., stream);
// No sync needed: same-stream ordering guarantees correctness
```

**Files**:
- `csrc/core/operator/nccl/allreduce/allreduce_op.cpp` — removed `ctx_->Synchronize()`
- `csrc/core/operator/nccl/allgather/allgather_op.cpp` — removed pre/post `Synchronize()`

### 1.6 NCCL Graph-Safe (IsGraphUnsafe removed)

NCCL AllReduce and AllGather are now marked graph-safe (can be captured in CUDA graphs).

```cpp
// allreduce_op.h / allgather_op.h
// bool IsGraphUnsafe() const override { return true; }  // COMMENTED OUT
```

### 1.7 MLA Attention Graph-Safe

MLA attention operator also marked graph-safe for models that use it (DeepSeek, etc.).

```cpp
// mla_attn_op.h
// bool IsGraphUnsafe() const override { return true; }  // COMMENTED OUT
```

### 1.8 SpinBarrier for Cross-Rank Graph Launch

For multi-GPU, NCCL graph replay requires all ranks to `cudaGraphLaunch` simultaneously. Added `SpinBarrier` synchronization.

```cpp
// csrc/core/model/model.h
class SpinBarrier {
  explicit SpinBarrier(int num_threads);
  void wait();  // spin until all threads arrive
};
```

Barrier is created by the engine after model build (`as_engine.cpp`) and shared across all rank's models. Used at two points:
1. **Before graph capture** in `CudaGraphBuildPlan` — ensures all ranks enter capture simultaneously
2. **Before graph launch** in `CudaGraphRunPiecewise` — ensures synchronized replay

**Files**:
- `csrc/core/model/model.h` — `SpinBarrier` class, `graph_launch_barrier_` member, `SetGraphLaunchBarrier()`
- `csrc/core/model/model.cpp` — barrier wait before capture and launch
- `csrc/common/as_engine.cpp` — barrier creation after `BuildModel`
- `csrc/common/engine_worker.h` — `SetGraphLaunchBarrier()` passthrough

### 1.9 CudaGraphPreCapture Infrastructure

Pre-capture method for future use (warmup-time graph pre-capture).

```cpp
AsStatus CudaGraphPreCapture(int max_batch, const std::vector<int>& user_batch_sizes = {});
```

Currently NOT called during warmup (warmup graphs are stale). The lazy capture approach works: first real decode at each bucket triggers capture. Pre-capture can be enabled once the stale-state issue is resolved.

### 1.10 Flash-Attention v2.8.3 Patch

Created reusable script `fix_flash_attn_v283.sh` for patching flash-attention v2.8.3 for standalone (non-PyTorch) builds. Run after `build.sh` fails on flash-attn patch.

Also fixed `flashv2.h` to use `flash::` namespace prefix for v2.8.3's `FLASH_NAMESPACE` wrapping.

**Files**:
- `fix_flash_attn_v283.sh` — reusable patch script
- `csrc/core/kernel/cuda/flashv2/flashv2.h` — `flash::Flash_fwd_params`
- `csrc/core/kernel/cuda/flashv2/flash_attention_operator.cu` — `flash::run_mha_fwd`

---

## 2. Benchmark Results (72B Qwen2.5, 2× H200)

### 2.1 DashInfer vs sglang Throughput

| Config | DashInfer (Eager) | sglang | Ratio |
|--------|------------------|--------|-------|
| Single, 256 out | 36.8 t/s | 46.3 t/s | 0.79× |
| Single, 512 out | 37.0 t/s | 46.2 t/s | 0.80× |
| 4 concurrent, 256 out | 72.9 t/s | 180.3 t/s | 0.40× |
| 8 concurrent, 256 out | 140.5 t/s | 357.0 t/s | 0.39× |

### 2.2 CUDA Graph Impact (Single GPU, 7B)

| Config | Eager | CudaGraph (2-seg) | Improvement |
|--------|-------|-------------------|-------------|
| Single request | 36.8 t/s | 40.6 t/s | **+10%** |
| Various batch (1-12) | — | 7/7 correct | **All pass** |

### 2.3 Graph Plan Structure

| Setup | Segments | Eager Ops | Main Graph Size |
|-------|----------|-----------|-----------------|
| Single GPU | **2** | 2 (embedding) | 340 ops (whole decoder) |
| 2-GPU (NCCL graph-safe) | **2** | 2 (embedding) | 340 ops (whole decoder + NCCL) |
| 2-GPU (NCCL graph-unsafe) | 57 | 60 | ~5 ops per segment |

---

## 3. Known Issues

### 3.1 Multi-GPU NCCL Graph Replay Hangs

**Status**: Capture succeeds, replay hangs.

Both ranks successfully capture graphs with NCCL inside (2 segments, 340 ops). But `cudaGraphLaunch` hangs because the NCCL collective inside the graph requires all ranks to launch simultaneously.

The `SpinBarrier` was added to synchronize launch, but the hang persists. Possible causes:
- SpinBarrier timing not tight enough (spin-wait vs NCCL's internal sync expectations)
- NCCL graph replay has additional requirements beyond launch synchronization
- Need `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH` to diagnose

**Workaround**: Set `ALLSPARK_EXECUTE_MODE=eager` or keep NCCL ops as `IsGraphUnsafe`. To revert:
```cpp
// In allreduce_op.h and allgather_op.h, uncomment:
bool IsGraphUnsafe() const override { return true; }
```

### 3.2 Warmup-Captured Graphs Are Stale

Graphs captured during warmup requests produce garbage output when replayed with real requests. Root cause: warmup uses zero-filled inputs and fresh KV cache; the captured graph contains kernel launches that reference these specific buffer states.

**Current fix**: Clear all graphs after warmup, rely on lazy capture during first real inference.

**Future fix**: Implement proper pre-capture with dummy GenerateContexts that have valid request objects.

### 3.3 prompt_logprobs Feature Incomplete

The `prompt_logprobs` field was added to `GenerateConfig` in a previous session but the feature is not functional (OOM on large sequences without chunk prefill). References to `prompt_logprobs` in `generate_op.cpp`, `get_last_line.cpp`, and `allspark_binding_common.h` have been stubbed out with `0` or commented out to avoid compilation errors.

---

## 4. File Change Summary

### Core CUDA Graph
| File | Changes |
|------|---------|
| `csrc/interface/allspark.h` | `AsExecuteMode` enum, `execute_mode`, `cuda_graph_batch_sizes` in `AsModelConfig` |
| `csrc/core/model/model.h` | `SpinBarrier`, `execute_mode_`, `graph_launch_barrier_`, `CudaGraphPreCapture`, public `CudaGraphClear` |
| `csrc/core/model/model.cpp` | Execute mode init, batch bucketing, no StopRequest invalidation, barrier sync |
| `csrc/common/as_engine.cpp` | SpinBarrier creation, warmup graph clearing, `cuda_graph_batch_sizes_` storage |
| `csrc/common/as_engine.h` | `cuda_graph_batch_sizes_` member |
| `csrc/common/engine_worker.h` | `CudaGraphPreCapture`, `CudaGraphClear`, `SetGraphLaunchBarrier` |

### NCCL Operators
| File | Changes |
|------|---------|
| `csrc/core/operator/nccl/allreduce/allreduce_op.h` | `IsGraphUnsafe` commented out |
| `csrc/core/operator/nccl/allreduce/allreduce_op.cpp` | `ctx_->Synchronize()` removed |
| `csrc/core/operator/nccl/allgather/allgather_op.h` | `IsGraphUnsafe` commented out |
| `csrc/core/operator/nccl/allgather/allgather_op.cpp` | Pre/post `Synchronize()` removed |
| `csrc/core/operator/generate_opt/mla_attn/mla_attn_op.h` | `IsGraphUnsafe` commented out |

### Python Bindings
| File | Changes |
|------|---------|
| `python/allspark_binding.cpp` | `AsExecuteMode` enum, `execute_mode`, `cuda_graph_batch_sizes` |
| `python/dashinfer/allspark/runtime_config.py` | `execute_mode()`, `cuda_graph_batch_sizes()` builder methods |

### Flash-Attention
| File | Changes |
|------|---------|
| `fix_flash_attn_v283.sh` | Reusable patch script for v2.8.3 |
| `csrc/core/kernel/cuda/flashv2/flashv2.h` | `flash::Flash_fwd_params` namespace |
| `csrc/core/kernel/cuda/flashv2/flash_attention_operator.cu` | `flash::run_mha_fwd` namespace |

### Bug Fixes (from earlier)
| File | Changes |
|------|---------|
| `csrc/device/bfc_allocator.h` | `static` → `extern` ODR fix |
| `csrc/device/bfc_allocator.cpp` | ChunkHandle==0 false-positive fix |
| `build.sh` | `set -e`, ninja check, nvcc PATH |
| `cmake/lmfe.cmake` | Library path fix |
| `cmake/hie-dnn.cmake` | Build dir guarantee |

### Stubs (prompt_logprobs, not functional)
| File | Changes |
|------|---------|
| `csrc/common/engine_runtime.h` | `prompt_logprobs` → `0` |
| `csrc/core/operator/general/get_last_line/get_last_line.cpp` | `prompt_logprobs` → `0` |
| `csrc/core/operator/generate_opt/generate/generate_op.cpp` | `prompt_logprobs` → `0` |
| `python/allspark_binding_common.h` | prompt logprobs bindings commented out |

---

## 5. Build Instructions

```bash
# Full clean build
cd /home/jzhang/dash-infer
source .venv/bin/activate
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
AS_CUDA_VERSION="12.6" AS_CUDA_SM="90a" AS_BUILD_PACKAGE=OFF AS_CXX11_ABI=ON bash build.sh

# If flash-attention patch fails:
bash fix_flash_attn_v283.sh
cd build && make -j16

# Python bindings:
cd build
cmake .. -DBUILD_PYTHON=ON -DPYTHON_LIB_DIRS="$(pwd)/python_out/dashinfer/allspark" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_PACKAGE=OFF -DCONFIG_ACCELERATOR_TYPE=CUDA \
  -DCONFIG_HOST_CPU_TYPE=X86 -DNCCL_VERSION=2.23.4 -DCUDA_VERSION=12.6 \
  -DCMAKE_CUDA_ARCHITECTURES="90a" -DUSE_SYSTEM_NV_LIB=OFF \
  -DENABLE_GLIBCXX11_ABI=ON -DENABLE_SPAN_ATTENTION=ON -DBUILD_HIEDNN=ON
make -j16 install

# Symlink:
ln -sf build/python_out/dashinfer/allspark/_allspark.cpython-310-x86_64-linux-gnu.so \
  python/dashinfer/allspark/
```

## 6. Test Commands

```bash
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$(pwd)/build/csrc:$LD_LIBRARY_PATH"
export PYTHONPATH="$(pwd)/python"

# Single GPU (CudaGraph default ON)
CUDA_VISIBLE_DEVICES=0 python3 test_cuda_graph_comprehensive.py

# 2-GPU
CUDA_VISIBLE_DEVICES=0,1 python3 test_cuda_graph_comprehensive.py --gpus 0,1

# Force eager mode
ALLSPARK_EXECUTE_MODE=eager CUDA_VISIBLE_DEVICES=0 python3 test_cuda_graph_comprehensive.py

# 72B benchmark
CUDA_VISIBLE_DEVICES=0,1 python3 bench_dashinfer_72b.py
```

## 7. Next Steps

1. **Debug multi-GPU NCCL graph hang** — run with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH`
2. **If SpinBarrier doesn't fix it** — implement custom AllReduce (NVLink P2P), ~200 lines CUDA
3. **Pre-capture during warmup** — fix dummy GenerateContext to have valid Request objects
4. **FlashAttention-3 SM90** — replace span-attention decode path for +20-30% single-request TPS
5. **HumanEval accuracy test** — verify graph mode doesn't affect output quality
6. **Benchmark with graph ON** — compare piecewise (57-seg multi-GPU) vs eager
