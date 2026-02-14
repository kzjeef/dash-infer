# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DashInfer (AllSpark) is a C++ LLM inference engine with Python bindings, supporting CUDA GPUs (SM70-SM100), x86 CPUs, and ARMv9. It implements continuous batching, paged attention (SpanAttention), prefix caching, quantization (INT8/INT4/FP8), and multi-GPU inference via NCCL.

## Build Commands

### Python wheel (primary build method)
```bash
cd python
AS_CUDA_VERSION="12.4" AS_CUDA_SM="'80;90a'" AS_PLATFORM="cuda" python3 setup.py bdist_wheel
pip install dist/dashinfer-*.whl --force-reinstall --no-deps
```

### C++ only (via build.sh)
```bash
AS_CUDA_VERSION="12.4" AS_CUDA_SM="'80;90a'" AS_PLATFORM="cuda" bash build.sh
```
This runs Conan dependency install (first time or when `AS_FORCE_CONAN=ON`), then cmake + make. Build output goes to `build/`.

### Key environment variables
- `AS_CUDA_VERSION`: CUDA toolkit version (default: 12.4)
- `AS_CUDA_SM`: Target SM architectures, semicolon-separated in quotes (default: `'80;90a'`)
- `AS_PLATFORM`: `cuda`, `x86`, or `armclang`
- `AS_NCCL_VERSION`: NCCL version (default: 2.23.4)
- `AS_BUILD_TYPE`: `Release` or `Debug`
- `ENABLE_SPAN_ATTENTION`: ON/OFF (default: ON for CUDA)

### Dependencies
Managed by Conan (conanfile in `conan/`): protobuf, gtest, glog, pybind11, zlib. Requires `conan` on PATH.

## Testing

### C++ tests (built via cmake)
Test binaries are in `build/bin/`:
- `cpp_interface_test` - Engine interface tests
- `cpp_operator_test` - Operator correctness tests
- `cpp_kernel_test` - CUDA kernel tests (requires `ENABLE_SPAN_ATTENTION`)
- `cpp_model_test` - Full model inference tests
- `model_stress_test` - Stress testing

Run a single test filter: `./build/bin/cpp_kernel_test --gtest_filter="MLA_KERNEL.*"`

### Python integration tests
See `run_full_test.sh` for the full pipeline: build wheel → install → run inference with Qwen2.5-7B.

## Architecture

### Layered design
```
Python API (engine.py, model_loader.py)
  ↓ pybind11 + DLPack + Protobuf
C++ Interface (csrc/interface/)
  ↓
Runtime (csrc/runtime/) — batching, cache mgmt, scheduling
  ↓
Model (csrc/core/model/) — operator graph orchestration
  ↓
Operator (csrc/core/operator/) — composite operations
  ↓
Kernel (csrc/core/kernel/cuda/, cpu/) — device-specific implementations
```

### Source file auto-discovery
`csrc/CMakeLists.txt` uses `GLOB_RECURSE` to find all `.cpp` files under `csrc/core/operator/`, `csrc/core/model/`, etc. New files placed in these directories are automatically compiled. CUDA kernels are similarly globbed from `csrc/core/kernel/cuda/`.

### Model registration
Models register via `REGISTER_MODEL("ModelName", ClassName)` macro in their `.cpp` files. The factory in `model.h` maps model type strings to constructors. All current models are registered in `csrc/core/model/` subdirectories (e.g., `qwen/qwen.cpp`, `llama/llama.cpp`).

### Operator registration
Operators register via `REGISTER_OP(OpName, DeviceType, ClassName)` in their `.cpp` files. The factory maps `(op_type_string, device_type)` pairs to constructors. Operators live under `csrc/core/operator/general/` (Gemm, LayerNorm, etc.) and `csrc/core/operator/generate_opt/` (SpanAttn, MLAAttn, BatchMHA, etc.).

### Python model graph definition
Each model type has a Python class (e.g., `python/pyhie/allspark/model/qwen_v20.py`) with a `_build_graph()` method that constructs a `TransformerProto` computation graph. The graph is serialized to `.asgraph` + `.asparam` files, then loaded by the C++ engine. Python prefill op types (e.g., `"MLAAttention"`) are converted to decode op types (e.g., `"DecOptMLA"`) in `_gen_graph()`.

### KV cache system
- `CacheSpanManager` manages paged KV cache frames
- Flow: free frames → `PresFrame(N)` pre-reserves → `ClaimSpanFromPres()` allocates
- `model.cpp` finds attention ops via `dynamic_cast<SpanAttnOp*>()` and `dynamic_cast<MLAAttnOp*>()` to allocate cache
- `kv_cache_dim` field in ConfigProto overrides cache sizing (used by MLA for compressed KV)

### Multi-GPU
Tensor parallelism via NCCL AllReduce/AllGather operators. Weight split strategies (VSPLIT, HSPLIT, QKVSPLIT) defined per-layer in Python model definitions.

### Protobuf schema
`csrc/proto/allspark.proto` defines `ConfigProto`, `TransformerProto`, `GraphProto`, `OperatorProto`, `TensorProto`. This is the bridge between Python graph construction and C++ execution.

### Flash Attention configuration
`cmake/flash-attention.cmake` has `TARGET_HEADDIM_LIST` controlling which head dimensions are compiled. Default is `"128"`. Add sizes like `"192"` for MLA (128 nope + 64 rope).

### Weight loading pipeline
`HuggingFaceModel` (in `model_loader.py`) loads HuggingFace models via `AutoConfig`, converts weights using regex-based name adapters, and serializes to DashInfer format. Supports lazy loading (`LazySafetensorsDict`) for large models (671B+).

## Key directories
- `csrc/core/model/` — Model implementations (Qwen, LLaMA, ChatGLM, DeepSeek)
- `csrc/core/operator/` — All operator implementations
- `csrc/core/kernel/cuda/` — CUDA kernels (attention, cache, MOE, etc.)
- `csrc/runtime/cache/` — KV cache management
- `csrc/proto/` — Protobuf definitions
- `python/pyhie/allspark/model/` — Python model graph definitions
- `python/pyhie/allspark/engine.py` — Python inference API
- `python/pyhie/allspark/model_loader.py` — HuggingFace model loading
- `cmake/` — CMake modules (flash-attention, cutlass, NCCL, etc.)
- `span-attention/` — Paged attention library
- `HIE-DNN/` — High-performance DNN operator library
- `examples/python/0_basic/` — Basic inference examples
