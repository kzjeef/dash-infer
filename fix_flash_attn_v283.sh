#!/bin/bash
# Apply manual patches to flash-attention v2.8.3 for standalone (non-PyTorch) builds.
# Run this after build.sh fails on flash-attn patch, then re-run make.
set -e

FA_SRC="${1:-build/flash-attention/src/project_flashattn}"
FA_CSRC="${FA_SRC}/csrc"
FA_FLASH="${FA_CSRC}/flash_attn/src"

if [ ! -d "$FA_SRC" ]; then
  echo "ERROR: flash-attention source not found at $FA_SRC"
  exit 1
fi

echo "Patching flash-attention v2.8.3 at $FA_SRC ..."

# 1. Create csrc/CMakeLists.txt
if [ ! -f "$FA_CSRC/CMakeLists.txt" ]; then
  cp third_party/patch/flash-attn-cmakelists.txt "$FA_CSRC/CMakeLists.txt" 2>/dev/null || \
  python3 -c "
# Generate CMakeLists.txt from the patch
import re
with open('third_party/patch/flash-attn.patch') as f:
    content = f.read()
# Extract the CMakeLists.txt content between +++ and the next diff
m = re.search(r'\+\+\+ b/csrc/CMakeLists\.txt\n@@ .+?\n((?:\+.*\n)+)', content)
if m:
    lines = [l[1:] for l in m.group(1).splitlines()]  # remove leading +
    with open('$FA_CSRC/CMakeLists.txt', 'w') as out:
        out.write('\n'.join(lines) + '\n')
    print('Created csrc/CMakeLists.txt')
"
fi

# 2. Create flash.cu
cat > "$FA_FLASH/flash.cu" << 'EOF'
#include "namespace_config.h"
#include "cuda.h"
#include "flash.h"
#include "static_switch.h"
#include <cutlass/numeric_types.h>

namespace FLASH_NAMESPACE {
void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                if (params.num_splits <= 1 && !force_split_kernel) {
                    run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                } else {
                    run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                }
            });
        });
    });
}
}  // namespace FLASH_NAMESPACE
EOF
echo "Created flash.cu"

# 3. Patch flash.h — disable ATen, add dprop, add run_mha_fwd decl
cd "$FA_FLASH"

if grep -q "ATen/cuda/CUDAGeneratorImpl.h" flash.h; then
  sed -i 's|#include <ATen/cuda/CUDAGeneratorImpl.h>.*|#if 0\n#include <ATen/cuda/CUDAGeneratorImpl.h>\n#endif|' flash.h
  sed -i '/^namespace FLASH_NAMESPACE {$/a #if 0' flash.h
  sed -i '/^constexpr int D_DIM = 2;$/a #endif' flash.h
  sed -i 's|at::PhiloxCudaState philox_args;|// at::PhiloxCudaState philox_args;|' flash.h
  # Add dprop field
  sed -i '/bool seqlenq_ngroups_swapped;/a\\n    // Cuda Device Properties\n    const cudaDeviceProp* dprop;' flash.h
  # Add SetCudaConfig
  sed -i '/struct Flash_fwd_params : public Qkv_params {/a\    void SetCudaConfig(const cudaDeviceProp* dprop_) { dprop = dprop_; }' flash.h
  # Add run_mha_fwd declaration
  sed -i '/^template<typename T, int Headdim, bool Is_causal> void run_mha_fwd_/i void run_mha_fwd(Flash_fwd_params \&params, cudaStream_t stream, bool force_split_kernel=false);\n' flash.h
  echo "Patched flash.h"
fi

# 4. Patch flash_fwd_kernel.h — disable philox/dropout
if grep -q 'philox_unpack.cuh' flash_fwd_kernel.h; then
  sed -i 's|#include "philox_unpack.cuh".*|// #include "philox_unpack.cuh"|' flash_fwd_kernel.h
  # Wrap the philox block in #if 0
  sed -i '/auto seed_offset = at::cuda::philox::unpack/i #if 0  // Disabled for non-PyTorch builds' flash_fwd_kernel.h
  sed -i '/params.rng_state\[1\] = std::get<1>(seed_offset);/{n;s/    }/    }\n#endif/}' flash_fwd_kernel.h
  echo "Patched flash_fwd_kernel.h (philox)"
fi

# 5. Patch flash_fwd_launch_template.h — replace c10 macros
if grep -q 'c10/cuda/CUDAException.h' flash_fwd_launch_template.h; then
  sed -i 's|#include <c10/cuda/CUDAException.h>.*|// #include <c10/cuda/CUDAException.h>|' flash_fwd_launch_template.h
  # Add CUDA_CHECK macro after includes
  sed -i '/#include "flash_fwd_kernel.h"/a\\n#define CUDA_CHECK(status) { cudaError_t error = status; if (error != cudaSuccess) { std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) << " at line: " << __LINE__ << std::endl; exit(EXIT_FAILURE); } }\n#define CUDA_KERNEL_LAUNCH_CHECK() CUDA_CHECK(cudaGetLastError())' flash_fwd_launch_template.h
  sed -i 's/C10_CUDA_CHECK/CUDA_CHECK/g' flash_fwd_launch_template.h
  sed -i 's/C10_CUDA_KERNEL_LAUNCH_CHECK/CUDA_KERNEL_LAUNCH_CHECK/g' flash_fwd_launch_template.h
  echo "Patched flash_fwd_launch_template.h"
fi

# 6. Patch static_switch.h — add headdim_switch.h guard
if ! grep -q 'headdim_switch.h' static_switch.h; then
  sed -i '/^#define HEADDIM_SWITCH/i #include "headdim_switch.h"\n\n#ifndef HEADDIM_SWITCH' static_switch.h
  echo '#endif' >> static_switch.h
  echo "Patched static_switch.h"
fi

cd - > /dev/null

# 7. Skip rebuild by touching stamps
touch build/flash-attention/src/project_flashattn-stamp/project_flashattn-{download,update,patch}

echo "Flash-attention patches applied. Run 'cd build && make -j16' to continue."
