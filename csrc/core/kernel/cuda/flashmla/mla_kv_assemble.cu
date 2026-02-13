/*!
 * Copyright contributors to the DashInfer Project
 * @file    mla_kv_assemble.cu
 *
 * CUDA kernels for MLA K/V assembly during prefill.
 *
 * After the kv_b_proj up-projection, we have:
 *   kv_full: [M, H, d_nope + d_v]
 * And from kv_compressed (after RoPE on k_rope portion):
 *   k_rope:  [M, d_rope]  (single head, needs broadcast to all H heads)
 *
 * We need to assemble:
 *   K: [M, H, d_nope + d_rope]  where each head's K = [k_nope_h, k_rope]
 *   V: [M, H, d_v]             extracted from kv_full
 *
 * Also provides a strided RoPE kernel for k_rope within kv_compressed.
 */

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>

#include "cuda/cuda_common.h"
#include "cuda/cuda_kernel.h"

namespace allspark {
namespace cuda {

// ---------------------------------------------------------------------------
// K/V assembly kernel
// ---------------------------------------------------------------------------
// Assembles K and V tensors from kv_full and k_rope for flash-attention prefill.
//
// kv_full:     [M, H * (d_nope + d_v)]  - contiguous per-head blocks of [d_nope, d_v]
// k_rope:      [M, d_rope]              - single head, to be broadcast
// k_assembled: [M, H, d_nope + d_rope]  - output K (qk_head_dim per head)
// v_assembled: [M, H, d_v]              - output V
template <typename T>
__global__ void mla_kv_assemble_kernel(
    T* __restrict__ k_assembled, T* __restrict__ v_assembled,
    const T* __restrict__ kv_full, const T* __restrict__ k_rope,
    int total_tokens, int num_heads, int d_nope, int d_rope, int d_v,
    int kv_full_head_stride,  // d_nope + d_v
    int k_rope_stride         // stride between tokens in k_rope (may be > d_rope if strided)
) {
  // Grid: [total_tokens, num_heads]
  // Threads: handle all dims within one head
  int token = blockIdx.x;
  int head = blockIdx.y;
  if (token >= total_tokens || head >= num_heads) return;

  int qk_head_dim = d_nope + d_rope;

  // Source pointers for this token and head
  const T* kv_full_head =
      kv_full + (int64_t)token * num_heads * kv_full_head_stride +
      head * kv_full_head_stride;
  const T* k_rope_token = k_rope + (int64_t)token * k_rope_stride;

  // Destination pointers
  T* k_out = k_assembled + (int64_t)token * num_heads * qk_head_dim +
             head * qk_head_dim;
  T* v_out = v_assembled + (int64_t)token * num_heads * d_v + head * d_v;

  // Copy k_nope from kv_full (first d_nope dims)
  for (int d = threadIdx.x; d < d_nope; d += blockDim.x) {
    k_out[d] = kv_full_head[d];
  }

  // Copy k_rope (broadcast from single head) into k_out[d_nope:]
  for (int d = threadIdx.x; d < d_rope; d += blockDim.x) {
    k_out[d_nope + d] = k_rope_token[d];
  }

  // Copy v from kv_full (last d_v dims)
  for (int d = threadIdx.x; d < d_v; d += blockDim.x) {
    v_out[d] = kv_full_head[d_nope + d];
  }
}

template <typename T>
void MLAKVAssembleLauncher(T* k_assembled, T* v_assembled, const T* kv_full,
                           const T* k_rope, int total_tokens, int num_heads,
                           int d_nope, int d_rope, int d_v,
                           int k_rope_stride, cudaStream_t stream) {
  int kv_full_head_stride = d_nope + d_v;
  int threads = min(max(d_nope, max(d_rope, d_v)), 256);
  dim3 grid(total_tokens, num_heads);

  mla_kv_assemble_kernel<<<grid, threads, 0, stream>>>(
      k_assembled, v_assembled, kv_full, k_rope, total_tokens, num_heads,
      d_nope, d_rope, d_v, kv_full_head_stride, k_rope_stride);
}

// Explicit instantiations
template void MLAKVAssembleLauncher<float>(float*, float*, const float*,
                                           const float*, int, int, int, int,
                                           int, int, cudaStream_t);
#ifdef ENABLE_FP16
template void MLAKVAssembleLauncher<half>(half*, half*, const half*,
                                          const half*, int, int, int, int, int,
                                          int, cudaStream_t);
#endif
#ifdef ENABLE_BF16
template void MLAKVAssembleLauncher<hie::bfloat16>(
    hie::bfloat16*, hie::bfloat16*, const hie::bfloat16*,
    const hie::bfloat16*, int, int, int, int, int, int, cudaStream_t);
#endif

// ---------------------------------------------------------------------------
// Strided K RoPE kernel
// ---------------------------------------------------------------------------
// Applies RoPE to k_rope that lives within kv_compressed at offset kv_lora_rank.
// The data is strided: each token's k_rope starts at kv_compressed + kv_lora_rank
// with stride = kv_lora_rank + d_rope between tokens.
template <typename T>
__global__ void mla_strided_rope_k_kernel(T* kv_compressed, const float* inv_freq,
                                          const int* step_list,
                                          int total_tokens, int d_rope,
                                          int kv_lora_rank, int stride) {
  int token_idx = blockIdx.x;
  int rope_pair = threadIdx.x;

  if (token_idx >= total_tokens || rope_pair >= d_rope / 2) return;

  int pos;
  if (step_list) {
    pos = step_list[token_idx];
  } else {
    pos = token_idx;
  }

  float freq = inv_freq[rope_pair];
  float angle = (float)pos * freq;
  float cos_val = cosf(angle);
  float sin_val = sinf(angle);

  // k_rope starts at offset kv_lora_rank within each token's kv_compressed
  int base = token_idx * stride + kv_lora_rank;
  int i0 = base + 2 * rope_pair;
  int i1 = base + 2 * rope_pair + 1;

  float x0 = (float)kv_compressed[i0];
  float x1 = (float)kv_compressed[i1];

  kv_compressed[i0] = (T)(x0 * cos_val - x1 * sin_val);
  kv_compressed[i1] = (T)(x1 * cos_val + x0 * sin_val);
}

template <typename T>
void MLAStridedRoPEKLauncher(T* kv_compressed, const float* inv_freq,
                             const int* step_list, int total_tokens,
                             int d_rope, int kv_lora_rank,
                             cudaStream_t stream) {
  int stride = kv_lora_rank + d_rope;
  int threads = d_rope / 2;
  mla_strided_rope_k_kernel<<<total_tokens, threads, 0, stream>>>(
      kv_compressed, inv_freq, step_list, total_tokens, d_rope, kv_lora_rank,
      stride);
}

// Explicit instantiations
template void MLAStridedRoPEKLauncher<float>(float*, const float*, const int*,
                                             int, int, int, cudaStream_t);
#ifdef ENABLE_FP16
template void MLAStridedRoPEKLauncher<half>(half*, const float*, const int*,
                                            int, int, int, cudaStream_t);
#endif
#ifdef ENABLE_BF16
template void MLAStridedRoPEKLauncher<hie::bfloat16>(hie::bfloat16*,
                                                      const float*, const int*,
                                                      int, int, int,
                                                      cudaStream_t);
#endif

// ---------------------------------------------------------------------------
// Copy compressed KV to span cache
// ---------------------------------------------------------------------------
// Copies kv_compressed tokens to non-contiguous cache spans.
// Used for both prefill (many tokens) and decode (single token per request).
template <typename T>
__global__ void mla_copy_to_span_cache_kernel(
    T* const* __restrict__ span_ptrs,  // [num_spans] pointers to span data
    const T* __restrict__ kv_compressed,
    int total_tokens, int kv_dim, int span_len, int start_pos) {
  int token = blockIdx.x;
  if (token >= total_tokens) return;

  int global_pos = start_pos + token;
  int span_idx = global_pos / span_len;
  int span_offset = global_pos % span_len;

  T* dst = span_ptrs[span_idx] + (int64_t)span_offset * kv_dim;
  const T* src = kv_compressed + (int64_t)token * kv_dim;

  for (int d = threadIdx.x; d < kv_dim; d += blockDim.x) {
    dst[d] = src[d];
  }
}

template <typename T>
void MLACopyToSpanCacheLauncher(T* const* span_ptrs, const T* kv_compressed,
                                int total_tokens, int kv_dim, int span_len,
                                int start_pos, cudaStream_t stream) {
  int threads = min(kv_dim, 256);
  mla_copy_to_span_cache_kernel<<<total_tokens, threads, 0, stream>>>(
      span_ptrs, kv_compressed, total_tokens, kv_dim, span_len, start_pos);
}

template void MLACopyToSpanCacheLauncher<float>(float* const*, const float*,
                                                int, int, int, int,
                                                cudaStream_t);
#ifdef ENABLE_FP16
template void MLACopyToSpanCacheLauncher<half>(half* const*, const half*, int,
                                               int, int, int, cudaStream_t);
#endif
#ifdef ENABLE_BF16
template void MLACopyToSpanCacheLauncher<hie::bfloat16>(
    hie::bfloat16* const*, const hie::bfloat16*, int, int, int, int,
    cudaStream_t);
#endif

}  // namespace cuda
}  // namespace allspark
#endif  // ENABLE_CUDA
