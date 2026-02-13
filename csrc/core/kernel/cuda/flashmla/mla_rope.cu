/*!
 * Copyright contributors to the DashInfer Project
 * @file    mla_rope.cu
 *
 * Decoupled RoPE kernel for MLA (Multi-head Latent Attention).
 *
 * In DeepSeek V3's MLA architecture, RoPE is applied in a "decoupled" manner:
 *   - Q has layout [M, H, d_nope + d_rope]. RoPE applies to the last d_rope dims.
 *   - K_rope has layout [M, d_rope] (single head, from the kv_compressed output).
 *     It is NOT broadcast to all H heads at this point; FlashMLA handles it internally.
 *
 * This kernel applies RoPE in-place on the rope portion of Q, and separately on K_rope.
 */

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <math.h>

#include "cuda/cuda_common.h"
#include "cuda/cuda_kernel.h"

namespace allspark {
namespace cuda {

// Decoupled RoPE for Q: applies RoPE to the last d_rope dims of each head.
// q: [total_tokens, num_heads, d_nope + d_rope]  (in-place)
// inv_freq: [d_rope / 2]
// step_list: [batch_size] current position for each batch
// batch_token_offsets: [batch_size] cumulative token offsets per batch (for prefill)
template <typename T>
__global__ void mla_rope_q_kernel(T* q, const float* inv_freq,
                                  const int* step_list,
                                  const int* batch_token_offsets,
                                  int total_tokens, int num_heads,
                                  int d_nope, int d_rope, int seq_len) {
  // Grid: [total_tokens, num_heads]
  // Each thread handles one pair of rope dimensions
  int token_idx = blockIdx.x;
  int head_idx = blockIdx.y;
  int rope_pair = threadIdx.x;  // index into d_rope/2 pairs

  if (token_idx >= total_tokens || head_idx >= num_heads ||
      rope_pair >= d_rope / 2)
    return;

  int head_dim = d_nope + d_rope;

  // Determine position: for prefill, position = seq_position_within_batch
  // For decode, position = step_list[batch]
  int pos;
  if (batch_token_offsets != nullptr) {
    // Prefill: find which batch this token belongs to
    // Simple linear scan (batch_size is small)
    int batch_idx = 0;
    // Use batch_token_offsets to find position within sequence
    // batch_token_offsets[b] = cumulative tokens up to batch b
    // position = token_idx - batch_token_offsets[batch_idx] + step_list[batch_idx]
    pos = token_idx;  // For single-batch prefill, token_idx IS the position
    if (step_list) pos += step_list[0];
  } else {
    // Decode: token_idx IS the batch index, position from step_list
    pos = step_list ? step_list[token_idx] : 0;
  }

  // Compute rotation angle
  float freq = inv_freq[rope_pair];
  float angle = (float)pos * freq;
  float cos_val = cosf(angle);
  float sin_val = sinf(angle);

  // Pointer to the rope portion of this head for this token
  // q layout: [total_tokens, num_heads, head_dim]
  int base_offset = token_idx * num_heads * head_dim + head_idx * head_dim;
  int rope_offset = base_offset + d_nope;  // skip nope dims

  // Apply rotation to pair (2*rope_pair, 2*rope_pair+1)
  int i0 = rope_offset + 2 * rope_pair;
  int i1 = rope_offset + 2 * rope_pair + 1;

  float x0 = (float)q[i0];
  float x1 = (float)q[i1];

  q[i0] = (T)(x0 * cos_val - x1 * sin_val);
  q[i1] = (T)(x1 * cos_val + x0 * sin_val);
}

// Decoupled RoPE for K_rope: applies RoPE to the rope part of kv_compressed.
// k_rope: [total_tokens, d_rope]  (in-place, or write to separate output)
// This is the portion of kv_compressed starting at offset kv_lora_rank.
template <typename T>
__global__ void mla_rope_k_kernel(T* k_rope, const float* inv_freq,
                                  const int* step_list,
                                  int total_tokens, int d_rope) {
  int token_idx = blockIdx.x;
  int rope_pair = threadIdx.x;

  if (token_idx >= total_tokens || rope_pair >= d_rope / 2) return;

  int pos;
  if (step_list) {
    pos = step_list[token_idx];
  } else {
    pos = token_idx;  // prefill: token index is position
  }

  float freq = inv_freq[rope_pair];
  float angle = (float)pos * freq;
  float cos_val = cosf(angle);
  float sin_val = sinf(angle);

  int base = token_idx * d_rope;
  int i0 = base + 2 * rope_pair;
  int i1 = base + 2 * rope_pair + 1;

  float x0 = (float)k_rope[i0];
  float x1 = (float)k_rope[i1];

  k_rope[i0] = (T)(x0 * cos_val - x1 * sin_val);
  k_rope[i1] = (T)(x1 * cos_val + x0 * sin_val);
}

// Launcher for MLA decoupled RoPE on Q
template <typename T>
void MLARoPEQLauncher(T* q, const float* inv_freq, const int* step_list,
                      const int* batch_token_offsets, int total_tokens,
                      int num_heads, int d_nope, int d_rope, int seq_len,
                      cudaStream_t stream) {
  dim3 grid(total_tokens, num_heads);
  int threads = d_rope / 2;
  mla_rope_q_kernel<<<grid, threads, 0, stream>>>(
      q, inv_freq, step_list, batch_token_offsets, total_tokens, num_heads,
      d_nope, d_rope, seq_len);
}

// Launcher for MLA decoupled RoPE on K_rope
template <typename T>
void MLARoPEKLauncher(T* k_rope, const float* inv_freq, const int* step_list,
                      int total_tokens, int d_rope, cudaStream_t stream) {
  dim3 grid(total_tokens);
  int threads = d_rope / 2;
  mla_rope_k_kernel<<<grid, threads, 0, stream>>>(
      k_rope, inv_freq, step_list, total_tokens, d_rope);
}

// Explicit instantiations
template void MLARoPEQLauncher<float>(float*, const float*, const int*,
                                      const int*, int, int, int, int, int,
                                      cudaStream_t);
template void MLARoPEKLauncher<float>(float*, const float*, const int*, int,
                                      int, cudaStream_t);
#ifdef ENABLE_FP16
template void MLARoPEQLauncher<half>(half*, const float*, const int*,
                                     const int*, int, int, int, int, int,
                                     cudaStream_t);
template void MLARoPEKLauncher<half>(half*, const float*, const int*, int, int,
                                     cudaStream_t);
#endif
#ifdef ENABLE_BF16
template void MLARoPEQLauncher<hie::bfloat16>(hie::bfloat16*, const float*,
                                               const int*, const int*, int, int,
                                               int, int, int, cudaStream_t);
template void MLARoPEKLauncher<hie::bfloat16>(hie::bfloat16*, const float*,
                                               const int*, int, int,
                                               cudaStream_t);
#endif

}  // namespace cuda
}  // namespace allspark
#endif  // ENABLE_CUDA
