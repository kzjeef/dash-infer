/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    mla_decode_naive.cu
 *
 * Naive MLA decode attention kernel using the "absorbed" attention approach.
 * This is a fallback when the FlashMLA library is not linked.
 *
 * For each (batch, head):
 *   1. Absorb: q_absorbed[kv_lora_rank] = q_nope @ W_k_nope^T
 *   2. Iterate cached tokens (paged spans), compute:
 *      score = dot(q_absorbed, kv_latent) + dot(q_rope, k_rope_cached)
 *   3. Online softmax + weighted sum of kv_latent
 *   4. Project: output[v_head_dim] = weighted_kv @ W_v
 */

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <math.h>

#include "../cuda_common.h"
#include "flashmla.h"

namespace allspark {
namespace cuda {

// Warp-level sum reduction
__device__ __forceinline__ float warpReduceSum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// Block-level sum reduction using shared memory
__device__ float blockReduceSum(float val, float* shared_buf) {
  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;
  int num_warps = (blockDim.x + 31) / 32;

  val = warpReduceSum(val);
  if (lane_id == 0) shared_buf[warp_id] = val;
  __syncthreads();

  if (tid == 0) {
    float total = 0.0f;
    for (int w = 0; w < num_warps; w++) total += shared_buf[w];
    shared_buf[0] = total;
  }
  __syncthreads();
  return shared_buf[0];
}

/**
 * One block per (batch, head).
 * All threads cooperate to process tokens one at a time.
 *
 * Shared memory layout:
 *   [0 .. kv_lora_rank)              : q_absorbed (float)
 *   [kv_lora_rank .. kv_lora_rank+qk_rope) : q_rope_f (float)
 *   [.. +kv_lora_rank)              : accum_kv (float)
 *   [.. +3)                         : running_max, running_sum, prev_max
 *   [.. +8)                         : warp reduction scratch
 */
template <typename T>
__global__ void mla_decode_naive_kernel(
    T* __restrict__ output,
    const T* __restrict__ q_full,
    const T* __restrict__ kv_b_proj,
    void* const* __restrict__ kv_span_ptrs,
    const int* __restrict__ cache_seqlens,
    int num_heads, int kv_lora_rank, int qk_nope, int qk_rope, int v_head_dim,
    int kv_stride, int span_len, int max_spans, float softmax_scale) {
  int bid = blockIdx.x;
  int batch_idx = bid / num_heads;
  int head_idx = bid % num_heads;
  int tid = threadIdx.x;
  int BS = blockDim.x;

  int seq_len = cache_seqlens[batch_idx];
  if (seq_len <= 0) return;

  int qk_hd = qk_nope + qk_rope;
  int kv_hd = qk_nope + v_head_dim;
  int kv_full_cols = num_heads * kv_hd;

  const T* q_h = q_full + (int64_t)batch_idx * num_heads * qk_hd +
                 (int64_t)head_idx * qk_hd;

  // Shared memory
  extern __shared__ char smem_raw[];
  float* smem = (float*)smem_raw;
  float* s_qa = smem;
  float* s_qr = s_qa + kv_lora_rank;
  float* s_akv = s_qr + qk_rope;
  float* s_ctrl = s_akv + kv_lora_rank;  // [3]: running_max, running_sum, prev_max
  float* s_warp = s_ctrl + 3;            // [8]: warp reduction

  // Step 1: Compute q_absorbed[j] = sum_i q_nope[i] * W_k[j, head*stride+i]
  for (int j = tid; j < kv_lora_rank; j += BS) {
    float val = 0.0f;
    const T* w_row =
        kv_b_proj + (int64_t)j * kv_full_cols + (int64_t)head_idx * kv_hd;
    for (int i = 0; i < qk_nope; i++) {
      val += (float)q_h[i] * (float)w_row[i];
    }
    s_qa[j] = val;
  }

  for (int i = tid; i < qk_rope; i += BS) {
    s_qr[i] = (float)q_h[qk_nope + i];
  }

  for (int j = tid; j < kv_lora_rank; j += BS) {
    s_akv[j] = 0.0f;
  }

  if (tid == 0) {
    s_ctrl[0] = -1e30f;  // running_max
    s_ctrl[1] = 0.0f;    // running_sum
    s_ctrl[2] = -1e30f;  // prev_max
  }
  __syncthreads();

  // Step 2: Online softmax attention over cached tokens
  for (int t = 0; t < seq_len; t++) {
    int si = t / span_len;
    int so = t % span_len;
    const T* kv =
        (const T*)kv_span_ptrs[batch_idx * max_spans + si] +
        (int64_t)so * kv_stride;

    // Compute partial score
    float partial = 0.0f;
    for (int j = tid; j < kv_lora_rank; j += BS) {
      partial += s_qa[j] * (float)kv[j];
    }
    for (int j = tid; j < qk_rope; j += BS) {
      partial += s_qr[j] * (float)kv[kv_lora_rank + j];
    }

    // Block reduction
    float score = blockReduceSum(partial, s_warp) * softmax_scale;

    // Online softmax update (all threads read from shared)
    if (tid == 0) {
      s_ctrl[2] = s_ctrl[0];  // prev_max
      s_ctrl[0] = fmaxf(s_ctrl[0], score);
      float rescale = expf(s_ctrl[2] - s_ctrl[0]);
      float weight = expf(score - s_ctrl[0]);
      s_ctrl[1] = s_ctrl[1] * rescale + weight;
    }
    __syncthreads();

    float rescale = expf(s_ctrl[2] - s_ctrl[0]);
    float weight = expf(score - s_ctrl[0]);

    // Rescale accumulator and add weighted kv_latent
    for (int j = tid; j < kv_lora_rank; j += BS) {
      s_akv[j] = s_akv[j] * rescale + weight * (float)kv[j];
    }
    __syncthreads();
  }

  // Normalize
  float inv_sum = (s_ctrl[1] > 0.0f) ? (1.0f / s_ctrl[1]) : 0.0f;
  for (int j = tid; j < kv_lora_rank; j += BS) {
    s_akv[j] *= inv_sum;
  }
  __syncthreads();

  // Step 3: Project through W_v to get output
  // output[batch, head, d] = accum_kv @ W_v[kv_lora_rank, v_head_dim]
  // W_v for head h: kv_b_proj[:, h*kv_hd + qk_nope : h*kv_hd + qk_nope + v_head_dim]
  T* out_h = output + (int64_t)batch_idx * num_heads * v_head_dim +
             (int64_t)head_idx * v_head_dim;
  for (int d = tid; d < v_head_dim; d += BS) {
    float val = 0.0f;
    for (int j = 0; j < kv_lora_rank; j++) {
      val += s_akv[j] *
             (float)kv_b_proj[(int64_t)j * kv_full_cols +
                              (int64_t)head_idx * kv_hd + qk_nope + d];
    }
    out_h[d] = (T)val;
  }
}

template <typename T>
void MLADecodeNaiveLauncher(T* output, const T* q_full, const T* kv_b_proj,
                            void* const* kv_span_ptrs,
                            const int* cache_seqlens, int batch_size,
                            int num_heads, int kv_lora_rank, int qk_nope,
                            int qk_rope, int v_head_dim, int kv_stride,
                            int span_len, int max_spans, float softmax_scale,
                            cudaStream_t stream) {
  int grid = batch_size * num_heads;
  int block = 256;

  // Shared memory: q_absorbed + q_rope + accum_kv + ctrl(3) + warp(8)
  int smem_bytes = (kv_lora_rank + qk_rope + kv_lora_rank + 3 + 8) * sizeof(float);

  mla_decode_naive_kernel<T><<<grid, block, smem_bytes, stream>>>(
      output, q_full, kv_b_proj, kv_span_ptrs, cache_seqlens, num_heads,
      kv_lora_rank, qk_nope, qk_rope, v_head_dim, kv_stride, span_len,
      max_spans, softmax_scale);
}

// Explicit instantiations
template void MLADecodeNaiveLauncher<float>(
    float* output, const float* q_full, const float* kv_b_proj,
    void* const* kv_span_ptrs, const int* cache_seqlens, int batch_size,
    int num_heads, int kv_lora_rank, int qk_nope, int qk_rope, int v_head_dim,
    int kv_stride, int span_len, int max_spans, float softmax_scale,
    cudaStream_t stream);

#ifdef ENABLE_FP16
template void MLADecodeNaiveLauncher<half>(
    half* output, const half* q_full, const half* kv_b_proj,
    void* const* kv_span_ptrs, const int* cache_seqlens, int batch_size,
    int num_heads, int kv_lora_rank, int qk_nope, int qk_rope, int v_head_dim,
    int kv_stride, int span_len, int max_spans, float softmax_scale,
    cudaStream_t stream);
#endif

#ifdef ENABLE_BF16
template void MLADecodeNaiveLauncher<hie::bfloat16>(
    hie::bfloat16* output, const hie::bfloat16* q_full,
    const hie::bfloat16* kv_b_proj, void* const* kv_span_ptrs,
    const int* cache_seqlens, int batch_size, int num_heads, int kv_lora_rank,
    int qk_nope, int qk_rope, int v_head_dim, int kv_stride, int span_len,
    int max_spans, float softmax_scale, cudaStream_t stream);
#endif

}  // namespace cuda
}  // namespace allspark
#endif  // ENABLE_CUDA
