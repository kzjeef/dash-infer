/*!
 * Copyright contributors to the DashInfer Project
 * @file    flashmla.h
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include "cuda/cuda_common.h"

// FlashMLA wrapper is always compiled (provides stubs).
// The actual FlashMLA kernel dispatch requires ENABLE_FLASH_MLA=ON at build time.

namespace allspark {

namespace cuda {

struct FlashMLAParams {
  // Dimensions
  int batch_size;
  int num_heads_q;       // number of query heads (128 for DeepSeek V3)
  int num_heads_kv;      // number of KV heads (1 for MQA)
  int head_dim_qk;       // query/key head dim (576 = 512 + 64)
  int head_dim_v;        // value head dim (512)
  int page_block_size;   // page block size (64)

  // Input pointers (device memory)
  const void* q;             // [batch, 1, num_heads_q, head_dim_qk]
  const void* kv_cache;      // paged KV cache
  void* output;              // [batch, 1, num_heads_q, head_dim_v]

  // Paged attention tables
  const int* block_table;    // [batch, max_num_blocks_per_seq]
  const int* cache_seqlens;  // [batch]
  int block_table_batch_stride;

  // Softmax
  float softmax_scale;
  bool is_causal;

  // Data type
  bool is_bf16;

  // Workspace
  void* splitkv_out;
  float* splitkv_lse;
  void* tile_scheduler_metadata;
  int* num_splits;
  bool metadata_initialized;

  // CUDA device properties
  const cudaDeviceProp* dprop;
};

/// Clear all parameters to zero
void flashmla_clear_param(FlashMLAParams& params);

/// Set static parameters (dimensions, data types, device properties)
void flashmla_set_static_param(FlashMLAParams& params,
                               cudaDeviceProp& dprop,
                               cudaDataType_t dtype,
                               int batch_size,
                               int num_heads_q,
                               int num_heads_kv,
                               int head_dim_qk,
                               int head_dim_v,
                               int page_block_size,
                               bool is_causal);

/// Set runtime parameters (pointers, scales)
void flashmla_set_runtime_param(FlashMLAParams& params,
                                const void* q,
                                const void* kv_cache,
                                void* output,
                                const int* block_table,
                                const int* cache_seqlens,
                                int block_table_batch_stride,
                                float softmax_scale,
                                void* splitkv_out,
                                float* splitkv_lse,
                                void* tile_scheduler_metadata,
                                int* num_splits);

/// Calculate workspace sizes needed for split-KV decode
void flashmla_get_workspace_sizes(int batch_size,
                                  int num_heads_q,
                                  int head_dim_v,
                                  int max_seqlen_kv,
                                  int page_block_size,
                                  size_t& splitkv_out_bytes,
                                  size_t& splitkv_lse_bytes,
                                  size_t& metadata_bytes,
                                  size_t& num_splits_bytes);

/// Dispatch the FlashMLA decode kernel
void flashmla_dispatch(FlashMLAParams& params, cudaStream_t stream);

/// Naive MLA decode attention (fallback when FlashMLA library is not linked).
/// Uses the "absorbed" attention approach: pre-absorbs kv_b_proj into Q,
/// then iterates over paged compressed KV cache with online softmax.
///
/// @param output         [batch, num_heads * v_head_dim]
/// @param q_full         [batch, num_heads * (qk_nope + qk_rope)]
/// @param kv_b_proj      [kv_lora_rank, num_heads * (qk_nope + v_head_dim)]
/// @param kv_span_ptrs   [batch * max_spans] device pointers to span data
/// @param cache_seqlens  [batch] number of cached tokens per batch item
/// @param kv_stride      elements per token in cache (kv_lora_rank + qk_rope)
template <typename T>
void MLADecodeNaiveLauncher(T* output, const T* q_full, const T* kv_b_proj,
                            void* const* kv_span_ptrs,
                            const int* cache_seqlens, int batch_size,
                            int num_heads, int kv_lora_rank, int qk_nope,
                            int qk_rope, int v_head_dim, int kv_stride,
                            int span_len, int max_spans, float softmax_scale,
                            cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
