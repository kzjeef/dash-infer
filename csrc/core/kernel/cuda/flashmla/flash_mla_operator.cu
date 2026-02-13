/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    flash_mla_operator.cu
 */

#ifdef ENABLE_CUDA
#include "flashmla.h"

#include <cstring>

namespace allspark {
namespace cuda {

void flashmla_clear_param(FlashMLAParams& params) {
  memset(&params, 0, sizeof(params));
}

void flashmla_set_static_param(FlashMLAParams& params,
                               cudaDeviceProp& dprop,
                               cudaDataType_t dtype,
                               int batch_size,
                               int num_heads_q,
                               int num_heads_kv,
                               int head_dim_qk,
                               int head_dim_v,
                               int page_block_size,
                               bool is_causal) {
  switch (dtype) {
    case cudaDataType_t::CUDA_R_16BF:
      params.is_bf16 = true;
      break;
    case cudaDataType_t::CUDA_R_16F:
      params.is_bf16 = false;
      break;
    default:
      return;
  }

  params.dprop = &dprop;
  params.batch_size = batch_size;
  params.num_heads_q = num_heads_q;
  params.num_heads_kv = num_heads_kv;
  params.head_dim_qk = head_dim_qk;
  params.head_dim_v = head_dim_v;
  params.page_block_size = page_block_size;
  params.is_causal = is_causal;
}

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
                                int* num_splits) {
  params.q = q;
  params.kv_cache = kv_cache;
  params.output = output;
  params.block_table = block_table;
  params.cache_seqlens = cache_seqlens;
  params.block_table_batch_stride = block_table_batch_stride;
  params.softmax_scale = softmax_scale;
  params.splitkv_out = splitkv_out;
  params.splitkv_lse = splitkv_lse;
  params.tile_scheduler_metadata = tile_scheduler_metadata;
  params.num_splits = num_splits;
}

void flashmla_get_workspace_sizes(int batch_size,
                                  int num_heads_q,
                                  int head_dim_v,
                                  int max_seqlen_kv,
                                  int page_block_size,
                                  size_t& splitkv_out_bytes,
                                  size_t& splitkv_lse_bytes,
                                  size_t& metadata_bytes,
                                  size_t& num_splits_bytes) {
  int max_num_blocks = (max_seqlen_kv + page_block_size - 1) / page_block_size;
  int max_splits = (max_num_blocks + 3) / 4;
  if (max_splits < 1) max_splits = 1;
  if (max_splits > 256) max_splits = 256;

  splitkv_out_bytes = (size_t)batch_size * max_splits * num_heads_q * head_dim_v * 2;
  splitkv_lse_bytes = (size_t)batch_size * max_splits * num_heads_q * sizeof(float);
  metadata_bytes = (size_t)batch_size * max_num_blocks * 32;
  num_splits_bytes = (size_t)(batch_size + 1) * sizeof(int);
}

void flashmla_dispatch(FlashMLAParams& params, cudaStream_t stream) {
#ifdef FLASH_MLA_LIB
  // When FlashMLA library is linked (ENABLE_FLASH_MLA=ON), call the C API
  // TODO: Bridge to flash_mla_dense_decode() from the FlashMLA C API
  LOG(ERROR) << "FlashMLA dispatch: library call not yet wired";
#else
  // Stub: FlashMLA library not available
  LOG(WARNING) << "FlashMLA dispatch called but library not linked. "
               << "Build with -DENABLE_FLASH_MLA=ON to enable. "
               << "Decode attention output will be zero.";
#endif
  params.metadata_initialized = true;
}

}  // namespace cuda
}  // namespace allspark
#endif  // ENABLE_CUDA
