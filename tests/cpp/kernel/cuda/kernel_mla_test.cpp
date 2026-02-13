/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    kernel_mla_test.cpp
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <sstream>

#include "common.hpp"
#include "test_common.h"

#include <core/kernel/cuda/cuda_kernel.h>
#include <cuda/cuda_context.h>

#if CUDA_VERSION >= 12030
#include <core/kernel/cuda/flashmla/flashmla.h>
#endif

namespace {

// ---------------------------------------------------------------------------
//  MLA Configuration Constants (DeepSeek V3)
// ---------------------------------------------------------------------------
constexpr int KV_LORA_RANK = 512;       // Latent dimension for KV compression
constexpr int Q_LORA_RANK = 1536;       // Query compression rank
constexpr int QK_NOPE_HEAD_DIM = 128;   // Non-RoPE part of QK head dim
constexpr int QK_ROPE_HEAD_DIM = 64;    // RoPE part of QK head dim
constexpr int V_HEAD_DIM = 128;         // Value head dimension
constexpr int NUM_HEADS = 128;          // Number of attention heads

// Derived dimensions
constexpr int QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM;  // 192
constexpr int KV_CACHE_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM;     // 576

// ---------------------------------------------------------------------------
//  Utility
// ---------------------------------------------------------------------------
int get_sm_version() {
  int device_id;
  cudaGetDevice(&device_id);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);
  return prop.major * 10 + prop.minor;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
//  Test: MLA dimension verification
// ---------------------------------------------------------------------------
TEST(MLA_KERNEL, DimensionSanityCheck) {
  // Verify MLA architecture dimensions match DeepSeek V3 spec
  EXPECT_EQ(QK_HEAD_DIM, 192);
  EXPECT_EQ(KV_CACHE_DIM, 576);

  // KV cache compression ratio: full KV vs MLA
  // Full KV: 2 * NUM_HEADS * V_HEAD_DIM = 2 * 128 * 128 = 32768 per token
  // MLA:     KV_CACHE_DIM = 576 per token
  // Ratio:   32768 / 576 ≈ 56.9x compression
  int full_kv_size = 2 * NUM_HEADS * V_HEAD_DIM;
  float compression_ratio = (float)full_kv_size / KV_CACHE_DIM;
  EXPECT_GT(compression_ratio, 50.0f);  // Should be ~57x

  // Q projection chain dimensions
  // hidden_size (7168) -> q_lora_rank (1536) -> NUM_HEADS * QK_HEAD_DIM (128*192 = 24576)
  EXPECT_EQ(NUM_HEADS * QK_HEAD_DIM, 24576);
  EXPECT_LT(Q_LORA_RANK, NUM_HEADS * QK_HEAD_DIM);  // Compression in Q path

  // KV projection chain dimensions
  // hidden_size (7168) -> kv_lora_rank + rope_dim (576) -> NUM_HEADS * (nope + v) (128*256 = 32768)
  EXPECT_EQ(NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), 32768);
  EXPECT_LT(KV_CACHE_DIM, NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM));
}

// ---------------------------------------------------------------------------
//  Test: FlashMLA wrapper parameter setup
// ---------------------------------------------------------------------------
#if CUDA_VERSION >= 12030
TEST(MLA_KERNEL, FlashMLAParamSetup) {
  int sm = get_sm_version();
  if (sm < 90) {
    GTEST_SKIP() << "FlashMLA requires SM90+, current SM=" << sm;
  }

  allspark::cuda::FlashMLAParams params;
  allspark::cuda::flashmla_clear_param(params);

  cudaDeviceProp dprop;
  int device_id;
  cudaGetDevice(&device_id);
  cudaGetDeviceProperties(&dprop, device_id);

  // Test static parameter setup
  allspark::cuda::flashmla_set_static_param(
      params, dprop,
      CUDA_R_16BF,   // bf16
      4,             // batch_size
      NUM_HEADS,     // num_heads_q
      1,             // num_heads_kv (MQA)
      KV_CACHE_DIM,  // head_dim_qk (576)
      KV_LORA_RANK,  // head_dim_v (512)
      64,            // page_block_size
      true           // is_causal
  );

  EXPECT_EQ(params.batch_size, 4);
  EXPECT_EQ(params.num_heads_q, NUM_HEADS);
  EXPECT_EQ(params.num_heads_kv, 1);
  EXPECT_EQ(params.head_dim_qk, KV_CACHE_DIM);
  EXPECT_EQ(params.head_dim_v, KV_LORA_RANK);
  EXPECT_EQ(params.page_block_size, 64);
  EXPECT_TRUE(params.is_bf16);
  EXPECT_TRUE(params.is_causal);

  // Test workspace size calculation
  size_t splitkv_out_bytes, splitkv_lse_bytes, metadata_bytes, num_splits_bytes;
  allspark::cuda::flashmla_get_workspace_sizes(
      4, NUM_HEADS, KV_LORA_RANK, 4096, 64,
      splitkv_out_bytes, splitkv_lse_bytes,
      metadata_bytes, num_splits_bytes);

  EXPECT_GT(splitkv_out_bytes, 0);
  EXPECT_GT(splitkv_lse_bytes, 0);
  EXPECT_GT(metadata_bytes, 0);
  EXPECT_GT(num_splits_bytes, 0);
}

TEST(MLA_KERNEL, FlashMLAParamFP16) {
  int sm = get_sm_version();
  if (sm < 90) {
    GTEST_SKIP() << "FlashMLA requires SM90+, current SM=" << sm;
  }

  allspark::cuda::FlashMLAParams params;
  allspark::cuda::flashmla_clear_param(params);

  cudaDeviceProp dprop;
  int device_id;
  cudaGetDevice(&device_id);
  cudaGetDeviceProperties(&dprop, device_id);

  // Test FP16 setup
  allspark::cuda::flashmla_set_static_param(
      params, dprop,
      CUDA_R_16F,    // fp16
      1,             // batch_size
      NUM_HEADS,     // num_heads_q
      1,             // num_heads_kv
      KV_CACHE_DIM,  // head_dim_qk
      KV_LORA_RANK,  // head_dim_v
      64,            // page_block_size
      false          // is_causal
  );

  EXPECT_FALSE(params.is_bf16);
  EXPECT_FALSE(params.is_causal);
}
#endif  // CUDA_VERSION >= 12030

// ---------------------------------------------------------------------------
//  Test: MLA KV cache layout verification
// ---------------------------------------------------------------------------
TEST(MLA_KERNEL, KVCacheLayoutBF16) {
  // Verify that the KV cache layout matches what FlashMLA expects
  //
  // FlashMLA expects kv_cache shaped as:
  //   [num_blocks, page_block_size, num_kv_heads, head_dim_qk]
  //
  // For DeepSeek V3:
  //   num_kv_heads = 1 (single latent shared across all query heads)
  //   head_dim_qk = 576 (512 latent + 64 rope)
  //   page_block_size = 64
  //
  // Each page block stores 64 tokens, each with 576 bf16 values = 73728 bytes

  const int page_block_size = 64;
  const int num_kv_heads = 1;
  const int head_dim_qk = KV_CACHE_DIM;  // 576

  size_t bytes_per_page = page_block_size * num_kv_heads * head_dim_qk * sizeof(uint16_t);
  EXPECT_EQ(bytes_per_page, 64 * 1 * 576 * 2);  // 73728 bytes per page

  // For a 4K context with 64-token pages
  int max_seq = 4096;
  int num_pages = (max_seq + page_block_size - 1) / page_block_size;
  EXPECT_EQ(num_pages, 64);

  size_t total_cache_bytes = num_pages * bytes_per_page;
  // ~4.5MB per request for 4K context (vs ~128MB for full KV cache)
  EXPECT_LT(total_cache_bytes, 5 * 1024 * 1024);
}

// ---------------------------------------------------------------------------
//  Test: MLA projection dimensions
// ---------------------------------------------------------------------------
TEST(MLA_KERNEL, ProjectionDimensions) {
  const int hidden_size = 7168;

  // Q projection chain
  // Step 1: hidden_size -> q_lora_rank (down-projection)
  int q_a_proj_in = hidden_size;
  int q_a_proj_out = Q_LORA_RANK;
  EXPECT_EQ(q_a_proj_out, 1536);

  // Step 2: q_lora_rank -> num_heads * qk_head_dim (up-projection)
  int q_b_proj_in = Q_LORA_RANK;
  int q_b_proj_out = NUM_HEADS * QK_HEAD_DIM;
  EXPECT_EQ(q_b_proj_out, 24576);

  // KV projection chain
  // Step 1: hidden_size -> kv_lora_rank + qk_rope_head_dim (down-projection with rope)
  int kv_a_proj_in = hidden_size;
  int kv_a_proj_out = KV_LORA_RANK + QK_ROPE_HEAD_DIM;
  EXPECT_EQ(kv_a_proj_out, 576);

  // Step 2: kv_lora_rank -> num_heads * (qk_nope_head_dim + v_head_dim) (up-projection)
  int kv_b_proj_in = KV_LORA_RANK;
  int kv_b_proj_out = NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM);
  EXPECT_EQ(kv_b_proj_out, 32768);

  // Verify compression ratios
  // Q compression: 7168 -> 1536 -> 24576
  // Q bottleneck ratio: 24576 / 1536 = 16x expansion
  float q_expansion = (float)q_b_proj_out / q_b_proj_in;
  EXPECT_FLOAT_EQ(q_expansion, 16.0f);

  // KV compression: 7168 -> 576 (stored in cache) -> 32768 (reconstructed)
  // KV bottleneck ratio: 32768 / 512 = 64x expansion
  float kv_expansion = (float)kv_b_proj_out / KV_LORA_RANK;
  EXPECT_FLOAT_EQ(kv_expansion, 64.0f);
}

// ---------------------------------------------------------------------------
//  Benchmark: MLA workspace allocation
// ---------------------------------------------------------------------------
TEST(MLA_KERNEL, WorkspaceAllocation) {
  // Test that workspace sizes are reasonable for typical configurations
  const int batch_sizes[] = {1, 8, 32, 64};
  const int max_seq_len = 4096;
  const int page_block_size = 64;

#if CUDA_VERSION >= 12030
  for (int bs : batch_sizes) {
    size_t splitkv_out_bytes, splitkv_lse_bytes, metadata_bytes,
        num_splits_bytes;
    allspark::cuda::flashmla_get_workspace_sizes(
        bs, NUM_HEADS, KV_LORA_RANK, max_seq_len, page_block_size,
        splitkv_out_bytes, splitkv_lse_bytes,
        metadata_bytes, num_splits_bytes);

    size_t total = splitkv_out_bytes + splitkv_lse_bytes + metadata_bytes +
                   num_splits_bytes;

    // Workspace should be reasonable (< 1GB for any batch size)
    EXPECT_LT(total, 1024ULL * 1024 * 1024)
        << "Workspace too large for batch_size=" << bs;

    // Should grow roughly linearly with batch size
    if (bs > 1) {
      size_t splitkv_out_bytes_1, splitkv_lse_bytes_1, metadata_bytes_1,
          num_splits_bytes_1;
      allspark::cuda::flashmla_get_workspace_sizes(
          1, NUM_HEADS, KV_LORA_RANK, max_seq_len, page_block_size,
          splitkv_out_bytes_1, splitkv_lse_bytes_1,
          metadata_bytes_1, num_splits_bytes_1);
      size_t total_1 = splitkv_out_bytes_1 + splitkv_lse_bytes_1 +
                        metadata_bytes_1 + num_splits_bytes_1;
      // Should be within 2x of linear scaling
      EXPECT_LT(total, total_1 * bs * 2);
    }
  }
#else
  GTEST_SKIP() << "FlashMLA requires CUDA >= 12.3";
#endif
}
