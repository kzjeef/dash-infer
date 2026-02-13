/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    mla_attn_op_cuda.h
 */
#pragma once

#ifdef ENABLE_CUDA

#include "core/kernel/kernel.h"
#include "mla_attn_op.h"

#include <core/kernel/cuda/flashmla/flashmla.h>
#include <core/kernel/cuda/flashv2/flashv2.h>
#include <cuda/cuda_context.h>

namespace allspark {

class MLAAttnOpCUDA : public MLAAttnOp {
 public:
  explicit MLAAttnOpCUDA(const std::string& op_type = "")
      : MLAAttnOp(op_type) {}

  virtual ~MLAAttnOpCUDA() = default;

 protected:
  AsStatus deviceInit() override;
  AsStatus deviceReshape(const RuntimeContext* runtime_ctx) override;
  AsStatus runContext(RuntimeContext* runtime_ctx) override;
  AsStatus runDecoder(RuntimeContext* runtime_ctx) override;

 private:
  cuda::FlashMLAParams flash_mla_params_;
#ifdef FLASH_ATTN_V2
  cuda::flashv2_t flash_v2_params_;
#endif
  cudaDeviceProp dprop_;

  // Workspace tensors for intermediate computations
  // q_compressed: output of q_a_proj, shape [batch, seq, q_lora_rank]
  std::unique_ptr<AsTensor> q_compressed_tensor_;
  // q_normed: output of q_a_norm, shape [batch, seq, q_lora_rank]
  std::unique_ptr<AsTensor> q_normed_tensor_;
  // q_full: output of q_b_proj, shape [batch, seq, num_heads * qk_head_dim]
  std::unique_ptr<AsTensor> q_full_tensor_;
  // kv_compressed: output of kv_a_proj, shape [batch, seq, kv_lora_rank + qk_rope_head_dim]
  std::unique_ptr<AsTensor> kv_compressed_tensor_;
  // kv_normed: output of kv_a_norm on latent part, shape [batch, seq, kv_lora_rank]
  std::unique_ptr<AsTensor> kv_normed_tensor_;
  // kv_full: output of kv_b_proj, shape [batch, seq, num_heads * (qk_nope_head_dim + v_head_dim)]
  std::unique_ptr<AsTensor> kv_full_tensor_;
  // attn_output: raw attention output before o_proj
  std::unique_ptr<AsTensor> attn_output_tensor_;

  // FlashMLA workspace buffers
  std::unique_ptr<AsTensor> splitkv_out_tensor_;
  std::unique_ptr<AsTensor> splitkv_lse_tensor_;
  std::unique_ptr<AsTensor> tile_scheduler_metadata_tensor_;
  std::unique_ptr<AsTensor> num_splits_tensor_;

  // Decoder-specific
  std::unique_ptr<AsTensor> decoder_q_tensor_;
  std::unique_ptr<AsTensor> decoder_seq_len_tensor_device_;
  std::unique_ptr<AsTensor> decoder_seq_len_tensor_host_;

  // Block table for paged attention
  std::unique_ptr<AsTensor> block_table_tensor_;
  std::unique_ptr<AsTensor> cache_seqlens_tensor_;

  // RoPE inverse frequencies for the rope dimensions
  // Shape: [qk_rope_head_dim / 2]
  std::unique_ptr<AsTensor> rope_inv_freq_tensor_;

  // Step list for RoPE position computation
  std::unique_ptr<AsTensor> step_list_tensor_;

  // Prefill attention workspace (softmax LSE)
  std::unique_ptr<AsTensor> prefill_workspace_tensor_;

  // Assembled K for prefill: [M, H, qk_head_dim] (nope + rope combined)
  std::unique_ptr<AsTensor> k_assembled_tensor_;
  // V for prefill: [M, H, v_head_dim]
  std::unique_ptr<AsTensor> v_assembled_tensor_;
  // V padded to qk_head_dim for flash-attention (which requires K and V same head dim)
  std::unique_ptr<AsTensor> v_padded_tensor_;
  // Flash-attention output with qk_head_dim per head (extracted to v_head_dim after)
  std::unique_ptr<AsTensor> flash_output_tensor_;

  // Span pointer arrays for paged cache (like span-attention)
  std::unique_ptr<AsTensor> kv_span_array_tensor_host_;
  std::unique_ptr<AsTensor> kv_span_array_tensor_device_;

  // Helper: compute RoPE inv_freq
  void computeInvFreq(float rope_base);
};

}  // namespace allspark
#endif  // ENABLE_CUDA
