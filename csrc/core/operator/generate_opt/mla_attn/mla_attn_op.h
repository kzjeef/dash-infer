/*!
 * Copyright contributors to the DashInfer Project
 * @file    mla_attn_op.h
 */
#pragma once

#include <memory>

#include "core/operator/operator.h"

namespace allspark {

/**
 * @brief Multi-head Latent Attention (MLA) operator for DeepSeek V3.
 *
 * MLA compresses the KV cache by storing a low-rank latent representation
 * instead of the full K and V tensors. This reduces KV cache size by ~28x.
 *
 * Architecture:
 *   - KV latent dim (kv_lora_rank): 512
 *   - Q compression rank (q_lora_rank): 1536
 *   - QK nope head dim: 128
 *   - QK rope head dim: 64
 *   - V head dim: 128
 *   - Num heads: 128
 *   - Num KV heads: 1 (MQA-like, single latent shared)
 *
 * KV cache per token: 512 (latent) + 64 (k_rope) = 576 dimensions in bf16.
 *
 * Inputs:
 *   [0] hidden_states: (batch, seq_len, hidden_size)
 *   [1] attention_mask
 *   [2] gen_ctx (when generating)
 *
 * Outputs:
 *   [0] attn_out: (batch, seq_len, hidden_size)
 *
 * The operator expects the following weight tensors in tensor_map:
 *   - {prefix}.q_a_proj.weight    : (hidden_size, q_lora_rank)
 *   - {prefix}.q_a_norm.gamma     : (q_lora_rank,)
 *   - {prefix}.q_b_proj.weight    : (q_lora_rank, num_heads * (qk_nope_head_dim + qk_rope_head_dim))
 *   - {prefix}.kv_a_proj.weight   : (hidden_size, kv_lora_rank + qk_rope_head_dim)
 *   - {prefix}.kv_a_norm.gamma    : (kv_lora_rank,)
 *   - {prefix}.kv_b_proj.weight   : (kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim))
 *   - {prefix}.output.dense.weight: (num_heads * v_head_dim, hidden_size)
 */
class MLAAttnOp : public AsOperator {
 public:
  explicit MLAAttnOp(const std::string& op_type = "")
      : AsOperator(op_type),
        dtype_(DATATYPE_UNDEFINED),
        layer_num_(0),
        batch_size_(1),
        seq_len_(1),
        kv_lora_rank_(512),
        q_lora_rank_(1536),
        qk_nope_head_dim_(128),
        qk_rope_head_dim_(64),
        v_head_dim_(128),
        num_heads_(128),
        hidden_size_(7168),
        causal_mask_(true) {}

  virtual ~MLAAttnOp() {}

  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map,
                TensorMap* tensor_map) override final;
  AsStatus Reshape(RuntimeContext* runtime_ctx) override final;
  AsStatus Forward(RuntimeContext* runtime_ctx) override final;
  AsStatus Alloc(RuntimeContext* runtime_ctx) override final;

 protected:
  virtual AsStatus deviceInit() = 0;
  virtual AsStatus deviceReshape(const RuntimeContext* runtime_ctx) = 0;
  virtual AsStatus runContext(RuntimeContext* runtime_ctx) = 0;
  virtual AsStatus runDecoder(RuntimeContext* runtime_ctx) = 0;

 protected:
  DataType dtype_;
  int layer_num_;
  int batch_size_;
  int seq_len_;

  // MLA architecture dimensions
  int kv_lora_rank_;       // 512: latent dimension for KV compression
  int q_lora_rank_;        // 1536: query compression rank
  int qk_nope_head_dim_;   // 128: non-RoPE part of QK head dim
  int qk_rope_head_dim_;   // 64: RoPE part of QK head dim
  int v_head_dim_;         // 128: value head dimension
  int num_heads_;          // 128: number of attention heads
  int hidden_size_;        // 7168: model hidden size

  bool causal_mask_;

  // RoPE base frequency (default 10000 for DeepSeek V3)
  float rope_base_ = 10000.0f;

  // KV cache dimension per token = kv_lora_rank + qk_rope_head_dim = 576
  int kv_cache_dim() const { return kv_lora_rank_ + qk_rope_head_dim_; }
  // Full QK head dim = qk_nope_head_dim + qk_rope_head_dim = 192
  int qk_head_dim() const { return qk_nope_head_dim_ + qk_rope_head_dim_; }
};

}  // namespace allspark
