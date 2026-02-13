/*!
 * Copyright contributors to the DashInfer Project
 * @file    mla_attn_op.cpp
 */

#include "mla_attn_op.h"

#include <cmath>

#include "core/kernel/kernel.h"
#include "utility/datatype_dispatcher.h"

namespace allspark {

AsStatus MLAAttnOp::Init(const OperatorProto& op_proto,
                         const DeviceContext& ctx,
                         const TensorMap& weights_map,
                         TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));

  layer_num_ = get_layer_num(this->op_name_);
  if (layer_num_ < 0) {
    LOG(ERROR) << "MLAAttnOp: cannot get layer_num from op name";
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  // type inference
  dtype_ = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype_);

  // Load MLA-specific attributes from op_proto
  auto& attr_map = op_proto.attr();

  if (attr_map.find("kv_lora_rank") != attr_map.end()) {
    kv_lora_rank_ = *(int*)(attr_map.at("kv_lora_rank").c_str());
  }
  if (attr_map.find("q_lora_rank") != attr_map.end()) {
    q_lora_rank_ = *(int*)(attr_map.at("q_lora_rank").c_str());
  }
  if (attr_map.find("qk_nope_head_dim") != attr_map.end()) {
    qk_nope_head_dim_ = *(int*)(attr_map.at("qk_nope_head_dim").c_str());
  }
  if (attr_map.find("qk_rope_head_dim") != attr_map.end()) {
    qk_rope_head_dim_ = *(int*)(attr_map.at("qk_rope_head_dim").c_str());
  }
  if (attr_map.find("v_head_dim") != attr_map.end()) {
    v_head_dim_ = *(int*)(attr_map.at("v_head_dim").c_str());
  }
  if (attr_map.find("num_heads") != attr_map.end()) {
    num_heads_ = *(int*)(attr_map.at("num_heads").c_str());
  }

  if (attr_map.find("rope_base") != attr_map.end()) {
    rope_base_ = *(float*)(attr_map.at("rope_base").c_str());
  }

  hidden_size_ = ctx_->GetNumberHeads() * ctx_->GetSizePerHead();
  num_heads_ = ctx_->GetNumberHeads();

  causal_mask_ = true;

  AS_CHECK_STATUS(deviceInit());

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus MLAAttnOp::Reshape(RuntimeContext* runtime_ctx) {
  const AsTensor* x = tensor_map_->at(in_names_[0]).get();
  const Shape& x_shape = x->GetShape();
  batch_size_ = x_shape[0];
  seq_len_ = x_shape[1];

  // Output shape: (batch, seq_len, hidden_size)
  // hidden_size = num_heads * v_head_dim
  int output_hidden = num_heads_ * v_head_dim_;
  Shape y_shape({batch_size_, seq_len_, output_hidden});
  tensor_map_->at(out_names_[0])->SetShape(std::move(y_shape));

  AS_CHECK_STATUS(deviceReshape(runtime_ctx));

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus MLAAttnOp::Forward(RuntimeContext* runtime_ctx) {
  if (runtime_ctx->is_context) {
    return runContext(runtime_ctx);
  } else {
    return runDecoder(runtime_ctx);
  }
}

AsStatus MLAAttnOp::Alloc(RuntimeContext* runtime_ctx) {
  // MLA uses the same paged KV cache infrastructure as span-attention.
  // Cache allocation is handled by the virtual_k_cache / virtual_v_cache
  // in GenerateContext, with kv_cache_dim() elements per token per head.
  const int max_seq_len = ctx_->GetModelMaxLength();
  const int span_len = ctx_->GetCacheSpanSize();

  if (runtime_ctx->is_context) {
    GenerateContext* gen_ctx = runtime_ctx->GetContextGenCtx();
    const int cache_increment = seq_len_;
    (void)gen_ctx->virtual_k_cache->GetCache(layer_num_, cache_increment);
    (void)gen_ctx->virtual_v_cache->GetCache(layer_num_, cache_increment);
  } else {
    for (int batch = 0; batch < batch_size_; batch++) {
      GenerateContext* gen_ctx = runtime_ctx->GetGenCtx(batch);
      const int cache_increment = seq_len_;
      (void)gen_ctx->virtual_k_cache->GetCache(layer_num_, cache_increment);
      (void)gen_ctx->virtual_v_cache->GetCache(layer_num_, cache_increment);
    }
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

}  // namespace allspark
