/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    moe_inefficient_op.h
 */

#pragma once

#include <core/operator/operator.h>

#include <dnnl.hpp>

namespace allspark {

class MoeInefficientOp : public AsOperator {
 public:
  explicit MoeInefficientOp(const std::string& op_type = "")
      : AsOperator(op_type) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape() override;
  AsStatus Forward() override;

 private:
  DataType dtype_;
  int num_expert_;
  int num_expert_pertoken_;
  int total_token_;
  int hidden_size_;
  int64_t ws_size_;
  int block_size_;
  int proj_size_;
  int64_t expert_size_;
  bool first_moe_ = true;
  std::unique_ptr<AsTensor> experts_score_;
  std::unique_ptr<AsTensor> float_gate_score_;
  std::unique_ptr<AsTensor> topk_value_;
  std::unique_ptr<AsTensor> topk_indice_;
  std::unique_ptr<AsTensor> experts_idx_;
  std::unique_ptr<AsTensor> experts_seq_;
  std::unique_ptr<AsTensor> indice_source_;
  std::unique_ptr<AsTensor> total_tokens_post_pad_;

  std::unique_ptr<AsTensor> gate_up_proj_array_ptr;
  std::unique_ptr<AsTensor> down_proj_array_ptr;
  std::unique_ptr<AsTensor> reorder_data_array_ptr;
  std::unique_ptr<AsTensor> gate_up_proj_out_array_ptr;
  std::unique_ptr<AsTensor> mid_result_array_ptr;
  std::unique_ptr<AsTensor> final_result_array_ptr;
  //
  void* reorder_data;
  void* gate_up_proj_out;
  void* mid_result;
  void* final_result;
  void** gate_up_proj_array;
  void** down_proj_array;
  void** reorder_data_array;
  void** gate_up_proj_out_array;
  void** mid_result_array;
  void** final_result_array;
};

}  // namespace allspark