/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * Copyright (c) 2025-2026 DashInfer Team.
 * @file    get_last_line.cpp
 */

#include "get_last_line.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>
#ifdef ENABLE_DNNL
using dnnl::memory;
#endif

namespace allspark {

AsStatus GetLastLineOp::Init(const OperatorProto& op_proto,
                             const DeviceContext& ctx,
                             const TensorMap& weights_map,
                             TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // type inference
  DataType dtype = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype);
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus GetLastLineOp::Reshape(RuntimeContext* runtime_ctx) {
  Shape in_shape = tensor_map_->at(in_names_[0])->GetShape();
  batch_ = in_shape[0];
  seq_len_ = in_shape[1];
  hidden_size_ = in_shape[2];

  // When prompt_logprobs is enabled during prefill, pass full sequence through
  // so that lm_head computes logits for all positions (not just the last one).
  bool pass_full_seq = false;
  if (runtime_ctx->is_context) {
    GenerateContext* gen_ctx = runtime_ctx->GetContextGenCtx();
    if (0 > 0) {
      pass_full_seq = true;
    }
  }

  if (pass_full_seq) {
    // Full sequence: [batch, seq_len, hidden_size]
    AS_CHECK_STATUS(tensor_map_->at(out_names_[0])
                        ->SetShape(Shape({batch_, seq_len_, hidden_size_})));
  } else {
    // WARMUP: Pre-allocate for the worst case.
    // When prompt_logprobs may be used at runtime, the output could be
    // [1, max_prefill_length, hidden_size] which is larger than the default
    // [max_batch, 1, hidden_size]. Allocate for whichever is bigger so that
    // later SetShape calls don't trigger runtime cudaMalloc.
    int max_batch = ctx_->GetModelMaxBatch();
    int max_prefill = ctx_->GetModelMaxPrefillLength();
    // max elements: max(max_batch * 1, 1 * max_prefill) * hidden_size
    int warmup_dim0 = std::max(max_batch, max_prefill);
    AS_CHECK_STATUS(tensor_map_->at(out_names_[0])
                        ->SetShape(Shape({warmup_dim0, 1,
                                          hidden_size_})));  // WARMUP
    AS_CHECK_STATUS(tensor_map_->at(out_names_[0])
                        ->SetShape(Shape({batch_, 1, hidden_size_})));
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
void GetLastLineOp::UpdateHiddenStates(RuntimeContext* runtime_ctx,
                                       AsTensor* out_tensor) {
  if (rank_info_.rank_id == 0) {
    std::string tensor_name = "hidden_states";
    if (runtime_ctx->is_context) {
      GenerateContext* gen_ctx = runtime_ctx->GetContextGenCtx();
      if (gen_ctx->gen_cfg.enable_tensors_from_model_inference) {
        std::shared_ptr<AsTensor> output_tensor = std::make_shared<AsTensor>(
            tensor_name, DeviceType::CPU, out_tensor->GetDataType(),
            DataMode::DENSE, Shape({seq_len_, hidden_size_}));
        int data_size = SizeofType(output_tensor->GetDataType());
        CopyData(output_tensor->GetDataPtr(), output_tensor->GetDeviceType(),
                 out_tensor->GetDataPtr(), out_tensor->GetDeviceType(),
                 seq_len_ * hidden_size_ * data_size, ctx_);
        gen_ctx->request->tensors_from_model_inference_list[tensor_name]
            .push_back(output_tensor);
      }
    } else {
      int batch_size = runtime_ctx->GetGenCtxListSize();
      for (int i = 0; i < batch_size; i++) {
        GenerateContext* gen_ctx = runtime_ctx->GetGenCtx(i);
        if (gen_ctx->gen_cfg.enable_tensors_from_model_inference) {
          std::shared_ptr<AsTensor> output_tensor = std::make_shared<AsTensor>(
              tensor_name, DeviceType::CPU, out_tensor->GetDataType(),
              DataMode::DENSE, Shape({seq_len_, hidden_size_}));
          int data_size = SizeofType(output_tensor->GetDataType());
          CopyData(output_tensor->GetDataPtr(), output_tensor->GetDeviceType(),
                   out_tensor->GetDataPtr() + i * hidden_size_ * data_size,
                   out_tensor->GetDeviceType(), hidden_size_ * data_size, ctx_);
          gen_ctx->request->tensors_from_model_inference_list[tensor_name]
              .push_back(output_tensor);
        }
      }
    }
  }
}

AsStatus GetLastLineOp::Forward(RuntimeContext* runtime_ctx) {
  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  AsTensor* out_tensor = tensor_map_->at(out_names_[0]).get();

  // When prompt_logprobs is enabled during prefill, copy the full sequence
  // so lm_head can compute logits for all positions.
  bool pass_full_seq = false;
  if (runtime_ctx->is_context) {
    GenerateContext* gen_ctx = runtime_ctx->GetContextGenCtx();
    if (0 > 0) {
      pass_full_seq = true;
    }
  }

  if (pass_full_seq) {
    // Copy full sequence: [batch, seq_len, hidden_size]
    out_tensor->CopyDataFrom(
        in_tensor->GetDataPtr(),
        batch_ * seq_len_ * hidden_size_ *
            SizeofType(in_tensor->GetDataType()),
        ctx_->GetDeviceType(), ctx_);
  } else {
    // Default: copy only the last position
    out_tensor->CopyDataFrom(
        (char*)in_tensor->GetDataPtr() +
            (seq_len_ - 1) * hidden_size_ *
                SizeofType(in_tensor->GetDataType()),
        batch_ * 1 * hidden_size_ * SizeofType(in_tensor->GetDataType()),
        ctx_->GetDeviceType(), ctx_);
  }
  UpdateHiddenStates(runtime_ctx, in_tensor);
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(GetLastLine, CUDA, GetLastLineOp)
REGISTER_OP(GetLastLine, CPU, GetLastLineOp)
}  // namespace allspark
