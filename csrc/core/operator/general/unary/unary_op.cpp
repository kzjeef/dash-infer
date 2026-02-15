/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * Copyright (c) 2025-2026 DashInfer Team.
 * @file    unary_op.cpp
 */

#include "unary_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <cpu/cpu_common.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif

namespace allspark {
AsStatus UnaryOp::Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                       const TensorMap& weights_map, TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  DataType dtype = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype);
  auto& attr_map = op_proto.attr();
  if (attr_map.find("unary_type") == attr_map.end()) {
    LOG(ERROR) << "UnaryOp : can't find unary_type attribute." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  unary_type_ = *(UnaryType*)(attr_map.at("unary_type").c_str());
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus UnaryOp::Reshape() {
  Shape out_shape = tensor_map_->at(in_names_[0])->GetShape();
  AS_CHECK_STATUS(
      tensor_map_->at(out_names_[0])->SetShape(std::move(out_shape)));
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus UnaryOp::Forward() {
  AsTensor* x_tensor = tensor_map_->at(in_names_[0]).get();
  AsTensor* y_tensor = tensor_map_->at(out_names_[0]).get();
  int64_t count = x_tensor->GetShape().Count();

  switch (ctx_->GetDeviceType()) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      auto functor = [&]<typename T>() {
        T* typed_out = static_cast<T*>(y_tensor->GetDataPtr());
        const T* typed_in = static_cast<const T*>(x_tensor->GetDataPtr());
        cuda::UnaryKernelLauncher(typed_out, typed_in, count, unary_type_,
                                  gpu_ctx->GetStream());
      };
      DispatchCUDA(x_tensor->GetDataType(), functor);
      break;
    }
#endif
    case DeviceType::CPU: {
      const float* in = static_cast<const float*>(x_tensor->GetDataPtr());
      float* out = static_cast<float*>(y_tensor->GetDataPtr());

      switch (unary_type_) {
        case UnaryType::SILU: {
          // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
          cpu::parallel_for(count, [&](int64_t i) {
            out[i] = in[i] / (1.0f + expf(-in[i]));
          });
          break;
        }
        case UnaryType::RELU: {
          cpu::parallel_for(count, [&](int64_t i) {
            out[i] = in[i] > 0.0f ? in[i] : 0.0f;
          });
          break;
        }
        case UnaryType::TANH: {
          cpu::parallel_for(count, [&](int64_t i) {
            out[i] = tanhf(in[i]);
          });
          break;
        }
        case UnaryType::GELU_ERF: {
          // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
          cpu::parallel_for(count, [&](int64_t i) {
            out[i] = 0.5f * in[i] * (1.0f + erff(in[i] * 0.7071067811865475f));
          });
          break;
        }
        case UnaryType::GELU_TANH: {
          // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
          cpu::parallel_for(count, [&](int64_t i) {
            float x = in[i];
            out[i] = 0.5f * x *
                (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
          });
          break;
        }
        default:
          LOG(ERROR) << "Unsupported unary type: "
                     << UnaryType_Name(unary_type_) << std::endl;
          return AsStatus::ALLSPARK_PARAM_ERROR;
      }
      break;
    }
    default:
      break;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(Unary, CUDA, UnaryOp)
REGISTER_OP(Unary, CPU, UnaryOp)
}  // namespace allspark
