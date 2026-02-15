/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * Copyright (c) 2025-2026 DashInfer Team.
 * @file    binary_op.cpp
 */

#include "binary_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <cpu/cpu_common.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif

namespace allspark {
AsStatus BinaryOp::Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                        const TensorMap& weights_map, TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  DataType dtype = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype);
  auto& attr_map = op_proto.attr();
  if (attr_map.find("binary_type") == attr_map.end()) {
    LOG(ERROR) << "BinaryOp : can't find binary_type attribute." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  binary_type_ = *(BinaryType*)(attr_map.at("binary_type").c_str());
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus BinaryOp::Reshape(RuntimeContext* runtime_ctx) {
  Shape out_shape = tensor_map_->at(in_names_[0])->GetShape();
  AS_CHECK_STATUS(
      tensor_map_->at(out_names_[0])->SetShape(std::move(out_shape)));
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus BinaryOp::Forward(RuntimeContext* runtime_ctx) {
  AsTensor* x_tensor = tensor_map_->at(in_names_[0]).get();
  AsTensor* y_tensor = tensor_map_->at(in_names_[1]).get();
  AsTensor* z_tensor = tensor_map_->at(out_names_[0]).get();
  int64_t count = x_tensor->GetShape().Count();

  switch (ctx_->GetDeviceType()) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      auto functor = [&]<typename T>() {
        T* typed_out = static_cast<T*>(z_tensor->GetDataPtr());
        const T* typed_in1 = static_cast<const T*>(x_tensor->GetDataPtr());
        const T* typed_in2 = static_cast<const T*>(y_tensor->GetDataPtr());
        cuda::BinaryKernelLauncher(typed_out, typed_in1, typed_in2, count,
                                   binary_type_, gpu_ctx->GetStream());
      };
      DispatchCUDA(x_tensor->GetDataType(), functor);
      break;
    }
#endif
    case DeviceType::CPU: {
      float* out = static_cast<float*>(z_tensor->GetDataPtr());
      const float* x = static_cast<const float*>(x_tensor->GetDataPtr());
      const float* y = static_cast<const float*>(y_tensor->GetDataPtr());

      if (binary_type_ == BinaryType::SWIGLU) {
        // out[i] = x[i] * SiLU(y[i]) = x[i] * y[i] / (1 + exp(-y[i]))
        cpu::parallel_for(count, [&](int64_t i) {
          float yi = y[i];
          out[i] = x[i] * yi / (1.0f + expf(-yi));
        });
      } else if (binary_type_ == BinaryType::GEGLU) {
        // out[i] = x[i] * GELU(y[i])
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        cpu::parallel_for(count, [&](int64_t i) {
          float yi = y[i];
          float gelu = 0.5f * yi *
              (1.0f + tanhf(0.7978845608f * (yi + 0.044715f * yi * yi * yi)));
          out[i] = x[i] * gelu;
        });
      } else if (binary_type_ == BinaryType::ADD) {
        cpu::parallel_for(count, [&](int64_t i) {
          out[i] = x[i] + y[i];
        });
      } else if (binary_type_ == BinaryType::MUL) {
        cpu::parallel_for(count, [&](int64_t i) {
          out[i] = x[i] * y[i];
        });
      } else {
        LOG(ERROR) << "Unsupported binary type: "
                   << BinaryType_Name(binary_type_) << std::endl;
        return AsStatus::ALLSPARK_PARAM_ERROR;
      }
      break;
    }
    default:
      break;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
REGISTER_OP(Binary, CUDA, BinaryOp)
REGISTER_OP(Binary, CPU, BinaryOp)
}  // namespace allspark
