/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * Copyright (c) 2025-2026 DashInfer Team.
 * @file    gemm_op_x86_spr.cpp
 */

#if defined(__x86_64__) || defined(_M_X64)
#include "gemm_op_x86_spr.h"

#include <core/kernel/kernel.h>
#include <cpu/cpu_common.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>

#include "hgemm_f32f16f32_simple.h"

namespace allspark {

template <typename T1, typename T2>
static void convert_datatype(T1* input, T2* output, int64_t num_elements,
                             const DeviceContext& ctx) {
  const CPUContext* cpu_ctx = static_cast<const CPUContext*>(&ctx);
  int num_threads = cpu_ctx->GetNumThread();
  int64_t num_elem_per_thread = std::ceil(num_elements * 1.0 / num_threads);
  cpu::parallel_for(num_threads, [&](int n) {
    int64_t min_idx = n * num_elem_per_thread;
    int64_t max_idx = std::min((n + 1) * num_elem_per_thread, num_elements);
    for (int64_t i = min_idx; i < max_idx; i++) {
      output[i] = (T2)input[i];
    }
  });
}

AsStatus GemmOpSpr::Init(const OperatorProto& op_proto,
                         const DeviceContext& ctx, const TensorMap& weights_map,
                         TensorMap* tensor_map) {
  LOG(ERROR) << "GemmOpSpr only support InitV2()" << std::endl;
  return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
}

AsStatus GemmOpSpr::InitV2(const OperatorProto& op_proto,
                           const DeviceContext& ctx,
                           const TensorMap& weights_map,
                           TensorMap& weights_buffer, TensorMap* tensor_map,
                           RuntimeContext* runtime_ctx) {
  DLOG(INFO) << "GemmOpSpr::InitV2()" << std::endl;

  if (!is_spr_ && ctx.GetMatmulPrecision() == PrecisionLevel::MEDIUM_FP16) {
    LOG(WARNING) << "Current CPU does not support fp16 GEMM";
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  if (ctx.GetMatmulPrecision() == PrecisionLevel::MEDIUM_BF16) {
    weight_data_type_ = DataType::BFLOAT16;
  } else if (ctx.GetMatmulPrecision() == PrecisionLevel::MEDIUM_FP16) {
    weight_data_type_ = DataType::FLOAT16;
  } else {
    weight_data_type_ = DataType::FLOAT32;
  }

  if (weight_data_type_ == DataType::FLOAT32 ||
      weight_data_type_ == DataType::BFLOAT16) {
    // FP32 and BF16 both use GemmOpCPU with MKL
    // (cblas_sgemm for FP32, cblas_gemm_bf16bf16f32 for BF16)
    AS_CHECK_STATUS(GemmOpCPU::InitV2(op_proto, ctx, weights_map,
                                      weights_buffer, tensor_map, runtime_ctx));
  } else if (weight_data_type_ == DataType::FLOAT16) {
#ifdef ENABLE_FP16
    AS_CHECK_STATUS(GemmOpBase::InitV2(
        op_proto, ctx, weights_map, weights_buffer, tensor_map, runtime_ctx));

    // intel gemm for FP16
    AsTensor* mutable_weight = const_cast<AsTensor*>(weights_[0]);
    Shape weight_shape = weights_[0]->GetShape();

    auto as_weight_fp16 = std::make_unique<AsTensor>(
        weights_[0]->GetName() + "_fp16", weights_[0]->GetDeviceType(),
        DataType::FLOAT16, weights_[0]->GetDataMode(), weight_shape);
    TensorUtils::Memset(*as_weight_fp16, 0);

    float* wei_fp32_ptr = (float*)mutable_weight->GetDataPtr();
    float16_t* wei_fp16_ptr = (float16_t*)as_weight_fp16->GetDataPtr();
    convert_datatype(wei_fp32_ptr, wei_fp16_ptr,
                     mutable_weight->GetShape().Count(), ctx);

    auto as_weight_pack = std::make_unique<AsTensor>(
        weights_[0]->GetName() + "_fp16_pack", *as_weight_fp16);
    ig_hgemm_f32f16f32_packb(transB_, n_, k_,
                             (const float16_t*)as_weight_fp16->GetDataPtr(),
                             ldb_, (float16_t*)as_weight_pack->GetDataPtr());

    mutable_weight->Free();
    mutable_weight->SetName(as_weight_pack->GetName());
    mutable_weight->SetDataType(DataType::FLOAT16);
    mutable_weight->SetShape(std::move(weight_shape));
    TensorUtils::DeepCopyWholeAsync(*mutable_weight, *as_weight_pack, ctx_);
#endif
  }

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus GemmOpSpr::Reshape(RuntimeContext* runtime_ctx) {
  if (weight_data_type_ == DataType::FLOAT32 ||
      weight_data_type_ == DataType::BFLOAT16) {
    AS_CHECK_STATUS(GemmOpCPU::Reshape(runtime_ctx));
  } else if (weight_data_type_ == DataType::FLOAT16) {
    AS_CHECK_STATUS(GemmOpBase::Reshape(n_));
  } else {
    LOG(ERROR) << "Unsupported matmul precision";
    return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus GemmOpSpr::Forward(RuntimeContext* runtime_ctx) {
  if (weight_data_type_ == DataType::FLOAT32 ||
      weight_data_type_ == DataType::BFLOAT16) {
    // Both FP32 and BF16 handled by GemmOpCPU (MKL)
    AS_CHECK_STATUS(GemmOpCPU::Forward(runtime_ctx));
  } else if (weight_data_type_ == DataType::FLOAT16) {
#ifdef ENABLE_FP16
    AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
    void* in = in_tensor->GetDataPtr();
    void* out = tensor_map_->at(out_names_[0])->GetDataPtr();
    void* bin_in = (in_names_.size() == 2)
                       ? tensor_map_->at(in_names_[1])->GetDataPtr()
                       : nullptr;
    if (is_split_k_) {
      in = (char*)in + k_ * rank_id_ * SizeofType(dtype_);
    }
    const AsTensor* weight = static_cast<const AsTensor*>(weights_[0]);
    void* bias = (weights_.size() == 2) ? weights_[1]->GetDataPtr() : nullptr;
    if (bias) {
      if (activation_ == UnaryType::RELU) {
        ig_hgemm_f32f16f32_compute_biasadd_relu(
            transB_, m_, n_, k_, 1.0f, (const float*)in, lda_,
            (const float16_t*)weight->GetDataPtr(), 0.0f, (float*)out, ldc_,
            (const float*)bias);
      } else if (binary_type_ == BinaryType::ADD) {
        ig_hgemm_f32f16f32_compute_residential(
            transB_, m_, n_, k_, 1.0f, (const float*)in, lda_,
            (const float16_t*)weight->GetDataPtr(), 0.0f, (float*)out, ldc_,
            (const float*)bias, (const float*)bin_in, ldc_);
      } else {
        ig_hgemm_f32f16f32_compute_biasadd(
            transB_, m_, n_, k_, 1.0f, (const float*)in, lda_,
            (const float16_t*)weight->GetDataPtr(), 0.0f, (float*)out, ldc_,
            (const float*)bias);
      }
    } else {
      if (activation_ == UnaryType::SILU) {
        ig_hgemm_f32f16f32_compute_silu(
            transB_, m_, n_, k_, 1.0f, (const float*)in, lda_,
            (const float16_t*)weight->GetDataPtr(), 0.0f, (float*)out, ldc_);
      } else if (binary_type_ == BinaryType::ADD) {
        ig_hgemm_f32f16f32_compute_residential(
            transB_, m_, n_, k_, 1.0f, (const float*)in, lda_,
            (const float16_t*)weight->GetDataPtr(), 0.0f, (float*)out, ldc_,
            (const float*)bias, (const float*)bin_in, ldc_);
      } else if (binary_type_ == BinaryType::MUL) {
        ig_hgemm_f32f16f32_compute_resmul(
            transB_, m_, n_, k_, 1.0f, (const float*)in, lda_,
            (const float16_t*)weight->GetDataPtr(), 0.0f, (float*)out, ldc_,
            (const float*)bin_in, ldc_);
      } else {
        ig_hgemm_f32f16f32_compute(transB_, m_, n_, k_, 1.0f, (const float*)in,
                                   lda_, (const float16_t*)weight->GetDataPtr(),
                                   0.0f, (float*)out, ldc_);
      }
    }
#endif
  } else {
    LOG(ERROR) << "Unsupported matmul precision";
    return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(Gemm, CPU, GemmOpSpr)
}  // namespace allspark
#endif
