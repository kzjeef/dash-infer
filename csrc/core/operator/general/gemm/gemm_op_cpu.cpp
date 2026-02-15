/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * Copyright (c) 2025-2026 DashInfer Team.
 * @file    gemm_op_cpu.cpp
 */

#include "gemm_op_cpu.h"

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <cpu/cpu_common.h>
#include <utility/datatype_dispatcher.h>

namespace allspark {

// Fast f32 -> bf16 conversion with round-to-nearest-even
static inline void f32_to_bf16(const float* src, uint16_t* dst, int64_t count) {
  for (int64_t i = 0; i < count; i++) {
    uint32_t bits;
    memcpy(&bits, &src[i], 4);
    uint32_t rounding = (bits >> 16) & 1;
    bits += 0x7FFF + rounding;
    dst[i] = (uint16_t)(bits >> 16);
  }
}

AsStatus GemmOpCPU::Init(const OperatorProto& op_proto,
                         const DeviceContext& ctx, const TensorMap& weights_map,
                         TensorMap* tensor_map) {
  LOG(ERROR) << "GemmOpCPU only support InitV2()" << std::endl;
  return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
}

AsStatus GemmOpCPU::InitV2(const OperatorProto& op_proto,
                           const DeviceContext& ctx,
                           const TensorMap& weights_map,
                           TensorMap& weights_buffer, TensorMap* tensor_map,
                           RuntimeContext* runtime_ctx) {
  AS_CHECK_STATUS(GemmOpBase::InitV2(op_proto, ctx, weights_map, weights_buffer,
                                     tensor_map, runtime_ctx));

  // For BF16 mode: convert FP32 weights to BF16 once and cache.
  // If weights are already BF16 (pre-stored), no conversion needed.
  if (weight_data_type_ == DataType::BFLOAT16 &&
      weights_[0]->GetDataType() != DataType::BFLOAT16) {
    int64_t w_count = weights_[0]->GetShape().Count();
    w_bf16_cache_.resize(w_count);
    f32_to_bf16(static_cast<const float*>(weights_[0]->GetDataPtr()),
                w_bf16_cache_.data(), w_count);
  }

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus GemmOpCPU::Reshape(RuntimeContext* runtime_ctx) {
  int yn = n_;
  AS_CHECK_STATUS(GemmOpBase::Reshape(yn));

  // Pre-allocate BF16 input buffer for the current m_ dimension.
  // This avoids malloc/free in every Forward call.
  if (weight_data_type_ == DataType::BFLOAT16) {
    int64_t in_count = m_ * k_;
    if ((int64_t)in_bf16_buf_.size() < in_count) {
      in_bf16_buf_.resize(in_count);
    }
  }

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus GemmOpCPU::Forward(RuntimeContext* runtime_ctx) {
  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  void* in = in_tensor->GetDataPtr();
  void* out = tensor_map_->at(out_names_[0])->GetDataPtr();
  void* bias = (weights_.size() == 2) ? weights_[1]->GetDataPtr() : nullptr;
  void* bin_in = (in_names_.size() == 2)
                     ? tensor_map_->at(in_names_[1])->GetDataPtr()
                     : nullptr;

  if (is_split_k_) {
    in = (char*)in + k_ * rank_id_ * SizeofType(dtype_);
  }

  if (weight_data_type_ == DataType::BFLOAT16) {
    // BF16 path: MKL cblas_gemm_bf16bf16f32 (AMX-accelerated)
    int64_t in_count = m_ * k_;
    f32_to_bf16(static_cast<const float*>(in), in_bf16_buf_.data(), in_count);

    const MKL_BF16* w_bf16_ptr;
    if (weights_[0]->GetDataType() == DataType::BFLOAT16) {
      w_bf16_ptr = reinterpret_cast<const MKL_BF16*>(
          weights_[0]->GetDataPtr());
    } else {
      w_bf16_ptr = reinterpret_cast<const MKL_BF16*>(w_bf16_cache_.data());
    }

    float* out_f = static_cast<float*>(out);
    float beta = 0.0f;
    if (bias) {
      const float* b = static_cast<const float*>(bias);
      for (int64_t i = 0; i < m_; i++)
        memcpy(out_f + i * n_, b, n_ * sizeof(float));
      beta = 1.0f;
    }
    if (bin_in) {
      memcpy(out_f, bin_in, m_ * n_ * sizeof(float));
      beta = 1.0f;
    }

    CBLAS_TRANSPOSE transB_flag = transB_ ? CblasTrans : CblasNoTrans;
    cblas_gemm_bf16bf16f32(
        CblasRowMajor, CblasNoTrans, transB_flag,
        m_, n_, k_, alpha_,
        reinterpret_cast<const MKL_BF16*>(in_bf16_buf_.data()), lda_,
        w_bf16_ptr, ldb_,
        beta, out_f, n_);
  } else {
    // FP32 path: use cblas_sgemv for decode (m=1) — avoids GEMM API overhead
    float* out_f = static_cast<float*>(out);
    const float* in_f = static_cast<const float*>(in);
    const float* w_f = static_cast<const float*>(weights_[0]->GetDataPtr());

    if (m_ == 1 && !bin_in) {
      // Decode (m=1): matrix-vector multiply is faster via sgemv
      // y = alpha * W^T * x + beta * y  (or y = alpha * W * x)
      if (bias) {
        memcpy(out_f, bias, n_ * sizeof(float));
        if (transB_) {
          // W is [n, k], compute out = alpha * W * x + bias
          cblas_sgemv(CblasRowMajor, CblasNoTrans, n_, k_, alpha_,
                      w_f, ldb_, in_f, 1, 1.0f, out_f, 1);
        } else {
          // W is [k, n], compute out = alpha * W^T * x + bias
          cblas_sgemv(CblasRowMajor, CblasTrans, k_, n_, alpha_,
                      w_f, ldb_, in_f, 1, 1.0f, out_f, 1);
        }
      } else {
        if (transB_) {
          cblas_sgemv(CblasRowMajor, CblasNoTrans, n_, k_, alpha_,
                      w_f, ldb_, in_f, 1, 0.0f, out_f, 1);
        } else {
          cblas_sgemv(CblasRowMajor, CblasTrans, k_, n_, alpha_,
                      w_f, ldb_, in_f, 1, 0.0f, out_f, 1);
        }
      }
    } else {
      // Prefill (m>1) or binary: use cblas_sgemm via GemmWraper
      cpu::GemmWraper<float>(
          out_f, in_f, w_f,
          static_cast<const float*>(bias),
          m_, n_, k_,
          false, transB_,
          lda_, ldb_, n_,
          alpha_, 0.0f,
          static_cast<const float*>(bin_in));
    }
  }

  // Post-op: fused activation
  if (activation_ != UNARYTYPE_UNDEFINED) {
    float* out_f = static_cast<float*>(out);
    int64_t total = m_ * n_;
    if (activation_ == UnaryType::SILU) {
      for (int64_t i = 0; i < total; i++)
        out_f[i] = out_f[i] / (1.0f + expf(-out_f[i]));
    } else if (activation_ == UnaryType::RELU) {
      for (int64_t i = 0; i < total; i++)
        out_f[i] = out_f[i] > 0.0f ? out_f[i] : 0.0f;
    } else if (activation_ == UnaryType::GELU_ERF) {
      for (int64_t i = 0; i < total; i++)
        out_f[i] = 0.5f * out_f[i] * (1.0f + erff(out_f[i] * 0.7071067811865475f));
    } else if (activation_ == UnaryType::TANH) {
      for (int64_t i = 0; i < total; i++)
        out_f[i] = tanhf(out_f[i]);
    }
  }

  return AsStatus::ALLSPARK_SUCCESS;
}

}  // namespace allspark
