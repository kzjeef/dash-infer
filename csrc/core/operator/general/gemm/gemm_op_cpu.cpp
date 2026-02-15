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

  // Only fuse binary ADD with GEMM; other binary ops applied as post-ops
  void* gemm_bin_in = (binary_type_ == BinaryType::ADD) ? bin_in : nullptr;

  if (is_split_k_) {
    in = (char*)in + k_ * rank_id_ * SizeofType(dtype_);
  }

  if (weight_data_type_ == DataType::BFLOAT16) {
    // BF16 path: MKL cblas_gemm_bf16bf16f32 (AMX-accelerated)
    // Respect lda_ stride when split-K is enabled
    const float* in_f = static_cast<const float*>(in);
    uint16_t* dst = in_bf16_buf_.data();
    for (int64_t row = 0; row < m_; row++) {
      f32_to_bf16(in_f + row * lda_, dst + row * k_, k_);
    }

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
    if (gemm_bin_in) {
      memcpy(out_f, gemm_bin_in, m_ * n_ * sizeof(float));
      beta = 1.0f;
    }

    CBLAS_TRANSPOSE transB_flag = transB_ ? CblasTrans : CblasNoTrans;
    cblas_gemm_bf16bf16f32(
        CblasRowMajor, CblasNoTrans, transB_flag,
        m_, n_, k_, alpha_,
        reinterpret_cast<const MKL_BF16*>(in_bf16_buf_.data()), k_,
        w_bf16_ptr, ldb_,
        beta, out_f, n_);
  } else {
    // FP32 path: MKL cblas_sgemm (faster than sgemv even for m=1 with many threads)
    cpu::GemmWraper<float>(
        static_cast<float*>(out),
        static_cast<const float*>(in),
        static_cast<const float*>(weights_[0]->GetDataPtr()),
        static_cast<const float*>(bias),
        m_, n_, k_,
        false, transB_,
        lda_, ldb_, n_,
        alpha_, 0.0f,
        static_cast<const float*>(gemm_bin_in));
  }

  // Post-op: non-ADD binary operations
  if (bin_in && binary_type_ != BinaryType::ADD) {
    float* out_f = static_cast<float*>(out);
    const float* bin_f = static_cast<const float*>(bin_in);
    int64_t total = m_ * n_;
    if (binary_type_ == BinaryType::MUL) {
      for (int64_t i = 0; i < total; i++)
        out_f[i] *= bin_f[i];
    } else {
      LOG(WARNING) << "Unsupported binary_type " << (int)binary_type_
                   << " for CPU GEMM; ignoring binary input";
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
