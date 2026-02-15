/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * Copyright (c) 2025-2026 DashInfer Team.
 * @file    unary_glu_op.h
 */

#pragma once

#include <core/operator/operator.h>

#ifdef ENABLE_DNNL
#include <dnnl.hpp>
#endif

namespace allspark {

class UnaryGLUOp : public AsOperator {
 public:
  explicit UnaryGLUOp(const std::string& op_type = "") : AsOperator(op_type) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape() override;
  AsStatus Forward() override;

 private:
  UnaryType unary_type_ = UNARYTYPE_UNDEFINED;
  size_t outer_size_;
  size_t inner_size_;
};

}  // namespace allspark
