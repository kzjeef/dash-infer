/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * Copyright (c) 2025-2026 DashInfer Team.
 * @file    binary_op.h
 */

#pragma once

#include <core/operator/operator.h>

#ifdef ENABLE_DNNL
#include <dnnl.hpp>
#endif

namespace allspark {

class BinaryOp : public AsOperator {
 public:
  explicit BinaryOp(const std::string& op_type = "") : AsOperator(op_type) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape(RuntimeContext* runtime_ctx);
  AsStatus Forward(RuntimeContext* runtime_ctx);

 private:
  BinaryType binary_type_ = BINARYTYPE_UNDEFINED;
};

}  // namespace allspark
