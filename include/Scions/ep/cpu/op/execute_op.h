//
// Created by sid on 13/11/23.
//
#pragma once
#include "core/op/tensor_op.h"
#include "core/op/op_ids.h"
#include "tensor_ops.h"

namespace scions::ep::cpu{
inline void executeOp(op::OpDesc& op, std::span<uint8_t *>& memory){
    using namespace scions::op::tensor;

    switch (op.op_id) {
    case TENSOR_ADD_OP_ID:
        _internal::tensorAdd(op, memory);
        break;
    case TENSOR_MULTIPLY_OP_ID:
        break;
    default:
        std::cout << "CPU EP: Op ID \"" << op.op_id << "\" Either not implemented in CPU EP"
        << " or does not exist\n";
    }
}
}