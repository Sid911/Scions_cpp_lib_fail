//
// Created by sid on 13/11/23.
//
#pragma once
#include "Scions/core/op/op_ids.h"
#include "Scions/core/op/tensor_op.h"
#include "Scions/ep/cpu/mem/mem_manager.h"
#include "tensor_ops.h"

namespace scions::ep::cpu {
template<size_t Mem, uint64_t Size>
inline void executeOp(const op::OpDesc &op, CpuMemoryManager<Mem, Size> &manager) {
    using namespace scions::op::tensor;

    switch (op.op_id) {
    case TENSOR_ADD_OP_ID:
        _internal::tensorAdd(op, manager);
        break;
    case TENSOR_MULTIPLY_OP_ID:
        _internal::tensorMul(op, manager);
        break;
    default:
        print(fg(fmt::color::blue_violet), "CPU EP: ");
        fmt::print("Op ID \"{}\" Either not implemented in CPU EP or does not exist\n", op.op_id);
        throw std::runtime_error("Cpu EP: OP not found");
    }
}
}  // namespace scions::ep::cpu
