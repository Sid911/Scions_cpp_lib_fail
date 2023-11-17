//
// Created by sid on 13/11/23.
//
#pragma once

#include "Scions/core/op/op.h"
#include "Scions/ep/cpu/mem/mem_manager.h"

namespace scions::ep::cpu::_internal{
template<size_t Mem, uint64_t Size>
inline void tensorAdd(op::OpDesc& desc, CpuMemoryManager<Mem, Size>& manager) {

}
template<size_t Mem, uint64_t Size>
inline void tensorMul(op::OpDesc& desc, CpuMemoryManager<Mem, Size>& manager) {

}
}