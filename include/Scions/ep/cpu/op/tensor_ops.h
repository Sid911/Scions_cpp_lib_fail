//
// Created by sid on 13/11/23.
//
#pragma once

#include "Scions/core/op/op.h"
#include "Scions/ep/cpu/mem/mem_manager.h"
#include "Scions/ep/common/common.h"

namespace scions::ep::cpu::_internal {
template<size_t Mem, uint64_t Size>
void tensorAdd(const op::OpDesc &desc, CpuMemoryManager<Mem, Size> &manager) {
  auto &inputs          = desc.info.inputs;
  auto &outputs         = desc.info.outputs;
  const auto numInputs  = desc.num_inputs;
  const auto numOutputs = desc.num_outputs;

  if (numInputs == 2 && numOutputs == 1) {
    const CpuMemRef& one_ref = manager.mem_refs.at(inputs[0]);
    const CpuMemRef& two_ref = manager.mem_refs.at(inputs[1]);
    const CpuMemRef& out_ref = manager.mem_refs.at(outputs[0]);

  }

}

template<size_t Mem, uint64_t Size>
void tensorMul(const op::OpDesc &desc, CpuMemoryManager<Mem, Size> &manager) {
  auto &inputs  = desc.info.inputs;
  auto &outputs = desc.info.outputs;

  fmt::print("Inputs :");
  for (size_t input : inputs) { if (input) fmt::print(" {} ", input); }
  fmt::print("\n");

  fmt::print("Outputs :");
  for (size_t output : outputs) { if (output) fmt::print(" {} ", output); }
  fmt::print("\n");
}
}
