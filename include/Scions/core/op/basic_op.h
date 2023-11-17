//
// Created by sid on 17/10/23.
//
#pragma once
#include "Scions/common/common.h"
#include "op.h"
#include "op_ids.h"

namespace scions::op::mem {
class MemCopy : public OpDesc {
public:
  constexpr MemCopy(const std::array<const size_t, SC_OP_INPUT_MAX> &input,
    const std::array<const size_t, SC_OP_OUTPUT_MAX> &output)
    : OpDesc(MEM_COPY_OP_ID, input, output, "Memory Copy Op") {}

  constexpr MemCopy(const size_t &input1, const size_t &output)
    : OpDesc(MEM_COPY_OP_ID, { input1 }, { output }, "Memory Copy Op") {}
};

class MemFillRandom : public OpDesc {
public:
  constexpr MemFillRandom(
    const std::array<const size_t, SC_OP_INPUT_MAX> &input,
    const std::array<const size_t, SC_OP_OUTPUT_MAX> &output)
    : OpDesc(MEM_FILL_RANDOM_OP_ID,
      input,
      output,
      "Memory Fill Random Op") {}

  constexpr MemFillRandom(const size_t &input1, const size_t &output)
    : OpDesc(MEM_FILL_RANDOM_OP_ID,
      { input1 },
      { output },
      "Memory Fill Random Op") {}
};

class MemFree : public OpDesc {
public:
  constexpr MemFree(const std::array<const size_t, SC_OP_INPUT_MAX> &input,
    const std::array<const size_t, SC_OP_OUTPUT_MAX> &output)
    : OpDesc(MEM_FREE_OP_ID, input, output, "Memory Free Op") {}

  constexpr MemFree(const size_t &input1, const size_t &output)
    : OpDesc(MEM_FREE_OP_ID, { input1 }, { output }, "Memory Free Op") {}
};

} // namespace scions::op::mem
