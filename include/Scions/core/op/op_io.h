//
// Created by sid on 17/10/23.
//
#pragma once

#include "Scions/common/common.h"
#include "Scions/core/mem/mem_desc.h"

namespace scions::op {
/**
 * @brief Structure that represents the input and output indices information for a particular operation.
 *
 * Indexed by 1.
 *
 * This structure holds the input and output indices for a given operation. The input indices are stored in
 * an array of size_t with a maximum size of SC_OP_INPUT_MAX, and the output indices are stored in a separate
 * array of size_t with a maximum size of SC_OP_OUTPUT_MAX.
 */
struct OpIOIndicesInfo {
  const std::array<const size_t, SC_OP_INPUT_MAX> inputs;
  const std::array<const size_t, SC_OP_OUTPUT_MAX> outputs;
};

/**
 * @brief The OpIONameInfo struct provides information about the input and output names of an operation.
 */
struct OpIONameInfo {
  const std::array<const std::string_view, SC_OP_INPUT_MAX> inputs;
  const std::array<const std::string_view, SC_OP_OUTPUT_MAX> outputs;
};
} // namespace scions::op
