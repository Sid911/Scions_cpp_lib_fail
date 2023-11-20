//
// Created by sid on 13/10/23.
//

#pragma once

#include <utility>

#include "Scions/common/common.h"
#include "op_io.h"

namespace scions::op {

/**
 * @class OpDesc
 * @brief Represents the description of an operation.
 *
 * This class contains the information about an operation, including its unique identifier, input and output indices, and description.
 */
class OpDesc {
public:
  int32_t op_id;
  OpIOIndicesInfo info;
  uint64_t num_inputs;
  uint64_t num_outputs;

  /**
   * @brief Constructs an OpDesc object.
   *
   * This constructor is used to create an OpDesc object with the specified parameters.
   *
   * @param id The unique identifier of the operation.
   * @param info_ The information about input and output indices of the operation.
   * @param desc The description of the operation.
   */
  constexpr OpDesc(int32_t id, OpIOIndicesInfo info_, const std::string_view &desc)
    : op_id(id), op_desc(desc), info(info_) {}

  /**
   * @brief Constructs an OpDesc object.
   *
   * This constructor is used to create an OpDesc object with the specified parameters.
   *
   * @param id The unique identifier of the operation.
   * @param inp The array of input indices of the operation.
   * @param out The array of output indices of the operation.
   * @param desc The description of the operation.
   */
  constexpr OpDesc(int32_t id,
    const std::array<const size_t, SC_OP_INPUT_MAX> &inp,
    const std::array<const size_t, SC_OP_OUTPUT_MAX> &out,
    const std::string_view &desc)
    : op_id(id), op_desc(desc), info({ inp, out }) {
    num_inputs = std::distance(inp.begin(),std::ranges::find(inp,0)) + 1;
    num_outputs = std::distance(out.begin(),std::ranges::find(out,0)) + 1;
  }

  /**
   * @brief Returns the description of the operation.
   *
   * This function returns a constexpr std::string_view that represents the description of the operation.
   *
   * @return A constexpr std::string_view containing the description of the operation.
   */
  constexpr std::string_view getDesc() { return op_desc; }

private:
  const std::string_view op_desc;
};

} // namespace scions::op
// op
