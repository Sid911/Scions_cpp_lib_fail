#pragma once
#include "op_type.hpp"
#include "tensor.hpp"

namespace manifold {
struct ExpressionReflection {
  // Expression part
  using INP_TYPE   = std::array<uint32_t, MANIFOLD_MAX_EXP_REF_INPUT>;
  using OUT_TYPE   = std::array<uint32_t, MANIFOLD_MAX_EXP_REF_OUTPUT>;
  using PARAM_TYPE = std::array<std::byte, MANIFOLD_PARAM_BYTES_MAX>;

  // OpType of the expression
  OpType type;
  uint32_t id;
  uint32_t num_inputs;
  uint32_t num_outputs;
  INP_TYPE inputs;
  OUT_TYPE outputs;
  PARAM_TYPE params;

  constexpr ExpressionReflection() : type(), num_inputs(0), num_outputs(0) {}

  [[nodiscard]] constexpr ExpressionReflection(uint32_t _id,
    OpType _type,
    uint32_t _num_inputs,
    INP_TYPE &_inputs,
    uint32_t _num_outputs,
    OUT_TYPE &_outputs,
    PARAM_TYPE &_params) noexcept
    : id(_id), type(_type), num_inputs(_num_inputs), num_outputs(_num_outputs), inputs(_inputs), outputs(_outputs),
      params(std::move(_params)) {}

#undef inp_type
#undef out_type
#undef param_type
};
#pragma endregion
}  // namespace manifold