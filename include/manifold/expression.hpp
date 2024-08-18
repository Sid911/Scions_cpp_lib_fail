#pragma once
#include "manifold/constants.hpp"
#include "op_type.hpp"
#include <print>

namespace manifold {
struct ExpressionReflection {
  // Expression part
  using INP_TYPE   = std::array<uint32_t, MANIFOLD_MAX_EXP_INPUT>;
  using OUT_TYPE   = std::array<uint32_t, MANIFOLD_MAX_EXP_OUTPUT>;
  using PARAM_TYPE = std::array<std::byte, MANIFOLD_PARAM_BYTES_MAX>;

  // OpType of the expression
  OpType type;
  DType data_type;
  uint32_t id;
  uint32_t num_inputs;
  uint32_t num_outputs;
  INP_TYPE inputs;
  OUT_TYPE outputs;
  PARAM_TYPE params;

  constexpr ExpressionReflection() = default;

  [[nodiscard]] constexpr ExpressionReflection(uint32_t _id,
    OpType _type,
    DType _data_type,
    uint32_t _num_inputs,
    INP_TYPE &_inputs,
    uint32_t _num_outputs,
    OUT_TYPE &_outputs,
    PARAM_TYPE &_params) noexcept
    : id(_id), type(_type), data_type(_data_type), num_inputs(_num_inputs), num_outputs(_num_outputs), inputs(_inputs),
      outputs(_outputs), params(_params) {}

#undef inp_type
#undef out_type
#undef param_type
};
#pragma endregion
}  // namespace manifold

template<>
struct std::formatter<manifold::ExpressionReflection> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  template<typename FormatContext>
  constexpr auto format(const manifold::ExpressionReflection &exp, FormatContext &ctx) const {
    std::format_to(ctx.out(),
      "{}:\nid: {}\nnum_inputs: {}\nnum_outputs: {}\n",
      manifold::optypeToString(exp.type),
      exp.id,
      exp.num_inputs,
      exp.num_outputs);
    std::format_to(ctx.out(), "Inputs: ");
    for (size_t i = 0; i < exp.num_inputs; ++i) { std::format_to(ctx.out(), "{} ", exp.inputs.at(i)); }
    std::format_to(ctx.out(), "\nOutputs: ");
    for (size_t i = 0; i < exp.num_outputs; ++i) { std::format_to(ctx.out(), "{} ", exp.outputs.at(i)); }
    return std::format_to(ctx.out(), "\n");
  }
};
