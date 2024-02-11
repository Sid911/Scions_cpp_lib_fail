#pragma once

#include "../common.hpp"
#include "../concepts.hpp"
#include "../expression.hpp"
#include "../op_type.hpp"

namespace manifold::op {
template<typename T, std::size_t N>
constexpr ExpressionReflection array_elm_op(uint32_t id, const OpType type, const T &out, const std::array<T, N> &inp)
  requires _internal::IsTensor<T>
{
  static_assert(N <= MANIFOLD_MAX_EXP_REF_INPUT,
    "Manifold: Input count more than MANIFOLD_MAX_EXP_INPUT, please define it as per your needs");

  using inp_type   = std::array<uint32_t, MANIFOLD_MAX_EXP_REF_INPUT>;
  using out_type   = std::array<uint32_t, MANIFOLD_MAX_EXP_REF_OUTPUT>;
  using param_type = std::array<std::byte, MANIFOLD_PARAM_BYTES_MAX>;

  auto inputs  = inp_type();
  auto outputs = out_type();
  auto params  = param_type();

  outputs.at(0) = out.id;
  for (size_t i = 0; i < inp.size(); ++i) { inputs.at(i) = inp.at(i).id; }
  return { id, type, N, inputs, 1, outputs, params };
}

template<typename T, std::size_t N>
constexpr ExpressionReflection array_add(uint32_t id, const T &out, const std::array<T, N> &inp)
  requires _internal::IsTensor<T>
{
  return array_elm_op(id, ARRAY_ELM_ADD, out, inp);
}

// Memory init
template<typename T>
struct fillParams {
  T value;
};

template<typename PARAM>
constexpr std::array<std::byte, MANIFOLD_PARAM_BYTES_MAX> copyStructToByteArray(const PARAM &obj)
  requires std::is_trivially_copyable_v<PARAM>
{
  std::array result = std::array<std::byte, MANIFOLD_PARAM_BYTES_MAX>();
  static_assert(
    MANIFOLD_PARAM_BYTES_MAX >= sizeof(PARAM), "Error Copying Params, please increase the 'MANIFOLD_PARAM_BYTES_MAX'");

  auto byteArray = std::bit_cast<std::array<std::byte, sizeof(PARAM)>>(obj);
  std::copy(byteArray.begin(), byteArray.end(), result.begin());
  return result;
}

template<typename OUT, typename IN, size_t N>
constexpr ExpressionReflection array_fill(uint32_t id, const std::array<OUT, N> &out, IN value)
  requires(_internal::IsTensor<OUT> && IsCompatibleDType<OUT::data_type, IN>)
{
  static_assert(N <= MANIFOLD_MAX_EXP_REF_OUTPUT,
    "Manifold: Output count more than MANIFOLD_MAX_EXP_OUTPUT, please define it as per your needs");

  auto inputs       = std::array<uint32_t, MANIFOLD_MAX_EXP_REF_INPUT>();
  auto outputs      = std::array<uint32_t, MANIFOLD_MAX_EXP_REF_OUTPUT>();
  auto param_struct = fillParams<IN>{ value };
  auto params       = copyStructToByteArray(param_struct);

  for (size_t i = 0; i < out.size(); ++i) { outputs.at(i) = out.at(i).id; }

  return { id, FILL_ELM, 0, inputs, N, outputs, params };
}
}  // namespace manifold::op
#pragma endregion
