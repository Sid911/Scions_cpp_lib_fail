#pragma once

#include "../concepts.hpp"
#include "../expression.hpp"
#include "../op_type.hpp"
#include "manifold/constants.hpp"
#include "manifold/macro.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace manifold::op {
// ------------------------------------------------ Helper utilities --------------------------------------------------

//
template<typename T>
struct OneValue {
  T value;
};
template<typename T, size_t N>
struct OneArray {
  std::array<T, N> arr;
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

// ------------------------------------------------ Base functions --------------------------------------------------
template<typename T, std::size_t N>
constexpr ExpressionReflection array_elm_op(uint32_t id, const OpType type, const T &out, const std::array<T, N> &inp)
  requires _internal::IsTensor<T>
{
  static_assert(N <= MANIFOLD_MAX_EXP_INPUT,
    "Manifold: Input count more than MANIFOLD_MAX_EXP_INPUT, please define it as per your needs");

  using inp_type   = std::array<uint32_t, MANIFOLD_MAX_EXP_INPUT>;
  using out_type   = std::array<uint32_t, MANIFOLD_MAX_EXP_OUTPUT>;
  using param_type = std::array<std::byte, MANIFOLD_PARAM_BYTES_MAX>;

  auto inputs  = inp_type();
  auto outputs = out_type();
  auto params  = param_type();

  outputs.at(0) = out.id;
  for (size_t i = 0; i < inp.size(); ++i) { inputs.at(i) = inp.at(i).id; }
  return { id, type, T::data_type, N, inputs, 1, outputs, params };
}

// Scalar version
template<typename T, typename Inp>
constexpr ExpressionReflection array_elm_op_scalar(uint32_t id, const OpType type, const T &out, const Inp &inp)
  requires _internal::IsTensor<T> && IsCompatibleDType<T::data_type, Inp>
{

  using inp_type = std::array<uint32_t, MANIFOLD_MAX_EXP_INPUT>;
  using out_type = std::array<uint32_t, MANIFOLD_MAX_EXP_OUTPUT>;

  auto inputs   = inp_type();
  auto outputs  = out_type();
  auto param    = copyStructToByteArray(OneValue<Inp>{ inp });
  outputs.at(0) = out.id;

  return { id, type, T::data_type, 0, inputs, 1, outputs, param };
}

// ------------------------------------------------ ELM ops --------------------------------------------------

// Many to one ops :
// These ops can take many multiple inputs and result in single output, for convinience also supports
// single input

template<OpType Op, OpType ScalarType>
constexpr auto create_elm_op() {
  return []<typename T, typename Inp>(uint32_t id, const T &out, const Inp &inp) constexpr
    requires _internal::IsTensor<T>
             && (_internal::is_std_array_v<Inp> || IsCompatibleDType<T::data_type, Inp> || _internal::IsTensor<Inp>)
  {
    constexpr auto is_primitive = IsCompatibleDType<T::data_type, Inp>;
    constexpr auto is_array     = _internal::is_std_array_v<Inp>;
    if constexpr (is_array) {
      if constexpr (_internal::IsTensor<typename Inp::value_type>) { return array_elm_op(id, Op, out, inp); }
    } else if constexpr (is_primitive) {
      return array_elm_op_scalar(id, ScalarType, out, inp);
    } else {
      return array_elm_op(id, Op, out, std::array{ inp, out });
    }
  };
}

//
constexpr auto elm_add = create_elm_op<OpType::ELM_ADD, OpType::SCL_ELM_ADD>();
constexpr auto elm_mul = create_elm_op<OpType::ELM_MUL, OpType::SCL_ELM_MUL>();
constexpr auto elm_div = create_elm_op<OpType::ELM_DIV, OpType::SCL_ELM_DIV>();
constexpr auto elm_sub = create_elm_op<OpType::ELM_SUB, OpType::SCL_ELM_SUB>();


// One to one :
// Supports single output per input
template<typename T, typename Inp>
constexpr ExpressionReflection exp(uint32_t id, const T &out, const Inp &inp)
  requires _internal::IsTensor<T>
{
  if constexpr (std::is_array_v<Inp>) {
    if constexpr (!_internal::IsTensor<typename Inp::value_type>) {
      return array_elm_op(id, OpType::EXPONENTIAL, out, inp);
    }
  }
  return array_elm_op(id, OpType::EXPONENTIAL, out, std::array{ inp });
}

// ------------------------------------------------ Memory --------------------------------------------------

// One to many op
template<typename OUT, typename IN, size_t N>
constexpr ExpressionReflection array_fill(uint32_t id, const std::array<OUT, N> &out, IN value)
  requires(_internal::IsTensor<OUT> ,IsCompatibleDType<OUT::data_type, IN>)
{
  static_assert(N <= MANIFOLD_MAX_EXP_OUTPUT,
    "Manifold: Output count more than MANIFOLD_MAX_EXP_OUTPUT, please define it as per your needs");

  auto inputs       = std::array<uint32_t, MANIFOLD_MAX_EXP_INPUT>();
  auto outputs      = std::array<uint32_t, MANIFOLD_MAX_EXP_OUTPUT>();
  auto param_struct = OneValue<IN>{ value };
  auto params       = copyStructToByteArray(param_struct);

  for (size_t i = 0; i < out.size(); ++i) { outputs.at(i) = out.at(i).id; }

  return { id, OpType::ELM_FILL, OUT::data_type, 0, inputs, N, outputs, params };
}


// Todo : Fix this for coping between locations, in and out should have different types
template<typename T>
constexpr ExpressionReflection copy(uint32_t id, const T &out, const T &in)
  requires _internal::IsTensor<T>
{
  auto inputs  = std::array<uint32_t, MANIFOLD_MAX_EXP_INPUT>();
  auto outputs = std::array<uint32_t, MANIFOLD_MAX_EXP_OUTPUT>();
  auto params  = std::array<std::byte, MANIFOLD_PARAM_BYTES_MAX>();

  outputs[0] = out.id;
  inputs[0]  = in.id;

  return { id, OpType::COPY, T::data_type, 1, inputs, 1, outputs, params };
}
#pragma endregion
}  // namespace manifold::op
