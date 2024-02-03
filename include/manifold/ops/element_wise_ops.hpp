//
// Created by sid on 29/11/23.
//
#pragma once

#include "../common.hpp"
#include "../concepts.hpp"
#include "../expression.hpp"
#include "../op_type.hpp"

namespace manifold::op {
#pragma region Element Wise OPs
/**
 * \brief Per element operation
 *
 * \tparam T Type of Structure to be added. Must satisfy _internal::is_array_like concept.
 * \tparam N Size of inp array
 * \param type
 * \param out output Structure to store the addition
 * \param inp Input array containing all the structures to element wise add
 * \return Expression object describing the operation
 */
template<typename T, std::size_t N>
constexpr ExpressionReflection array_elm_op(const OpType type, T &out, const std::array<T, N> &inp)
  requires _internal::is_array_like<T>
{
  static_assert(N <= MANIFOLD_MAX_EXP_REF_INPUT,
    "Manifold: Input count more than MANIFOLD_MAX_EXP_INPUT, please define it as per your needs");

  using arr_type = std::array<ExpressionReflection, MANIFOLD_MAX_EXP_REF_INPUT>;
  auto inputs    = std::make_unique<arr_type>();
  auto outputs   = std::make_unique<arr_type>();
  outputs->at(0) = ExpressionReflection(out);
  for (size_t i = 0; i < inp.size(); ++i) { inputs->at(i) = ExpressionReflection(inp.at(i)); }
  return { type, N, inputs, 1, outputs, {}};
}

template<typename T, std::size_t N>
constexpr ExpressionReflection array_elm_op(const OpType type, const T &out, const std::array<T, N> &inp)
  requires(_internal::is_array_like<const T>)
{

  static_assert(N <= MANIFOLD_MAX_EXP_REF_INPUT,
    "Manifold: Input count more than MANIFOLD_MAX_EXP_INPUT, please define it as per your needs");

  using arr_type = std::array<ExpressionReflection, MANIFOLD_MAX_EXP_REF_INPUT>;
  auto inputs    = std::make_unique<arr_type>();
  auto outputs   = std::make_unique<arr_type>();
  outputs->at(0) = ExpressionReflection(out);
  for (size_t i = 0; i < inp.size(); ++i) { inputs->at(i) = ExpressionReflection(inp.at(i)); }
  return { type, N, inputs, 1, outputs, {} };
}

template<typename T, std::size_t N>
constexpr ExpressionReflection array_add(const T &out, const std::array<T, N> &inp)
  requires(_internal::is_array_like<const T>)
{
  using OutType = std::remove_const_t<T>;
  return array_elm_op(ARRAY_ELM_ADD, out, inp);
}

template<typename T, std::size_t N>
constexpr ExpressionReflection array_mul(T &out, const std::array<T, N> &inp)
  requires(_internal::is_array_like<T>)
{
  return array_elm_op(ARRAY_ELM_MUL, out, inp);
}

template<typename T, std::size_t N>
constexpr ExpressionReflection array_mul(const T &out, const std::array<T, N> &inp)
  requires(_internal::is_array_like<const T>)
{
  return array_elm_op(ARRAY_ELM_MUL, out, inp);
}

template<typename T, std::size_t N>
ExpressionReflection array_sub(T &out, const std::array<T, N> &inp)
  requires(_internal::is_array_like<T>)
{
  return array_elm_op(ARRAY_ELM_SUB, out, inp);
}

template<typename T, std::size_t N>
ExpressionReflection array_sub(const T &out, const std::array<T, N> &inp)
  requires(_internal::is_array_like<const T>)
{
  return array_elm_op<T, N>(ARRAY_ELM_SUB, out, inp);
}
#pragma endregion
}  // namespace manifold::op