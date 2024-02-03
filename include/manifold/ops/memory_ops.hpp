//
// Created by sid on 1/1/24.
//

#pragma once
#include "../common.hpp"
#include "../concepts.hpp"
#include "../expression.hpp"
#include "../op_type.hpp"
#include "element_wise_ops.hpp"

namespace manifold::op {
template<typename T, std::size_t N>
constexpr ExpressionReflection fill(const T& out, const std::array<T, N>& inp) requires _internal::is_array_like<T> {
  using OutType = std::remove_const_t<T>;
  return array_elm_op(FILL_ELM,);
}
}