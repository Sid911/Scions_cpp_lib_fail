//
// Created by sid on 19/11/23.
//
#pragma once

#include "Scions/ep/common/common.h"

namespace scions::ep::cpu::_internal {
// template<class T, class Op1, class Op2, int... Dims>
// class TensorExpression {
//   Op1 op1; // left operand
//   Op2 op2; // right operand
// public:
//   TensorExpression(const Op1& a, const Op2& b) : op1(a), op2(b) {} // constructor
//   T operator()(int... indices) const { // evaluate the expression at given indices
//     return op1(indices...) + op2(indices...); // for example, this is the expression for addition
//   }
//   static constexpr int rank = sizeof...(Dims); // rank of the expression
//   static constexpr int size = (... * Dims); // total number of elements
// };
}

namespace scions::ep::cpu {}