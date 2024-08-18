//
// Created by sid on 27/12/23.
//

#pragma once
#include "scions/common/common.hpp"

namespace scions::cpu {
template<typename T, size_t N, size_t IN_S>
void element_wise_add(T *out, std::array<T *, IN_S> &in)
  requires std::is_arithmetic_v<T>
{
  for (size_t i = 0; i < N; ++i) {
    T sum{ in[0][i] };
    for (size_t j = 1; j < IN_S; ++j) { sum += in[j][i]; }
    out[i] = sum;
  }
}

template<typename T, size_t N, size_t IN_S>
void element_wise_mul(T *out, std::array<T *, IN_S> &in)
  requires std::is_arithmetic_v<T>
{
  for (size_t i = 0; i < N; ++i) {
    T prod{ in[0][i] };
    for (size_t j = 1; j < IN_S; ++j) { prod *= in[j][i]; }
    out[i] = prod;
  }
}

}  // namespace scions::cpu