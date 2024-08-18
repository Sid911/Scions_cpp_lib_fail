//
// Created by sid on 27/12/23.
//
#pragma once
#include "manifold/tensor.hpp"
namespace scions::cpu {
template<typename T = void>
  requires std::is_arithmetic_v<T> || std::is_void_v<T>
struct RawData {
  T *data_ptr;
  manifold::TensorReflection *meta;
};
}  // namespace scions::cpu
