//
// Created by sid on 29/11/23.
//
#pragma once

#include "expression.hpp"
#include "tensor.hpp"


namespace manifold {

template<size_t TSize, size_t ExpSize>
struct SymbolContainer {
  std::array<TensorReflection, TSize> tensors;
  std::array<ExpressionReflection, ExpSize> exprs;
};


struct GraphMetadata {
  size_t max_in;
  size_t max_out;
  // size_t max_params;
  size_t graph_op_size;
  size_t graph_data_size;
  size_t total;

  uint16_t u8_tensors;
  uint16_t u16_tensors;
  uint16_t u32_tensors;
  uint16_t u64_tensors;
  uint16_t i8_tensors;
  uint16_t i16_tensors;
  uint16_t i32_tensors;
  uint16_t i64_tensors;
  uint16_t f32_tensors;
  uint16_t f64_tensors;

  size_t u8_size;
  size_t u16_size;
  size_t u32_size;
  size_t u64_size;

  size_t i8_size;
  size_t i16_size;
  size_t i32_size;
  size_t i64_size;

  size_t f32_size;
  size_t f64_size;
};
}  // namespace manifold
