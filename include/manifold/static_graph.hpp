//
// Created by sid on 29/11/23.
//
#pragma once
#include "common.hpp"
#include "expression.hpp"
#include "tensor.hpp"
#include "utility.hpp"

namespace manifold {

template<std::uint16_t DataLen = MANIFOLD_STATIC_GRAPH_MAX_MEM,
  std::uint16_t ExpLen         = MANIFOLD_STATIC_GRAPH_MAX_OP,
  size_t MaxInput              = MANIFOLD_MAX_EXP_INPUT,
  size_t MaxOutput             = MANIFOLD_MAX_EXP_OUTPUT>
struct StaticGraph {
  std::uint16_t data_len;
  std::uint16_t exp_len;
  std::array<TensorReflection, DataLen> data;
  std::array<Expression, ExpLen> expressions;

  [[nodiscard]] constexpr StaticGraph(const std::uint16_t data_len_,
    std::array<TensorReflection, DataLen> data_,
    const std::uint16_t exp_len_,
    std::array<Expression, ExpLen> expressions_)
    : data_len(data_len_), exp_len(exp_len_), data(std::move(data_)), expressions(std::move(expressions_)) {}

  template<size_t N>
  constexpr explicit StaticGraph(const std::array<ExpressionReflection *, N> &reflections)
    : data_len(0), exp_len(0), data(), expressions{} {
    for (const ExpressionReflection *exp_ref : reflections) {
      extractRefs<DataLen, ExpLen>(exp_ref, data, expressions, data_len, exp_len);
    }
  }
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

template<std::uint16_t CDataLen, std::uint16_t CExpLen, size_t CMaxInput, size_t CMaxOutput>
struct CompactStaticGraph {
  std::array<TensorReflection, CDataLen> data;
  std::array<CompactExpression<CMaxInput, CMaxOutput>, CExpLen> expressions;

  template<std::uint16_t DataLen_, std::uint16_t ExpLen_, size_t MaxInput_, size_t MaxOutput_>
  constexpr explicit CompactStaticGraph(const StaticGraph<DataLen_, ExpLen_, MaxInput_, MaxOutput_> &other)
    : data(), expressions() {
    for (size_t i = 0; i < DataLen_ && i < other.data_len; i++) { data.at(i) = other.data.at(i); }
    for (size_t i = 0; i < CExpLen && i < other.exp_len; i++) {
      expressions.at(i) = CompactExpression<CMaxInput, CMaxOutput>(other.expressions.at(i));
    }
  }
};
consteval auto getMetadata(const StaticGraph<> &inp_graph) {
  size_t max_exp_in  = 0;
  size_t max_exp_out = 0;
  // size_t max_params = 0;
  size_t total_size  = 0;
  std::array<size_t, NUM_DTYPE> sizes{};
  std::array<uint8_t, NUM_DTYPE> counts{};


  for (size_t i = 0; i < inp_graph.exp_len; ++i) {
    auto &exp   = inp_graph.expressions.at(i);
    max_exp_in  = exp.num_inputs > max_exp_in ? exp.num_inputs : max_exp_in;
    max_exp_out = exp.num_outputs > max_exp_out ? exp.num_outputs : max_exp_out;
  }

  for (size_t i = 0; i < inp_graph.data_len; ++i) {
    const TensorReflection &ref = inp_graph.data.at(i);
    total_size += ref.size * sizeOf(ref.data_type);
    const size_t type_index = static_cast<uint8_t>(ref.data_type);
    sizes[type_index] += ref.size;
    counts[type_index] += 1;
  }

  return GraphMetadata{
    max_exp_in,
    max_exp_out,
    inp_graph.exp_len,
    inp_graph.data_len,
    total_size,
    counts[0],
    counts[1],
    counts[2],
    counts[3],
    counts[4],
    counts[5],
    counts[6],
    counts[7],
    counts[8],
    counts[9],
    sizes[0],
    sizes[1],
    sizes[2],
    sizes[3],
    sizes[4],
    sizes[5],
    sizes[6],
    sizes[7],
    sizes[8],
    sizes[9],
  };
}

template<GraphMetadata C>
constexpr auto compact(const StaticGraph<> &inp_graph) {
  return CompactStaticGraph<C.graph_data_size, C.graph_op_size, C.max_in, C.max_out>(inp_graph);
}

}  // namespace manifold