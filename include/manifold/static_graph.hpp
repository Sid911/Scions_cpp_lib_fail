#pragma once

#include "manifold/dag.hpp"

namespace manifold {

template<size_t TSize, size_t ExpSize>
struct SymbolContainer {
  std::array<TensorReflection, TSize> tensors;
  std::array<ExpressionReflection, ExpSize> exprs;

  [[nodiscard]] constexpr StaticDAG<TSize, ExpSize> to_dag() const { return StaticDAG<TSize, ExpSize>(tensors, exprs); }
};
}  // namespace manifold

template<size_t A, size_t B>
struct std::formatter<manifold::SymbolContainer<A, B>> : std::formatter<std::string> {
  template<typename FormatContext>
  constexpr auto format(const manifold::SymbolContainer<A, B> &sym, FormatContext &ctx) const {
    for (size_t i = 0; i < sym.tensors.size(); ++i) {
      std::format_to(ctx.out(), "\nTensor Pos : {}\n{}\n", i, sym.tensors.at(i));
    }
    for (size_t i = 0; i < sym.exprs.size(); ++i) {
      std::format_to(ctx.out(), "\nExpr Pos : {}\n{}\n", i, sym.exprs.at(i));
    }
    return std::format_to(ctx.out(), "\n\n");
  }
};
// Todo : might need this when implementing mem stores
// struct GraphMetadata {
//   size_t max_in;
//   size_t max_out;
//   // size_t max_params;
//   size_t graph_op_size;
//   size_t graph_data_size;
//   size_t total;

//   uint16_t u8_tensors;
//   uint16_t u16_tensors;
//   uint16_t u32_tensors;
//   uint16_t u64_tensors;
//   uint16_t i8_tensors;
//   uint16_t i16_tensors;
//   uint16_t i32_tensors;
//   uint16_t i64_tensors;
//   uint16_t f32_tensors;
//   uint16_t f64_tensors;

//   size_t u8_size;
//   size_t u16_size;
//   size_t u32_size;
//   size_t u64_size;

//   size_t i8_size;
//   size_t i16_size;
//   size_t i32_size;
//   size_t i64_size;

//   size_t f32_size;
//   size_t f64_size;
// };
