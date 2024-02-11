#pragma once

#include "expression.hpp"
#include "static_graph.hpp"

template<>
struct std::formatter<manifold::ExpressionReflection> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  template<typename FormatContext>
  constexpr auto format(const manifold::ExpressionReflection &exp, FormatContext &ctx) const {
    std::format_to(ctx.out(),
      "{}:\nid: {}\nnum_inputs: {}\nnum_outputs: {}\n",
      manifold::optypeToString(exp.type),
      exp.id,
      exp.num_inputs,
      exp.num_outputs);
    std::format_to(ctx.out(), "Inputs: ");
    for (size_t i = 0; i < exp.num_inputs; ++i) { std::format_to(ctx.out(), "{} ", exp.inputs.at(i)); }
    std::format_to(ctx.out(), "\nOutputs: ");
    for (size_t i = 0; i < exp.num_outputs; ++i) { std::format_to(ctx.out(), "{} ", exp.outputs.at(i)); }
    return std::format_to(ctx.out(), "\n");
  }
};
namespace manifold {
constexpr std::string_view dtypeToString(const DType &type) {
  switch (type) {
  case DType::UINT8: return { "UINT8" };
  case DType::UINT16: return { "UINT16" };
  case DType::UINT32: return { "UINT32" };
  case DType::UINT64: return { "UINT64" };
  case DType::INT8: return { "INT8" };
  case DType::INT16: return { "INT16" };
  case DType::INT32: return { "INT32" };
  case DType::INT64: return { "INT64" };
  case DType::F32: return { "F32" };
  case DType::F64: return { "F64" };
  }
  return {};
}

constexpr std::string_view storeToString(const Store &store) {
  switch (store) {
  case Store::HOST: return {"HOST"};
  case Store::DEVICE: return {"DEVICE"};
  }
  return {};
}

constexpr std::string_view layoutToString(const layout &layout) {
  switch (layout) {
  case layout::ROW_MAJOR: return {"ROW MAJOR"};
  case layout::COL_MAJOR: return {"COL MAJOR"};
  }
  return {};
}
}  // namespace manifold

template<>
struct std::formatter<manifold::ShapeReflection> : std::formatter<std::string> {
  template<typename FormatContext>
  constexpr auto format(const manifold::ShapeReflection &ref, FormatContext &ctx) const {
    std::format_to(ctx.out(), "Rank : {}\nShape : (", ref.rank);
    for (size_t i{}; i < ref.rank - 1; i++) { std::format_to(ctx.out(), " {},", ref.shape.at(i)); }
    return format_to(ctx.out(), " {} )", ref.shape.at(ref.rank - 1));
  }
};

template<>
struct std::formatter<manifold::TensorReflection> : std::formatter<std::string> {
  template<typename FormatContext>
  constexpr auto format(const manifold::TensorReflection &ten, FormatContext &ctx) const {
    std::format_to(ctx.out(),
      "id : {}\ntype : {}\nsize : {}\nstorage : {}\nlayout : {}",
      ten.id,
      dtypeToString(ten.data_type),
      ten.size,
      storeToString(ten.storage_type),
      layoutToString(ten.storage_layout));
    return std::format_to(ctx.out(), "\n{}", ten.shape);
  }
};

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
