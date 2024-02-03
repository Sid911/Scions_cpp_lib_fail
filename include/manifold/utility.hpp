//
// Created by sid on 29/11/23.
//
#pragma once
#include "common.hpp"
#include "expression.hpp"
#include "static_graph.hpp"

namespace manifold {
template<std::uint16_t DataLen, std::uint16_t ExpLen>
constexpr uint32_t extractRefs(const ExpressionReflection *exp_ref,
  std::array<TensorReflection, DataLen> &data,
  std::array<Expression, ExpLen> &exp_arr,
  std::uint16_t &data_len,
  std::uint16_t &exp_len) {

  if (exp_ref->isTensor()) {
    // Check if exp_ref_tensor.value().id is already in data:
    auto it = std::ranges::find_if(
      data, [id = exp_ref->tensor.value().id](const TensorReflection &ten) { return ten.id == id; });

    if (it == data.end()) {
      data.at(data_len++) = exp_ref->tensor.value();
      return data_len - 1;
    }
    // data_len++;
    return static_cast<uint32_t>(std::distance(data.begin(), it));
  }
  std::array<uint32_t, MANIFOLD_MAX_EXP_INPUT> input_arr{};
  std::array<uint32_t, MANIFOLD_MAX_EXP_OUTPUT> output_arr{};
  uint32_t const id = UINT16_MAX + 1 + exp_len;

  for (size_t i = 0; i < exp_ref->num_inputs; ++i) {
    const auto &current = exp_ref->inputs->at(i);
    input_arr.at(i)     = extractRefs<DataLen, ExpLen>(&current, data, exp_arr, data_len, exp_len);
  }

  for (size_t i = 0; i < exp_ref->num_outputs; ++i) {
    const auto &current = exp_ref->outputs->at(i);
    output_arr.at(i)    = extractRefs<DataLen, ExpLen>(&current, data, exp_arr, data_len, exp_len);
  }

  auto expression = Expression(exp_ref->type, id, exp_ref->num_inputs, input_arr, exp_ref->num_outputs, output_arr);
  auto it =
    std::ranges::find_if(exp_arr, [hash = expression.hash](const Expression &e) { return hash_value(e) == hash; });
  if (it == exp_arr.end()) {
    exp_arr.at(exp_len++) = expression;
    return exp_len;
  }
  return static_cast<uint32_t>(std::distance(exp_arr.begin(), it));
}

constexpr uint sizeOf(const DType type) { return DTYPE_SIZES[static_cast<uint8_t>(type)]; }


}  // namespace manifold


template<>
struct std::formatter<manifold::Expression> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  template<typename FormatContext>
  constexpr auto format(const manifold::Expression &exp, FormatContext &ctx) {
    std::format_to(ctx.out(),
      "{}:\nid: {}\nnum_inputs: {}\nnum_outputs: {}\nhash: {}\n",
      manifold::optypeToString(exp.type),
      exp.id,
      exp.num_inputs,
      exp.num_outputs,
      exp.hash);
    std::format_to(ctx.out(), "Inputs: ");
    for (size_t i = 0; i < exp.num_inputs; ++i) { std::format_to(ctx.out(), "{} ", exp.input_indices.at(i)); }
    std::format_to(ctx.out(), "\nOutputs: ");
    for (size_t i = 0; i < exp.num_outputs; ++i) { std::format_to(ctx.out(), "{} ", exp.output_indices.at(i)); }
    return std::format_to(ctx.out(), "\n");
  }
};

template<uint16_t MaxInput, uint16_t MaxOutput>
struct std::formatter<manifold::CompactExpression<MaxInput, MaxOutput >> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }
  template<typename FormatContext>
  constexpr auto format(const manifold::CompactExpression<MaxInput, MaxOutput> &exp,
    FormatContext &ctx) const {
    std::format_to(ctx.out(),
      "{}:\nid: {}\nInSize: {}\nOutSize: {}\nhash: {}\n",
      manifold::optypeToString(exp.type),
      exp.id,
      exp.inp_size,
      exp.out_size,
      exp.hash);
    std::format_to(ctx.out(), "Inputs: ");
    for (auto index : exp.input_indices) { std::format_to(ctx.out(), "{} ", index); }
    std::format_to(ctx.out(), "\nOutputs: ");
    for (auto index : exp.output_indices) { std::format_to(ctx.out(), "{} ", index); }
    return std::format_to(ctx.out(), "\n");
  }
};

namespace manifold {
constexpr std::string_view dtypeToString(const DType &type) {
  switch (type) {
  case DType::UINT8: return std::string_view("UINT8");
  case DType::UINT16: return std::string_view("UINT16");
  case DType::UINT32: return std::string_view("UINT32");
  case DType::UINT64: return std::string_view("UINT64");
  case DType::INT8: return std::string_view("INT8");
  case DType::INT16: return std::string_view("INT16");
  case DType::INT32: return std::string_view("INT32");
  case DType::INT64: return std::string_view("INT64");
  case DType::F32: return std::string_view("F32");
  case DType::F64: return std::string_view("F64");
  }
  return {};
}

constexpr std::string_view storeToString(const Store &store) {
  switch (store) {
  case Store::HOST: return std::string_view("HOST");
  case Store::DEVICE: return std::string_view("DEVICE");
  }
  return {};
}

constexpr std::string_view layoutToString(const layout &layout) {
  switch (layout) {
  case layout::ROW_MAJOR: return std::string_view("ROW MAJOR");
  case layout::COL_MAJOR: return std::string_view("COL MAJOR");
  }
  return {};
}
}  // namespace manifold

template<>
struct std::formatter<manifold::TensorReflection> : std::formatter<std::string> {
  template<typename FormatContext>
  constexpr auto format(const manifold::TensorReflection &ten, FormatContext &ctx) const {
    return std::format_to(ctx.out(),
      "id:{}\ntype:{}\nsize:{}\nstorage:{}\nlayout:{}",
      ten.id,
      dtypeToString(ten.data_type),
      ten.size,
      storeToString(ten.storage_type),
      layoutToString(ten.storage_layout));

    std::format_to(ctx.out(), "Rank / Shape: {} , (", ten.shape.rank);
    for (size_t i{}; i < ten.shape.rank; i++) { std::format_to(ctx.out(), " {},", ten.shape.shape.at(i)); }
    return std::format_to(ctx.out(), " )\n");
  }
};
