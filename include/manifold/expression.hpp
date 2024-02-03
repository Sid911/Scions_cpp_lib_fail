//
// Created by sid on 29/11/23.
//
#pragma once
#include "op_type.hpp"
#include "tensor.hpp"

namespace manifold {

struct Expression {
  OpType type;
  uint64_t id;
  uint32_t num_inputs;
  uint32_t num_outputs;
  std::array<std::uint32_t, MANIFOLD_MAX_EXP_INPUT> input_indices;
  std::array<std::uint32_t, MANIFOLD_MAX_EXP_OUTPUT> output_indices;
  std::size_t hash;

  constexpr Expression()
    : type(), id(0), num_inputs(0), num_outputs(0), input_indices(), output_indices(), hash(0) {}

  [[nodiscard]] constexpr Expression(const OpType type_,
    const uint64_t id_,
    const uint32_t num_inputs_,
    std::array<std::uint32_t, MANIFOLD_MAX_EXP_INPUT> &input_indices_,
    const uint32_t num_outputs_,
    std::array<std::uint32_t, MANIFOLD_MAX_EXP_OUTPUT> &output_indices_)
    : type(type_), id(id_), num_inputs(num_inputs_), num_outputs(num_outputs_), input_indices(input_indices_),
      output_indices(output_indices_), hash(0) {
    // do no initialize in constructor
    hash = hash_value(*this);
  }
  // NOLINTBEGIN
  constexpr friend std::size_t hash_value(const Expression &obj) {
    if (obj.hash != 0) { return obj.hash; }
    std::size_t seed = 0x071EF957;
    seed ^= (seed << 6) + (seed >> 2) + 0x4D863D4C + static_cast<std::size_t>(obj.type);
    // seed ^= (seed << 6) + (seed >> 2) + 0x75319139 + static_cast<std::size_t>(obj.id);
    seed ^= (seed << 6) + (seed >> 2) + 0x6BE220B3 + static_cast<std::size_t>(obj.num_inputs);
    seed ^= (seed << 6) + (seed >> 2) + 0x3831C880 + static_cast<std::size_t>(obj.num_outputs);
    for (size_t i = 0; i < obj.num_inputs; i++) {
      seed ^= (seed << 6) + (seed >> 2) + 0x27889449 + static_cast<std::size_t>(obj.input_indices.at(i));
    }
    for (size_t i = 0; i < obj.num_outputs; i++) {
      seed ^= (seed << 6) + (seed >> 2) + 0x021A6FAF + static_cast<std::size_t>(obj.output_indices.at(i));
    }
    return seed;
  }
  // NOLINTEND
};

template<uint16_t CMaxInput, uint16_t CMaxOutput>
struct CompactExpression {
  OpType type;
  uint64_t id;
  uint16_t inp_size;
  uint16_t out_size;
  std::array<std::uint32_t, CMaxInput> input_indices;
  std::array<std::uint32_t, CMaxOutput> output_indices;
  // std::array<std::uint8_t[], CMaxParams> params;
  std::size_t hash;

  constexpr CompactExpression() = default;

  template<size_t MaxInput_ = MANIFOLD_MAX_EXP_INPUT, size_t MaxOutput_ = MANIFOLD_MAX_EXP_OUTPUT>
  explicit constexpr CompactExpression(const Expression &exp)
    : type(exp.type), id(exp.id), input_indices(), output_indices(), hash(exp.hash) {
    const auto exp_inp = static_cast<uint16_t>(exp.num_inputs);
    const auto exp_out = static_cast<uint16_t>(exp.num_outputs);
    inp_size           = exp_inp < CMaxInput ? exp_inp : CMaxInput;
    out_size           = exp_out < CMaxOutput ? exp_out : CMaxOutput;
    for (size_t i = 0; i < inp_size; i++) { input_indices.at(i) = exp.input_indices.at(i); }
    for (size_t i = 0; i < out_size; i++) { output_indices.at(i) = exp.output_indices.at(i); }
    // for (size_t i = 0; i < CMaxParams; ++i) { params.at(i) = exp.params.at(i); }
  }
};


#pragma region reflection

struct ExpressionReflection {
  // Expression part

  // OpType of the expression
  OpType type;
  uint32_t num_inputs;
  uint32_t num_outputs;
  std::unique_ptr<std::array<ExpressionReflection, MANIFOLD_MAX_EXP_REF_INPUT>> inputs;
  std::unique_ptr<std::array<ExpressionReflection, MANIFOLD_MAX_EXP_REF_INPUT>> outputs;
  // const std::unique_ptr<const uint8_t[]> params;

  // TensorReflection part
  std::optional<TensorReflection> tensor;
  constexpr ExpressionReflection() : type(), num_inputs(0), num_outputs(0) {}

  // delete the copy constructor
  ExpressionReflection(const ExpressionReflection &) = delete;

  // if you also want to delete the copy assignment operator, do this as well:
  ExpressionReflection &operator=(const ExpressionReflection &) = delete;

  ExpressionReflection(ExpressionReflection &&)            = default;  // Move constructor
  ExpressionReflection &operator=(ExpressionReflection &&) = default;  // Move assignment operator

  ~ExpressionReflection() = default;

  [[nodiscard]] constexpr ExpressionReflection(const OpType type_,
    const uint32_t num_inputs_,
    std::unique_ptr<std::array<ExpressionReflection, MANIFOLD_MAX_EXP_REF_INPUT>> &inputs_,
    const uint32_t num_outputs_,
    std::unique_ptr<std::array<ExpressionReflection, MANIFOLD_MAX_EXP_REF_INPUT>> &outputs_,
    const std::optional<TensorReflection> &tensor_)
    : type(type_), num_inputs(num_inputs_), num_outputs(num_outputs_), inputs(std::move(inputs_)),
      outputs(std::move(outputs_)), tensor(tensor_) {}

  // ReSharper disable once CppNonExplicitConvertingConstructor
  template<DType Type, std::size_t Size, Store Storage = Store::HOST>
  // ReSharper disable once CppNonExplicitConvertingConstructor
  constexpr ExpressionReflection(Array<Type, Size, Storage> &array) : ExpressionReflection(TensorReflection(array)) {}

  // ReSharper disable once CppNonExplicitConvertingConstructor
  template<DType Type,
    std::size_t Rows,
    std::size_t Cols,
    layout Layout = layout::ROW_MAJOR,
    Store Storage = Store::HOST>
  // ReSharper disable once CppNonExplicitConvertingConstructor
  constexpr ExpressionReflection(const Matrix<Type, Rows, Cols, Layout, Storage> &mat)
    : ExpressionReflection(TensorReflection(mat)) {}

  [[nodiscard]] constexpr explicit ExpressionReflection(const TensorReflection &tensor_)
    : type(), num_inputs(0), num_outputs(0), tensor(tensor_) {}

  // NOLINTBEGIN
  [[nodiscard]] constexpr bool isTensor() const { return tensor.has_value(); }
  constexpr friend std::size_t hash_value(const ExpressionReflection &obj) {
    std::size_t seed = 0x713065A4;
    if (obj.isTensor()) {
      seed ^= (seed << 6) + (seed >> 2) + 0x26072FDF + hash_value(obj.tensor.value());
    } else {
      seed ^= (seed << 6) + (seed >> 2) + 0x19D2EDF0 + static_cast<std::size_t>(obj.type);
      seed ^= (seed << 6) + (seed >> 2) + 0x14604031 + static_cast<std::size_t>(obj.num_inputs);
      seed ^= (seed << 6) + (seed >> 2) + 0x17E21DA7 + static_cast<std::size_t>(obj.num_outputs);
      for (size_t i = 0; i < obj.num_inputs; ++i) {
        seed ^= (seed << 6) + (seed >> 2) + 0x591C6A0B + hash_value(obj.inputs->at(i));
      }
      for (size_t i = 0; i < obj.num_inputs; ++i) {
        seed ^= (seed << 6) + (seed >> 2) + 0x1F5BB1BC + hash_value(obj.outputs->at(i));
      }
    }
    return seed;
  }
  // NOLINTEND
};
#pragma endregion
}  // namespace manifold