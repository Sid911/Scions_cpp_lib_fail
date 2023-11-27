//
// Created by sid on 20/10/23.
//

#pragma once
#include "Scions/common/common.h"
#include "op.h"
#include "op_ids.h"

namespace scions::op::tensor {
class TensorAddOpDesc : public OpDesc {
public:
  constexpr TensorAddOpDesc(const std::array<const size_t, SC_OP_INPUT_MAX> &input,
    const std::array<const size_t, SC_OP_OUTPUT_MAX> &output)
    : OpDesc(TENSOR_ADD_OP_ID, input, output, "Tensor Add Op") {}

  constexpr TensorAddOpDesc(const size_t &input1, const size_t &input2, const size_t &output1)
    : OpDesc(TENSOR_ADD_OP_ID, { input1 + 1, input2 + 1 }, { output1 + 1 }, "Tensor Add Op") {}
};

class TensorSubtractOpDesc : public OpDesc {
public:
  constexpr TensorSubtractOpDesc(const std::array<const size_t, SC_OP_INPUT_MAX> &input,
    const std::array<const size_t, SC_OP_OUTPUT_MAX> &output)
    : OpDesc(TENSOR_SUBTRACT_OP_ID, input, output, "Tensor Subtract Op") {}

  constexpr TensorSubtractOpDesc(const size_t &input1, const size_t &input2, const size_t &output1)
    : OpDesc(TENSOR_SUBTRACT_OP_ID, { input1 + 1, input2 + 1 }, { output1 + 1 }, "Tensor Subtract Op") {}
};

// Class for describing a multiplication operation
class TensorMultiplyOpDesc : public OpDesc {
public:
  constexpr TensorMultiplyOpDesc(const std::array<const size_t, SC_OP_INPUT_MAX> &input,
    const std::array<const size_t, SC_OP_OUTPUT_MAX> &output)
    : OpDesc(TENSOR_MULTIPLY_OP_ID, input, output, "Tensor Multiply Op") {}

  constexpr TensorMultiplyOpDesc(const size_t &input1, const size_t &input2, const size_t &output1)
    : OpDesc(TENSOR_MULTIPLY_OP_ID, { input1 + 1, input2 + 1 }, { output1 + 1 }, "Tensor Multiply Op") {}
};

// Class for describing a division operation
class TensorDivideOpDesc : public OpDesc {
public:
  constexpr TensorDivideOpDesc(const std::array<const size_t, SC_OP_INPUT_MAX> &input,
    const std::array<const size_t, SC_OP_OUTPUT_MAX> &output)
    : OpDesc(TENSOR_DIVIDE_OP_ID, input, output, "Tensor Divide Op") {}

  constexpr TensorDivideOpDesc(const size_t &input1, const size_t &input2, const size_t &output1)
    : OpDesc(TENSOR_DIVIDE_OP_ID, { input1 + 1, input2 + 1 }, { output1 + 1 }, "Tensor Divide Op") {}
};

// Class for describing an exponentiation operation
class TensorExponentiateOpDesc : public OpDesc {
public:
  constexpr TensorExponentiateOpDesc(const std::array<const size_t, SC_OP_INPUT_MAX> &input,
    const std::array<const size_t, SC_OP_OUTPUT_MAX> &output)
    : OpDesc(TENSOR_EXPONENTIATION_OP_ID, input, output, "Tensor Exponentiate Op") {}

  constexpr TensorExponentiateOpDesc(const size_t &input1, const size_t &input2, const size_t &output1)
    : OpDesc(TENSOR_EXPONENTIATION_OP_ID, { input1 + 1, input2 + 1 }, { output1 + 1 }, "Tensor Exponentiate Op") {}
};

// Class for describing a logarithm operation
class TensorLogOpDesc : public OpDesc {
public:
  constexpr TensorLogOpDesc(const std::array<const size_t, SC_OP_INPUT_MAX> &input,
    const std::array<const size_t, SC_OP_OUTPUT_MAX> &output)
    : OpDesc(TENSOR_LOG_OP_ID, input, output, "Tensor Log Op") {}

  constexpr TensorLogOpDesc(const size_t &input1, const size_t &output1)
    : OpDesc(TENSOR_LOG_OP_ID, { input1 + 1 }, { output1 + 1 }, "Tensor Log Op") {}
};


// Class for describing a square root operation
class TensorSquareRootOpDesc : public OpDesc {
public:
  constexpr TensorSquareRootOpDesc(const std::array<const size_t, SC_OP_INPUT_MAX> &input,
    const std::array<const size_t, SC_OP_OUTPUT_MAX> &output)
    : OpDesc(TENSOR_SQUARE_ROOT_OP_ID, input, output, "Tensor Square Root Op") {}

  constexpr TensorSquareRootOpDesc(const size_t &input1, const size_t &output1)
    : OpDesc(TENSOR_SQUARE_ROOT_OP_ID, { input1 + 1 }, { output1 + 1 }, "Tensor Square Root Op") {}
};

// Class for describing a mean operation
class TensorMeanOpDesc : public OpDesc {
public:
  constexpr TensorMeanOpDesc(const std::array<const size_t, SC_OP_INPUT_MAX> &input,
    const std::array<const size_t, SC_OP_OUTPUT_MAX> &output)
    : OpDesc(TENSOR_MEAN_OP_ID, input, output, "Tensor Mean Op") {}

  constexpr TensorMeanOpDesc(const size_t &input1, const size_t &output1)
    : OpDesc(TENSOR_MEAN_OP_ID, { input1 + 1 }, { output1 + 1 }, "Tensor Mean Op") {}
};

}  // namespace scions::op::tensor
