//
// Created by sid on 13/11/23.
//
#pragma once
#include "Scions/common/common.h"

namespace scions::op::tensor {
static const int32_t TENSOR_ADD_OP_ID            = 1;
static const int32_t TENSOR_SUBTRACT_OP_ID       = 2;
static const int32_t TENSOR_MULTIPLY_OP_ID       = 3;  // New operation ID
static const int32_t TENSOR_DIVIDE_OP_ID         = 4;  // New operation ID
static const int32_t TENSOR_EXPONENTIATION_OP_ID = 5;  // New operation ID
static const int32_t TENSOR_LOG_OP_ID            = 6;  // New operation ID
static const int32_t TENSOR_SQUARE_ROOT_OP_ID    = 7;  // New operation ID
static const int32_t TENSOR_MEAN_OP_ID           = 8;  // New operation ID
}  // namespace scions::op::tensor

namespace scions::op::mem {
static const int32_t MEM_COPY_OP_ID        = -1;
static const int32_t MEM_FILL_RANDOM_OP_ID = -2;
static const int32_t MEM_FREE_OP_ID        = -3;
}  // namespace scions::op::mem
