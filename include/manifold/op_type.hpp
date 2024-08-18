#pragma once
#include "manifold/constants.hpp"
#include <cstdint>

namespace manifold {
enum class OpType : uint8_t {
  // Element wise add two arrays
  ELM_ADD,
  ELM_SUB,
  ELM_MUL,
  ELM_DIV,
  ELM_RANDOM,

  // memory based
  ELM_FILL,
  COPY,

  // MATH
  EXPONENTIAL,
  SIN,
  COS,
  ABS,
  // Array Scalar ops
  SCL_ELM_ADD,
  SCL_ELM_SUB,
  SCL_ELM_MUL,
  SCL_ELM_DIV,


  // AXPY is vector scalar product
  ARRAY_AXPY,
  ARRAY_SUM,
  ARRAY_MEAN,

  // Matrix Matrix operations
  MAT_ADD,
  MAT_SUB,
  MAT_TRAN,
  MAT_MUL,
  MAT_INV,

  // Matrix Array ops
  MAT_ARR_MUL,
  MAT_ARR_ADD,

  // Special Ops

  EXP_GROUP,

  // Derivative op type
  D_ZERO,
  D_IDENTITY,
  BRUH
};


constexpr inline uint16_t GetParamSize(OpType op, DType type) {
  switch (op) {
  case OpType::ELM_FILL: return DTYPE_SIZES[static_cast<uint8_t>(type)];
  case OpType::SCL_ELM_ADD: return DTYPE_SIZES[static_cast<uint8_t>(type)];
  case OpType::SCL_ELM_SUB: return DTYPE_SIZES[static_cast<uint8_t>(type)];
  case OpType::SCL_ELM_DIV: return DTYPE_SIZES[static_cast<uint8_t>(type)];
  case OpType::SCL_ELM_MUL: return DTYPE_SIZES[static_cast<uint8_t>(type)];
  default: return 0;
  }
}

constexpr inline std::string_view optypeToString(OpType e) {
  switch (e) {
  case OpType::ELM_ADD: return "ELM_ADD";
  case OpType::ELM_SUB: return "ELM_SUB";
  case OpType::ELM_MUL: return "ELM_MUL";
  case OpType::ELM_DIV: return "ELM_DIV";
  case OpType::ELM_RANDOM: return "RANDOM_ELM";
  case OpType::ELM_FILL: return "FILL_ELM";
  case OpType::COPY: return "COPY";
  case OpType::ARRAY_AXPY: return "ARRAY_AXPY";
  case OpType::ARRAY_SUM: return "ARRAY_SUM";
  case OpType::ARRAY_MEAN: return "ARRAY_MEAN";
  case OpType::MAT_ADD: return "MAT_ADD";
  case OpType::MAT_SUB: return "MAT_SUB";
  case OpType::MAT_TRAN: return "MAT_TRAN";
  case OpType::MAT_MUL: return "MAT_MUL";
  case OpType::MAT_INV: return "MAT_INV";
  case OpType::MAT_ARR_MUL: return "MAT_ARR_MUL";
  case OpType::MAT_ARR_ADD: return "MAT_ARR_ADD";
  default: return "UNKNOWN";
  }
}
}  // namespace manifold
