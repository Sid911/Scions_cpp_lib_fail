//
// Created by sid on 29/11/23.
//
#pragma once
#include "common.hpp"

namespace manifold {
enum OpType : uint8_t {
  // Element wise add two arrays
  ARRAY_ELM_ADD,
  ARRAY_ELM_SUB,
  ARRAY_ELM_MUL,
  ARRAY_ELM_DIV,
  RANDOM_ELM,
  FILL_ELM,
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
  MAT_ARR_ADD
};

inline std::string_view optypeToString(OpType e) {
  switch (e) {
  case ARRAY_ELM_ADD: return "ARRAY_ELM_ADD";
  case ARRAY_ELM_SUB: return "ARRAY_ELM_SUB";
  case ARRAY_ELM_MUL: return "ARRAY_ELM_MUL";
  case ARRAY_ELM_DIV: return "ARRAY_ELM_DIV";
  case RANDOM_ELM: return "RANDOM_ELM";
  case ARRAY_AXPY: return "ARRAY_AXPY";
  case ARRAY_SUM: return "ARRAY_SUM";
  case ARRAY_MEAN: return "ARRAY_MEAN";
  case MAT_ADD: return "MAT_ADD";
  case MAT_SUB: return "MAT_SUB";
  case MAT_TRAN: return "MAT_TRAN";
  case MAT_MUL: return "MAT_MUL";
  case MAT_INV: return "MAT_INV";
  case MAT_ARR_MUL: return "MAT_ARR_MUL";
  case MAT_ARR_ADD: return "MAT_ARR_ADD";
  default: return {};
  }
}

inline uint8_t optypeParams(OpType e) {
  switch (e) {
  case FILL_ELM: return 1;
  case RANDOM_ELM: return 1;
  default: return 0;
  }
}
}  // namespace manifold
