#pragma once

#include "manifold/constants.hpp"

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
  case Store::HOST: return { "HOST" };
  case Store::DEVICE: return { "DEVICE" };
  }
  return {};
}

constexpr std::string_view layoutToString(const layout &layout) {
  switch (layout) {
  case layout::ROW_MAJOR: return { "ROW MAJOR" };
  case layout::COL_MAJOR: return { "COL MAJOR" };
  }
  return {};
}
}  // namespace manifold

