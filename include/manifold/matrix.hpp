//
// Created by sid on 29/11/23.
//
#pragma once

#include "array.hpp"
#include "common.hpp"
#include "constants.hpp"
#include "shape.hpp"

namespace manifold {
template<DType Type,
  std::size_t Rows,
  std::size_t Cols,
  layout Layout = layout::ROW_MAJOR,
  Store Storage = Store::HOST>
struct Matrix {
  static constexpr DType data_type = Type;
  static constexpr std::size_t size = Rows * Cols;
  static constexpr Shape<Rows, Cols> shape{};
  static constexpr layout storage_layout = Layout;
  static constexpr Store storage_type    = Storage;
  uint32_t id;


  constexpr Matrix() : id(UINT32_MAX) {}

  [[nodiscard]] constexpr explicit Matrix(const uint32_t id_) : id(id_) {}

  [[nodiscard]] constexpr auto to_array() const { return Array<Type, size, Storage>(id); }
};
}  // namespace manifold