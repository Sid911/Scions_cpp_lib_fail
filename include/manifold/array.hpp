//
// Created by sid on 29/11/23.
//

#pragma once

#include "common.hpp"
#include "constants.hpp"

namespace manifold {
template<DType Type, std::size_t Size, Store Storage = Store::HOST>
struct Array {
  static constexpr DType data_type   = Type;
  static constexpr std::size_t size   = Size;
  static constexpr Store storage_type = Storage;
  uint32_t id;

  constexpr Array() : id(UINT32_MAX) {}

  [[nodiscard]] constexpr explicit Array(const uint32_t id_) : id(id_) {}
};

}  // namespace manifold