//
// Created by sid on 29/11/23.
//

#pragma once
#include "common.hpp"
#include <cstdint>

namespace manifold {
template<std::uint32_t... ShapeList>
struct Shape {
  static constexpr std::size_t rank                      = sizeof...(ShapeList);
  static constexpr std::array<std::uint32_t, rank> shape = { ShapeList... };
};

struct ShapeReflection {
  std::size_t rank;
  std::array<std::uint32_t, MANIFOLD_MAX_RANK> shape;

  constexpr ShapeReflection() : rank(0), shape() {}

  [[nodiscard]] constexpr ShapeReflection(const std::uint32_t rank_,
    const std::array<std::uint32_t, MANIFOLD_MAX_RANK> &shape_)
    : rank(rank_), shape(shape_) {}

  // NOLINTBEGIN
  [[nodiscard]] constexpr ShapeReflection(const std::uint32_t size) : rank(1), shape({ size }) {}

  constexpr friend std::size_t hash_value(const ShapeReflection &obj) {
    std::size_t seed = 0x48335BC4;
    seed ^= (seed << 6) + (seed >> 2) + 0x29A54A39 + static_cast<std::size_t>(obj.rank);
    for (size_t i = 0; i < obj.rank; i++) {
      seed ^= static_cast<uint32_t>(obj.shape[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
  // NOLINTEND
  [[nodiscard]] constexpr ShapeReflection(const std::uint32_t M, const std::uint32_t N) : rank(2), shape({ M, N }) {}
};

}  // namespace manifold