#pragma once
#include "manifold/common.hpp"
namespace manifold {

struct ShapeReflection {
  std::size_t rank;
  std::array<std::uint32_t, MANIFOLD_MAX_RANK> shape;

  //  constexpr ShapeReflection() : rank(UINT64_MAX), shape() {}

  [[nodiscard]] constexpr ShapeReflection(const std::uint32_t rank_,
    const std::array<std::uint32_t, MANIFOLD_MAX_RANK> &shape_)
    : rank(rank_), shape(shape_) {}
};

template<std::uint32_t... ShapeList>
struct Shape {
  static constexpr std::size_t rank                      = sizeof...(ShapeList);
  static constexpr std::array<std::uint32_t, rank> shape = { ShapeList... };
  constexpr auto reflect() const {
    std::array<uint32_t, MANIFOLD_MAX_RANK> shape_arr{};
    static_assert(rank <= MANIFOLD_MAX_RANK, "Could not reflect Large rank, consider increasing MANIFOLD_MAX_RANK");
    std::copy(shape.begin(), shape.end(), shape_arr.begin());
    return ShapeReflection(rank, shape_arr);
  }
};

}  // namespace manifold