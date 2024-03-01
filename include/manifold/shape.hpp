#pragma once
#include "manifold/common.hpp"
namespace manifold {

struct ShapeReflection {
  std::size_t rank;
  std::array<std::uint32_t, MANIFOLD_MAX_RANK> shape;

  constexpr ShapeReflection() : rank(UINT64_MAX), shape() {}

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

template<>
struct std::formatter<manifold::ShapeReflection> : std::formatter<std::string> {
  template<typename FormatContext>
  constexpr auto format(const manifold::ShapeReflection &ref, FormatContext &ctx) const {
    std::format_to(ctx.out(), "Rank : {}\nShape : (", ref.rank);
    for (size_t i{}; i < ref.rank - 1; i++) { std::format_to(ctx.out(), " {},", ref.shape.at(i)); }
    return format_to(ctx.out(), " {} )", ref.shape.at(ref.rank - 1));
  }
};
