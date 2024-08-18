//
// Created by sid on 29/11/23.
//
#pragma once
#include "manifold/expression.hpp"
#include <ranges>
#include <type_traits>

#pragma region Manifold concepts
namespace manifold::_internal {
template<typename T>
concept has_id = requires(T typ) {
  { typ.id } -> std::same_as<const unsigned &>;
};

template<typename T>
concept IsTBase = requires {
  T::data_type;
  T::size;
  std::declval<T>().shape;
};
template<typename T>
concept IsShape = requires {
  { T::rank } -> std::same_as<std::size_t>;  // Requires rank member variable
  { T::shape.size() } -> std::same_as<std::size_t>;  // Requires size() member function
};

template<typename T>
concept IsTensor = requires {
  T::data_type;  // Requires a data_type member type
  T::storage_type;  // Requires a storage_type member type
  T::shape;  // Requires a shape member
  T::id;
};

template<typename T>
concept IsExprReflection = requires { std::is_same_v<T, ExpressionReflection>; };

template<typename T>
concept IsScalar = requires {
  { T::shape.rank == 1 };
};

template<typename T>
struct is_std_array : std::false_type {};

template<typename T, std::size_t N>
struct is_std_array<std::array<T, N>> : std::true_type {};

template<typename T>
inline constexpr bool is_std_array_v = is_std_array<T>::value;

template<typename T>
concept IsStdArray = requires { is_std_array_v<T>; };
}  // namespace manifold::_internal
#pragma endregion
