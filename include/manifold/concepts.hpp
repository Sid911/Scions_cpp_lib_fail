//
// Created by sid on 29/11/23.
//
#pragma once
#include "common.hpp"
#include "constants.hpp"

#pragma region Manifold concepts
namespace manifold::_internal {
template<typename T>
concept has_id = requires(T typ) {
  { typ.id } -> std::same_as<const unsigned &>;
};

template<typename T>
concept has_size = requires(T typ) {
  // remember : auto != decltype(auto)
  { typ.size } -> std::same_as<const unsigned long &>;
};

template<typename T>
concept is_array_like = requires(T typ) {
  { typ.storage_type } -> std::same_as<const Store &>;
  // { typ.to_tensor() } -> std::same_as<decltype(typ.to_tensor())>;
} && has_id<T> && has_size<T> && std::is_class_v<T>;
}  // namespace manifold::_internal
#pragma endregion
