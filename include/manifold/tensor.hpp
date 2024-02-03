//
// Created by sid on 29/11/23.
//
#pragma once
#include "common.hpp"
#include "constants.hpp"
#include "matrix.hpp"
#include "shape.hpp"

namespace manifold {

template<DType Type, std::uint32_t... Shapes>
struct TBase {
  static constexpr DType data_type = Type;
  static constexpr std::size_t size = (Shapes * ...);
  static constexpr Shape<Shapes...> shape{};
};

template<typename TensorBase, Store Storage = Store::HOST, layout Layout = layout::ROW_MAJOR>
struct Tensor : TensorBase {
  static constexpr layout storage_layout = Layout;
  static constexpr Store storage_type    = Storage;
  uint32_t id;

  constexpr Tensor() : id(UINT32_MAX) {}

  [[nodiscard]] constexpr explicit Tensor(const uint32_t id_) : id(id_) {}

  [[nodiscard]] constexpr auto to_array() const { return Array<TensorBase::data_type, TensorBase::size, Storage>(id); }

  template<std::size_t M, std::size_t N>
  [[nodiscard]] constexpr auto to_matrix() const {
    static_assert(TensorBase::size == M * N, "total Matrix size should be equal total tensor size");
    return Matrix<TensorBase::data_type, M, N, Layout, Storage>(id);
  }
};

#pragma region reflections
// Reflections
struct TensorReflection {
  DType data_type;
  std::size_t size;
  uint32_t id;
  ShapeReflection shape;
  layout storage_layout;
  Store storage_type;

  constexpr TensorReflection() : data_type(), size(0), id(0), storage_layout(), storage_type() {}

  [[nodiscard]] constexpr TensorReflection(const DType data_type_,
    const std::size_t size_,
    const uint32_t id_,
    const ShapeReflection &shape_reflection,
    const layout storage_layout_,
    const Store storage_type_)
    : data_type(data_type_), size(size_), id(id_), shape(shape_reflection), storage_layout(storage_layout_),
      storage_type(storage_type_) {}


  template<DType Type, std::size_t Size, Store Storage = Store::HOST>
  constexpr TensorReflection(const Array<Type, Size, Storage> &arr)
    : data_type(Type), size(Size), id(arr.id), shape(Size), storage_layout(layout::ROW_MAJOR), storage_type(Storage) {}

  template<DType Type,
    std::size_t Rows,
    std::size_t Cols,
    layout Layout = layout::ROW_MAJOR,
    Store Storage = Store::HOST>
  constexpr TensorReflection(const Matrix<Type, Rows, Cols, Layout, Storage> &mat)
    : data_type(Type), size(Rows * Cols), id(mat.id), shape(Rows, Cols), storage_layout(layout::ROW_MAJOR),
      storage_type(Storage) {}

  // NOLINTBEGIN
  constexpr friend std::size_t hash_value(const TensorReflection &obj) {
    std::size_t seed = 0x66D26DCF;
    seed ^= (seed << 6) + (seed >> 2) + 0x1B5EB086 + static_cast<std::size_t>(obj.data_type);
    seed ^= (seed << 6) + (seed >> 2) + 0x0F42AAAE + static_cast<std::size_t>(obj.size);
    seed ^= (seed << 6) + (seed >> 2) + 0x426382BB + static_cast<std::size_t>(obj.id);
    seed ^= (seed << 6) + (seed >> 2) + 0x3BF8AAF1 + hash_value(obj.shape);
    seed ^= (seed << 6) + (seed >> 2) + 0x0FCF80F0 + static_cast<std::size_t>(obj.storage_layout);
    seed ^= (seed << 6) + (seed >> 2) + 0x5167620F + static_cast<std::size_t>(obj.storage_type);
    return seed;
  }
  // NOLINTEND
};
#pragma endregion


}  // namespace manifold