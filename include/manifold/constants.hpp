//
// Created by sid on 29/11/23.
//

#pragma once

namespace manifold {
enum class DType : std::uint8_t { UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64, F32, F64 };

constexpr std::array DTYPE_SIZES = { 1u, 2u, 4u, 8u, 1u, 2u, 4u, 8u, 4u, 8u };
constexpr uint8_t NUM_DTYPE      = 10;

static_assert(NUM_DTYPE == DTYPE_SIZES.size());

// Enumeration for storage types
enum class Store : std::uint8_t { HOST, DEVICE };

// Enumeration for layout types
enum class layout : std::uint8_t { ROW_MAJOR, COL_MAJOR };
}  // namespace manifold


namespace manifold {
#pragma region Dtype to Primitive
template<DType T>
struct DTypeToPrimitive {
  using type = float;
};

template<>
struct DTypeToPrimitive<DType::F32> {
  using type = float;
};

template<>
struct DTypeToPrimitive<DType::F64> {
  using type = double;
};

template<>
struct DTypeToPrimitive<DType::UINT8> {
  using type = uint8_t;
};

template<>
struct DTypeToPrimitive<DType::UINT16> {
  using type = uint16_t;
};
#pragma endregion
}  // namespace manifold
