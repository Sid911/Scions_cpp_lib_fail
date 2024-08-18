//
// Created by sid on 29/11/23.
//

#pragma once

#include <cstdint>
namespace manifold {
enum class DType : std::uint8_t { UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64, F32, F64 };

constexpr uint8_t NUM_DTYPE                          = 10;
constexpr std::array<uint8_t, NUM_DTYPE> DTYPE_SIZES = { 1u, 2u, 4u, 8u, 1u, 2u, 4u, 8u, 4u, 8u };

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
template<DType Type, typename T>
concept IsCompatibleDType =
    (Type == DType::F32 && std::is_same_v<T, float>) || (Type == DType::F64 && std::is_same_v<T, float>)
    || (Type == DType::INT32 && std::is_same_v<T, int>) || (Type == DType::UINT8 && std::is_same_v<T, uint8_t>)
    || (Type == DType::UINT16 && std::is_same_v<T, uint16_t>);
}  // namespace manifold
