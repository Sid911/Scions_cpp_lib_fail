//
// Created by sid on 20/10/23.
//
#pragma once
#include <cstdint>

namespace scions::d_type {
enum TYPE {
    INT8,
    INT16,
    INT32,
    INT64,
    F16,
    F32,
    F64,
};

constexpr uint8_t getDTypeSizeBytes(const TYPE t) {
    switch (t) {
    case INT8:
        return sizeof(int8_t);
    case INT16:
        return sizeof(int16_t);
    case INT32:
        return sizeof(int32_t);
    case INT64:
        return sizeof(int64_t);
    case F16:
        return sizeof(int16_t);
    case F32:
        return sizeof(float_t);
    case F64:
        return sizeof(double);
    }
}

} // namespace scions::d_type