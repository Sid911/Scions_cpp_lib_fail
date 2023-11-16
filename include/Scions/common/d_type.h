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

constexpr uint8_t getDTypeSizeBytes(const TYPE t){
    switch (t) {
    case INT8:
        return 1;
    case INT16:
        return 2;
    case INT32:
        return 4;
    case INT64:
        return 8;
    case F16:
        return 2;
    case F32:
        return 4;
    case F64:
        return 8;
    }
}

} // namespace scions::d_type