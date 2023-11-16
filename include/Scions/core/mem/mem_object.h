//
// Created by sid on 13/10/23.
//

#pragma once
#include "Scions/common/common.h"
#include "Scions/common/d_type.h"

namespace scions::mem {

class StaticMemObject {
  public:
    // total size in bytes
    uint64_t total_size;
    const std::string_view name;
    const uint32_t dimension;
    const d_type::TYPE data_type;
    std::array<uint32_t, SC_MAX_DIMS> shape;


    consteval StaticMemObject(const uint32_t size_, std::string_view name_,
                              const d_type::TYPE type_ = d_type::F16)
        : total_size(size_), name(name_), dimension(1), data_type(type_),
          shape({size_}) {
        if (total_size % d_type::getDTypeSizeBytes(data_type) != 0) {
            throw std::logic_error("Shape of input (bytes) is not divisible by "
                                   "size of data type. Please correct.");
        }
    }


    constexpr StaticMemObject(uint32_t x, uint32_t y, std::string_view name_,
                              d_type::TYPE type_ = d_type::F16)
        : total_size(x * y * d_type::getDTypeSizeBytes(type_)), name(name_),
          dimension(2), data_type(type_), shape({x, y}) {}


    constexpr StaticMemObject(uint32_t x, uint32_t y, uint32_t z,
                              std::string_view name_,
                              d_type::TYPE type_ = d_type::F16)
        : total_size(x * y * z * d_type::getDTypeSizeBytes(type_)), name(name_),
          dimension(3), data_type(type_), shape({x, y, z}) {}


    consteval StaticMemObject(std::initializer_list<uint32_t> list,
                              std::string_view name_,
                              d_type::TYPE type_ = d_type::F16)
        : name(name_),
          dimension(SC_MAX_DIMS > list.size() ? list.size() : SC_MAX_DIMS),
          data_type(type_), shape() {

        total_size = 1;
        uint16_t i = 0;
        for (const uint32_t val : list) {
            if (i >= dimension)
                break;
            total_size *= val;
            shape[i] = val;
            i++;
        }
        // just keep it as total number of elements
        total_size *= d_type::getDTypeSizeBytes(type_);
    }
};

class MemObject {
  public:
    uint64_t bytes;
    uint64_t id;
    std::string_view name;
    d_type::TYPE type;

    constexpr MemObject()
        : bytes(0), id(UINT64_MAX), name("bruh"), type(d_type::TYPE::F16){};

    constexpr MemObject(const uint64_t size, const std::string_view name_,
                        const uint64_t id_,
                        const d_type::TYPE type_ = d_type::F16)
        : name(name_), bytes(size), id(id_), type(type_) {
        if (bytes % d_type::getDTypeSizeBytes(type_) != 0) {
            throw std::logic_error("Size of input (bytes) is not divisible by "
                                   "size of data type. Please correct.");
        }
    }

    constexpr MemObject(const StaticMemObject &object, const uint64_t id_)
        : name(object.name), bytes(object.total_size), id(id_),
          type(object.data_type) {
        if (bytes % d_type::getDTypeSizeBytes(type) != 0) {
            throw std::logic_error("Size of input (bytes) is not divisible by "
                                   "size of data type. Please correct.");
        }
    }
};
} // namespace scions::mem