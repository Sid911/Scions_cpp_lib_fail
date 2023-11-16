//
// Created by sid on 12/11/23.
//
#pragma once

#include "ep/common/common.h"
namespace scions::ep::cpu {
struct  MemoryManager {
    std::vector<std::unique_ptr<uint8_t[]>> static_memory;
    std::vector<std::unique_ptr<uint8_t[]>> dynamic_memory;
};
} // namespace scions::ep::cpu