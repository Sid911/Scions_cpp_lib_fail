//
// Created by sid on 12/11/23.
//
#pragma once

#include "Scions/ep/common/common.h"
#include "Scions/core/mem/mem_desc.h"

namespace scions::ep::cpu {

template<size_t Mem, uint64_t StaticSize>
class CpuMemoryManager {
public:
  std::array<uint8_t, StaticSize> static_memory;
  // Not used as of now
  // std::vector<std::shared_ptr<uint8_t[]>> dynamic_memory;

  CpuMemoryManager(const mem::MemDescriptor<Mem> &d_)
    : descriptor(d_) {}

  template<typename T>
  std::span<T> getSpan(size_t index) {
    const auto &obj = descriptor.memoryObjects.at(index);
    fmt::println("CpuMemManager: offset = {}", obj.offset);
    fmt::println("CpuMemManager: bytes = {}, size = {}", obj.bytes, obj.bytes/ sizeof(T));
    auto span       = std::span<T>(
      reinterpret_cast<T *>(static_memory.data() + obj.offset),
      obj.bytes / sizeof(T)
      );
    return span;
  }

  template<typename T>
  std::span<T> getSpan(size_t index) const {
    const auto &obj = descriptor.memoryObjects.at(index);
    fmt::println("CpuMemManager: offset = {}", obj.offset);
    fmt::println("CpuMemManager: bytes = {}, size = {}", obj.bytes, obj.bytes/ sizeof(T));
    auto span       = std::span<T>(
      reinterpret_cast<T *>(static_memory.data() + obj.offset),
      obj.bytes / sizeof(T)
      );
    return span;
  }

private:
  const mem::MemDescriptor<Mem> &descriptor;
};
} // namespace scions::ep::cpu
