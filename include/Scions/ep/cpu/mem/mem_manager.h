//
// Created by sid on 12/11/23.
//
#pragma once

#include "Scions/core/mem/mem_desc.h"
#include "Scions/ep/common/common.h"

namespace scions::ep::cpu {
namespace _internal {
  struct CpuMemRef {
    std::span<uint8_t> memory_bytes_ref;
    d_type::TYPE type;
    std::span<const uint32_t> shape;
    uint32_t dimension;
  };
}  // namespace _internal

template<size_t Mem, uint64_t StaticSize>
class CpuMemoryManager {
public:
  std::array<uint8_t, StaticSize> static_memory;

  std::vector<_internal::CpuMemRef> mem_refs;
  // Not used as of now
  // std::vector<std::shared_ptr<uint8_t[]>> dynamic_memory;

  CpuMemoryManager(const mem::MemDescriptor<Mem> d_) : descriptor(d_), static_memory() {
    mem_refs.reserve(descriptor.memoryObjects.size());
    for (size_t i = 0; const mem::MemObject &obj : descriptor.memoryObjects) {
      const auto ref_span   = std::span<uint8_t>(static_memory.begin() + obj.offset, obj.bytes);
      const auto shape_span = std::span(obj.shape.begin(), obj.dimension);
      mem_refs.emplace_back(ref_span, obj.type, shape_span, obj.dimension);
      i++;
    }
  }

  template<typename T>
  std::span<T> getSpan(size_t index) {
    const auto &obj = descriptor.memoryObjects.at(index);
    fmt::println("CpuMemManager: offset = {}", obj.offset);
    fmt::println("CpuMemManager: bytes = {}, size = {}", obj.bytes, obj.bytes / sizeof(T));
    auto span = std::span<T>(reinterpret_cast<T *>(static_memory.data() + obj.offset), obj.bytes / sizeof(T));
    return span;
  }

  _internal::CpuMemRef getMemRef(size_t index) const { return mem_refs.at(index); }

private:
  const mem::MemDescriptor<Mem> descriptor;
};
}  // namespace scions::ep::cpu
