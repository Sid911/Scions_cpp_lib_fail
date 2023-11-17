//
// Created by sid on 13/10/23.
//

#pragma once
#include "Scions/common/common.h"
#include "mem_object.h"

namespace scions::mem {
struct MemRef {
  size_t index;
  std::string_view name;
  d_type::TYPE data_type;
};

template<size_t N>
class MemDescriptor {
public:
  std::array<MemObject, N> memoryObjects;

  /**
   * @class MemDescriptor
   * @brief Represents a memory descriptor for a collection of static memory
   * objects.
   *
   * The MemDescriptor class provides a way to manage a collection of static
   * memory objects. It calculates the total size of the memory objects and
   * provides methods to access the individual memory objects.
   *
   * @tparam N The number of memory objects in the collection.
   */
  constexpr MemDescriptor(const std::array<StaticMemObject, N> &objects)
    : memoryObjects() {
    uint64_t bytes = 0;
    for (int i = 0; i < N; i++) {
      memoryObjects[i] =
        MemObject(objects[i].total_size,
          objects[i].name,
          i,
          bytes,
          objects[i].data_type
          );
      bytes+= objects[i].total_size;
    }

    total_bytes = bytes;
  }

  [[nodiscard]] constexpr MemRef getRef(size_t index) const { return MemRef{ index, memoryObjects[index].name }; }

  /**
   * @brief Get a reference to a memory object by name.
   *
   * This method searches for a memory object with the given name in the
   * collection of memory objects. If found, it returns a reference to the
   * memory object. Otherwise, it throws a runtime_error indicating that the
   * memory object was not found.
   *
   * @param search The name of the memory object to search for.
   * @return A MemRef object representing the reference to the memory object.
   * @throws std::runtime_error if the memory object is not found.
   */
  [[nodiscard]] consteval MemRef getRef(std::string_view search) const {
    auto index = -1;
    for (int i = 0; i < N; ++i) {
      if (memoryObjects[i].name == search) {
        index = i;
        break;
      }
    }
    if (index == -1) {
      throw std::runtime_error(std::string(
        "memory object named " + std::string(search) + "not found"));
    }

    return MemRef{ index, memoryObjects[index].name };
  }

  constexpr uint64_t getTotalBytes() const noexcept { return total_bytes; }

private:
  uint64_t total_bytes;

  static consteval std::array<MemObject, N>
    buildMemObjects(const std::array<StaticMemObject, N> &obj) {
    std::array<MemObject, N> temp;

    for (int i = 0; i < N; i++) {
      temp[i] =
        MemObject(obj[i].total_size, obj[i].name, i, obj[i].data_type);
    }
    return temp;
  }
};

} // namespace scions::mem
// mem
