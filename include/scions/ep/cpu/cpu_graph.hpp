//
#pragma once

#include "scions/common/common.hpp"

namespace scions {
template<size_t N>
class CpuGraph {
public:
    [[nodiscard]] CpuGraph(std::array<std::function<void()>, N> _o) noexcept : ops(_o) {}

private:
    std::array<std::function<void()>, N> ops;
};

#undef __COMPACT_TEMP_PARAMS
}  // namespace scions