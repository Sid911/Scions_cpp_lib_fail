
#pragma once

#include <algorithm>
#include <cstdint>
#include <expected>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

// FMT
#include <fmt/color.h>
#include <fmt/core.h>

#ifndef SC_MAX_DIMS
#define SC_MAX_DIMS 5
#endif

#ifndef SC_OP_INPUT_MAX
#define SC_OP_INPUT_MAX 10
#endif

#ifndef SC_OP_OUTPUT_MAX
#define SC_OP_OUTPUT_MAX 5
#endif
