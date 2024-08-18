#pragma once

#include "manifold/expression.hpp"
#include "manifold/macro.hpp"
#include "manifold/op_type.hpp"
#include <cstddef>
#include <cstdint>

namespace manifold::op {


template<std::size_t N>
constexpr ExpressionReflection
exp_group(uint32_t id, const std::array<ExpressionReflection, N> &inp, bool pin = false) {
    static_assert(N <= MANIFOLD_MAX_EXP_INPUT,
                  "Manifold: Input count more than MANIFOLD_MAX_EXP_INPUT + MANIFOLD_MAX_EXP, \
    please define it as per your needs (this is special case)");

    static_assert(N > 1, "Manifold: A group should Have more than 1 expressions");

    using inp_type   = std::array<uint32_t, MANIFOLD_MAX_EXP_INPUT>;
    using out_type   = std::array<uint32_t, MANIFOLD_MAX_EXP_OUTPUT>;
    using param_type = std::array<std::byte, MANIFOLD_PARAM_BYTES_MAX>;

    auto inputs  = inp_type();
    auto outputs = out_type();
    auto param   = param_type();
    for (size_t i = 0; i < N; i++) {
        inputs.at(i) = inp.at(i).id;
    }

    static_assert(MANIFOLD_MAX_EXP_OUTPUT > 0,
                  "Manifold: MANIFOLD_MAX_EXP_OUTPUT < 1 , this op needs to \
    save metadata consider increasing it");

    outputs.at(0) = pin;
    return { id, OpType::EXP_GROUP, {}, N, inputs, 0, outputs, param };
}
}  // namespace manifold::op
