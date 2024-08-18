#pragma once
#include "manifold/expression.hpp"
#include "manifold/tensor.hpp"
#include <cstddef>
#include <format>

namespace manifold {

struct TensorNode : TensorReflection {
    uint32_t total_out{};
    std::array<uint32_t, MANIFOLD_TENSORNODE_MAX_OUT> outgoing{};
    uint32_t incoming{ UINT32_MAX };

    constexpr TensorNode() = default;

    constexpr TensorNode(TensorReflection ten_ref,
                         uint32_t out_num,
                         std::array<uint32_t, MANIFOLD_TENSORNODE_MAX_OUT> out,
                         uint32_t inp = UINT32_MAX)
        : TensorReflection(ten_ref), total_out(out_num), outgoing(out), incoming(inp) {}

    constexpr explicit TensorNode(TensorReflection ten_ref) : TensorReflection(ten_ref), outgoing() {}
};

struct ExprEdge : ExpressionReflection {
    std::array<uint32_t, MANIFOLD_MAX_EXP_INPUT> inp_idxs{};
    std::array<uint32_t, MANIFOLD_MAX_EXP_OUTPUT> out_idxs{};

    constexpr ExprEdge() = default;

    constexpr ExprEdge(ExpressionReflection exp_ref,
                       std::array<uint32_t, MANIFOLD_MAX_EXP_INPUT> &input_idxs,
                       std::array<uint32_t, MANIFOLD_MAX_EXP_OUTPUT> &output_idxs)
        : ExpressionReflection(exp_ref), inp_idxs(input_idxs), out_idxs(output_idxs) {}

    constexpr explicit ExprEdge(const ExpressionReflection &exp_ref) : ExpressionReflection(exp_ref), inp_idxs(), out_idxs() {}
};


}  // namespace manifold


template<>
struct std::formatter<manifold::TensorNode> : std::formatter<manifold::TensorReflection> {
    template<typename FormatContext>
    constexpr auto format(const manifold::TensorNode &ten, FormatContext &ctx) const {
        std::format_to(ctx.out(), "{}\n", static_cast<manifold::TensorReflection>(ten));
        std::format_to(ctx.out(), "incoming idx: {}\ntotal out: {}\n outgoing: [ ", ten.incoming, ten.total_out);

        for (size_t i{}; i < ten.total_out; i++) {
            std::format_to(ctx.out(), "{} ", ten.outgoing.at(i));
        }
        return std::format_to(ctx.out(), "]");
    }
};


template<>
struct std::formatter<manifold::ExprEdge> : std::formatter<manifold::ExpressionReflection> {
    template<typename FormatContext>
    constexpr auto format(const manifold::ExprEdge &expr, FormatContext &ctx) const {
        std::format_to(ctx.out(), "{}\n", static_cast<manifold::ExpressionReflection>(expr));
        std::format_to(ctx.out(), "input idxs: [ ");

        for (size_t i{}; i < expr.num_inputs; i++) {
            std::format_to(ctx.out(), "{} ", expr.inp_idxs.at(i));
        }
        std::format_to(ctx.out(), "]\noutput idxs: [ ");

        for (size_t i{}; i < expr.num_outputs; i++) {
            std::format_to(ctx.out(), "{} ", expr.out_idxs.at(i));
        }
        return std::format_to(ctx.out(), " ]");
    }
};
