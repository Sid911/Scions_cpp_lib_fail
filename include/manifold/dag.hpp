#pragma once

#include "manifold/concepts.hpp"
#include "manifold/expression.hpp"
#include "manifold/macro.hpp"
#include "manifold/op_type.hpp"
#include "manifold/tensor.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "manifold/dag_node.hpp"

namespace manifold {

template<size_t InSize, size_t OutSize>
struct DagIOIdx {
    std::array<uint32_t, InSize> in_tensor_idxs;
    std::array<uint32_t, OutSize> out_tensor_idxs;
};

template<size_t TSize, size_t ESize>
struct StaticDAG {
    std::array<TensorNode, TSize> data;
    std::array<ExprEdge, ESize> edges;
    std::array<int64_t, ESize> group_mask;

    constexpr StaticDAG(std::array<TensorNode, TSize> &_data,
                        std::array<ExprEdge, ESize> &_edges,
                        std::array<int64_t, ESize> &masks)
        : data(_data), edges(_edges), group_mask(masks) {}

    constexpr StaticDAG(const std::array<TensorReflection, TSize> &tensors,
                        const std::array<ExpressionReflection, ESize> &exprs)
        : edges(), data(), group_mask() {

        for (uint32_t i = 0; i < TSize; ++i) {
            data[i] = TensorNode(tensors[i]);
        }
        auto new_exprs = generateMaskWithIdxs(exprs, group_mask);
        generateIndices(new_exprs, tensors, edges, data);
    }

    template<typename T>
    static inline constexpr auto generateMaskWithIdxs(const std::array<T, ESize> &exprs,
            _internal::IsStdArray auto &mask) {
        std::array<T, ESize> exprs_c(exprs);

        // generate mask for fragmented groups
        for (uint i = 0; i < ESize; i++) {
            const ExpressionReflection &e = exprs[i];

            if (e.type != OpType::EXP_GROUP) {
                mask[i] = i;
                continue;
            }
            mask[i] = -static_cast<int64_t>(i);

            for (uint j = 0; j < e.num_inputs; j++) {
                auto expected_mask = e.outputs[0] == 1 ? -static_cast<int64_t>(i) : i;
                // check if next is expected id element
                if (exprs_c[i + j + 1].id == e.inputs[j]) {
                    mask[i + j + 1] = expected_mask;
                    continue;
                }
                // else rotate the array from right of the i to expected id element
                uint dis = i + 1;
                while (dis < ESize) {
                    if (exprs_c[dis].id == e.inputs[j]) {
                        mask[dis] = expected_mask;
                        break;
                    }
                    dis++;
                }
                if (dis == ESize) {
                    throw std::logic_error(
                        "Manifold: Member expression of a group not found toward right of the group in the array");
                }
                for (size_t x{ dis }; x >= i + j + 2; x--) {
                    std::swap(mask[x], mask[x - 1]);
                    std::swap(exprs_c[x], exprs_c[x - 1]);
                }
                if (exprs_c[i + j + 1].id != e.inputs[j]) {
                    throw "bruh";
                }
            }
            i += e.num_inputs;
        }
        return exprs_c;
    }

    static inline constexpr void generateIndices(const auto &exprs, const auto &tensors, auto &edges, auto &data) {
        for (uint32_t i = 0; i < ESize; ++i) {
            // Reset expression
            const ExpressionReflection &expr = exprs.at(i);
            // weird way but okay
            ExprEdge &edge = edges.at(i);
            edge           = ExprEdge(expr);

            if (edge.type == OpType::EXP_GROUP) {
                continue;
            }
            for (uint32_t j = 0; j < expr.num_outputs; ++j) {
                // Search for the tensor with the output ID
                auto is_ID    = [id = expr.outputs.at(j)](auto &ref) {
                    return ref.id == id;
                };
                auto iterator = std::ranges::find_if(tensors, is_ID);
                if (iterator == tensors.end()) {
                    throw std::logic_error("Internal Error: Could not find tensor with an ID");
                }
                auto idx              = static_cast<uint32_t>(std::ranges::distance(tensors.begin(), iterator));
                data.at(idx).incoming = i;
                edge.out_idxs.at(j)   = idx;
            }

            for (uint32_t j = 0; j < expr.num_inputs; ++j) {
                auto is_ID    = [id = expr.inputs.at(j)](auto &ref) {
                    return ref.id == id;
                };
                auto iterator = std::ranges::find_if(tensors, is_ID);
                if (iterator == tensors.end()) {
                    throw std::logic_error("Internal Error: Could not find tensor with an ID");
                }

                auto idx        = static_cast<uint32_t>(std::ranges::distance(tensors.begin(), iterator));
                TensorNode &ten = data.at(idx);
                if (ten.total_out > MANIFOLD_TENSORNODE_MAX_OUT) {
                    throw std::runtime_error("Error: more than MANIFOLD_TENSORNODE_MAX_OUT outgoing edges from a node");
                }
                ten.outgoing.at(ten.total_out++) = i;
                edge.inp_idxs.at(j)              = idx;
            }
        }
    }

    static constexpr int64_t computeRanks(auto &exprs, auto &tens, auto &ranks, uint32_t idx) {
        ExprEdge &expr = exprs.at(idx);
        if (ranks.at(idx)) {
            return ranks.at(idx);
        }
        if (!expr.num_inputs) {
            return 0;
        }

        uint32_t rank = expr.num_inputs;
        for (uint32_t i{}; i < exprs.at(idx).num_inputs; i++) {
            auto &ten = tens.at(expr.inputs.at(i));
            if (ten.incoming == UINT32_MAX) {
                continue;
            }
            rank += computeRanks(exprs, tens, ranks, ten.incoming);
        }

        ranks.at(idx) = rank;
        return rank;
    }

    constexpr void topoSortGroup(std::array<ExprEdge, ESize> &exprs,
                                 std::array<int64_t, ESize> &ranks,
                                 uint32_t idx,
                                 uint32_t inp_len) const {
        for (auto i = idx; i < idx + inp_len; i++) {
            ExprEdge &expr = exprs.at(i);
            if (expr.type == OpType::EXP_GROUP) {
                if (expr.outputs[0] == 1) {
                    i += expr.num_inputs;
                    continue;
                }
                auto max_len = std::min(expr.num_inputs, uint32_t(ESize) - idx - 1);
                topoSortGroup(exprs, ranks, idx + 1, max_len);
            }
            // sort exprs using ranks skipping negative ranks
            for (uint32_t j = i + 1; j < idx + inp_len; j++) {
                if (ranks[j] > 0 &&  // Skip negative ranks
                        ranks[j] < ranks[i]) {
                    std::swap(exprs[i], exprs[j]);
                    std::swap(ranks[i], ranks[j]);
                }
            }
        }
    }

    //! Topologically sorts and generates new DAG based on effective ranks.
    //!
    //! Note : doesn't sort tensors but only the OP/ Expressions.
    constexpr StaticDAG<TSize, ESize> topologicalSort() const {
        std::array<ExprEdge, ESize> e_cpy(edges);
        std::array<TensorNode, TSize> t_cpy(data);
        std::array<int64_t, ESize> masks{};

        // Generate ranks
        std::array<int64_t, ESize> ranks{};
        for (uint32_t i{}; i < ESize; i++) {
            if (group_mask[i] < 0) {
                ranks[i] = group_mask[i];
                continue;
            }
            computeRanks(e_cpy, t_cpy, ranks, i);
        }
        // Sort WRT Pinned groups
        // Group Expression is never sorted nor does pinned expressions ie. any negative ranked
        topoSortGroup(e_cpy, ranks, 0, ESize);

        for (auto &ten : t_cpy) {
            ten.total_out = 0;
        }
        generateIndices(e_cpy, data, e_cpy, t_cpy);
        auto new_exprs = generateMaskWithIdxs(e_cpy, masks);
        return StaticDAG{ t_cpy, new_exprs, masks };
    }

    [[nodiscard]] constexpr uint32_t tensorIdxFromID(uint32_t iden) const {
        for (uint32_t i{}; i < TSize; i++) {
            if (data.at(i).id == iden) {
                return i;
            }
        }
        return UINT32_MAX;
    }

    [[nodiscard]] constexpr uint32_t opIdxFromID(uint32_t iden) const {
        for (uint32_t i{}; i < ESize; i++) {
            if (edges.at(i).id == iden) {
                return i;
            }
        }
        return UINT32_MAX;
    }

    //-------------------------------------------------- Dot Convert -----------------------------------------------------


    //! Generates Dot representation of the graph (https://graphviz.org/docs/layouts/dot/). Remember your 'Dot'
    //! interpreter might topologically sort the DAG which might be misrepresentation of the underlying graph itself.
    //! Be sure to look at the index number and id.
    //!
    //! Note: This function is not really constexpr supported right now as it modifies string to generate the
    //!       Dot String, a buffer implementation would be much rather suited for constexpr evaluation
    //!
    //! Todo: Create actual subgraphs based on subgroups
    [[nodiscard]] constexpr std::string toDot(const std::string &graph_name = "G",
            const std::string &rank_dir                                           = "LR") const {
        std::string str = std::format("digraph {} {}\n  rankdir=\"{}\"\n", graph_name, "{", rank_dir);
        using namespace std::literals;
        constexpr auto t_style  = "[shape=box,color=aliceblue,style=\"filled,rounded\"]"sv;
        constexpr auto op_style = "[shape=circle,color=lavender,style=\"filled,rounded\"]"sv;

        // Todo: Actually add subgroups
        // constexpr auto colors = std::array{ "grey0"sv, "grey10"sv, "grey20"sv, "grey30"sv, "grey40"sv, "grey50"sv };

        std::vector<std::string> t_str(TSize);
        std::vector<std::string> op_str(ESize);

        for (size_t i{}; i < TSize; i++) {
            const TensorNode &node = data.at(i);
            t_str[i]               = std::format("T{}_{}", node.id, i);
            str += std::format("  {} {};\n", t_str[i], t_style);
        }

        for (size_t i{}; i < ESize; i++) {
            const ExpressionReflection &expr = edges.at(i);
            if (expr.type == OpType::EXP_GROUP) {
                continue;
            }
            auto mask   = group_mask.at(i);
            auto pinned = mask == int64_t(i) ? ""sv : mask < 0 ? "_t"sv : "_f"sv;
            op_str[i]   = std::format("OP{}_{}_{}{}", expr.id, i, abs(mask), pinned);
            str += std::format("  {} {};\n", op_str[i], op_style);
        }

        for (size_t i{}; i < TSize; i++) {
            const TensorNode &node = data.at(i);
            str += "  " + t_str.at(i) + " -> { ";
            for (size_t j{}; j < node.total_out; j++) {
                str += op_str.at(node.outgoing.at(j)) + " ";
            }
            str += "};\n";
        }

        for (size_t i{}; i < ESize; i++) {
            const ExprEdge &edg = edges.at(i);
            if (edg.type == OpType::EXP_GROUP) {
                continue;
            }
            str += "  " + op_str.at(i) + " -> { ";
            for (size_t j{}; j < edg.num_outputs; j++) {
                str += t_str.at(edg.out_idxs.at(j)) + " ";
            }
            str += "};\n";
        }

        str += "}";
        return str;
    }

    //--------------------------------------------------- Sub graph ------------------------------------------------------

    // Todo: This only works with constexpr workflow but not normal
    [[nodiscard]] constexpr std::pair<uint32_t, uint32_t> ioCount() const noexcept {
        uint32_t in{};
        uint32_t out{};
        for (const TensorNode &dat : data) {
            if (dat.incoming == UINT32_MAX) {
                in++;
            }
            if (dat.total_out == 0) {
                out++;
            }
        }
        return { in, out };
    }


    template<size_t OutSize>
    [[nodiscard]] constexpr std::pair<std::array<bool, TSize>, std::array<bool, ESize>> depSearch(
                const std::array<uint32_t, OutSize> &arr,
                std::array<bool, TSize> t_visited,
                std::array<bool, ESize> e_visited,
    const uint32_t max = OutSize) const noexcept {

        // Dep search (backtrace)
        for (uint i{}; i < max; i++) {
            auto data_idx = arr.at(i);
            // find deps
            TensorNode curr = data.at(data_idx);
            if (t_visited.at(data_idx)) {
                continue;
            }
            t_visited.at(data_idx) = true;

            if (curr.incoming == UINT32_MAX) {
                continue;
            }
            ExprEdge incom              = edges.at(curr.incoming);
            e_visited.at(curr.incoming) = true;
            auto [r_t, r_e]             = depSearch(incom.inp_idxs, t_visited, e_visited, incom.num_inputs);
            // merge the results
            std::transform(t_visited.begin(), t_visited.end(), r_t.begin(), t_visited.begin(), std::bit_or<>());
            std::transform(e_visited.begin(), e_visited.end(), r_e.begin(), e_visited.begin(), std::bit_or<>());
        }
        return { t_visited, e_visited };
    }


    //! @param out : "Output Indices" of required output tensors.
    //! @return : Returns pair of Tensor and Edge Count. This is usually used to construct
    //!           subGraph
    template<size_t OutSize>
    [[nodiscard]] consteval std::pair<uint32_t, uint32_t> subGraphCountByIdx(
        const std::array<uint32_t, OutSize> &out) const {
        auto [r_t, r_e] = depSearch(out, {}, {});

        auto t_size = static_cast<uint32_t>(std::count_if(r_t.begin(), r_t.end(), [](bool value) {
            return value;
        }));
        auto e_size = static_cast<uint32_t>(std::count_if(r_e.begin(), r_e.end(), [](bool value) {
            return value;
        }));

        return { t_size, e_size };
    }

    template<size_t OutSize>
    [[nodiscard]] consteval std::pair<uint32_t, uint32_t> subGraphCount(const std::array<uint32_t, OutSize> &out) const {

        std::array<uint32_t, OutSize> ids{};
        for (size_t i{}; i < OutSize; i++) {
            ids[i] = tensorIdxFromID(out[i]);
        }

        return subGraphCountByIdx(ids);
    }


    //! You should already know the value of template @param T and @param E this can be done
    //! This can be done by calling subGraphCount
    template<uint32_t T, uint32_t E, size_t OutSize>
    [[nodiscard]] constexpr StaticDAG<T, E> subgraphByIdx(const std::array<uint32_t, OutSize> &out) const {
        auto [r_t, r_e] = depSearch(out, {}, {});

        // Count and add those to the list of elements
        std::array<TensorNode, T> tensors{};
        std::array<ExprEdge, E> exprs{};

        uint32_t jx = 0;
        for (size_t i = 0; i < TSize; i++) {
            if (r_t.at(i)) {
                tensors[jx++] = data.at(i);
            }
        }
        jx = 0;
        for (size_t i = 0; i < ESize; i++) {
            if (r_e.at(i)) {
                exprs[jx++] = edges.at(i);
            }
        }

        return { tensors, exprs };
    }

    template<uint32_t T, uint32_t E, size_t OutSize>
    [[nodiscard]] constexpr StaticDAG<T, E> subgraph(const std::array<uint32_t, OutSize> &out) const {

        std::array<uint32_t, OutSize> ids{};
        for (size_t i{}; i < OutSize; i++) {
            ids[i] = tensorIdxFromID(out[i]);
        }

        return subgraphByIdx<T, E>(ids);
    }


    template<std::pair<uint32_t, uint32_t> P, size_t OutSize>
    [[nodiscard]] constexpr StaticDAG<P.first, P.second> subgraph(const std::array<uint32_t, OutSize> &out) const {
        return subgraph<P.first, P.second>(out);
    }

    template<uint32_t in, uint32_t out>
    [[nodiscard]] constexpr DagIOIdx<in, out> dagIOIndices() const {
        std::array<uint32_t, in> in_idxs;
        std::array<uint32_t, out> out_idxs;

        uint32_t i{};
        uint32_t jx{};
        for (uint32_t kx{}; const TensorNode &dat : data) {
            if (dat.incoming == UINT32_MAX) {
                in_idxs[i++] = kx;
            }
            if (dat.total_out == 0) {
                out_idxs[jx++] = kx;
            }
            kx++;
        }
        return { in_idxs, out_idxs };
    }

    template<const std::pair<uint32_t, uint32_t> Pair>
    [[nodiscard]] constexpr DagIOIdx<Pair.first, Pair.second> dagIOIndices() const {
        return dagIOIndices<Pair.first, Pair.second>();
    }
};
}  // namespace manifold


template<size_t TSize, size_t ESize>
struct std::formatter<manifold::StaticDAG<TSize, ESize>> : std::formatter<std::string> {
    template<typename FormatContext>
    constexpr auto format(const manifold::StaticDAG<TSize, ESize> &dag, FormatContext &ctx) const {
        std::format_to(ctx.out(), "data: [\n");
        for (const manifold::TensorNode &ten : dag.data) {
            std::format_to(ctx.out(), "{}\n\n", ten);
        }
        std::format_to(ctx.out(), "]\nedges : [\n");
        for (const manifold::ExprEdge &expr : dag.edges) {
            std::format_to(ctx.out(), "{}\n\n", expr);
        }
        return std::format_to(ctx.out(), "]\n");
    }
};
