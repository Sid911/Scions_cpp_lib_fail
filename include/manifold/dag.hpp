#pragma once

#include "manifold/expression.hpp"
#include "manifold/macro.hpp"
#include "manifold/tensor.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

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

  constexpr TensorNode(TensorReflection ten_ref) : TensorReflection(ten_ref), outgoing() {}
};

struct ExprEdge : ExpressionReflection {
  std::array<uint32_t, MANIFOLD_MAX_EXP_REF_INPUT> inp_idxs;
  std::array<uint32_t, MANIFOLD_MAX_EXP_REF_OUTPUT> out_idxs;

  constexpr ExprEdge() = default;

  constexpr ExprEdge(ExpressionReflection exp_ref,
    std::array<uint32_t, MANIFOLD_MAX_EXP_REF_INPUT> &input_idxs,
    std::array<uint32_t, MANIFOLD_MAX_EXP_REF_OUTPUT> &output_idxs)
    : ExpressionReflection(exp_ref), inp_idxs(input_idxs), out_idxs(output_idxs) {}

  constexpr ExprEdge(const ExpressionReflection &exp_ref) : ExpressionReflection(exp_ref), inp_idxs(), out_idxs() {}
};


template<size_t TSize, size_t ESize>
struct StaticDAG {
  std::array<TensorNode, TSize> data;
  std::array<ExprEdge, ESize> edges;

  constexpr StaticDAG(std::array<TensorNode, TSize> &_data, std::array<ExprEdge, ESize> &_edges)
    : data(_data), edges(_edges) {}

  constexpr StaticDAG(const std::array<TensorReflection, TSize> &tensors,
    const std::array<ExpressionReflection, ESize> &exprs)
    : edges(), data() {

    for (uint32_t i = 0; i < TSize; ++i) { data[i] = tensors[i]; }
    generateIndices(exprs, tensors, edges, data);
  }

  static inline constexpr void generateIndices(const auto &exprs, const auto &tensors, auto &edges, auto &data) {
    for (uint32_t i = 0; i < ESize; ++i) {
      const ExpressionReflection &expr = exprs.at(i);
      ExprEdge &edge                   = edges.at(i);
      edge                             = expr;

      for (uint32_t j = 0; j < expr.num_outputs; ++j) {
        // Search for the tensor with the output ID
        auto is_ID    = [id = expr.outputs.at(j)](auto &ref) { return ref.id == id; };
        auto iterator = std::ranges::find_if(tensors, is_ID);
        if (iterator == tensors.end()) { throw std::runtime_error("Internal Error: Could not find tensor with an ID"); }
        auto idx              = static_cast<uint32_t>(std::ranges::distance(tensors.begin(), iterator));
        data.at(idx).incoming = i;
        edge.out_idxs.at(j)   = idx;
      }

      for (uint32_t j = 0; j < expr.num_inputs; ++j) {
        auto is_ID    = [id = expr.inputs.at(j)](auto &ref) { return ref.id == id; };
        auto iterator = std::ranges::find_if(tensors, is_ID);
        if (iterator == tensors.end()) { throw std::runtime_error("Internal Error: Could not find tensor with an ID"); }

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

  static inline constexpr uint32_t computeRanks(auto &exprs, auto &tens, auto &ranks, uint32_t idx) {
    auto &expr = exprs.at(idx);
    if (ranks.at(idx)) { return ranks.at(idx); }
    if (!expr.num_inputs) { return 0; }

    uint32_t rank = expr.num_inputs;
    for (uint32_t i{}; i < exprs.at(idx).num_inputs; i++) {
      auto &ten = tens.at(expr.inputs.at(i));
      if (ten.incoming == UINT32_MAX) { continue; }
      rank += computeRanks(exprs, tens, ranks, ten.incoming);
    }

    ranks.at(idx) = rank;
    return rank;
  }

  inline constexpr StaticDAG<TSize, ESize> topologicalSort() const {
    std::array<ExprEdge, ESize> e_cpy(edges);
    std::array<TensorNode, TSize> t_cpy(data);

    // very simple sort
    // std::ranges::sort(e_cpy, [](auto &x, auto &y) { return x.num_inputs < y.num_inputs; });

    // Compute effective ranks for acyclic graph
    std::array<uint32_t, ESize> ranks{};
    for (uint32_t i{}; i < ESize; i++) { computeRanks(e_cpy, t_cpy, ranks, i); }

    // Custom insertion sort to sort based on ranks
    auto it_e = e_cpy.begin();
    for (auto it_r = ranks.begin(); it_r != ranks.end(); it_r++ ) {
      auto const insertion_r = std::upper_bound(ranks.begin(),it_r,*it_r);
      auto const insertion_e = e_cpy.begin() + std::distance(ranks.begin(),insertion_r);
      std::rotate(insertion_r, it_r, it_r+1);
      std::rotate(insertion_e, it_e, it_e+1);
      it_e++;
    }

    for (auto &ten : t_cpy) { ten.total_out = 0; }
    generateIndices(e_cpy, data, e_cpy, t_cpy);

    return StaticDAG<TSize, ESize>{ t_cpy, e_cpy };
  }


  // Generates Dot representation of the graph(https://graphviz.org/docs/layouts/dot/)
  // Note: This function is not really constexpr supported right now as it modifies string to generate the
  //       Dot String
  [[nodiscard]] constexpr std::string toDot(const std::string &graph_name = "G",
    const std::string &rank_dir                                           = "LR") const {
    std::string str = std::format("digraph {} {}\n  rankdir=\"{}\"\n", graph_name, "{", rank_dir);

    constexpr std::string_view t_style  = "[shape=box,color=aliceblue,style=\"filled,rounded\"]";
    constexpr std::string_view op_style = "[shape=circle,color=lavender,style=\"filled,rounded\"]";
    std::vector<std::string> t_str(TSize);
    std::vector<std::string> op_str(ESize);

    for (size_t i{}; i < TSize; i++) {
      const TensorNode &node = data.at(i);
      t_str[i]               = std::format("T{}_{}", node.id, i);
      str += std::format("  {} {};\n", t_str[i], t_style);
    }

    for (size_t i{}; i < ESize; i++) {
      const ExpressionReflection &expr = edges.at(i);
      op_str[i]                        = std::format("OP{}_{}", expr.id, i);
      str += std::format("  {} {};\n", op_str[i], op_style);
    }

    for (size_t i{}; i < TSize; i++) {
      const TensorNode &node = data.at(i);
      str += "  " + t_str.at(i) + " -> { ";
      for (size_t j{}; j < node.total_out; j++) { str += op_str.at(node.outgoing.at(j)) + " "; }
      str += "};\n";
    }

    for (size_t i{}; i < ESize; i++) {
      const ExprEdge &edg = edges.at(i);
      str += "  " + op_str.at(i) + " -> { ";
      for (size_t j{}; j < edg.num_outputs; j++) { str += t_str.at(edg.out_idxs.at(j)) + " "; }
      str += "};\n";
    }

    str += "}";
    return str;
  }
};


}  // namespace manifold
