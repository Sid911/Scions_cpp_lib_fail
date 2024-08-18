
#include "manifold/dag.hpp"
#include "manifold/dag_node.hpp"
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <sys/types.h>
#include <tuple>

namespace manifold {
template<size_t TSize, size_t ESize, size_t InSize, size_t OutSize>
struct ComputeGraph {
    std::array<TensorNode, TSize> data;
    std::array<ExprEdge, ESize> edges;
    std::array<uint32_t, InSize> in_ids;
    std::array<uint32_t, OutSize> out_ids;

    //! Note: Here **in_data** and **out_data** is actual index of the Tensors that are
    //! in the @param _data
    constexpr ComputeGraph(const std::array<TensorNode, TSize> &_data,
                           const std::array<ExprEdge, ESize> &_edges,
                           const std::array<uint32_t, InSize> &_in_data,
                           const std::array<uint32_t, OutSize> &_out_data)
        : data(_data), edges(_edges), in_ids(_in_data), out_ids(_out_data) {}
//! @param dag : Subgraph generated for the specific @param _out_data, can be done using
    //! @StaticDAG::subgraph or @StaticDAG::subgraphByIdx
    //!
    //! Note: Here **in_data** and **out_data** is actual index of the Tensors that are
    //! in the @dag.data
    constexpr ComputeGraph(const StaticDAG<TSize, ESize> &dag,
                           const std::array<uint32_t, InSize> &_in_data,
                           const std::array<uint32_t, OutSize> &_out_data)
        : ComputeGraph(dag.data, dag.edges, _in_data, _out_data) {}


    constexpr ComputeGraph(const StaticDAG<TSize, ESize> &dag, const DagIOIdx<InSize, OutSize> &dag_io)
        : ComputeGraph(dag.data, dag.edges, dag_io.in_tensor_idxs, dag_io.out_tensor_idxs) {}


    constexpr void partialAdj() {

    }

    // partial adjoint = v_(i->j) = v_i bar * partial_adj(vj, vi)
    constexpr void nodeDiff(std::vector<TensorNode>& tens, std::vector<ExprEdge>& exprs, uint32_t expr_idx) const {
        const auto& expr = edges.at(expr_idx);


    }

    constexpr auto reverse(uint32_t idx= 0) const {
        std::vector<TensorNode> tens;
        std::vector<ExprEdge> expers;

        // Go from output 0 by default

    }

    constexpr std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> autoDiffCount() const {
        std::vector<uint32_t> curr;
        curr.reserve(ESize);
        for (uint32_t i = OutSize - 1; i > 0; i--) {
            TensorNode out_ten = data.at(i);
            if (out_ten.incoming == UINT32_MAX) {
                continue;
            }
        }
        return this;
    }
};
}  // namespace manifold
