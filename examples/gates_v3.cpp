#include "manifold/compute_graph.hpp"
#include "manifold/constants.hpp"
#include "manifold/ops/element_wise_ops.hpp"
#include "manifold/ops/special_ops.hpp"
#include "manifold/static_graph.hpp"
#include "manifold/tensor.hpp"
#include <array>
#include <print>

// Note: Std variant and overload idiom
// std::apply ? std::visit

[[nodiscard]]
consteval auto graphGen() {
  using namespace manifold;

  const auto v1 = Scalar<DType::F32>(0);
  const auto v2 = Scalar<DType::F32>(1);
  const auto v3 = Scalar<DType::F32>(2);
  const auto v4 = Scalar<DType::F32>(3);

  const auto exp = op::exp(4, v2, v1);
  // We should be able to optimize this right?
  const auto copy_v2_v3 = op::copy(5, v3, v2);
  const auto add_2      = op::elm_add(6, v3, 1.0F);
  const auto group      = op::exp_group(7, std::array{ copy_v2_v3, add_2 }, true);

  const auto mul = op::elm_mul(8, v4, std::array{ v2, v3 });
  const auto f   = op::elm_div(9, v4, std::array{ v1, v2 });
  // const auto group_all = op::exp_group(9, std::array{ exp, group, mul });

  // For now first operation of the container must not be a group expression ^ above will not work if passed
  // directly at first
  const auto container = SymbolContainer{ std::array{ v1.reflect(), v2.reflect(), v3.reflect(), v4.reflect() },
    std::array{ exp, group, mul, add_2, copy_v2_v3 } };

  return container.to_dag();
}

int main() {
  using namespace manifold;
  static constexpr auto graph = graphGen().topologicalSort();
  constexpr auto io_count     = graph.ioCount();
  constexpr auto io           = graph.dagIOIndices<io_count>();
  // constexpr auto comp_graph   = ComputeGraph(graph, io);

  // static constexpr auto size = sizeof(decltype(graph));
  // static constexpr auto edge_size = sizeof(ExprEdge);

  std::print("{}", graph.toDot());
  std::ranges::for_each(io.in_tensor_idxs, [](auto i) { std::print("{}", i); });
}
