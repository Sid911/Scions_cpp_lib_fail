#include "manifold/constants.hpp"
#include "manifold/ops/element_wise_ops.hpp"
#include "manifold/static_graph.hpp"
#include "manifold/tensor.hpp"
#include <print>

// Note: Std variant and overload idiom
// std::apply ? std::visit

[[nodiscard]]
consteval auto graphGen() {
  using namespace manifold;

  // uint32_t tensors{}, ops{};

  auto ten1                    = Tensor<TBase<DType::F32, 10, 3, 10>>(0);
  auto ten2                    = Tensor<TBase<DType::F32, 10, 3, 10>>(1);
  auto ten3                    = Tensor<TBase<DType::F32, 10, 10, 3>>(2);
  auto ten4                    = ten3.reshape<10, 3, 10>();
  auto ten5                    = Tensor<TBase<DType::F32, 10, 3, 10>>(3);
  ExpressionReflection t_fill  = op::array_fill(4, std::array{ ten1, ten4 }, 5.0F);
  ExpressionReflection t_fill2 = op::array_fill(5, std::array{ ten2 }, 10.0F);
  ExpressionReflection exp     = op::array_add(6, ten5, std::array{ ten1, ten2, ten4 });

  // ids need to be sequential and index based as of right now ids can't be random
  SymbolContainer container{
    std::array{ ten1.reflect(), ten2.reflect(), ten3.reflect(), ten5.reflect() },  // data
    std::array{ exp, t_fill, t_fill2 }  // ops
  };

  return container.to_dag().topologicalSort();
}

int main() {
  using namespace manifold;
  static constexpr auto graph = graphGen();
  auto dot                    = graph.toDot();
  std::println("{}", dot);
}
