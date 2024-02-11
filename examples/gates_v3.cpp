#include "manifold/constants.hpp"
#include "manifold/ops/element_wise_ops.hpp"
#include "manifold/static_graph.hpp"
#include "manifold/tensor.hpp"
#include "manifold/utility.hpp"
#include <print>
// Note: Std variant and overload idiom
// std::apply ? std::visit

[[nodiscard]]
consteval auto graphGen() {
  using namespace manifold;

  // uint32_t tensors{}, ops{};

  auto ten1   = Tensor<TBase<DType::F32, 10, 3, 10>>(0);
  auto ten2   = Tensor<TBase<DType::F32, 10, 3, 10>>(1);
  auto ten3   = Tensor<TBase<DType::F32, 10, 10, 3>>(2);
  auto ten4   = ten3.reshape<10, 3, 10>();
  auto t_fill = op::array_fill(3, std::array{ ten1, ten2 }, 5.0F);
  auto exp    = op::array_add(4, ten4, std::array{ ten1, ten2, ten4 });

  SymbolContainer graph{ std::array{ ten1.reflect(), ten2.reflect(), ten3.reflect() }, std::array{ t_fill, exp } };
  return graph;
}

int main() {
  auto graph = graphGen();
  std::print("{}", graph);

  using namespace manifold;
  std::print("{} {} {} {}",
    sizeof(TensorReflection),
    sizeof(ExpressionReflection),
    sizeof(std::variant<TensorReflection, ExpressionReflection>),
    sizeof(decltype(graph)));
}
