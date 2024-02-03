//
#include "manifold/array.hpp"
#include "manifold/constants.hpp"
#include "manifold/ops/element_wise_ops.hpp"
#include "manifold/static_graph.hpp"
#include "manifold/tensor.hpp"
#include "manifold/utility.hpp"
#include "scions/ep/cpu/cpu_mem_store.hpp"
#include "scions/ep/cpu/exec_graph_gen.hpp"

consteval auto buildGraph() noexcept {
  using namespace manifold;
  // Example usage
  constexpr auto arr1 = Array<DType::F32, 200>(1);
  constexpr auto arr2 = Array<DType::F32, 200>(2);
  constexpr auto arr3 = Array<DType::F32, 200>(3);

  constexpr auto ten     = Tensor<TBase<DType::F32, 10, 20>>(4);
  constexpr auto ten_arr = ten.to_array();

  // ArrayAdd just reutrns an expression template class which includes input and output
  // and other relevent data
  auto add_1_2         = op::array_add(ten_arr, std::array{ arr1, arr2 });
  auto mul_1_2         = op::array_mul(arr3, std::array{ arr1, arr2, ten_arr });
  const std::array exp = { &add_1_2, &mul_1_2, &add_1_2 };
  return StaticGraph(exp);
}


int main() {
  static constexpr auto res     = buildGraph();
  static constexpr auto meta    = getMetadata(res);
  static constexpr auto compact = manifold::compact<meta>(res);

  scions::cpu::CpuMemStore<meta> mem_store(compact);
  mem_store.initializeMemory();

  scions::cpu::exec_cpu_graph<compact>(mem_store);

  std::println("");
  std::print("Tensor IDs : \n");
  for (const auto &data : compact.data) { std::print("{}\n", data.id); }
  std::print("\nExpression hashes: \n");
  for (const auto &exp : compact.expressions) { std::print("{}\n", exp); }
  std::print("Compact MemStore vs compact = {}:{}\n", sizeof(decltype(mem_store)), sizeof(decltype(compact)));

  return 0;
}