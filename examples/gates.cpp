#include "Scions/core/graph/sequential_graph.h"
#include "Scions/core/mem/mem_desc.h"
#include "Scions/core/op/basic_op.h"
#include "Scions/core/op/op.h"
#include "Scions/core/op/tensor_op.h"
#include "Scions/ep/cpu/cpu_ep.h"
#include "Scions/ep/cpu/mem/mem_manager.h"
#include "fmt/core.h"

using namespace std;

// builds the graph at compile time
[[nodiscard]] consteval auto buildGraph() {

  // Alright here will be the example start
  using namespace scions;

  static constexpr array objects = {
    mem::StaticMemObject(16, "inp1"),
    mem::StaticMemObject(1, 2, 3, "inp2"),
    mem::StaticMemObject({ 1024 * 10, 1024 * 10 }, "inp3"),
    mem::StaticMemObject({ 64, 64, 64 }, "inp4"),
  };

  constexpr mem::MemDescriptor desc = mem::MemDescriptor(objects);

  constexpr std::array<op::OpDesc, 2> tensors = {
    op::tensor::TensorAddOpDesc(0, 1, 2),
    op::tensor::TensorMultiplyOpDesc(0, 3, 2),
  };

  constexpr graph::SequentialGraph graph = { tensors, desc };

  return graph;
}

int main() {
  using namespace scions;
  using namespace scions::ep;
  // Compile time graph
  static constexpr auto res   = buildGraph();
  static constexpr auto &desc = res.memDescriptor;
  static constexpr auto size  = desc.memoryObjects.size();
  static constexpr auto by    = desc.getTotalBytes();
  // set CPUExecutionProvider Options
  constexpr cpu::CPUOptions options;

  // Note: Always set this to static if you don't want to blow past
  // the stack size
  static auto manager  = cpu::CpuMemoryManager<size, by>(res.memDescriptor);
  const span<int> span = manager.getSpan<>(2);
  span[2]              = 3;

  fmt::println("{}, size : {}, sizeof manager", span[2], span.size(), sizeof(manager));

  auto provider = cpu::CPUStaticExecutionProvider(res, manager,options);

  return 0;
}
