#include "Scions/core/graph/sequential_graph.h"
#include "Scions/core/mem/mem_desc.h"
#include "Scions/core/op/basic_op.h"
#include "Scions/core/op/op.h"
#include "Scions/core/op/tensor_op.h"
#include "Scions/ep/cpu/cpu_ep.h"
#include "Scions/ep/cpu/mem/mem_manager.h"
#include "type_traits"
#include "Scions/ep/cpu/op/execute_op.h"

using namespace std;
#define OP_SIZE 2


template<int i,size_t Ops, size_t Mem, int32_t OpID>
struct GraphExecutor {
  static void execute(const scions::graph::SequentialGraph<Ops,Mem>& graph) {

  }
};


template<int i, size_t Ops, size_t Mem>
struct GraphExecutor<i,Ops, Mem,scions::op::tensor::TENSOR_MULTIPLY_OP_ID>  {
  static void execute(const scions::graph::SequentialGraph<Ops,Mem>& graph) {
    print(fg(fmt::color::green_yellow),"MULITPLY!\n");
  }
};


template<int i, size_t Ops, size_t Mem>
struct GraphExecutor<i,Ops, Mem,scions::op::tensor::TENSOR_ADD_OP_ID>  {
  static void execute(const scions::graph::SequentialGraph<Ops,Mem>& graph) {
    print(fg(fmt::color::green_yellow),"ADD!\n");
  }
};


template<size_t Mem, uint64_t Size>
consteval auto executeCPUGraph(const auto& graph, const scions::ep::cpu::CpuMemoryManager<Mem, Size>& manager) {
    using namespace std::chrono;
    using namespace scions;
    const auto start = high_resolution_clock::now();

    for (const op::OpDesc &op : graph.ops) { ep::cpu::executeOp(op, manager); }
    const auto end      = high_resolution_clock::now();
    const auto duration = duration_cast<milliseconds>(end - start);
    return ep::cpu::CPUExecutionStats{ duration };
}

// builds the graph at compile time
[[nodiscard]] consteval auto buildGraph() {

  // Alright here will be the example start
  using namespace scions;

  static constexpr array mems = {
    mem::StaticMem(1024, 1024, "inp1"),
    mem::StaticMem(1024, 1024, "inp2"),
    mem::StaticMem(1024, 1024, "inp3"),
  };

  constexpr mem::MemDescriptor desc = mem::MemDescriptor(mems);

  static constexpr std::array<op::OpDesc, OP_SIZE> ops = {
    op::tensor::TensorAddOpDesc(0, 1, 2),
    op::tensor::TensorMultiplyOpDesc(0, 3, 2),
  };

  constexpr graph::SequentialGraph graph = { ops, desc };

  return graph;
}

int main() {
  using namespace scions;
  using namespace scions::ep;
  // Compile time graph
  static constexpr auto res   = buildGraph();
  static constexpr auto &desc = res.memDescriptor;
  static constexpr auto mem_size  = desc.memoryObjects.size();
  static constexpr auto by    = desc.getTotalBytes();

  // set CPUExecutionProvider Options
  constexpr cpu::CPUOptions options{ true };
  // Note: Always set this to static if you don't want to blow past
  // the stack size
  static auto manager = cpu::CpuMemoryManager<mem_size, by>(desc);

  // auto provider = cpu::CPUStaticExecutionProvider<OP_SIZE, mem_size, by>(res, manager, options);

  cpu::CPUExecutionStats stats = executeCPUGraph(res,manager);
  return 0;
}
