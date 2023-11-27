//
// Created by sid on 23/10/23.
//

#pragma once

#include "cpu_execution_stats.h"
#include "op/execute_op.h"
#include "Scions/core/op/op.h"
#include "Scions/ep/common/common.h"
#include "Scions/ep/common/runtime_sequential_graph.h"
#include "Scions/ep/cpu/mem/mem_manager.h"
#include "mkl.h"

constexpr std::string formatBytes(const size_t bytes) {
    constexpr std::array units{ "b", "KiB", "MiB", "GiB", "TiB" };
    size_t i    = 0;
    double size = static_cast<double>(bytes);
    while (size > 1024 && i < 5) {
        size /= 1024;
        i++;
    }
    return fmt::format("{:.2f} {}", size, units[i]);
}


namespace scions::ep::cpu {
struct CPUOptions {
    bool is_debug = false;
};

/* # Todo : Make Memory Engine which manages memory
 * # Because This will allow Execution providers to generate EP
 * # Ops at compile time. Greatly increasing optimisation potential
 * # Also, it should be this way because managing memory will be big
 * # for future revisions where dynamic data structures are used
 */

template<size_t Op, size_t Mem, uint64_t Size>
class CPUStaticExecutionProvider {
public:
    [[nodiscard(
          "Message: Returning object of CPUExecutionProvider.")]]
    constexpr CPUStaticExecutionProvider(
        const graph::SequentialGraph<Op, Mem> &graph_,
        CpuMemoryManager<Mem, Size> &manager_,
        const CPUOptions &options_)
        : graph(graph_), options(options_), manager(manager_) {}

    // Todo: Constructor without manger passed in

    template<>
    std::expected<CPUExecutionStats, std::string> executeGraph() {
        using namespace std::chrono;
        const auto start = high_resolution_clock::now();

        for (const op::OpDesc &op : graph.ops) {
            executeOp(op, manager);
        }
        const auto end      = high_resolution_clock::now();
        const auto duration = duration_cast<milliseconds>(end - start);
        return CPUExecutionStats{ duration };
    }

private:
    const graph::SequentialGraph<Op, Mem> graph;
    CpuMemoryManager<Mem, Size> &manager;
    const CPUOptions options;

};

} // namespace scions::ep::cpu
