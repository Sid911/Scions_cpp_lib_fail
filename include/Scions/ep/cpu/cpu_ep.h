//
// Created by sid on 23/10/23.
//

#pragma once
#include "Scions/core/op/op.h"
#include "cpu_execution_stats.h"
#include "Scions/ep/common/common.h"
#include "Scions/ep/common/runtime_sequential_graph.h"
#include "mkl.h"

constexpr std::string formatBytes(size_t bytes) {
    const char *units[] = {"b", "KiB", "MiB", "GiB", "TiB"};
    int i = 0;
    double size = bytes;
    while (size > 1024 && i < 5) {
        size /= 1024;
        i++;
    }
    std::locale locale;
    return fmt::format("{:.2f} {}", size, units[i]);
}
namespace scions::ep::cpu {
struct CPUOptions {
    bool is_debug;
};

class CPUDynamicExecutionProvider {
  public:
    [[nodiscard("Message: Returning object of CPUExecutionProvider.")]]
    CPUDynamicExecutionProvider(graph::RuntimeSequentialGraph &graph_,
                                CPUOptions &options_);
    [[nodiscard("Message: Returning object of CPUExecutionProvider.")]]
    CPUDynamicExecutionProvider(const graph::RuntimeSequentialGraph &graph_,
                                const CPUOptions &options_);

    std::expected<bool, std::string> allocateAll();

    std::expected<bool, std::string>
    allocateAll(std::unordered_map<size_t, uint8_t *> &&memory_map);

    std::expected<bool, std::string>
    allocateAll(std::unordered_map<std::string_view, uint8_t *> &&memory_map);

  private:
    graph::RuntimeSequentialGraph graph;
    const CPUOptions options;

    std::vector<std::unique_ptr<uint8_t[]>> static_memory;
};

/* # Todo : Make Memory Engine which manages memory
 * # Because This will allow Execution providers to generate EP
 * # Ops at compile time. Greatly increasing optimisation potential
 * # Also, it should be this way because managing memory will be big
 * # for future revisions where dynamic data structures are used
 */

template <size_t Op, size_t Mem> class CPUStaticExecutionProvider {
  public:
    [[nodiscard(
        "Message: Returning object of CPUExecutionProvider.")]] constexpr
    CPUStaticExecutionProvider(graph::SequentialGraph<Op, Mem> &graph_,
                               CPUOptions &options_)
        : graph(graph_), options(options_){};

    [[nodiscard(
        "Message: Returning object of CPUExecutionProvider.")]] constexpr
    CPUStaticExecutionProvider(const graph::SequentialGraph<Op, Mem> &graph_,
                               const CPUOptions &options_)
        : graph(graph_), options(options_){};

    std::expected<bool, std::string> allocateAll() {
        auto &memObjects = graph.memDescriptor.memoryObjects;
        for (auto i = 0; i < Mem; i++) {
            static_memory[i] = new uint8_t[memObjects[i].bytes];
            if (options.is_debug) {
                print(fg(fmt::color::dim_gray), "CPU EP: Allocating {} \n",
                      formatBytes(memObjects[i].bytes));
            }
        }
        return true;
    }

    std::expected<bool, std::string>
    allocateAll(std::unordered_map<size_t, uint8_t *> &&memory_map) {
        auto &memObjects = graph.memDescriptor.memoryObjects;
        for (int i = 0; i < memObjects.size(); ++i) {
            if (memory_map.contains(i)) {
                static_memory[i] = memory_map[i];
            } else {
                static_memory[i] = new uint8_t[memObjects[i].bytes];
                if (options.is_debug) {
                    print(fg(fmt::color::yellow_green),
                          "CPU EP: Allocating {} \n",
                          formatBytes(memObjects[i].bytes));
                }
            }
        }
        return true;
    }

    std::expected<bool, std::string>
    allocateAll(std::unordered_map<std::string_view, uint8_t *> &&memory_map) {
        auto &memObjects = graph.memDescriptor.memoryObjects;
        for (int i = 0; i < memObjects.size(); ++i) {
            if (auto &current_name = memObjects[i].name;
                memory_map.contains(current_name)) {
                static_memory[i] = memory_map[current_name];
            } else {
                static_memory[i] = new uint8_t[memObjects[i].bytes];
                if (options.is_debug) {
                    print(fg(fmt::color::yellow_green),
                          "CPU EP: Allocating {} \n",
                          formatBytes(memObjects[i].bytes));
                }
            }
        }
        return true;
    }

    uint8_t *getMemory(size_t index) { return static_memory[index]; }

    std::expected<CPUExecutionStats, std::string> executeGraph() {
        using namespace std::chrono;
        const auto start = high_resolution_clock::now();
        for (op::OpDesc &op : graph.ops) {
            executeOp(op, static_memory);
        }
        const auto end = high_resolution_clock::now();
        const auto duration = duration_cast<milliseconds>(end - start);
        return CPUExecutionStats{duration};
    }

  private:
    const graph::SequentialGraph<Op, Mem> graph;
    const CPUOptions options;

    std::array<uint8_t *, Mem> static_memory;
};

} // namespace scions::ep::cpu
