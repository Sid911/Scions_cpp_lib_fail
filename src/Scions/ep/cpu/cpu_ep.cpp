//
// Created by sid on 23/10/23.
//

#include "Scions/ep/cpu/cpu_ep.h"

namespace scions::ep::cpu {
CPUDynamicExecutionProvider::CPUDynamicExecutionProvider(
    graph::RuntimeSequentialGraph &graph_, CPUOptions &options_)
    : graph(graph_), options(options_) {}

CPUDynamicExecutionProvider::CPUDynamicExecutionProvider(
    const graph::RuntimeSequentialGraph &graph_, const CPUOptions &options_)
    : graph(graph_), options(options_) {}

std::expected<bool, std::string> CPUDynamicExecutionProvider::allocateAll() {
    for (const auto &mem_object : graph.mem_objects) {
        if (options.is_debug) {
            print(fg(fmt::color::yellow_green), "CPU EP: Allocating {} \n",
                  formatBytes(mem_object.bytes));
        }
        static_memory.push_back(std::make_unique<uint8_t[]>(mem_object.bytes));
    }
    return true;
}

std::expected<bool, std::string> CPUDynamicExecutionProvider::allocateAll(
    std::unordered_map<size_t, uint8_t *> &&memory_map) {
    for (int i = 0; i < graph.mem_objects.size(); ++i) {
        if (memory_map.contains(i)) {
            static_memory[i].reset(memory_map[i]);
        } else {
            if (options.is_debug) {
                print(fg(fmt::color::yellow_green), "CPU EP: Allocating {} \n",
                      formatBytes(graph.mem_objects[i].bytes));
            }
            static_memory.push_back(
                std::make_unique<uint8_t[]>(graph.mem_objects[i].bytes));
        }
    }
    return true;
}

std::expected<bool, std::string> CPUDynamicExecutionProvider::allocateAll(
    std::unordered_map<std::string_view, uint8_t *> &&memory_map) {
    for (int i = 0; i < graph.mem_objects.size(); ++i) {
        if (auto &current_name = graph.mem_objects[i].name;
            memory_map.contains(current_name)) {
            static_memory[i].reset(memory_map[current_name]);
        } else {
            if (options.is_debug) {
                print(fg(fmt::color::yellow_green), "CPU EP: Allocating {} \n",
                      formatBytes(graph.mem_objects[i].bytes));
            }
            static_memory.push_back(
                std::make_unique<uint8_t[]>(graph.mem_objects[i].bytes));
        }
    }
    return true;
}
} // namespace scions::ep::cpu
