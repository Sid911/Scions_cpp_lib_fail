//
// Created by sid on 9/12/23.
//
#pragma once
#include "manifold/static_graph.hpp"
#include "manifold/utility.hpp"
#include "raw_data.hpp"
#include "scions/common/common.hpp"


namespace scions::cpu {

#define __COMPACT_TEMP_PARAMS G.graph_data_size, G.graph_op_size, G.max_in, G.max_out
template<manifold::GraphMetadata G>
class CpuMemStore {
public:
  [[nodiscard]] explicit CpuMemStore(const manifold::CompactStaticGraph<__COMPACT_TEMP_PARAMS> &graph) noexcept
    : _graph(graph) {}

  void initializeMemory() {
    if constexpr (G.f32_size) { f32_container = std::make_unique<std::array<float, G.f32_size>>(); }
    if constexpr (G.u8_size) { u8_container = std::make_unique<std::array<uint8_t, G.u8_size>>(); }
    static_assert(G.f32_size != 0);
    //! Note :
    //! probably should use an array of pointers and just increment them
    //! but for two it should be fine
    //! std::array<size_t, manifold::NUM_DTYPE> offsets;
    size_t f32_offset{}, u8_offset{};

    for (size_t i{}; i < G.graph_data_size; ++i) {
      manifold::TensorReflection &tensor = _graph.data[i];
      void *data_ptr                     = nullptr;

      switch (tensor.data_type) {
      case manifold::DType::UINT8:
        data_ptr = &u8_container->at(u8_offset);
        u8_offset += tensor.size + 1;
        break;
      case manifold::DType::UINT16: break;
      case manifold::DType::UINT32: break;
      case manifold::DType::UINT64: break;
      case manifold::DType::INT8: break;
      case manifold::DType::INT16: break;
      case manifold::DType::INT32: break;
      case manifold::DType::INT64: break;
      case manifold::DType::F32:
        data_ptr = &f32_container->at(f32_offset);
        f32_offset += tensor.size + 1;
        break;
      case manifold::DType::F64: break;
      }
      if (data_ptr == nullptr) {
        // Todo : Fix tensor print problem
        std::print("Tensor :\n{}", tensor);
        throw std::format_error("Something went wrong initializing the tensor, maybe no implementation for the type");
      }

      tensor_refs[i] = RawData{ data_ptr, &tensor };
    }
  }

  [[nodiscard]] consteval static CpuMemStore fromStaticGraph(const manifold::StaticGraph<> &graph) noexcept {
    return CpuMemStore(manifold::compact<G>(graph));
  }

  CpuMemStore(const CpuMemStore &other)       = delete;
  CpuMemStore(CpuMemStore &&other)            = delete;
  CpuMemStore &operator=(CpuMemStore &&other) = delete;

  std::array<RawData<>, G.graph_data_size> tensor_refs;

private:
  std::unique_ptr<std::array<float, G.f32_size>> f32_container;
  std::unique_ptr<std::array<uint8_t, G.u8_size>> u8_container;

  manifold::CompactStaticGraph<__COMPACT_TEMP_PARAMS> _graph;
};
}  // namespace scions