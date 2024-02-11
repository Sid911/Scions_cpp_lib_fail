//
// Created by sid on 24/12/23.
//

#pragma once
#include "cpu_graph.hpp"
#include "manifold/constants.hpp"
#include "manifold/op_type.hpp"
#include "ops/element_wise_cpu.hpp"
#include "scions/common/common.hpp"

namespace scions::cpu {
namespace _internal {
  using CPU_FUNCTION_TYPE = std::function<void()>;
  consteval bool isOpEqual(auto ex_arr, size_t N, manifold::OpType &type) { return ex_arr[N].type == type; }

  inline void invalidCpuOp() { throw std::runtime_error(std::format("No CPU OP compatible op found")); }

  template<typename T, size_t N, auto SEL_ARR, size_t TEN_SIZE>
  inline std::array<T *, N> generatePointerArr(std::array<RawData<>, TEN_SIZE> &ptrs) {
    std::array<T *, N> refs{};
    static_assert(N <= SEL_ARR.size());
    for (size_t i = 0; i < N; ++i) { refs[i] = reinterpret_cast<T *>(ptrs[SEL_ARR[i]].data_ptr); }
    return refs;
  }

  template<typename T, auto exp, auto D_ARR>
  inline void SwitchIMPL(auto &memStore) {
    using namespace manifold;
    constexpr auto OP                   = exp.type;
    constexpr auto IN_IND               = exp.input_indices;
    constexpr auto OUT_IND              = exp.output_indices;
    constexpr TensorReflection OUT_DATA = D_ARR[OUT_IND[0]];
    auto ten_ptrs                       = memStore.tensor_refs;
    auto in_arr                         = generatePointerArr<T, exp.inp_size, IN_IND>(ten_ptrs);
    auto out_arr                        = generatePointerArr<T, exp.out_size, OUT_IND>(ten_ptrs);

//    if constexpr (OP == ARRAY_ELM_ADD) { element_wise_add<T, OUT_DATA.size, exp.inp_size>(out_arr[0], in_arr); }
//    if constexpr (OP == ARRAY_ELM_MUL) { element_wise_mul<T, OUT_DATA.size, exp.inp_size>(out_arr[0], in_arr); }
    //    invalidCpuOp();
  }
}  // namespace _internal

template<auto graph, size_t N = 0>
inline auto exec_cpu_graph(auto &memStore) {
  static constexpr auto &D_ARR  = graph.data;
  static constexpr auto &EX_ARR = graph.expressions;

  using inp_type = typename manifold::DTypeToPrimitive<D_ARR[EX_ARR[N].output_indices[0]].data_type>::type;

  _internal::SwitchIMPL<inp_type, EX_ARR[N], D_ARR>(memStore);
  if constexpr (N < EX_ARR.size() - 1) { exec_cpu_graph<graph, N + 1>(memStore); }
}
}  // namespace scions::cpu