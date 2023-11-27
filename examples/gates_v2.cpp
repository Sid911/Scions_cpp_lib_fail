//
// Created by sid on 21/11/23.
//

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <fmt/core.h>
#include <memory>
#include <optional>
#include <type_traits>

// NOLINTBEGIN
#ifndef MANIFOLD_MAX_EXP_REF_INPUT
#define MANIFOLD_MAX_EXP_REF_INPUT 5
#endif

#ifndef MANIFOLD_MAX_EXP_REF_OUTPUT
#define MANIFOLD_MAX_EXP_REF_OUTPUT 5
#endif


#ifndef MANIFOLD_MAX_EXP_INPUT
#define MANIFOLD_MAX_EXP_INPUT 10
#endif


#ifndef MANIFOLD_MAX_EXP_OUTPUT
#define MANIFOLD_MAX_EXP_OUTPUT 2
#endif


#ifndef MANIFOLD_MAX_RANK
#define MANIFOLD_MAX_RANK 5
#endif


#ifndef MANIFOLD_STATIC_GRAPH_MAX_MEM
#define MANIFOLD_STATIC_GRAPH_MAX_MEM 20
#endif

#ifndef MANIFOLD_STATIC_GRAPH_MAX_OP
#define MANIFOLD_STATIC_GRAPH_MAX_OP 10
#endif
// NOLINTEND

namespace manifold {
// Enumeration for data types
enum class d_type : std::uint8_t { F32, F64 };

// Enumeration for storage types
enum class Store : std::uint8_t { HOST, DEVICE };

// Enumeration for layout types
enum class layout : std::uint8_t { ROW_MAJOR, COL_MAJOR };

// Helper structure for shapes of a tensor
template<std::uint32_t... Shapes>
struct Shape {
  static constexpr std::size_t rank                      = sizeof...(Shapes);
  static constexpr std::array<std::uint32_t, rank> shape = { Shapes... };
};

#pragma region Mat Array
// Definition for array
template<d_type Type, std::size_t Size, Store Storage = Store::HOST>
struct Array {
  static constexpr d_type data_type   = Type;
  static constexpr std::size_t size   = Size;
  static constexpr Store storage_type = Storage;
  uint32_t id;

  constexpr Array() : id(UINT32_MAX) {}

  [[nodiscard]] constexpr explicit Array(const uint32_t id_) : id(id_) {}
};

// Definition for matrix
template<d_type Type,
  std::size_t Rows,
  std::size_t Cols,
  layout Layout = layout::ROW_MAJOR,
  Store Storage = Store::HOST>
struct Matrix {
  static constexpr d_type data_type = Type;
  static constexpr std::size_t size = Rows * Cols;
  static constexpr Shape<Rows, Cols> shape{};
  static constexpr layout storage_layout = Layout;
  static constexpr Store storage_type    = Storage;
  uint32_t id;


  constexpr Matrix() : id(UINT32_MAX) {}

  [[nodiscard]] constexpr explicit Matrix(const uint32_t id_) : id(id_) {}

  [[nodiscard]] constexpr auto to_array() const { return Array<Type, size, Storage>(id); }
};
#pragma endregion
#pragma region Tensor
template<d_type Type, std::uint32_t... Shapes>
struct TBase {
  static constexpr d_type data_type = Type;
  static constexpr std::size_t size = (Shapes * ...);
  static constexpr Shape<Shapes...> shape{};
};

template<typename TensorBase, Store Storage = Store::HOST, layout Layout = layout::ROW_MAJOR>
struct Tensor : TensorBase {
  static constexpr layout storage_layout = Layout;
  static constexpr Store storage_type    = Storage;
  uint32_t id;

  constexpr Tensor() : id(UINT32_MAX) {}

  [[nodiscard]] constexpr explicit Tensor(const uint32_t id_) : id(id_) {}

  [[nodiscard]] constexpr auto to_array() const { return Array<TensorBase::data_type, TensorBase::size, Storage>(id); }

  template<std::size_t M, std::size_t N>
  [[nodiscard]] constexpr auto to_matrix() const {
    static_assert(TensorBase::size == M * N, "total Matrix size should be equal total tensor size");
    return Matrix<TensorBase::data_type, M, N, Layout, Storage>(id);
  }
};
#pragma endregion

#pragma region reflections
// Reflections
struct ShapeReflection {
  std::size_t rank;
  std::array<std::uint32_t, MANIFOLD_MAX_RANK> shape;

  constexpr ShapeReflection() : rank(0), shape() {}

  [[nodiscard]] constexpr ShapeReflection(const std::uint32_t rank,
    const std::array<std::uint32_t, MANIFOLD_MAX_RANK> &shape)
    : rank(rank), shape(shape) {}

  // NOLINTBEGIN
  [[nodiscard]] constexpr ShapeReflection(const std::uint32_t size) : rank(1), shape({ size }) {}

  constexpr friend std::size_t hash_value(const ShapeReflection &obj) {
    std::size_t seed = 0x48335BC4;
    seed ^= (seed << 6) + (seed >> 2) + 0x29A54A39 + static_cast<std::size_t>(obj.rank);
    for (auto i = 0; i < obj.rank; i++) {
      seed ^= static_cast<uint32_t>(obj.shape[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
  // NOLINTEND
  [[nodiscard]] constexpr ShapeReflection(const std::uint32_t M, const std::uint32_t N) : rank(2), shape({ M, N }) {}
};

struct TensorReflection {
  d_type data_type;
  std::size_t size;
  uint32_t id;
  ShapeReflection shape;
  layout storage_layout;
  Store storage_type;

  constexpr TensorReflection() : data_type(), size(0), id(0), storage_layout(), storage_type() {}

  [[nodiscard]] constexpr TensorReflection(const d_type data_type,
    const std::size_t size,
    const uint32_t id,
    const ShapeReflection &shape_reflection,
    const layout storage_layout,
    const Store storage_type)
    : data_type(data_type), size(size), id(id), shape(shape_reflection), storage_layout(storage_layout),
      storage_type(storage_type) {}


  template<d_type Type, std::size_t Size, Store Storage = Store::HOST>
  constexpr TensorReflection(const Array<Type, Size, Storage> &arr)
    : data_type(Type), size(Size), id(arr.id), shape(Size), storage_layout(layout::ROW_MAJOR), storage_type(Storage) {}

  template<d_type Type,
    std::size_t Rows,
    std::size_t Cols,
    layout Layout = layout::ROW_MAJOR,
    Store Storage = Store::HOST>
  constexpr TensorReflection(const Matrix<Type, Rows, Cols, Layout, Storage> &mat)
    : data_type(Type), size(Rows * Cols), id(mat.id), shape(Rows, Cols), storage_layout(layout::ROW_MAJOR),
      storage_type(Storage) {}

  // NOLINTBEGIN
  constexpr friend std::size_t hash_value(const TensorReflection &obj) {
    std::size_t seed = 0x66D26DCF;
    seed ^= (seed << 6) + (seed >> 2) + 0x1B5EB086 + static_cast<std::size_t>(obj.data_type);
    seed ^= (seed << 6) + (seed >> 2) + 0x0F42AAAE + static_cast<std::size_t>(obj.size);
    seed ^= (seed << 6) + (seed >> 2) + 0x426382BB + static_cast<std::size_t>(obj.id);
    seed ^= (seed << 6) + (seed >> 2) + 0x3BF8AAF1 + hash_value(obj.shape);
    seed ^= (seed << 6) + (seed >> 2) + 0x0FCF80F0 + static_cast<std::size_t>(obj.storage_layout);
    seed ^= (seed << 6) + (seed >> 2) + 0x5167620F + static_cast<std::size_t>(obj.storage_type);
    return seed;
  }
  // NOLINTEND
};
#pragma endregion

enum OpType : uint8_t {
  // Array operations

  // Element wise add two arrays
  ARRAY_ELM_ADD,
  ARRAY_ELM_SUB,
  ARRAY_ELM_MUL,
  ARRAY_ELM_DIV,
  RANDOM_ELM,
  // AXPY is vector scalar product
  ARRAY_AXPY,
  ARRAY_SUM,
  ARRAY_MEAN,

  // Matrix Matrix operations
  MAT_ADD,
  MAT_SUB,
  MAT_TRAN,
  MAT_MUL,
  MAT_INV,

  // Matrix Array ops
  MAT_ARR_MUL,
  MAT_ARR_ADD


};
const char *to_string(OpType e) {
  switch (e) {
  case ARRAY_ELM_ADD: return "ARRAY_ELM_ADD";
  case ARRAY_ELM_SUB: return "ARRAY_ELM_SUB";
  case ARRAY_ELM_MUL: return "ARRAY_ELM_MUL";
  case ARRAY_ELM_DIV: return "ARRAY_ELM_DIV";
  case RANDOM_ELM: return "RANDOM_ELM";
  case ARRAY_AXPY: return "ARRAY_AXPY";
  case ARRAY_SUM: return "ARRAY_SUM";
  case ARRAY_MEAN: return "ARRAY_MEAN";
  case MAT_ADD: return "MAT_ADD";
  case MAT_SUB: return "MAT_SUB";
  case MAT_TRAN: return "MAT_TRAN";
  case MAT_MUL: return "MAT_MUL";
  case MAT_INV: return "MAT_INV";
  case MAT_ARR_MUL: return "MAT_ARR_MUL";
  case MAT_ARR_ADD: return "MAT_ARR_ADD";
  default: return "unknown";
  }
}

#pragma region expression
struct ExpressionReflection {
  // Expression part

  // OpType of the expression
  OpType type;
  uint32_t num_inputs;
  uint32_t num_outputs;
  std::unique_ptr<std::array<ExpressionReflection, MANIFOLD_MAX_EXP_REF_INPUT>> inputs;
  std::unique_ptr<std::array<ExpressionReflection, MANIFOLD_MAX_EXP_REF_INPUT>> outputs;

  // TensorReflection part
  std::optional<TensorReflection> tensor;
  constexpr ExpressionReflection() : type(), num_inputs(0), num_outputs(0) {}

  // delete the copy constructor
  ExpressionReflection(const ExpressionReflection &) = delete;

  // if you also want to delete the copy assignment operator, do this as well:
  ExpressionReflection &operator=(const ExpressionReflection &) = delete;

  ExpressionReflection(ExpressionReflection &&)            = default;  // Move constructor
  ExpressionReflection &operator=(ExpressionReflection &&) = default;  // Move assignment operator

  ~ExpressionReflection() = default;

  [[nodiscard]] constexpr ExpressionReflection(const OpType type_,
    const uint32_t num_inputs_,
    std::unique_ptr<std::array<ExpressionReflection, MANIFOLD_MAX_EXP_REF_INPUT>> &inputs_,
    const uint32_t num_outputs_,
    std::unique_ptr<std::array<ExpressionReflection, MANIFOLD_MAX_EXP_REF_INPUT>> &outputs_,
    const std::optional<TensorReflection> &tensor_)
    : type(type_), num_inputs(num_inputs_), num_outputs(num_outputs_), inputs(std::move(inputs_)),
      outputs(std::move(outputs_)), tensor(tensor_) {}

  // ReSharper disable once CppNonExplicitConvertingConstructor
  template<d_type Type, std::size_t Size, Store Storage = Store::HOST>
  // ReSharper disable once CppNonExplicitConvertingConstructor
  constexpr ExpressionReflection(Array<Type, Size, Storage> &array) : ExpressionReflection(TensorReflection(array)) {}

  // ReSharper disable once CppNonExplicitConvertingConstructor
  template<d_type Type,
    std::size_t Rows,
    std::size_t Cols,
    layout Layout = layout::ROW_MAJOR,
    Store Storage = Store::HOST>
  // ReSharper disable once CppNonExplicitConvertingConstructor
  constexpr ExpressionReflection(const Matrix<Type, Rows, Cols, Layout, Storage> &mat)
    : ExpressionReflection(TensorReflection(mat)) {}

  [[nodiscard]] constexpr explicit ExpressionReflection(const TensorReflection &tensor)
    : type(), num_inputs(0), num_outputs(0), tensor(tensor) {}

  // NOLINTBEGIN
  [[nodiscard]] constexpr bool isTensor() const { return tensor.has_value(); }
  constexpr friend std::size_t hash_value(const ExpressionReflection &obj) {
    std::size_t seed = 0x713065A4;
    if (obj.isTensor()) {
      seed ^= (seed << 6) + (seed >> 2) + 0x26072FDF + hash_value(obj.tensor.value());
    } else {
      seed ^= (seed << 6) + (seed >> 2) + 0x19D2EDF0 + static_cast<std::size_t>(obj.type);
      seed ^= (seed << 6) + (seed >> 2) + 0x14604031 + static_cast<std::size_t>(obj.num_inputs);
      seed ^= (seed << 6) + (seed >> 2) + 0x17E21DA7 + static_cast<std::size_t>(obj.num_outputs);
      for (auto i = 0; i < obj.num_inputs; ++i) {
        seed ^= (seed << 6) + (seed >> 2) + 0x591C6A0B + hash_value(obj.inputs->at(i));
      }
      for (auto i = 0; i < obj.num_inputs; ++i) {
        seed ^= (seed << 6) + (seed >> 2) + 0x1F5BB1BC + hash_value(obj.outputs->at(i));
      }
    }
    return seed;
  }
  // NOLINTEND
};

template<size_t MaxInput = MANIFOLD_MAX_EXP_INPUT, size_t MaxOutput = MANIFOLD_MAX_EXP_OUTPUT>
struct Expression {
  OpType type;
  uint64_t id;
  uint32_t num_inputs;
  uint32_t num_outputs;
  std::array<std::uint32_t, MANIFOLD_MAX_EXP_INPUT> input_indices;
  std::array<std::uint32_t, MANIFOLD_MAX_EXP_OUTPUT> output_indices;
  std::size_t hash;

  constexpr Expression() : type(), id(0), num_inputs(0), num_outputs(0), input_indices(), output_indices(), hash(0) {}

  [[nodiscard]] constexpr Expression(const OpType type,
    const uint64_t id,
    const uint32_t num_inputs,
    const std::array<std::uint32_t, MaxInput> &input_indices,
    const uint32_t num_outputs,
    const std::array<std::uint32_t, MaxOutput> &output_indices)
    : type(type), id(id), num_inputs(num_inputs), num_outputs(num_outputs), input_indices(input_indices),
      output_indices(output_indices), hash(0) {
    // do no initialize in constructor
    hash = hash_value(*this);
  }
  // NOLINTBEGIN
  constexpr friend std::size_t hash_value(const Expression &obj) {
    if (obj.hash != 0) { return obj.hash; }
    std::size_t seed = 0x071EF957;
    seed ^= (seed << 6) + (seed >> 2) + 0x4D863D4C + static_cast<std::size_t>(obj.type);
    // seed ^= (seed << 6) + (seed >> 2) + 0x75319139 + static_cast<std::size_t>(obj.id);
    seed ^= (seed << 6) + (seed >> 2) + 0x6BE220B3 + static_cast<std::size_t>(obj.num_inputs);
    seed ^= (seed << 6) + (seed >> 2) + 0x3831C880 + static_cast<std::size_t>(obj.num_outputs);
    for (auto i = 0; i < obj.num_inputs; i++) {
      seed ^= (seed << 6) + (seed >> 2) + 0x27889449 + static_cast<std::size_t>(obj.input_indices.at(i));
    }
    for (auto i = 0; i < obj.num_outputs; i++) {
      seed ^= (seed << 6) + (seed >> 2) + 0x021A6FAF + static_cast<std::size_t>(obj.output_indices.at(i));
    }
    return seed;
  }
  // NOLINTEND
};

template<size_t CMaxInput, size_t CMaxOutput>
struct CompactExpression {
  OpType type;
  uint64_t id;
  std::array<std::uint32_t, CMaxInput> input_indices;
  std::array<std::uint32_t, CMaxOutput> output_indices;
  std::size_t hash;

  constexpr CompactExpression() = default;

  template<size_t MaxInput_ = MANIFOLD_MAX_EXP_INPUT, size_t MaxOutput_ = MANIFOLD_MAX_EXP_OUTPUT>
  explicit constexpr CompactExpression(const Expression<MaxInput_, MaxOutput_> &exp)
    : type(exp.type), id(exp.id), input_indices(), output_indices(), hash(exp.hash) {
    for (size_t i = 0; i < exp.num_inputs && i < CMaxInput; i++) { input_indices.at(i) = exp.input_indices.at(i); }

    for (size_t i = 0; i < exp.num_outputs && i < CMaxOutput; i++) { output_indices.at(i) = exp.output_indices.at(i); }
  }
};

#pragma endregion


#pragma region Manifold concepts
namespace _internal {
  template<typename T>
  concept has_id = requires(T typ) {
    { typ.id } -> std::same_as<const unsigned &>;
  };

  template<typename T>
  concept has_size = requires(T typ) {
    // remember : auto != decltype(auto)
    { typ.size } -> std::same_as<const unsigned long &>;
  };

  template<typename T>
  concept is_array_like = requires(T typ) {
    { typ.storage_type } -> std::same_as<const Store &>;
    // { typ.to_tensor() } -> std::same_as<decltype(typ.to_tensor())>;
  } && has_id<T> && has_size<T> && std::is_class_v<T>;
}  // namespace _internal
#pragma endregion
#pragma region Element Wise OPs
/**
 * \brief Per element operation
 *
 * \tparam T Type of Structure to be added. Must satisfy _internal::is_array_like concept.
 * \tparam N Size of inp array
 * \param type
 * \param out output Structure to store the addition
 * \param inp Input array containing all the structures to element wise add
 * \return Expression object describing the operation
 */
template<typename T, std::size_t N>
constexpr ExpressionReflection array_elm_op(const OpType type, T &out, const std::array<T, N> &inp)
  requires(_internal::is_array_like<T>)
{


  static_assert(N <= MANIFOLD_MAX_EXP_REF_INPUT,
    "Manifold: Input count more than MANIFOLD_MAX_EXP_INPUT, please define it as per your needs");

  using arr_type = std::array<ExpressionReflection, MANIFOLD_MAX_EXP_REF_INPUT>;
  auto inputs    = std::make_unique<arr_type>();
  auto outputs   = std::make_unique<arr_type>();
  outputs->at(0) = std::move(ExpressionReflection(out));
  for (int i = 0; i < inp.size(); ++i) { inputs->at(i) = std::move(ExpressionReflection(inp.at(i))); }
  return { type, N, inputs, 1, outputs, {} };
}

template<typename T, std::size_t N>
constexpr ExpressionReflection array_elm_op(const OpType type, const T &out, const std::array<T, N> &inp)
  requires(_internal::is_array_like<const T>)
{

  static_assert(N <= MANIFOLD_MAX_EXP_REF_INPUT,
    "Manifold: Input count more than MANIFOLD_MAX_EXP_INPUT, please define it as per your needs");

  using arr_type = std::array<ExpressionReflection, MANIFOLD_MAX_EXP_REF_INPUT>;
  auto inputs    = std::make_unique<arr_type>();
  auto outputs   = std::make_unique<arr_type>();
  outputs->at(0) = std::move(ExpressionReflection(out));
  for (int i = 0; i < inp.size(); ++i) { inputs->at(i) = std::move(ExpressionReflection(inp.at(i))); }
  return { type, N, inputs, 1, outputs, {} };
}

template<typename T, std::size_t N>
constexpr ExpressionReflection array_add(T &out, const std::array<T, N> &inp)
  requires(_internal::is_array_like<T>)
{
  return array_elm_op(ARRAY_ELM_ADD, out, inp);
}

template<typename T, std::size_t N>
constexpr ExpressionReflection array_add(const T &out, const std::array<T, N> &inp)
  requires(_internal::is_array_like<const T>)
{
  return array_elm_op(ARRAY_ELM_ADD, out, inp);
}

template<typename T, std::size_t N>
constexpr ExpressionReflection array_mul(T &out, const std::array<T, N> &inp)
  requires(_internal::is_array_like<T>)
{
  return array_elm_op(ARRAY_ELM_MUL, out, inp);
}

template<typename T, std::size_t N>
constexpr ExpressionReflection array_mul(const T &out, const std::array<T, N> &inp)
  requires(_internal::is_array_like<const T>)
{
  return array_elm_op(ARRAY_ELM_MUL, out, inp);
}

template<typename T, std::size_t N>
ExpressionReflection array_sub(T &out, const std::array<T, N> &inp)
  requires(_internal::is_array_like<T>)
{
  return array_elm_op(ARRAY_ELM_SUB, out, inp);
}

template<typename T, std::size_t N>
ExpressionReflection array_sub(const T &out, const std::array<T, N> &inp)
  requires(_internal::is_array_like<const T>)
{
  return array_elm_op<T, N>(ARRAY_ELM_SUB, out, inp);
}
#pragma endregion

template<std::uint16_t DataLen, std::uint16_t ExpLen>
constexpr uint32_t extractRefs(const ExpressionReflection *exp_ref,
  std::array<TensorReflection, DataLen> &data,
  std::array<Expression<>, ExpLen> &exp_arr,
  std::uint16_t &data_len,
  std::uint16_t &exp_len) {
  if (exp_ref->isTensor()) {
    // Check if exp_ref_tensor.value().id is already in data:
    auto it = std::ranges::find_if(
      data, [id = exp_ref->tensor.value().id](const TensorReflection &ten) { return ten.id == id; });

    if (it == data.end()) {
      data.at(data_len++) = exp_ref->tensor.value();
      return data_len - 1;
    }
    // data_len++;
    return std::distance(data.begin(), it);
  }
  std::array<uint32_t, MANIFOLD_MAX_EXP_INPUT> input_arr{};
  std::array<uint32_t, MANIFOLD_MAX_EXP_OUTPUT> output_arr{};
  uint32_t const id = UINT16_MAX + 1 + exp_len;

  for (auto i = 0; i < exp_ref->num_inputs; ++i) {
    const auto &current = exp_ref->inputs->at(i);
    input_arr.at(i)     = extractRefs<DataLen, ExpLen>(&current, data, exp_arr, data_len, exp_len);
  }

  for (auto i = 0; i < exp_ref->num_outputs; ++i) {
    const auto &current = exp_ref->outputs->at(i);
    output_arr.at(i)    = extractRefs<DataLen, ExpLen>(&current, data, exp_arr, data_len, exp_len);
  }
  auto expression = Expression(exp_ref->type, id, exp_ref->num_inputs, input_arr, exp_ref->num_outputs, output_arr);
  auto it =
    std::ranges::find_if(exp_arr, [hash = expression.hash](const Expression<> &e) { return hash_value(e) == hash; });
  if (it == exp_arr.end()) {
    exp_arr.at(exp_len++) = expression;
    return exp_len;
  }
  return std::distance(exp_arr.begin(), it);
}

template<std::uint16_t DataLen = MANIFOLD_STATIC_GRAPH_MAX_MEM,
  std::uint16_t ExpLen         = MANIFOLD_STATIC_GRAPH_MAX_OP,
  size_t MaxInput              = MANIFOLD_MAX_EXP_INPUT,
  size_t MaxOutput             = MANIFOLD_MAX_EXP_OUTPUT>
struct StaticGraph {
  std::uint16_t data_len;
  std::uint16_t exp_len;
  std::array<TensorReflection, DataLen> data;
  std::array<Expression<MaxInput, MaxOutput>, ExpLen> expressions;

  [[nodiscard]] constexpr StaticGraph(const std::uint16_t data_len,
    std::array<TensorReflection, DataLen> data_,
    const std::uint16_t exp_len,
    std::array<Expression<MaxInput, MaxOutput>, ExpLen> expressions_)
    : data_len(data_len), exp_len(exp_len), data(std::move(data_)), expressions(std::move(expressions_)) {}

  template<size_t N>
  constexpr explicit StaticGraph(const std::array<ExpressionReflection *, N> &reflections)
    : data_len(0), exp_len(0), data(), expressions{} {
    for (const ExpressionReflection *exp_ref : reflections) {
      extractRefs<DataLen, ExpLen>(exp_ref, data, expressions, data_len, exp_len);
    }
  }
};

struct CompConstrains {
  size_t exp_in_size;
  size_t exp_out_size;
  size_t graph_op_size;
  size_t graph_data_size;
};

template<std::uint16_t CDataLen, std::uint16_t CExpLen, size_t CMaxInput, size_t CMaxOutput>
struct CompactStaticGraph {
  std::array<TensorReflection, CDataLen> data;
  std::array<CompactExpression<CMaxInput, CMaxOutput>, CExpLen> expressions;

  template<std::uint16_t DataLen_, std::uint16_t ExpLen_, size_t MaxInput_, size_t MaxOutput_>
  constexpr explicit CompactStaticGraph(const StaticGraph<DataLen_, ExpLen_, MaxInput_, MaxOutput_> &other)
    : data(), expressions() {
    for (size_t i = 0; i < DataLen_ && i < other.data_len; i++) { data[i] = other.data.at(i); }
    for (size_t i = 0; i < CExpLen && i < other.exp_len; i++) {
      expressions.at(i) = CompactExpression<CMaxInput, CMaxOutput>(other.expressions.at(i));
    }
  }
};

template<uint16_t DataLen, uint16_t ExpLen>
consteval auto genConstrains(const StaticGraph<DataLen, ExpLen> &inp_graph) {
  size_t max_exp_in  = 0;
  size_t max_exp_out = 0;
  for (int i = 0; i < inp_graph.exp_len; ++i) {
    auto &exp   = inp_graph.expressions.at(i);
    max_exp_in  = exp.num_inputs > max_exp_in ? exp.num_inputs : max_exp_in;
    max_exp_out = exp.num_outputs > max_exp_out ? exp.num_outputs : max_exp_out;
  }
  return CompConstrains{ max_exp_in, max_exp_out, inp_graph.exp_len, inp_graph.data_len };
}

template<CompConstrains C, std::uint16_t DataLen_, std::uint16_t ExpLen_, size_t MaxInput_, size_t MaxOutput_>
constexpr auto compact(const StaticGraph<DataLen_, ExpLen_, MaxInput_, MaxOutput_> &inp_graph) {
  return CompactStaticGraph<C.graph_data_size, C.graph_op_size, C.exp_in_size, C.exp_out_size>(inp_graph);
}
}  // namespace manifold


template<>
struct fmt::formatter<manifold::Expression<>> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  template<typename FormatContext>
  constexpr auto format(const manifold::Expression<> &exp, FormatContext &ctx) {
    format_to(ctx.out(),
      "{}:\nid: {}\nnum_inputs: {}\nnum_outputs: {}\nhash: {}\n",
      to_string(exp.type),
      exp.id,
      exp.num_inputs,
      exp.num_outputs,
      exp.hash);
    format_to(ctx.out(), "Inputs: ");
    for (int i = 0; i < exp.num_inputs; ++i) { format_to(ctx.out(), "{} ", exp.input_indices.at(i)); }
    format_to(ctx.out(), "\nOutputs: ");
    for (int i = 0; i < exp.num_outputs; ++i) { format_to(ctx.out(), "{} ", exp.output_indices.at(i)); }
    return format_to(ctx.out(), "\n");
  }
};

template<size_t MaxInput, size_t MaxOutput>
struct fmt::formatter<manifold::CompactExpression<MaxInput, MaxOutput>> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  template<typename FormatContext>
  constexpr auto format(const manifold::CompactExpression<MaxInput, MaxOutput> &exp, FormatContext &ctx) {
    format_to(ctx.out(),
      "{}:\nid: {}\nInSize: {}\nOutSize: {}\nhash: {}\n",
      to_string(exp.type),
      exp.id,
      MaxInput,
      MaxOutput,
      exp.hash);
    format_to(ctx.out(), "Inputs: ");
    for (auto index : exp.input_indices) { format_to(ctx.out(), "{} ", index); }
    format_to(ctx.out(), "\nOutputs: ");
    for (auto index : exp.output_indices) { format_to(ctx.out(), "{} ", index); }
    return format_to(ctx.out(), "\n");
  }
};

consteval auto buildGraph() noexcept {

  using namespace manifold;
  // Example usage
  constexpr auto arr1 = Array<d_type::F32, 200>(3);
  constexpr auto arr2 = Array<d_type::F32, 200>(1);
  constexpr auto arr3 = Array<d_type::F32, 200>(2);

  constexpr auto ten     = Tensor<TBase<d_type::F32, 10, 20>>(4);
  constexpr auto ten_arr = ten.to_array();

  // ArrayAdd is just an expression template class which includes input and output
  // and other relevent data
  auto add_1_2         = array_add(ten_arr, std::array{ arr1, arr2 });
  auto mul_1_2         = array_mul(arr3, std::array{ arr1, arr2, ten_arr });
  const std::array exp = { &add_1_2, &mul_1_2, &add_1_2 };
  return StaticGraph(exp);
}

int main() noexcept {
  constexpr auto res            = buildGraph();
  constexpr auto con            = genConstrains(res);
  static constexpr auto compact = manifold::compact<con>(res);

  fmt::print("Tensor IDs : \n");
  for (const auto &data : compact.data) { fmt::print("{}\n", data.id); }
  fmt::print("\nExpression hashes: \n");
  for (const auto &exp : compact.expressions) { fmt::print("{}\n", exp); }
  fmt::print("Compact Static Graph vs Non = {}:{}\n", sizeof(decltype(res)), sizeof(decltype(compact)));
  return 0;
}
