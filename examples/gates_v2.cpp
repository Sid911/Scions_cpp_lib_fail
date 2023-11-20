//
// Created by sid on 21/11/23.
//

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <array>
#include <optional>
#include <queue>
#include <type_traits>
#include <fmt/core.h>

enum : uint8_t {
  MANIFOLD_MAX_EXP_INPUT = 5,
  MANIFOLD_MAX_EXP_OUTPUT = 2,
  MANIFOLD_MAX_RANK = 5,
};

namespace manifold {
// Enumeration for data types
enum class d_type: std::uint8_t { F32, F64 };

// Enumeration for storage types
enum class Store: std::uint8_t { HOST, DEVICE };

// Enumeration for layout types
enum class layout: std::uint8_t { ROW_MAJOR, COL_MAJOR };

// Helper structure for shapes of a tensor
template<std::uint32_t... Shapes>
struct Shape {
  static constexpr std::size_t rank                      = sizeof...(Shapes);
  static constexpr std::array<std::uint32_t, rank> shape = { Shapes... };
};

// Definition for array
template<d_type Type, std::size_t Size, Store Storage = Store::HOST>
struct Array {
  static constexpr d_type data_type   = Type;
  static constexpr std::size_t size   = Size;
  static constexpr Store storage_type = Storage;
  uint32_t id;

  [[nodiscard]] constexpr explicit Array(const uint32_t id_)
    : id(id_) {}
};

// Definition for matrix
template<d_type Type, std::size_t Rows, std::size_t Cols, layout Layout = layout::ROW_MAJOR, Store
  Storage = Store::HOST>
struct Matrix {
  static constexpr d_type data_type = Type;
  static constexpr std::size_t size = Rows * Cols;
  static constexpr Shape<Rows, Cols> shape{};
  static constexpr layout storage_layout = Layout;
  static constexpr Store storage_type    = Storage;
  uint32_t id;


  [[nodiscard]] constexpr explicit Matrix(const uint32_t id_)
    : id(id_) {}

  [[nodiscard]] constexpr auto to_array() const { return Array<Type, size, Storage>(id); }
};

template<d_type Type, std::uint32_t ... Shapes>
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

  [[nodiscard]] constexpr explicit Tensor(const uint32_t id_)
    : id(id_) {}

  [[nodiscard]] constexpr auto to_array() const { return Array<TensorBase::data_type, TensorBase::size, Storage>(id); }

  template<std::size_t M, std::size_t N>
  [[nodiscard]] constexpr auto to_matrix() const {
    static_assert(TensorBase::size == M * N, "total Matrix size should be equal total tensor size");
    return Matrix<TensorBase::data_type, M, N, Layout, Storage>(id);
  }
};


// Reflections
struct ShapeReflection {
  std::size_t rank;
  std::array<std::uint32_t, MANIFOLD_MAX_RANK> shape;

  [[nodiscard]] constexpr ShapeReflection(const std::uint32_t rank,
    const std::array<std::uint32_t, MANIFOLD_MAX_RANK> &shape)
    : rank(rank), shape(shape) {}

  [[nodiscard]] constexpr ShapeReflection(const std::uint32_t size)
    : rank(1), shape({ size }) {}

  [[nodiscard]] constexpr ShapeReflection(const std::uint32_t M, const std::uint32_t N)
    : rank(2), shape({ M, N }) {}
};

struct TensorReflection {
  d_type data_type;
  std::size_t size;
  uint32_t id;
  ShapeReflection shape;
  layout storage_layout;
  Store storage_type;

  [[nodiscard]] constexpr TensorReflection(const d_type data_type,
    const std::size_t size,
    const uint32_t id,
    const ShapeReflection &shape_reflection,
    const layout storage_layout,
    const Store storage_type)
    : data_type(data_type),
      size(size),
      id(id),
      shape(shape_reflection),
      storage_layout(storage_layout),
      storage_type(storage_type) {}


  template<d_type Type, std::size_t Size, Store Storage = Store::HOST>
  constexpr TensorReflection(const Array<Type, Size, Storage> &arr)
    : data_type(Type),
      size(Size),
      id(arr.id),
      shape(Size),
      storage_layout(layout::ROW_MAJOR),
      storage_type(Storage) {}

  template<d_type Type, std::size_t Rows, std::size_t Cols, layout Layout = layout::ROW_MAJOR, Store
    Storage = Store::HOST>
  constexpr TensorReflection(const Matrix<Type, Rows, Cols, Layout, Storage> &mat)
    : data_type(Type),
      size(Rows * Cols),
      id(mat.id),
      shape(Rows, Cols),
      storage_layout(layout::ROW_MAJOR),
      storage_type(Storage) {}
};

enum OpType: uint8_t {
  // Array operations

  //Element wise add two arrays
  ARRAY_ELM_ADD,
  ARRAY_ELM_SUB,
  ARRAY_ELM_MUL,
  ARRAY_ELM_DIV,
  //AXPY is vector scalar product
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

struct Expression {
  //Expression part

  // OpType of the expression
  OpType type;
  uint32_t num_inputs;
  uint32_t num_outputs;
  std::shared_ptr<std::array<Expression, MANIFOLD_MAX_EXP_INPUT>> inputs;
  std::shared_ptr<std::array<Expression, MANIFOLD_MAX_EXP_INPUT>> outputs;

  //TensorReflection part
  std::optional<TensorReflection> tensor;

  [[nodiscard]] Expression(const OpType type_,
    const uint32_t num_inputs_,
    const std::shared_ptr<std::array<Expression, MANIFOLD_MAX_EXP_INPUT>> &inputs_,
    const uint32_t num_outputs_,
    const std::shared_ptr<std::array<Expression, MANIFOLD_MAX_EXP_INPUT>> &outputs_,
    const std::optional<TensorReflection> &tensor_)
    : type(type_),
      num_inputs(num_inputs_),
      num_outputs(num_outputs_),
      inputs(inputs_),
      outputs(outputs_),
      tensor(tensor_) {}

  template<d_type Type, std::size_t Size, Store Storage = Store::HOST>
  Expression(Array<Type, Size, Storage> &array)
    : Expression(TensorReflection(array)) {}

  template<d_type Type, std::size_t Rows, std::size_t Cols,
    layout Layout = layout::ROW_MAJOR, Store Storage = Store::HOST>
  Expression(const Matrix<Type, Rows, Cols, Layout, Storage> &mat)
    : Expression(TensorReflection(mat)) {}

  [[nodiscard]] constexpr explicit Expression(const TensorReflection &tensor)
    : type(), num_inputs(0), num_outputs(0), tensor(tensor) {}

  [[nodiscard]] constexpr bool isTensor() const { return tensor.has_value(); }

};


namespace _internal {
  template<typename T>
  concept has_id = requires(T typ) { { typ.id } -> std::same_as<const unsigned &>; };

  template<typename T>
  concept has_size = requires(T typ) { { typ.size } -> std::same_as<const unsigned long &>; };

  template<typename T>
  concept is_array_like = requires(T typ)
  {
    { typ.storage_type } -> std::same_as<const Store &>;
    // { typ.to_tensor() } -> std::same_as<decltype(typ.to_tensor())>;
  } && has_id<T> && has_size<T> && std::is_class_v<T>;
}

/**
 * \brief Adds each element from
 *
 * \tparam T Type of Structure to be added. Must satisfy _internal::is_array_like concept.
 * \tparam N Size of inp array
 * \param out output Structure to store the addition
 * \param inp Input array containing all the structures to element wise add
 * \return Expression object describing the operation
 */
template<typename T, std::size_t N>
Expression array_add(T &out, const std::array<T, N> &inp)
  requires (_internal::is_array_like<T>) {

  static_assert(
    N <= MANIFOLD_MAX_EXP_INPUT,
    "Manifold: Input count more than MANIFOLD_MAX_EXP_INPUT, please define it as per your needs");
  using arr_type = std::array<T, MANIFOLD_MAX_EXP_INPUT>;
  std::shared_ptr<arr_type> inputs = std::make_shared<arr_type>({});
  for (int i = 0; i < inp.size(); ++i) { inputs[i] = Expression(inp[i]); }
  const auto exp = Expression(ARRAY_ELM_ADD, N, inputs, 1, Expression(out), {});
  return exp;
}

template<typename T, std::size_t N>
Expression array_add(const T &out, const std::array<T, N> &inp)
  requires (_internal::is_array_like<const T>) {

  static_assert(
    N <= MANIFOLD_MAX_EXP_INPUT,
    "Manifold: Input count more than MANIFOLD_MAX_EXP_INPUT, please define it as per your needs");

  const auto exp = Expression(ARRAY_ELM_ADD, N, {}, 1, {}, {});
  return exp;
}

} // namespace manifold

struct Node {
  int data;
  std::unique_ptr<Node> left  = nullptr;
  std::unique_ptr<Node> right = nullptr;
};

std::unique_ptr<Node> newNode(int data) {
  auto node  = std::make_unique<Node>();
  node->data = data;
  return node;
}

void generateTree(std::unique_ptr<Node> &root, std::size_t depth) {
  std::queue<Node *> nodes;
  nodes.push(root.get());

  int level = 0;
  while (!nodes.empty() && level < depth) {
    // calculate the number of nodes at the current level
    int numNodes = nodes.size();

    // process all nodes at the current level
    for (int i = 0; i < numNodes; ++i) {
      Node *current_node = nodes.front();
      nodes.pop();

      current_node->left  = newNode(current_node->data - 1);
      current_node->right = newNode(current_node->data - 1);

      nodes.push(current_node->left.get());
      nodes.push(current_node->right.get());
    }

    // move to the next level
    level++;
  }
}

consteval auto buildGraph() {
  std::array<std::optional<int>,5> arr{2,3,4,5,0};
  return arr;
}

int main() {struct i{int64_t ayo; uint16_t n; uint8_t there;};
  static auto res = buildGraph();
  fmt::print("Optional Array index 2 : {}, size of Optional Array {}\n",res[2].value(), sizeof(std::optional<i>));
  using namespace manifold;
  // Example usage
  static constexpr auto arr1 = Matrix<d_type::F32, 20, 10>(0);
  static constexpr auto arr2 = Matrix<d_type::F32, 20, 10>(1);
  static constexpr auto arr3 = Matrix<d_type::F32, 20, 10>(2);

  // ArrayAdd is just an expression template class which includes input and output
  // and other relevent data
  constexpr std::array inp{ arr1.to_array(), arr2.to_array() };
  auto add_1_2_in3 = array_add(arr3.to_array(), inp);
  // size of entire program till now
  fmt::print("Size of Expression {}", sizeof(Expression) + sizeof(decltype(arr1)) * 6);

  constexpr auto mat     = Matrix<d_type::F32, 1024, 1024>(3);
  constexpr auto ten     = Tensor<TBase<d_type::F32, 5, 6, 7, 7, 8>>(4);
  constexpr auto ten_mat = ten.to_matrix<210, 56>();

  return 0;
}
