#pragma once
#include "concepts.hpp"
#include "constants.hpp"
#include "manifold/utility.hpp"
#include "shape.hpp"
#include <cstdint>

namespace manifold {
// Reflection
struct TensorReflection {
    DType data_type;
    std::size_t size;
    uint32_t id;
    ShapeReflection shape;
    layout storage_layout;
    Store storage_type;

    constexpr TensorReflection() = default;

    [[nodiscard]] constexpr TensorReflection(const DType _data_type,
            const std::size_t _size,
            const uint32_t _id,
            const ShapeReflection &shape_reflection,
            const layout _storage_layout,
            const Store _storage_type)
        : data_type(_data_type), size(_size), id(_id), shape(shape_reflection), storage_layout(_storage_layout),
          storage_type(_storage_type) {}
};
#pragma endregion

template<DType Type, std::uint32_t... Shapes>
struct TBase {
    static constexpr manifold::DType data_type = Type;
    static constexpr std::size_t size          = (Shapes * ...);
    static constexpr Shape<Shapes...> shape{};
};

template<typename TensorBase, Store Storage = Store::HOST, layout Layout = layout::ROW_MAJOR>
requires _internal::IsTBase<TensorBase>
struct Tensor : TensorBase {
    static constexpr layout storage_layout = Layout;
    static constexpr Store storage_type    = Storage;
    uint32_t id;

    constexpr Tensor() : id(UINT32_MAX) {}

    [[nodiscard]] constexpr explicit Tensor(const uint32_t id_) : id(id_) {}


    // Reshape method
    template<std::size_t... NewShapes>
    requires(TensorBase::size == TBase<TensorBase::data_type, NewShapes...>::size)
    constexpr auto reshape() const {
        return Tensor<TBase<TensorBase::data_type, NewShapes...>, Storage, Layout>(id);
    }
    constexpr auto reflect() const {
        return TensorReflection{
            TensorBase::data_type, TensorBase::size, id, TensorBase::shape.reflect(), storage_layout, storage_type
        };
    }
};


template<DType Type, Store Storage = Store::HOST, layout Layout = layout::ROW_MAJOR>
constexpr Tensor<TBase<Type, 1>, Storage, Layout> Scalar(uint32_t _id) {
    return Tensor<TBase<Type, 1>, Storage, Layout>(_id);
}

}  // namespace manifold


template<>
struct std::formatter<manifold::TensorReflection> : std::formatter<std::string> {
    template<typename FormatContext>
    constexpr auto format(const manifold::TensorReflection &ten, FormatContext &ctx) const {
        std::format_to(ctx.out(),
                       "id : {}\ntype : {}\nsize : {}\nstorage : {}\nlayout : {}",
                       ten.id,
                       dtypeToString(ten.data_type),
                       ten.size,
                       storeToString(ten.storage_type),
                       layoutToString(ten.storage_layout));
        return std::format_to(ctx.out(), "\n{}", ten.shape);
    }
};
