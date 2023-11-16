//
// Created by sid on 21/10/23.
//

#pragma once

#include "Scions/common/common.h"
#include "Scions/core/op/op.h"
#include "Scions/core/mem/mem_desc.h"

namespace scions::graph {
template <size_t Ops, size_t Mem>
struct SequentialGraph {
    std::array<op::OpDesc, Ops>  ops;
    mem::MemDescriptor<Mem> memDescriptor;
};
}