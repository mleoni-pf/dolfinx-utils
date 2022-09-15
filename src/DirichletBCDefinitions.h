#pragma once

#include <dolfinx/fem/Function.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/mesh/MeshTags.h>

using DirichletBCFunction_ret =
        std::pair<std::vector<PetscScalar>, std::vector<std::size_t>>;
using DirichletBCFunction = const std::function<DirichletBCFunction_ret(
        std::experimental::mdspan<const double,
                                  std::experimental::extents<
                                          std::size_t,
                                          3,
                                          std::experimental::dynamic_extent>>)>;

using DirichletBCList_key_type =
        std::pair<const dolfinx::mesh::MeshTags<int> &, int>;

namespace std
{
template <>
struct less<DirichletBCList_key_type>
{
    bool operator()(const DirichletBCList_key_type &lhs,
                    const DirichletBCList_key_type &rhs) const
    {
        if (lhs.first.dim() == rhs.first.dim())
            return lhs.second < rhs.second;

        return lhs.first.dim() < rhs.first.dim();
    }
};
}

#include "TimeDependentFunction.h"

using DirichletBCList =
        std::vector<std::pair<DirichletBCList_key_type,
                              std::shared_ptr<TimeDependentFunction>>>;
