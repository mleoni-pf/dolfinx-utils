#pragma once

#include <dolfinx/fem/Function.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/mesh/MeshTags.h>

using DirichletBCFunction_ret = xt::xarray<PetscScalar>;
using DirichletBCFunction = const std::function<DirichletBCFunction_ret(
        const xt::xtensor<double, 2> &)>;

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
