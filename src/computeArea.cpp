#include "computeArea.h"

#include <dolfinx/fem/assembler.h>

#include "area.h"

using namespace dolfinx;

PetscReal computeArea(const dolfinx::mesh::MeshTags<int>& tags, const int index)
{
    std::vector<int> values(tags.values().size());

    std::transform(tags.values().cbegin(),
                   tags.values().cend(),
                   values.begin(),
                   [=](const auto& val) { return val == index ? 0 : 1; });
    auto integrationTags =
            mesh::MeshTags<int>(tags.mesh(), 2, tags.indices(), values);

    auto a = std::make_shared<fem::Constant<PetscScalar>>(1);
    auto M_area = std::make_shared<fem::Form<PetscScalar>>(
            fem::create_form<PetscReal>(
                    *form_area_area,
                    {},
                    {},
                    {{"a", a}},
                    {{fem::IntegralType::exterior_facet, &integrationTags}},
                    integrationTags.mesh()));

    auto area_loc = fem::assemble_scalar(*M_area);
    decltype(area_loc) area = 0;
    MPI_Allreduce(&area_loc,
                  &area,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  integrationTags.mesh()->comm());
    return area;
}
