#include "computeArea.h"

#include <dolfinx/fem/assembler.h>

#include "area.h"

using namespace dolfinx;

PetscReal computeArea(const std::shared_ptr<const mesh::Mesh<double>> mesh,
                      const mesh::MeshTags<int>& tags,
                      const int index)
{
    std::vector<int> values(tags.values().size());

    std::transform(tags.values().begin(),
                   tags.values().end(),
                   values.begin(),
                   [=](const auto& val) { return val == index ? 0 : 1; });
    auto integrationTags = mesh::MeshTags<int>(
            tags.topology(),
            2,
            std::vector<std::int32_t>(tags.indices().begin(),
                                      tags.indices().end()),
            values);

    auto a = std::make_shared<fem::Constant<PetscScalar>>(1);

    std::vector<std::pair<int, std::vector<std::int32_t>>>
            exteriorIntegrationDomains = {
                    {0,
                     fem::compute_integration_domains(
                             fem::IntegralType::exterior_facet,
                             *tags.topology(),
                             tags.find(0))}};
    std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>>
            exteriorIntegrationDomainsSpans;
    for (const auto& [i, v] : exteriorIntegrationDomains)
    {
        exteriorIntegrationDomainsSpans.push_back(
                std::make_pair(i, std::span(v)));
    }
    auto M_area = std::make_shared<fem::Form<PetscScalar>>(
            fem::create_form<PetscReal>(*form_area_area,
                                        {},
                                        {},
                                        {{"a", a}},
                                        {{fem::IntegralType::exterior_facet,
                                          exteriorIntegrationDomainsSpans}},
                                        {},
                                        mesh));

    auto area_loc = fem::assemble_scalar(*M_area);
    decltype(area_loc) area = 0;
    MPI_Allreduce(&area_loc, &area, 1, MPI_DOUBLE, MPI_SUM, mesh->comm());
    return area;
}
