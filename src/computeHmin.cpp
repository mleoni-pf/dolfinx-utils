#include "computeHmin.h"

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/mesh/utils.h>

using namespace dolfinx;

PetscReal computeHmin(const std::shared_ptr<mesh::Mesh<double>>& mesh)
{
    auto dim = mesh->topology()->dim();
    auto map = mesh->topology()->index_map(dim);
    const std::size_t num_cells = map->size_local() + map->num_ghosts();
    std::vector<std::int32_t> cells(num_cells);
    std::iota(cells.begin(), cells.end(), 0);
    auto h = mesh::h(*mesh, cells, dim);
    auto hminLoc = *std::min_element(h.cbegin(), h.cend());
    decltype(hminLoc) hminGlob = 0;
    MPI_Allreduce(&hminLoc, &hminGlob, 1, MPI_DOUBLE, MPI_MIN, mesh->comm());

    return hminGlob;
}
