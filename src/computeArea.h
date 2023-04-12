#pragma once

#include <petscsystypes.h>

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>

PetscReal computeArea(
        const std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh,
        const dolfinx::mesh::MeshTags<int>& tags,
        const int index);
