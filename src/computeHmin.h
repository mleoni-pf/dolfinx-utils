#pragma once

#include <petscsystypes.h>

#include <dolfinx/mesh/Mesh.h>

PetscReal computeHmin(const std::shared_ptr<dolfinx::mesh::Mesh>& mesh);
