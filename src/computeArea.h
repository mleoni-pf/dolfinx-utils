#pragma once

#include <petscsystypes.h>

#include <dolfinx/mesh/MeshTags.h>

PetscReal computeArea(const dolfinx::mesh::MeshTags<int>& tags,
                      const int index);
