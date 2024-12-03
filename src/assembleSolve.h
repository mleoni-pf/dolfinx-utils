#pragma once

#include <petscsystypes.h>

#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/la/petsc.h>

KSPConvergedReason
        assembleSolve(const std::vector<fem::DirichletBC<PetscScalar>> bcs,
                      const fem::Form<PetscScalar>& a,
                      const fem::Form<PetscScalar>& L,
                      std::shared_ptr<la::petsc::Matrix> A,
                      std::shared_ptr<la::Vector<PetscScalar>> b,
                      la::petsc::KrylovSolver& solver,
                      std::shared_ptr<dolfinx::fem::Function<PetscScalar>> u);
