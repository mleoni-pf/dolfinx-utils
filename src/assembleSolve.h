#pragma once

#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/la/petsc.h>

KSPConvergedReason
        assembleSolve(std::vector<std::shared_ptr<
                              const dolfinx::fem::DirichletBC<PetscScalar>>> bc,
                      std::shared_ptr<dolfinx::fem::Form<PetscScalar>> a,
                      std::shared_ptr<dolfinx::fem::Form<PetscScalar>> L,
                      std::shared_ptr<dolfinx::la::petsc::Matrix> A,
                      std::shared_ptr<dolfinx::la::Vector<PetscScalar>> b,
                      dolfinx::la::petsc::KrylovSolver& solver,
                      std::shared_ptr<dolfinx::fem::Function<PetscScalar>> u);
