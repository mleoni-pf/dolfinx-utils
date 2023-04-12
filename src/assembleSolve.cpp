#include "assembleSolve.h"

#include <dolfinx/common/log.h>
#include <dolfinx/fem/assembler.h>

using namespace dolfinx;

KSPConvergedReason
        assembleSolve(std::vector<std::shared_ptr<
                              const dolfinx::fem::DirichletBC<PetscScalar>>> bc,
                      std::shared_ptr<dolfinx::fem::Form<PetscScalar>> a,
                      std::shared_ptr<dolfinx::fem::Form<PetscScalar>> L,
                      std::shared_ptr<dolfinx::la::petsc::Matrix> A,
                      std::shared_ptr<dolfinx::la::Vector<PetscScalar>> b,
                      dolfinx::la::petsc::KrylovSolver& solver,
                      std::shared_ptr<dolfinx::fem::Function<PetscScalar>> u)
{
    const auto& V = u->function_space();
    LOG(INFO) << "Assembling matrix of size "
              << V->dofmap()->index_map->size_global()
                    * V->dofmap()->index_map_bs();
    MatZeroEntries(A->mat());
    if (V->dofmap()->element_dof_layout().block_size() == 1)
    {
        fem::assemble_matrix(la::petsc::Matrix::set_fn(A->mat(), ADD_VALUES),
                             *a,
                             bc);
    }
    else
    {
        fem::assemble_matrix(
                la::petsc::Matrix::set_block_fn(A->mat(), ADD_VALUES),
                *a,
                bc);
    }
    A->apply(la::petsc::Matrix::AssemblyType::FLUSH);
    fem::set_diagonal(la::petsc::Matrix::set_fn(A->mat(), INSERT_VALUES),
                      *V,
                      bc);
    A->apply(la::petsc::Matrix::AssemblyType::FINAL);
    LOG(INFO) << "Matrix assembly done";

    LOG(INFO) << "Assembling vector";
    b->set(0);
    fem::assemble_vector(b->mutable_array(), *L);
    fem::apply_lifting<PetscScalar, double>(b->mutable_array(),
                                            {a},
                                            {{bc}},
                                            {},
                                            static_cast<PetscScalar>(1));
    b->scatter_rev(std::plus<PetscScalar>());
    fem::set_bc<PetscScalar, double>(b->mutable_array(), {bc});
    LOG(INFO) << "Vector assembly done";

    solver.set_operator(A->mat());
    la::petsc::Vector _u(la::petsc::create_vector_wrap(*u->x()), false);
    la::petsc::Vector _b(la::petsc::create_vector_wrap(*b), false);
    auto nit = solver.solve(_u.vec(), _b.vec());
    LOG(INFO) << "Solver done in " << nit << " iterations";

    KSPConvergedReason r;
    KSPGetConvergedReason(solver.ksp(), &r);
    return r;
}
