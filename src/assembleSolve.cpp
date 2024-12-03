#include "assembleSolve.h"

#include <dolfinx/common/log.h>
#include <dolfinx/fem/assembler.h>

using namespace dolfinx;

KSPConvergedReason
        assembleSolve(const std::vector<fem::DirichletBC<PetscScalar>> bcs,
                      const fem::Form<PetscScalar>& a,
                      const fem::Form<PetscScalar>& L,
                      std::shared_ptr<la::petsc::Matrix> A,
                      std::shared_ptr<la::Vector<PetscScalar>> b,
                      la::petsc::KrylovSolver& solver,
                      std::shared_ptr<fem::Function<PetscScalar>> u)
{
    std::vector<std::reference_wrapper<const fem::DirichletBC<PetscScalar>>>
            bcs_rw;
    std::transform(std::cbegin(bcs),
                   std::cend(bcs),
                   std::back_inserter(bcs_rw),
                   [](const auto& bc) { return std::cref(bc); });

    const auto& V = u->function_space();
    spdlog::info("Assembling matrix of size {}",
                 V->dofmap()->index_map->size_global()
                         * V->dofmap()->index_map_bs());
    MatZeroEntries(A->mat());
    if (V->dofmap()->element_dof_layout().block_size() == 1)
    {
        fem::assemble_matrix(la::petsc::Matrix::set_fn(A->mat(), ADD_VALUES),
                             a,
                             bcs_rw);
    }
    else
    {
        fem::assemble_matrix(
                la::petsc::Matrix::set_block_fn(A->mat(), ADD_VALUES),
                a,
                bcs_rw);
    }
    A->apply(la::petsc::Matrix::AssemblyType::FLUSH);
    fem::set_diagonal(la::petsc::Matrix::set_fn(A->mat(), INSERT_VALUES),
                      *V,
                      bcs_rw);
    A->apply(la::petsc::Matrix::AssemblyType::FINAL);
    spdlog::info("Matrix assembly done");

    spdlog::info("Assembling vector");
    b->set(0);
    fem::assemble_vector(b->mutable_array(), L);
    fem::apply_lifting<PetscScalar, double>(b->mutable_array(),
                                            {a},
                                            {bcs_rw},
                                            {},
                                            static_cast<PetscScalar>(1));
    b->scatter_rev(std::plus<PetscScalar>());
    for (const auto& bc : bcs_rw)
    {
        bc.get().set(b->mutable_array(), std::nullopt);
    }
    spdlog::info("Vector assembly done");

    solver.set_operator(A->mat());
    la::petsc::Vector _u(la::petsc::create_vector_wrap(*u->x()), false);
    la::petsc::Vector _b(la::petsc::create_vector_wrap(*b), false);
    auto nit = solver.solve(_u.vec(), _b.vec());
    spdlog::info("Solver done in {} iterations", nit);

    u->x()->scatter_fwd();

    KSPConvergedReason r;
    KSPGetConvergedReason(solver.ksp(), &r);
    return r;
}
