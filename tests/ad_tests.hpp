#ifndef XPED_AT_TESTS_HPP_
#define XPED_AT_TESTS_HPP_

#include "doctest/doctest.h"

#include "Xped/AD/finite_diff.hpp"

namespace Xped::internal {

template <typename Scalar, typename Symmetry, typename F>
void check_ad_vs_fd(const F& f, std::size_t basis_size, Xped::mpi::XpedWorld& world)
{
    Xped::Qbasis<Symmetry> B1, B2;
    if(world.rank == 0) {
        B1.setRandom(basis_size);
        B2.setRandom(basis_size);
    }
    Xped::mpi::broadcast(B1, world.rank, 0, world);
    Xped::mpi::broadcast(B2, world.rank, 0, world);
    // Xped::mpi::broadcast(B2, world.rank, 0, world);
    // std::filesystem::path p("/home/user/__LIBS__/j1j2_ipeps_states/single-site_pg-C4v-A1/j20.0/state_1s_A1_j20.0_D2_chi_opt48.json");
    // auto tmp = Xped::IO::loadU0JsonTensor<Symmetry>(p);
    // Xped::Tensor<double, 4, 1, Symmetry, true> t = tmp;
    Xped::Tensor<double, 2, 2, Symmetry, true> t({{B1, B2}}, {{B1, B2}}, world);
    t.setRandom();
    // t.print(std::cout, true);
    // std::cout << std::endl;
    stan::math::nested_rev_autodiff nested;
    auto res = f(t);
    stan::math::grad(res.vi_);
    auto grad_ad = t.adj();
    auto grad_fd = finite_diff_gradient(f, t.val_op());

    // grad_ad.print(std::cout, true);
    // std::cout << std::endl;
    // grad_fd.print(std::cout, true);
    // std::cout << std::endl;
    CHECK((grad_ad - grad_fd).norm() == doctest::Approx(0.));
}

} // namespace Xped::internal

#endif
