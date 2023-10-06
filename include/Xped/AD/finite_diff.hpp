#ifndef XPED_FINITE_DIFF_HPP_
#define XPED_FINITE_DIFF_HPP_

#include "Xped/PEPS/iPEPS.hpp"

namespace Xped::internal {

template <typename F, typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry>
Tensor<Scalar, Rank, CoRank, Symmetry, false> finite_diff_gradient(const F& f, Tensor<Scalar, Rank, CoRank, Symmetry, false>& t)
{
    auto t_copy = t;
    auto f_plain = [&f, &t_copy](const Eigen::VectorXd& xs) {
        // std::cout << "xs: " << xs.transpose() << std::endl;
        t_copy.set_data(xs.data(), xs.size());
        return f(t_copy);
    };
    auto plain_data = t_copy.data();
    Eigen::VectorXd xs(Eigen::Map<Eigen::VectorXd>(plain_data, t.plainSize()));
    Eigen::VectorXd grad_fd_plain;
    double res_fd;
    stan::math::finite_diff_gradient_auto(f_plain, xs, res_fd, grad_fd_plain);
    auto grad_fd = t;
    grad_fd.set_data(grad_fd_plain.data(), grad_fd_plain.size());
    return grad_fd;
}

template <typename F, typename Scalar, typename Symmetry, bool ALL_OUT_LEGS>
auto finite_diff_gradient(const F& f, iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, false>& Psi)
{
    auto psi_copy = Psi;
    auto f_plain = [&f, &psi_copy](const Eigen::VectorXd& xs) {
        // std::cout << "xs: " << xs.transpose() << std::endl;
        psi_copy.set_data(xs.data(), false);
        return f(psi_copy);
    };
    auto plain_data = psi_copy.data();
    Eigen::VectorXd xs(Eigen::Map<Eigen::VectorXd>(plain_data.data(), plain_data.size()));
    Eigen::VectorXd grad_fd_plain;
    double res_fd;
    stan::math::finite_diff_gradient_auto(f_plain, xs, res_fd, grad_fd_plain);
    auto grad_fd = Psi;
    grad_fd.set_data(grad_fd_plain.data());
    return grad_fd;
}

} // namespace Xped::internal

#endif
