#ifndef XPED_FINITE_DIFF_HPP_
#define XPED_FINITE_DIFF_HPP_

#include "Xped/PEPS/iPEPS.hpp"

namespace Xped::internal {

template <typename F, typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry>
Tensor<Scalar, Rank, CoRank, Symmetry, false> finite_diff_gradient(const F& f, Tensor<Scalar, Rank, CoRank, Symmetry, false>& t)
{
    auto t_copy = t;
    auto f_plain = [&f, &t_copy](const Eigen::Vector<Scalar, Eigen::Dynamic>& xs) {
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
        if constexpr(ScalarTraits<Scalar>::IS_COMPLEX()) {
            const std::complex<double>* params_compl = reinterpret_cast<const std::complex<double>*>(xs.data());
            psi_copy.set_data(params_compl, false);
        } else {
            psi_copy.set_data(xs.data(), false);
        }
        return f(psi_copy);
    };
    auto plain_data = psi_copy.data();
    Eigen::VectorXd grad_fd_plain;
    double res_fd;
    if constexpr(ScalarTraits<Scalar>::IS_COMPLEX()) {
        double* plain_data_real = reinterpret_cast<double*>(plain_data.data());
        Eigen::VectorXd xs(Eigen::Map<Eigen::VectorXd>(plain_data_real, 2 * plain_data.size()));
        stan::math::finite_diff_gradient_auto(f_plain, xs, res_fd, grad_fd_plain);
    } else {
        Eigen::VectorXd xs(Eigen::Map<Eigen::VectorXd>(plain_data.data(), plain_data.size()));
        stan::math::finite_diff_gradient_auto(f_plain, xs, res_fd, grad_fd_plain);
    }
    auto grad_fd = Psi;
    if constexpr(ScalarTraits<Scalar>::IS_COMPLEX()) {
        std::complex<double>* grad_fd_plain_compl = reinterpret_cast<std::complex<double>*>(grad_fd_plain.data());
        grad_fd.set_data(grad_fd_plain_compl);
    } else {
        grad_fd.set_data(grad_fd_plain.data());
    }
    return grad_fd;
}

} // namespace Xped::internal

#endif
