#ifndef XPED_FINITE_DIFF_HPP_
#define XPED_FINITE_DIFF_HPP_

namespace Xped::internal {

template <typename F, typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry>
Xped::Tensor<Scalar, Rank, CoRank, Symmetry, false> finite_diff_gradient(const F& f, Xped::Tensor<Scalar, Rank, CoRank, Symmetry, false>& t)
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

} // namespace Xped::internal

#endif
