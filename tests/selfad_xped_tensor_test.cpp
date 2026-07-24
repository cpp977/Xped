// End-to-end check of the self-contained AD engine against *real* Xped code:
// block-sparse U(1) tensors with AD tracking, contraction / permutation /
// adjoint / norm / trace / SVD backward passes, gradients validated with
// finite differences.  This is the same computational pattern as
// docs/snippets/ad.cpp, extended by a truncated-SVD pipeline.
//
// Build (see tests/README_selfad or CI): requires only Xped's ordinary
// header-only dependencies -- NO stan-math, NO TBB, NO SUNDIALS.

#include <cstdio>
#include <random>

// canonical Xped include order (see examples/ipeps_ad.cpp)
#include "stan/math/rev.hpp"

#include "Xped/Util/Macros.hpp"
#include "Xped/Util/Mpi.hpp"

#include "Xped/Interfaces/PlainInterface.hpp"

#include "Xped/Core/Qbasis.hpp"
#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U1.hpp"
#include "Xped/Symmetry/kind_dummies.hpp"

#include "Xped/Core/AdjointOp.hpp"
#include "Xped/Core/Tensor.hpp"

#include "Xped/AD/ADTensor.hpp"

static int g_failures = 0;
#define CHECK(cond)                                                                                                                                  \
    do {                                                                                                                                             \
        if(!(cond)) {                                                                                                                                \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                                                              \
            ++g_failures;                                                                                                                            \
        }                                                                                                                                            \
    } while(0)

// finite-difference gradient of a scalar function of a plain tensor; this is
// the same construction as Xped::internal::finite_diff_gradient (AD/finite_diff.hpp)
template <typename F, typename PlainT>
static PlainT fd_gradient(const F& f, const PlainT& t)
{
    auto t_copy = t;
    auto f_plain = [&f, &t_copy](const Eigen::Vector<double, Eigen::Dynamic>& xs) {
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

// f(T) = ||(7 T) (3 T^dagger)|| + tr((T^dagger T)) -- scalar function of the tensor
template <typename TensorT>
static auto scalar_pipeline(TensorT& T)
{
    auto A = (7. * T) * (3. * T.adjoint());
    auto B = T.adjoint() * T;
    return A.norm() + B.trace();
}

template <typename Symmetry>
static void run_for(const char* name)
{
    using PlainT = Xped::Tensor<double, 2, 1, Symmetry, false>;
    using ADT = Xped::Tensor<double, 2, 1, Symmetry, true>;
    std::printf("---- %s ----\n", name);

    Xped::Qbasis<Symmetry, 1> B1, B2;
    B1.setRandom(6);
    B2.setRandom(6);
    auto C = B1.combine(B2).forgetHistory();

    // ---------------- AD gradient ----------------
    PlainT T0({{B1, B2}}, {{C}});
    std::mt19937 engine(0);
    T0.setRandom(engine);

    double e_ad = 0;
    PlainT grad_ad;
    {
        stan::math::nested_rev_autodiff nested;
        ADT T(T0);
        auto res = scalar_pipeline(T);
        e_ad = res.val();
        stan::math::grad(res.vi_);
        grad_ad = T.adj();
    }

    // ---------------- finite differences ----------------
    auto f_plain = [](PlainT& t) {
        auto v = scalar_pipeline(t);
        return v;
    };
    auto grad_fd = fd_gradient(f_plain, T0);

    auto e_plain = scalar_pipeline(T0);
    CHECK(std::abs(e_ad - e_plain) < 1e-12 * (1 + std::abs(e_plain)));

    double diff = (grad_ad - grad_fd).norm();
    double scale = grad_fd.norm();
    std::printf("f = %.12f, |grad_ad - grad_fd| / |grad_fd| = %.3e\n", e_ad, diff / scale);
    CHECK(diff / scale < 1e-7);

    // ---------------- SVD backward ----------------
    auto f_svd = [](auto& t) {
        double truncWeight = 0;
        auto [U, S, Vdag] = t.template tSVD<true>(4ul, 1.e-14, truncWeight, false);
        return S.template trace<true>();
    };
    auto f_svd_plain = [](PlainT& t) {
        double truncWeight = 0;
        auto [U, S, Vdag] = t.tSVD(4ul, 1.e-14, truncWeight, false);
        return S.trace();
    };

    double s_ad = 0;
    PlainT grad_svd_ad;
    {
        stan::math::nested_rev_autodiff nested;
        ADT T(T0);
        auto res = f_svd(T);
        s_ad = res.val();
        stan::math::grad(res.vi_);
        grad_svd_ad = T.adj();
    }
    auto grad_svd_fd = fd_gradient(f_svd_plain, T0);
    double sdiff = (grad_svd_ad - grad_svd_fd).norm();
    double sscale = grad_svd_fd.norm();
    std::printf("truncated SVD: sum(S) = %.12f, |grad_ad - grad_fd| / |grad_fd| = %.3e\n", s_ad, sdiff / sscale);
    CHECK(sdiff / sscale < 1e-6);
}

int main()
{
    run_for<Xped::Sym::U1<Xped::Sym::SpinU1>>("U(1) symmetry");
    run_for<Xped::Sym::SU2<Xped::Sym::SpinSU2>>("SU(2) symmetry (self-contained Wigner coefficients)");

    if(g_failures == 0) {
        std::printf("\nXPED TENSOR AD TEST PASSED (self-contained engine)\n");
        return 0;
    }
    std::printf("\n%d FAILURES\n", g_failures);
    return 1;
}
