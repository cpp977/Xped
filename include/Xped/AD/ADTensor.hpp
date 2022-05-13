#ifndef VAR_VALUE_HPP_
#define VAR_VALUE_HPP_

#include "Xped/Core/Tensor.hpp"

#include "Xped/AD/reverse_pass_callback_alloc.hpp"
#include "Xped/AD/vari_value.hpp"

namespace Xped {

template <typename Scalar_, std::size_t Rank, std::size_t CoRank, typename Symmetry_, typename AllocationPolicy_>
class Tensor<Scalar_, Rank, CoRank, Symmetry_, true, AllocationPolicy_>
{
public:
    using Scalar = Scalar_;
    using RealScalar = typename ScalarTraits<Scalar>::Real;

    using Symmetry = Symmetry_;
    using qType = typename Symmetry::qType;

    using AllocationPolicy = AllocationPolicy_;

    using IndexType = PlainInterface::Indextype;
    using VectorType = PlainInterface::VType<Scalar>;
    using MatrixType = PlainInterface::MType<Scalar>;
    using MatrixMapType = PlainInterface::MapMType<Scalar>;
    using MatrixcMapType = PlainInterface::cMapMType<Scalar>;
    using TensorType = PlainInterface::TType<Scalar, Rank + CoRank>;
    using TensorMapType = PlainInterface::MapTType<Scalar, Rank + CoRank>;
    using TensorcMapType = PlainInterface::cMapTType<Scalar, Rank + CoRank>;
    using value_type = Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>; // type in vari_value -->ArenaTensor or Tensor.
    using vari_type = stan::math::vari_value<value_type>;

    vari_type* vi_;
    bool grad_tracking = true;

    inline void nograd() { grad_tracking = false; }
    inline void grad() { grad_tracking = true; }

    inline bool is_uninitialized() noexcept { return (vi_ == nullptr); }

    Tensor()
        : vi_(nullptr)
    {}

    Tensor(const Xped::Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>& x)
        : vi_(new vari_type(x, false))
    {}

    Tensor(vari_type* vi)
        : vi_(vi)
    {}

    Tensor(const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, Rank>& basis_domain,
           const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, CoRank>& basis_codomain,
           mpi::XpedWorld& world = mpi::getUniverse())
        : vi_(new vari_type(basis_domain, basis_codomain, world))
    {}

    void setRandom() { val_op().setRandom(); }
    void setIdentity() { val_op().setIdentity(); }
    void setZero() { val_op().setZero(); }
    void setConstant(Scalar c) { val_op().setConstant(c); }

    inline const auto& val() const noexcept { return vi_->val(); }
    inline auto& val_op() noexcept { return vi_->val_op(); }

    inline auto& adj() noexcept { return vi_->adj(); }
    inline auto& adj() const noexcept { return vi_->adj(); }
    inline auto& adj_op() noexcept { return vi_->adj(); }

    constexpr std::size_t rank() const noexcept { return vi_->rank(); }
    constexpr std::size_t corank() const noexcept { return vi_->corank(); }
    const std::string name() const { return val().name(); }

    inline const auto& sector() const { return val().sector(); }
    inline const qType sector(std::size_t i) const { return val().sector(i); }

    constexpr bool CONTIGUOUS_STORAGE() const { return val().CONTIGUOUS_STORAGE(); }

    std::size_t plainSize() const { return val().plainSize(); }

    inline const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, Rank>& uncoupledDomain() const { return val().uncoupledDomain(); }
    inline const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, CoRank>& uncoupledCodomain() const { return val().uncoupledCodomain(); }

    inline const Qbasis<Symmetry, Rank, AllocationPolicy>& coupledDomain() const { return val().coupledDomain(); }
    inline const Qbasis<Symmetry, CoRank, AllocationPolicy>& coupledCodomain() const { return val().coupledCodomain(); }

    inline auto begin() const { return val_op().begin(); }
    inline auto end() const { return val_op().end(); }

    inline const auto cbegin() const { return val().cbegin(); }
    inline const auto cend() const { return val().cend(); }

    inline auto gradbegin() const { return adj_op().begin(); }
    inline auto gradend() const { return adj_op().end(); }

    inline const auto cgradbegin() const { return adj().cbegin(); }
    inline const auto cgradend() const { return adj().cend(); }

    inline void set_data(const Scalar* data, std::size_t size) { val_op().set_data(data, size); }

    inline vari_type& operator*() { return *vi_; }

    inline vari_type* operator->() { return vi_; }

    inline Tensor<Scalar, Rank, CoRank, Symmetry, true, AllocationPolicy> eval() const { return *this; }

    auto adjoint() const
    {
        Xped::Tensor<Scalar, CoRank, Rank, Symmetry, true, AllocationPolicy> out(val().adjoint().eval());
        if(grad_tracking) {
            stan::math::reverse_pass_callback([curr = *this, out]() mutable {
                curr.adj() += out.adj().adjoint().eval();
                SPDLOG_WARN("reverse adjoint of {}, input adj norm={}, output adj norm={}", curr.name(), out.adj().norm(), curr.adj().norm());
            });
        }
        return out;
    }

    template <int shift, std::size_t... p>
    auto permute() const
    {
        Xped::Tensor<Scalar, Rank - shift, CoRank + shift, Symmetry, true, AllocationPolicy> out(val().template permute<shift, p...>());
        if(grad_tracking) {
            stan::math::reverse_pass_callback([curr = *this, out]() mutable {
                using inverse = decltype(Xped::util::constFct::inverse_permutation<seq::iseq<std::size_t, p...>>());
                curr.adj() += out.adj().template permute<-shift>(inverse{});
                SPDLOG_WARN("reverse permute of {}, input adj norm={}, output adj norm={}", curr.name(), out.adj().norm(), curr.adj().norm());
            });
        }
        return out;
    }

    template <int shift, std::size_t... p>
    Tensor<Scalar, Rank - shift, CoRank + shift, Symmetry, true, AllocationPolicy> permute(seq::iseq<std::size_t, p...>) const
    {
        return permute<shift, p...>();
    }

#if XPED_HAS_NTTP
    template <auto a1, auto a2, std::size_t ResRank, std::size_t OtherRank, std::size_t OtherCoRank, bool ENABLE_AD>
    auto contract(const Tensor<Scalar, OtherRank, OtherCoRank, Symmetry, ENABLE_AD, AllocationPolicy>& other) XPED_CONST
    {
        constexpr auto perms = util::constFct::get_permutations<a1, Rank, a2, OtherRank, ResRank>();
        constexpr auto p1 = std::get<0>(perms);
        constexpr auto shift1 = std::get<1>(perms);
        SPDLOG_INFO("shift1={}, p1={}", shift1, p1);
        constexpr auto p2 = std::get<2>(perms);
        constexpr auto shift2 = std::get<3>(perms);
        SPDLOG_INFO("shift2={}, p2={}", shift2, p2);
        constexpr auto pres = std::get<4>(perms);
        constexpr auto shiftres = std::get<5>(perms);
        SPDLOG_INFO("shiftres={}, pres={}", shiftres, pres);
        return (this->template permute<shift1>(util::constFct::as_sequence<p1>()) * other.template permute<shift2>(util::constFct::as_sequence<p2>()))
            .template permute<shiftres>(util::constFct::as_sequence<pres>());
    }
#endif

    std::tuple<Tensor<Scalar, Rank, 1, Symmetry, true, AllocationPolicy>,
               Tensor<RealScalar, 1, 1, Symmetry, true, AllocationPolicy>,
               Tensor<Scalar, 1, CoRank, Symmetry, true, AllocationPolicy>>
    tSVD(std::size_t maxKeep,
         RealScalar eps_svd,
         RealScalar& truncWeight,
         RealScalar& entropy,
         std::map<qarray<Symmetry::Nq>, VectorType>& SVspec,
         bool PRESERVE_MULTIPLETS = true,
         bool RETURN_SPEC = true) XPED_CONST
    {
        auto [Uval, Sval, Vdagval] = val().tSVD(maxKeep, eps_svd, truncWeight, entropy, SVspec, PRESERVE_MULTIPLETS, RETURN_SPEC);
        Tensor<Scalar, Rank, 1, Symmetry, true, AllocationPolicy> U(Uval);
        Tensor<Scalar, 1, 1, Symmetry, true, AllocationPolicy> S(Sval);
        Tensor<Scalar, 1, CoRank, Symmetry, true, AllocationPolicy> Vdag(Vdagval);

        if(grad_tracking) {
            stan::math::reverse_pass_callback([curr = *this, U, S, Vdag]() mutable {
                for(std::size_t i = 0; i < curr.sector().size(); ++i) {
                    SPDLOG_INFO("i={}", i);
                    if(i >= S.val().sector().size()) {
                        // curr.adj().block(i) = PlainInterface::construct_with_zero<Scalar>(
                        //     PlainInterface::rows(curr.val().block(i)), PlainInterface::cols(curr.val().block(i)), *curr.val().world());
                        continue;
                    }
                    auto U_b = U.val().block(i);
                    auto S_b = S.val().block(i);
                    auto Vdag_b = Vdag.val().block(i);

                    SPDLOG_INFO("i={}:\tU.val: {}x{}, U.adj: {}x{}, Vdag.val: {}x{}, Vdag.adj: {}x{}, S.val: {}x{}",
                                i,
                                U_b.rows(),
                                U_b.cols(),
                                U.adj().block(i).rows(),
                                U.adj().block(i).cols(),
                                Vdag_b.rows(),
                                Vdag_b.cols(),
                                Vdag.adj().block(i).rows(),
                                Vdag.adj().block(i).cols(),
                                S_b.rows(),
                                S_b.cols());

                    auto F_inv = PlainInterface::construct<Scalar>(PlainInterface::rows(S_b), PlainInterface::cols(S_b), *S.val().world());
                    PlainInterface::vec_diff(S_b.diagonal().eval(), F_inv);
                    auto F = PlainInterface::unaryFunc<Scalar>(F_inv, [](Scalar d) { return (d < 1.e-12) ? d / (d * d + 1.e-12) : 1. / d; });
                    // fmt::print("S={}\n", S_b.diagonal().transpose());
                    PlainInterface::MType<Scalar> tmp = S_b.diagonal().asDiagonal().inverse();
                    // fmt::print("S_inv={}\n", tmp.diagonal().transpose());
                    // fmt::print("F=\n{}\n", F);
                    auto G_inv = PlainInterface::construct<Scalar>(PlainInterface::rows(S_b), PlainInterface::cols(S_b), *S.val().world());
                    PlainInterface::vec_add(S_b.diagonal().eval(), G_inv);
                    PlainInterface::MType<Scalar> G =
                        PlainInterface::unaryFunc<Scalar>(G_inv, [](Scalar d) { return (d < 1.e-12) ? d / (d * d + 1.e-12) : 1. / d; });
                    G.diagonal().setZero();
                    // fmt::print("G=\n{}\n", G);
                    auto Udag_dU = U_b.adjoint() * U.adj().block(i);
                    auto Vdag_dV = Vdag_b * Vdag.adj().block(i).adjoint();
                    auto Su = ((F + G).array() * (Udag_dU - Udag_dU.adjoint()).array()).matrix() / 2;
                    auto Sv = ((F - G).array() * (Vdag_dV - Vdag_dV.adjoint()).array()).matrix() / 2;
                    curr.adj().block(i) += U_b * (Su + Sv + S.adj().block(i)) * Vdag_b;
                    // fmt::print("dA=\n{}\n", curr.adj().block(i));
                    if(U_b.rows() > S_b.rows()) {
                        curr.adj().block(i) += (PlainInterface::Identity<Scalar>(U_b.rows(), U_b.rows(), *U.val().world()) - U_b * U_b.adjoint()) *
                                               U.adj().block(i) * S_b.diagonal().asDiagonal().inverse() * Vdag_b;
                    }
                    if(Vdag_b.cols() > S_b.rows()) {
                        curr.adj().block(i) +=
                            U_b * S_b.diagonal().asDiagonal().inverse() * Vdag.adj().block(i) *
                            (PlainInterface::Identity<Scalar>(Vdag_b.cols(), Vdag_b.cols(), *Vdag.val().world()) - Vdag_b.adjoint() * Vdag_b);
                    }
                    // fmt::print("max dA={}\n", curr.adj().block(i).maxCoeff());
                }
                SPDLOG_WARN("reverse svd. U.adj.norm()={}, S.adj.norm()={}, Vdag.adj.norm()={}, output adj norm={}",
                            U.adj().norm(),
                            S.adj().norm(),
                            Vdag.adj().norm(),
                            curr.adj().norm());
            });
        }
        return std::make_tuple(U, S, Vdag);
    }

    std::tuple<Tensor<Scalar, Rank, 1, Symmetry, true, AllocationPolicy>,
               Tensor<RealScalar, 1, 1, Symmetry, true, AllocationPolicy>,
               Tensor<Scalar, 1, CoRank, Symmetry, true, AllocationPolicy>>
    tSVD(std::size_t maxKeep, RealScalar eps_svd, RealScalar& truncWeight, bool PRESERVE_MULTIPLETS = true) XPED_CONST
    {
        RealScalar S_dumb;
        std::map<qarray<Symmetry::Nq>, VectorType> SVspec_dumb;
        return tSVD(maxKeep, eps_svd, truncWeight, S_dumb, SVspec_dumb, PRESERVE_MULTIPLETS, false); // false: Dont return singular value spectrum
    }

    stan::math::var_value<Scalar> norm() const
    {
        Scalar tmp = val().norm();
        stan::math::var_value<Scalar> res(tmp);
        if(grad_tracking) {
            stan::math::reverse_pass_callback([curr = *this, res]() mutable {
                curr.adj() += (curr.val() * (res.adj() / res.val())).eval();
                SPDLOG_WARN("reverse norm of {}, input adj norm={}, output adj norm={}", curr.name(), res.adj(), curr.adj().norm());
            });
        }
        return res;
    }

    stan::math::var_value<Scalar> trace() const
    {
        Scalar tmp = val().trace();
        stan::math::var_value<Scalar> res(tmp);
        if(grad_tracking) {
            stan::math::reverse_pass_callback([curr = *this, res]() mutable {
                auto Id =
                    Tensor<Scalar, Rank, CoRank, Symmetry, false>::Identity(curr.uncoupledDomain(), curr.uncoupledCodomain(), *curr.adj().world());
                curr.adj() += Id * res.adj();
                SPDLOG_WARN("reverse trace of {}, input adj norm={}, output adj norm={}", curr.name(), res.adj(), curr.adj().norm());
            });
        }
        return res;
    }

    // Tensor<Scalar, Rank, CoRank, Symmetry, true, AllocationPolicy> sqrt() const
    // {
    //     Tensor<Scalar, Rank, CoRank, Symmetry, true, AllocationPolicy> res(val().sqrt().eval());
    //     stan::math::reverse_pass_callback([curr = *this, res]() mutable {
    //         SPDLOG_WARN("reverse sqrt, in adj norm={}", res.adj().norm());
    //         curr.adj() += res.adj().binaryExpr((res.val().inv()) / 2., [](Scalar d1, Scalar d2) { return d1 * d2; });
    //     });
    //     return res;
    // }

    Tensor<Scalar, Rank, CoRank, Symmetry, true, AllocationPolicy> diag_sqrt() const
    {
        Tensor<Scalar, Rank, CoRank, Symmetry, true, AllocationPolicy> res(val().diag_sqrt().eval());
        if(grad_tracking) {
            stan::math::reverse_pass_callback([curr = *this, res]() mutable {
                curr.adj() += res.adj().binaryExpr(res.val().diag_inv().eval(), [](Scalar d1, Scalar d2) { return d1 * d2 * 0.5; });
                SPDLOG_WARN("reverse sqrt of {}, input adj norm={}, output adj norm={}", curr.name(), res.adj().norm(), curr.adj().norm());
            });
        }
        return res;
    }

    Tensor<Scalar, Rank, CoRank, Symmetry, true, AllocationPolicy> diag_inv() const
    {
        Tensor<Scalar, Rank, CoRank, Symmetry, true, AllocationPolicy> res(val().diag_inv().eval());
        if(grad_tracking) {
            stan::math::reverse_pass_callback([curr = *this, res]() mutable {
                curr.adj() += res.adj().diagBinaryExpr((res.val().square()) * -1., [](Scalar d1, Scalar d2) { return d1 * d2; }).eval();
                SPDLOG_WARN("reverse diag_inv of {}, input adj norm={}, output adj norm={}", curr.name(), res.adj().norm(), curr.adj().norm());
            });
        }
        return res;
    }

    Tensor<Scalar, Rank, CoRank, Symmetry, true, AllocationPolicy>& operator-=(const Scalar s)
    {
        val_op() = val() - s;
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const Tensor<Scalar, Rank, CoRank, Symmetry, true, AllocationPolicy>& v)
    {
        if(v.vi_ == nullptr) { return os << "uninitialized"; }
        return os << v.val();
    }
};

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry>
Tensor<Scalar, Rank, CoRank, Symmetry, true> operator-(const Tensor<Scalar, Rank, CoRank, Symmetry, true>& t, Scalar s)
{
    SPDLOG_CRITICAL("BLOCKER");
    Tensor<Scalar, Rank, CoRank, Symmetry, true> out(t.val() - s);
    return out;
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry>
Tensor<Scalar, Rank, CoRank, Symmetry> operator+(const Tensor<Scalar, Rank, CoRank, Symmetry, true>& t, Scalar s)
{
    SPDLOG_CRITICAL("BLOCKER");
    Tensor<Scalar, Rank, CoRank, Symmetry> out(t.val() + s);
    return out;
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry>
Tensor<Scalar, Rank, CoRank, Symmetry, true> operator*(const Tensor<Scalar, Rank, CoRank, Symmetry, true>& t, Scalar s)
{
    Tensor<Scalar, Rank, CoRank, Symmetry, true> res((t.val() * s).eval());
    if(t.grad_tracking) {
        stan::math::reverse_pass_callback([res, t, s]() mutable {
            t.adj() += (res.adj() * s).eval();
            SPDLOG_WARN("reverse vt*d with vt={}, input adj norm={}, output adj norm={}", t.name(), res.adj().norm(), t.adj().norm());
        });
    }
    return res;
}

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry>
Tensor<Scalar, Rank, CoRank, Symmetry, true> operator*(const Tensor<Scalar, Rank, CoRank, Symmetry, true>& t, stan::math::var_value<Scalar> s)
{
    Tensor<Scalar, Rank, CoRank, Symmetry, true> res((t.val() * s.val()).eval());
    if(t.grad_tracking) {
        stan::math::reverse_pass_callback([res, t, s]() mutable {
            t.adj() += (res.adj() * s.val()).eval();
            s.adj() += (res.adj() * t.val().adjoint()).trace();
            SPDLOG_WARN(
                "reverse vt*v with vt={}, input adj norm={}, vt adj norm={}, v adj norm={}", t.name(), res.adj().norm(), t.adj().norm(), s.adj());
        });
    }
    return res;
}

template <typename Scalar, std::size_t Rank, std::size_t MiddleRank, std::size_t CoRank, typename Symmetry>
Xped::Tensor<Scalar, Rank, CoRank, Symmetry, true> operator*(const Tensor<Scalar, Rank, MiddleRank, Symmetry, false>& left,
                                                             const Tensor<Scalar, MiddleRank, CoRank, Symmetry, true>& right)
{
    Tensor<Scalar, Rank, CoRank, Symmetry, true> res(left * right.val());
    if(right.grad_tracking) {
        Xped::reverse_pass_callback_alloc([res, left, right]() mutable {
            right.adj() += (left.adjoint() * res.adj());
            SPDLOG_WARN("reverse t*vt with t={} and vt={}, input adj norm={}, output adj norm={}",
                        left.name(),
                        right.name(),
                        res.adj().norm(),
                        right.adj().norm());
        });
    }
    return res;
}

template <typename Scalar, std::size_t Rank, std::size_t MiddleRank, std::size_t CoRank, typename Symmetry>
Xped::Tensor<Scalar, Rank, CoRank, Symmetry, true> operator*(const Tensor<Scalar, Rank, MiddleRank, Symmetry, true>& left,
                                                             const Tensor<Scalar, MiddleRank, CoRank, Symmetry, false>& right)
{
    Tensor<Scalar, Rank, CoRank, Symmetry, true> res(left.val() * right);
    if(left.grad_tracking) {
        Xped::reverse_pass_callback_alloc([res, left, right]() mutable {
            left.adj() += res.adj() * right.adjoint();
            SPDLOG_WARN("reverse vt*t with vt={} and t={}, input adj norm={}, output adj norm={}",
                        left.name(),
                        right.name(),
                        res.adj().norm(),
                        left.adj().norm());
        });
    }
    return res;
}

template <typename Scalar, std::size_t Rank, std::size_t MiddleRank, std::size_t CoRank, typename Symmetry>
Xped::Tensor<Scalar, Rank, CoRank, Symmetry, true> operator*(const Tensor<Scalar, Rank, MiddleRank, Symmetry, true>& left,
                                                             const Tensor<Scalar, MiddleRank, CoRank, Symmetry, true>& right)
{
    assert(left.grad_tracking == right.grad_tracking);
    Tensor<Scalar, Rank, CoRank, Symmetry, true> res(left.val() * right.val());
    if(left.grad_tracking) {
        stan::math::reverse_pass_callback([res, left, right]() mutable {
            right.adj() += (left.val().adjoint() * res.adj());
            left.adj() += res.adj() * right.val().adjoint();
            SPDLOG_WARN("reverse vt*vt with vtl={} and vtr={}, in adj norm={}, vtl adj norm={}, vtr adj norm={}",
                        left.name(),
                        right.name(),
                        res.adj().norm(),
                        left.adj().norm(),
                        right.adj().norm());
        });
    }
    return res;
}

template <typename Scalar, std::size_t Rank, typename Symmetry>
stan::math::var operator*(const Tensor<Scalar, 0, Rank, Symmetry, true>& left, const Tensor<Scalar, Rank, 0, Symmetry, true>& right)
{
    assert(left.grad_tracking == right.grad_tracking);
    stan::math::var res = (left.val() * right.val()).block(0)(0, 0);
    if(left.grad_tracking) {
        stan::math::reverse_pass_callback([res, left, right]() mutable {
            right.adj() += left.val().adjoint() * res.adj();
            left.adj() += res.adj() * right.val().adjoint();
        });
    }
    return res;
}

template <typename Scalar, std::size_t Rank, typename Symmetry>
stan::math::var operator*(const Tensor<Scalar, 0, Rank, Symmetry, false>& left, const Tensor<Scalar, Rank, 0, Symmetry, true>& right)
{
    stan::math::var res = (left * right.val()).block(0)(0, 0);
    if(right.grad_tracking) {
        Xped::reverse_pass_callback_alloc([res, left, right]() mutable { right.adj() += left.adjoint() * res.adj(); });
    }
    return res;
}

// template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry>
// stan::math::var_value<Xped::Tensor<Scalar, CoRank, Rank, Symmetry>>
// adjoint(const stan::math::var_value<Xped::Tensor<Scalar, Rank, CoRank, Symmetry>>& t)
// {
//     stan::math::var_value<Xped::Tensor<Scalar, CoRank, Rank, Symmetry>> res(t.val().adjoint().eval());
//     stan::math::reverse_pass_callback([res, t]() mutable { t.adj() += res.adj().adjoint().eval(); });
//     return res;
// }

} // namespace Xped
#endif
