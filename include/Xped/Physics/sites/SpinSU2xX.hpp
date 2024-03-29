#ifndef XPED_SPINSU2xX_HPP_
#define XPED_SPINSU2xX_HPP_

#include "Xped/Symmetry/kind_dummies.hpp"

#include "Xped/Symmetry/CombSym.hpp"
#include "Xped/Symmetry/SU2.hpp"

namespace Xped {

template <typename Symmetry, std::size_t order>
class Spin;

template <typename Symmetry_>
class Spin<Sym::Combined<Sym::SU2<Sym::SpinSU2>, Symmetry_>>
{
    using Scalar = double;
    using Symmetry = Sym::Combined<Sym::SU2<Sym::SpinSU2>, Symmetry_>;
    using OperatorType = SiteOperator<Scalar, Symmetry>;

public:
    Spin(){};
    Spin(std::size_t D_input);

    OperatorType Id_1s() const { return Id_1s_; }

    OperatorType S_1s() const { return S_1s_; }

    OperatorType Q_1s() const { return Q_1s_; }

    Qbasis<Symmetry, 1> basis_1s() const { return basis_1s_; }

protected:
    std::size_t D;

    Qbasis<Symmetry, 1> basis_1s_;

    OperatorType Id_1s_; // identity
    OperatorType S_1s_; // spin
    OperatorType Sdag_1s_; // spin
    OperatorType Q_1s_; // quadrupole moment
};

template <typename Symmetry_>
Spin<Sym::Combined<Sym::SU2<Sym::SpinSU2>, Symmetry_>, 0ul>::Spin(std::size_t D_input)
    : D(D_input)
{
    // create basis for one Spin Site
    typename Symmetry::qType Q = {static_cast<int>(D), Symmetry_::qvacuum()[0]};
    basis_1s_.push_back(Q, 1);

    Id_1s_ = OperatorType(Symmetry::qvacuum(), basis_1s_);
    Id_1s_.setIdentity();
    S_1s_ = OperatorType({3, Symmetry_::qvacuum()[0]}, basis_1s_);
    S_1s_.data.setZero();

    Scalar locS = 0.5 * static_cast<double>(D - 1);
    S_1s_(Q, Q)(0, 0) = std::sqrt(locS * (locS + 1.));
    Sdag_1s_ = S_1s_.adjoint();
    Q_1s_ = std::sqrt(2.) * OperatorType::prod(S_1s_, S_1s_, {5, Symmetry_::qvacuum()[0]});
}

} // namespace Xped

#endif
