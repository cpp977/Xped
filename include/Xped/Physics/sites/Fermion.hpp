#ifndef XPED_FERMIONSITE_HPP_
#define XPED_FERMIONSITE_HPP_

#include "Xped/Physics/SiteOperator.hpp"
#include "Xped/Physics/SpinIndex.hpp"
#include "Xped/Physics/SpinOp.hpp"
#include "Xped/Symmetry/U0.hpp"

namespace Xped {

template <typename Symmetry_>
class Fermion
{
    using Scalar = double;
    using Symmetry = Symmetry_;
    using OperatorType = SiteOperator<Scalar, Symmetry>;
    using qType = typename Symmetry::qType;

public:
    Fermion(){};
    Fermion(bool REMOVE_DOUBLE, bool REMOVE_EMPTY, bool REMOVE_SINGLE, int mfactor_input = 1);

    OperatorType Id_1s() const { return Id_1s_; }
    OperatorType F_1s() const { return F_1s_; }

    OperatorType c_1s(SPIN_INDEX sigma) const
    {
        if(sigma == SPIN_INDEX::UP) { return cup_1s_; }
        return cdn_1s_;
    }
    OperatorType cdag_1s(SPIN_INDEX sigma) const
    {
        if(sigma == SPIN_INDEX::UP) { return cdagup_1s_; }
        return cdagdn_1s_;
    }

    OperatorType n_1s() const { return n_1s_; }
    OperatorType n_1s(SPIN_INDEX sigma) const
    {
        if(sigma == SPIN_INDEX::UP) { return nup_1s_; }
        return ndn_1s_;
    }
    OperatorType ns_1s() const { return n_1s() - 2. * d_1s(); }
    OperatorType nh_1s() const { return 2. * d_1s() - n_1s() + Id_1s(); }
    OperatorType d_1s() const { return d_1s_; }

    OperatorType Sz_1s() const { return Sz_1s_; }
    OperatorType Sp_1s() const { return Sp_1s_; }
    OperatorType Sm_1s() const { return Sm_1s_; }

    OperatorType Tz_1s() const { return 0.5 * (n_1s() - Id_1s()); }
    OperatorType cc_1s() const { return cc_1s_; }
    OperatorType cdagcdag_1s() const { return cdagcdag_1s_; }

    Qbasis<Symmetry, 1> basis_1s() const { return basis_1s_; }

protected:
    int mfactor = 1;

    void fill_basis(bool REMOVE_DOUBLE, bool REMOVE_EMPTY, bool REMOVE_SINGLE);
    void fill_SiteOps(bool REMOVE_DOUBLE, bool REMOVE_EMPTY, bool REMOVE_SINGLE);

    qType getQ(SPIN_INDEX sigma, int Delta) const;
    qType getQ(SPINOP_LABEL Sa) const;

    Qbasis<Symmetry, 1> basis_1s_;
    std::unordered_map<std::string, std::pair<qType, std::size_t>> labels;

    OperatorType Id_1s_; // identity
    OperatorType F_1s_; // Fermionic sign

    OperatorType cup_1s_; // annihilation
    OperatorType cdn_1s_; // annihilation

    OperatorType cdagup_1s_; // creation
    OperatorType cdagdn_1s_; // creation

    OperatorType n_1s_; // particle number
    OperatorType nup_1s_; // particle number
    OperatorType ndn_1s_; // particle number
    OperatorType d_1s_; // double occupancy

    OperatorType Sz_1s_; // orbital spin
    OperatorType Sp_1s_; // orbital spin
    OperatorType Sm_1s_; // orbital spin

    OperatorType Tz_1s_; // orbital pseude spin
    OperatorType cc_1s_; // pairing
    OperatorType cdagcdag_1s_; // pairing adjoint
};

template <typename Symmetry_>
Fermion<Symmetry_>::Fermion(bool REMOVE_DOUBLE, bool REMOVE_EMPTY, bool REMOVE_SINGLE, int mfactor_input)
    : mfactor(mfactor_input)
{
    // create basis for one Fermionic Site
    fill_basis(REMOVE_DOUBLE, REMOVE_EMPTY, REMOVE_SINGLE);
    std::cout << "single site basis" << std::endl << this->basis_1s_ << std::endl;
    fmt::print("labels=\n{}\n", labels);
    fill_SiteOps(REMOVE_DOUBLE, REMOVE_EMPTY, REMOVE_SINGLE);
    //	cout << "fill_SiteOps done!" << endl;
}

template <typename Symmetry_>
void Fermion<Symmetry_>::fill_SiteOps(bool REMOVE_DOUBLE, bool REMOVE_EMPTY, bool REMOVE_SINGLE)
{
    // create operators for one site
    Id_1s_ = OperatorType(Symmetry::qvacuum(), basis_1s_, labels);
    F_1s_ = OperatorType(Symmetry::qvacuum(), basis_1s_, labels);

    cup_1s_ = OperatorType(getQ(SPIN_INDEX::UP, -1), basis_1s_, labels);
    cup_1s_.setZero();
    cdn_1s_ = OperatorType(getQ(SPIN_INDEX::DN, -1), basis_1s_, labels);
    cdn_1s_.setZero();

    cdagup_1s_ = OperatorType(getQ(SPIN_INDEX::UP, +1), basis_1s_, labels);
    cdagdn_1s_ = OperatorType(getQ(SPIN_INDEX::DN, +1), basis_1s_, labels);

    n_1s_ = OperatorType(Symmetry::qvacuum(), basis_1s_, labels);
    nup_1s_ = OperatorType(Symmetry::qvacuum(), basis_1s_, labels);
    ndn_1s_ = OperatorType(Symmetry::qvacuum(), basis_1s_, labels);
    d_1s_ = OperatorType(Symmetry::qvacuum(), basis_1s_, labels);

    Sz_1s_ = OperatorType(getQ(SPINOP_LABEL::SZ), basis_1s_, labels);
    Sz_1s_.setZero();
    Sp_1s_ = OperatorType(getQ(SPINOP_LABEL::SP), basis_1s_, labels);
    Sp_1s_.setZero();
    Sm_1s_ = OperatorType(getQ(SPINOP_LABEL::SM), basis_1s_, labels);
    Sm_1s_.setZero();

    cc_1s_ = OperatorType(getQ(SPIN_INDEX::UPDN, -1), basis_1s_, labels);
    cdagcdag_1s_ = OperatorType(getQ(SPIN_INDEX::UPDN, +1), basis_1s_, labels);

    Id_1s_.setIdentity();

    F_1s_.setZero();

    if(!REMOVE_EMPTY) F_1s_("empty", "empty") = 1.;
    if(!REMOVE_DOUBLE) F_1s_("double", "double") = 1.;
    if(!REMOVE_SINGLE) {
        F_1s_("up", "up") = -1.;
        F_1s_("dn", "dn") = -1.;
    }

    fmt::print("Setting up cup.\n");
    if(!REMOVE_EMPTY and !REMOVE_SINGLE) { cup_1s_("empty", "up") = 1.; }
    if(!REMOVE_DOUBLE and !REMOVE_SINGLE) { cup_1s_("dn", "double") = 1.; }

    fmt::print("Setting up cdn.\n");
    if(!REMOVE_EMPTY and !REMOVE_SINGLE) { cdn_1s_("empty", "dn") = 1.; }
    if(!REMOVE_EMPTY and !REMOVE_SINGLE) { cdn_1s_("up", "double") = -1.; }

    fmt::print("Computing cdagup.\n");
    cdagup_1s_ = cup_1s_.adjoint();
    fmt::print("Computing cdagdn.\n");
    cdagdn_1s_ = cdn_1s_.adjoint();

    std::cout << std::fixed << cdagup_1s_ << std::endl
              << cdagup_1s_.plain()[0] << std::endl
              << cup_1s_ << std::endl
              << cup_1s_.plain()[0] << std::endl;

    fmt::print("Computing nup.\n");
    nup_1s_ = cdagup_1s_ * cup_1s_;
    auto check = nup_1s_.data.template permute<0, 1, 0, 2>().template trim<2>(); //(cup_1s_ * cdagup_1s_).data.template trim<2>();
    // [[maybe_unused]] double t_weight;
    // auto [U, S, Vdag] = check.tSVD(std::numeric_limits<std::size_t>::max(), 0, t_weight, false);
    // fmt::print("Singular values of nup.\n");
    // S.print(std::cout, true);
    // std::cout << std::endl;
    // auto chchd = check.template contract<std::array{-1, 1}, std::array{1, -2}, 1>(check.adjoint().eval());
    // for(auto r = 0ul; r < chchd.rank(); ++r) {
    //     if(chchd.uncoupledDomain()[r].IS_CONJ()) { chchd = chchd.twist(r); }
    // }
    // auto [D, V] = chchd.eigh();
    // fmt::print("Eigenvalues of nup*nup.\n");
    // D.print(std::cout, true);
    // std::cout << std::endl;
    // SiteOperator<Scalar, Symmetry> nupc(Symmetry::qvacuum(), cup_1s_.data.coupledDomain());
    // Qbasis<Symmetry, 1> Otarget_op;
    // Otarget_op.push_back(Symmetry::qvacuum(), 1);
    // Tensor<double, 2, 1, Symmetry, false> couple({{cup_1s_.data.uncoupledCodomain()[1], cdagup_1s_.data.uncoupledCodomain()[1]}}, {{Otarget_op}});
    // couple.setConstant(1.);
    // auto tmp1 = cup_1s_.data.template contract<std::array{1, -3, -4}, std::array{-1, 1, -2}, 2>(cdagup_1s_.data);
    // nupc.data = tmp1.template contract<std::array{-1, 1, -2, 2}, std::array{2, 1, -3}, 1>(couple);

    // auto nup1 = cup_1s_.data.adjoint().eval().template contract<std::array{-1, 1, 2}, std::array{2, -2, 1}, 1>(cup_1s_.data);
    // auto nup2 = cup_1s_.data.template contract<std::array{2, -2, 1}, std::array{-1, 1, 2}, 1>(cup_1s_.data.adjoint().eval());

    fmt::print("Computing ndn.\n");
    ndn_1s_ = cdagdn_1s_ * cdn_1s_;
    std::cout << std::fixed << nup_1s_ << std::endl << nup_1s_.plain()[0] << std::endl << std::endl;
    std::cout << std::fixed << check << std::endl << check.plainTensor() << std::endl << std::endl;
    auto Id = Xped::Tensor<double, 1, 1, Symmetry>::Identity({{check.coupledDomain().conj()}}, {{check.coupledDomain().conj()}});
    auto trt = check.template contract<std::array{1, 2}, std::array{1, 2}, 0>(Id);
    check = check.twist(0);
    fmt::print("tr(n)={}, tr(check)={}, with contract: {}\n", nup_1s_.data.template trim<2>().trace(), check.trace(), trt.block(0)(0, 0));

    // std::cout << std::fixed << nup1 << std::endl << nup1.plainTensor() << std::endl << std::endl;
    // std::cout << std::fixed << nup2 << std::endl << nup2.plainTensor() << std::endl << std::endl;
    fmt::print("Computing n.\n");
    n_1s_ = nup_1s_ + ndn_1s_;

    d_1s_.setZero();
    if(!REMOVE_DOUBLE) d_1s_("double", "double") = 1.;

    fmt::print("Computing cc.\n");
    cc_1s_ = cdn_1s_ * cup_1s_;
    fmt::print("Computing cdagcdag.\n");
    cdagcdag_1s_ = cc_1s_.adjoint();

    if(!REMOVE_SINGLE) {
        Sz_1s_ = 0.5 * (nup_1s_ - ndn_1s_);
        Sp_1s_ = cup_1s_.adjoint() * cdn_1s_;
        Sm_1s_ = Sp_1s_.adjoint();
    }
}

template <typename Symmetry_>
void Fermion<Symmetry_>::fill_basis(bool REMOVE_DOUBLE, bool REMOVE_EMPTY, bool REMOVE_SINGLE)
{
    bool U_IS_INFINITE = false;
    bool UPH_IS_INFINITE = false;

    // if constexpr(std::is_same<Symmetry, Sym::S1xS2<Sym::U1<Sym::SpinU1>, Sym::U1<Sym::ChargeU1>>>::value) // U1xU1
    // {
    //     qType Q;

    //     if(!REMOVE_EMPTY) {
    //         Q = {0, 0};
    //         labels.insert(std::make_pair("empty", std::make_pair(Symmetry::qvacuum(), 0)));
    //         this->basis_1s_.push_back(Q, 1);
    //     }

    //     if(!REMOVE_SINGLE) {
    //         Q = {+mfactor, 1};
    //         this->basis_1s_.push_back(Q, 1);
    //         labels.insert(std::make_pair("up", std::make_pair(Q, 1)));

    //         Q = {-mfactor, 1};
    //         this->basis_1s_.push_back(Q, 1);
    //         labels.insert(std::make_pair("dn", std::make_pair(Q, 1)));
    //     }

    //     if(!REMOVE_DOUBLE) {
    //         Q = {0, 2};
    //         this->basis_1s_.push_back(Q, 1);
    //         labels.insert(std::make_pair("double", std::make_pair(Q, 1)));
    //     }
    // } else
    if constexpr(std::is_same<Symmetry, Sym::U0<Scalar>>::value) // U0
    {
        qType Q = {};

        if(!UPH_IS_INFINITE and U_IS_INFINITE) {
            this->basis_1s_.push_back(Q, 3);
            labels.insert(std::make_pair("empty", std::make_pair(Q, 0)));
            labels.insert(std::make_pair("up", std::make_pair(Q, 1)));
            labels.insert(std::make_pair("dn", std::make_pair(Q, 2)));
        } else if(!U_IS_INFINITE and !UPH_IS_INFINITE) {
            this->basis_1s_.push_back(Q, 4);
            labels.insert(std::make_pair("empty", std::make_pair(Q, 0)));
            labels.insert(std::make_pair("up", std::make_pair(Q, 1)));
            labels.insert(std::make_pair("dn", std::make_pair(Q, 2)));
            labels.insert(std::make_pair("double", std::make_pair(Q, 3)));
        } else {
            this->basis_1s_.push_back(Q, 2);
            labels.insert(std::make_pair("up", std::make_pair(Q, 1)));
            labels.insert(std::make_pair("dn", std::make_pair(Q, 2)));
        }
    } else if constexpr(std::is_same<Symmetry, Sym::U1<Sym::SpinU1>>::value) // spin U1
    {
        typename Symmetry::qType Q;

        if(!REMOVE_EMPTY) {
            Q = {0};
            this->basis_1s_.push_back(Q, 1);
            labels.insert(std::make_pair("empty", std::make_pair(Q, 0)));
        }

        if(!REMOVE_DOUBLE) {
            Q = {0};
            this->basis_1s_.push_back(Q, 2);
            labels.insert(std::make_pair("empty", std::make_pair(Q, 0)));
            labels.insert(std::make_pair("double", std::make_pair(Q, 1)));
        }

        if(!REMOVE_SINGLE) {
            Q = {+mfactor};
            this->basis_1s_.push_back(Q, 1);
            labels.insert(std::make_pair("up", std::make_pair(Q, 0)));

            Q = {-mfactor};
            this->basis_1s_.push_back(Q, 1);
            labels.insert(std::make_pair("dn", std::make_pair(Q, 0)));
        }
    } else if constexpr(std::is_same<Symmetry, Sym::U1<Sym::SpinU1>>::value) // spin U1
    {
        typename Symmetry::qType Q;

        if(!REMOVE_EMPTY) {
            Q = {0};
            this->basis_1s_.push_back(Q, 1);
            labels.insert(std::make_pair("empty", std::make_pair(Q, 0)));
        }

        if(!REMOVE_DOUBLE) {
            Q = {0};
            this->basis_1s_.push_back(Q, 2);
            labels.insert(std::make_pair("empty", std::make_pair(Q, 0)));
            labels.insert(std::make_pair("double", std::make_pair(Q, 1)));
        }

        if(!REMOVE_SINGLE) {
            Q = {+mfactor};
            this->basis_1s_.push_back(Q, 1);
            labels.insert(std::make_pair("up", std::make_pair(Q, 0)));

            Q = {-mfactor};
            this->basis_1s_.push_back(Q, 1);
            labels.insert(std::make_pair("dn", std::make_pair(Q, 0)));
        }
    } else if constexpr(std::is_same<Symmetry, Sym::U1<Sym::ChargeU1>>::value or std::is_same<Symmetry, Sym::U1<Sym::FChargeU1>>::value or
                        std::is_same<Symmetry, Sym::ZN<Sym::FChargeU1, 2>>::value) // charge U1
    {
        typename Symmetry::qType Q;

        if(!REMOVE_EMPTY and !REMOVE_DOUBLE) {
            Q = {0};
            this->basis_1s_.push_back(Q, 2);
            labels.insert(std::make_pair("empty", std::make_pair(Q, 0)));
            labels.insert(std::make_pair("double", std::make_pair(Q, 1)));
        }

        if(!REMOVE_SINGLE) {
            Q = {1};
            this->basis_1s_.push_back(Q, 2);
            labels.insert(std::make_pair("up", std::make_pair(Q, 0)));
            labels.insert(std::make_pair("dn", std::make_pair(Q, 1)));
        }
    }
}

template <typename Symmetry_>
typename Symmetry_::qType Fermion<Symmetry_>::getQ(SPIN_INDEX sigma, int Delta) const
{
    if constexpr(Symmetry::IS_TRIVIAL) {
        return {};
    } else if constexpr(Symmetry::Nq == 1) {
        if constexpr(Symmetry::kind()[0] == Sym::KIND::N or Symmetry::kind()[0] == Sym::KIND::FN) // return particle number as good quantum number.
        {
            typename Symmetry::qType out;
            if(sigma == SPIN_INDEX::UP) {
                out = {Delta};
            } else if(sigma == SPIN_INDEX::DN) {
                out = {Delta};
            } else if(sigma == SPIN_INDEX::UPDN) {
                out = {2 * Delta};
            } else if(sigma == SPIN_INDEX::NOSPIN) {
                out = Symmetry::qvacuum();
            }
            if(sigma == SPIN_INDEX::UP) {
                out = {1};
            } else if(sigma == SPIN_INDEX::DN) {
                out = {1};
            } else if(sigma == SPIN_INDEX::UPDN) {
                out = Symmetry::qvacuum();
            } else if(sigma == SPIN_INDEX::NOSPIN) {
                out = Symmetry::qvacuum();
            }
            return out;
        } else if constexpr(Symmetry::kind()[0] == Sym::KIND::M) // return magnetization as good quantum number.
        {
            typename Symmetry::qType out;
            if(sigma == SPIN_INDEX::UP) {
                out = {mfactor * Delta};
            } else if(sigma == SPIN_INDEX::DN) {
                out = {-mfactor * Delta};
            } else if(sigma == SPIN_INDEX::UPDN) {
                out = Symmetry::qvacuum();
            } else if(sigma == SPIN_INDEX::NOSPIN) {
                out = Symmetry::qvacuum();
            }
            return out;
        } else if constexpr(Symmetry::kind()[0] == Sym::KIND::Z2) // return parity as good quantum number.
        {
            typename Symmetry::qType out;
            if(sigma == SPIN_INDEX::UP) {
                out = {util::posmod<2>(Delta)};
            } else if(sigma == SPIN_INDEX::DN) {
                out = {util::posmod<2>(-Delta)};
            } else if(sigma == SPIN_INDEX::UPDN) {
                out = Symmetry::qvacuum();
            } else if(sigma == SPIN_INDEX::NOSPIN) {
                out = Symmetry::qvacuum();
            }
            return out;
        } else {
            assert(false and "Ill defined KIND of the used Symmetry.");
        }
    } else if constexpr(Symmetry::Nq == 2) {
        typename Symmetry::qType out;
        if constexpr(Symmetry::kind()[0] == Sym::KIND::N and Symmetry::kind()[1] == Sym::KIND::M) {
            if(sigma == SPIN_INDEX::UP) {
                out = {Delta, mfactor * Delta};
            } else if(sigma == SPIN_INDEX::DN) {
                out = {Delta, -mfactor * Delta};
            } else if(sigma == SPIN_INDEX::UPDN) {
                out = {2 * Delta, 0};
            } else if(sigma == SPIN_INDEX::NOSPIN) {
                out = Symmetry::qvacuum();
            }
        } else if constexpr(Symmetry::kind()[0] == Sym::KIND::M and Symmetry::kind()[1] == Sym::KIND::N) {
            if(sigma == SPIN_INDEX::UP) {
                out = {mfactor * Delta, Delta};
            } else if(sigma == SPIN_INDEX::DN) {
                out = {-mfactor * Delta, Delta};
            } else if(sigma == SPIN_INDEX::UPDN) {
                out = {0, 2 * Delta};
            } else if(sigma == SPIN_INDEX::NOSPIN) {
                out = Symmetry::qvacuum();
            }
        }
        // Not possible to use mfactor with these?
        else if constexpr(Symmetry::kind()[0] == Sym::KIND::Nup and Symmetry::kind()[1] == Sym::KIND::Ndn) {
            if(sigma == SPIN_INDEX::UP) {
                out = {Delta, 0};
            } else if(sigma == SPIN_INDEX::DN) {
                out = {0, Delta};
            } else if(sigma == SPIN_INDEX::UPDN) {
                out = {Delta, Delta};
            } else if(sigma == SPIN_INDEX::NOSPIN) {
                out = Symmetry::qvacuum();
            }
        } else if constexpr(Symmetry::kind()[0] == Sym::KIND::Ndn and Symmetry::kind()[1] == Sym::KIND::Nup) {
            if(sigma == SPIN_INDEX::UP) {
                out = {0, Delta};
            } else if(sigma == SPIN_INDEX::DN) {
                out = {Delta, 0};
            } else if(sigma == SPIN_INDEX::UPDN) {
                out = {Delta, Delta};
            } else if(sigma == SPIN_INDEX::NOSPIN) {
                out = Symmetry::qvacuum();
            }
        }
        return out;
    }
    static_assert("You inserted a Symmetry which can not be handled by FermionBase.");
}

template <typename Symmetry_>
typename Symmetry_::qType Fermion<Symmetry_>::getQ(SPINOP_LABEL Sa) const
{
    if constexpr(Symmetry::IS_TRIVIAL) {
        return {};
    } else if constexpr(Symmetry::Nq == 1) {
        if constexpr(Symmetry::kind()[0] == Sym::KIND::N or Symmetry::kind()[0] == Sym::KIND::Z2 or
                     Symmetry::kind()[0] == Sym::KIND::FN) // return particle number as good quantum number.
        {
            return Symmetry::qvacuum();
        } else if constexpr(Symmetry::kind()[0] == Sym::KIND::M) // return magnetization as good quantum number.
        {
            assert(Sa != SPINOP_LABEL::SX and Sa != SPINOP_LABEL::iSY);

            typename Symmetry::qType out;
            if(Sa == SPINOP_LABEL::SZ) {
                out = {0};
            } else if(Sa == SPINOP_LABEL::SP) {
                out = {+2};
            } else if(Sa == SPINOP_LABEL::SM) {
                out = {-2};
            }
            return out;
        } else {
            assert(false and "Ill defined KIND of the used Symmetry.");
        }
    } else if constexpr(Symmetry::Nq == 2) {
        assert(Sa != SPINOP_LABEL::SX and Sa != SPINOP_LABEL::iSY);

        typename Symmetry::qType out;
        if constexpr(Symmetry::kind()[0] == Sym::KIND::N and Symmetry::kind()[1] == Sym::KIND::M) {
            if(Sa == SPINOP_LABEL::SZ) {
                out = {0, 0};
            } else if(Sa == SPINOP_LABEL::SP) {
                out = {0, +2};
            } else if(Sa == SPINOP_LABEL::SM) {
                out = {0, -2};
            }
        } else if constexpr(Symmetry::kind()[0] == Sym::KIND::M and Symmetry::kind()[1] == Sym::KIND::N) {
            if(Sa == SPINOP_LABEL::SZ) {
                out = {0, 0};
            } else if(Sa == SPINOP_LABEL::SP) {
                out = {+2, 0};
            } else if(Sa == SPINOP_LABEL::SM) {
                out = {-2, 0};
            }
        } else if constexpr(Symmetry::kind()[0] == Sym::KIND::Nup and Symmetry::kind()[1] == Sym::KIND::Ndn) {
            if(Sa == SPINOP_LABEL::SZ) {
                out = {0, 0};
            } else if(Sa == SPINOP_LABEL::SP) {
                out = {+1, -1};
            } else if(Sa == SPINOP_LABEL::SM) {
                out = {-1, +1};
            }
        } else if constexpr(Symmetry::kind()[0] == Sym::KIND::Ndn and Symmetry::kind()[1] == Sym::KIND::Nup) {
            if(Sa == SPINOP_LABEL::SZ) {
                out = {0, 0};
            } else if(Sa == SPINOP_LABEL::SP) {
                out = {-1, +1};
            } else if(Sa == SPINOP_LABEL::SM) {
                out = {+1, -1};
            }
        }
        return out;
    }
    static_assert("You inserted a Symmetry which can not be handled by FermionBase.");
}

} // namespace Xped
#endif // FERMIONSITE_H_
