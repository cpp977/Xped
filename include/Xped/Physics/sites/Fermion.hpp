#ifndef XPED_FERMIONSITE_HPP_
#define XPED_FERMIONSITE_HPP_

#include "Xped/Physics/SiteOperator.hpp"
#include "Xped/Physics/SpinIndex.hpp"
#include "Xped/Physics/SpinOp.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Util/Constfct.hpp"

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
    Fermion(bool REMOVE_DOUBLE, bool REMOVE_EMPTY, bool REMOVE_SINGLE, bool CONSIDER_SPIN = true, bool CONSIDER_CHARGE = true);

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
    void fill_basis(bool REMOVE_DOUBLE, bool REMOVE_EMPTY, bool REMOVE_SINGLE);
    void fill_SiteOps(bool REMOVE_DOUBLE, bool REMOVE_EMPTY, bool REMOVE_SINGLE);

    qType getQ(SPIN_INDEX sigma, int Delta) const;
    qType getQ(SPINOP_LABEL Sa) const;

    Qbasis<Symmetry, 1> basis_1s_;
    std::unordered_map<std::string, std::pair<qType, std::size_t>> labels;

    std::size_t sp_index = 0;
    std::size_t ch_index = 0;

    bool HAS_SPIN;
    bool HAS_CHARGE;

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
Fermion<Symmetry_>::Fermion(bool REMOVE_DOUBLE, bool REMOVE_EMPTY, bool REMOVE_SINGLE, bool CONSIDER_SPIN, bool CONSIDER_CHARGE)
{
    if(CONSIDER_SPIN) {
        for(std::size_t q = 0; q < Symmetry::Nq; ++q) {
            if(Symmetry::IS_SPIN[q]) {
                sp_index = q;
                HAS_SPIN = true;
                break;
            }
        }
    } else {
        HAS_SPIN = false;
    }

    if(CONSIDER_CHARGE) {
        for(std::size_t q = 0; q < Symmetry::Nq; ++q) {
            if(Symmetry::IS_FERMIONIC[q] or Symmetry::IS_BOSONIC[q]) {
                ch_index = q;
                HAS_CHARGE = true;
                break;
            }
        }
    } else {
        HAS_CHARGE = false;
    }

    // create basis for one Fermionic Site
    fill_basis(REMOVE_DOUBLE, REMOVE_EMPTY, REMOVE_SINGLE);
    basis_1s_.sort();
    fill_SiteOps(REMOVE_DOUBLE, REMOVE_EMPTY, REMOVE_SINGLE);
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

    if(!REMOVE_EMPTY and !REMOVE_SINGLE) { cup_1s_("empty", "up") = 1.; }
    if(!REMOVE_DOUBLE and !REMOVE_SINGLE) { cup_1s_("dn", "double") = 1.; }

    if(!REMOVE_EMPTY and !REMOVE_SINGLE) { cdn_1s_("empty", "dn") = 1.; }
    if(!REMOVE_EMPTY and !REMOVE_SINGLE) { cdn_1s_("up", "double") = -1.; }

    cdagup_1s_ = cup_1s_.adjoint();
    cdagdn_1s_ = cdn_1s_.adjoint();

    // std::cout << std::fixed << cdagup_1s_ << std::endl
    //           << cdagup_1s_.plain()[0] << std::endl
    //           << cup_1s_ << std::endl
    //           << cup_1s_.plain()[0] << std::endl;

    nup_1s_ = cdagup_1s_ * cup_1s_;
    ndn_1s_ = cdagdn_1s_ * cdn_1s_;
    n_1s_ = nup_1s_ + ndn_1s_;

    d_1s_.setZero();
    if(!REMOVE_DOUBLE) d_1s_("double", "double") = 1.;

    cc_1s_ = cdn_1s_ * cup_1s_;
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
    if constexpr(Symmetry::ALL_IS_TRIVIAL) // U0
    {
        qType Q = Symmetry::qvacuum();
        std::size_t dim = 0;
        if(not REMOVE_EMPTY) { labels.insert(std::make_pair("empty", std::make_pair(Q, dim++))); }
        if(not REMOVE_SINGLE) {
            labels.insert(std::make_pair("up", std::make_pair(Q, dim++)));
            labels.insert(std::make_pair("dn", std::make_pair(Q, dim++)));
        }
        if(not REMOVE_DOUBLE) { labels.insert(std::make_pair("double", std::make_pair(Q, dim++))); }
        this->basis_1s_.push_back(Q, dim);
        return;
    } else {
        if(not HAS_SPIN and HAS_CHARGE) {
            qType Q = Symmetry::qvacuum();
            if(not REMOVE_EMPTY) {
                this->basis_1s_.push_back(Q, 1);
                labels.insert(std::make_pair("empty", std::make_pair(Q, 0)));
            }
            if(not REMOVE_DOUBLE) {
                std::size_t dim = (Symmetry::MOD_N[ch_index] == 2) ? 1 : 0;
                Q[ch_index] = Symmetry::IS_MODULAR[ch_index] ? util::constFct::posmod(2, Symmetry::MOD_N[ch_index]) : 2;
                this->basis_1s_.push_back(Q, 1);
                labels.insert(std::make_pair("double", std::make_pair(Q, dim)));
            }
            if(not REMOVE_SINGLE) {
                Q[ch_index] = 1;
                this->basis_1s_.push_back(Q, 2);
                labels.insert(std::make_pair("up", std::make_pair(Q, 0)));
                labels.insert(std::make_pair("dn", std::make_pair(Q, 1)));
            }
        } else if(HAS_SPIN and not HAS_CHARGE) {
            qType Q = Symmetry::qvacuum();
            std::size_t dim = 0;
            if(not REMOVE_EMPTY) { labels.insert(std::make_pair("empty", std::make_pair(Q, dim++))); }
            if(not REMOVE_DOUBLE) { labels.insert(std::make_pair("double", std::make_pair(Q, dim++))); }
            this->basis_1s_.push_back(Q, dim);
            if(not REMOVE_SINGLE) {
                Q[sp_index] = 1;
                this->basis_1s_.push_back(Q, 1);
                labels.insert(std::make_pair("up", std::make_pair(Q, 0)));
                Q[sp_index] = Symmetry::IS_MODULAR[sp_index] ? util::constFct::posmod(-1, Symmetry::MOD_N[sp_index]) : -1;
                this->basis_1s_.push_back(Q, 1);
                labels.insert(std::make_pair("dn", std::make_pair(Q, 0)));
            }
        } else {
            qType Q = Symmetry::qvacuum();
            if(not REMOVE_EMPTY) {
                this->basis_1s_.push_back(Q, 1);
                labels.insert(std::make_pair("empty", std::make_pair(Q, 0)));
            }
            if(not REMOVE_DOUBLE) {
                std::size_t dim = (Symmetry::MOD_N[ch_index] == 2) ? 1 : 0;
                Q[ch_index] = Symmetry::IS_MODULAR[ch_index] ? util::constFct::posmod(2, Symmetry::MOD_N[ch_index]) : 2;
                this->basis_1s_.push_back(Q, 1);
                labels.insert(std::make_pair("double", std::make_pair(Q, dim)));
            }
            if(not REMOVE_SINGLE) {
                Q[ch_index] = 1;
                Q[sp_index] = 1;
                this->basis_1s_.push_back(Q, 1);
                labels.insert(std::make_pair("up", std::make_pair(Q, 0)));
                Q[sp_index] = Symmetry::IS_MODULAR[sp_index] ? util::constFct::posmod(-1, Symmetry::MOD_N[ch_index]) : -1;
                this->basis_1s_.push_back(Q, 1);
                labels.insert(std::make_pair("dn", std::make_pair(Q, 0)));
            }
        }
    }
}

template <typename Symmetry_>
typename Symmetry_::qType Fermion<Symmetry_>::getQ(SPIN_INDEX sigma, int Delta) const
{
    auto Q = Symmetry::qvacuum();
    if constexpr(Symmetry::ALL_IS_TRIVIAL) {
        return Q;
    } else {
        if(not HAS_SPIN and HAS_CHARGE) {
            if(sigma == SPIN_INDEX::UP or sigma == SPIN_INDEX::DN) {
                Q[ch_index] = Symmetry::IS_MODULAR[ch_index] ? util::constFct::posmod(Delta, Symmetry::MOD_N[ch_index]) : Delta;
                return Q;
            }
            if(sigma == SPIN_INDEX::UPDN) {
                Q[ch_index] = Symmetry::IS_MODULAR[ch_index] ? util::constFct::posmod(2 * Delta, Symmetry::MOD_N[ch_index]) : 2 * Delta;
                return Q;
            }
            return Q;
        } else if(HAS_SPIN and not HAS_CHARGE) {
            if(sigma == SPIN_INDEX::UP) {
                Q[sp_index] = Symmetry::IS_MODULAR[sp_index] ? util::constFct::posmod(Delta, Symmetry::MOD_N[sp_index]) : Delta;
                return Q;
            }
            if(sigma == SPIN_INDEX::DN) {
                Q[sp_index] = Symmetry::IS_MODULAR[sp_index] ? util::constFct::posmod(-Delta, Symmetry::MOD_N[sp_index]) : -Delta;
                return Q;
            }
            return Q;
        } else {
            if(sigma == SPIN_INDEX::UP) {
                Q[sp_index] = Symmetry::IS_MODULAR[sp_index] ? util::constFct::posmod(Delta, Symmetry::MOD_N[sp_index]) : Delta;
                Q[ch_index] = Symmetry::IS_MODULAR[ch_index] ? util::constFct::posmod(Delta, Symmetry::MOD_N[ch_index]) : Delta;
                return Q;
            }
            if(sigma == SPIN_INDEX::DN) {
                Q[sp_index] = Symmetry::IS_MODULAR[sp_index] ? util::constFct::posmod(-Delta, Symmetry::MOD_N[sp_index]) : -Delta;
                Q[ch_index] = Symmetry::IS_MODULAR[ch_index] ? util::constFct::posmod(Delta, Symmetry::MOD_N[ch_index]) : Delta;
                return Q;
            }
            if(sigma == SPIN_INDEX::UPDN) {
                Q[ch_index] = Symmetry::IS_MODULAR[ch_index] ? util::constFct::posmod(2 * Delta, Symmetry::MOD_N[ch_index]) : 2 * Delta;
                return Q;
            }
            return Q;
        }
    }
}

template <typename Symmetry_>
typename Symmetry_::qType Fermion<Symmetry_>::getQ(SPINOP_LABEL Sa) const
{
    auto Q = Symmetry::qvacuum();
    if constexpr(Symmetry::ALL_IS_TRIVIAL) {
        return Q;
    } else {
        if(HAS_SPIN) {
            assert(Sa != SPINOP_LABEL::SX and Sa != SPINOP_LABEL::iSY and "Sx and Sy break the U1 spin symmetry.");
            if(Sa == SPINOP_LABEL::SZ) {
                return Q;
            } else if(Sa == SPINOP_LABEL::SP) {
                Q[sp_index] = Symmetry::IS_MODULAR[sp_index] ? util::constFct::posmod(2, Symmetry::MOD_N[sp_index]) : 2;
            } else if(Sa == SPINOP_LABEL::SM) {
                Q[sp_index] = Symmetry::IS_MODULAR[sp_index] ? util::constFct::posmod(-2, Symmetry::MOD_N[sp_index]) : -2;
            }
            return Q;
        }
    }
    return Q;
}

} // namespace Xped
#endif // FERMIONSITE_H_
