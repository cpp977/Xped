#ifndef XPED_SPINLESS_FERMIONSITE_HPP_
#define XPED_SPINLESS_FERMIONSITE_HPP_

#include "fmt/ostream.h"
#include "fmt/ranges.h"
#include "fmt/std.h"

#include "Xped/Physics/SiteOperator.hpp"
#include "Xped/Physics/SpinIndex.hpp"
#include "Xped/Physics/SpinOp.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Util/Constfct.hpp"

namespace Xped {

template <typename Symmetry_>
class SpinlessFermion
{
    using Scalar = double;
    using Symmetry = Symmetry_;
    using OperatorType = SiteOperator<Scalar, Symmetry>;
    using qType = typename Symmetry::qType;

public:
    explicit SpinlessFermion(bool CONSIDER_SPIN = true, bool CONSIDER_CHARGE = true);

    OperatorType Id_1s() const { return Id_1s_; }
    OperatorType F_1s() const { return F_1s_; }

    OperatorType c_1s() const { return c_1s_; }
    OperatorType cdag_1s() const { return cdag_1s_; }

    OperatorType n_1s() const { return n_1s_; }

    Qbasis<Symmetry, 1> basis_1s() const { return basis_1s_; }

protected:
    void fill_basis();
    void fill_SiteOps();

    qType getQ(int Delta) const;

    Qbasis<Symmetry, 1> basis_1s_;
    std::unordered_map<std::string, std::pair<qType, std::size_t>> labels;

    std::size_t sp_index = 0;
    std::size_t ch_index = 0;

    bool HAS_SPIN;
    bool HAS_CHARGE;

    OperatorType Id_1s_; // identity
    OperatorType F_1s_; // Fermionic sign

    OperatorType c_1s_; // annihilation

    OperatorType cdag_1s_; // creation

    OperatorType n_1s_; // particle number
};

template <typename Symmetry_>
SpinlessFermion<Symmetry_>::SpinlessFermion(bool CONSIDER_SPIN, bool CONSIDER_CHARGE)
{
    HAS_SPIN = false;
    if(CONSIDER_SPIN) {
        for(std::size_t q = 0; q < Symmetry::Nq; ++q) {
            if(Symmetry::IS_SPIN[q]) {
                sp_index = q;
                HAS_SPIN = true;
                break;
            }
        }
    }

    HAS_CHARGE = false;
    if(CONSIDER_CHARGE) {
        for(std::size_t q = 0; q < Symmetry::Nq; ++q) {
            if(Symmetry::IS_FERMIONIC[q] or Symmetry::IS_BOSONIC[q]) {
                ch_index = q;
                HAS_CHARGE = true;
                break;
            }
        }
    }
    fmt::print("HAS_SPIN={}, HAS_CHARGE={}\n", HAS_SPIN, HAS_CHARGE);
    // create basis for one Fermionic Site
    fill_basis();
    for(auto [label, value] : labels) { fmt::print("label={}: Q={}, deg={}\n", label, fmt::streamed(value.first), value.second); }
    basis_1s_.sort();
    fmt::print("{}\n", basis_1s_.print());
    fill_SiteOps();
}

template <typename Symmetry_>
void SpinlessFermion<Symmetry_>::fill_SiteOps()
{
    // create operators for one site
    Id_1s_ = OperatorType(Symmetry::qvacuum(), basis_1s_, labels);
    F_1s_ = OperatorType(Symmetry::qvacuum(), basis_1s_, labels);

    c_1s_ = OperatorType(getQ(-1), basis_1s_, labels);
    c_1s_.setZero();

    cdag_1s_ = OperatorType(getQ(+1), basis_1s_, labels);

    n_1s_ = OperatorType(Symmetry::qvacuum(), basis_1s_, labels);

    Id_1s_.setIdentity();

    F_1s_.setZero();

    F_1s_("empty", "empty") = 1.;
    F_1s_("occ", "occ") = -1.;

    c_1s_("empty", "occ") = 1.;

    cdag_1s_ = c_1s_.adjoint();

    // std::cout << std::fixed << cdagup_1s_ << std::endl
    //           << cdagup_1s_.plain()[0] << std::endl
    //           << cup_1s_ << std::endl
    //           << cup_1s_.plain()[0] << std::endl;

    n_1s_ = cdag_1s_ * c_1s_;
}

template <typename Symmetry_>
void SpinlessFermion<Symmetry_>::fill_basis()
{
    if constexpr(Symmetry::ALL_IS_TRIVIAL) // U0
    {
        qType Q = Symmetry::qvacuum();
        std::size_t dim = 0;
        labels.insert(std::make_pair("empty", std::make_pair(Q, dim++)));
        labels.insert(std::make_pair("occ", std::make_pair(Q, dim++)));
        this->basis_1s_.push_back(Q, dim);
        return;
    } else {
        if(not HAS_SPIN and HAS_CHARGE) {
            qType Q = Symmetry::qvacuum();
            this->basis_1s_.push_back(Q, 1);
            labels.insert(std::make_pair("empty", std::make_pair(Q, 0)));
            Q[ch_index] = 1;
            this->basis_1s_.push_back(Q, 1);
            labels.insert(std::make_pair("occ", std::make_pair(Q, 0)));
        } else if(HAS_SPIN and not HAS_CHARGE) {
            qType Q = Symmetry::qvacuum();
            std::size_t dim = 0;
            labels.insert(std::make_pair("empty", std::make_pair(Q, dim++)));
            labels.insert(std::make_pair("occ", std::make_pair(Q, dim++)));
            this->basis_1s_.push_back(Q, dim);
        } else {
            qType Q = Symmetry::qvacuum();
            this->basis_1s_.push_back(Q, 1);
            labels.insert(std::make_pair("empty", std::make_pair(Q, 0)));
            Q[ch_index] = 1;
            this->basis_1s_.push_back(Q, 1);
            labels.insert(std::make_pair("occ", std::make_pair(Q, 0)));
        }
    }
}

template <typename Symmetry_>
typename Symmetry_::qType SpinlessFermion<Symmetry_>::getQ(int Delta) const
{
    auto Q = Symmetry::qvacuum();
    if constexpr(Symmetry::ALL_IS_TRIVIAL) {
        return Q;
    } else {
        if(HAS_CHARGE) {
            Q[ch_index] = Symmetry::IS_MODULAR[ch_index] ? util::constFct::posmod(Delta, Symmetry::MOD_N[ch_index]) : Delta;
            return Q;
        } else {
            return Q;
        }
    }
}

} // namespace Xped
#endif // FERMIONSITE_H_
