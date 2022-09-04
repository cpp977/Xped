#ifndef XPED_FERMIONSSU2xU1_H_
#define XPED_FERMIONSSU2xU1_H_

#include "Xped/Physics/SiteOperator.hpp"
#include "Xped/Symmetry/S1xS2.hpp"
#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U1.hpp"
#include "Xped/Symmetry/kind_dummies.hpp"

namespace Xped {
template <typename Symmetry>
class Fermion;

template <typename Symmetry_>
class Fermion<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>, Symmetry_>>
{
    typedef double Scalar;
    typedef Sym::S1xS2<Sym::SU2<Sym::SpinSU2>, Symmetry_> Symmetry;
    typedef SiteOperator<double, Symmetry> OperatorType;
    using qType = typename Symmetry::qType;

public:
    Fermion(){};
    Fermion(bool REMOVE_DOUBLE, bool REMOVE_EMPTY, bool REMOVE_SINGLE, bool = true, bool CONSIDER_CHARGE = true);

    OperatorType Id_1s() const { return Id_1s_; }
    OperatorType F_1s() const { return F_1s_; }

    OperatorType c_1s() const { return c_1s_; }
    OperatorType cdag_1s() const { return cdag_1s_; }

    OperatorType n_1s() const { return n_1s_; }
    OperatorType ns_1s() const { return n_1s() - 2. * d_1s(); }
    OperatorType nh_1s() const { return 2. * d_1s() - n_1s() + Id_1s(); }
    OperatorType d_1s() const { return d_1s_; }

    OperatorType S_1s() const { return S_1s_; }

    OperatorType Tz_1s() const { return 0.5 * (n_1s() - Id_1s()); }
    OperatorType cc_1s() const { return p_1s_; }
    OperatorType cdagcdag_1s() const { return pdag_1s_; }

    Qbasis<Symmetry, 1> basis_1s() const { return basis_1s_; }

protected:
    void fill_basis(bool REMOVE_DOUBLE, bool REMOVE_EMPTY, bool REMOVE_SINGLE);
    Qbasis<Symmetry, 1> basis_1s_;

    std::unordered_map<std::string, std::pair<qType, std::size_t>> labels;
    std::size_t ch_index = 0;
    bool HAS_CHARGE;

    OperatorType Id_1s_; // identity
    OperatorType F_1s_; // Fermionic sign
    OperatorType c_1s_; // annihilation
    OperatorType cdag_1s_; // creation
    OperatorType n_1s_; // particle number
    OperatorType d_1s_; // double occupancy
    OperatorType S_1s_; // orbital spin
    OperatorType p_1s_; // pairing
    OperatorType pdag_1s_; // pairing adjoint
};

template <typename Symmetry_>
Fermion<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>, Symmetry_>>::Fermion(bool REMOVE_DOUBLE, bool REMOVE_EMPTY, bool REMOVE_SINGLE, bool, bool CONSIDER_CHARGE)
{
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

    fill_basis(REMOVE_DOUBLE, REMOVE_EMPTY, REMOVE_SINGLE);

    qType Q_c = Symmetry::qvacuum();
    Q_c[0] = 2;
    if constexpr(Symmetry::Nq > 1) { Q_c[1] = Symmetry::IS_MODULAR[1] ? util::constFct::posmod(-1, Symmetry::MOD_N[1]) : -1; }
    qType Q_S = Symmetry::qvacuum();
    Q_S[0] = 3;
    Id_1s_ = OperatorType(Symmetry::qvacuum(), basis_1s_, labels);
    Id_1s_.setIdentity();
    F_1s_ = OperatorType(Symmetry::qvacuum(), basis_1s_, labels);
    F_1s_.setIdentity();
    c_1s_ = OperatorType(Q_c, basis_1s_, labels);
    c_1s_.setZero();
    d_1s_ = OperatorType(Symmetry::qvacuum(), basis_1s_, labels);
    d_1s_.setZero();
    S_1s_ = OperatorType(Q_S, basis_1s_, labels);
    S_1s_.setZero();

    // create operators one orbitals

    if(not REMOVE_SINGLE) { F_1s_("single", "single") = -1.; }

    if(not REMOVE_EMPTY and not REMOVE_SINGLE) { c_1s_("empty", "single") = std::sqrt(2.); }
    if(not REMOVE_DOUBLE and not REMOVE_SINGLE) { c_1s_("single", "double") = 1.; }
    cdag_1s_ = c_1s_.adjoint();
    n_1s_ = std::sqrt(2.) * OperatorType::prod(cdag_1s_, c_1s_, Symmetry::qvacuum());
    if(not REMOVE_DOUBLE) { d_1s_("double", "double") = 1.; }
    if(not REMOVE_SINGLE) { S_1s_("single", "single") = std::sqrt(0.75); }
    qType Q_p = Symmetry::qvacuum();
    if constexpr(Symmetry::Nq > 1) { Q_p[1] = Symmetry::IS_MODULAR[1] ? util::constFct::posmod(-2, Symmetry::MOD_N[1]) : -2; }
    p_1s_ = -std::sqrt(0.5) * OperatorType::prod(c_1s_, c_1s_, Q_p); // The sign convention corresponds to c_DN c_UP
    pdag_1s_ = p_1s_.adjoint(); // The sign convention corresponds to (c_DN c_UP)†=c_UP† c_DN†
}

template <typename Symmetry_>
void Fermion<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>, Symmetry_>>::fill_basis(bool REMOVE_DOUBLE, bool REMOVE_EMPTY, bool REMOVE_SINGLE)
{
    if(not HAS_CHARGE) {
        qType Q = Symmetry::qvacuum();
        std::size_t dim = 0;
        if(not REMOVE_EMPTY) { labels.insert(std::make_pair("empty", std::make_pair(Q, dim++))); }
        if(not REMOVE_DOUBLE) { labels.insert(std::make_pair("double", std::make_pair(Q, dim++))); }
        this->basis_1s_.push_back(Q, dim);
        if(not REMOVE_SINGLE) {
            Q[0] = 2;
            this->basis_1s_.push_back(Q, 1);
            labels.insert(std::make_pair("single", std::make_pair(Q, 0)));
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
            Q[0] = 2;
            this->basis_1s_.push_back(Q, 1);
            labels.insert(std::make_pair("single", std::make_pair(Q, 0)));
        }
    }
}

} // namespace Xped
#endif // XPED_FERMIONSU2xU1_H_
