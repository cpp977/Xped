#ifndef XPED_FERMIONSSU2xSU2xU1_H_
#define XPED_FERMIONSSU2xSU2xU1_H_

#include "Xped/Physics/SiteOperator.hpp"
#include "Xped/Symmetry/CombSym.hpp"
#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U1.hpp"
#include "Xped/Symmetry/kind_dummies.hpp"

namespace Xped {
template <typename Symmetry>
class Fermion;

template <typename Symmetry_>
class Fermion<Sym::Combined<Sym::SU2<Sym::SpinSU2>, Sym::SU2<Sym::SpinSU2>, Symmetry_>>
{
    using Scalar = double;
    using Symmetry = Sym::Combined<Sym::SU2<Sym::SpinSU2>, Sym::SU2<Sym::SpinSU2>, Symmetry_>;
    using OperatorType = SiteOperator<double, Symmetry>;
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
    void fill_basis();
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
Fermion<Sym::Combined<Sym::SU2<Sym::SpinSU2>, Sym::SU2<Sym::SpinSU2>, Symmetry_>>::Fermion(bool, bool, bool, bool, bool CONSIDER_CHARGE)
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

    fill_basis();
    basis_1s_.sort();

    fmt::print("labels={}\n", labels);
    std::cout << basis_1s_.info() << std::endl << basis_1s_ << std::endl;

    qType Q_c = Symmetry::qvacuum();
    Q_c[0] = 2;
    Q_c[1] = 2;
    if constexpr(Symmetry::Nq > 2) { Q_c[2] = Symmetry::IS_MODULAR[2] ? util::constFct::posmod(-1, Symmetry::MOD_N[2]) : -1; }
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

    F_1s_("single", "single") = -1.;

    c_1s_("empty", "single") = std::sqrt(2.) * std::sqrt(2.);
    c_1s_("single", "spin_double") = std::sqrt(1.5);
    c_1s_("single", "orbital_double") = std::sqrt(1.5);
    c_1s_("spin_double", "triple") = -std::sqrt(2);
    c_1s_("orbital_double", "triple") = std::sqrt(2);
    c_1s_("triple", "quadrupel") = 1.;
    cdag_1s_ = c_1s_.adjoint();
    n_1s_ = std::sqrt(2.) * std::sqrt(2.) * OperatorType::prod(cdag_1s_, c_1s_, Symmetry::qvacuum());
    // S_1s_("single", "single") = std::sqrt(0.75);
    qType Q_p = Symmetry::qvacuum();
    if constexpr(Symmetry::Nq > 2) { Q_p[2] = Symmetry::IS_MODULAR[2] ? util::constFct::posmod(-2, Symmetry::MOD_N[2]) : -2; }
    p_1s_ = -std::sqrt(0.5) * OperatorType::prod(c_1s_, c_1s_, Q_p); // The sign convention corresponds to c_DN c_UP
    pdag_1s_ = p_1s_.adjoint(); // The sign convention corresponds to (c_DN c_UP)†=c_UP† c_DN†
}

template <typename Symmetry_>
void Fermion<Sym::Combined<Sym::SU2<Sym::SpinSU2>, Sym::SU2<Sym::SpinSU2>, Symmetry_>>::fill_basis()
{
    if(not HAS_CHARGE) {
        qType Q = Symmetry::qvacuum();
        labels.insert(std::make_pair("empty", std::make_pair(Q, 0)));
        labels.insert(std::make_pair("quadruple", std::make_pair(Q, 1)));
        this->basis_1s_.push_back(Q, 2);
        Q[0] = 2;
        Q[1] = 2;
        this->basis_1s_.push_back(Q, 2);
        labels.insert(std::make_pair("single", std::make_pair(Q, 0)));
        labels.insert(std::make_pair("triple", std::make_pair(Q, 1)));
        Q[0] = 3;
        Q[1] = 1;
        this->basis_1s_.push_back(Q, 1);
        labels.insert(std::make_pair("spin_double", std::make_pair(Q, 0)));
        Q[0] = 1;
        Q[1] = 3;
        this->basis_1s_.push_back(Q, 1);
        labels.insert(std::make_pair("orbital_double", std::make_pair(Q, 0)));
    } else {
        qType Q = Symmetry::qvacuum();
        this->basis_1s_.push_back(Q, 1);
        labels.insert(std::make_pair("empty", std::make_pair(Q, 0)));
        std::size_t dim = (Symmetry::MOD_N[ch_index] == 2 or Symmetry::MOD_N[ch_index] == 4) ? 1 : 0;
        Q[ch_index] = Symmetry::IS_MODULAR[ch_index] ? util::constFct::posmod(4, Symmetry::MOD_N[ch_index]) : 4;
        this->basis_1s_.push_back(Q, 1);
        labels.insert(std::make_pair("quadrupel", std::make_pair(Q, dim)));

        Q[0] = 2;
        Q[1] = 2;
        Q[ch_index] = 1;
        this->basis_1s_.push_back(Q, 1);
        labels.insert(std::make_pair("single", std::make_pair(Q, 0)));

        Q[ch_index] = Symmetry::IS_MODULAR[ch_index] ? util::constFct::posmod(3, Symmetry::MOD_N[ch_index]) : 3;
        this->basis_1s_.push_back(Q, 1);
        dim = (Symmetry::MOD_N[ch_index] == 2) ? 1 : 0;
        labels.insert(std::make_pair("triple", std::make_pair(Q, dim)));

        Q[0] = 3;
        Q[1] = 1;
        Q[ch_index] = Symmetry::IS_MODULAR[ch_index] ? util::constFct::posmod(2, Symmetry::MOD_N[ch_index]) : 2;
        this->basis_1s_.push_back(Q, 1);
        labels.insert(std::make_pair("spin_double", std::make_pair(Q, 0)));
        Q[0] = 1;
        Q[1] = 3;
        this->basis_1s_.push_back(Q, 1);
        labels.insert(std::make_pair("orbital_double", std::make_pair(Q, 0)));
    }
}

} // namespace Xped
#endif // XPED_FERMIONSU2xU1_H_
