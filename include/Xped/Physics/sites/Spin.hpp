#ifndef XPED_SPIN_HPP_
#define XPED_SPIN_HPP_

#include "Xped/Physics/SiteOperator.hpp"

// #include "Xped/Symmetry/S1xS2.hpp"
#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"

namespace Xped {

enum SPINOP_LABEL
{
    SZ,
    SP,
    SM,
    SX,
    SY,
    iSY,
    QZ,
    QPZ,
    QMZ,
    QP,
    QM
};

template <typename Symmetry_, std::size_t order = 0>
class Spin
{
    typedef double Scalar;
    typedef Symmetry_ Symmetry;
    using OperatorType = SiteOperator<Scalar, Symmetry>;
    using qType = typename Symmetry::qType;

public:
    Spin(){};
    Spin(std::size_t D_input);

    OperatorType Id_1s() const { return Id_1s_; }
    OperatorType Zero_1s() const { return Zero_1s_; }
    OperatorType F_1s() const { return F_1s_; }

    // OperatorType n_1s() const { return n_1s(UP) + n_1s(DN); }

    // dipole
    OperatorType Sz_1s() const { return Sz_1s_; }
    OperatorType Sp_1s() const { return Sp_1s_; }
    OperatorType Sm_1s() const { return Sm_1s_; }

    // quadrupole
    OperatorType Qz_1s() const { return Qz_1s_; }
    OperatorType Qp_1s() const { return Qp_1s_; }
    OperatorType Qm_1s() const { return Qm_1s_; }
    OperatorType Qpz_1s() const { return Qpz_1s_; }
    OperatorType Qmz_1s() const { return Qmz_1s_; }

    OperatorType exp_i_pi_Sx() const { return exp_i_pi_Sx_1s_; }
    OperatorType exp_i_pi_Sy() const { return exp_i_pi_Sy_1s_; }
    OperatorType exp_i_pi_Sz() const { return exp_i_pi_Sz_1s_; }

    Qbasis<Symmetry, 1> basis_1s() const { return basis_1s_; }

protected:
    std::size_t D;

    void fill_basis();
    void fill_SiteOps();

    /**Returns the quantum numbers of the operators for the different combinations of U1 symmetries.*/
    qType getQ(SPINOP_LABEL Sa) const;

    Qbasis<Symmetry, 1> basis_1s_;
    std::unordered_map<std::string, std::pair<qType, std::size_t>> labels;

    OperatorType Id_1s_; // identity
    OperatorType Zero_1s_; // zero
    OperatorType F_1s_; // fermionic sign

    OperatorType n_1s_; // particle number

    // orbital spin
    OperatorType Sz_1s_;
    OperatorType Sp_1s_;
    OperatorType Sm_1s_;

    // orbital quadrupole
    OperatorType Qz_1s_;
    OperatorType Qp_1s_;
    OperatorType Qm_1s_;
    OperatorType Qpz_1s_;
    OperatorType Qmz_1s_;

    OperatorType exp_i_pi_Sx_1s_;
    OperatorType exp_i_pi_Sy_1s_;
    OperatorType exp_i_pi_Sz_1s_;
};

template <typename Symmetry_, std::size_t order>
Spin<Symmetry_, order>::Spin(std::size_t D_input)
    : D(D_input)
{
    // create basis for one spin site
    fill_basis();

    // cout << "single site basis" << endl << this->basis_1s_ << endl;
    fill_SiteOps();
}

template <typename Symmetry_, std::size_t order>
void Spin<Symmetry_, order>::fill_SiteOps()
{
    Id_1s_ = OperatorType(Symmetry::qvacuum(), basis_1s_);
    Id_1s_.setIdentity();

    Zero_1s_ = OperatorType(Symmetry::qvacuum(), basis_1s_);
    Zero_1s_.setZero();

    F_1s_ = OperatorType(Symmetry::qvacuum(), basis_1s_);

    Sz_1s_ = OperatorType(getQ(SZ), basis_1s_);
    Sp_1s_ = OperatorType(getQ(SP), basis_1s_);
    Sm_1s_ = OperatorType(getQ(SM), basis_1s_);

    Qz_1s_ = OperatorType(getQ(SZ), basis_1s_);
    Qp_1s_ = OperatorType(getQ(QP), basis_1s_);
    Qm_1s_ = OperatorType(getQ(QM), basis_1s_);
    Qpz_1s_ = OperatorType(getQ(SP), basis_1s_);
    Qmz_1s_ = OperatorType(getQ(SM), basis_1s_);

    exp_i_pi_Sz_1s_ = OperatorType(getQ(SZ), basis_1s_);
    if constexpr(Symmetry::IS_TRIVIAL) {
        exp_i_pi_Sx_1s_ = OperatorType(getQ(SX), basis_1s_);
        exp_i_pi_Sy_1s_ = OperatorType(getQ(iSY), basis_1s_);
    }

    OperatorType Sbase = OperatorType(getQ(SP), basis_1s_, labels);
    Sbase.setZero();

    double S = 0.5 * (D - 1);
    std::size_t Sx2 = D - 1;

    for(std::size_t i = 0; i < D - 1; ++i) {
        int Q = -static_cast<int>(Sx2) + 2 * static_cast<int>(i);
        double m = -S + static_cast<double>(i);
        Sbase(std::to_string(Q + 2), std::to_string(Q)) = 0.5 * sqrt(S * (S + 1.) - m * (m + 1.));
    }

    Sm_1s_ = 2. * Sbase;
    Sp_1s_ = Sm_1s_.adjoint();
    Sz_1s_ = 0.5 * (Sp_1s_ * Sm_1s_ - Sm_1s_ * Sp_1s_);

    F_1s_ = 0.5 * Id_1s_ - Sz_1s_;

    //	cout << "Spin:" << endl;
    //	cout << "Id=" << endl << MatrixXd(Id_1s_.template plain<double>().data) << endl;
    //	cout << "Sbase=" << endl << MatrixXd(Sbase.template plain<double>().data) << endl;
    //	cout << "Sp=" << endl << MatrixXd(Sp_1s_.template plain<double>().data) << endl;
    //	cout << "Sm=" << endl << MatrixXd(Sm_1s_.template plain<double>().data) << endl;
    //	cout << "Sz=" << endl << MatrixXd(Sz_1s_.template plain<double>().data) << endl;
    //	cout << Id_1s_ << endl;

    Qz_1s_ = 1. / sqrt(3.) * (3. * Sz_1s_ * Sz_1s_ - S * (S + 1.) * Id_1s_);
    Qp_1s_ = Sp_1s_ * Sp_1s_;
    Qm_1s_ = Sm_1s_ * Sm_1s_;
    Qpz_1s_ = Sp_1s_ * Sz_1s_ + Sz_1s_ * Sp_1s_;
    Qmz_1s_ = Sm_1s_ * Sz_1s_ + Sz_1s_ * Sm_1s_;

    // if constexpr(Symmetry::IS_TRIVIAL) {
    //     // The exponentials are only correct for integer spin S=1,2,3,...!
    //     // for (size_t i=0; i<D; ++i) // <- don't want this basis order
    //     for(int i = D - 1; i >= 0; --i) {
    //         int Q1 = -static_cast<int>(Sx2) + 2 * static_cast<int>(i);
    //         int Q2 = +static_cast<int>(Sx2) - 2 * static_cast<int>(i);
    //         stringstream ssQ1;
    //         ssQ1 << Q1;
    //         stringstream ssQ2;
    //         ssQ2 << Q2;

    //         // exp(i*pi*Sx) has -1 on the antidiagonal for S=1,3,5,...
    //         // and +1 for S=2,4,6,...
    //         exp_i_pi_Sx_1s_(ssQ1.str(), ssQ2.str()) = pow(-1., D);

    //         // exp(i*pi*Sy) has alternating +-1 on the antidiagonal
    //         // starting with -1 for even D and with +1 for odd D
    //         exp_i_pi_Sy_1s_(ssQ1.str(), ssQ2.str()) = pow(-1., D + 1) * pow(-1, i);
    //     }
    // }

    // for(int i = D - 1; i >= 0; --i) {
    //     double m = -S + static_cast<double>(i);
    //     int Q = -static_cast<int>(Sx2) + 2 * static_cast<int>(i);
    //     stringstream ssQ;
    //     ssQ << Q;
    //     exp_i_pi_Sz_1s_(ssQ.str(), ssQ.str()) = pow(-1., m);
    // }

    return;
}

template <typename Symmetry_, std::size_t order>
void Spin<Symmetry_, order>::fill_basis()
{
    // double S = 0.5 * (D - 1);
    std::size_t Sx2 = D - 1;

    if constexpr(Symmetry::NO_SPIN_SYM()) // U0
    {
        basis_1s_.push_back(Symmetry::qvacuum(), D);
        // for(int i = D - 1; i >= 0; --i) {
        for(int i = 0; i < D; ++i) {
            int Qint = -static_cast<int>(Sx2) + 2 * static_cast<int>(i);
            labels.insert(std::make_pair(std::to_string(Qint), std::make_pair(Symmetry::qvacuum(), i)));
        }
    } else if constexpr(Symmetry::IS_SPIN_U1()) {
        typename Symmetry::qType Q;
        assert(D >= 1);
        // for(int i = D - 1; i >= 0; --i) {
        for(int i = 0; i < D; ++i) {
            int Qint = -static_cast<int>(Sx2) + 2 * static_cast<int>(i);
            if constexpr(Symmetry::Nq > 1) {
                for(std::size_t q = 0; q < Symmetry::Nq; q++) { Q[q] = (Symmetry::kind()[q] == Sym::KIND::M and q == order) ? Qint : 0; }
            } else {
                if constexpr(Symmetry::IS_MODULAR) {
                    Q[0] = util::constFct::posmod<Symmetry::MOD_N>(Qint);
                } else {
                    Q[0] = Qint;
                }
            }
            labels.insert(std::make_pair(std::to_string(Qint), std::make_pair(Q, 0)));
            basis_1s_.push_back(Q, 1);
        }
        basis_1s_.sort();
    }
}

template <typename Symmetry_, std::size_t order>
typename Symmetry_::qType Spin<Symmetry_, order>::getQ(SPINOP_LABEL Sa) const
{
    if constexpr(Symmetry::NO_SPIN_SYM()) {
        return Symmetry::qvacuum();
    } else if constexpr(Symmetry::Nq == 1) {
        if constexpr(Symmetry::kind()[0] == Sym::KIND::N or Symmetry::kind()[0] == Sym::KIND::Z2) // return particle number as good quantum number.
        {
            return Symmetry::qvacuum();
        } else if constexpr(Symmetry::kind()[0] == Sym::KIND::M) // return magnetization as good quantum number.
        {
            assert(Sa != SX and Sa != iSY);

            typename Symmetry::qType out;
            if(Sa == SZ or Sa == QZ) {
                out = {0};
            } else if(Sa == SP or Sa == QPZ) {
                out = {+2};
            } else if(Sa == SM or Sa == QMZ) {
                out = Symmetry::conj({+2});
            } else if(Sa == QP) {
                out = {+4};
            } else if(Sa == QM) {
                out = Symmetry::conj({+4});
            }
            return out;
        } else {
            assert(false and "Ill defined KIND of the used Symmetry.");
        }
    } else if constexpr(Symmetry::Nq == 2) {
        assert(Sa != SX and Sa != iSY and Sa != QP and Sa != QM);

        typename Symmetry::qType out;
        if constexpr(Symmetry::kind()[0] == Sym::KIND::N and Symmetry::kind()[1] == Sym::KIND::M) {
            if(Sa == SZ) {
                out = {0, 0};
            } else if(Sa == SP) {
                out = {0, +2};
            } else if(Sa == SM) {
                out = {0, -2};
            }
        } else if constexpr(Symmetry::kind()[0] == Sym::KIND::M and Symmetry::kind()[1] == Sym::KIND::N) {
            if(Sa == SZ) {
                out = {0, 0};
            } else if(Sa == SP) {
                out = {+2, 0};
            } else if(Sa == SM) {
                out = {-2, 0};
            }
        } else if constexpr(Symmetry::kind()[0] == Sym::KIND::Nup and Symmetry::kind()[1] == Sym::KIND::Ndn) {
            if(Sa == SZ) {
                out = {0, 0};
            } else if(Sa == SP) {
                out = {+1, -1};
            } else if(Sa == SM) {
                out = {-1, +1};
            }
        } else if constexpr(Symmetry::kind()[0] == Sym::KIND::Ndn and Symmetry::kind()[1] == Sym::KIND::Nup) {
            if(Sa == SZ) {
                out = {0, 0};
            } else if(Sa == SP) {
                out = {-1, +1};
            } else if(Sa == SM) {
                out = {+1, -1};
            }
        } else if constexpr(Symmetry::kind()[0] == Sym::KIND::M and Symmetry::kind()[1] == Sym::KIND::M) {
            if(order == 0ul) {
                if(Sa == SZ) {
                    out = {0, 0};
                } else if(Sa == SP) {
                    out = {+2, 0};
                } else if(Sa == SM) {
                    out = {-2, 0};
                }
            } else {
                if(Sa == SZ) {
                    out = {0, 0};
                } else if(Sa == SP) {
                    out = {0, +2};
                } else if(Sa == SM) {
                    out = {0, -2};
                }
            }
        }
        //		cout << "order=" << order << ", Sa=" << Sa << ", out=" << out << endl;
        return out;
    }
    static_assert("You inserted a Symmetry which can not be handled by SpinBase.");
}

} // namespace Xped

#endif
