#ifndef XPED_PEPS_HEISENBERG_HPP_
#define XPED_PEPS_HEISENBERG_HPP_

#include <map>
#include <string>

#include <highfive/H5File.hpp>

#include "Xped/Core/Tensor.hpp"
#include "Xped/PEPS/Models/Hamiltonian.hpp"
#include "Xped/PEPS/OneSiteObservable.hpp"
#include "Xped/PEPS/TwoSiteObservable.hpp"
#include "Xped/Physics/SpinBase.hpp"
#include "Xped/Util/Param.hpp"

namespace Xped {

/* gs energy:
   square: E/N = -0.66944
   triang: E/N = −0.5445
     j1j2: E/N =
*/
template <typename Symmetry, typename Scalar = double>
class Heisenberg : public Hamiltonian<Scalar, Symmetry>
{
public:
    Heisenberg(std::map<std::string, Param>& params_in, const Pattern& pat_in, Opts::Bond bond_in = Opts::Bond::H | Opts::Bond::V)
        : Hamiltonian<Scalar, Symmetry>(params_in, pat_in, bond_in, "Heisenberg")
    {
        if constexpr(std::is_same_v<Symmetry, Sym::SU2<Sym::SpinSU2>>) {
            this->used_params = {"J"};
        } else {
            this->used_params = {"Jxy", "Jz", "Bz"};
        }
        if((this->bond & Opts::Bond::D1) == Opts::Bond::D1 or (this->bond & Opts::Bond::D2) == Opts::Bond::D2) { this->used_params.push_back("J2"); }

        B = SpinBase<Symmetry>(1, 2);

        Tensor<double, 2, 2, Symmetry> gate_nn, gate_d, gate_nnv;
        if constexpr(std::is_same_v<Symmetry, Sym::SU2<Sym::SpinSU2>>) {
            gate_nn = this->params["J"].template get<double>() * (std::sqrt(3.) * tprod(B.Sdag(0), B.S(0, false))).eval();
            gate_nnv = this->params["J"].template get<double>() * (std::sqrt(3.) * tprod(B.Sdag(0, false), B.S(0))).eval();
            gate_d = this->params["J2"].template get<double>() * (std::sqrt(3.) * tprod(B.Sdag(0), B.S(0))).eval();
        } else if constexpr(Symmetry::ALL_ABELIAN) {
            gate_d = this->params["J2"].template get<double>() * tprod(B.Sz(), B.Sz()) +
                     0.5 * this->params["J2"].template get<double>() * (tprod(B.Sp(), B.Sm()) + tprod(B.Sm(), B.Sp()));
            bool SUBL = true;
            gate_nn = this->params["Jz"].template get<double>() * tprod(B.Sz(), B.Sz(0, SUBL)) +
                      0.5 * this->params["Jxy"].template get<double>() * (tprod(B.Sp(), B.Sm(0, SUBL)) + tprod(B.Sm(), B.Sp(0, SUBL)));
            gate_nnv = this->params["Jz"].template get<double>() * tprod(B.Sz(0, SUBL), B.Sz()) +
                       0.5 * this->params["Jxy"].template get<double>() * (tprod(B.Sp(0, SUBL), B.Sm()) + tprod(B.Sm(0, SUBL), B.Sp()));
        } else {
            assert(false and "Symmetry is not supported in Heisenberg model.");
        }

        if((this->bond & Opts::Bond::H) == Opts::Bond::H) {
            for(auto& t : this->data_h) { t = gate_nn; }
        }
        if((this->bond & Opts::Bond::V) == Opts::Bond::V) {
            for(auto& t : this->data_v) { t = gate_nnv; }
        }
        if((this->bond & Opts::Bond::D1) == Opts::Bond::D1) {
            for(auto& t : this->data_d1) { t = gate_d; }
        }
        if((this->bond & Opts::Bond::D2) == Opts::Bond::D2) {
            for(auto& t : this->data_d2) { t = gate_d; }
        }
    }

    virtual void setDefaultObs() override
    {
        if constexpr(std::is_same_v<Symmetry, Sym::SU2<Sym::SpinSU2>>) {
            auto SS = std::make_unique<Xped::TwoSiteObservable<double, Symmetry>>(this->pat, Xped::Opts::Bond::H, "SS");
            for(auto& t : SS->data_h) { t = std::sqrt(3.) * Xped::tprod(B.Sdag(), B.S()); }
            this->obs.push_back(std::move(SS));
            return;

            // auto SS = std::make_unique<Xped::TwoSiteObservable<double, Symmetry>>(
            //     this->pat, Xped::Opts::Bond::H | Xped::Opts::Bond::V | Xped::Opts::Bond::D1 | Xped::Opts::Bond::D2, "SS");
            // for(auto& t : SS->data_h) { t = std::sqrt(3.) * Xped::tprod(B.Sdag(), B.S()); }
            // for(auto& t : SS->data_v) { t = std::sqrt(3.) * Xped::tprod(B.Sdag(), B.S()); }
            // for(auto& t : SS->data_d1) { t = std::sqrt(3.) * Xped::tprod(B.Sdag(), B.S()); }
            // for(auto& t : SS->data_d2) { t = std::sqrt(3.) * Xped::tprod(B.Sdag(), B.S()); }
            // this->obs.push_back(std::move(SS));

        } else if constexpr(Symmetry::ALL_ABELIAN) {
            auto Sz = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(this->pat, "Sz");
            for(auto& t : Sz->data) { t = B.Sz().data.template trim<2>(); }
            this->obs.push_back(std::move(Sz));
            if constexpr(Symmetry::ALL_IS_TRIVIAL) {
                auto Sx = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(this->pat, "Sx");
                for(auto& t : Sx->data) { t = B.Sx().data.template trim<2>(); }
                this->obs.push_back(std::move(Sx));
            }
            auto SzSz = std::make_unique<Xped::TwoSiteObservable<double, Symmetry>>(
                this->pat, Xped::Opts::Bond::H | Xped::Opts::Bond::V | Xped::Opts::Bond::D1 | Xped::Opts::Bond::D2, "SzSz");
            for(auto& t : SzSz->data_h) { t = Xped::tprod(B.Sz(), B.Sz()); }
            for(auto& t : SzSz->data_v) { t = Xped::tprod(B.Sz(), B.Sz()); }
            for(auto& t : SzSz->data_d1) { t = Xped::tprod(B.Sz(), B.Sz()); }
            for(auto& t : SzSz->data_d2) { t = Xped::tprod(B.Sz(), B.Sz()); }
            this->obs.push_back(std::move(SzSz));
            if constexpr(not Symmetry::ANY_IS_SPIN) {
                auto SxSx = std::make_unique<TwoSiteObservable<double, Symmetry>>(
                    this->pat, Opts::Bond::H | Opts::Bond::V | Opts::Bond::D1 | Opts::Bond::D2, "SxSx");
                for(auto& t : SxSx->data_h) { t = tprod(B.Sx(), B.Sx()); }
                for(auto& t : SxSx->data_v) { t = tprod(B.Sx(), B.Sx()); }
                for(auto& t : SxSx->data_d1) { t = tprod(B.Sx(), B.Sx()); }
                for(auto& t : SxSx->data_d2) { t = tprod(B.Sx(), B.Sx()); }
                this->obs.push_back(std::move(SxSx));
                auto Sy = std::make_unique<Xped::OneSiteObservable<std::complex<double>, Symmetry>>(this->pat, "Sy");
                for(auto& t : Sy->data) { t = B.Sy().data.template trim<2>(); }
                this->obs.push_back(std::move(Sy));
            }
        }
    }

    SpinBase<Symmetry> B;
};

} // namespace Xped
#endif
