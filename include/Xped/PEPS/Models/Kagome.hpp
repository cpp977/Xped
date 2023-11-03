#ifndef XPED_PEPS_KAGOME_HPP_
#define XPED_PEPS_KAGOME_HPP_

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

/*
  gs energy:
   square: E/N = -0.66944
   triang: E/N = −0.5445
     j1j2: E/N =
*/
template <typename Symmetry, typename Scalar = double>
class Kagome : public Hamiltonian<Scalar, Symmetry>
{
public:
    Kagome(std::map<std::string, Param>& params_in, const Pattern& pat_in, Opts::Bond bond_in = Opts::Bond::H | Opts::Bond::V)
        : Hamiltonian<Scalar, Symmetry>(params_in, pat_in, bond_in, "Kagome")
    {
        if constexpr(std::is_same_v<Symmetry, Sym::SU2<Sym::SpinSU2>>) {
            this->used_params = {"J"};
        } else {
            this->used_params = {"Jxy", "Jz", "Bz"};
        }

        B = SpinBase<Symmetry>(3, 2);

        Tensor<double, 2, 2, Symmetry> gate_nnh, gate_nnv, local;
        if constexpr(std::is_same_v<Symmetry, Sym::SU2<Sym::SpinSU2>>) {
            local = 0.25 * this->params["J"].template get<double>() * std::sqrt(3.) * (B.Sdag(0) * B.S(1) + B.Sdag(1) * B.S(2));
            gate_nnh = this->params["J"].template get<double>() * std::sqrt(3.) * (tprod(B.Sdag(2), B.S(0)) + tprod(B.Sdag(2), B.S(1)));
            gate_nnv = this->params["J"].template get<double>() * std::sqrt(3.) * (tprod(B.Sdag(1), B.S(0)) + tprod(B.Sdag(2), B.S(0)));
        } else if constexpr(Symmetry::ALL_ABELIAN) {
            auto SdotS = [this](auto i, auto j) {
                return (tprod(B.Sz(i), B.Sz(j)) + 0.5 * (tprod(B.Sp(i), B.Sm(j)) + tprod(B.Sm(i), B.Sp(j)))).eval();
            };
            auto SdotSloc = [this](auto i, auto j) { return B.Sz(i) * B.Sz(j) + 0.5 * (B.Sp(i) * B.Sm(j) + B.Sm(i) * B.Sp(j)); };
            auto local_op = this->params["J"].template get<double>() * (SdotSloc(0, 1) + SdotSloc(1, 2));
            local = 0.25 * (tprod(local_op, B.Id()) + tprod(B.Id(), local_op));
            gate_nnh = this->params["J"].template get<double>() * (SdotS(2, 0) + SdotS(2, 1));
            gate_nnv = this->params["J"].template get<double>() * (SdotS(1, 0) + SdotS(2, 0));
        } else {
            assert(false and "Symmetry is not supported in Kagome model.");
        }

        if((this->bond & Opts::Bond::H) == Opts::Bond::H) {
            for(auto& t : this->data_h) { t = gate_nnh + local; }
        }
        if((this->bond & Opts::Bond::V) == Opts::Bond::V) {
            for(auto& t : this->data_v) { t = gate_nnv + local; }
        }
        // if((this->bond & Opts::Bond::D1) == Opts::Bond::D1) {
        //     for(auto& t : this->data_d1) { t = gate_d; }
        // }
        // if((this->bond & Opts::Bond::D2) == Opts::Bond::D2) {
        //     for(auto& t : this->data_d2) { t = gate_d; }
        // }
    }

    virtual void setDefaultObs() override
    {
        if constexpr(std::is_same_v<Symmetry, Sym::SU2<Sym::SpinSU2>>) {
            return;
        } else if constexpr(Symmetry::ALL_ABELIAN) {
            auto Sz0 = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(this->pat, "Sz0");
            for(auto& t : Sz0->data) { t = B.Sz(0).data.template trim<2>(); }
            auto Sz1 = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(this->pat, "Sz1");
            for(auto& t : Sz1->data) { t = B.Sz(1).data.template trim<2>(); }
            auto Sz2 = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(this->pat, "Sz2");
            for(auto& t : Sz2->data) { t = B.Sz(2).data.template trim<2>(); }
            this->obs.push_back(std::move(Sz0));
            this->obs.push_back(std::move(Sz1));
            this->obs.push_back(std::move(Sz2));
            if constexpr(not Symmetry::ANY_IS_SPIN) {
                auto Sx0 = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(this->pat, "Sx0");
                for(auto& t : Sx0->data) { t = B.Sx(0).data.template trim<2>(); }
                auto Sx1 = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(this->pat, "Sx1");
                for(auto& t : Sx1->data) { t = B.Sx(1).data.template trim<2>(); }
                auto Sx2 = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(this->pat, "Sx2");
                for(auto& t : Sx2->data) { t = B.Sx(2).data.template trim<2>(); }
                this->obs.push_back(std::move(Sx0));
                this->obs.push_back(std::move(Sx1));
                this->obs.push_back(std::move(Sx2));
            }
            return;
        }
    }

    SpinBase<Symmetry> B;
};

} // namespace Xped
#endif
