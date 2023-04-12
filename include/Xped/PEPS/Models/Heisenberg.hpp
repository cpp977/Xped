#ifndef XPED_PEPS_HEISENBERG_HPP_
#define XPED_PEPS_HEISENBERG_HPP_

#include <map>
#include <string>

#include <highfive/H5File.hpp>

#include "Xped/Core/Tensor.hpp"
#include "Xped/PEPS/Models/Helpers.hpp"
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
template <typename Symmetry>
class Heisenberg : public TwoSiteObservable<double, Symmetry>
{
public:
    Heisenberg(std::map<std::string, Param>& params_in, const Pattern& pat_in, Opts::Bond bond = Opts::Bond::H | Opts::Bond::V)
        : TwoSiteObservable<double, Symmetry>(pat_in, bond)
        , params(params_in)
        , pat(pat_in)
    {
        if constexpr(std::is_same_v<Symmetry, Sym::SU2<Sym::SpinSU2>>) {
            used_params = {"J"};
        } else {
            used_params = {"Jxy", "Jz"};
        }
        if((bond & Opts::Bond::D1) == Opts::Bond::D1 or (bond & Opts::Bond::D2) == Opts::Bond::D2) { used_params.push_back("J2"); }

        B = SpinBase<Symmetry>(1, 2);
        Tensor<double, 2, 2, Symmetry> gate;
        if constexpr(std::is_same_v<Symmetry, Sym::SU2<Sym::SpinSU2>>) {
            gate = params["J"].get<double>() * (std::sqrt(3.) * tprod(B.Sdag(0), B.S(0))).eval();
        } else if constexpr(Symmetry::ALL_ABELIAN) {
            gate = params["Jz"].get<double>() * tprod(B.Sz(), B.Sz()) +
                   0.5 * params["Jxy"].get<double>() * (tprod(B.Sp(), B.Sm()) + tprod(B.Sm(), B.Sp()));
        } else {
            assert(false and "Symmetry is not supported in Heisenberg model.");
        }

        if((bond & Opts::Bond::H) == Opts::Bond::H) {
            for(auto& t : this->data_h) { t = gate; }
        }
        if((bond & Opts::Bond::V) == Opts::Bond::V) {
            for(auto& t : this->data_v) { t = gate; }
        }
        if((bond & Opts::Bond::D1) == Opts::Bond::D1) {
            for(auto& t : this->data_d1) { t = params["J2"].get<double>() * gate; }
        }
        if((bond & Opts::Bond::D2) == Opts::Bond::D2) {
            for(auto& t : this->data_d2) { t = params["J2"].get<double>() * gate; }
        }
    }

    virtual void setDefaultObs() override
    {
        if constexpr(std::is_same_v<Symmetry, Sym::SU2<Sym::SpinSU2>>) {
            auto SS = std::make_unique<Xped::TwoSiteObservable<double, Symmetry>>(
                pat, Xped::Opts::Bond::H | Xped::Opts::Bond::V | Xped::Opts::Bond::D1 | Xped::Opts::Bond::D2, "SS");
            for(auto& t : SS->data_h) { t = std::sqrt(3.) * Xped::tprod(B.Sdag(), B.S()); }
            for(auto& t : SS->data_v) { t = std::sqrt(3.) * Xped::tprod(B.Sdag(), B.S()); }
            for(auto& t : SS->data_d1) { t = std::sqrt(3.) * Xped::tprod(B.Sdag(), B.S()); }
            for(auto& t : SS->data_d2) { t = std::sqrt(3.) * Xped::tprod(B.Sdag(), B.S()); }
            obs.push_back(std::move(SS));

        } else if constexpr(Symmetry::ALL_ABELIAN) {
            auto Sz = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(pat, "Sz");
            for(auto& t : Sz->data) { t = B.Sz().data.template trim<2>(); }
            obs.push_back(std::move(Sz));
            if constexpr(Symmetry::ALL_IS_TRIVIAL) {
                auto Sx = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(pat, "Sx");
                for(auto& t : Sx->data) { t = B.Sx().data.template trim<2>(); }
                obs.push_back(std::move(Sx));
            }
            auto SzSz = std::make_unique<Xped::TwoSiteObservable<double, Symmetry>>(
                pat, Xped::Opts::Bond::H | Xped::Opts::Bond::V | Xped::Opts::Bond::D1 | Xped::Opts::Bond::D2, "SzSz");
            for(auto& t : SzSz->data_h) { t = Xped::tprod(B.Sz(), B.Sz()); }
            for(auto& t : SzSz->data_v) { t = Xped::tprod(B.Sz(), B.Sz()); }
            for(auto& t : SzSz->data_d1) { t = Xped::tprod(B.Sz(), B.Sz()); }
            for(auto& t : SzSz->data_d2) { t = Xped::tprod(B.Sz(), B.Sz()); }
            obs.push_back(std::move(SzSz));
            if constexpr(not Symmetry::ANY_IS_SPIN) {
                auto Sx = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(pat, "Sx");
                for(auto& t : Sx->data) { t = B.Sx().data.template trim<2>(); }

                auto SxSx = std::make_unique<TwoSiteObservable<double, Symmetry>>(
                    pat, Opts::Bond::H | Opts::Bond::V | Opts::Bond::D1 | Opts::Bond::D2, "SxSx");
                for(auto& t : SxSx->data_h) { t = tprod(B.Sx(), B.Sx()); }
                for(auto& t : SxSx->data_v) { t = tprod(B.Sx(), B.Sx()); }
                for(auto& t : SxSx->data_d1) { t = tprod(B.Sx(), B.Sx()); }
                for(auto& t : SxSx->data_d2) { t = tprod(B.Sx(), B.Sx()); }
                obs.push_back(std::move(Sx));
                obs.push_back(std::move(SxSx));
                auto Sy = std::make_unique<Xped::OneSiteObservable<std::complex<double>, Symmetry>>(pat, "Sy");
                for(auto& t : Sy->data) { t = B.Sy().data.template trim<2>(); }
                obs.push_back(std::move(Sy));
            }
        }
    }

    virtual std::string file_name() const override { return internal::create_filename("Heisenberg", params, used_params); }

    virtual std::string format() const override
    {
        return internal::format_params(fmt::format("Heisenberg[sym={}]", Symmetry::name()), params, used_params);
    }

    virtual void computeObs(XPED_CONST CTM<double, Symmetry, 2, false, Opts::CTMCheckpoint{}>& env) override
    {
        for(auto& ob : obs) {
            if(auto* one = dynamic_cast<OneSiteObservable<double, Symmetry>*>(ob.get()); one != nullptr) { avg(env, *one); }
            if(auto* one_c = dynamic_cast<OneSiteObservable<std::complex<double>, Symmetry>*>(ob.get()); one_c != nullptr) { avg(env, *one_c); }
            if(auto* two = dynamic_cast<TwoSiteObservable<double, Symmetry>*>(ob.get()); two != nullptr) { avg(env, *two); }
        }
    }

    virtual void computeObs(XPED_CONST CTM<std::complex<double>, Symmetry, 2, false, Opts::CTMCheckpoint{}>& env) override
    {
        for(auto& ob : obs) {
            if(auto* one = dynamic_cast<OneSiteObservable<double, Symmetry>*>(ob.get()); one != nullptr) { avg(env, *one); }
            if(auto* one_c = dynamic_cast<OneSiteObservable<std::complex<double>, Symmetry>*>(ob.get()); one_c != nullptr) { avg(env, *one_c); }
            if(auto* two = dynamic_cast<TwoSiteObservable<double, Symmetry>*>(ob.get()); two != nullptr) { avg(env, *two); }
        }
    }

    virtual void computeObs(XPED_CONST CTM<double, Symmetry, 1, false, Opts::CTMCheckpoint{}>& env) override
    {
        for(auto& ob : obs) {
            if(auto* one = dynamic_cast<OneSiteObservable<double, Symmetry>*>(ob.get()); one != nullptr) { avg(env, *one); }
            if(auto* one_c = dynamic_cast<OneSiteObservable<std::complex<double>, Symmetry>*>(ob.get()); one_c != nullptr) { avg(env, *one_c); }
            if(auto* two = dynamic_cast<TwoSiteObservable<double, Symmetry>*>(ob.get()); two != nullptr) { avg(env, *two); }
        }
    }

    virtual std::string getObsString(const std::string& offset) const override
    {
        std::string out;
        for(const auto& ob : obs) {
            out.append(ob->getResString(offset));
            if(&ob != &obs.back()) { out.push_back('\n'); }
        }
        return out;
    }

    virtual void obsToFile(HighFive::File& file, const std::string& root = "/") const override
    {
        for(const auto& ob : obs) { ob->toFile(file, root); }
    }

    std::map<std::string, Param> params;
    Pattern pat;
    std::vector<std::string> used_params;
    SpinBase<Symmetry> B;
    std::vector<std::unique_ptr<ObservableBase>> obs;
};

} // namespace Xped
#endif
