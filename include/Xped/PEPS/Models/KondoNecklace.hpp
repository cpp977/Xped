#ifndef XPED_KONDO_NECKLACE_HPP_
#define XPED_KONDO_NECKLACE_HPP_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <highfive/H5File.hpp>

#include "Xped/Core/Tensor.hpp"
#include "Xped/PEPS/Models/Helpers.hpp"
#include "Xped/PEPS/OneSiteObservable.hpp"
#include "Xped/PEPS/TwoSiteObservable.hpp"
#include "Xped/Physics/SpinBase.hpp"
#include "Xped/Util/Param.hpp"

namespace Xped {

template <typename Symmetry>
class KondoNecklace : public TwoSiteObservable<Symmetry>
{
public:
    KondoNecklace(std::map<std::string, Param>& params_in, const Pattern& pat_in, Opts::Bond bond = Opts::Bond::H | Opts::Bond::V)
        : TwoSiteObservable<Symmetry>(pat, bond)
        , params(params_in)
        , pat(pat_in)
    {
        B = SpinBase<Symmetry>(2, 2);
        Tensor<double, 2, 2, Symmetry> gate, bond_gate;
        if constexpr(std::is_same_v<Symmetry, Sym::SU2<Sym::SpinSU2>>) {
            used_params = {"J", "Jk", "I"};
        } else {
            used_params = {"Jxy", "Jz", "Jkxy", "Jz", "Ixy", "Iz"};
        }

        if constexpr(std::is_same_v<Symmetry, Sym::SU2<Sym::SpinSU2>>) {
            auto SS = (std::sqrt(3.) * tprod(B.Sdag(0), B.S(0))).eval();

            auto ss = (std::sqrt(3.) * tprod(B.Sdag(1), B.S(1))).eval();

            auto IdxSs = tprod(B.Id(), std::sqrt(3.) * SiteOperator<double, Symmetry>::prod(B.Sdag(0), B.S(1), Symmetry::qvacuum()));

            auto SsxId = tprod(std::sqrt(3.) * SiteOperator<double, Symmetry>::prod(B.Sdag(0), B.S(1), Symmetry::qvacuum()), B.Id());

            gate = params["J"].get<double>() * ss + 0.25 * params["Jk"].get<double>() * (SsxId + IdxSs) + params["I"].get<double>() * SS;
            bond_gate = params["J"].get<double>() * ss + params["I"].get<double>() * SS;
        } else {
            auto SzSz = tprod(B.Sz(0), B.Sz(0));
            auto SpSm = tprod(B.Sp(0), B.Sm(0));
            auto SmSp = tprod(B.Sm(0), B.Sp(0));

            auto szsz = tprod(B.Sz(1), B.Sz(1));
            auto spsm = tprod(B.Sp(1), B.Sm(1));
            auto smsp = tprod(B.Sm(1), B.Sp(1));

            auto IdxSzsz = tprod(B.Id(), B.Sz(0) * B.Sz(1));
            auto IdxSpsm = tprod(B.Id(), B.Sp(0) * B.Sm(1));
            auto IdxSmsp = tprod(B.Id(), B.Sm(0) * B.Sp(1));

            auto SzszxId = tprod(B.Sz(0) * B.Sz(1), B.Id());
            auto SpsmxId = tprod(B.Sp(0) * B.Sm(1), B.Id());
            auto SmspxId = tprod(B.Sm(0) * B.Sp(1), B.Id());

            gate = params["Jz"].get<double>() * szsz + 0.5 * params["Jxy"].get<double>() * (spsm + smsp) +
                   0.25 * (params["Jkz"].get<double>() * (SzszxId + IdxSzsz) +
                           0.5 * params["Jkxy"].get<double>() * ((SpsmxId + IdxSpsm) + (SmspxId + IdxSmsp))) +
                   params["Iz"].get<double>() * SzSz + 0.5 * params["Ixy"].get<double>() * (SpSm + SmSp);
            bond_gate = params["Jz"].get<double>() * szsz + 0.5 * params["Jxy"].get<double>() * (spsm + smsp) + params["Iz"].get<double>() * SzSz +
                        0.5 * params["Ixy"].get<double>() * (SpSm + SmSp);
        }

        if((bond & Opts::Bond::H) == Opts::Bond::H) {
            for(auto& t : this->data_h) { t = gate; }
        }
        if((bond & Opts::Bond::V) == Opts::Bond::V) {
            for(auto& t : this->data_v) { t = gate; }
        }
        if((bond & Opts::Bond::D1) == Opts::Bond::D1) {
            for(auto& t : this->data_d1) { t = bond_gate; }
        }
        if((bond & Opts::Bond::D2) == Opts::Bond::D2) {
            for(auto& t : this->data_d2) { t = bond_gate; }
        }
    }

    virtual void setDefaultObs() override
    {
        auto Sz = std::make_unique<Xped::OneSiteObservable<Symmetry>>(pat);
        for(auto& t : Sz->data) { t = B.Sz(0).data.template trim<2>(); }
        auto sz = std::make_unique<Xped::OneSiteObservable<Symmetry>>(pat);
        for(auto& t : sz->data) { t = B.Sz(1).data.template trim<2>(); }
        obs.push_back(std::move(Sz));
        obs.push_back(std::move(sz));
    }

    virtual std::string file_name() const override { return internal::create_filename("KondoNecklace", params, used_params); }

    virtual std::string format() const override
    {
        return internal::format_params(fmt::format("KondoNecklace[sym={}]", Symmetry::name()), params, used_params);
    }

    virtual void computeObs(XPED_CONST CTM<double, Symmetry, 2, false, Opts::CTMCheckpoint{}>& env) override
    {
        for(auto& ob : obs) {
            if(auto* one = dynamic_cast<OneSiteObservable<Symmetry>*>(ob.get()); one != nullptr) { avg(env, *one); }
            if(auto* two = dynamic_cast<TwoSiteObservable<Symmetry>*>(ob.get()); two != nullptr) { avg(env, *two); }
        }
    }

    virtual void computeObs(XPED_CONST CTM<double, Symmetry, 1, false, Opts::CTMCheckpoint{}>& env) override
    {
        for(auto& ob : obs) {
            if(auto* one = dynamic_cast<OneSiteObservable<Symmetry>*>(ob.get()); one != nullptr) { avg(env, *one); }
            if(auto* two = dynamic_cast<TwoSiteObservable<Symmetry>*>(ob.get()); two != nullptr) { avg(env, *two); }
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

    virtual void obsToFile(HighFive::File& file) const override
    {
        for(const auto& ob : obs) { ob->toFile(file); }
    }

    virtual void initObsfile(HighFive::File& file) const override
    {
        for(const auto& ob : obs) { ob->initFile(file); }
    }

    std::map<std::string, Param> params;
    Pattern pat;
    std::vector<std::string> used_params;
    SpinBase<Symmetry> B;
    std::vector<std::unique_ptr<ObservableBase>> obs;
};

} // namespace Xped
#endif
