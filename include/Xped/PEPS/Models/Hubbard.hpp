#ifndef XPED_HUBBARD_HPP_
#define XPED_HUBBARD_HPP_

#include <map>
#include <string>

#include <highfive/H5File.hpp>

#include "Xped/Core/Tensor.hpp"
#include "Xped/PEPS/Models/Helpers.hpp"
#include "Xped/PEPS/TwoSiteObservable.hpp"
#include "Xped/Physics/FermionBase.hpp"
#include "Xped/Util/Param.hpp"

namespace Xped {

template <typename Symmetry>
class Hubbard : public TwoSiteObservable<Symmetry>
{
public:
    Hubbard(std::map<std::string, Param>& params_in, const Pattern& pat_in, Opts::Bond bond = Opts::Bond::H | Opts::Bond::V)
        : TwoSiteObservable<Symmetry>(pat_in, bond)
        , params(params_in)
        , pat(pat_in)
    {
        F = FermionBase<Symmetry>(1);
        used_params = {"U", /*"μ"*/ "mu", "t"};
        Tensor<double, 2, 2, Symmetry> gate, bond_gate;
        if constexpr(std::is_same_v<Symmetry,
                                    Xped::Sym::Combined<Xped::Sym::SU2<Xped::Sym::SpinSU2>,
                                                        Xped::Sym::SU2<Xped::Sym::SpinSU2>,
                                                        Xped::Sym::ZN<Xped::Sym::FChargeU1, 2>>>) {
            gate = -params["t"].get<double>() * (tprod(F.cdag(), F.c()) - tprod(F.c(), F.cdag())) +
                   0.25 * 0.5 * params["U"].get<double>() * (tprod(F.n() * (F.n() - F.Id()), F.Id()) + tprod(F.Id(), F.n() * (F.n() - F.Id()))) -
                   0.25 * (params["mu"].get<double>() + 1.5 * params["U"].get<double>()) * (tprod(F.n(), F.Id()) + tprod(F.Id(), F.n()));
            bond_gate = -params["t"].get<double>() * (tprod(F.cdag(), F.c()) - tprod(F.c(), F.cdag()));
        } else {
            gate = -params["t"].get<double>() *
                       (tprod(F.cdag(SPIN_INDEX::UP), F.c(SPIN_INDEX::UP)) + tprod(F.cdag(SPIN_INDEX::DN), F.c(SPIN_INDEX::DN)) -
                        tprod(F.c(SPIN_INDEX::UP), F.cdag(SPIN_INDEX::UP)) - tprod(F.c(SPIN_INDEX::DN), F.cdag(SPIN_INDEX::DN))) +
                   0.25 * params["U"].get<double>() * (tprod(F.d(), F.Id()) + tprod(F.Id(), F.d())) -
                   0.25 * params["mu"].get<double>() * (tprod(F.n(), F.Id()) + tprod(F.Id(), F.n()));
            // auto [Es, Vs] = gate.eigh();
            // gate.print(std::cout, true);
            // std::cout << std::endl;
            // Es.print(std::cout, true);
            // std::cout << std::endl;
            bond_gate = -params["t"].get<double>() *
                        (tprod(F.cdag(SPIN_INDEX::UP), F.c(SPIN_INDEX::UP)) + tprod(F.cdag(SPIN_INDEX::DN), F.c(SPIN_INDEX::DN)) -
                         tprod(F.c(SPIN_INDEX::UP), F.cdag(SPIN_INDEX::UP)) - tprod(F.c(SPIN_INDEX::DN), F.cdag(SPIN_INDEX::DN)));
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
        auto n = std::make_unique<Xped::OneSiteObservable<Symmetry>>(pat, "n");
        for(auto& t : n->data) { t = F.n().data.template trim<2>(); }
        // auto nup = std::make_unique<Xped::OneSiteObservable<Symmetry>>(pat, "nup");
        // for(auto& t : nup->data) { t = F.n(Xped::SPIN_INDEX::UP).data.template trim<2>(); }
        // auto ndn = std::make_unique<Xped::OneSiteObservable<Symmetry>>(pat, "ndn");
        // for(auto& t : ndn->data) { t = F.n(Xped::SPIN_INDEX::DN).data.template trim<2>(); }
        // auto d = std::make_unique<Xped::OneSiteObservable<Symmetry>>(pat, "d");
        // for(auto& t : d->data) { t = F.d().data.template trim<2>(); }
        obs.push_back(std::move(n));
        // obs.push_back(std::move(nup));
        // obs.push_back(std::move(ndn));
        // obs.push_back(std::move(d));
    }

    virtual std::string file_name() const override { return internal::create_filename("Hubbard", params, used_params); }

    virtual std::string format() const override
    {
        return internal::format_params(fmt::format("Hubbard[sym={}]", Symmetry::name()), params, used_params);
    }

    virtual void computeObs(XPED_CONST CTM<double, Symmetry, 2>& env) override
    {
        for(auto& ob : obs) {
            if(auto* one = dynamic_cast<OneSiteObservable<Symmetry>*>(ob.get()); one != nullptr) { avg(env, *one); }
            if(auto* two = dynamic_cast<TwoSiteObservable<Symmetry>*>(ob.get()); two != nullptr) { avg(env, *two); }
        }
    }

    virtual void computeObs(XPED_CONST CTM<double, Symmetry, 1>& env) override
    {
        for(auto& ob : obs) {
            if(auto* one = dynamic_cast<OneSiteObservable<Symmetry>*>(ob.get()); one != nullptr) { avg(env, *one); }
            if(auto* two = dynamic_cast<TwoSiteObservable<Symmetry>*>(ob.get()); two != nullptr) { avg(env, *two); }
        }
    }

    virtual std::string getObsString(const std::string& offset) const override
    {
        std::string out;
        for(auto i = 0ul; const auto& ob : obs) {
            out.append(ob->getResString(offset));
            if(i++ < obs.size() - 1) { out.push_back('\n'); }
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
    FermionBase<Symmetry> F;
    std::vector<std::unique_ptr<ObservableBase>> obs;
};

} // namespace Xped
#endif
