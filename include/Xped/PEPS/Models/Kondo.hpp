#ifndef XPED_KONDO_HPP_
#define XPED_KONDO_HPP_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <highfive/H5File.hpp>

#include "Xped/Core/Tensor.hpp"
#include "Xped/PEPS/Models/Helpers.hpp"
#include "Xped/PEPS/OneSiteObservable.hpp"
#include "Xped/PEPS/TwoSiteObservable.hpp"
#include "Xped/Physics/FermionBase.hpp"
#include "Xped/Physics/SpinBase.hpp"
#include "Xped/Util/Param.hpp"

namespace Xped {

template <typename Symmetry>
class Kondo : public TwoSiteObservable<Symmetry>
{
    using Op = SiteOperator<double, Symmetry>;

public:
    Kondo(std::map<std::string, Param>& params_in, const Pattern& pat_in, Opts::Bond bond = Opts::Bond::H | Opts::Bond::V)
        : TwoSiteObservable<Symmetry>(pat_in, bond)
        , params(params_in)
        , pat(pat_in)
    {
        B = SpinBase<Symmetry>(1, 2);
        F = FermionBase<Symmetry>(1);
        Tensor<double, 2, 2, Symmetry> gate, local_gate;
        if constexpr(Symmetry::IS_SPIN[0] and Symmetry::NON_ABELIAN[0] and Symmetry::ABELIAN[1]) {
            used_params = {"t", "Jk", "I"};
        } else if constexpr(Symmetry::ALL_ABELIAN) {
            used_params = {"t", "Jkxy", "Jkz", "Ixy", "Iz"};
        }

        if constexpr(Symmetry::IS_SPIN[0] and Symmetry::NON_ABELIAN[0] and Symmetry::ABELIAN[1]) {
            auto SS = (std::sqrt(3.) * tprod(Op::outerprod(F.Id(), B.Sdag()), Op::outerprod(F.Id(), B.S()))).eval();

            auto hop = (std::sqrt(2.) * tprod(Op::outerprod(F.cdag(), B.Id()), Op::outerprod(F.c(), B.Id())) +
                        std::sqrt(2.) * tprod(Op::outerprod(F.c(), B.Id()), Op::outerprod(F.cdag(), B.Id())))
                           .eval();

            auto IdxSs = tprod(Op::outerprod(F.Id(), B.Id()), std::sqrt(3.) * Op::outerprod(F.Sdag(), B.S(), Symmetry::qvacuum()));
            auto SsxId = tprod(std::sqrt(3.) * Op::outerprod(F.Sdag(), B.S(), Symmetry::qvacuum()), Op::outerprod(F.Id(), B.Id()));
            std::cout << hop << std::endl << SS << std::endl;
            gate = -params["t"].get<double>() * hop + params["I"].get<double>() * SS;
            local_gate = gate + 0.25 * params["Jk"].get<double>() * (SsxId + IdxSs);
        } else if constexpr(Symmetry::ALL_ABELIAN) {
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
            gate = params["Jz"].get<double>() * szsz + 0.5 * params["Jxy"].get<double>() * (spsm + smsp);
            gate = gate + params["Iz"].get<double>() * SzSz + 0.5 * params["Ixy"].get<double>() * (SpSm + SmSp);
            local_gate = gate + 0.25 * (params["Jkz"].get<double>() * (SzszxId + IdxSzsz) +
                                        0.5 * params["Jkxy"].get<double>() * ((SpsmxId + IdxSpsm) + (SmspxId + IdxSmsp)));
        } else {
            assert(false and "Symmetry is not supported in Kondo model.");
        }

        if((bond & Opts::Bond::H) == Opts::Bond::H) {
            for(auto& t : this->data_h) { t = local_gate; }
        }
        if((bond & Opts::Bond::V) == Opts::Bond::V) {
            for(auto& t : this->data_v) { t = local_gate; }
        }
        if((bond & Opts::Bond::D1) == Opts::Bond::D1) {
            for(auto& t : this->data_d1) { t = gate; }
        }
        if((bond & Opts::Bond::D2) == Opts::Bond::D2) {
            for(auto& t : this->data_d2) { t = gate; }
        }
    }

    virtual void setDefaultObs() override
    {
        if constexpr(Symmetry::IS_SPIN[0] and Symmetry::NON_ABELIAN[0] and Symmetry::ABELIAN[1]) {
            auto Sdags = std::make_unique<OneSiteObservable<Symmetry>>(pat, "Sdags");
            for(auto& t : Sdags->data) { t = std::sqrt(3.) * Op::outerprod(F.Sdag(), B.S(), Symmetry::qvacuum()).data.template trim<2>(); }
            obs.push_back(std::move(Sdags));
            auto cdagc = std::make_unique<TwoSiteObservable<Symmetry>>(pat, Opts::Bond::H | Opts::Bond::V, "cdagc");
            for(auto& t : cdagc->data_h) {
                t = (std::sqrt(2.) * tprod(Op::outerprod(F.cdag(), B.Id()), Op::outerprod(F.c(), B.Id())) +
                     std::sqrt(2.) * tprod(Op::outerprod(F.c(), B.Id()), Op::outerprod(F.cdag(), B.Id())))
                        .eval();
            }
            for(auto& t : cdagc->data_v) {
                t = (std::sqrt(2.) * tprod(Op::outerprod(F.cdag(), B.Id()), Op::outerprod(F.c(), B.Id())) +
                     std::sqrt(2.) * tprod(Op::outerprod(F.c(), B.Id()), Op::outerprod(F.cdag(), B.Id())))
                        .eval();
            }
            auto SdagS = std::make_unique<TwoSiteObservable<Symmetry>>(pat, Opts::Bond::H | Opts::Bond::V, "SdagS");
            for(auto& t : SdagS->data_h) { t = (std::sqrt(3.) * tprod(Op::outerprod(F.Id(), B.Sdag()), Op::outerprod(F.Id(), B.S()))).eval(); }
            for(auto& t : SdagS->data_v) { t = (std::sqrt(3.) * tprod(Op::outerprod(F.Id(), B.Sdag()), Op::outerprod(F.Id(), B.S()))).eval(); }
            obs.push_back(std::move(cdagc));
            obs.push_back(std::move(SdagS));
        } else if constexpr(Symmetry::ALL_ABELIAN) {
            auto Sz = std::make_unique<Xped::OneSiteObservable<Symmetry>>(pat, "Sz");
            for(auto& t : Sz->data) { t = Op::outerprod(F.Id(), B.Sz(0)).data.template trim<2>(); }
            auto sz = std::make_unique<Xped::OneSiteObservable<Symmetry>>(pat, "sz");
            for(auto& t : sz->data) { t = B.Sz(1).data.template trim<2>(); }
            auto Szsz = std::make_unique<Xped::OneSiteObservable<Symmetry>>(pat, "Szsz");
            for(auto& t : Szsz->data) { t = (B.Sz(0) * B.Sz(1)).data.template trim<2>(); }
            auto Spsm = std::make_unique<Xped::OneSiteObservable<Symmetry>>(pat, "Spsm");
            for(auto& t : Spsm->data) { t = (B.Sp(0) * B.Sm(1)).data.template trim<2>(); }
            auto Smsp = std::make_unique<Xped::OneSiteObservable<Symmetry>>(pat, "Smsp");
            for(auto& t : Smsp->data) { t = (B.Sm(0) * B.Sp(1)).data.template trim<2>(); }
            obs.push_back(std::move(Sz));
            obs.push_back(std::move(sz));
            obs.push_back(std::move(Szsz));
            obs.push_back(std::move(Spsm));
            obs.push_back(std::move(Smsp));

            auto szsz = std::make_unique<Xped::TwoSiteObservable<Symmetry>>(
                pat, Xped::Opts::Bond::H | Xped::Opts::Bond::V | Xped::Opts::Bond::D1 | Xped::Opts::Bond::D2, "szsz");
            for(auto& t : szsz->data_h) { t = Xped::tprod(B.Sz(1), B.Sz(1)); }
            for(auto& t : szsz->data_v) { t = Xped::tprod(B.Sz(1), B.Sz(1)); }
            for(auto& t : szsz->data_d1) { t = Xped::tprod(B.Sz(1), B.Sz(1)); }
            for(auto& t : szsz->data_d2) { t = Xped::tprod(B.Sz(1), B.Sz(1)); }
            auto spsm = std::make_unique<Xped::TwoSiteObservable<Symmetry>>(
                pat, Xped::Opts::Bond::H | Xped::Opts::Bond::V | Xped::Opts::Bond::D1 | Xped::Opts::Bond::D2, "spsm");
            for(auto& t : spsm->data_h) { t = Xped::tprod(B.Sp(1), B.Sm(1)); }
            for(auto& t : spsm->data_v) { t = Xped::tprod(B.Sp(1), B.Sm(1)); }
            for(auto& t : spsm->data_d1) { t = Xped::tprod(B.Sp(1), B.Sm(1)); }
            for(auto& t : spsm->data_d2) { t = Xped::tprod(B.Sp(1), B.Sm(1)); }
            auto smsp = std::make_unique<Xped::TwoSiteObservable<Symmetry>>(
                pat, Xped::Opts::Bond::H | Xped::Opts::Bond::V | Xped::Opts::Bond::D1 | Xped::Opts::Bond::D2, "smsp");
            for(auto& t : smsp->data_h) { t = Xped::tprod(B.Sm(1), B.Sp(1)); }
            for(auto& t : smsp->data_v) { t = Xped::tprod(B.Sm(1), B.Sp(1)); }
            for(auto& t : smsp->data_d1) { t = Xped::tprod(B.Sm(1), B.Sp(1)); }
            for(auto& t : smsp->data_d2) { t = Xped::tprod(B.Sm(1), B.Sp(1)); }
            auto SzSz = std::make_unique<Xped::TwoSiteObservable<Symmetry>>(
                pat, Xped::Opts::Bond::H | Xped::Opts::Bond::V | Xped::Opts::Bond::D1 | Xped::Opts::Bond::D2, "SzSz");
            for(auto& t : SzSz->data_h) { t = Xped::tprod(B.Sz(0), B.Sz(0)); }
            for(auto& t : SzSz->data_v) { t = Xped::tprod(B.Sz(0), B.Sz(0)); }
            for(auto& t : SzSz->data_d1) { t = Xped::tprod(B.Sz(0), B.Sz(0)); }
            for(auto& t : SzSz->data_d2) { t = Xped::tprod(B.Sz(0), B.Sz(0)); }
            auto SpSm = std::make_unique<Xped::TwoSiteObservable<Symmetry>>(
                pat, Xped::Opts::Bond::H | Xped::Opts::Bond::V | Xped::Opts::Bond::D1 | Xped::Opts::Bond::D2, "SpSm");
            for(auto& t : SpSm->data_h) { t = Xped::tprod(B.Sp(0), B.Sm(0)); }
            for(auto& t : SpSm->data_v) { t = Xped::tprod(B.Sp(0), B.Sm(0)); }
            for(auto& t : SpSm->data_d1) { t = Xped::tprod(B.Sp(0), B.Sm(0)); }
            for(auto& t : SpSm->data_d2) { t = Xped::tprod(B.Sp(0), B.Sm(0)); }
            auto SmSp = std::make_unique<Xped::TwoSiteObservable<Symmetry>>(
                pat, Xped::Opts::Bond::H | Xped::Opts::Bond::V | Xped::Opts::Bond::D1 | Xped::Opts::Bond::D2, "SmSp");
            for(auto& t : SmSp->data_h) { t = Xped::tprod(B.Sm(0), B.Sp(0)); }
            for(auto& t : SmSp->data_v) { t = Xped::tprod(B.Sm(0), B.Sp(0)); }
            for(auto& t : SmSp->data_d1) { t = Xped::tprod(B.Sm(0), B.Sp(0)); }
            for(auto& t : SmSp->data_d2) { t = Xped::tprod(B.Sm(0), B.Sp(0)); }
            obs.push_back(std::move(szsz));
            obs.push_back(std::move(spsm));
            obs.push_back(std::move(smsp));
            obs.push_back(std::move(SzSz));
            obs.push_back(std::move(SpSm));
            obs.push_back(std::move(SmSp));
        }
    }

    virtual std::string file_name() const override { return internal::create_filename("Kondo", params, used_params); }

    virtual std::string format() const override
    {
        return internal::format_params(fmt::format("Kondo[sym={}]", Symmetry::name()), params, used_params);
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
    FermionBase<Symmetry> F;
    std::vector<std::unique_ptr<ObservableBase>> obs;
};

} // namespace Xped
#endif
