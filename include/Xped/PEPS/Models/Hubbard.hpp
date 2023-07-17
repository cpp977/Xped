#ifndef XPED_HUBBARD_HPP_
#define XPED_HUBBARD_HPP_

#include <map>
#include <string>

#include <highfive/H5File.hpp>

#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"
#include "Xped/Symmetry/ZN.hpp"

#include "Xped/Core/Tensor.hpp"
#include "Xped/PEPS/Models/Helpers.hpp"
#include "Xped/PEPS/OneSiteObservable.hpp"
#include "Xped/PEPS/TwoSiteObservable.hpp"
#include "Xped/Physics/FermionBase.hpp"
#include "Xped/Util/Param.hpp"

namespace Xped {

template <typename Symmetry>
class Hubbard : public TwoSiteObservable<std::complex<double>, Symmetry>
{
public:
    Hubbard(std::map<std::string, Param>& params_in, const Pattern& pat_in, Opts::Bond bond = Opts::Bond::H | Opts::Bond::V)
        : TwoSiteObservable<double, Symmetry>(pat_in, bond)
        , params(params_in)
        , pat(pat_in)
    {
        F = FermionBase<Symmetry>(1);
        used_params = {"U", /*"μ"*/ "mu", "tprime", "t"};
        Tensor<double, 2, 2, Symmetry> gate, bond_gate, hopping, hubbard, occ;
        if constexpr(Symmetry::Nq == 2 and Symmetry::ANY_NON_ABELIAN) {
            if constexpr(Symmetry::IS_SPIN[0] and Symmetry::NON_ABELIAN[0] and Symmetry::ABELIAN[1]) {
                hopping = (std::sqrt(2.) * tprod(F.cdag(), F.c()) + std::sqrt(2.) * tprod(F.c(), F.cdag())).eval();
                hubbard = 0.25 * (tprod(F.d(), F.Id()) + tprod(F.Id(), F.d()));
                occ = 0.25 * (tprod(F.n(), F.Id()) + tprod(F.Id(), F.n()));
                if((bond & Opts::Bond::H) == Opts::Bond::H) {
                    for(auto& t : this->data_h) {
                        t = -params["t"].get<double>() * hopping + params["U"].get<double>() * hubbard - params["mu"].get<double>() * occ;
                    }
                }
                if((bond & Opts::Bond::V) == Opts::Bond::V) {
                    for(auto& t : this->data_v) {
                        t = -params["t"].get<double>() * hopping + params["U"].get<double>() * hubbard - params["mu"].get<double>() * occ;
                    }
                }
                if((bond & Opts::Bond::D1) == Opts::Bond::D1) {
                    // strange sign here but the code works fine
                    for(auto& t : this->data_d1) { t = +params["tprime"].get<double>() * hopping; }
                }
                if((bond & Opts::Bond::D2) == Opts::Bond::D2) {
                    // strange sign here but the code works fine
                    for(auto& t : this->data_d2) { t = +params["tprime"].get<double>() * hopping; }
                }
            }
        } else if constexpr(std::is_same_v<Symmetry,
                                           Xped::Sym::Combined<Xped::Sym::SU2<Xped::Sym::SpinSU2>,
                                                               Xped::Sym::SU2<Xped::Sym::SpinSU2>,
                                                               Xped::Sym::ZN<Xped::Sym::FChargeU1, 2>>>) {
            hopping = (tprod(F.cdag(), F.c()) - tprod(F.c(), F.cdag())).eval();
            hubbard = 0.25 * (tprod(F.n() * (F.n() - F.Id()), F.Id()) + tprod(F.Id(), F.n() * (F.n() - F.Id())));
            occ = 0.25 * (tprod(F.n(), F.Id()) + tprod(F.Id(), F.n()));
            gate = -params["t"].get<double>() * (tprod(F.cdag(), F.c()) - tprod(F.c(), F.cdag())) +
                   0.25 * 0.5 * params["U"].get<double>() * (tprod(F.n() * (F.n() - F.Id()), F.Id()) + tprod(F.Id(), F.n() * (F.n() - F.Id()))) -
                   0.25 * (params["mu"].get<double>() + 1.5 * params["U"].get<double>()) * (tprod(F.n(), F.Id()) + tprod(F.Id(), F.n()));
            bond_gate = -params["t"].get<double>() * (tprod(F.cdag(), F.c()) - tprod(F.c(), F.cdag()));

            if((bond & Opts::Bond::H) == Opts::Bond::H) {
                for(auto& t : this->data_h) {
                    t = -params["t"].get<double>() * hopping + 0.5 * params["U"].get<double>() * hubbard -
                        (params["mu"].get<double>() + 1.5 * params["U"].get<double>()) * occ;
                }
            }
            if((bond & Opts::Bond::V) == Opts::Bond::V) {
                for(auto& t : this->data_v) {
                    t = -params["t"].get<double>() * hopping + 0.5 * params["U"].get<double>() * hubbard -
                        (params["mu"].get<double>() + 1.5 * params["U"].get<double>()) * occ;
                }
            }
            if((bond & Opts::Bond::D1) == Opts::Bond::D1) {
                for(auto& t : this->data_d1) { t = -params["tprime"].get<double>() * hopping; }
            }
            if((bond & Opts::Bond::D2) == Opts::Bond::D2) {
                for(auto& t : this->data_d2) { t = -params["tprime"].get<double>() * hopping; }
            }
        } else if constexpr(Symmetry::ALL_ABELIAN) {
            used_params.push_back("Bx");
            used_params.push_back("By");
            used_params.push_back("Bz");
            hopping = (tprod(F.cdag(SPIN_INDEX::UP), F.c(SPIN_INDEX::UP)) + tprod(F.cdag(SPIN_INDEX::DN), F.c(SPIN_INDEX::DN)) -
                       tprod(F.c(SPIN_INDEX::UP), F.cdag(SPIN_INDEX::UP)) - tprod(F.c(SPIN_INDEX::DN), F.cdag(SPIN_INDEX::DN)))
                          .eval();

            hubbard = 0.25 * (tprod(F.d(), F.Id()) + tprod(F.Id(), F.d())).eval();
            occ = 0.25 * (tprod(F.n(), F.Id()) + tprod(F.Id(), F.n())).eval();
            // std::cout << hopping << std::endl;
            // std::exit(0);
            // Es.print(std::cout, true);
            // std::cout << std::endl;
            auto Bx = params["Bx"].get<TMatrix<double>>();
            auto By = params["By"].get<TMatrix<double>>();
            auto Bz = params["Bz"].get<TMatrix<double>>();
            if((bond & Opts::Bond::H) == Opts::Bond::H) {
                for(auto pos = 0ul; pos < this->data_h.size(); ++pos) {
                    auto [x, y] = pat.coord(pos);
                    this->data_h(x, y) = -params["t"].get<double>() * hopping + params["U"].get<double>() * hubbard -
                                         params["mu"].get<double>() * occ - Bx(x, y) * 0.25 * (tprod(F.Bx(), F.Id()) + tprod(F.Id(), F.Bx())).eval() -
                                         By(x, y) * 0.25 * (tprod(F.By(), F.Id()) + tprod(F.Id(), F.By())).eval() -
                                         Bz(x, y) * 0.25 * (tprod(F.Bz(), F.Id()) + tprod(F.Id(), F.Bz())).eval();

                    // t = params["t"].get<double>() * hopping;
                }
            }
            if((bond & Opts::Bond::V) == Opts::Bond::V) {
                for(auto pos = 0ul; pos < this->data_v.size(); ++pos) {
                    auto [x, y] = pat.coord(pos);
                    this->data_v(x, y) = -params["t"].get<double>() * hopping + params["U"].get<double>() * hubbard -
                                         params["mu"].get<double>() * occ - Bx(x, y) * 0.25 * (tprod(F.Bx(), F.Id()) + tprod(F.Id(), F.Bx())).eval() -
                                         By(x, y) * 0.25 * (tprod(F.By(), F.Id()) + tprod(F.Id(), F.By())).eval() -
                                         Bz(x, y) * 0.25 * (tprod(F.Bz(), F.Id()) + tprod(F.Id(), F.Bz())).eval();

                    // t = params["t"].get<double>() * hopping;
                }
            }
            if((bond & Opts::Bond::D1) == Opts::Bond::D1) {
                for(auto& t : this->data_d1) {
                    // strange sign here but the code works fine
                    t = +params["tprime"].get<double>() * hopping;
                }
            }
            if((bond & Opts::Bond::D2) == Opts::Bond::D2) {
                for(auto& t : this->data_d2) {
                    // strange sign here but the code works fine
                    t = +params["tprime"].get<double>() * hopping;
                }
            }
        } else {
            assert(false and "Symmetry is not supported in Hubbard model.");
        }
        // this->loadFromMatlab(std::filesystem::path("/home/user/matlab-tmp/hubbard_D2.mat"), "cpp");
    }

    virtual void setDefaultObs() override
    {
        if constexpr(Symmetry::Nq == 2 and Symmetry::ANY_NON_ABELIAN) {
            if constexpr(Symmetry::IS_SPIN[0] and Symmetry::NON_ABELIAN[0] and Symmetry::ABELIAN[1]) {
                auto n = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(pat, "n");
                for(auto& t : n->data) { t = F.n().data.template trim<2>(); }
                obs.push_back(std::move(n));
                auto d = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(pat, "d");
                for(auto& t : d->data) { t = F.d().data.template trim<2>(); }
                obs.push_back(std::move(d));
                auto hopping = (std::sqrt(2.) * tprod(F.cdag(), F.c()) + std::sqrt(2.) * tprod(F.c(), F.cdag())).eval();
                auto cdagc = std::make_unique<TwoSiteObservable<double, Symmetry>>(
                    pat, Opts::Bond::H | Opts::Bond::V | Opts::Bond::D1 | Opts::Bond::D2, "cdagc");
                for(auto& t : cdagc->data_h) { t = hopping; }
                for(auto& t : cdagc->data_v) { t = hopping; }
                for(auto& t : cdagc->data_d1) { t = -1. * hopping; }
                for(auto& t : cdagc->data_d2) { t = -1. * hopping; }
                obs.push_back(std::move(cdagc));
            }
        } else if constexpr(std::is_same_v<Symmetry,
                                           Xped::Sym::Combined<Xped::Sym::SU2<Xped::Sym::SpinSU2>,
                                                               Xped::Sym::SU2<Xped::Sym::SpinSU2>,
                                                               Xped::Sym::ZN<Xped::Sym::FChargeU1, 2>>>) {

            auto n = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(pat, "n");
            for(auto& t : n->data) { t = F.n().data.template trim<2>(); }
            obs.push_back(std::move(n));
        } else if constexpr(Symmetry::ALL_ABELIAN) {
            auto nup = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(pat, "nup");
            for(auto& t : nup->data) { t = F.n(Xped::SPIN_INDEX::UP).data.template trim<2>(); }
            auto ndn = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(pat, "ndn");
            for(auto& t : ndn->data) { t = F.n(Xped::SPIN_INDEX::DN).data.template trim<2>(); }
            auto n = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(pat, "n");
            for(auto& t : n->data) { t = F.n().data.template trim<2>(); }
            auto d = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(pat, "d");
            for(auto& t : d->data) { t = F.d().data.template trim<2>(); }
            auto Sz = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(pat, "Sz");
            for(auto& t : Sz->data) { t = F.Sz().data.template trim<2>(); }
            obs.push_back(std::move(nup));
            obs.push_back(std::move(ndn));
            obs.push_back(std::move(n));
            obs.push_back(std::move(d));
            obs.push_back(std::move(Sz));
            auto hopping = (tprod(F.cdag(SPIN_INDEX::UP), F.c(SPIN_INDEX::UP)) + tprod(F.cdag(SPIN_INDEX::DN), F.c(SPIN_INDEX::DN)) -
                            tprod(F.c(SPIN_INDEX::UP), F.cdag(SPIN_INDEX::UP)) - tprod(F.c(SPIN_INDEX::DN), F.cdag(SPIN_INDEX::DN)))
                               .eval();
            auto cdagc =
                std::make_unique<TwoSiteObservable<double, Symmetry>>(pat, Opts::Bond::H | Opts::Bond::V | Opts::Bond::D1 | Opts::Bond::D2, "cdagc");
            for(auto& t : cdagc->data_h) { t = hopping; }
            for(auto& t : cdagc->data_v) { t = hopping; }
            for(auto& t : cdagc->data_d1) { t = -1. * hopping; }
            for(auto& t : cdagc->data_d2) { t = -1. * hopping; }

            auto SzSz =
                std::make_unique<TwoSiteObservable<double, Symmetry>>(pat, Opts::Bond::H | Opts::Bond::V | Opts::Bond::D1 | Opts::Bond::D2, "SzSz");
            for(auto& t : SzSz->data_h) { t = tprod(F.Sz(), F.Sz()); }
            for(auto& t : SzSz->data_v) { t = tprod(F.Sz(), F.Sz()); }
            for(auto& t : SzSz->data_d1) { t = tprod(F.Sz(), F.Sz()); }
            for(auto& t : SzSz->data_d2) { t = tprod(F.Sz(), F.Sz()); }

            obs.push_back(std::move(cdagc));
            obs.push_back(std::move(SzSz));
            if constexpr(not Symmetry::ANY_IS_SPIN) {
                auto Sx = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(pat, "Sx");
                for(auto& t : Sx->data) { t = F.Sx().data.template trim<2>(); }
                auto SxSx = std::make_unique<TwoSiteObservable<double, Symmetry>>(
                    pat, Opts::Bond::H | Opts::Bond::V | Opts::Bond::D1 | Opts::Bond::D2, "SxSx");
                for(auto& t : SxSx->data_h) { t = tprod(F.Sx(), F.Sx()); }
                for(auto& t : SxSx->data_v) { t = tprod(F.Sx(), F.Sx()); }
                for(auto& t : SxSx->data_d1) { t = tprod(F.Sx(), F.Sx()); }
                for(auto& t : SxSx->data_d2) { t = tprod(F.Sx(), F.Sx()); }
                obs.push_back(std::move(Sx));
                obs.push_back(std::move(SxSx));
                auto Sy = std::make_unique<Xped::OneSiteObservable<std::complex<double>, Symmetry>>(pat, "Sy");
                for(auto& t : Sy->data) { t = F.Sy().data.template trim<2>(); }
                obs.push_back(std::move(Sy));
            }
        }
    }

    virtual std::string file_name() const override
    {
        return internal::create_filename(fmt::format("Hubbard_Lx={}_Ly={}", pat.Lx, pat.Ly), params, used_params);
    }

    virtual std::string format() const override
    {
        return internal::format_params(fmt::format("Hubbard[sym={}]", Symmetry::name()), params, used_params);
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
            if(auto* one = dynamic_cast<OneSiteObservable<double, Symmetry, false>*>(ob.get()); one != nullptr) { avg(env, *one); }
            if(auto* one_c = dynamic_cast<OneSiteObservable<std::complex<double>, Symmetry, false>*>(ob.get()); one_c != nullptr) {
                avg(env, *one_c);
            }
            if(auto* two = dynamic_cast<TwoSiteObservable<double, Symmetry, false>*>(ob.get()); two != nullptr) { avg(env, *two); }
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
        for(auto i = 0ul; const auto& ob : obs) {
            out.append(ob->getResString(offset));
            if(i++ < obs.size() - 1) { out.push_back('\n'); }
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
    FermionBase<Symmetry> F;
    std::vector<std::unique_ptr<ObservableBase>> obs;
};

} // namespace Xped
#endif
