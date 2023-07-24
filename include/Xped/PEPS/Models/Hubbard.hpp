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
#include "Xped/PEPS/Models/Hamiltonian.hpp"
#include "Xped/PEPS/OneSiteObservable.hpp"
#include "Xped/PEPS/TwoSiteObservable.hpp"
#include "Xped/Physics/FermionBase.hpp"
#include "Xped/Util/Param.hpp"

namespace Xped {

template <typename Symmetry, typename Scalar = double>
class Hubbard : public Hamiltonian<Scalar, Symmetry>
{
public:
    Hubbard(std::map<std::string, Param>& params_in, const Pattern& pat_in, Opts::Bond bond_in = Opts::Bond::H | Opts::Bond::V)
        : Hamiltonian<Scalar, Symmetry>(params_in, pat_in, bond_in, "Hubbard")
    {
        F = FermionBase<Symmetry>(1);
        this->used_params = {"U", /*"μ"*/ "mu", "tprime", "t"};
        Tensor<double, 2, 2, Symmetry> gate, bond_gate, hopping, hubbard, occ;
        if constexpr(Symmetry::Nq == 2 and Symmetry::ANY_NON_ABELIAN) {
            if constexpr(Symmetry::IS_SPIN[0] and Symmetry::NON_ABELIAN[0] and Symmetry::ABELIAN[1]) {
                hopping = (std::sqrt(2.) * tprod(F.cdag(), F.c()) + std::sqrt(2.) * tprod(F.c(), F.cdag())).eval();
                hubbard = 0.25 * (tprod(F.d(), F.Id()) + tprod(F.Id(), F.d()));
                occ = 0.25 * (tprod(F.n(), F.Id()) + tprod(F.Id(), F.n()));
                if((this->bond & Opts::Bond::H) == Opts::Bond::H) {
                    for(auto& t : this->data_h) {
                        t = -this->params["t"].template get<double>() * hopping + this->params["U"].template get<double>() * hubbard -
                            this->params["mu"].template get<double>() * occ;
                    }
                }
                if((this->bond & Opts::Bond::V) == Opts::Bond::V) {
                    for(auto& t : this->data_v) {
                        t = -this->params["t"].template get<double>() * hopping + this->params["U"].template get<double>() * hubbard -
                            this->params["mu"].template get<double>() * occ;
                    }
                }
                if((this->bond & Opts::Bond::D1) == Opts::Bond::D1) {
                    // strange sign here but the code works fine
                    for(auto& t : this->data_d1) { t = +this->params["tprime"].template get<double>() * hopping; }
                }
                if((this->bond & Opts::Bond::D2) == Opts::Bond::D2) {
                    // strange sign here but the code works fine
                    for(auto& t : this->data_d2) { t = +this->params["tprime"].template get<double>() * hopping; }
                }
            }
        } else if constexpr(std::is_same_v<Symmetry,
                                           Xped::Sym::Combined<Xped::Sym::SU2<Xped::Sym::SpinSU2>,
                                                               Xped::Sym::SU2<Xped::Sym::SpinSU2>,
                                                               Xped::Sym::ZN<Xped::Sym::FChargeU1, 2>>>) {
            hopping = (tprod(F.cdag(), F.c()) - tprod(F.c(), F.cdag())).eval();
            hubbard = 0.25 * (tprod(F.n() * (F.n() - F.Id()), F.Id()) + tprod(F.Id(), F.n() * (F.n() - F.Id())));
            occ = 0.25 * (tprod(F.n(), F.Id()) + tprod(F.Id(), F.n()));
            gate = -this->params["t"].template get<double>() * (tprod(F.cdag(), F.c()) - tprod(F.c(), F.cdag())) +
                   0.25 * 0.5 * this->params["U"].template get<double>() *
                       (tprod(F.n() * (F.n() - F.Id()), F.Id()) + tprod(F.Id(), F.n() * (F.n() - F.Id()))) -
                   0.25 * (this->params["mu"].template get<double>() + 1.5 * this->params["U"].template get<double>()) *
                       (tprod(F.n(), F.Id()) + tprod(F.Id(), F.n()));
            bond_gate = -this->params["t"].template get<double>() * (tprod(F.cdag(), F.c()) - tprod(F.c(), F.cdag()));

            if((this->bond & Opts::Bond::H) == Opts::Bond::H) {
                for(auto& t : this->data_h) {
                    t = -this->params["t"].template get<double>() * hopping + 0.5 * this->params["U"].template get<double>() * hubbard -
                        (this->params["mu"].template get<double>() + 1.5 * this->params["U"].template get<double>()) * occ;
                }
            }
            if((this->bond & Opts::Bond::V) == Opts::Bond::V) {
                for(auto& t : this->data_v) {
                    t = -this->params["t"].template get<double>() * hopping + 0.5 * this->params["U"].template get<double>() * hubbard -
                        (this->params["mu"].template get<double>() + 1.5 * this->params["U"].template get<double>()) * occ;
                }
            }
            if((this->bond & Opts::Bond::D1) == Opts::Bond::D1) {
                for(auto& t : this->data_d1) { t = -this->params["tprime"].template get<double>() * hopping; }
            }
            if((this->bond & Opts::Bond::D2) == Opts::Bond::D2) {
                for(auto& t : this->data_d2) { t = -this->params["tprime"].template get<double>() * hopping; }
            }
        } else if constexpr(Symmetry::ALL_ABELIAN) {
            this->used_params.push_back("Bx");
            this->used_params.push_back("By");
            this->used_params.push_back("Bz");
            hopping = (tprod(F.cdag(SPIN_INDEX::UP), F.c(SPIN_INDEX::UP)) + tprod(F.cdag(SPIN_INDEX::DN), F.c(SPIN_INDEX::DN)) -
                       tprod(F.c(SPIN_INDEX::UP), F.cdag(SPIN_INDEX::UP)) - tprod(F.c(SPIN_INDEX::DN), F.cdag(SPIN_INDEX::DN)))
                          .eval();

            hubbard = 0.25 * (tprod(F.d(), F.Id()) + tprod(F.Id(), F.d())).eval();
            occ = 0.25 * (tprod(F.n(), F.Id()) + tprod(F.Id(), F.n())).eval();
            // std::cout << hopping << std::endl;
            // std::exit(0);
            // Es.print(std::cout, true);
            // std::cout << std::endl;
            auto Bx = this->params["Bx"].template get<TMatrix<double>>();
            auto By = this->params["By"].template get<TMatrix<double>>();
            auto Bz = this->params["Bz"].template get<TMatrix<double>>();
            if((this->bond & Opts::Bond::H) == Opts::Bond::H) {
                for(auto pos = 0ul; pos < this->data_h.size(); ++pos) {
                    auto [x, y] = this->pat.coords(pos);
                    this->data_h(x, y) = -this->params["t"].template get<double>() * hopping + this->params["U"].template get<double>() * hubbard -
                                         this->params["mu"].template get<double>() * occ;
                    if constexpr(std::is_same_v<Scalar, std::complex<double>>) {
                        this->data_h(x, y) = this->data_h(x, y) - Bx(x, y) * 0.25 * (tprod(F.Sx(), F.Id()) + tprod(F.Id(), F.Sx())).eval() -
                                             By(x, y) * 0.25 * (tprod(F.Sy(), F.Id()) + tprod(F.Id(), F.Sy())).eval() -
                                             Bz(x, y) * 0.25 * (tprod(F.Sz(), F.Id()) + tprod(F.Id(), F.Sz())).eval();
                    }
                    // t = this->params["t"].template get<double>() * hopping;
                }
            }
            if((this->bond & Opts::Bond::V) == Opts::Bond::V) {
                for(auto pos = 0ul; pos < this->data_v.size(); ++pos) {
                    auto [x, y] = this->pat.coords(pos);
                    this->data_v(x, y) = -this->params["t"].template get<double>() * hopping + this->params["U"].template get<double>() * hubbard -
                                         this->params["mu"].template get<double>() * occ;
                    if constexpr(std::is_same_v<Scalar, std::complex<double>>) {
                        this->data_v(x, y) = this->data_v(x, y) - Bx(x, y) * 0.25 * (tprod(F.Sx(), F.Id()) + tprod(F.Id(), F.Sx())).eval() -
                                             By(x, y) * 0.25 * (tprod(F.Sy(), F.Id()) + tprod(F.Id(), F.Sy())).eval() -
                                             Bz(x, y) * 0.25 * (tprod(F.Sz(), F.Id()) + tprod(F.Id(), F.Sz())).eval();
                    }
                    // t = this->params["t"].template get<double>() * hopping;
                }
            }
            if((this->bond & Opts::Bond::D1) == Opts::Bond::D1) {
                for(auto& t : this->data_d1) {
                    // strange sign here but the code works fine
                    t = +this->params["tprime"].template get<double>() * hopping;
                }
            }
            if((this->bond & Opts::Bond::D2) == Opts::Bond::D2) {
                for(auto& t : this->data_d2) {
                    // strange sign here but the code works fine
                    t = +this->params["tprime"].template get<double>() * hopping;
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
                auto n = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(this->pat, "n");
                for(auto& t : n->data) { t = F.n().data.template trim<2>(); }
                this->obs.push_back(std::move(n));
                auto d = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(this->pat, "d");
                for(auto& t : d->data) { t = F.d().data.template trim<2>(); }
                this->obs.push_back(std::move(d));
                auto hopping = (std::sqrt(2.) * tprod(F.cdag(), F.c()) + std::sqrt(2.) * tprod(F.c(), F.cdag())).eval();
                auto cdagc = std::make_unique<TwoSiteObservable<double, Symmetry>>(
                    this->pat, Opts::Bond::H | Opts::Bond::V | Opts::Bond::D1 | Opts::Bond::D2, "cdagc");
                for(auto& t : cdagc->data_h) { t = hopping; }
                for(auto& t : cdagc->data_v) { t = hopping; }
                for(auto& t : cdagc->data_d1) { t = -1. * hopping; }
                for(auto& t : cdagc->data_d2) { t = -1. * hopping; }
                this->obs.push_back(std::move(cdagc));
            }
        } else if constexpr(std::is_same_v<Symmetry,
                                           Xped::Sym::Combined<Xped::Sym::SU2<Xped::Sym::SpinSU2>,
                                                               Xped::Sym::SU2<Xped::Sym::SpinSU2>,
                                                               Xped::Sym::ZN<Xped::Sym::FChargeU1, 2>>>) {

            auto n = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(this->pat, "n");
            for(auto& t : n->data) { t = F.n().data.template trim<2>(); }
            this->obs.push_back(std::move(n));
        } else if constexpr(Symmetry::ALL_ABELIAN) {
            auto nup = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(this->pat, "nup");
            for(auto& t : nup->data) { t = F.n(Xped::SPIN_INDEX::UP).data.template trim<2>(); }
            auto ndn = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(this->pat, "ndn");
            for(auto& t : ndn->data) { t = F.n(Xped::SPIN_INDEX::DN).data.template trim<2>(); }
            auto n = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(this->pat, "n");
            for(auto& t : n->data) { t = F.n().data.template trim<2>(); }
            auto d = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(this->pat, "d");
            for(auto& t : d->data) { t = F.d().data.template trim<2>(); }
            auto Sz = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(this->pat, "Sz");
            for(auto& t : Sz->data) { t = F.Sz().data.template trim<2>(); }
            this->obs.push_back(std::move(nup));
            this->obs.push_back(std::move(ndn));
            this->obs.push_back(std::move(n));
            this->obs.push_back(std::move(d));
            this->obs.push_back(std::move(Sz));
            auto hopping = (tprod(F.cdag(SPIN_INDEX::UP), F.c(SPIN_INDEX::UP)) + tprod(F.cdag(SPIN_INDEX::DN), F.c(SPIN_INDEX::DN)) -
                            tprod(F.c(SPIN_INDEX::UP), F.cdag(SPIN_INDEX::UP)) - tprod(F.c(SPIN_INDEX::DN), F.cdag(SPIN_INDEX::DN)))
                               .eval();
            auto cdagc = std::make_unique<TwoSiteObservable<double, Symmetry>>(
                this->pat, Opts::Bond::H | Opts::Bond::V | Opts::Bond::D1 | Opts::Bond::D2, "cdagc");
            for(auto& t : cdagc->data_h) { t = hopping; }
            for(auto& t : cdagc->data_v) { t = hopping; }
            for(auto& t : cdagc->data_d1) { t = -1. * hopping; }
            for(auto& t : cdagc->data_d2) { t = -1. * hopping; }

            auto SzSz = std::make_unique<TwoSiteObservable<double, Symmetry>>(
                this->pat, Opts::Bond::H | Opts::Bond::V | Opts::Bond::D1 | Opts::Bond::D2, "SzSz");
            for(auto& t : SzSz->data_h) { t = tprod(F.Sz(), F.Sz()); }
            for(auto& t : SzSz->data_v) { t = tprod(F.Sz(), F.Sz()); }
            for(auto& t : SzSz->data_d1) { t = tprod(F.Sz(), F.Sz()); }
            for(auto& t : SzSz->data_d2) { t = tprod(F.Sz(), F.Sz()); }

            this->obs.push_back(std::move(cdagc));
            this->obs.push_back(std::move(SzSz));
            if constexpr(not Symmetry::ANY_IS_SPIN) {
                auto Sx = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(this->pat, "Sx");
                for(auto& t : Sx->data) { t = F.Sx().data.template trim<2>(); }
                auto SxSx = std::make_unique<TwoSiteObservable<double, Symmetry>>(
                    this->pat, Opts::Bond::H | Opts::Bond::V | Opts::Bond::D1 | Opts::Bond::D2, "SxSx");
                for(auto& t : SxSx->data_h) { t = tprod(F.Sx(), F.Sx()); }
                for(auto& t : SxSx->data_v) { t = tprod(F.Sx(), F.Sx()); }
                for(auto& t : SxSx->data_d1) { t = tprod(F.Sx(), F.Sx()); }
                for(auto& t : SxSx->data_d2) { t = tprod(F.Sx(), F.Sx()); }
                this->obs.push_back(std::move(Sx));
                this->obs.push_back(std::move(SxSx));
                auto Sy = std::make_unique<Xped::OneSiteObservable<std::complex<double>, Symmetry>>(this->pat, "Sy");
                for(auto& t : Sy->data) { t = F.Sy().data.template trim<2>(); }
                this->obs.push_back(std::move(Sy));
            }
        }
    }

    FermionBase<Symmetry> F;
};

} // namespace Xped
#endif
