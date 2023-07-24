#ifndef XPED_KONDO_HPP_
#define XPED_KONDO_HPP_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <highfive/H5File.hpp>

#include "Xped/Core/Tensor.hpp"
#include "Xped/PEPS/Models/Hamiltonian.hpp"
#include "Xped/PEPS/OneSiteObservable.hpp"
#include "Xped/PEPS/TwoSiteObservable.hpp"
#include "Xped/Physics/FermionBase.hpp"
#include "Xped/Physics/SpinBase.hpp"
#include "Xped/Util/Param.hpp"

namespace Xped {

template <typename Symmetry, typename Scalar = double>
class Kondo : public Hamiltonian<Scalar, Symmetry>
{
    using Op = SiteOperator<double, Symmetry>;

public:
    Kondo(std::map<std::string, Param>& params_in, const Pattern& pat_in, Opts::Bond bond_in = Opts::Bond::H | Opts::Bond::V)
        : Hamiltonian<Scalar, Symmetry>(params_in, pat_in, bond_in, "Kondo")
    {
        B = SpinBase<Symmetry>(1, 2);
        F = FermionBase<Symmetry>(1);
        Tensor<double, 2, 2, Symmetry> gate, gate2, local_gate;
        if constexpr(Symmetry::Nq > 0 and Symmetry::ANY_NON_ABELIAN) {
            if constexpr(Symmetry::IS_SPIN[0] and Symmetry::NON_ABELIAN[0] and Symmetry::ABELIAN[1]) {
                this->used_params = {"t", "Jk", "I", "tprime", "Iprime"};
            }
        } else if constexpr(Symmetry::ALL_ABELIAN) {
            this->used_params = {"t", "Jkxy", "Jkz", "Ixy", "Iz", "tprime", "Izprime", "Ixyprime"};
        }
        if constexpr(Symmetry::Nq > 0 and Symmetry::ANY_NON_ABELIAN) {
            if constexpr(Symmetry::IS_SPIN[0] and Symmetry::NON_ABELIAN[0] and Symmetry::ABELIAN[1]) {
                auto SS = (std::sqrt(3.) * tprod(Op::outerprod(F.Id(), B.Sdag()), Op::outerprod(F.Id(), B.S()))).eval();

                auto hop = (std::sqrt(2.) * tprod(Op::outerprod(F.cdag(), B.Id()), Op::outerprod(F.c(), B.Id())) +
                            std::sqrt(2.) * tprod(Op::outerprod(F.c(), B.Id()), Op::outerprod(F.cdag(), B.Id())))
                               .eval();

                auto IdxSs = tprod(Op::outerprod(F.Id(), B.Id()), std::sqrt(3.) * Op::outerprod(F.Sdag(), B.S(), Symmetry::qvacuum()));
                auto SsxId = tprod(std::sqrt(3.) * Op::outerprod(F.Sdag(), B.S(), Symmetry::qvacuum()), Op::outerprod(F.Id(), B.Id()));
                gate = -this->params["t"].template get<double>() * hop + this->params["I"].template get<double>() * SS;
                gate2 = this->params["tprime"].template get<double>() * hop + this->params["Iprime"].template get<double>() * SS;
                local_gate = gate + 0.25 * this->params["Jk"].template get<double>() * (SsxId + IdxSs);
            }
        } else if constexpr(Symmetry::ALL_ABELIAN) {
            auto SzSz = tprod(Op::outerprod(F.Id(), B.Sz()), Op::outerprod(F.Id(), B.Sz()));
            auto SpSm = tprod(Op::outerprod(F.Id(), B.Sp()), Op::outerprod(F.Id(), B.Sm()));
            auto SmSp = tprod(Op::outerprod(F.Id(), B.Sm()), Op::outerprod(F.Id(), B.Sp()));

            auto IdxSzsz = tprod(Op::outerprod(F.Id(), B.Id()), Op::outerprod(F.Sz(), B.Sz()));
            auto IdxSpsm = tprod(Op::outerprod(F.Id(), B.Id()), Op::outerprod(F.Sp(), B.Sm()));
            auto IdxSmsp = tprod(Op::outerprod(F.Id(), B.Id()), Op::outerprod(F.Sm(), B.Sp()));

            auto SzszxId = tprod(Op::outerprod(F.Sz(), B.Sz()), Op::outerprod(F.Id(), B.Id()));
            auto SpsmxId = tprod(Op::outerprod(F.Sp(), B.Sm()), Op::outerprod(F.Id(), B.Id()));
            auto SmspxId = tprod(Op::outerprod(F.Sm(), B.Sp()), Op::outerprod(F.Id(), B.Id()));

            auto hop = (tprod(Op::outerprod(F.cdag(SPIN_INDEX::UP), B.Id()), Op::outerprod(F.c(SPIN_INDEX::UP), B.Id())) +
                        tprod(Op::outerprod(F.cdag(SPIN_INDEX::DN), B.Id()), Op::outerprod(F.c(SPIN_INDEX::DN), B.Id())) -
                        tprod(Op::outerprod(F.c(SPIN_INDEX::UP), B.Id()), Op::outerprod(F.cdag(SPIN_INDEX::UP), B.Id())) -
                        tprod(Op::outerprod(F.c(SPIN_INDEX::DN), B.Id()), Op::outerprod(F.cdag(SPIN_INDEX::DN), B.Id())))
                           .eval();

            gate = -this->params["t"].template get<double>() * hop + this->params["Iz"].template get<double>() * SzSz +
                   0.5 * this->params["Ixy"].template get<double>() * (SpSm + SmSp);
            gate2 = this->params["tprime"].template get<double>() * hop + this->params["Izprime"].template get<double>() * SzSz +
                    0.5 * this->params["Ixyprime"].template get<double>() * (SpSm + SmSp);
            local_gate = gate + 0.25 * (this->params["Jkz"].template get<double>() * (SzszxId + IdxSzsz) +
                                        0.5 * this->params["Jkxy"].template get<double>() * ((SpsmxId + IdxSpsm) + (SmspxId + IdxSmsp)));
        } else {
            assert(false and "Symmetry is not supported in Kondo model.");
        }

        if((this->bond & Opts::Bond::H) == Opts::Bond::H) {
            for(auto& t : this->data_h) { t = local_gate; }
        }
        if((this->bond & Opts::Bond::V) == Opts::Bond::V) {
            for(auto& t : this->data_v) { t = local_gate; }
        }
        if((this->bond & Opts::Bond::D1) == Opts::Bond::D1) {
            for(auto& t : this->data_d1) { t = gate2; }
        }
        if((this->bond & Opts::Bond::D2) == Opts::Bond::D2) {
            for(auto& t : this->data_d2) { t = gate2; }
        }
    }

    virtual void setDefaultObs() override
    {
        if constexpr(Symmetry::Nq > 0 and Symmetry::ANY_NON_ABELIAN) {
            if constexpr(Symmetry::IS_SPIN[0] and Symmetry::NON_ABELIAN[0] and Symmetry::ABELIAN[1]) {
                auto Sdags = std::make_unique<OneSiteObservable<double, Symmetry>>(this->pat, "Sdags");
                for(auto& t : Sdags->data) { t = std::sqrt(3.) * Op::outerprod(F.Sdag(), B.S(), Symmetry::qvacuum()).data.template trim<2>(); }
                this->obs.push_back(std::move(Sdags));
                auto cdagc = std::make_unique<TwoSiteObservable<double, Symmetry>>(this->pat, Opts::Bond::H | Opts::Bond::V, "cdagc");
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
                for(auto& t : cdagc->data_d1) {
                    t = (-1. * (std::sqrt(2.) * tprod(Op::outerprod(F.cdag(), B.Id()), Op::outerprod(F.c(), B.Id())) +
                                std::sqrt(2.) * tprod(Op::outerprod(F.c(), B.Id()), Op::outerprod(F.cdag(), B.Id()))))
                            .eval();
                }
                for(auto& t : cdagc->data_d2) {
                    t = (-1. * (std::sqrt(2.) * tprod(Op::outerprod(F.cdag(), B.Id()), Op::outerprod(F.c(), B.Id())) +
                                std::sqrt(2.) * tprod(Op::outerprod(F.c(), B.Id()), Op::outerprod(F.cdag(), B.Id()))))
                            .eval();
                }

                auto SdagS = std::make_unique<TwoSiteObservable<double, Symmetry>>(this->pat, Opts::Bond::H | Opts::Bond::V, "SdagS");
                for(auto& t : SdagS->data_h) { t = (std::sqrt(3.) * tprod(Op::outerprod(F.Id(), B.Sdag()), Op::outerprod(F.Id(), B.S()))).eval(); }
                for(auto& t : SdagS->data_v) { t = (std::sqrt(3.) * tprod(Op::outerprod(F.Id(), B.Sdag()), Op::outerprod(F.Id(), B.S()))).eval(); }
                for(auto& t : SdagS->data_d1) { t = (std::sqrt(3.) * tprod(Op::outerprod(F.Id(), B.Sdag()), Op::outerprod(F.Id(), B.S()))).eval(); }
                for(auto& t : SdagS->data_d2) { t = (std::sqrt(3.) * tprod(Op::outerprod(F.Id(), B.Sdag()), Op::outerprod(F.Id(), B.S()))).eval(); }
                this->obs.push_back(std::move(cdagc));
                this->obs.push_back(std::move(SdagS));
            }
        } else if constexpr(Symmetry::ALL_ABELIAN) {
            auto Sz = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(this->pat, "Sz");
            for(auto& t : Sz->data) { t = Op::outerprod(F.Id(), B.Sz(0)).data.template trim<2>(); }
            auto sz = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(this->pat, "sz");
            for(auto& t : sz->data) { t = Op::outerprod(F.Sz(0), B.Id()).data.template trim<2>(); }
            auto nup = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(this->pat, "nup");
            for(auto& t : nup->data) { t = Op::outerprod(F.n(Xped::SPIN_INDEX::UP), B.Id()).data.template trim<2>(); }
            auto ndn = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(this->pat, "ndn");
            for(auto& t : ndn->data) { t = Op::outerprod(F.n(Xped::SPIN_INDEX::DN), B.Id()).data.template trim<2>(); }
            auto n = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(this->pat, "n");
            for(auto& t : n->data) { t = Op::outerprod(F.n(), B.Id()).data.template trim<2>(); }
            auto d = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(this->pat, "d");
            for(auto& t : d->data) { t = Op::outerprod(F.d(), B.Id()).data.template trim<2>(); }
            this->obs.push_back(std::move(nup));
            this->obs.push_back(std::move(ndn));
            this->obs.push_back(std::move(n));
            this->obs.push_back(std::move(d));
            this->obs.push_back(std::move(Sz));
            this->obs.push_back(std::move(sz));
            if constexpr(not Symmetry::ANY_IS_SPIN) {
                auto Sx = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(this->pat, "Sx");
                for(auto& t : Sx->data) { t = Op::outerprod(F.Id(), B.Sx(0)).data.template trim<2>(); }
                auto sx = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(this->pat, "sx");
                for(auto& t : sx->data) { t = Op::outerprod(F.Sx(0), B.Id()).data.template trim<2>(); }

                this->obs.push_back(std::move(Sx));
                this->obs.push_back(std::move(sx));
            }
        }
    }

    SpinBase<Symmetry> B;
    FermionBase<Symmetry> F;
};

} // namespace Xped
#endif
