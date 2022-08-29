#ifndef XPED_HUBBARD_HPP_
#define XPED_HUBBARD_HPP_

#include <any>
#include <map>
#include <string>

#include "Xped/Core/Tensor.hpp"
#include "Xped/PEPS/TwoSiteObservable.hpp"
#include "Xped/Physics/FermionBase.hpp"

namespace Xped {

template <typename Symmetry>
class Hubbard : public TwoSiteObservable<Symmetry>
{
public:
    Hubbard(std::map<std::string, std::any>& params, const Pattern& pat, Opts::Bond bond = Opts::Bond::H | Opts::Bond::V)
        : TwoSiteObservable<Symmetry>(pat, bond)
    {
        this->name =
            "Hubbard[U=" + std::to_string(std::any_cast<double>(params["U"])) + ", μ=" + std::to_string(std::any_cast<double>(params["mu"])) + "]";
        FermionBase<Symmetry> F(1);
        Tensor<double, 2, 2, Symmetry> gate;
        if constexpr(std::is_same_v<Symmetry, Sym::SU2<Sym::SpinSU2>>) {
            gate.setZero();
        } else {
            gate = -std::any_cast<double>(params["t"]) *
                       (tprod(F.cdag(SPIN_INDEX::UP), F.c(SPIN_INDEX::UP)) + tprod(F.cdag(SPIN_INDEX::DN), F.c(SPIN_INDEX::DN)) -
                        tprod(F.c(SPIN_INDEX::UP), F.cdag(SPIN_INDEX::UP)) - tprod(F.c(SPIN_INDEX::DN), F.cdag(SPIN_INDEX::DN))) +
                   0.25 * std::any_cast<double>(params["U"]) * (tprod(F.d(), F.Id()) + tprod(F.Id(), F.d())) -
                   0.25 * std::any_cast<double>(params["mu"]) * (tprod(F.n(), F.Id()) + tprod(F.Id(), F.n()));
            auto [Es, Vs] = gate.eigh();
            gate.print(std::cout, true);
            std::cout << std::endl;
            Es.print(std::cout, true);
            std::cout << std::endl;
        }

        if((bond & Opts::Bond::H) == Opts::Bond::H) {
            for(auto& t : this->data_h) { t = gate; }
        }
        if((bond & Opts::Bond::V) == Opts::Bond::V) {
            for(auto& t : this->data_v) { t = gate; }
        }
        if((bond & Opts::Bond::D1) == Opts::Bond::D1) {
            for(auto& t : this->data_d1) { t = gate; }
        }
        if((bond & Opts::Bond::D2) == Opts::Bond::D2) {
            for(auto& t : this->data_d2) { t = gate; }
        }
    }
};

} // namespace Xped
#endif
