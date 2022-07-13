#ifndef XPED_PEPS_HEISENBERG_HPP_
#define XPED_PEPS_HEISENBERG_HPP_

#include <any>
#include <map>
#include <string>

#include "Xped/Core/Tensor.hpp"
#include "Xped/PEPS/TwoSiteObservable"
#include "Xped/Physics/SpinBase.hpp"

namespace Xped {

/* gs energy:
   square: E/N = -0.66944
   triang: E/N = −0.5445
     j1j2: E/N =
*/
template <typename Symmetry>
class Heisenberg : public TwoSiteObservable<Symmetry>
{
public:
    Heisenberg(std::map<std::string, std::any>& params, const Pattern& pat, Opts::Bond bond = Opts::Bond::H | Opts::Bond::V)
        : TwoSiteObservable<Symmetry>(pat, bond)
    {
        this->name = "Heisenberg";
        SpinBase<Symmetry> B(1, 2);
        Tensor<double, 2, 2, Symmetry> gate;
        if constexpr(std::is_same_v<Symmetry, Sym::SU2<Sym::SpinSU2>>) {
            gate = std::any_cast<double>(params["J"]) * (std::sqrt(3.) * tprod(B.Sdag(0), B.S(0))).eval();
        } else {
            gate = std::any_cast<double>(params["Jz"]) * tprod(B.Sz(), B.Sz()) +
                   0.5 * std::any_cast<double>(params["Jxy"]) * (tprod(B.Sp(), B.Sm()) + tprod(B.Sm(), B.Sp()));
        }

        if((bond & Opts::Bond::H) == Opts::Bond::H) {
            for(auto& t : this->data_h) { t = gate; }
        }
        if((bond & Opts::Bond::V) == Opts::Bond::V) {
            for(auto& t : this->data_v) { t = gate; }
        }
        if((bond & Opts::Bond::D1) == Opts::Bond::D1) {
            for(auto& t : this->data_d1) { t = std::any_cast<double>(params["J2"]) * gate; }
        }
        if((bond & Opts::Bond::D2) == Opts::Bond::D2) {
            for(auto& t : this->data_d2) { t = std::any_cast<double>(params["J2"]) * gate; }
        }
    }
};

} // namespace Xped
#endif
