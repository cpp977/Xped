#ifndef XPED_KONDO_NECKLACE_HPP_
#define XPED_KONDO_NECKLACE_HPP_

#include <any>
#include <map>
#include <string>

#include "Xped/Core/Tensor.hpp"
#include "Xped/Physics/SpinBase.hpp"

namespace Xped {

template <typename Symmetry>
class KondoNecklace : public TwoSiteObservable<Symmetry>
{
public:
    KondoNecklace(std::map<std::string, std::any>& params, const Pattern& pat, Opts::Bond bond = Opts::Bond::H | Opts::Bond::V)
        : TwoSiteObservable<Symmetry>(pat, bond)
    {
        this->name = "KondoNecklace";
        SpinBase<Symmetry> B(2, 2);
        Tensor<double, 2, 2, Symmetry> gate;
        if constexpr(std::is_same_v<Symmetry, Sym::SU2<Sym::SpinSU2>>) {
            auto SS = (std::sqrt(3.) * tprod(B.Sdag(0), B.S(0))).eval();

            auto ss = (std::sqrt(3.) * tprod(B.Sdag(1), B.S(1))).eval();

            auto IdxSs = tprod(B.Id(), std::sqrt(3.) * SiteOperator<double, Symmetry>::prod(B.Sdag(0), B.S(1), Symmetry::qvacuum()));

            auto SsxId = tprod(std::sqrt(3.) * SiteOperator<double, Symmetry>::prod(B.Sdag(0), B.S(1), Symmetry::qvacuum()), B.Id());

            gate = std::any_cast<double>(params["J"]) * ss + 0.25 * std::any_cast<double>(params["Jk"]) * (SsxId + IdxSs) +
                   std::any_cast<double>(params["I"]) * SS;
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

            gate = std::any_cast<double>(params["Jz"]) * szsz + 0.5 * std::any_cast<double>(params["Jxy"]) * (spsm + smsp) +
                   0.25 * (std::any_cast<double>(params["Jkz"]) * (SzszxId + IdxSzsz) +
                           0.5 * std::any_cast<double>(params["Jkxy"]) * ((SpsmxId + IdxSpsm) + (SmspxId + IdxSmsp))) +
                   std::any_cast<double>(params["Iz"]) * SzSz + 0.5 * std::any_cast<double>(params["Ixy"]) * (SpSm + SmSp);
        }

        if((bond & Opts::Bond::H) == Opts::Bond::H) {
            for(auto& t : this->data_h) { t = gate; }
        }
        if((bond & Opts::Bond::H) == Opts::Bond::V) {
            for(auto& t : this->data_v) { t = gate; }
        }
        if((bond & Opts::Bond::D1) == Opts::Bond::D1) {
            for(auto& t : this->data_d1) { t = gate; }
        }
        if((bond & Opts::Bond::D2) == Opts::Bond::D2) {
            for(auto& t : this->data_d2) { t = gate; }
        }

        for(auto& t : this->data_h) { t = gate; }
        for(auto& t : this->data_v) { t = gate; }
    }
};

} // namespace Xped
#endif
