#ifndef XPED_PEPS_SPINLESS_FERMIONS_HPP_
#define XPED_PEPS_SPINLESS_FERMIONS_HPP_

#include <map>
#include <string>

#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"
#include "Xped/Symmetry/ZN.hpp"

#include "Xped/Core/Tensor.hpp"
#include "Xped/PEPS/Models/Hamiltonian.hpp"
#include "Xped/PEPS/OneSiteObservable.hpp"
#include "Xped/PEPS/TwoSiteObservable.hpp"
#include "Xped/Physics/SpinlessFermionBase.hpp"
#include "Xped/Util/Param.hpp"

namespace Xped {

template <typename Symmetry, typename Scalar = double>
class SpinlessFermions : public Hamiltonian<Scalar, Symmetry>
{
public:
    SpinlessFermions(std::map<std::string, Param>& params_in, const Pattern& pat_in, Opts::Bond bond_in = Opts::Bond::H | Opts::Bond::V)
        : Hamiltonian<Scalar, Symmetry>(params_in, pat_in, bond_in, "SpinlessFermions")
    {
        F = SpinlessFermionBase<Symmetry>(1);
        this->used_params = {"V", /*"μ"*/ "mu", "tprime1", "tprime2", "Vprime", "th", "tv"};
        Tensor<double, 2, 2, Symmetry> hopping, v_int, occ;

        if constexpr(Symmetry::ALL_ABELIAN) {
            hopping = (tprod(F.cdag(), F.c()) - tprod(F.c(), F.cdag())).eval();
            v_int = (tprod(F.n(), F.n())).eval();
            occ = 0.25 * (tprod(F.n(), F.Id()) + tprod(F.Id(), F.n())).eval();
			auto th = this->params["th"].template get<TMatrix<Scalar>>();
			auto tv = this->params["tv"].template get<TMatrix<Scalar>>();
			auto tprime1 = this->params["tprime1"].template get<TMatrix<Scalar>>();
			auto tprime2 = this->params["tprime2"].template get<TMatrix<Scalar>>();
			
			if((this->bond & Opts::Bond::H) == Opts::Bond::H) {
				for(auto pos = 0ul; pos < this->data_h.size(); ++pos) {
					auto [x, y] = this->pat.coords(pos);
					fmt::print("th({},{})={}\n", x,y,th(x,y));
					this->data_h(x,y) = -th(x,y) * hopping + this->params["V"].template get<double>() * v_int -
						this->params["mu"].template get<double>() * occ;
				}
            }
            if((this->bond & Opts::Bond::V) == Opts::Bond::V) {
				for(auto pos = 0ul; pos < this->data_v.size(); ++pos) {
					auto [x, y] = this->pat.coords(pos);
					fmt::print("tv({},{})={}\n", x,y,tv(x,y));
                    this->data_v(x,y) = -tv(x,y) * hopping + this->params["V"].template get<double>() * v_int -
                        this->params["mu"].template get<double>() * occ;
                }
            }
            if((this->bond & Opts::Bond::D1) == Opts::Bond::D1) {
				for(auto pos = 0ul; pos < this->data_d1.size(); ++pos) {
					auto [x, y] = this->pat.coords(pos);
					fmt::print("td1({},{})={}\n", x,y,tprime1(x,y));
                    // strange sign here but the code works fine
                    this->data_d1(x,y) = +tprime1(x,y) * hopping + this->params["Vprime"].template get<double>() * v_int;
                }
            }
            if((this->bond & Opts::Bond::D2) == Opts::Bond::D2) {
				for(auto pos = 0ul; pos < this->data_d2.size(); ++pos) {
					auto [x, y] = this->pat.coords(pos);
					fmt::print("td2({},{})={}\n", x,y,tprime2(x,y));
                    // strange sign here but the code works fine
                    this->data_d2(x,y) = +tprime2(x,y) * hopping + this->params["Vprime"].template get<double>() * v_int;
                }
            }
        } else {
            assert(false and "Symmetry is not supported in Hubbard model.");
        }
    }

    virtual void setDefaultObs() override
    {
        if constexpr(Symmetry::ALL_ABELIAN) {
            auto n = std::make_unique<Xped::OneSiteObservable<double, Symmetry>>(this->pat, "n");
            for(auto& t : n->data) { t = F.n().data.template trim<2>(); }
            this->obs.push_back(std::move(n));
        }
    }

private:
    SpinlessFermionBase<Symmetry> F;
};

} // namespace Xped
#endif
