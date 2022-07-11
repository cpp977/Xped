#ifndef XPED_TWO_SITE_OBSERVABLE_HPP_
#define XPED_TWO_SITE_OBSERVABLE_HPP_

#include "Xped/Core/Tensor.hpp"
#include "Xped/PEPS/ObservableBase.hpp"
#include "Xped/PEPS/TMatrix.hpp"

namespace Xped {

template <typename Symmetry>
struct TwoSiteObservable : public ObservableBase
{
    /*
      x,y: O_h(x,y; x+1,y)
      x,y: O_v(x,y; x,y+1)
      x,y: O_d1(x,y; x+1,y+1)
      x,y: O_d2(x,y; x-1,y+1)
     */
    TwoSiteObservable(const Pattern& pat, Opts::Bond bond)
        : bond(bond)
    {
        if((bond & Opts::Bond::H) == Opts::Bond::H) {
            data_h = TMatrix<Tensor<double, 2, 2, Symmetry, false>>(pat);
            obs_h = TMatrix<double>(pat);
        }
        if((bond & Opts::Bond::V) == Opts::Bond::V) {
            data_v = TMatrix<Tensor<double, 2, 2, Symmetry, false>>(pat);
            obs_v = TMatrix<double>(pat);
        }
        if((bond & Opts::Bond::D1) == Opts::Bond::D1) {
            data_d1 = TMatrix<Tensor<double, 2, 2, Symmetry, false>>(pat);
            obs_d1 = TMatrix<double>(pat);
        }
        if((bond & Opts::Bond::D2) == Opts::Bond::D2) {
            data_d2 = TMatrix<Tensor<double, 2, 2, Symmetry, false>>(pat);
            obs_d2 = TMatrix<double>(pat);
        }
    }
    TMatrix<Tensor<double, 2, 2, Symmetry, false>> data_h;
    TMatrix<Tensor<double, 2, 2, Symmetry, false>> data_v;
    TMatrix<Tensor<double, 2, 2, Symmetry, false>> data_d1;
    TMatrix<Tensor<double, 2, 2, Symmetry, false>> data_d2;
    TMatrix<double> obs_h;
    TMatrix<double> obs_v;
    TMatrix<double> obs_d1;
    TMatrix<double> obs_d2;
    Opts::Bond bond;
};

} // namespace Xped

#endif
