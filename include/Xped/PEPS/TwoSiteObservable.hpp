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
    explicit TwoSiteObservable(const Pattern& pat)
        : data_h(pat)
        , data_v(pat)
        , data_d1(pat)
        , data_d2(pat)
        , obs_h(pat)
        , obs_v(pat)
        , obs_d1(pat)
        , obs_d2(pat)
    {}
    TMatrix<Tensor<double, 2, 2, Symmetry, false>> data_h;
    TMatrix<Tensor<double, 2, 2, Symmetry, false>> data_v;
    TMatrix<Tensor<double, 2, 2, Symmetry, false>> data_d1;
    TMatrix<Tensor<double, 2, 2, Symmetry, false>> data_d2;
    TMatrix<double> obs_h;
    TMatrix<double> obs_v;
    TMatrix<double> obs_d1;
    TMatrix<double> obs_d2;
};

} // namespace Xped

#endif
