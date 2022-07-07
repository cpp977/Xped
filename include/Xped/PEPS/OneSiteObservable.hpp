#ifndef XPED_ONE_SITE_OBSERVABLE_HPP_
#define XPED_ONE_SITE_OBSERVABLE_HPP_

#include "Xped/Core/Tensor.hpp"
#include "Xped/PEPS/ObservableBase.hpp"
#include "Xped/PEPS/TMatrix.hpp"

namespace Xped {

template <typename Symmetry>
struct OneSiteObservable : public ObservableBase
{
    explicit OneSiteObservable(const Pattern& pat)
        : data(pat)
        , obs(pat)
    {}
    TMatrix<Tensor<double, 1, 1, Symmetry, false>> data;
    TMatrix<double> obs;
};

} // namespace Xped

#endif
