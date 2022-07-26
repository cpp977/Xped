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
    OneSiteObservable<Symmetry> shiftQN(const TMatrix<typename Symmetry::qType>& charges)
    {
        OneSiteObservable<Symmetry> out(data.pat);
        for(int x = 0; x < data.pat.Lx; ++x) {
            for(int y = 0; y < data.pat.Ly; ++y) {
                if(not data.pat.isUnique(x, y)) { continue; }
                out.data(x, y) = data(x, y).template shiftQN<0, 1>(charges(x, y));
            }
        }
        return out;
    }
};

} // namespace Xped

#endif
