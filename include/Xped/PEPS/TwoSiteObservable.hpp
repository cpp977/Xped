#ifndef XPED_TWO_SITE_OBSERVABLE_HPP_
#define XPED_TWO_SITE_OBSERVABLE_HPP_

#include "Xped/Core/Tensor.hpp"
#include "Xped/PEPS/Bonds.hpp"
#include "Xped/PEPS/ObservableBase.hpp"
#include "Xped/PEPS/TMatrix.hpp"

namespace Xped {

template <typename Symmetry>
struct TwoSiteObservable : public ObservableBase
{
    TwoSiteObservable() = default;

    /*
      x,y: O_h(x,y; x+1,y)
      x,y: O_v(x,y; x,y+1)
      x,y: O_d1(x,y; x+1,y+1)
      x,y: O_d2(x,y; x+1,y-1)
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

    TwoSiteObservable<Symmetry> shiftQN(const TMatrix<typename Symmetry::qType>& charges)
    {
        TwoSiteObservable<Symmetry> out(data_h.pat, bond);
        for(int x = 0; x < data_h.pat.Lx; ++x) {
            for(int y = 0; y < data_h.pat.Ly; ++y) {
                if(not data_h.pat.isUnique(x, y)) { continue; }
                if((bond & Opts::Bond::H) == Opts::Bond::H) {
                    out.data_h(x, y) = data_h(x, y).template shiftQN<0, 2>(charges(x, y)).template shiftQN<1, 3>(charges(x + 1, y));
                }
                if((bond & Opts::Bond::V) == Opts::Bond::V) {
                    out.data_v(x, y) = data_v(x, y).template shiftQN<0, 2>(charges(x, y)).template shiftQN<1, 3>(charges(x, y + 1));
                }
                if((bond & Opts::Bond::D1) == Opts::Bond::D1) {
                    out.data_d1(x, y) = data_d1(x, y).template shiftQN<0, 2>(charges(x, y)).template shiftQN<1, 3>(charges(x + 1, y + 1));
                }
                if((bond & Opts::Bond::D2) == Opts::Bond::D2) {
                    out.data_d2(x, y) = data_d2(x, y).template shiftQN<0, 2>(charges(x, y)).template shiftQN<1, 3>(charges(x + 1, y - 1));
                }
            }
        }
        return out;
    }

    template <typename Ar>
    void serialize(Ar& ar)
    {
        ar& YAS_OBJECT_NVP("TwoSiteobservable",
                           ("data_h", data_h),
                           ("data_v", data_v),
                           ("data_d1", data_d1),
                           ("data_d2", data_d2),
                           ("obs_h", obs_h),
                           ("obs_v", obs_v),
                           ("obs_d1", obs_d1),
                           ("obs_d2", obs_d2),
                           ("bond", bond));
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
