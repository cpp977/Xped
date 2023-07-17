#ifndef XPED_TIME_PROPAGATOR_H_
#define XPED_TIME_PROPAGATOR_H_

#include <memory>

#include "Xped/PEPS/ImagOpts.hpp"
#include "Xped/PEPS/SimpleUpdate.hpp"
#include "Xped/PEPS/TwoSiteObservable.hpp"
#include "Xped/PEPS/iPEPS.hpp"

namespace Xped {

namespace Opts {

enum class GATE_ORDER
{
    HDV,
    VDH
};

}

template <typename Scalar_, typename TimeScalar_, typename HamScalar_, typename Symmetry_>
class TimePropagator
{
public:
    using Scalar = Scalar_;
    using TimeScalar = TimeScalar_;
    using HamScalar = HamScalar_;
    using Symmetry = Symmetry_;

    TimePropagator() = delete;

    explicit TimePropagator(const TwoSiteObservable<HamScalar, Symmetry>& H_in,
                            TimeScalar dt_in,
                            const Opts::Update& update_in,
                            const TMatrix<typename Symmetry::qType>& charges_in)
        : H(H_in)
        , cell_(H_in.data_h.pat)
        , dt(dt_in)
        , update(update_in)
        , charges(charges_in)
    {
        initU();
        spectrum_h.resize(cell_.pattern);
        spectrum_v.resize(cell_.pattern);
    }

    void t_step(iPEPS<Scalar, Symmetry>& Psi);

    TMatrix<Tensor<Scalar, 1, 1, Symmetry>> spectrum_h;
    TMatrix<Tensor<Scalar, 1, 1, Symmetry>> spectrum_v;

private:
    const TwoSiteObservable<HamScalar, Symmetry>& H;
    UnitCell cell_;
    TimeScalar dt;
    Opts::Update update;
    TMatrix<typename Symmetry::qType> charges;

    TwoSiteObservable<HamScalar, Symmetry> U;
    TwoSiteObservable<HamScalar, Symmetry> Usqrt;
    TwoSiteObservable<HamScalar, Symmetry> Usq;

    void t_step_h(iPEPS<Scalar, Symmetry>& Psi, int x, int y);
    void t_step_v(iPEPS<Scalar, Symmetry>& Psi, int x, int y);

    void t_step_d1(iPEPS<Scalar, Symmetry>& Psi, int x, int y, Opts::GATE_ORDER gate_order, bool UPDATE_BOTH_DIAGONALS = false);
    void t_step_d2(iPEPS<Scalar, Symmetry>& Psi, int x, int y, Opts::GATE_ORDER gate_order, bool UPDATE_BOTH_DIAGONALS = false);

    std::tuple<Tensor<Scalar, 2, 1, Symmetry>, Tensor<Scalar, 1, 1, Symmetry>, Tensor<Scalar, 1, 2, Symmetry>>
    renormalize(const Tensor<Scalar, 2, 2, Symmetry>& bond,
                const Tensor<Scalar, 3, 1, Symmetry>& left,
                const Tensor<Scalar, 1, 3, Symmetry>& right,
                std::size_t max_keep) const;

    std::tuple<Tensor<Scalar, 2, 1, Symmetry>,
               Tensor<Scalar, 1, 1, Symmetry>,
               Tensor<Scalar, 4, 1, Symmetry>,
               Tensor<Scalar, 1, 1, Symmetry>,
               Tensor<Scalar, 1, 2, Symmetry>>
    renormalize_d1(const Tensor<Scalar, 2, 5, Symmetry>& bond, std::size_t max_keep) const;

    std::tuple<Tensor<Scalar, 1, 2, Symmetry>,
               Tensor<Scalar, 1, 1, Symmetry>,
               Tensor<Scalar, 4, 1, Symmetry>,
               Tensor<Scalar, 1, 1, Symmetry>,
               Tensor<Scalar, 1, 2, Symmetry>>
    renormalize_d2(const Tensor<Scalar, 2, 5, Symmetry>& bond, std::size_t max_keep) const;

    void initU();
};

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "PEPS/TimePropagator.cpp"
#endif

#endif
