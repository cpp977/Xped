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

template <typename Scalar_, typename TimeScalar_, typename Symmetry_>
class TimePropagator
{
public:
    using Scalar = Scalar_;
    using TimeScalar = TimeScalar_;
    using Symmetry = Symmetry_;

    TimePropagator() = delete;

    explicit TimePropagator(const TwoSiteObservable<Symmetry>& H_in, std::shared_ptr<iPEPS<Scalar, Symmetry>> Psi_in, const Opts::Update& update_in)
        : H(H_in)
        , Psi(Psi_in)
        , cell_(Psi_in->cell())
        , update(update_in)
    {
        Psi->initWeightTensors();
        spectrum_h.resize(Psi->cell().pattern);
        spectrum_v.resize(Psi->cell().pattern);
    }

    void t_step(TimeScalar_ dt);

    TMatrix<Tensor<Scalar, 1, 1, Symmetry>> spectrum_h;
    TMatrix<Tensor<Scalar, 1, 1, Symmetry>> spectrum_v;

private:
    const TwoSiteObservable<Symmetry>& H;
    std::shared_ptr<iPEPS<Scalar, Symmetry>> Psi;
    UnitCell cell_;
    Opts::Update update;

    void t_step_h(int x, int y, TimeScalar_ dt);
    void t_step_v(int x, int y, TimeScalar_ dt);

    void t_step_d1(int x, int y, TimeScalar_ dt, Opts::GATE_ORDER gate_order, TimeScalar_ s = 1.);
    void t_step_d2(int x, int y, TimeScalar_ dt, Opts::GATE_ORDER gate_order, TimeScalar_ s = 1.);

    std::tuple<Tensor<Scalar, 2, 1, Symmetry>, Tensor<Scalar, 1, 1, Symmetry>, Tensor<Scalar, 1, 2, Symmetry>>
    renormalize(const Tensor<Scalar, 2, 2, Symmetry>& bond,
                const Tensor<Scalar, 3, 1, Symmetry>& left,
                const Tensor<Scalar, 1, 3, Symmetry>& right) const;

    std::tuple<Tensor<Scalar, 2, 1, Symmetry>,
               Tensor<Scalar, 1, 1, Symmetry>,
               Tensor<Scalar, 4, 1, Symmetry>,
               Tensor<Scalar, 1, 1, Symmetry>,
               Tensor<Scalar, 1, 2, Symmetry>>
    renormalize_d1(const Tensor<Scalar, 2, 5, Symmetry>& bond) const;

    std::tuple<Tensor<Scalar, 1, 2, Symmetry>,
               Tensor<Scalar, 1, 1, Symmetry>,
               Tensor<Scalar, 4, 1, Symmetry>,
               Tensor<Scalar, 1, 1, Symmetry>,
               Tensor<Scalar, 1, 2, Symmetry>>
    renormalize_d2(const Tensor<Scalar, 2, 5, Symmetry>& bond) const;
};

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "PEPS/TimePropagator.cpp"
#endif

#endif
