#ifndef XPED_TIME_PROPAGATOR_H_
#define XPED_TIME_PROPAGATOR_H_

#include <memory>

#include "Xped/PEPS/SimpleUpdate.hpp"
#include "Xped/PEPS/TwoSiteObservable.hpp"
#include "Xped/PEPS/iPEPS.hpp"

namespace Xped {

template <typename Scalar_, typename TimeScalar_, typename Symmetry_, typename Update = SimpleUpdate<Scalar_, Symmetry_>>
class TimePropagator
{
public:
    using Scalar = Scalar_;
    using TimeScalar = TimeScalar_;
    using Symmetry = Symmetry_;

    TimePropagator() = delete;

    explicit TimePropagator(const TwoSiteObservable<Symmetry>& H_in, std::shared_ptr<iPEPS<Scalar, Symmetry>> Psi_in, const Update& updater_in)
        : H(H_in)
        , Psi(Psi_in)
        , cell_(Psi_in->cell())
        , updater(updater_in)
    {
        Psi->initWeightTensors();
    }

    void t_step(TimeScalar_ dt);

private:
    const TwoSiteObservable<Symmetry>& H;
    std::shared_ptr<iPEPS<Scalar, Symmetry>> Psi;
    UnitCell cell_;
    Update updater;

    void t_step_h(int x, int y, TimeScalar_ dt);
    void t_step_v(int x, int y, TimeScalar_ dt);
};

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "PEPS/TimePropagator.cpp"
#endif

#endif
