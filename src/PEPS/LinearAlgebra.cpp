#include "Xped/PEPS/LinearAlgebra.hpp"

#include "spdlog/spdlog.h"

namespace Xped {

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
std::pair<PlainInterface::MType<std::conditional_t<ENABLE_AD, stan::math::var, Scalar>>,
          PlainInterface::MType<std::conditional_t<ENABLE_AD, stan::math::var, Scalar>>>
avg(XPED_CONST CTM<Scalar, Symmetry, ENABLE_AD>& env, XPED_CONST Tensor<Scalar, 2, 2, Symmetry, false>& op)
{
    assert(env.RDM_COMPUTED());

    PlainInterface::MType<std::conditional_t<ENABLE_AD, stan::math::var, Scalar>> o_h(env.cell.rows(), env.cell.cols());
    PlainInterface::MType<std::conditional_t<ENABLE_AD, stan::math::var, Scalar>> o_v(env.cell.rows(), env.cell.cols());
    for(int x = 0; x < env.cell.rows(); ++x) {
        for(int y = 0; y < env.cell.cols(); ++y) {
            if(not env.cell.pattern.isUnique(x, y)) {
                auto [x_unique, y_unique] = env.cell.pattern.coords(env.cell.pattern.uniqueIndex(x, y));
                o_h(x, y) = o_h(x_unique, y_unique);
                o_v(x, y) = o_v(x_unique, y_unique);
                continue;
            }
            o_h(x, y) = env.rho_h(x, y).template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(op).trace();
            o_v(x, y) = env.rho_v(x, y).template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(op).trace();
        }
    }
    return std::make_pair(o_h, o_v);
}
} // namespace Xped