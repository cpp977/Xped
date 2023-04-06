#include "Xped/Physics/SiteOperator.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry>
SiteOperator<Scalar, Symmetry>::SiteOperator(qType Q, const Qbasis<Symmetry, 1>& basis, const mpi::XpedWorld& world)
    : Q(Q)
{
    Qbasis<Symmetry, 1> opBasis;
    opBasis.push_back(Q, 1);
    data = Tensor<Scalar, 1, 2, Symmetry, false>({{basis}}, {{basis, opBasis}}, world);
}

template <typename Scalar, typename Symmetry>
SiteOperator<Scalar, Symmetry>::SiteOperator(qType Q,
                                             const Qbasis<Symmetry, 1>& basis,
                                             const std::unordered_map<std::string, std::pair<qType, std::size_t>>& labels,
                                             const mpi::XpedWorld& world)
    : Q(Q)
    , label_dict(labels)
{
    Qbasis<Symmetry, 1> opBasis;
    opBasis.push_back(Q, 1);
    data = Tensor<Scalar, 1, 2, Symmetry, false>({{basis}}, {{basis, opBasis}}, world);
}

template <typename Scalar, typename Symmetry>
SiteOperator<Scalar, Symmetry> SiteOperator<Scalar, Symmetry>::adjoint() XPED_CONST
{
    SiteOperator<Scalar, Symmetry> out(Symmetry::conj(Q), data.coupledDomain());
    out.data = data.adjoint().eval().template permute<+1, 0, 2, 1>();
    return out;
}

template <typename Scalar, typename Symmetry>
SiteOperator<Scalar, Symmetry> SiteOperator<Scalar, Symmetry>::prod(XPED_CONST SiteOperator<Scalar, Symmetry>& O1,
                                                                    XPED_CONST SiteOperator<Scalar, Symmetry>& O2,
                                                                    const qType& target)
{
    SiteOperator<Scalar, Symmetry> out(target, O1.data.coupledDomain());
    Qbasis<Symmetry, 1> Otarget_op;
    Otarget_op.push_back(target, 1);
    Tensor<double, 2, 1, Symmetry, false> couple({{O2.data.uncoupledCodomain()[1], O1.data.uncoupledCodomain()[1]}}, {{Otarget_op}}, O1.data.world());
    couple.setConstant(1.);
    auto tmp1 = O1.data.template contract<std::array{-1, 1, -2}, std::array{1, -3, -4}, 2>(O2.data);
    out.data = tmp1.template contract<std::array{-1, 1, -2, 2}, std::array{2, 1, -3}, 1>(couple);
    return out;
}

template <typename Scalar, typename Symmetry>
SiteOperator<Scalar, Symmetry> SiteOperator<Scalar, Symmetry>::outerprod(XPED_CONST SiteOperator<Scalar, Symmetry>& O1,
                                                                         XPED_CONST SiteOperator<Scalar, Symmetry>& O2,
                                                                         const qType& target)
{
    auto fuse_bb =
        Tensor<Scalar, 2, 1, Symmetry, false>::Identity({{O1.data.uncoupledCodomain()[0], O2.data.uncoupledCodomain()[0]}},
                                                        {{O1.data.uncoupledCodomain()[0].combine(O2.data.uncoupledCodomain()[0]).forgetHistory()}});

    auto fuse_kk =
        Tensor<Scalar, 1, 2, Symmetry, false>::Identity({{O1.data.uncoupledDomain()[0].combine(O2.data.uncoupledDomain()[0]).forgetHistory()}},
                                                        {{O1.data.uncoupledDomain()[0], O2.data.uncoupledDomain()[0]}});

    SiteOperator<Scalar, Symmetry> out(target, O1.data.coupledDomain().combine(O1.data.coupledDomain()).forgetHistory());
    Qbasis<Symmetry, 1> Otarget_op;
    Otarget_op.push_back(target, 1);
    Tensor<double, 2, 1, Symmetry, false> couple({{O2.data.uncoupledCodomain()[1], O1.data.uncoupledCodomain()[1]}}, {{Otarget_op}}, O1.data.world());
    couple.setConstant(1.);
    // auto tmp1 = couple.template contract<std::array{-1, 1, -2}, std::array{-3, -4, 1}, 2>(O1.data);
    auto tmp1 = O1.data.template contract<std::array{-1, -2, 1}, std::array{-3, 1, -4}, 2>(couple);
    auto tmp2 = O2.data.template contract<std::array{-2, -5, 1}, std::array{-1, -4, 1, -3}, 3>(tmp1);
    auto tmp3 = fuse_kk.template contract<std::array{-1, 1, 2}, std::array{1, 2, -2, -3, -4}, 2>(tmp2);
    out.data = tmp3.template contract<std::array{-1, -3, 1, 2}, std::array{1, 2, -2}, 1>(fuse_bb);

    // auto tmp1 = O1.data.template contract<std::array{-1, 1, -2}, std::array{1, -4, -3}, 2>(couple);
    // auto tmp2 = tmp1.template contract<std::array{-4, -1, 1, -3}, std::array{-2, 1, -5}, 3>(O2.data);
    // auto tmp3 = fuse_kk.template contract<std::array{-1, 1, 2}, std::array{1, 2, -2, -3, -4}, 2>(tmp2);
    // out.data = tmp3.template contract<std::array{-1, -2, 1, 2}, std::array{1, 2, -3}, 2>(fuse_bb);
    // out.data = (((O1.data.adjoint().eval().template permute<false, -1, 0, 1, 2>() * couple.template permute<+1, 0, 2, 1>())
    //                  .template permute<false, -1, 0, 1, 3, 2>()) *
    //             O2.data.template permute<false, +1, 1, 2, 0>())
    //                .template permute<false, 0, 0, 4, 2, 1, 3>();
    return out;
}

template <typename Scalar, typename Symmetry>
Scalar SiteOperator<Scalar, Symmetry>::norm() const
{
    return SiteOperator<Scalar, Symmetry>::prod(*this, *this, Symmetry::qvacuum()).data.norm();
}

template <typename Scalar, typename Symmetry>
std::vector<typename SiteOperator<Scalar, Symmetry>::MatrixType> SiteOperator<Scalar, Symmetry>::plain() const
{
    auto T = data.plainTensor();
    std::vector<MatrixType> out(Symmetry::degeneracy(Q));
    for(std::size_t i = 0; i < Symmetry::degeneracy(Q); ++i) {
        auto M = PlainInterface::construct<Scalar>(data.uncoupledDomain()[0].fullDim(), data.uncoupledCodomain()[0].fullDim(), data.world());
        for(std::size_t row = 0; row < data.uncoupledDomain()[0].fullDim(); ++row) {
            for(std::size_t col = 0; col < data.uncoupledCodomain()[0].fullDim(); ++col) {
                PlainInterface::setVal(
                    M, row, col, PlainInterface::getVal(T, {{static_cast<long>(row), static_cast<long>(col), static_cast<long>(i)}}));
            }
        }
        out[i] = M;
    }
    return out;
}

template <typename Scalar, typename Symmetry>
std::string SiteOperator<Scalar, Symmetry>::print() const
{
    std::stringstream ss;
    ss << "QN=" << Sym::format<Symmetry>(Q) << ": ";
    data.print(ss, true);
    return ss.str();
}

} // namespace Xped
