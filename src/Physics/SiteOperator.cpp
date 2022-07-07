#include "Xped/Physics/SiteOperator.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry>
SiteOperator<Scalar, Symmetry>::SiteOperator(qType Q, const Qbasis<Symmetry, 1>& basis, mpi::XpedWorld& world)
    : Q(Q)
{
    Qbasis<Symmetry, 1> opBasis;
    opBasis.push_back(Q, 1);
    data = Tensor<Scalar, 2, 1, Symmetry, false>({{basis, opBasis}}, {{basis}}, world);
}

template <typename Scalar, typename Symmetry>
SiteOperator<Scalar, Symmetry>::SiteOperator(qType Q,
                                             const Qbasis<Symmetry, 1>& basis,
                                             const std::unordered_map<std::string, std::pair<qType, std::size_t>>& labels,
                                             mpi::XpedWorld& world)
    : Q(Q)
    , label_dict(labels)
{
    Qbasis<Symmetry, 1> opBasis;
    opBasis.push_back(Q, 1);
    data = Tensor<Scalar, 2, 1, Symmetry, false>({{basis, opBasis}}, {{basis}}, world);
}

template <typename Scalar, typename Symmetry>
SiteOperator<Scalar, Symmetry> SiteOperator<Scalar, Symmetry>::adjoint() const
{
    SiteOperator<Scalar, Symmetry> out(Symmetry::conj(Q), data.coupledCodomain());
    out.data = data.adjoint().eval().template permute<false, -1, 0, 2, 1>();
    return out;
}

template <typename Scalar, typename Symmetry>
SiteOperator<Scalar, Symmetry>
SiteOperator<Scalar, Symmetry>::prod(const SiteOperator<Scalar, Symmetry>& O1, const SiteOperator<Scalar, Symmetry>& O2, const qType& target)
{
    SiteOperator<Scalar, Symmetry> out(target, O1.data.coupledCodomain());
    Qbasis<Symmetry, 1> Otarget_op;
    Otarget_op.push_back(target, 1);
    Tensor<double, 1, 2, Symmetry, false> couple({{Otarget_op}}, {{O1.data.uncoupledDomain()[1], O2.data.uncoupledDomain()[1]}}, O1.data.world());
    couple.setConstant(1.);
    auto tmp1 = O1.data.template contract<std::array{-1, -2, 1}, std::array{1, -3, -4}, 2>(O2.data);
    out.data = couple.template contract<std::array{-2, 1, 2}, std::array{-1, 1, 2, -3}, 2>(tmp1);
    return out;
}

template <typename Scalar, typename Symmetry>
SiteOperator<Scalar, Symmetry>
SiteOperator<Scalar, Symmetry>::outerprod(const SiteOperator<Scalar, Symmetry>& O1, const SiteOperator<Scalar, Symmetry>& O2, const qType& target)
{
    auto fuse_bb =
        Tensor<Scalar, 2, 1, Symmetry, false>::Identity({{O1.data.uncoupledCodomain()[0], O2.data.uncoupledCodomain()[0]}},
                                                        {{O1.data.uncoupledCodomain()[0].combine(O2.data.uncoupledCodomain()[0]).forgetHistory()}});

    auto fuse_kk =
        Tensor<Scalar, 1, 2, Symmetry, false>::Identity({{O1.data.uncoupledDomain()[0].combine(O2.data.uncoupledDomain()[0]).forgetHistory()}},
                                                        {{O1.data.uncoupledDomain()[0], O2.data.uncoupledDomain()[0]}});

    SiteOperator<Scalar, Symmetry> out(target, O1.data.coupledCodomain().combine(O1.data.coupledCodomain()).forgetHistory());
    Qbasis<Symmetry, 1> Otarget_op;
    Otarget_op.push_back(target, 1);
    Tensor<double, 1, 2, Symmetry, false> couple({{Otarget_op}}, {{O1.data.uncoupledDomain()[1], O2.data.uncoupledDomain()[1]}}, O1.data.world());
    couple.setConstant(1.);
    auto tmp1 = couple.template contract<std::array{-1, 1, -2}, std::array{-3, 1, -4}, 2>(O1.data);
    auto tmp2 = tmp1.template contract<std::array{-3, 1, -1, -4}, std::array{-2, 1, -5}, 3>(O2.data);
    auto tmp3 = fuse_kk.template contract<std::array{-1, 1, 2}, std::array{1, 2, -2, -3, -4}, 2>(tmp2);
    out.data = tmp3.template contract<std::array{-1, -2, 1, 2}, std::array{1, 2, -3}, 2>(fuse_bb);

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
        auto M = PlainInterface::construct<Scalar>(data.uncoupledDomain()[0].fullDim(), data.uncoupledCodomain()[0].fullDim());
        for(std::size_t row = 0; row < data.uncoupledDomain()[0].fullDim(); ++row) {
            for(std::size_t col = 0; col < data.uncoupledCodomain()[0].fullDim(); ++col) {
                PlainInterface::setVal(
                    M, row, col, PlainInterface::getVal(T, {{static_cast<long>(row), static_cast<long>(i), static_cast<long>(col)}}));
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
    data.print(ss);
    return ss.str();
}

} // namespace Xped