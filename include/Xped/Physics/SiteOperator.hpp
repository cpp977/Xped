#ifndef XPED_SITEOPERATOR_HPP_
#define XPED_SITEOPERATOR_HPP_

#include "Xped/AD/ADTensor.hpp"
#include "Xped/Core/Tensor.hpp"

namespace Xped {

template <typename Scalar_, typename Symmetry_>
struct SiteOperator
{
    using Scalar = Scalar_;
    using Symmetry = Symmetry_;
    using qType = typename Symmetry::qType;
    using MatrixType = typename Tensor<Scalar, 1, 2, Symmetry, false>::MatrixType;

    SiteOperator() = default;

    SiteOperator(qType Q, const Qbasis<Symmetry, 1>& basis, const mpi::XpedWorld& world = mpi::getUniverse());

    SiteOperator(qType Q,
                 const Qbasis<Symmetry, 1>& basis,
                 const std::unordered_map<std::string, std::pair<qType, std::size_t>>& labels,
                 const mpi::XpedWorld& world = mpi::getUniverse());

    const auto operator()(const qType& bra, const qType& ket, bool DUAL = false) const
    {
        FusionTree<2, Symmetry> k{
            .q_uncoupled = {ket, Q}, .q_coupled = bra, .dims = {data.uncoupledCodomain()[0].inner_dim(ket), 1}, .IS_DUAL = {DUAL, false}};
        k.computeDim();
        FusionTree<1, Symmetry> b{.q_uncoupled = {bra}, .q_coupled = bra, .dims = {data.coupledDomain().inner_dim(bra)}, .IS_DUAL = {DUAL}};
        b.computeDim();
        return data.subMatrix(b, k);
    }
    auto operator()(const qType& bra, const qType& ket, bool DUAL = false)
    {
        FusionTree<2, Symmetry> k{
            .q_uncoupled = {ket, Q}, .q_coupled = bra, .dims = {data.uncoupledCodomain()[0].inner_dim(ket), 1}, .IS_DUAL = {DUAL, false}};
        k.computeDim();
        FusionTree<1, Symmetry> b{.q_uncoupled = {bra}, .q_coupled = bra, .dims = {data.coupledDomain().inner_dim(bra)}, .IS_DUAL = {DUAL}};
        b.computeDim();
        return data.subMatrix(b, k);
    }

    const Scalar& operator()(const std::string& bra, const std::string& ket) const
    {
        auto it_bra = label_dict.find(bra);
        assert(it_bra != label_dict.end() and "label_dict in SiteOperator does not contain bra label");
        auto it_ket = label_dict.find(ket);
        assert(it_ket != label_dict.end() and "label_dict in SiteOperator does not contain ket label");

        return this->operator()(it_bra->second.first, it_ket->second.first)(it_bra->second.second, it_ket->second.second);
    }
    Scalar& operator()(const std::string& bra, const std::string& ket)
    {
        auto it_bra = label_dict.find(bra);
        assert(it_bra != label_dict.end() and "label_dict in SiteOperator does not contain bra label");
        auto it_ket = label_dict.find(ket);
        assert(it_ket != label_dict.end() and "label_dict in SiteOperator does not contain ket label");
        return this->operator()(it_bra->second.first, it_ket->second.first)(it_bra->second.second, it_ket->second.second);
    }

    SiteOperator<Scalar, Symmetry> adjoint() XPED_CONST;

    void setZero() { data.setZero(); }
    void setIdentity() { data.setIdentity(); }
    void setRandom() { data.setRandom(); }

    static SiteOperator<Scalar, Symmetry>
    prod(XPED_CONST SiteOperator<Scalar, Symmetry>& O1, XPED_CONST SiteOperator<Scalar, Symmetry>& O2, const qType& target);
    static SiteOperator<Scalar, Symmetry>
    outerprod(XPED_CONST SiteOperator<Scalar, Symmetry>& O1, XPED_CONST SiteOperator<Scalar, Symmetry>& O2, const qType& target);
    static SiteOperator<Scalar, Symmetry> outerprod(XPED_CONST SiteOperator<Scalar, Symmetry>& O1, XPED_CONST SiteOperator<Scalar, Symmetry>& O2)
    {
        auto target = Symmetry::reduceSilent(O1.Q, O2.Q);
        assert(target.size() == 1 and "Use outerprod overload with specification of fuse quantum number!");
        return SiteOperator<Scalar, Symmetry>::outerprod(O1, O2, target[0]);
    }

    Scalar norm() const;

    std::vector<MatrixType> plain() const;

    std::string print() const;
    // template <typename OtherScalar>
    // SiteOperator<OtherScalar, Symmetry> cast() const;

    std::string& label() { return label_; }
    Tensor<Scalar, 1, 2, Symmetry, false> data;
    qType Q;
    std::unordered_map<std::string, std::pair<qType, std::size_t>> label_dict;
    std::string label_;
};

template <typename Scalar, typename Symmetry, typename OtherScalar>
SiteOperator<std::common_type_t<Scalar, OtherScalar>, Symmetry> operator*(OtherScalar s, XPED_CONST SiteOperator<Scalar, Symmetry>& op)
{
    SiteOperator<std::common_type_t<Scalar, OtherScalar>, Symmetry> out(op.Q, op.data.coupledDomain(), op.data.world());
    out.data = s * op.data;
    return out;
}

template <typename Scalar, typename Symmetry, typename OtherScalar>
SiteOperator<std::common_type_t<Scalar, OtherScalar>, Symmetry> operator*(OtherScalar s, SiteOperator<Scalar, Symmetry>&& op)
{
    SiteOperator<Scalar, Symmetry>& tmp_op = op;
    return s * tmp_op;
}

template <typename Scalar, typename Symmetry, typename OtherScalar>
SiteOperator<std::common_type_t<Scalar, OtherScalar>, Symmetry> operator*(XPED_CONST SiteOperator<Scalar, Symmetry>& op, OtherScalar s)
{
    SiteOperator<std::common_type_t<Scalar, OtherScalar>, Symmetry> out(op.Q, op.data.coupledDomain(), op.data.world());
    out.data = s * op.data;
    return out;
}

template <typename Scalar, typename Symmetry>
SiteOperator<Scalar, Symmetry> operator*(XPED_CONST SiteOperator<Scalar, Symmetry>& O1, XPED_CONST SiteOperator<Scalar, Symmetry>& O2)
{
    auto Qtots = Symmetry::reduceSilent(O1.Q, O2.Q);
    assert(Qtots.size() == 1 and
           "The operator * for SiteOperator can only be used if the target quantumnumber is unique. Use SiteOperator::prod() instead.");
    auto Oout = SiteOperator<Scalar, Symmetry>::prod(O1, O2, Qtots[0]);
    return Oout;
}

template <typename Scalar, typename Symmetry>
SiteOperator<Scalar, Symmetry> operator+(XPED_CONST SiteOperator<Scalar, Symmetry>& O1, XPED_CONST SiteOperator<Scalar, Symmetry>& O2)
{
    assert(O1.Q == O2.Q and "For addition of SiteOperator the operator quantum number needs to be the same.");
    SiteOperator<Scalar, Symmetry> out(O1.Q, O1.data.coupledDomain());
    out.data = O1.data + O2.data;
    return out;
}

template <typename Scalar, typename Symmetry>
SiteOperator<Scalar, Symmetry> operator-(XPED_CONST SiteOperator<Scalar, Symmetry>& O1, XPED_CONST SiteOperator<Scalar, Symmetry>& O2)
{
    assert(O1.Q == O2.Q and "For subtraction of SiteOperator the operator quantum number needs to be the same.");
    SiteOperator<Scalar, Symmetry> out(O1.Q, O1.data.coupledDomain());
    out.data = O1.data - O2.data;
    return out;
}

template <typename Scalar, typename Symmetry>
SiteOperator<Scalar, Symmetry> kroneckerProduct(XPED_CONST SiteOperator<Scalar, Symmetry>& O1, XPED_CONST SiteOperator<Scalar, Symmetry>& O2)
{
    return SiteOperator<Scalar, Symmetry>::outerprod(O1, O2);
}

template <typename LeftScalar, typename RightScalar, typename Symmetry>
Tensor<std::common_type_t<LeftScalar, RightScalar>, 2, 2, Symmetry, false> tprod(XPED_CONST SiteOperator<LeftScalar, Symmetry>& O1,
                                                                                 XPED_CONST SiteOperator<RightScalar, Symmetry>& O2,
                                                                                 bool ADD_TWIST = false,
                                                                                 bool REVERSE_ORDER = false)
{
    if(not REVERSE_ORDER) {
        Qbasis<Symmetry, 1> Otarget_op;
        Otarget_op.push_back(Symmetry::qvacuum(), 1);
        Tensor<double, 2, 1, Symmetry, false> couple(
            {{O2.data.uncoupledCodomain()[1], O1.data.uncoupledCodomain()[1]}}, {{Otarget_op}}, O1.data.world());
        couple.setConstant(1.);
        auto tmp1 = ADD_TWIST ? O1.data.twist(2).template contract<std::array{-1, -2, 1}, std::array{-3, 1, -4}, 2>(couple)
                              : O1.data.template contract<std::array{-1, -2, 1}, std::array{-3, 1, -4}, 2>(couple);
        return tmp1.template contract<std::array{-1, -3, 1, -5}, std::array{-2, -4, 1}, 2>(O2.data).template trim<4>();
    } else {
        Qbasis<Symmetry, 1> Otarget_op;
        Otarget_op.push_back(Symmetry::qvacuum(), 1);
        Tensor<double, 2, 1, Symmetry, false> couple(
            {{O1.data.uncoupledCodomain()[1], O2.data.uncoupledCodomain()[1]}}, {{Otarget_op}}, O1.data.world());
        couple.setConstant(1.);
        auto tmp1 = ADD_TWIST ? O1.data.twist(2).template contract<std::array{-1, -2, 1}, std::array{1, -3, -4}, 2>(couple)
                              : O1.data.template contract<std::array{-1, -2, 1}, std::array{1, -3, -4}, 2>(couple);
        return tmp1.template contract<std::array{-1, -3, 1, -5}, std::array{-2, -4, 1}, 2>(O2.data).template trim<4>();
    }
}

template <typename LeftScalar, typename RightScalar, typename Symmetry>
auto tprod(SiteOperator<LeftScalar, Symmetry>&& O1, SiteOperator<RightScalar, Symmetry>&& O2)
{
    SiteOperator<LeftScalar, Symmetry> tmp_O1 = O1;
    SiteOperator<RightScalar, Symmetry> tmp_O2 = O2;
    return tprod(tmp_O1, tmp_O2);
}

template <typename Scalar, typename Symmetry>
std::ostream& operator<<(std::ostream& os, const SiteOperator<Scalar, Symmetry>& Op)
{
    os << Op.print();
    return os;
}

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "Physics/SiteOperator.cpp"
#endif

#endif
