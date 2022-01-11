#include "Xped/Interfaces/VectorInterface.hpp"

#include "Xped/Interfaces/VectorInterface_Eigen_impl.hpp"

namespace Xped {

// typedefs
template <typename Scalar>
using VType = Eigen::Array<Scalar, Eigen::Dynamic, 1>;

typedef Eigen::Index VIndextype;

// constructors
template <typename Scalar>
VType<Scalar> VectorInterface<EigenVectorLib>::construct(const VIndextype& elems, mpi::XpedWorld& world)
{
    return VType<Scalar>(elems);
}

template <typename Scalar>
VType<Scalar> VectorInterface<EigenVectorLib>::construct_with_zero(const VIndextype& elems, mpi::XpedWorld& world)
{
    VType<Scalar> vec(elems);
    vec.setZero();
    return vec;
}

template <typename Scalar>
void VectorInterface<EigenVectorLib>::resize(VType<Scalar>& V, const VIndextype& new_elems)
{
    V.resize(new_elems);
}

// initialization
template <typename Scalar>
void VectorInterface<EigenVectorLib>::setZero(VType<Scalar>& V)
{
    V.setZero();
}

template <typename Scalar>
void VectorInterface<EigenVectorLib>::setRandom(VType<Scalar>& V)
{
    V.setRandom();
}

template <typename Scalar>
void VectorInterface<EigenVectorLib>::setConstant(VType<Scalar>& V, const Scalar& val)
{
    V.setConstant(val);
}

// shape
template <typename Scalar>
VIndextype VectorInterface<EigenVectorLib>::length(const VType<Scalar>& V)
{
    return V.size();
}

template <typename Scalar>
const Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar, Scalar>, const VType<Scalar>, const VType<Scalar>>
VectorInterface<EigenVectorLib>::sum(const VType<Scalar>& V1, const VType<Scalar>& V2)
{
    return (V1 + V2);
}

template <typename Scalar>
const Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar, Scalar>, const VType<Scalar>, const VType<Scalar>>
VectorInterface<EigenVectorLib>::substract(const VType<Scalar>& V1, const VType<Scalar>& V2)
{
    return (V1 - V2);
}

template <typename Scalar>
const Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<Scalar, Scalar>, Scalar, const VType<Scalar>>
VectorInterface<EigenVectorLib>::scale(const VType<Scalar>& V, const Scalar& val)
{
    return (val * V);
}

// block
template <typename Scalar>
const Eigen::VectorBlock<const VType<Scalar>>
VectorInterface<EigenVectorLib>::sub(const VType<Scalar>& V, const VIndextype& off, const VIndextype& elems)
{
    return V(Eigen::seqN(off, elems));
}

template <typename Scalar>
std::string VectorInterface<EigenVectorLib>::print(const VType<Scalar>& V)
{
    std::stringstream ss;
    ss << V;
    return ss.str();
}

template <typename Scalar>
void VectorInterface<EigenVectorLib>::vec_to_stdvec(const VType<Scalar>& V, std::vector<Scalar>& vec)
{
    vec = std::vector<Scalar>(V.data(), V.data() + V.size());
}

} // namespace Xped
