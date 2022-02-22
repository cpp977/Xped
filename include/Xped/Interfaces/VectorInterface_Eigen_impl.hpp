#ifndef VECTOR_INTERFACE_EIGEN_IMPL_H_
#define VECTOR_INTERFACE_EIGEN_IMPL_H_

#include "Eigen/Dense"

#include "Xped/Util/Mpi.hpp"

namespace Xped {

struct VectorInterface
{
    // typedefs
    template <typename Scalar>
    using VType = Eigen::Array<Scalar, Eigen::Dynamic, 1>;

    typedef Eigen::Index VIndextype;

    // constructors
    template <typename Scalar>
    static VType<Scalar> construct(const VIndextype& elems, mpi::XpedWorld& world = mpi::getUniverse());

    template <typename Scalar>
    static VType<Scalar> construct_with_zero(const VIndextype& elems, mpi::XpedWorld& world = mpi::getUniverse());

    template <typename Scalar>
    static void resize(VType<Scalar>& V, const VIndextype& new_elems);

    // initialization
    template <typename Scalar>
    static void setZero(VType<Scalar>& V);

    template <typename Scalar>
    static void setRandom(VType<Scalar>& V);

    template <typename Scalar>
    static void setConstant(VType<Scalar>& V, const Scalar& val);

    // shape
    template <typename Scalar>
    static VIndextype length(const VType<Scalar>& V);

    template <typename Scalar>
    static const Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar, Scalar>, const VType<Scalar>, const VType<Scalar>>
    sum(const VType<Scalar>& V1, const VType<Scalar>& V2);

    template <typename Scalar>
    static const Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar, Scalar>, const VType<Scalar>, const VType<Scalar>>
    substract(const VType<Scalar>& V1, const VType<Scalar>& V2);

    template <typename Scalar>
    static const Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<Scalar, Scalar>, Scalar, const VType<Scalar>> scale(const VType<Scalar>& V,
                                                                                                                             const Scalar& val);

    // block
    template <typename Scalar>
    static const Eigen::VectorBlock<const VType<Scalar>> sub(const VType<Scalar>& V, const VIndextype& off, const VIndextype& elems);

    template <typename Scalar>
    static std::string print(const VType<Scalar>& V);

    template <typename Scalar>
    static void vec_to_stdvec(const VType<Scalar>& V, std::vector<Scalar>& vec);
};

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "Interfaces/VectorInterface_Eigen_impl.cpp"
#endif

#endif
