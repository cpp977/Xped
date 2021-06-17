#ifndef VECTOR_INTERFACE_EIGEN_IMPL_H_
#define VECTOR_INTERFACE_EIGEN_IMPL_H_

#include "Eigen/Dense"

template <>
struct VectorInterface<EigenVectorLib>
{
    // typedefs
    template <typename Scalar>
    using VType = Eigen::Array<Scalar, Eigen::Dynamic, 1>;

    typedef Eigen::Index VIndextype;

    // constructors
    template <typename Scalar>
    static VType<Scalar> construct(const VIndextype& elems)
    {
        return VType<Scalar>(elems);
    }

    template <typename Scalar>
    static VType<Scalar> construct_with_zero(const VIndextype& elems)
    {
        VType<Scalar> vec(elems);
        vec.setZero();
        return vec;
    }

    template <typename Scalar>
    static void resize(VType<Scalar>& V, const VIndextype& new_elems)
    {
        V.resize(new_elems);
    }

    // initialization
    template <typename Scalar>
    static void setZero(VType<Scalar>& V)
    {
        V.setZero();
    }

    template <typename Scalar>
    static void setRandom(VType<Scalar>& V)
    {
        V.setRandom();
    }

    template <typename Scalar>
    static void setConstant(VType<Scalar>& V, const Scalar& val)
    {
        V.setConstant(val);
    }

    // shape
    template <typename Scalar>
    static VIndextype length(const VType<Scalar>& V)
    {
        return V.size();
    }

    template <typename Scalar>
    static auto sum(const VType<Scalar>& V1, const VType<Scalar>& V2)
    {
        return (V1 + V2);
    }

    template <typename Scalar>
    static auto substract(const VType<Scalar>& V1, const VType<Scalar>& V2)
    {
        return (V1 - V2);
    }

    template <typename Scalar>
    static auto scale(const VType<Scalar>& V, const Scalar& val)
    {
        return (val * V);
    }

    // block
    template <typename Scalar>
    static auto sub(const VType<Scalar>& V, const VIndextype& off, const VIndextype& elems)
    {
        return V(Eigen::seqN(off, elems));
    }

    template <typename Scalar>
    static std::string print(const VType<Scalar>& V)
    {
        std::stringstream ss;
        ss << V;
        return ss.str();
    }
};

#endif
