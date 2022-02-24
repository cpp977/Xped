#ifndef VECTOR_INTERFACE_CYCLOPS_IMPL_H_
#define VECTOR_INTERFACE_CYCLOPS_IMPL_H_

#include "ctf.hpp"

namespace Xped {

struct VectorInterface
{
    // typedefs
    template <typename Scalar>
    using VType = CTF::Vector<Scalar>;

    typedef int VIndextype;

    // constructors
    template <typename Scalar>
    static VType<Scalar> construct(const VIndextype& elems, CTF::World& world);

    template <typename Scalar>
    static VType<Scalar> construct_with_zero(const VIndextype& elems, CTF::World& world);

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

    // template <typename Scalar, typename VT1, typename VT2>
    // static auto add(VT1&& V1, VT2&& V2)
    // {
    //     VType<Scalar> res(V1.len);
    //     res["i"] = V1["i"] + V2["i"];
    //     return res;
    // }

    // template <typename Scalar, typename VT1, typename VT2>
    // static auto difference(VT1&& V1, VT2&& V2)
    // {
    //     VType<Scalar> res(V1.len);
    //     res["i"] = V1["i"] - V2["i"];
    //     return res;
    // }

    template <typename Scalar, typename VT1>
    static VType<Scalar> scale(VT1&& V, const Scalar& val);

    // block
    template <typename Scalar>
    static VType<Scalar> sub(const VType<Scalar>& V, const VIndextype& off, const VIndextype& elems);

    template <typename Scalar>
    static std::string print(const VType<Scalar>& V);

    template <typename Scalar, typename VT>
    static void vec_to_stdvec(VT&& V, std::vector<Scalar>& vec);
};

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "Interfaces/VectorInterface_Cyclops_impl.cpp"
#endif

#endif
