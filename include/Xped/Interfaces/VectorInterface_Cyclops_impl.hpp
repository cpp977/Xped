#ifndef VECTOR_INTERFACE_CYCLOPS_IMPL_H_
#define VECTOR_INTERFACE_CYCLOPS_IMPL_H_

#include "ctf.hpp"

template <>
struct VectorInterface<CyclopsVectorLib>
{
    // typedefs
    template <typename Scalar>
    using VType = CTF::Vector<Scalar>;

    typedef int VIndextype;

    // constructors
    template <typename Scalar>
    static VType<Scalar> construct(const VIndextype& elems, CTF::World& world)
    {
        return VType<Scalar>(elems, world);
    }

    template <typename Scalar>
    static VType<Scalar> construct_with_zero(const VIndextype& elems, CTF::World& world)
    {
        return VType<Scalar>(elems, world);
    }

    template <typename Scalar>
    static void resize(VType<Scalar>& V, const VIndextype& new_elems)
    {
        V = VType<Scalar>(new_elems, *V.wrld);
    }

    // initialization
    template <typename Scalar>
    static void setZero(VType<Scalar>& V)
    {
        V["i"] = 0;
    }

    template <typename Scalar>
    static void setRandom(VType<Scalar>& V)
    {
        V.fill_random();
    }

    template <typename Scalar>
    static void setConstant(VType<Scalar>& V, const Scalar& val)
    {
        V["i"] = val;
    }

    // shape
    template <typename Scalar>
    static VIndextype length(const VType<Scalar>& V)
    {
        return V.len;
    }

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
    static VType<Scalar> scale(VT1&& V, const Scalar& val)
    {
        VType<Scalar> res(V.len, *V.wrld);
        res["i"] = val * V["i"];
        return res;
    }

    // block
    template <typename Scalar>
    static auto sub(const VType<Scalar>& V, const VIndextype& off, const VIndextype& elems)
    {
        std::array<int, 1> offs = {off};
        std::array<int, 1> ends = {off + elems};
        return V.scale(offs.data(), ends.data());
    }

    template <typename Scalar>
    static std::string print(const VType<Scalar>& V)
    {
        return V.print();
    }

    template <typename Scalar, typename VT>
    static void vec_to_stdvec(VT&& V, std::vector<Scalar>& vec)
    {
        int64_t nvals;
        Scalar* data;
        V.get_all_data(&nvals, &data);
        vec = std::vector<Scalar>(data, data + nvals);
        delete[] data;
    }
};

#endif
