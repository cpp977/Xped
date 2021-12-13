#include "Xped/Interfaces/VectorInterface.hpp"

#include "Xped/Interfaces/VectorInterface_Cyclops_impl.hpp"

// typedefs
template <typename Scalar>
using VType = CTF::Vector<Scalar>;

typedef int VIndextype;

// constructors
template <typename Scalar>
VType<Scalar> VectorInterface<CyclopsVectorLib>::construct(const VIndextype& elems, CTF::World& world)
{
    return VType<Scalar>(elems, world);
}

template <typename Scalar>
VType<Scalar> VectorInterface<CyclopsVectorLib>::construct_with_zero(const VIndextype& elems, CTF::World& world)
{
    return VType<Scalar>(elems, world);
}

template <typename Scalar>
void VectorInterface<CyclopsVectorLib>::resize(VType<Scalar>& V, const VIndextype& new_elems)
{
    V = VType<Scalar>(new_elems, *V.wrld);
}

// initialization
template <typename Scalar>
void VectorInterface<CyclopsVectorLib>::setZero(VType<Scalar>& V)
{
    V["i"] = 0;
}

template <typename Scalar>
void VectorInterface<CyclopsVectorLib>::setRandom(VType<Scalar>& V)
{
    V.fill_random();
}

template <typename Scalar>
void VectorInterface<CyclopsVectorLib>::setConstant(VType<Scalar>& V, const Scalar& val)
{
    V["i"] = val;
}

// shape
template <typename Scalar>
VIndextype VectorInterface<CyclopsVectorLib>::length(const VType<Scalar>& V)
{
    return V.len;
}

// template <typename Scalar, typename VT1, typename VT2>
// auto add(VT1&& V1, VT2&& V2)
// {
//     VType<Scalar> res(V1.len);
//     res["i"] = V1["i"] + V2["i"];
//     return res;
// }

// template <typename Scalar, typename VT1, typename VT2>
// auto difference(VT1&& V1, VT2&& V2)
// {
//     VType<Scalar> res(V1.len);
//     res["i"] = V1["i"] - V2["i"];
//     return res;
// }

template <typename Scalar, typename VT1>
VType<Scalar> VectorInterface<CyclopsVectorLib>::scale(VT1&& V, const Scalar& val)
{
    VType<Scalar> res(V.len, *V.wrld);
    res["i"] = val * V["i"];
    return res;
}

// block
template <typename Scalar>
VType<Scalar> VectorInterface<CyclopsVectorLib>::sub(const VType<Scalar>& V, const VIndextype& off, const VIndextype& elems)
{
    std::array<int, 1> offs = {off};
    std::array<int, 1> ends = {off + elems};
    return V.slice(offs.data(), ends.data());
}

template <typename Scalar>
std::string VectorInterface<CyclopsVectorLib>::print(const VType<Scalar>& V)
{
    return V.print();
}

template <typename Scalar, typename VT>
void VectorInterface<CyclopsVectorLib>::vec_to_stdvec(VT&& V, std::vector<Scalar>& vec)
{
    int64_t nvals;
    Scalar* data;
    V.get_all_data(&nvals, &data);
    vec = std::vector<Scalar>(data, data + nvals);
    delete[] data;
}