#ifndef TENSOR_TRAITS_HPP__
#define TENSOR_TRAITS_HPP__


struct EigenTensorLib {};

struct ArrayTensorLib {};

#ifdef XPED_USE_ARRAY_TENSOR_LIB
#define M_TENSORLIB ArrayTensorLib
#else
#define M_TENSORLIB EigenTensorLib
#endif

template<typename Library>
struct tensortraits
{
        // constructor with dim array [x] [x]
        // constructor with map [x] [x]

        // initialization
        // setZero [x] [x]
        // setRandom [x] [x]
        // setConstant [x] [x]
        
        // contract [x] [x]
        // shuffle [x] [x]

        // lvalue methods
        // shuffle ??? <-- not in Eigen
        // reshape with dims [x] [x]
        // slice [x] [x]

        // tensorProd [x] [x]
        
        // return dimensions [x] [x]
};

#include "NestedLoopIterator.h"
#include "seq/seq.h"

#ifdef XPED_USE_EIGEN_TENSOR_LIB
#include <unsupported/Eigen/CXX11/Tensor>
template<>
struct tensortraits<EigenTensorLib>
{
        // typedefs
        template<typename Scalar, std::size_t Rank> using Ttype = Eigen::Tensor<Scalar,Rank>;
        template<typename Scalar, std::size_t Rank> using cTtype = const Eigen::Tensor<Scalar,Rank>;

        template<typename Scalar, std::size_t Rank> using Maptype = Eigen::TensorMap<Ttype<Scalar,Rank> >;
        template<typename Scalar, std::size_t Rank> using cMaptype = Eigen::TensorMap<cTtype<Scalar,Rank> >;

        // template<typename Scalar, std::size_t Rank> using Indextype = typename Eigen::Tensor<Scalar,Rank>::Index;
        using Indextype = Eigen::Index;
        
        // constructors
        template<typename Scalar, std::size_t Rank>
        static Ttype<Scalar, Rank> construct(const std::array<Indextype, Rank>& dims) {
                return Ttype<Scalar, Rank>(dims);
        }

        template<typename Scalar, int Rank>
        static Ttype<Scalar, Rank> construct(const Maptype<Scalar,Rank>& map ) {
                return Ttype<Scalar, Rank>(map);
        }

        template<typename Scalar, int Rank>
        static cTtype<Scalar, Rank> construct(const cMaptype<Scalar,Rank>& map ) {
                return cTtype<Scalar, Rank>(map);
        }

        // map constructors
        template<typename Scalar, std::size_t Rank>
        static cMaptype<Scalar, Rank> cMap(const Scalar* data, const std::array<Indextype, Rank>& dims) {
                return cMaptype<Scalar,Rank>(data,dims);
        }

        template<typename Scalar, std::size_t Rank>
        static Maptype<Scalar,Rank> Map(Scalar* data, const std::array<Indextype, Rank>& dims) {
                return Maptype<Scalar,Rank>(data,dims);
        }

        // initialization
        template<typename Scalar, int Rank>
        static void setZero(Ttype<Scalar, Rank>& T) {
                T.setZero();
        }

        template<typename Scalar, int Rank>
        static void setRandom(Ttype<Scalar, Rank>& T) {
                T.setRandom();
        }

        template<typename Scalar, int Rank>
        static void setConstant(Ttype<Scalar, Rank>& T, const Scalar& val) {
                T.setConstant(val);
        }
        
        // shape info
        template<typename Scalar, int Rank>
        static std::array<Indextype, Rank> dimensions(const Ttype<Scalar,Rank>& T) {
                std::array<Indextype, Rank> out;
                auto dims = T.dimensions();
                std::copy(dims.cbegin(), dims.cend(), out.begin());
                return out;
        }
        
        // tensorProd
        template<typename Scalar, int Rank>
        static Ttype<Scalar, Rank> tensorProd(const Ttype<Scalar, Rank>& T1, const Ttype<Scalar, Rank>& T2) {
                typedef Ttype<Scalar, Rank> TensorType;
                typedef Indextype Index;
                std::array<Index, TensorType::NumIndices> dims;
                for (Index i=0; i<T1.rank(); i++) {
                        dims[i] = T1.dimensions()[i]*T2.dimensions()[i];
                }
                TensorType res(dims); res.setZero();
                std::array<Index, TensorType::NumIndices> extents = T2.dimensions();

                std::vector<std::size_t> vec_dims; for (const auto& d:T1.dimensions()) {vec_dims.push_back(d);}
                NestedLoopIterator Nelly(T1.rank(), vec_dims);
        
                for (std::size_t i = Nelly.begin(); i!=Nelly.end(); i++) {
                        std::array<Index, TensorType::NumIndices> indices;
                        for (Index j=0; j<T1.rank(); j++) {indices[j] = Nelly(j);}
                        std::array<Index, TensorType::NumIndices> offsets;
                        for (Index i=0; i<T1.rank(); i++) {
                                offsets[i] = indices[i] * T2.dimensions()[i];
                        }
        
                        res.slice(offsets,extents) = T1(indices) * T2;
                        ++Nelly;
                }
                return res;
        }

        // methods rvalue
        template<typename Scalar, std::size_t Rank1, std::size_t Rank2, Indextype... Is>
        static Ttype<double,Rank1+Rank2-sizeof...(Is)> contract (const Ttype<Scalar,Rank1>& T1, const Ttype<Scalar,Rank2>& T2) {
                static_assert(sizeof...(Is)%2 == 0);
                constexpr Indextype Ncon = sizeof...(Is)/2;
                static_assert(Rank1+Rank2>=Ncon);
                static_assert(Ncon <= 4, "Contractions of more than 4 legs is currently not implemented for EigenTensorLib");

                using seqC = seq::iseq<Indextype, Is...>;
                if constexpr (Ncon == 1) {
                        constexpr std::array<Eigen::IndexPair<Indextype >, Ncon> con = {Eigen::IndexPair<Indextype >(seq::at<0, seqC>, seq::at<1, seqC>)};
                        return T1.contract(T2,con);
                }
                else if constexpr (Ncon == 2) {
                        constexpr std::array<Eigen::IndexPair<Indextype >, Ncon> con = {Eigen::IndexPair<Indextype >(seq::at<0, seqC>, seq::at<1, seqC>),
                                                                                                      Eigen::IndexPair<Indextype >(seq::at<2, seqC>, seq::at<3, seqC>)};
                        return T1.contract(T2,con);
                }
                else if constexpr (Ncon == 3) {
                        constexpr std::array<Eigen::IndexPair<Indextype >, Ncon> con = {Eigen::IndexPair<Indextype >(seq::at<0, seqC>, seq::at<1, seqC>),
                                                                                                      Eigen::IndexPair<Indextype >(seq::at<2, seqC>, seq::at<3, seqC>),
                                                                                                      Eigen::IndexPair<Indextype >(seq::at<4, seqC>, seq::at<5, seqC>)};
                        return T1.contract(T2,con);
                }
                else if constexpr (Ncon == 4) {
                        constexpr std::array<Eigen::IndexPair<Indextype >, Ncon> con = {
                                Eigen::IndexPair<Indextype >(seq::at<0, seqC>, seq::at<1, seqC>),
                                Eigen::IndexPair<Indextype >(seq::at<2, seqC>, seq::at<3, seqC>),
                                Eigen::IndexPair<Indextype >(seq::at<4, seqC>, seq::at<5, seqC>),
                                Eigen::IndexPair<Indextype >(seq::at<6, seqC>, seq::at<7, seqC>)
                        };
                        return T1.contract(T2,con);
                }
        }

        template<typename Scalar, std::size_t Rank, Indextype... p>
        static Ttype<Scalar, Rank> shuffle(const Ttype<Scalar,Rank>& T)
        {
                static_assert(Rank == sizeof...(p));
                std::array<Indextype, Rank> dims = {p...};
                return Ttype<Scalar, Rank>(T.shuffle(dims));
        }

        template<typename Scalar, std::size_t Rank, Indextype... p>
        static Ttype<Scalar, Rank> shuffle(const Ttype<Scalar,Rank>& T, seq::iseq<Indextype, p...> s)
        {
                static_assert(Rank == sizeof...(p));
                std::array<Indextype, Rank> dims = {p...};
                return Ttype<Scalar, Rank>(T.shuffle(dims));
        }
        
        template<typename Scalar, int Rank1, std::size_t Rank2>
        static Ttype<Scalar,Rank2> reshape(const Ttype<Scalar, Rank1>& T, const std::array<Indextype, Rank2>& dims) {
                return T.reshape(dims);
        }

        //methods lvalue
        template<typename Scalar, std::size_t Rank1, std::size_t Rank2>
        static void setSubTensor(Ttype<Scalar, Rank1>& T, const std::array<Indextype, Rank2>& offsets, const std::array<Indextype, Rank2>& extents, 
                                 const Ttype<Scalar, Rank1>& S) {
                T.slice(offsets, extents) = S;
        }
        
        template<typename Scalar, int Rank1, std::size_t Rank2>
        static auto slice(Ttype<Scalar, Rank1>& T, const std::array<Indextype, Rank2>& offsets, const std::array<Indextype, Rank2>& extents) {
                return T.slice(offsets, extents);
        }

        template<typename Scalar, int Rank1, std::size_t Rank2>
        static auto reshape(Ttype<Scalar, Rank1>& T, const std::array<Indextype, Rank2>& dims) {
                return T.reshape(dims);
        }
};
#endif

#ifdef XPED_USE_ARRAY_TENSOR_LIB
#include <array/array.h>
#include <array/ein_reduce.h>

#include "../util/Random.hpp"

template<typename Index, Index oldVal, Index newVal, typename S>
using replace = seq::insert<seq::index_of<oldVal, S>,newVal,seq::remove<oldVal,S> >;

template<>
struct tensortraits<ArrayTensorLib>
{
        template <typename element_t, std::size_t N, size_t... Is>
        static nda::internal::tuple_of_n<element_t, N> as_tuple(std::array<element_t, N> const& arr,
                                                         std::index_sequence<Is...>)
        {
                return std::make_tuple(arr[Is]...);
        }

        template <typename element_t, std::size_t N>
        static nda::internal::tuple_of_n<element_t, N> as_tuple(std::array<element_t, N> const& arr)
        {
                return as_tuple(arr, std::make_index_sequence<N>{});
        }
        
        // typedefs
        template<typename Scalar, std::size_t Rank> using Ttype = nda::dense_array<Scalar, Rank>;
        template<typename Scalar, std::size_t Rank> using cTtype = const nda::dense_array<Scalar, Rank>;

        template<typename Scalar, std::size_t Rank> using Maptype = nda::dense_array_ref<Scalar, Rank>;
        template<typename Scalar, std::size_t Rank> using cMaptype = nda::const_dense_array_ref<Scalar, Rank>;

        // template<typename Scalar, std::size_t Rank> using Indextype = typename nda::dense_array<Scalar,Rank>::shape_type::index_type;
        using Indextype = nda::index_t;
        
        // constructors
        template<typename Scalar, std::size_t Rank>
        static Ttype<Scalar, Rank> construct(const std::array<Indextype, Rank>& dims) {
                return Ttype<Scalar,Rank>(as_tuple(dims));
        }

        template<typename Scalar, std::size_t Rank>
        static Ttype<Scalar, Rank> construct(const Maptype<Scalar,Rank>& map ) {
                Ttype<Scalar,Rank> out(map.shape());
                nda::copy(map,out);
                return out;
        }

        template<typename Scalar, std::size_t Rank>
        static cTtype<Scalar, Rank> construct(const cMaptype<Scalar,Rank>& map ) {
                Ttype<Scalar,Rank> tmp(map.shape());
                nda::copy(map,tmp);
                const cTtype<double,Rank> out(std::move(tmp));
                return out;
        }

        // map constructors
        template<typename Scalar, std::size_t Rank>
        static cMaptype<Scalar, Rank> cMap(const Scalar* data, const std::array<Indextype, Rank>& dims) {
                return cMaptype<Scalar,Rank>(data,as_tuple(dims));
        }

        template<typename Scalar, std::size_t Rank>
        static Maptype<Scalar,Rank> Map(Scalar* data, const std::array<Indextype, Rank>& dims) {
                return Maptype<Scalar,Rank>(data,as_tuple(dims));
        }

        // initialization
        template<typename Scalar, std::size_t Rank>
        static void setZero(Ttype<Scalar, Rank>& T) {
                T.for_each_value([](Scalar& d) {d=0.;});
        }

        template<typename Scalar, std::size_t Rank>
        static void setRandom(Ttype<Scalar, Rank>& T) {
                T.for_each_value([](Scalar& d) {d=util::random::threadSafeRandUniform<Scalar,Scalar>(-1., 1.);});
        }

        template<typename Scalar, std::size_t Rank>
        static void setConstant(Ttype<Scalar, Rank>& T, const Scalar& val) {
                T.for_each_value([val](Scalar& d) {d=val;});
        }
        
        // shape info
        template<typename Scalar, std::size_t Rank>
        static std::array<Indextype, Rank> dimensions(const Ttype<Scalar,Rank>& T) {
                std::array<Indextype, Rank> out;
                auto tmp = nda::internal::tuple_to_array<nda::dim<> >(T.shape().dims());
                for (std::size_t d=0; d<Rank; d++) {
                        out[d] = tmp[d].extent();
                }
                return out;
        }
        
        // tensorProd
        template<typename Scalar, std::size_t Rank>
        static Ttype<Scalar, Rank> tensorProd(const Ttype<Scalar, Rank>& T1, const Ttype<Scalar, Rank>& T2) {
                cMaptype<Scalar, Rank> T2m(T2.data(), T2.shape()); 
                typedef Ttype<Scalar, Rank> TensorType;
                typedef Indextype Index;
                std::array<Index, Rank> dims;
                std::array<Index, Rank> dim1_array;
                std::size_t tmp=0;
                for (const auto& d: nda::internal::tuple_to_array<nda::dim<> >(T1.shape().dims())) {dim1_array[tmp] = d.extent(); tmp++;}
                std::array<Index, Rank> dim2_array;
                tmp=0;
                for (const auto& d: nda::internal::tuple_to_array<nda::dim<> >(T2.shape().dims())) {dim2_array[tmp] = d.extent(); tmp++;}
                
                for (Index i=0; i<T1.rank(); i++) {
                        dims[i] = dim1_array[i]*dim2_array[i];
                }
        
                TensorType res(as_tuple(dims)); res.for_each_value([](Scalar& d) {d=0.;});
                std::array<Index, Rank> extents = dim2_array;

                std::vector<std::size_t> vec_dims; for (const auto& d:dim1_array) {vec_dims.push_back(d);}
                NestedLoopIterator Nelly(T1.rank(), vec_dims);
        
                for (std::size_t i = Nelly.begin(); i!=Nelly.end(); i++) {
                        std::array<Index, Rank> indices;
                        for (Index j=0; j<T1.rank(); j++) {indices[j] = Nelly(j);}
                        std::array<Index, Rank> offsets;
                        for (Index i=0; i<T1.rank(); i++) {
                                offsets[i] = indices[i] * dim2_array[i];
                        }

                        std::array<nda::interval<>, Rank> slices;
                        for (std::size_t r=0; r<Rank; r++) {slices[r] = nda::interval<>(offsets[r],extents[r]);}
                        nda::dense_shape<Rank> new_s(as_tuple(slices));
                        new_s.resolve();
                        T2m.set_shape(new_s);
                        nda::copy(T2m,res(as_tuple(slices)));
                        res(as_tuple(slices)).for_each_value([&T1, &indices](Scalar& val) {val=val*T1(as_tuple(indices));});
                        ++Nelly;
                }
                return res;
        }
 
        // methods rvalue
        template<typename Scalar, std::size_t Rank1, std::size_t Rank2, Indextype... Is1, Indextype... Is2, Indextype... Ist>
        static auto contract_helper (const Ttype<Scalar,Rank1>& T1, const Ttype<Scalar,Rank2>& T2,
                              seq::iseq<Indextype, Is1...> S1, seq::iseq<Indextype, Is2...> S2, seq::iseq<Indextype, Ist...> St)
        {
                return nda::make_ein_sum<Scalar,Ist...>(nda::ein<Is1...>(T1) * nda::ein<Is2...>(T2));
        }

        template<typename Scalar, std::size_t Rank1, std::size_t Rank2, Indextype... Is>
        static Ttype<double,Rank1+Rank2-sizeof...(Is)> contract (const Ttype<Scalar,Rank1>& T1, const Ttype<Scalar,Rank2>& T2) {
                static_assert(sizeof...(Is)%2 == 0);
                constexpr Indextype Ncon = sizeof...(Is)/2;
                static_assert(Ncon <= 4, "Contractions of more than 4 legs is currently not implemented for ArrayTensorLib");
                static_assert(Rank1+Rank2>=Ncon);
                constexpr Indextype Rank1i = static_cast<Indextype>(Rank1);
                constexpr Indextype Rank2i = static_cast<Indextype>(Rank2);
                
                using seq1 = seq::make<Indextype, Rank1i>; //0, 1, ..., Rank1-1
                using seq2 = seq::make<Indextype, Rank2i, Rank1i>; //Rank1, Rank+1, ..., Rank1+Rank2-1
                using seqC = seq::iseq<Indextype, Is...>;
                if constexpr (Ncon == 0) {
                        return contract_helper<double, Rank1, Rank2>(T1,T2,seq1{},seq2{},seq::concat<seq1,seq2>{});
                }
                else {
                        using mod_seq1 = replace<Indextype, seq::at<0,seqC>, 100, seq1>;
                        using mod_seq2 = replace<Indextype, seq::at<1,seqC>+Rank1i, 100, seq2>;
                        using rem_seq1 = seq::remove<seq::at<0,seqC>, seq1>;
                        using rem_seq2 = seq::remove<seq::at<1,seqC>+Rank1i, seq2>;
                        if constexpr (Ncon == 1) {
                                return contract_helper<double, Rank1, Rank2>(T1,T2,mod_seq1{},mod_seq2{},seq::concat<rem_seq1, rem_seq2>{});
                        }
                        else {
                                using mod_seq11 = replace<Indextype, seq::at<2,seqC>, 101, mod_seq1>;
                                using mod_seq22 = replace<Indextype, seq::at<3,seqC>+Rank1i, 101, mod_seq2>;
                                using rem_seq11 = seq::remove<seq::at<2,seqC>, rem_seq1>;
                                using rem_seq22 = seq::remove<seq::at<3,seqC>+Rank1i, rem_seq2>;
                                if constexpr (Ncon == 2) {
                                        return contract_helper<double, Rank1, Rank2>(T1,T2,mod_seq11{},mod_seq22{},seq::concat<rem_seq11, rem_seq22>{});
                                }
                                else {
                                        using mod_seq111 = replace<Indextype, seq::at<4,seqC>, 102, mod_seq11>;
                                        using mod_seq222 = replace<Indextype, seq::at<5,seqC>+Rank1i, 102, mod_seq22>;
                                        using rem_seq111 = seq::remove<seq::at<4,seqC>, rem_seq11>;
                                        using rem_seq222 = seq::remove<seq::at<5,seqC>+Rank1i, rem_seq22>;
                                        if constexpr (Ncon == 3) {
                                                return contract_helper<double, Rank1, Rank2>(T1,T2,mod_seq111{},mod_seq222{},seq::concat<rem_seq111, rem_seq222>{});
                                        }
                                        else {
                                                using mod_seq1111 = replace<Indextype, seq::at<6,seqC>, 103, mod_seq111>;
                                                using mod_seq2222 = replace<Indextype, seq::at<7,seqC>+Rank1i, 103, mod_seq222>;
                                                using rem_seq1111 = seq::remove<seq::at<6,seqC>, rem_seq111>;
                                                using rem_seq2222 = seq::remove<seq::at<7,seqC>+Rank1i, rem_seq222>;
                                                if constexpr (Ncon == 4) {
                                                        return contract_helper<double, Rank1, Rank2>(T1,T2,mod_seq1111{},mod_seq2222{},seq::concat<rem_seq1111, rem_seq2222>{});
                                                }       
                                        }
                                }
                        }
                }
        }

        template<typename Scalar, std::size_t Rank, Indextype... p>
        static Ttype<Scalar, Rank> shuffle(const Ttype<Scalar,Rank>& T)
        {
                static_assert(Rank == sizeof...(p));
                return construct<Scalar,Rank>(nda::transpose<p...>(T));
        }

        template<typename Scalar, std::size_t Rank, Indextype... p>
        static Ttype<Scalar, Rank> shuffle(const Ttype<Scalar,Rank>& T, seq::iseq<Indextype, p...> s)
        {
                static_assert(Rank == sizeof...(p));
                return construct<Scalar,Rank>(nda::transpose<p...>(T));
        }

        template<typename Scalar, std::size_t Rank1, std::size_t Rank2>
        static Ttype<Scalar,Rank2> reshape(const Ttype<Scalar, Rank1>& T, const std::array<Indextype, Rank2>& dims) {
                cMaptype<Scalar, Rank2> map(T.data(),as_tuple(dims));
                return construct<Scalar,Rank2>(map); 
        }

        //methods lvalue
        template<typename Scalar, std::size_t Rank1, std::size_t Rank2>
        static auto slice(Ttype<Scalar, Rank1>& T, const std::array<Indextype, Rank2>& offsets, const std::array<Indextype, Rank2>& extents) {
                std::array<nda::interval<>, Rank1> slices;
                for (std::size_t r=0; r<Rank1; r++) {slices[r] = nda::interval<>(offsets[r],extents[r]);}
                return T(as_tuple(slices));
        }

        template<typename Scalar, std::size_t Rank1, std::size_t Rank2>
        static void setSubTensor(Ttype<Scalar, Rank1>& T, const std::array<Indextype, Rank2>& offsets, const std::array<Indextype, Rank2>& extents, 
                                 const Ttype<Scalar, Rank1>& S) {
                cMaptype<Scalar,Rank1> Sm = S;
                std::array<nda::interval<>, Rank1> slices;
                for (std::size_t r=0; r<Rank1; r++) {slices[r] = nda::interval<>(offsets[r],extents[r]);}
                nda::dense_shape<Rank1> new_s(as_tuple(slices));
                new_s.resolve();
                Sm.set_shape(new_s);
                nda::copy(Sm,T(as_tuple(slices)));
                // return T(as_tuple(slices));
        }

        template<typename Scalar, std::size_t Rank1, std::size_t Rank2>
        static auto reshape(Ttype<Scalar, Rank1>& T, const std::array<Indextype, Rank2>& dims) {
                Maptype<Scalar, Rank2> map(T.data(),as_tuple(dims));
                return map;
        }
};
#endif

#endif
