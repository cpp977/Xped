#ifndef TENSOR_TRAITS_HPP__
#define TENSOR_TRAITS_HPP__

struct EigenTensorLib {};

template<typename Library>
struct tensortraits
{
        // constructor with dim array [x]
        // constructor with map [x]

        // initialization
        // setZero [x]
        // setRandom [x]
        // setConstant [x]
        
        // contract [x]
        // shuffle [x]

        // lvalue methods
        // shuffle ??? <-- not in Eigen
        // reshape with dims
        // slice

        // tensorProd [x]
        
        // return dimensions [x]
};

#include <unsupported/Eigen/CXX11/Tensor>
#include "NestedLoopIterator.h"

template<>
struct tensortraits<EigenTensorLib>
{
        // typedefs
        template<typename Scalar, std::size_t Rank> using Ttype = Eigen::Tensor<Scalar,Rank>;
        template<typename Scalar, std::size_t Rank> using cTtype = const Eigen::Tensor<Scalar,Rank>;

        template<typename Scalar, std::size_t Rank> using Maptype = Eigen::TensorMap<Ttype<Scalar,Rank> >;
        template<typename Scalar, std::size_t Rank> using cMaptype = Eigen::TensorMap<cTtype<Scalar,Rank> >;

        template<typename Scalar, std::size_t Rank> using Indextype = typename Eigen::Tensor<Scalar,Rank>::Index;
        
        // constructors
        template<typename Scalar, std::size_t Rank>
        static Ttype<Scalar, Rank> construct(const std::array<Indextype<Scalar,Rank>, Rank>& dims) {
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
        static cMaptype<Scalar, Rank> cMap(const Scalar* data, const std::array<Indextype<Scalar, Rank>, Rank>& dims) {
                return cMaptype<Scalar,Rank>(data,dims);
        }

        template<typename Scalar, std::size_t Rank>
        static Maptype<Scalar,Rank> Map(Scalar* data, const std::array<Indextype<Scalar, Rank>, Rank>& dims) {
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
        static void setConstant(Ttype<Scalar, Rank>& T) {
                T.setConstant();
        }
        
        // shape info
        template<typename Scalar, int Rank>
        static std::array<Indextype<Scalar,Rank>, Rank> dimensions(const Ttype<Scalar,Rank>& T) {
                std::array<Indextype<Scalar,Rank>, Rank> out;
                auto dims = T.dimensions();
                std::copy(dims.cbegin(), dims.cend(), out.begin());
                return out;
        }
        
        // tensorProd
        template<typename Scalar, int Rank>
        static Ttype<Scalar, Rank> tensorProd(const Ttype<Scalar, Rank>& T1, const Ttype<Scalar, Rank>& T2) {
                typedef Ttype<Scalar, Rank> TensorType;
                typedef Indextype<Scalar,Rank> Index;
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

        // methods
        template<typename Scalar, int Rank1, int Rank2, std::size_t Ncon>
        static Ttype<Scalar, Rank1+Rank2-2ul*Ncon> contract(const Ttype<Scalar, Rank1>& T1,
                                                            const Ttype<Scalar, Rank2>& T2,
                                                            const std::array<std::pair<Indextype<Scalar,Rank1>,Indextype<Scalar,Rank2> >, Ncon> con) {
                return T1.contract(T2,con);
        }
        
        template<typename Scalar, int Rank1, std::size_t Rank2>
        static Ttype<Scalar, Rank1> shuffle(const Ttype<Scalar, Rank1>& T, const std::array<Indextype<Scalar,Rank1>, Rank2>& dims) {
                static_assert(Rank1 == Rank2);
                return Ttype<Scalar, Rank1>(T.shuffle(dims));
        }

        // template<typename Scalar, int Rank>
        // static Ttype<Scalar, Rank> shuffle(const Ttype<Scalar, Rank>& T, const std::array<Indextype<Scalar,Rank>, Rank>& dims) {
        //         return Ttype<Scalar, Rank>(T.shuffle(dims));
        // }
};
#endif
