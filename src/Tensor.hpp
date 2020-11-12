#ifndef TENSOR_H_
#define TENSOR_H_

#include <vector>
#include <unordered_set>
#include <string>
#include <sstream>

#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/KroneckerProduct>

#include "NestedLoopIterator.h"

#include "FusionTree.hpp"
#include "Qbasis.hpp"

namespace util{
                
        template<std::size_t Rank1, std::size_t Rank2, typename Symmetry>
        std::pair<Qbasis<Symmetry,Rank2+Rank1>,std::array<Qbasis<Symmetry,1>,0> >
        build_FusionTree_Helper(const Qbasis<Symmetry,Rank2>& coupled, const std::array<Qbasis<Symmetry,1>,Rank1>& uncoupled)
        {
                if constexpr(Rank1 == 0) {return std::make_pair(coupled, uncoupled);}
                else if constexpr(Rank1 == 1) {std::array<Qbasis<Symmetry,1>,0> trivial; return std::make_pair(coupled.combine(uncoupled[0]), trivial);}
                else {
                        std::array<Qbasis<Symmetry,1>,Rank1-1> new_uncoupled;
                        std::copy(uncoupled.begin()+1, uncoupled.end(), new_uncoupled.begin());
                        return build_FusionTree_Helper(coupled.combine(uncoupled[0]), new_uncoupled);
                }
        }

        template<std::size_t Rank, typename Symmetry>
        Qbasis<Symmetry,Rank> build_FusionTree( const std::array<Qbasis<Symmetry,1>,Rank>& uncoupled )
        {
                if constexpr (Rank == 0) {
                        Qbasis<Symmetry,0> tmp; tmp.push_back(Symmetry::qvacuum(),1);
                        return tmp;
                }
                else {
                        std::array<Qbasis<Symmetry,1>,Rank-1> basis_domain_shrinked;
                        std::copy(uncoupled.begin()+1, uncoupled.end(), basis_domain_shrinked.begin());
                        auto [domain_, discard] = util::build_FusionTree_Helper(uncoupled[0], basis_domain_shrinked);
                        return domain_;
                }
        }
        
        template<typename MatrixType_>
        MatrixType_ zero_init() {
                if constexpr (std::is_same<MatrixType_,Eigen::MatrixXd>::value) {return Eigen::MatrixXd::Zero(1,1);}
                else if constexpr (std::is_same<MatrixType_,Eigen::SparseMatrix<double> >::value) {Eigen::SparseMatrix<double> M(1,1); return M;}
                else if constexpr (std::is_same<MatrixType_,Eigen::DiagonalMatrix<double,-1> >::value) {Eigen::DiagonalMatrix<double,-1> M(1); M.diagonal() << 0.; return M;}
        }

        template<typename MatrixType_>
        string print_matrix(const MatrixType_& mat) {
                std::stringstream ss;
                if constexpr (std::is_same<MatrixType_,Eigen::MatrixXd>::value) {ss << mat; return ss.str();}
                else if constexpr (std::is_same<MatrixType_,Eigen::SparseMatrix<double> >::value) {ss << mat; return ss.str();}
                else if constexpr (std::is_same<MatrixType_,Eigen::DiagonalMatrix<double,-1> >::value) {ss << mat.toDenseMatrix(); return ss.str();}
        }

        template<typename TensorType>
        TensorType tensorProd(const TensorType& T1, const TensorType& T2)
        {
                std::array<Eigen::Index, T1.NumIndices> dims;
                for (std::size_t i=0; i<T1.rank(); i++) {
                        dims[i] = T1.dimensions()[i]*T2.dimensions()[i];
                }
                TensorType res(dims); res.setZero();
                std::array<Eigen::Index, T1.NumIndices> extents = T2.dimensions();

                std::vector<std::size_t> vec_dims; for (const auto& d:T1.dimensions()) {vec_dims.push_back(d);}
                NestedLoopIterator Nelly(T1.rank(), vec_dims);
        
                for (std::size_t i = Nelly.begin(); i!=Nelly.end(); i++) {
                        std::vector<std::size_t> indices;
                        for (std::size_t j=0; j<T1.rank(); j++) {indices.push_back(Nelly(j));}
                        std::array<Eigen::Index, T1.NumIndices> offsets;
                        for (std::size_t i=0; i<T1.rank(); i++) {
                                offsets[i] = indices[i] * T2.dimensions()[i];
                        }
        
                        res.slice(offsets,extents) = T1(indices) * T2;
                        ++Nelly;
                }
                return res;
        }
}

template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_=Eigen::MatrixXd, typename TensorType_=Eigen::Tensor<double,Rank+CoRank> >
class Tensor
{
        template<std::size_t Rank_, std::size_t CoRank_, std::size_t MiddleRank, typename Symmetry_, typename MatrixType__, typename TensorType__>
        friend Tensor<Rank_, CoRank_, Symmetry_, MatrixType__, TensorType__>
        operator*(const Tensor<Rank_,MiddleRank,Symmetry_,MatrixType__,TensorType__> &T1, const Tensor<MiddleRank,CoRank_,Symmetry_,MatrixType__,TensorType__> &T2);

        template<std::size_t Rank_, std::size_t CoRank_, typename Symmetry_, typename MatrixType__, typename TensorType__>
        friend Tensor<Rank_, CoRank_, Symmetry_, MatrixType__, TensorType__>
        operator+(const Tensor<Rank_,CoRank_,Symmetry_,MatrixType__,TensorType__> &T1, const Tensor<Rank_,CoRank_,Symmetry_,MatrixType__,TensorType__> &T2);

        template<std::size_t Rank_, std::size_t CoRank_, typename Symmetry_, typename MatrixType__, typename TensorType__>
        friend Tensor<Rank_, CoRank_, Symmetry_, MatrixType__, TensorType__>
        operator-(const Tensor<Rank_,CoRank_,Symmetry_,MatrixType__,TensorType__> &T1, const Tensor<Rank_,CoRank_,Symmetry_,MatrixType__,TensorType__> &T2);
        
        template<std::size_t Rank_, std::size_t CoRank_, typename Symmetry_, typename MatrixType__, typename TensorType__> friend class Tensor;
        typedef Tensor<Rank,CoRank,Symmetry,MatrixType_,TensorType_> self;
public:
        typedef MatrixType_ MatrixType;
        typedef TensorType_ TensorType;

        /**Does nothing.*/
        Tensor() {};

        Tensor(const std::array<Qbasis<Symmetry,1>,Rank> basis_domain, const std::array<Qbasis<Symmetry,1>,CoRank> basis_codomain);
                
        typedef typename Symmetry::qType qType;
        typedef typename MatrixType::Scalar Scalar;

        // Eigen::Map<MatrixType> operator() (const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2) const;
        //Eigen::TensorMap<TensorType> operator() (const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2);
        //Eigen::TensorMap<TensorType> operator() (const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2) const;

        Eigen::TensorMap<TensorType> view(const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2);
        Eigen::TensorMap<const TensorType> view(const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2) const;

        TensorType subBlock(const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2) const;
        // MatrixType& operator() (const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2);

        std::string print(bool PRINT_MATRICES=true) const;

        void setRandom();
        void setZero();
        void setIdentity();

        //Apply the basis transformation of domain and codomain to the block matrizes to get a plain array/tensor
        TensorType plainTensor() const;
        // MatrixType plainMatrix() const;
        
        Tensor<CoRank,Rank,Symmetry,MatrixType_,TensorType_> adjoint() const;
        self conjugate() const;
        Tensor<CoRank,Rank,Symmetry,MatrixType_,TensorType_> transpose() const;

        Scalar trace() const;

        Scalar squaredNorm() const {
                return (*this * this->adjoint()).trace();
        }

        Scalar norm() const {
                return std::sqrt(squaredNorm());
        }

        // template<std::size_t NewRank, std::size_t NewCoRank>
        // Tensor<NewRank,NewCoRank,Symmetry,MatrixType_,TensorType_> permute(const std::array<std::size_t,NewRank>& new_domain, const std::array<std::size_t,NewCoRank>& new_codomain) const;

        self permute(const Permutation<Rank>& p_domain, const Permutation<CoRank>& p_codomain) const;
        
        std::vector<FusionTree<Rank,Symmetry> > domain_trees(const qType &q) const {return domain.tree(q);}
        std::vector<FusionTree<CoRank,Symmetry> > codomain_trees(const qType &q) const {return codomain.tree(q);}
        
private:
        std::vector<MatrixType> block;

        std::unordered_map<qType, std::size_t> dict; //sector --> number
        std::vector<qType> sector;
        
        std::array<Qbasis<Symmetry,1>,Rank> uncoupled_domain;
        std::array<Qbasis<Symmetry,1>,CoRank> uncoupled_codomain;
        Qbasis<Symmetry, Rank> domain;
        Qbasis<Symmetry, CoRank> codomain;

        void push_back(const qType& q, const MatrixType& M) {
                block.push_back(M);
                sector.push_back(q);
                dict.insert(std::make_pair(q, sector.size()-1));
        }
};

template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorType_>
Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_>::
Tensor(const std::array<Qbasis<Symmetry,1>,Rank> basis_domain, const std::array<Qbasis<Symmetry,1>,CoRank> basis_codomain)
        :uncoupled_domain(basis_domain), uncoupled_codomain(basis_codomain)
{
        domain = util::build_FusionTree(basis_domain);
        codomain = util::build_FusionTree(basis_codomain);
}

template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorType_>
void Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_>::
setRandom()
{
        std::unordered_set<qType> uniqueController;
        for (const auto& [q,dim,plain]: domain) {
                if ( auto it=uniqueController.find(q); it == uniqueController.end() and codomain.IS_PRESENT(q)) {
                        sector.push_back(q); uniqueController.insert(q);
                        dict.insert(std::make_pair(q,sector.size()-1));
                }
        }
        block.resize(sector.size());
        for (size_t i=0; i<sector.size(); i++) {
                MatrixType mat(domain.inner_dim(sector[i]), codomain.inner_dim(sector[i])); mat.setRandom();
                block[i] = mat;
        }
}

template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorType_>
void Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_>::
setZero()
{
        std::unordered_set<qType> uniqueController;
        for (const auto& [q,dim,plain]: domain) {
                if ( auto it=uniqueController.find(q); it == uniqueController.end() and codomain.IS_PRESENT(q)) {
                        sector.push_back(q); uniqueController.insert(q);
                        dict.insert(std::make_pair(q,sector.size()-1));
                }
        }
        block.resize(sector.size());
        for (size_t i=0; i<sector.size(); i++) {
                MatrixType mat(domain.inner_dim(sector[i]), codomain.inner_dim(sector[i])); mat.setZero();
                block[i] = mat;
        }
}
template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorType_>
void Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_>::
setIdentity()
{
        std::unordered_set<qType> uniqueController;
        for (const auto& [q,dim,plain]: domain) {
                if ( auto it=uniqueController.find(q); it == uniqueController.end() and codomain.IS_PRESENT(q)) {
                        sector.push_back(q); uniqueController.insert(q);
                        dict.insert(std::make_pair(q,sector.size()-1));
                }
        }
        block.resize(sector.size());
        for (size_t i=0; i<sector.size(); i++) {
                MatrixType mat(domain.inner_dim(sector[i]), codomain.inner_dim(sector[i])); mat.setIdentity();
                block[i] = mat;
        }
}

template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorType_>
Tensor<CoRank, Rank, Symmetry, MatrixType_, TensorType_> Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_>::
adjoint() const
{
        Tensor<CoRank, Rank, Symmetry, MatrixType_, TensorType_> T;
        T.domain = codomain;
        T.codomain = domain;
        T.uncoupled_domain = uncoupled_codomain;
        T.uncoupled_codomain = uncoupled_domain;
        T.sector = sector;
        T.dict = dict;
        T.block.resize(T.sector.size());
        for (size_t i=0; i<sector.size(); i++) {
                T.block[i] = block[i].adjoint();
        }
        return T;
}

template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorType_>
Tensor<CoRank, Rank, Symmetry, MatrixType_, TensorType_> Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_>::
transpose() const
{
        Tensor<CoRank, Rank, Symmetry, MatrixType_, TensorType_> T;
        T.domain = codomain;
        T.codomain = domain;
        T.uncoupled_domain = uncoupled_codomain;
        T.uncoupled_codomain = uncoupled_domain;
        T.sector = sector;
        T.dict = dict;
        T.block.resize(T.sector.size());
        for (size_t i=0; i<sector.size(); i++) {
                T.block[i] = block[i].transpose();
        }
        return T;
}

template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorType_>
Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_> Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_>::
conjugate() const
{
        self T;
        T.domain = domain;
        T.codomain = codomain;
        T.uncoupled_domain = uncoupled_domain;
        T.uncoupled_codomain = uncoupled_codomain;
        T.sector = sector;
        T.dict = dict;
        T.block.resize(T.sector.size());
        for (size_t i=0; i<sector.size(); i++) {
                T.block[i] = block[i].conjugate();
        }
        return T;
}

template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorType_>
typename MatrixType_::Scalar Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_>::
trace() const
{
        assert(domain == codomain);
        Scalar out=0.;
        for (size_t i=0; i<sector.size(); i++) {
                out += block[i].trace() * Symmetry::degeneracy(sector[i]);
        }
        return out;
}

template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorType_>
Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_> Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_>::
permute(const Permutation<Rank>& p_domain, const Permutation<CoRank>& p_codomain) const
{
        std::array<std::size_t,Rank+CoRank> total_p;
        auto it_total = std::copy(p_domain.pi.begin(), p_domain.pi.end(), total_p.begin());
        auto pi_codomain_shifted = p_codomain.pi;
        std::for_each(pi_codomain_shifted.begin(), pi_codomain_shifted.end(), [] (std::size_t& elem) {return elem+=Rank;});
        std::copy(pi_codomain_shifted.begin(), pi_codomain_shifted.end(), it_total);
        self out;
        out.uncoupled_codomain = uncoupled_codomain;
        p_codomain.apply(out.uncoupled_codomain);
        
        out.uncoupled_domain = uncoupled_domain;
        p_domain.apply(out.uncoupled_domain);

        out.domain = util::build_FusionTree(out.uncoupled_domain);
        out.codomain = util::build_FusionTree(out.uncoupled_codomain);

        for (size_t i=0; i<sector.size(); i++) {
                auto domain_trees = domain.tree(sector[i]);
                auto codomain_trees = codomain.tree(sector[i]);
                for (const auto& domain_tree:domain_trees)
                for (const auto& codomain_tree:codomain_trees) {
                        auto permuted_domain_trees = domain_tree.permute(p_domain);
                        auto permuted_codomain_trees = codomain_tree.permute(p_codomain);
                        for (const auto& [permuted_domain_tree, coeff_domain]:permuted_domain_trees)
                        for (const auto& [permuted_codomain_tree, coeff_codomain]:permuted_codomain_trees) {
                                if (std::abs(coeff_domain*coeff_codomain) < 1.e-10) {continue;}

                                auto tensor = this->subBlock(domain_tree,codomain_tree);
                                TensorType Tshuffle = tensor.shuffle(total_p);
                                
                                auto it = out.dict.find(sector[i]);
                                if (it == out.dict.end()) {
                                        MatrixType mat(out.domain.inner_dim(sector[i]), out.codomain.inner_dim(sector[i])); mat.setZero();
                                        // MatrixType tmp = coeff_domain*coeff_codomain* Eigen::Map<MatrixType>(Tshuffle.data(),domain_tree.dim, codomain_tree.dim);
                                        // std::size_t i1 = out.domain.leftOffset(permuted_domain_tree);
                                        // std::size_t i2 = out.codomain.leftOffset(permuted_codomain_tree);
                                        // mat.block(i1, i2, permuted_domain_tree.dim, permuted_codomain_tree.dim) =
                                        //         tmp;
                                        mat.block(out.domain.leftOffset(permuted_domain_tree), out.codomain.leftOffset(permuted_codomain_tree), permuted_domain_tree.dim, permuted_codomain_tree.dim) =
                                                coeff_domain*coeff_codomain * Eigen::Map<MatrixType>(Tshuffle.data(),domain_tree.dim, codomain_tree.dim);
                                        out.push_back(sector[i], mat);
                                }
                                else {
                                        // auto tensor = this->subBlock(domain_tree,codomain_tree);
                                        // TensorType Tshuffle = tensor.shuffle(total_p);

                                        out.block[it->second].block(out.domain.leftOffset(permuted_domain_tree), out.codomain.leftOffset(permuted_codomain_tree),
                                                                    permuted_domain_tree.dim, permuted_codomain_tree.dim) +=
                                                coeff_domain*coeff_codomain * Eigen::Map<MatrixType>(Tshuffle.data(),domain_tree.dim, codomain_tree.dim);
                                }
                        }
                }
        }
        return out;
}

// template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorType_>
// MatrixType_& Tensor<Rank, CoRank, Symmetry, MatrixType_>::
// operator() (const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2)
// {
//         assert(f1.q_coupled == f2.q_coupled);
//         std::array<std::size_t,Rank> zeros_domain = std::array<std::size_t,Rank>();
//         std::array<std::size_t,CoRank> zeros_codomain = std::array<std::size_t,CoRank>();
//         const auto left_offset_domain = domain.leftOffset(f1,zeros_domain);
//         const auto left_offset_codomain = codomain.leftOffset(f2,zeros_codomain);
//         const auto it = dict.find(f1.q_coupled);
//         return block[it->second].block(left_offset_domain, left_offset_codomain, f1.dim, f2.dim);
// }

// template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorType_>
// Eigen::Map<MatrixType_> Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_>::
// operator() (const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2) const
// {
//         if(f1.q_coupled != f2.q_coupled) {return util::zero_init<MatrixType>();}
//         std::array<std::size_t,Rank> zeros_domain = std::array<std::size_t,Rank>();
//         std::array<std::size_t,CoRank> zeros_codomain = std::array<std::size_t,CoRank>();
//         const auto left_offset_domain = domain.leftOffset(f1,zeros_domain);
//         const auto left_offset_codomain = codomain.leftOffset(f2,zeros_codomain);
//         const auto it = dict.find(f1.q_coupled);
//         return block[it->second].block(left_offset_domain, left_offset_codomain, f1.dim, f2.dim);
// }

template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorType_>
Eigen::TensorMap<TensorType_> Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_>::
view (const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2)
{
        if(f1.q_coupled != f2.q_coupled) {assert(false);}
        
        const auto left_offset_domain = domain.leftOffset(f1);
        const auto left_offset_codomain = codomain.leftOffset(f2);
        const auto it = dict.find(f1.q_coupled);
        std::array<std::size_t,Rank+CoRank> dims;
        for (size_t i=0; i<Rank; i++) {dims[i] = uncoupled_domain[i].inner_dim(f1.q_uncoupled[i]);}
        for (size_t i=0; i<CoRank; i++) {dims[i+Rank] = uncoupled_codomain[i].inner_dim(f2.q_uncoupled[i]);}
        Eigen::TensorMap<TensorType> tensorview(block[it->second].block(left_offset_domain, left_offset_codomain, f1.dim, f2.dim).data(), dims);
        return tensorview;
        //return Eigen::TensorMap<TensorType>(block[it->second].block(left_offset_domain, left_offset_codomain, f1.dim, f2.dim).data(), dims);
}

template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorType_>
Eigen::TensorMap<const TensorType_> Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_>::
view (const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2) const
{
        if(f1.q_coupled != f2.q_coupled) {assert(false);}
        
        const auto left_offset_domain = domain.leftOffset(f1);
        const auto left_offset_codomain = codomain.leftOffset(f2);
        const auto it = dict.find(f1.q_coupled);
        std::array<std::size_t,Rank+CoRank> dims;
        for (size_t i=0; i<Rank; i++) {dims[i] = uncoupled_domain[i].inner_dim(f1.q_uncoupled[i]);}
        for (size_t i=0; i<CoRank; i++) {dims[i+Rank] = uncoupled_codomain[i].inner_dim(f2.q_uncoupled[i]);}
        // std::cout << "matrix subblock is:" << endl << block[it->second].block(left_offset_domain, left_offset_codomain, f1.dim, f2.dim) << endl;
        // return Eigen::TensorMap<const TensorType>(block[it->second].block(left_offset_domain, left_offset_codomain, f1.dim, f2.dim).data(), dims);
        Eigen::TensorMap<const TensorType> tensorview(block[it->second].block(left_offset_domain, left_offset_codomain, f1.dim, f2.dim).data(), dims);
        TensorType T(tensorview);
        return tensorview;
}

template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorType_>
TensorType_ Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_>::
subBlock (const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2) const
{
        if(f1.q_coupled != f2.q_coupled) {assert(false);}
        
        const auto left_offset_domain = domain.leftOffset(f1);
        const auto left_offset_codomain = codomain.leftOffset(f2);
        const auto it = dict.find(f1.q_coupled);
        std::array<std::size_t,Rank+CoRank> dims;
        for (size_t i=0; i<Rank; i++) {dims[i] = uncoupled_domain[i].inner_dim(f1.q_uncoupled[i]);}
        for (size_t i=0; i<CoRank; i++) {dims[i+Rank] = uncoupled_codomain[i].inner_dim(f2.q_uncoupled[i]);}
        MatrixType submatrix = block[it->second].block(left_offset_domain, left_offset_codomain, f1.dim, f2.dim);
        Eigen::TensorMap<const TensorType> tensorview(submatrix.data(), dims);
        TensorType T(tensorview);
        return T;
}

template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorType_>
TensorType_ Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_>::
plainTensor () const
{
        auto sorted_domain = domain;
        sorted_domain.sort();
        auto sorted_codomain = codomain;
        sorted_codomain.sort();
        auto sorted_uncoupled_domain = uncoupled_domain;
        std::for_each(sorted_uncoupled_domain.begin(), sorted_uncoupled_domain.end(), [] (Qbasis<Symmetry,1>& q) {q.sort();});
        auto sorted_uncoupled_codomain = uncoupled_codomain;
        std::for_each(sorted_uncoupled_codomain.begin(), sorted_uncoupled_codomain.end(), [] (Qbasis<Symmetry,1>& q) {q.sort();});

        std::vector<std::size_t> index_sort(sector.size());
	std::iota(index_sort.begin(),index_sort.end(),0);
	std::sort (index_sort.begin(), index_sort.end(), [this] (std::size_t n1, std::size_t n2) {
                qarray<Symmetry::Nq> q1 = sector[n1];
                qarray<Symmetry::Nq> q2 = sector[n2];
                return Symmetry::compare(std::array{q1},std::array{q2});
        });

        auto sorted_sector = sector;
        auto sorted_block = block;
        for (std::size_t i=0; i<sector.size(); i++) {
                sorted_sector[i] = sector[index_sort[i]];
                sorted_block[i] = block[index_sort[i]];
        }
        MatrixType inner_mat(sorted_domain.fullDim(), sorted_codomain.fullDim()); inner_mat.setZero();
        for (std::size_t i=0; i<sorted_sector.size(); i++) {
                inner_mat.block(sorted_domain.full_outer_num(sorted_sector[i]), sorted_codomain.full_outer_num(sorted_sector[i]),
                                Symmetry::degeneracy(sorted_sector[i])*sorted_block[i].rows(), Symmetry::degeneracy(sorted_sector[i])*sorted_block[i].cols()) =
                        Eigen::kroneckerProduct (sorted_block[i], MatrixType::Identity(Symmetry::degeneracy(sorted_sector[i]),Symmetry::degeneracy(sorted_sector[i])));
        }
        Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > map(inner_mat.data(), sorted_domain.fullDim(), sorted_codomain.fullDim());
        Eigen::Tensor<Scalar, 2> inner_tensor(map);
        
        Eigen::array<Eigen::Index,Rank+1> dims_domain;
        for (size_t i=0; i<Rank; i++) {dims_domain[i] = sorted_uncoupled_domain[i].fullDim();}
        dims_domain[Rank] = sorted_domain.fullDim();
        Eigen::Tensor<Scalar, Rank+1> unitary_domain(dims_domain); unitary_domain.setZero();

        for (const auto& [q,num,plain] : sorted_domain) {
                for (const auto& tree: sorted_domain.tree(q)) {
                        std::size_t uncoupled_dim=1;
                        for (std::size_t i=0; i<Rank; i++) {
                                uncoupled_dim *= sorted_uncoupled_domain[i].inner_dim(tree.q_uncoupled[i]);
                        }
                        MatrixType id(uncoupled_dim,uncoupled_dim); id.setIdentity();
                        Eigen::TensorMap<Eigen::Tensor<Scalar,2> > Tid_mat(id.data(),id.rows(), id.cols());
                        std::array<std::size_t, Rank+1> dims;
                        for (std::size_t i=0; i<Rank; i++) {
                                dims[i] = sorted_uncoupled_domain[i].inner_dim(tree.q_uncoupled[i]);
                        }
                        dims[Rank] = uncoupled_dim;
                        Eigen::Tensor<Scalar,Rank+1> Tid = Tid_mat.reshape(dims);
                        
                        auto T=tree.asTensor();
                        // auto tdims = T.dimensions();
                        // Eigen::Index product=1;
                        // for (std::size_t i=0; i<Rank; i++) {product *= tdims[i];}
                        // Eigen::Tensor<Scalar,2> Tmat = T.reshape(std::array<Eigen::Index,2>{{product,tdims[Rank]}});
                        // Eigen::Map<MatrixType> M(Tmat.data(),product,tdims[Rank]);
                        // MatrixType total = Eigen::kroneckerProduct(id,M);
                        // Eigen::TensorMap<Eigen::Tensor<Scalar,2> > Tfull_mat(total.data(),total.rows(), total.cols());
                        // cout << "Tfull matrix:" << endl << Tfull_mat << endl << endl;
                        // std::array<std::size_t, Rank+1> dims;
                        // for (std::size_t i=0; i<Rank; i++) {
                        //         dims[i] = sorted_uncoupled_domain[i].inner_dim(tree.q_uncoupled[i])*Symmetry::degeneracy(tree.q_uncoupled[i]);
                        // }
                        // dims[Rank] = uncoupled_dim*Symmetry::degeneracy(tree.q_coupled);
                        
                        Eigen::Tensor<Scalar,Rank+1> Tfull = util::tensorProd(Tid,T); //Tfull_mat.reshape(dims);
                        // Tfull.setZero(); Tfull(0,0,0) = 1.; Tfull(0,1,1) = 1.; Tfull(1,0,2) = 1.; Tfull(1,1,3) = 1.; Tfull(0,2,4) = 1.; Tfull(1,2,5) = 1.;
                                                
                        std::array<std::size_t, Rank+1> offsets;
                        for (std::size_t i=0; i<Rank; i++) {
                                offsets[i] = sorted_uncoupled_domain[i].full_outer_num(tree.q_uncoupled[i]);
                        }
                        offsets[Rank] = sorted_domain.full_outer_num(q) + sorted_domain.leftOffset(tree)*Symmetry::degeneracy(q);
                        
                        std::array<std::size_t, Rank+1> extents;
                        for (std::size_t i=0; i<Rank; i++) {
                                extents[i] = sorted_uncoupled_domain[i].inner_dim(tree.q_uncoupled[i]) * Symmetry::degeneracy(tree.q_uncoupled[i]);
                        }
                        extents[Rank] = Tfull.dimensions()[Rank];
                        unitary_domain.slice(offsets,extents) += Tfull; //+= or =?
                }
        }
        
        Eigen::array<Eigen::Index,CoRank+1> dims_codomain;
        for (size_t i=0; i<CoRank; i++) {dims_codomain[i] = sorted_uncoupled_codomain[i].fullDim();}
        dims_codomain[CoRank] = sorted_codomain.fullDim();
        Eigen::Tensor<Scalar, CoRank+1> unitary_codomain(dims_codomain); unitary_codomain.setZero();


        for (const auto& [q,num,plain] : sorted_codomain) {
                for (const auto& tree: sorted_codomain.tree(q)) {
                        std::size_t uncoupled_dim=1;
                        for (std::size_t i=0; i<CoRank; i++) {
                                uncoupled_dim *= sorted_uncoupled_codomain[i].inner_dim(tree.q_uncoupled[i]);
                        }
                        MatrixType id(uncoupled_dim,uncoupled_dim); id.setIdentity();
                        Eigen::TensorMap<Eigen::Tensor<Scalar,2> > Tid_mat(id.data(),id.rows(), id.cols());
                        std::array<std::size_t, CoRank+1> dims;
                        for (std::size_t i=0; i<CoRank; i++) {
                                dims[i] = sorted_uncoupled_codomain[i].inner_dim(tree.q_uncoupled[i]);
                        }
                        dims[CoRank] = uncoupled_dim;
                        Eigen::Tensor<Scalar,CoRank+1> Tid = Tid_mat.reshape(dims);
                        auto T=tree.asTensor();
                        
                        // auto tdims = T.dimensions();
                        // Eigen::Index product=1;
                        // for (std::size_t i=0; i<CoRank; i++) {product *= tdims[i];}
                        // Eigen::Tensor<Scalar,2> Tmat = T.reshape(std::array<Eigen::Index,2>{{product,tdims[CoRank]}});
                        // Eigen::Map<MatrixType> M(Tmat.data(),product,tdims[CoRank]);
                        // MatrixType total = Eigen::kroneckerProduct(M,id);

                        // Eigen::TensorMap<Eigen::Tensor<Scalar,2> > Tfull_mat(total.data(),total.rows(), total.cols());

                        // std::array<std::size_t, CoRank+1> dims;
                        // for (std::size_t i=0; i<CoRank; i++) {
                        //         dims[i] = sorted_uncoupled_codomain[i].inner_dim(tree.q_uncoupled[i])*Symmetry::degeneracy(tree.q_uncoupled[i]);
                        // }
                        // dims[CoRank] = uncoupled_dim*Symmetry::degeneracy(tree.q_coupled);
                        
                        Eigen::Tensor<Scalar,CoRank+1> Tfull = util::tensorProd(Tid,T); //Tfull_mat.reshape(dims);
                        std::array<std::size_t, CoRank+1> offsets;
                        for (std::size_t i=0; i<CoRank; i++) {
                                offsets[i] = sorted_uncoupled_codomain[i].full_outer_num(tree.q_uncoupled[i]);
                        }
                        offsets[CoRank] = sorted_codomain.full_outer_num(q) + sorted_codomain.leftOffset(tree)*Symmetry::degeneracy(q);
                        
                        std::array<std::size_t, CoRank+1> extents;
                        for (std::size_t i=0; i<CoRank; i++) {
                                extents[i] = sorted_uncoupled_codomain[i].inner_dim(tree.q_uncoupled[i]) * Symmetry::degeneracy(tree.q_uncoupled[i]);
                        }
                        extents[CoRank] = Tfull.dimensions()[CoRank];
                        unitary_codomain.slice(offsets,extents) = Tfull;
                }
        }
                        
        Eigen::array<Eigen::Index,Rank+CoRank> dims_result;
        for (size_t i=0; i<Rank; i++) {dims_result[i] = sorted_uncoupled_domain[i].fullDim();}
        for (size_t i=0; i<CoRank; i++) {dims_result[i+Rank] = sorted_uncoupled_codomain[i].fullDim();}
        TensorType out(dims_result); out.setZero();

        std::array<Eigen::IndexPair<std::size_t>, 1> legs_domain;
        legs_domain[0] = Eigen::IndexPair<std::size_t>(Rank, 0);
        std::array<Eigen::IndexPair<std::size_t>, 1> legs_codomain;
        legs_codomain[0] = Eigen::IndexPair<std::size_t>(Rank, CoRank);
        out = (unitary_domain.contract(inner_tensor, legs_domain)).contract(unitary_codomain, legs_codomain);
        return out;
        
}

template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorType_>
std::string Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_>::
print(bool PRINT_MATRICES) const
{
        std::stringstream ss;
        ss << "domain:" << endl << domain << endl << "with trees:" << endl << domain.printTrees() << endl;
        ss << "codomain:" << endl << codomain << endl << "with trees:" << endl << codomain.printTrees() << endl;
        for (size_t i=0; i<sector.size(); i++) {
                ss << "Sector with QN=" << sector[i] << endl;
                if (PRINT_MATRICES) { ss << util::print_matrix(block[i]) << endl;}
        }
        return ss.str();
}

template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorType_>
std::ostream& operator<<(std::ostream& os, const Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_> &t)
{
	os << t.print();
	return os;
}

template<std::size_t Rank, std::size_t CoRank, std::size_t MiddleRank, typename Symmetry, typename MatrixType_, typename TensorType_>
Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_>
operator* (const Tensor<Rank,MiddleRank,Symmetry,MatrixType_,TensorType_> &T1, const Tensor<MiddleRank,CoRank,Symmetry,MatrixType_,TensorType_> &T2)
{
        assert(T1.codomain == T2.domain);
        Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_> Tout;
        Tout.domain = T1.domain;
        Tout.codomain = T2.codomain;
        Tout.uncoupled_domain = T1.uncoupled_domain;
        Tout.uncoupled_codomain = T2.uncoupled_codomain;
        Tout.sector = T1.sector;
        Tout.dict = T1.dict;
        Tout.block.resize(Tout.sector.size());
        for (size_t i=0; i<T1.sector.size(); i++){
                auto it = T2.dict.find(T1.sector[i]);
                Tout.block[i] = T1.block[i] * T2.block[it->second];
        }
        return Tout;
}

template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorType_>
Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_>
operator+ (const Tensor<Rank,CoRank,Symmetry,MatrixType_,TensorType_> &T1, const Tensor<Rank,CoRank,Symmetry,MatrixType_,TensorType_> &T2)
{
        assert(T1.domain == T2.domain);
        assert(T1.codomain == T2.codomain);
        Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_> Tout;
        Tout.domain = T1.domain;
        Tout.codomain = T1.codomain;
        Tout.uncoupled_domain = T1.uncoupled_domain;
        Tout.uncoupled_codomain = T1.uncoupled_codomain;
        Tout.sector = T1.sector;
        Tout.dict = T1.dict;
        Tout.block.resize(Tout.sector.size());
        for (size_t i=0; i<T1.sector.size(); i++){
                auto it = T2.dict.find(T1.sector[i]);
                Tout.block[i] = T1.block[i] + T2.block[it->second];
        }
        return Tout;
}

template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorType_>
Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_>
operator- (const Tensor<Rank,CoRank,Symmetry,MatrixType_,TensorType_> &T1, const Tensor<Rank,CoRank,Symmetry,MatrixType_,TensorType_> &T2)
{
        assert(T1.domain == T2.domain);
        assert(T1.codomain == T2.codomain);
        Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_> Tout;
        Tout.domain = T1.domain;
        Tout.codomain = T1.codomain;
        Tout.uncoupled_domain = T1.uncoupled_domain;
        Tout.uncoupled_codomain = T1.uncoupled_codomain;
        Tout.sector = T1.sector;
        Tout.dict = T1.dict;
        Tout.block.resize(Tout.sector.size());
        for (size_t i=0; i<T1.sector.size(); i++){
                auto it = T2.dict.find(T1.sector[i]);
                Tout.block[i] = T1.block[i] - T2.block[it->second];
        }
        return Tout;
}
#endif
