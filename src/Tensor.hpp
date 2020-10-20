#ifndef TENSOR_H_
#define TENSOR_H_

#include <vector>
#include <unordered_set>
#include <string>
#include <sstream>

#include <unsupported/Eigen/CXX11/Tensor>

#include "FusionTree.hpp"
#include "Qbasis.hpp"

namespace util{
                
        template<std::size_t Rank1, std::size_t Rank2, typename Symmetry>
        std::pair<Qbasis<Symmetry,Rank2+Rank1>,std::array<Qbasis<Symmetry,1>,0> >
        build_FusionTree(const Qbasis<Symmetry,Rank2>& coupled, const std::array<Qbasis<Symmetry,1>,Rank1>& uncoupled)
        {
                if constexpr(Rank1 == 0) {return std::make_pair(coupled, uncoupled);}
                else if constexpr(Rank1 == 1) {std::array<Qbasis<Symmetry,1>,0> trivial; return std::make_pair(coupled.combine(uncoupled[0]), trivial);}
                else {
                        std::array<Qbasis<Symmetry,1>,Rank1-1> new_uncoupled;
                        std::copy(uncoupled.begin()+1, uncoupled.end(), new_uncoupled.begin());
                        return build_FusionTree(coupled.combine(uncoupled[0]), new_uncoupled);
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
}

template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_=Eigen::MatrixXd, typename TensorType_=Eigen::Tensor<double,Rank+CoRank> >
class Tensor
{
        template<std::size_t Rank_, std::size_t CoRank_, std::size_t MiddleRank, typename Symmetry_, typename MatrixType__, typename TensorType__>
        friend Tensor<Rank_, CoRank_, Symmetry_, MatrixType__, TensorType__>
        operator*(const Tensor<Rank_,MiddleRank,Symmetry_,MatrixType__,TensorType__> &T1, const Tensor<MiddleRank,CoRank_,Symmetry_,MatrixType__,TensorType__> &T2);
        
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

        MatrixType operator() (const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2) const;
        TensorType getBlockTensor(const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2) const;
        
        // MatrixType& operator() (const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2);

        std::string print() const;

        void setRandom();
        void setZero();
        void setIdentity();

        Tensor<CoRank,Rank,Symmetry,MatrixType_,TensorType_> adjoint() const;
        self conjugate() const;
        Tensor<CoRank,Rank,Symmetry,MatrixType_,TensorType_> transpose() const;
        
        std::vector<FusionTree<Rank,Symmetry> > domain_trees(const qType &q) const {return domain.tree(q);}
        std::vector<FusionTree<CoRank,Symmetry> > codomain_trees(const qType &q) const {return codomain.tree(q);}
        
private:
        // std::unordered_map<qType, MatrixType> block_map;
        std::unordered_map<qType, std::size_t> dict;
        std::vector<MatrixType> block;
        std::vector<qType> sector;
        std::array<Qbasis<Symmetry,1>,Rank> uncoupled_domain;
        std::array<Qbasis<Symmetry,1>,CoRank> uncoupled_codomain;
        Qbasis<Symmetry, Rank> domain;
        Qbasis<Symmetry, CoRank> codomain;
};

template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorType_>
Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_>::
Tensor(const std::array<Qbasis<Symmetry,1>,Rank> basis_domain, const std::array<Qbasis<Symmetry,1>,CoRank> basis_codomain)
        :uncoupled_domain(basis_domain), uncoupled_codomain(basis_codomain)
{
        if constexpr (Rank == 0) {
                Qbasis<Symmetry,0> tmp; tmp.push_back(Symmetry::qvacuum(),1);
                domain = tmp;
        }
        else {
                std::array<Qbasis<Symmetry,1>,Rank-1> basis_domain_shrinked;
                std::copy(basis_domain.begin()+1, basis_domain.end(), basis_domain_shrinked.begin());
                auto [domain_, discard] = util::build_FusionTree(basis_domain[0], basis_domain_shrinked);
                domain = domain_;
        }
                
        if constexpr (CoRank == 0) {
                Qbasis<Symmetry,0> tmp; tmp.push_back(Symmetry::qvacuum(),1);
                codomain = tmp;
        }
        else {
                std::array<Qbasis<Symmetry,1>,CoRank-1> basis_codomain_shrinked;
                std::copy(basis_codomain.begin()+1, basis_codomain.end(), basis_codomain_shrinked.begin());
                auto [codomain_, discard2] = util::build_FusionTree(basis_codomain[0], basis_codomain_shrinked);
                codomain = codomain_;
        }
        
        std::unordered_set<qType> uniqueController;
        for (const auto& [q,dim,plain]: domain) {
                if ( auto it=uniqueController.find(q); it == uniqueController.end() and codomain.IS_PRESENT(q)) {
                        sector.push_back(q); uniqueController.insert(q);
                        dict.insert(std::make_pair(q,sector.size()-1));
                }
        }
        block.resize(sector.size());
        for (size_t i=0; i<sector.size(); i++)
                {
                        MatrixType mat(domain.inner_dim(sector[i]), codomain.inner_dim(sector[i]));
                        block[i] = mat;
                }
}

template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorType_>
void Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_>::
setRandom()
{ 
        for (size_t i=0; i<sector.size(); i++) {
                block[i].setRandom();
        }
}

template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorType_>
void Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_>::
setZero()
{ 
        for (size_t i=0; i<sector.size(); i++) {
                block[i].setZero();
        }
}
template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorType_>
void Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_>::
setIdentity()
{ 
        for (size_t i=0; i<sector.size(); i++) {
                block[i].setIdentity();
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

template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorType_>
MatrixType_ Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_>::
operator() (const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2) const
{
        if(f1.q_coupled != f2.q_coupled) {return util::zero_init<MatrixType>();}
        std::array<std::size_t,Rank> zeros_domain = std::array<std::size_t,Rank>();
        std::array<std::size_t,CoRank> zeros_codomain = std::array<std::size_t,CoRank>();
        const auto left_offset_domain = domain.leftOffset(f1,zeros_domain);
        const auto left_offset_codomain = codomain.leftOffset(f2,zeros_codomain);
        const auto it = dict.find(f1.q_coupled);
        return block[it->second].block(left_offset_domain, left_offset_codomain, f1.dim, f2.dim);
}

template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorType_>
TensorType_ Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_>::
getBlockTensor (const FusionTree<Rank,Symmetry>& f1, const FusionTree<CoRank,Symmetry>& f2) const
{
        MatrixType M = this->operator()(f1,f2);
        std::array<std::size_t,Rank+CoRank> dims;
        for (size_t i=0; i<Rank; i++) {dims[i] = uncoupled_domain[i].inner_dim(f1.q_uncoupled[i]);}
        for (size_t i=0; i<CoRank; i++) {dims[i+Rank] = uncoupled_codomain[i].inner_dim(f2.q_uncoupled[i]);}
        if constexpr (std::is_same<MatrixType, Eigen::MatrixXd>::value) {
                Eigen::TensorMap<TensorType> t(M.data(),dims);
                return t;
        }
        else if constexpr (std::is_same<MatrixType, Eigen::SparseMatrix<double> >::value) {
                        Eigen::TensorMap<TensorType> t(Eigen::MatrixXd(M).data(),dims);
                        return t;
                }
}

template<std::size_t Rank, std::size_t CoRank, typename Symmetry, typename MatrixType_, typename TensorType_>
std::string Tensor<Rank, CoRank, Symmetry, MatrixType_, TensorType_>::
print() const
{
        std::stringstream ss;
        ss << "domain:" << endl << domain << endl;
        ss << "codomain:" << endl << codomain << endl;
        for (size_t i=0; i<sector.size(); i++) {ss << "Sector with QN=" << sector[i] << endl << util::print_matrix(block[i]) << endl;}
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
#endif
