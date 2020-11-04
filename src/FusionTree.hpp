#ifndef FUSIONTREE_H_
#define FUSIONTREE_H_

#include <unsupported/Eigen/CXX11/Tensor>

#include "hash/hash.hpp"
#include "numeric_limits.h"
#include "Permutations.h"

namespace util
{
        constexpr std::size_t inter_dim(std::size_t Rank) {return (Rank == 1 or Rank == 0) ? 0 : Rank-2; }
        constexpr std::size_t mult_dim(std::size_t Rank) {return (Rank == 0) ? 0 : Rank-1; }
        constexpr std::size_t numberOfVertices(std::size_t Rank) {return (Rank == 0) ? 0 : Rank-1; }
        constexpr std::size_t numberOfInnerlines(std::size_t Rank) {return (Rank == 1 or Rank == 0) ? 0 : Rank-2; }
}

#include "FusionTreePrint.hpp"

template<std::size_t Rank, typename Symmetry>
struct FusionTree
{
        typedef typename Symmetry::qType qType;
        typedef typename Symmetry::Scalar Scalar;
        
        std::array<qType, Rank> q_uncoupled;
        qType q_coupled;
        std::size_t dim;
        std::array<qType, util::inter_dim(Rank)> q_intermediates;
        std::array<size_t, util::mult_dim(Rank)> multiplicities=std::array<size_t, util::mult_dim(Rank)>(); //only for non-Abelian symmetries with outermultiplicity.
        std::array<bool, Rank> IS_DUAL;

        bool operator< (const FusionTree<Rank,Symmetry>& other) const
        {
                if (Symmetry::compare(q_uncoupled,other.q_uncoupled)) return true;
                else {
                        if (q_uncoupled != other.q_uncoupled) {return false;}
                        else {
                                if (Symmetry::compare(q_intermediates,other.q_intermediates)) return true;
                                else {
                                        if (q_intermediates != other.q_intermediates) {return false;}
                                        else {
                                                if (q_coupled < other.q_coupled) return true;
                                                else {
                                                        if (q_coupled != other.q_coupled) {return false;}
                                                        else {
                                                                if (multiplicities < other.multiplicities) return true;
                                                                else {return false;}
                                                        }
                                                }
                                        }
                                        
                                }
                        }
                }
        }
        
        bool operator== (const FusionTree<Rank,Symmetry>& other) const
        {                
                return 
                        q_uncoupled == other.q_uncoupled and
                        q_intermediates == other.q_intermediates and
                        multiplicities == other.multiplicities and
                        q_coupled == other.q_coupled;
        }

        bool operator!= (const FusionTree<Rank,Symmetry>& other) const {return !this->operator==(other);}

        std::string draw() const {
                auto print_aligned = [](const qType& q, std::size_t alignment=4) -> std::string {
                        std::stringstream ss;
                        ss << Sym::format<Symmetry>(q);
                        std::string tmp = ss.str();
                        std::stringstream result;
                        if (tmp.size() == 0) {
                                result << "    ";
                        }
                        else if (tmp.size() == 1) {
                                result << " " << tmp << "  ";
                        }
                        else if (tmp.size() == 2) {
                                result << " " << tmp << " ";
                        }
                        else if (tmp.size() == 3) {
                                result << " " << tmp;
                        }
                        else if (tmp.size() == 4) {
                                result << tmp;
                        }
                        return result.str();
                };
                        
                std::array<std::string,Rank> uncoupled;
                std::array<std::string,util::inter_dim(Rank)> intermediates;
                std::array<std::string,util::mult_dim(Rank)> multiplicities;
                for (size_t i=0; i< util::mult_dim(Rank); i++) {
                        multiplicities[i] = " μ";
                }
                for (size_t i=0; i< Rank; i++) {
                        uncoupled[i] = print_aligned(q_uncoupled[i]);
                }
                for (size_t i=0; i<util::inter_dim(Rank); i++) {                        
                        intermediates[i] = print_aligned(q_intermediates[i]);
                }
                std::string coupled;
                coupled = print_aligned(q_coupled);
                return printTree<Rank>(uncoupled, intermediates, coupled, multiplicities);
        }

        std::string print() const
        {
                std::stringstream ss;
                ss << "Fusiontree with dim=" << dim << std::endl;
                ss << "uncoupled sectors: "; for (const auto& q : q_uncoupled) {ss << q << " ";} ss << std::endl;
                ss << "intermediate sectors: "; for (const auto& q : q_intermediates) {ss << q << " ";} ss << std::endl;
                ss << "coupled sector: " << q_coupled << std::endl;
                return ss.str();
        }

        Eigen::Tensor<Scalar,Rank+1> asTensor() const {
                if constexpr (Rank == 0) { Eigen::Tensor<Scalar,Rank+1> out(1); out(0)=1.; return out; }
                else if constexpr (Rank == 1) {
                        Eigen::Tensor<Scalar,Rank+1> out(Symmetry::degeneracy(q_uncoupled[0]), Symmetry::degeneracy(q_coupled)); out.setZero();
                        for (std::size_t i=0; i<Symmetry::degeneracy(q_uncoupled[0]); i++) {out(i,i) = 1.;}
                        return out;
                }
                else if constexpr (Rank == 2) {
                        return Symmetry::CGC(q_uncoupled[0], q_uncoupled[1], q_coupled, multiplicities[0]);
                }
                else if constexpr (Rank == 3) {
                        auto vertex1 = Symmetry::CGC(q_uncoupled[0], q_uncoupled[1], q_intermediates[0], multiplicities[0]);
                        auto vertex2 = Symmetry::CGC(q_intermediates[0], q_uncoupled[2], q_coupled, multiplicities[1]);
                        return vertex1.contract(vertex2, Eigen::array<Eigen::IndexPair<Eigen::Index>, 1>{{Eigen::IndexPair<Eigen::Index>(2,0)}});
                }
                else if constexpr (Rank == 4) {
                        auto vertex1 = Symmetry::CGC(q_uncoupled[0], q_uncoupled[1], q_intermediates[0], multiplicities[0]);
                        auto vertex2 = Symmetry::CGC(q_intermediates[0], q_uncoupled[2], q_intermediates[1], multiplicities[1]);
                        auto vertex3 = Symmetry::CGC(q_intermediates[1], q_uncoupled[3], q_coupled, multiplicities[2]);
                        return (vertex1.contract(vertex2, Eigen::array<Eigen::IndexPair<Eigen::Index>, 1>{{Eigen::IndexPair<Eigen::Index>(2,0)}}))
                                .contract(vertex3,Eigen::array<Eigen::IndexPair<Eigen::Index>, 1>{{Eigen::IndexPair<Eigen::Index>(3,0)}});
                }
                else if constexpr (Rank == 5) {
                        auto vertex1 = Symmetry::CGC(q_uncoupled[0], q_uncoupled[1], q_intermediates[0], multiplicities[0]);
                        auto vertex2 = Symmetry::CGC(q_intermediates[0], q_uncoupled[2], q_intermediates[1], multiplicities[1]);
                        auto vertex3 = Symmetry::CGC(q_intermediates[1], q_uncoupled[3], q_intermediates[2], multiplicities[2]);
                        auto vertex4 = Symmetry::CGC(q_intermediates[2], q_uncoupled[4], q_coupled, multiplicities[3]);
                        return (vertex1.contract(vertex2, Eigen::array<Eigen::IndexPair<Eigen::Index>, 1>{{Eigen::IndexPair<Eigen::Index>(2,0)}}))
                                .contract(vertex3,Eigen::array<Eigen::IndexPair<Eigen::Index>, 1>{{Eigen::IndexPair<Eigen::Index>(3,0)}})
                                          .contract(vertex4,Eigen::array<Eigen::IndexPair<Eigen::Index>, 1>{{Eigen::IndexPair<Eigen::Index>(4,0)}});
                }
                else {  assert(false); }
        }
        
        FusionTree<Rank+1, Symmetry> enlarge(const FusionTree<1, Symmetry>& other) const
        {
                FusionTree<Rank+1, Symmetry> out;
                out.dim = this->dim * other.dim;
                std::copy(this->q_uncoupled.begin(), this->q_uncoupled.end(), out.q_uncoupled.begin());
                out.q_uncoupled[Rank] = other.q_uncoupled[0];
                std::copy(this->q_intermediates.begin(), this->q_intermediates.end(), out.q_intermediates.begin());
                if (out.q_intermediates.size() > 0) {out.q_intermediates[out.q_intermediates.size()-1] = this->q_coupled;}


                std::copy(this->multiplicities.begin(), this->multiplicities.end(), out.multiplicities.begin());
                return out;
        }

        std::unordered_map<FusionTree<Rank, Symmetry>, typename Symmetry::Scalar> permute(const Permutation<Rank>& p) const        
        {
                std::unordered_map<FusionTree<Rank, Symmetry>, typename Symmetry::Scalar> tmp;
                std::unordered_map<FusionTree<Rank, Symmetry>, typename Symmetry::Scalar> out; out.insert(std::make_pair(*this, 1.));
                for (const auto& pos:p.decompose()) {
                        for (const auto& [tree,coeff] : out) {
                                auto tmp2 = tree.swap(pos);
                                for (const auto& [tree2,coeff2] : tmp2) {
                                        auto it = tmp.find(tree2);
                                        if (it == tmp.end()) {tmp.insert(std::make_pair(tree2, coeff*coeff2));}
                                        else (tmp[tree2] += coeff*coeff2);
                                }
                        }
                        out = tmp;
                        tmp.clear();
                }
                return out;
        }
        
        std::unordered_map<FusionTree<Rank, Symmetry>, typename Symmetry::Scalar> swap(const std::size_t& pos) const //swaps sites pos and pos+1
        {
                assert(pos < Rank-1 and "Invalid position for swap.");
                std::unordered_map<FusionTree<Rank, Symmetry>, typename Symmetry::Scalar> out;
                if (pos == 0) //easy case
                        {
                                if constexpr (Symmetry::HAS_MULTIPLICITIES) {
                                        assert(false and "Not implemented.");
                                }
                                else {
                                        auto ql = q_uncoupled[0];
                                        auto qr = q_uncoupled[1];
                                        auto qf = (Rank == 2) ? q_coupled : q_intermediates[0];
                                        FusionTree<Rank, Symmetry> tree;
                                        tree.q_coupled = q_coupled;
                                        tree.q_uncoupled = q_uncoupled;
                                        std::swap(tree.q_uncoupled[0], tree.q_uncoupled[1]);
                                        tree.IS_DUAL = IS_DUAL;
                                        std::swap(tree.IS_DUAL[0], tree.IS_DUAL[1]);
                                        tree.q_intermediates = q_intermediates;
                                        tree.multiplicities = multiplicities;
                                        tree.dim = dim;
                                        
                                        out.insert(std::make_pair(tree,Symmetry::coeff_swap(ql,qr,qf)));
                                        return out;
                                }
                        }
                if constexpr (Symmetry::HAS_MULTIPLICITIES) {
                        assert(false and "Not implemented.");
                }
                else {
                        auto q1 = (pos == 1) ? q_uncoupled[0] : q_intermediates[pos-2];
                        auto q2 = q_uncoupled[pos];
                        auto q3 = q_uncoupled[pos+1];
                        auto Q = (pos == Rank-2) ? q_coupled : q_intermediates[pos];
                        auto Q12 = q_intermediates[pos-1];
                        FusionTree<Rank, Symmetry> tree;
                        tree.q_coupled = q_coupled;
                        tree.q_uncoupled = q_uncoupled;
                        std::swap(tree.q_uncoupled[pos], tree.q_uncoupled[pos+1]);
                        tree.IS_DUAL = IS_DUAL;
                        std::swap(tree.IS_DUAL[pos], tree.IS_DUAL[pos+1]);
                        tree.q_intermediates = q_intermediates;
                        tree.multiplicities = multiplicities;
                        tree.dim = dim;
                        for (const auto& Q31: Symmetry::reduceSilent(q3,q1))
                                {
                                        tree.q_intermediates[pos-1] = Q31;                                        
                                        // auto cgc = Symmetry::coeff_swap(Q12,q3,Q) * std::conj(Symmetry::coeff_recouple(q1,q3,q2,Q,Q13,Q12)) * Symmetry::coeff_swap(q1,q3,Q13);
                                        auto cgc = Symmetry::coeff_swap(Q12,q3,Q) * Symmetry::coeff_recouple(q3,q1,q2,Q,Q31,Q12) * Symmetry::coeff_swap(q3,q1,Q31);
                                        if (std::abs(cgc) < mynumeric_limits<typename Symmetry::Scalar>::epsilon()) {continue;}
                                        out.insert(std::make_pair(tree,cgc));
                                }
                        return out;
                }
        }
};

template<std::size_t depth, typename Symmetry>
std::ostream& operator<<(std::ostream& os, const FusionTree<depth,Symmetry> &tree)
{
	os << tree.print();
	return os;
}
#endif
