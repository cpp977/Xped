#ifndef FUSIONTREE_H_
#define FUSIONTREE_H_

#ifdef XPED_CACHE_PERMUTE_OUTPUT
#    include "lru/lru.hpp"
#endif

#include "Interfaces/tensor_traits.hpp"

#include "Hash/hash.hpp"
#include "Permutations.h"
#include "Symmetry/functions.hpp"
#include "numeric_limits.h"

#include <unordered_map>

namespace util {
constexpr std::size_t inter_dim(std::size_t Rank) { return (Rank == 1 or Rank == 0) ? 0 : Rank - 2; }
constexpr std::size_t mult_dim(std::size_t Rank) { return (Rank == 0) ? 0 : Rank - 1; }
constexpr std::size_t numberOfVertices(std::size_t Rank) { return (Rank == 0) ? 0 : Rank - 1; }
constexpr std::size_t numberOfInnerlines(std::size_t Rank) { return (Rank == 1 or Rank == 0) ? 0 : Rank - 2; }
} // namespace util

#include "FusionTreePrint.hpp"

template <std::size_t Rank, typename Symmetry>
struct FusionTree
{
    typedef typename Symmetry::qType qType;
    typedef typename Symmetry::Scalar Scalar;

    std::array<qType, Rank> q_uncoupled;
    qType q_coupled;
    std::size_t dim;
    std::array<size_t, Rank> dims;
    std::array<qType, util::inter_dim(Rank)> q_intermediates;
    std::array<size_t, util::mult_dim(Rank)> multiplicities =
        std::array<size_t, util::mult_dim(Rank)>(); // only for non-Abelian symmetries with outermultiplicity.
    std::array<bool, Rank> IS_DUAL{};

    void computeDim() { dim = std::accumulate(dims.begin(), dims.end(), 1ul, std::multiplies<std::size_t>()); }

    bool operator<(const FusionTree<Rank, Symmetry>& other) const
    {
        if(Symmetry::compare(q_uncoupled, other.q_uncoupled))
            return true;
        else {
            if(q_uncoupled != other.q_uncoupled) {
                return false;
            } else {
                if(Symmetry::compare(q_intermediates, other.q_intermediates))
                    return true;
                else {
                    if(q_intermediates != other.q_intermediates) {
                        return false;
                    } else {
                        if(Symmetry::compare(q_coupled, other.q_coupled))
                            return true;
                        else {
                            if(q_coupled != other.q_coupled) {
                                return false;
                            } else {
                                if(multiplicities < other.multiplicities)
                                    return true;
                                else {
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    bool operator==(const FusionTree<Rank, Symmetry>& other) const
    {
        return q_uncoupled == other.q_uncoupled and q_intermediates == other.q_intermediates and multiplicities == other.multiplicities and
               q_coupled == other.q_coupled and IS_DUAL == other.IS_DUAL and dims == other.dims;
    }

    bool operator!=(const FusionTree<Rank, Symmetry>& other) const { return !this->operator==(other); }

    friend std::size_t hash_value(const FusionTree<Rank, Symmetry>& tree)
    {
        size_t seed = 0;
        boost::hash_combine(seed, tree.q_uncoupled);
        boost::hash_combine(seed, tree.q_coupled);
        boost::hash_combine(seed, tree.q_intermediates);
        boost::hash_combine(seed, tree.IS_DUAL);
        boost::hash_combine(seed, tree.multiplicities);
        boost::hash_combine(seed, tree.dims);
        return seed;
    }

    std::string draw() const
    {
        //, std::size_t alignment=4
        auto print_aligned = [](const qType& q) -> std::string {
            std::stringstream ss;
            ss << Sym::format<Symmetry>(q);
            std::string tmp = ss.str();
            std::stringstream result;
            if(tmp.size() == 0) {
                result << "    ";
            } else if(tmp.size() == 1) {
                result << " " << tmp << "  ";
            } else if(tmp.size() == 2) {
                result << " " << tmp << " ";
            } else if(tmp.size() == 3) {
                result << " " << tmp;
            } else if(tmp.size() == 4) {
                result << tmp;
            }
            return result.str();
        };

        std::array<std::string, Rank> uncoupled;
        std::array<std::string, util::inter_dim(Rank)> intermediates;
        std::array<std::string, util::mult_dim(Rank)> multiplicities;
        for(size_t i = 0; i < util::mult_dim(Rank); i++) { multiplicities[i] = " μ"; }
        for(size_t i = 0; i < Rank; i++) { uncoupled[i] = print_aligned(q_uncoupled[i]); }
        for(size_t i = 0; i < util::inter_dim(Rank); i++) { intermediates[i] = print_aligned(q_intermediates[i]); }
        std::string coupled;
        coupled = print_aligned(q_coupled);
        return printTree<Rank>(uncoupled, intermediates, coupled, multiplicities, IS_DUAL);
    }

    std::string print() const
    {
        std::stringstream ss;
        ss << "Fusiontree with dim=" << dim << std::endl;
        ss << "uncoupled sectors: ";
        for(const auto& q : q_uncoupled) { ss << q << " "; }
        ss << std::endl;
        ss << "intermediate sectors: ";
        for(const auto& q : q_intermediates) { ss << q << " "; }
        ss << std::endl;
        ss << "coupled sector: " << q_coupled << std::endl;
        return ss.str();
    }

    template <typename TensorLib_>
    typename tensortraits<TensorLib_>::template Ttype<Scalar, Rank + 1> asTensor() const
    {
        static_assert(Rank <= 5);
        typedef typename tensortraits<TensorLib_>::template Ttype<Scalar, Rank + 1> TensorType;
        typedef typename tensortraits<TensorLib_>::Indextype IndexType;
        TensorType out;
        if constexpr(Rank == 0) {
            out = tensortraits<TensorLib_>::template construct<Scalar, 1>(std::array<IndexType, 1>{1});
            tensortraits<TensorLib_>::template setConstant<Scalar, 1>(out, 1.);
            // out(0) = 1.;
        } else if constexpr(Rank == 1) {
            out = tensortraits<TensorLib_>::template construct<Scalar>(
                std::array<IndexType, 2>{Symmetry::degeneracy(q_uncoupled[0]), Symmetry::degeneracy(q_coupled)});
            // out = TensorType(Symmetry::degeneracy(q_uncoupled[0]), Symmetry::degeneracy(q_coupled));
            tensortraits<TensorLib_>::template setZero<Scalar, 2>(out);
            for(IndexType i = 0; i < static_cast<std::size_t>(Symmetry::degeneracy(q_uncoupled[0])); i++) {
                tensortraits<TensorLib_>::template setVal<Scalar, 2>(out, {{i, i}}, 1.);
            }

        } else if constexpr(Rank == 2) {
            out = Symmetry::template CGC<TensorLib_>(q_uncoupled[0], q_uncoupled[1], q_coupled, multiplicities[0]);
        } else if constexpr(Rank == 3) {
            auto vertex1 = Symmetry::template CGC<TensorLib_>(q_uncoupled[0], q_uncoupled[1], q_intermediates[0], multiplicities[0]);
            auto vertex2 = Symmetry::template CGC<TensorLib_>(q_intermediates[0], q_uncoupled[2], q_coupled, multiplicities[1]);
            out = tensortraits<TensorLib_>::template contract<Scalar, 3, 3, 2, 0>(vertex1, vertex2);
        } else if constexpr(Rank == 4) {
            auto vertex1 = Symmetry::template CGC<TensorLib_>(q_uncoupled[0], q_uncoupled[1], q_intermediates[0], multiplicities[0]);
            auto vertex2 = Symmetry::template CGC<TensorLib_>(q_intermediates[0], q_uncoupled[2], q_intermediates[1], multiplicities[1]);
            auto vertex3 = Symmetry::template CGC<TensorLib_>(q_intermediates[1], q_uncoupled[3], q_coupled, multiplicities[2]);
            auto intermediate = tensortraits<TensorLib_>::template contract<Scalar, 3, 3, 2, 0>(vertex1, vertex2);
            out = tensortraits<TensorLib_>::template contract<Scalar, 4, 3, 3, 0>(intermediate, vertex3);
        } else if constexpr(Rank == 5) {
            auto vertex1 = Symmetry::template CGC<TensorLib_>(q_uncoupled[0], q_uncoupled[1], q_intermediates[0], multiplicities[0]);
            auto vertex2 = Symmetry::template CGC<TensorLib_>(q_intermediates[0], q_uncoupled[2], q_intermediates[1], multiplicities[1]);
            auto vertex3 = Symmetry::template CGC<TensorLib_>(q_intermediates[1], q_uncoupled[3], q_intermediates[2], multiplicities[2]);
            auto vertex4 = Symmetry::template CGC<TensorLib_>(q_intermediates[2], q_uncoupled[4], q_coupled, multiplicities[3]);
            auto intermediate1 = tensortraits<TensorLib_>::template contract<Scalar, 3, 3, 2, 0>(vertex1, vertex2);
            auto intermediate2 = tensortraits<TensorLib_>::template contract<Scalar, 4, 3, 3, 0>(intermediate1, vertex3);
            out = tensortraits<TensorLib_>::template contract<Scalar, 5, 3, 4, 0>(intermediate2, vertex4);

            // out = (vertex1.contract(vertex2, Eigen::array<Eigen::IndexPair<Eigen::Index>, 1>{{Eigen::IndexPair<Eigen::Index>(2,0)}}))
            //         .contract(vertex3,Eigen::array<Eigen::IndexPair<Eigen::Index>, 1>{{Eigen::IndexPair<Eigen::Index>(3,0)}})
            //                   .contract(vertex4,Eigen::array<Eigen::IndexPair<Eigen::Index>, 1>{{Eigen::IndexPair<Eigen::Index>(4,0)}});
        } else {
            assert(false);
        }
        if constexpr(Rank == 0) {
            return out;
        } // 0
        else {
            if(IS_DUAL[0]) {
                auto one_j = Symmetry::template one_j_tensor<TensorLib_>(q_uncoupled[0]);
                TensorType tmp = tensortraits<TensorLib_>::template contract<Scalar, 2, Rank + 1, 1, 0>(one_j, out);
                out = tmp;
                // std::array<IndexType, Rank+1> shuffle_dims; std::iota(shuffle_dims.begin(), shuffle_dims.end(), 0);
                // for (std::size_t j=0; j<0; j++) {
                //         shuffle_dims[j]++;
                // }
                // shuffle_dims[0] = 0;
                TensorType tmp2 = tensortraits<TensorLib_>::template shuffle<Scalar, Rank + 1>(out, seq::make<IndexType, Rank + 1>{});
                out = tmp2;
            }
            if constexpr(Rank == 1) {
                return out;
            } // 1
            else {
                if(IS_DUAL[1]) {
                    auto one_j = Symmetry::template one_j_tensor<TensorLib_>(q_uncoupled[1]);
                    TensorType tmp = tensortraits<TensorLib_>::template contract<Scalar, 2, Rank + 1, 1, 1>(one_j, out);
                    out = tmp;
                    // std::array<IndexType, Rank+1> shuffle_dims; std::iota(shuffle_dims.begin(), shuffle_dims.end(), 0);
                    // for (std::size_t j=0; j<1; j++) {
                    //         shuffle_dims[j]++;
                    // }
                    // shuffle_dims[1] = 0;
                    TensorType tmp2 = tensortraits<TensorLib_>::template shuffle<Scalar, Rank + 1>(
                        out, seq::concat<seq::iseq<IndexType, 1, 0>, seq::make<IndexType, Rank + 1 - 2, 2>>{});
                    out = tmp2;
                }

                if constexpr(Rank == 2) {
                    return out;
                } else {
                    if(IS_DUAL[2]) {
                        auto one_j = Symmetry::template one_j_tensor<TensorLib_>(q_uncoupled[2]);
                        TensorType tmp = tensortraits<TensorLib_>::template contract<Scalar, 2, Rank + 1, 1, 2>(one_j, out);
                        out = tmp;
                        // std::array<IndexType, Rank+1> shuffle_dims; std::iota(shuffle_dims.begin(), shuffle_dims.end(), 0);
                        // for (std::size_t j=0; j<2; j++) {
                        //         shuffle_dims[j]++;
                        // }
                        // shuffle_dims[2] = 0;
                        TensorType tmp2 = tensortraits<TensorLib_>::template shuffle<Scalar, Rank + 1>(
                            out, seq::concat<seq::iseq<IndexType, 1, 2, 0>, seq::make<IndexType, Rank + 1 - 3, 3>>{});
                        out = tmp2;
                    }
                    if constexpr(Rank == 3) {
                        return out;
                    } else {
                        if(IS_DUAL[3]) {
                            auto one_j = Symmetry::template one_j_tensor<TensorLib_>(q_uncoupled[3]);
                            TensorType tmp = tensortraits<TensorLib_>::template contract<Scalar, 2, Rank + 1, 1, 3>(one_j, out);
                            out = tmp;
                            // std::array<IndexType, Rank+1> shuffle_dims; std::iota(shuffle_dims.begin(), shuffle_dims.end(), 0);
                            // for (std::size_t j=0; j<3; j++) {
                            //         shuffle_dims[j]++;
                            // }
                            // shuffle_dims[3] = 0;
                            TensorType tmp2 = tensortraits<TensorLib_>::template shuffle<Scalar, Rank + 1>(
                                out, seq::concat<seq::iseq<IndexType, 1, 2, 3, 0>, seq::make<IndexType, Rank + 1 - 4, 4>>{});
                            out = tmp2;
                        }
                        if constexpr(Rank == 4) {
                            return out;
                        } else {
                            if(IS_DUAL[4]) {
                                auto one_j = Symmetry::template one_j_tensor<TensorLib_>(q_uncoupled[4]);
                                TensorType tmp = tensortraits<TensorLib_>::template contract<Scalar, 2, Rank + 1, 1, 4>(one_j, out);
                                out = tmp;
                                // std::array<IndexType, Rank+1> shuffle_dims; std::iota(shuffle_dims.begin(), shuffle_dims.end(), 0);
                                // for (std::size_t j=0; j<4; j++) {
                                //         shuffle_dims[j]++;
                                // }
                                // shuffle_dims[4] = 0;
                                TensorType tmp2 = tensortraits<TensorLib_>::template shuffle<Scalar, Rank + 1>(
                                    out, seq::concat<seq::iseq<IndexType, 1, 2, 3, 4, 0>, seq::make<IndexType, Rank + 1 - 5, 5>>{});
                                out = tmp2;
                            }
                            if constexpr(Rank == 5) { return out; }
                            // else {static_assert(false);}
                        }
                    }
                }
            }
        }
        // for (std::size_t i=0; i<Rank; i++) {
        //         if (IS_DUAL[i]) {
        //                 TensorType tmp = tensortraits<TensorLib_>::template contract(Symmetry::template one_j_tensor<TensorLib_>(q_uncoupled[i]),
        //                                                                              out, std::array<std::pair<IndexType, IndexType>,
        //                                                                              1>{{std::make_pair(1,i)}});
        //                 out = tmp;
        //                 std::array<IndexType, Rank+1> shuffle_dims; std::iota(shuffle_dims.begin(), shuffle_dims.end(), 0);
        //                 for (std::size_t j=0; j<i; j++) {
        //                         shuffle_dims[j]++;
        //                 }
        //                 shuffle_dims[i] = 0;
        //                 TensorType tmp2 = tensortraits<TensorLib_>::template shuffle(out, shuffle_dims);
        //                 out = tmp2;
        //         }
        // }
        return out;
    }

    FusionTree<Rank + 1, Symmetry> enlarge(const FusionTree<1, Symmetry>& other) const
    {
        FusionTree<Rank + 1, Symmetry> out;
        std::copy(dims.begin(), dims.end(), out.dims.begin());
        out.dims[Rank] = other.dims[0];
        out.computeDim();
        std::copy(this->q_uncoupled.begin(), this->q_uncoupled.end(), out.q_uncoupled.begin());
        out.q_uncoupled[Rank] = other.q_uncoupled[0];
        std::copy(this->q_intermediates.begin(), this->q_intermediates.end(), out.q_intermediates.begin());
        if(out.q_intermediates.size() > 0) { out.q_intermediates[out.q_intermediates.size() - 1] = this->q_coupled; }
        std::copy(IS_DUAL.begin(), IS_DUAL.end(), out.IS_DUAL.begin());
        out.IS_DUAL[Rank] = other.IS_DUAL[0];
        std::copy(this->multiplicities.begin(), this->multiplicities.end(), out.multiplicities.begin());
        return out;
    }

    std::unordered_map<FusionTree<Rank, Symmetry>, typename Symmetry::Scalar> permute(const Permutation<Rank>& p) const
    {
        std::unordered_map<FusionTree<Rank, Symmetry>, typename Symmetry::Scalar> tmp;
        std::unordered_map<FusionTree<Rank, Symmetry>, typename Symmetry::Scalar> out;
        out.insert(std::make_pair(*this, 1.));
        for(const auto& pos : p.decompose()) {
            for(const auto& [tree, coeff] : out) {
                for(const auto& [tree2, coeff2] : tree.swap(pos)) {
                    auto it = tmp.find(tree2);
                    if(it == tmp.end()) {
                        tmp.insert(std::make_pair(tree2, coeff * coeff2));
                    } else {
                        if(std::abs(tmp[tree2] + coeff * coeff2) < mynumeric_limits<Scalar>::epsilon()) {
                            tmp.erase(tree2);
                        } else {
                            tmp[tree2] += coeff * coeff2;
                        }
                    };
                }
            }
            out = tmp;
            tmp.clear();
        }
        return out;
    }

    std::unordered_map<FusionTree<Rank, Symmetry>, typename Symmetry::Scalar> swap(const std::size_t& pos) const // swaps sites pos and pos+1
    {
        assert(pos < Rank - 1 and "Invalid position for swap.");
        std::unordered_map<FusionTree<Rank, Symmetry>, typename Symmetry::Scalar> out;
        if(pos == 0) {
            if constexpr(Symmetry::HAS_MULTIPLICITIES) {
                assert(false and "Not implemented.");
            } else {
                auto ql = q_uncoupled[0];
                auto qr = q_uncoupled[1];
                auto qf = (Rank == 2) ? q_coupled : q_intermediates[0];
                FusionTree<Rank, Symmetry> tree;
                tree.q_coupled = q_coupled;
                tree.q_uncoupled = q_uncoupled;
                std::swap(tree.q_uncoupled[0], tree.q_uncoupled[1]);
                tree.IS_DUAL = IS_DUAL;
                std::swap(tree.IS_DUAL[0], tree.IS_DUAL[1]);
                tree.dims = dims;
                std::swap(tree.dims[0], tree.dims[1]);
                tree.dim = dim;
                tree.q_intermediates = q_intermediates;
                tree.multiplicities = multiplicities;
                Scalar coeff = Symmetry::coeff_swap(ql, qr, qf);
                if(std::abs(coeff) < mynumeric_limits<typename Symmetry::Scalar>::epsilon()) { return out; }
                out.insert(std::make_pair(tree, coeff));
                return out;
            }
        }
        if constexpr(Symmetry::HAS_MULTIPLICITIES) {
            assert(false and "Not implemented.");
        } else {
            auto q1 = (pos == 1) ? q_uncoupled[0] : q_intermediates[pos - 2];
            auto q2 = q_uncoupled[pos];
            auto q3 = q_uncoupled[pos + 1];
            auto Q = (pos == Rank - 2) ? q_coupled : q_intermediates[pos];
            auto Q12 = q_intermediates[pos - 1];
            FusionTree<Rank, Symmetry> tree;
            tree.q_coupled = q_coupled;
            tree.q_uncoupled = q_uncoupled;
            std::swap(tree.q_uncoupled[pos], tree.q_uncoupled[pos + 1]);
            tree.IS_DUAL = IS_DUAL;
            std::swap(tree.IS_DUAL[pos], tree.IS_DUAL[pos + 1]);
            tree.dims = dims;
            std::swap(tree.dims[pos], tree.dims[pos + 1]);
            tree.dim = dim;
            tree.q_intermediates = q_intermediates;
            tree.multiplicities = multiplicities;
            for(const auto& Q31 : Symmetry::reduceSilent(q3, q1)) {
                tree.q_intermediates[pos - 1] = Q31;
                // auto cgc = Symmetry::coeff_swap(Q12,q3,Q) * std::conj(Symmetry::coeff_recouple(q1,q3,q2,Q,Q13,Q12)) *
                // Symmetry::coeff_swap(q1,q3,Q13);
                auto cgc = Symmetry::coeff_swap(Q12, q3, Q) * Symmetry::coeff_recouple(q3, q1, q2, Q, Q31, Q12) * Symmetry::coeff_swap(q3, q1, Q31);
                if(std::abs(cgc) < mynumeric_limits<typename Symmetry::Scalar>::epsilon()) { continue; }
                out.insert(std::make_pair(tree, cgc));
            }
            return out;
        }
    }
};

template <std::size_t depth, typename Symmetry>
std::ostream& operator<<(std::ostream& os, const FusionTree<depth, Symmetry>& tree)
{
    os << tree.print();
    return os;
}

namespace treepair {
//                           ☐
//                           _
//   c                 c     b
//   |                  \   /
//   |                   \ /
//  / \   ---> coeff*     |
// /   \                  |
// a     b                 a
template <std::size_t Rank, std::size_t CoRank, typename Symmetry>
std::unordered_map<std::pair<FusionTree<Rank + 1, Symmetry>, FusionTree<CoRank - 1, Symmetry>>, typename Symmetry::Scalar>
turn_right(const FusionTree<Rank, Symmetry>& t1, const FusionTree<CoRank, Symmetry>& t2)
{
    static_assert(CoRank > 0);
    assert(t1.q_coupled == t2.q_coupled);

    FusionTree<Rank + 1, Symmetry> t1p;
    std::copy(t1.q_uncoupled.begin(), t1.q_uncoupled.end(), t1p.q_uncoupled.begin());
    t1p.q_uncoupled.back() = Symmetry::conj(t2.q_uncoupled.back());
    std::copy(t1.q_intermediates.begin(), t1.q_intermediates.end(), t1p.q_intermediates.begin());
    if(Rank + 1 > 2) { t1p.q_intermediates.back() = t1.q_coupled; }
    std::copy(t1.IS_DUAL.begin(), t1.IS_DUAL.end(), t1p.IS_DUAL.begin());
    t1p.IS_DUAL.back() = !t2.IS_DUAL.back();
    if(CoRank == 1) {
        t1p.q_coupled = Symmetry::qvacuum();
    } else if(CoRank == 2) {
        t1p.q_coupled = t2.q_uncoupled[0];
    } else {
        t1p.q_coupled = t2.q_intermediates.back();
    }
    std::copy(t1.dims.begin(), t1.dims.end(), t1p.dims.begin());
    t1p.dims.back() = t2.dims.back();
    t1p.computeDim();
    FusionTree<CoRank - 1, Symmetry> t2p;
    std::copy(t2.q_uncoupled.begin(), t2.q_uncoupled.end() - 1, t2p.q_uncoupled.begin());
    std::copy(t2.IS_DUAL.begin(), t2.IS_DUAL.end() - 1, t2p.IS_DUAL.begin());
    if constexpr(CoRank > 2) { std::copy(t2.q_intermediates.begin(), t2.q_intermediates.end() - 1, t2p.q_intermediates.begin()); }
    if(CoRank == 1) {
        t2p.q_coupled = Symmetry::qvacuum();
    } else if(CoRank == 2) {
        t2p.q_coupled = t2.q_uncoupled[0];
    } else {
        t2p.q_coupled = t2.q_intermediates.back();
    }
    std::copy(t2.dims.begin(), t2.dims.end() - 1, t2p.dims.begin());
    t2p.computeDim();
    typename Symmetry::qType a;
    if(CoRank == 1) {
        a = Symmetry::qvacuum();
    } else if(CoRank == 2) {
        a = t2.q_uncoupled[0];
    } else {
        a = t2.q_intermediates.back();
    }
    typename Symmetry::qType b = t2.q_uncoupled.back();
    typename Symmetry::qType c = t2.q_coupled;
    auto coeff = Symmetry::coeff_turn(a, b, c);
    if(t2.IS_DUAL.back()) { coeff *= Symmetry::coeff_FS(Symmetry::conj(b)); }
    std::unordered_map<std::pair<FusionTree<Rank + 1, Symmetry>, FusionTree<CoRank - 1, Symmetry>>, typename Symmetry::Scalar> out;
    if(std::abs(coeff) < mynumeric_limits<typename Symmetry::Scalar>::epsilon()) { return out; }
    out.insert(std::make_pair(std::make_pair(t1p, t2p), coeff));
    return out;
}

template <std::size_t Rank, std::size_t CoRank, typename Symmetry>
std::unordered_map<std::pair<FusionTree<Rank - 1, Symmetry>, FusionTree<CoRank + 1, Symmetry>>, typename Symmetry::Scalar>
turn_left(const FusionTree<Rank, Symmetry>& t1, const FusionTree<CoRank, Symmetry>& t2)
{
    auto tmp = turn_right(t2, t1);
    std::unordered_map<std::pair<FusionTree<Rank - 1, Symmetry>, FusionTree<CoRank + 1, Symmetry>>, typename Symmetry::Scalar> out;
    for(const auto& [trees, coeff] : tmp) {
        auto [t1p, t2p] = trees;
        out.insert(std::make_pair(std::make_pair(t2p, t1p), coeff));
    }
    return out;
}

template <int shift, std::size_t Rank, std::size_t CoRank, typename Symmetry>
std::unordered_map<std::pair<FusionTree<Rank - shift, Symmetry>, FusionTree<CoRank + shift, Symmetry>>, typename Symmetry::Scalar>
turn(const FusionTree<Rank, Symmetry>& t1, const FusionTree<CoRank, Symmetry>& t2)
{
    assert(t1.q_coupled == t2.q_coupled);
    if constexpr(shift > 0) {
        static_assert(shift <= static_cast<int>(Rank));
    } else if constexpr(shift < 0) {
        static_assert(-shift <= static_cast<int>(CoRank));
    }
    if constexpr(shift == 0) {
        std::unordered_map<std::pair<FusionTree<Rank, Symmetry>, FusionTree<CoRank, Symmetry>>, typename Symmetry::Scalar> out;
        out.insert(std::make_pair(std::make_pair(t1, t2), 1.));
        return out;
    } else {
        constexpr std::size_t newRank = Rank - shift;
        constexpr std::size_t newCoRank = CoRank + shift;
        std::unordered_map<std::pair<FusionTree<newRank, Symmetry>, FusionTree<newCoRank, Symmetry>>, typename Symmetry::Scalar> out;
        if constexpr(shift > 0) {
            for(const auto& [trees1, coeff1] : turn_left(t1, t2)) {
                auto [t1p, t2p] = trees1;
                for(const auto& [trees2, coeff2] : turn<shift - 1>(t1p, t2p)) {
                    auto [t1pp, t2pp] = trees2;
                    out.insert(std::make_pair(std::make_pair(t1pp, t2pp), coeff1 * coeff2));
                }
            }
            return out;
        } else {
            for(const auto& [trees1, coeff1] : turn_right(t1, t2)) {
                auto [t1p, t2p] = trees1;
                for(const auto& [trees2, coeff2] : turn<shift + 1>(t1p, t2p)) {
                    auto [t1pp, t2pp] = trees2;
                    out.insert(std::make_pair(std::make_pair(t1pp, t2pp), coeff1 * coeff2));
                }
            }
            return out;
        }
    }
}

template <int shift, std::size_t Rank, std::size_t CoRank, typename Symmetry>
std::unordered_map<std::pair<FusionTree<Rank - shift, Symmetry>, FusionTree<CoRank + shift, Symmetry>>, typename Symmetry::Scalar>
permute(const FusionTree<Rank, Symmetry>& t1, const FusionTree<CoRank, Symmetry>& t2, const Permutation<Rank + CoRank>& p)
{
#ifdef XPED_CACHE_PERMUTE_OUTPUT
    if constexpr(Symmetry::NON_ABELIAN) {
        if(tree_cache<shift, Rank, CoRank, Symmetry>.cache.contains(std::make_tuple(t1, t2, p))) {
            return tree_cache<shift, Rank, CoRank, Symmetry>.cache.lookup(std::make_tuple(t1, t2, p));
        }
    }
#endif
    assert(t1.q_coupled == t2.q_coupled);

    // transform the permutation. Needed because turn<>() reverses the order of the FusionTree.
    std::array<std::size_t, Rank + CoRank> pi_id;
    std::iota(pi_id.begin(), pi_id.end(), 0ul);
    for(std::size_t i = 0; i < CoRank; i++) { pi_id[i + Rank] = (CoRank - 1) - i + Rank; }
    constexpr std::size_t newRank = Rank - shift;
    constexpr std::size_t newCoRank = CoRank + shift;
    std::array<std::size_t, Rank + CoRank> pi_tmp;
    for(std::size_t i = 0; i < Rank + CoRank; i++) { pi_tmp[i] = pi_id[p.pi[i]]; }
    std::array<std::size_t, Rank + CoRank> pi_corrected;
    std::copy(pi_tmp.begin(), pi_tmp.begin() + newRank, pi_corrected.begin());
    for(std::size_t i = 0; i < newCoRank; i++) { pi_corrected[i + newRank] = pi_tmp[(newCoRank - 1) - i + newRank]; }
    Permutation<Rank + CoRank> p_corrected(pi_corrected);

    constexpr int reshift = CoRank + shift;
    std::unordered_map<std::pair<FusionTree<Rank - shift, Symmetry>, FusionTree<CoRank + shift, Symmetry>>, typename Symmetry::Scalar> out;
    for(const auto& [trees, coeff1] : turn<-static_cast<int>(CoRank)>(t1, t2)) {
        auto [t1p, trivial] = trees;
        for(const auto& [t1pp, coeff2] : t1p.permute(p_corrected)) {
            for(const auto& [trees_final, coeff3] : turn<reshift>(t1pp, trivial)) {
                if(auto it = out.find(trees_final); it == out.end()) {
                    out.insert(std::make_pair(trees_final, coeff1 * coeff2 * coeff3));
                } else {
                    out[trees_final] += coeff1 * coeff2 * coeff3;
                }
            }
        }
    }
    std::size_t zero_count = 0ul;
    for(const auto& [trees, coeff] : out) {
        if(std::abs(coeff) < 1.e-9) { zero_count++; }
    }
    if(zero_count > 0) { cout << "permute pair operation created #=" << zero_count << " 0s." << endl; }
#ifdef XPED_CACHE_PERMUTE_OUTPUT
    if constexpr(Symmetry::NON_ABELIAN) { tree_cache<shift, Rank, CoRank, Symmetry>.cache.emplace(std::make_tuple(t1, t2, p), out); }
#endif
    return out;
}

} // end namespace treepair

#endif
