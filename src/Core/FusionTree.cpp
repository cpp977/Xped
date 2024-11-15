#include <iostream>
#include <string>
#include <unordered_map>

#include "Xped/Core/FusionTree.hpp"
#include "Xped/Core/ScalarTraits.hpp"
#include "Xped/Hash/hash.hpp"
#include "Xped/Interfaces/PlainInterface.hpp"

#include "Xped/Symmetry/CombSym.hpp"
#include "Xped/Symmetry/S1xS2.hpp"
#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"
#include "Xped/Symmetry/ZN.hpp"

#include "Xped/Symmetry/functions.hpp"
#include "Xped/Util/Macros.hpp"

using std::cout;
using std::endl;
using std::size_t;

namespace Xped {

template <std::size_t Rank, typename Symmetry>
bool FusionTree<Rank, Symmetry>::operator>(const FusionTree<Rank, Symmetry>& other) const
{
    return not this->operator<(other);
}

template <std::size_t Rank, typename Symmetry>
bool FusionTree<Rank, Symmetry>::operator<(const FusionTree<Rank, Symmetry>& other) const
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

template <std::size_t Rank, typename Symmetry>
bool FusionTree<Rank, Symmetry>::operator==(const FusionTree<Rank, Symmetry>& other) const
{
    if constexpr(Symmetry::ALL_IS_TRIVIAL) { return dims == other.dims; }
    if constexpr(not Symmetry::ANY_HAS_MULTIPLICITIES) {
#ifdef XPED_PEDANTIC_ASSERTS
        return q_uncoupled == other.q_uncoupled and q_coupled == other.q_coupled and dims == other.dims and IS_DUAL == other.IS_DUAL and
               q_intermediates == other.q_intermediates;
#else
        return q_uncoupled == other.q_uncoupled and q_coupled == other.q_coupled and dims == other.dims and q_intermediates == other.q_intermediates;
#endif
    }
#ifdef XPED_PEDANTIC_ASSERTS
    return q_uncoupled == other.q_uncoupled and q_intermediates == other.q_intermediates and multiplicities == other.multiplicities and
           q_coupled == other.q_coupled and IS_DUAL == other.IS_DUAL and dims == other.dims;
#else
    return q_uncoupled == other.q_uncoupled and q_intermediates == other.q_intermediates and multiplicities == other.multiplicities and
           q_coupled == other.q_coupled and dims == other.dims;
#endif
}

template <std::size_t Rank, typename Symmetry>
std::string FusionTree<Rank, Symmetry>::draw() const
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
    return printTree(uncoupled, intermediates, coupled, multiplicities);
}

template <std::size_t Rank, typename Symmetry>
std::string FusionTree<Rank, Symmetry>::print() const
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

template <std::size_t Rank, typename Symmetry>
void FusionTree<Rank, Symmetry>::computeIntermediates()
{
    assert(not Symmetry::ANY_NON_ABELIAN and "Cannot compute intermediate quantum numbers for non-Abelian symmetries.");
    if constexpr(Rank == 0) { return; }
    for(std::size_t i = 0; i < q_intermediates.size(); ++i) {
        q_intermediates[i] = (i == 0) ? Symmetry::reduceSilent(q_uncoupled[0], q_uncoupled[1]).front()
                                      : Symmetry::reduceSilent(q_intermediates[i - 1], q_uncoupled[i + 1]).front();
    }
    if constexpr(Rank < 2) {
        q_coupled = q_uncoupled.front();
    } else if constexpr(Rank == 2) {
        q_coupled = Symmetry::reduceSilent(q_uncoupled[0], q_uncoupled[1]).front();
    } else {
        q_coupled = Symmetry::reduceSilent(q_intermediates.back(), q_uncoupled.back()).front();
    }
}

template <std::size_t Rank, typename Symmetry>
template <typename PlainLib>
typename PlainLib::template TType<typename Symmetry::Scalar, Rank + 1> FusionTree<Rank, Symmetry>::asTensor(const mpi::XpedWorld& world) const
{
    typedef typename Symmetry::Scalar Scalar;

    assert(Rank <= 5);
    typedef typename PlainLib::template TType<Scalar, Rank + 1> TensorType;
    typedef typename PlainLib::Indextype IndexType;
    TensorType out;
    if constexpr(Rank == 0) {
        out = PlainLib::template construct<Scalar, 1>(std::array<IndexType, 1>{1}, world);
        PlainLib::template setConstant<Scalar, 1>(out, 1.);
    } else if constexpr(Rank == 1) {
        out = PlainLib::template construct<Scalar>(std::array<IndexType, 2>{Symmetry::degeneracy(q_uncoupled[0]), Symmetry::degeneracy(q_coupled)},
                                                   world);
        PlainLib::template setZero<Scalar, 2>(out);
        for(IndexType i = 0; i < static_cast<std::size_t>(Symmetry::degeneracy(q_uncoupled[0])); i++) {
            PlainLib::template setVal<Scalar, 2>(out, {{i, i}}, 1.);
        }
    } else if constexpr(Rank == 2) {
        out = Symmetry::template CGC<PlainLib>(q_uncoupled[0], q_uncoupled[1], q_coupled, multiplicities[0], world);
    } else if constexpr(Rank == 3) {
        auto vertex1 = Symmetry::template CGC<PlainLib>(q_uncoupled[0], q_uncoupled[1], q_intermediates[0], multiplicities[0], world);
        auto vertex2 = Symmetry::template CGC<PlainLib>(q_intermediates[0], q_uncoupled[2], q_coupled, multiplicities[1], world);
        out = PlainLib::template contract<Scalar, 3, 3, 2, 0>(vertex1, vertex2);
    } else if constexpr(Rank == 4) {
        auto vertex1 = Symmetry::template CGC<PlainLib>(q_uncoupled[0], q_uncoupled[1], q_intermediates[0], multiplicities[0], world);
        auto vertex2 = Symmetry::template CGC<PlainLib>(q_intermediates[0], q_uncoupled[2], q_intermediates[1], multiplicities[1], world);
        auto vertex3 = Symmetry::template CGC<PlainLib>(q_intermediates[1], q_uncoupled[3], q_coupled, multiplicities[2], world);
        auto intermediate = PlainLib::template contract<Scalar, 3, 3, 2, 0>(vertex1, vertex2);
        out = PlainLib::template contract<Scalar, 4, 3, 3, 0>(intermediate, vertex3);
    } else if constexpr(Rank == 5) {
        auto vertex1 = Symmetry::template CGC<PlainLib>(q_uncoupled[0], q_uncoupled[1], q_intermediates[0], multiplicities[0], world);
        auto vertex2 = Symmetry::template CGC<PlainLib>(q_intermediates[0], q_uncoupled[2], q_intermediates[1], multiplicities[1], world);
        auto vertex3 = Symmetry::template CGC<PlainLib>(q_intermediates[1], q_uncoupled[3], q_intermediates[2], multiplicities[2], world);
        auto vertex4 = Symmetry::template CGC<PlainLib>(q_intermediates[2], q_uncoupled[4], q_coupled, multiplicities[3], world);
        auto intermediate1 = PlainLib::template contract<Scalar, 3, 3, 2, 0>(vertex1, vertex2);
        auto intermediate2 = PlainLib::template contract<Scalar, 4, 3, 3, 0>(intermediate1, vertex3);
        out = PlainLib::template contract<Scalar, 5, 3, 4, 0>(intermediate2, vertex4);

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
            auto one_j = Symmetry::template one_j_tensor<PlainLib>(q_uncoupled[0], world);
            TensorType tmp = PlainLib::template contract<Scalar, 2, Rank + 1, 1, 0>(one_j, out);
            out = tmp;
            // std::array<IndexType, Rank+1> shuffle_dims; std::iota(shuffle_dims.begin(), shuffle_dims.end(), 0);
            // for (std::size_t j=0; j<0; j++) {
            //         shuffle_dims[j]++;
            // }
            // shuffle_dims[0] = 0;
            TensorType tmp2 = PlainLib::template shuffle<Scalar, Rank + 1>(out, seq::make<IndexType, Rank + 1>{});
            out = tmp2;
        }
        if constexpr(Rank == 1) {
            return out;
        } // 1
        else {
            if(IS_DUAL[1]) {
                auto one_j = Symmetry::template one_j_tensor<PlainLib>(q_uncoupled[1], world);
                TensorType tmp = PlainLib::template contract<Scalar, 2, Rank + 1, 1, 1>(one_j, out);
                out = tmp;
                // std::array<IndexType, Rank+1> shuffle_dims; std::iota(shuffle_dims.begin(), shuffle_dims.end(), 0);
                // for (std::size_t j=0; j<1; j++) {
                //         shuffle_dims[j]++;
                // }
                // shuffle_dims[1] = 0;
                TensorType tmp2 = PlainLib::template shuffle<Scalar, Rank + 1>(
                    out, seq::concat<seq::iseq<IndexType, 1, 0>, seq::make<IndexType, Rank + 1 - 2, 2>>{});
                out = tmp2;
            }

            if constexpr(Rank == 2) {
                return out;
            } else {
                if(IS_DUAL[2]) {
                    auto one_j = Symmetry::template one_j_tensor<PlainLib>(q_uncoupled[2], world);
                    TensorType tmp = PlainLib::template contract<Scalar, 2, Rank + 1, 1, 2>(one_j, out);
                    out = tmp;
                    // std::array<IndexType, Rank+1> shuffle_dims; std::iota(shuffle_dims.begin(), shuffle_dims.end(), 0);
                    // for (std::size_t j=0; j<2; j++) {
                    //         shuffle_dims[j]++;
                    // }
                    // shuffle_dims[2] = 0;
                    TensorType tmp2 = PlainLib::template shuffle<Scalar, Rank + 1>(
                        out, seq::concat<seq::iseq<IndexType, 1, 2, 0>, seq::make<IndexType, Rank + 1 - 3, 3>>{});
                    out = tmp2;
                }
                if constexpr(Rank == 3) {
                    return out;
                } else {
                    if(IS_DUAL[3]) {
                        auto one_j = Symmetry::template one_j_tensor<PlainLib>(q_uncoupled[3], world);
                        TensorType tmp = PlainLib::template contract<Scalar, 2, Rank + 1, 1, 3>(one_j, out);
                        out = tmp;
                        // std::array<IndexType, Rank+1> shuffle_dims; std::iota(shuffle_dims.begin(), shuffle_dims.end(), 0);
                        // for (std::size_t j=0; j<3; j++) {
                        //         shuffle_dims[j]++;
                        // }
                        // shuffle_dims[3] = 0;
                        TensorType tmp2 = PlainLib::template shuffle<Scalar, Rank + 1>(
                            out, seq::concat<seq::iseq<IndexType, 1, 2, 3, 0>, seq::make<IndexType, Rank + 1 - 4, 4>>{});
                        out = tmp2;
                    }
                    if constexpr(Rank == 4) {
                        return out;
                    } else {
                        if(IS_DUAL[4]) {
                            auto one_j = Symmetry::template one_j_tensor<PlainLib>(q_uncoupled[4], world);
                            TensorType tmp = PlainLib::template contract<Scalar, 2, Rank + 1, 1, 4>(one_j, out);
                            out = tmp;
                            // std::array<IndexType, Rank+1> shuffle_dims; std::iota(shuffle_dims.begin(), shuffle_dims.end(), 0);
                            // for (std::size_t j=0; j<4; j++) {
                            //         shuffle_dims[j]++;
                            // }
                            // shuffle_dims[4] = 0;
                            TensorType tmp2 = PlainLib::template shuffle<Scalar, Rank + 1>(
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
    //                 TensorType tmp = PlainLib::template contract(Symmetry::template
    //                 one_j_tensor<PlainLib>(q_uncoupled[i]),
    //                                                                              out, std::array<std::pair<IndexType, IndexType>,
    //                                                                              1>{{std::make_pair(1,i)}});
    //                 out = tmp;
    //                 std::array<IndexType, Rank+1> shuffle_dims; std::iota(shuffle_dims.begin(), shuffle_dims.end(), 0);
    //                 for (std::size_t j=0; j<i; j++) {
    //                         shuffle_dims[j]++;
    //                 }
    //                 shuffle_dims[i] = 0;
    //                 TensorType tmp2 = PlainLib::template shuffle(out, shuffle_dims);
    //                 out = tmp2;
    //         }
    // }
    return out;
}

template <std::size_t Rank, typename Symmetry>
FusionTree<Rank + 1, Symmetry> FusionTree<Rank, Symmetry>::enlarge(const FusionTree<1, Symmetry>& other) const
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

template <std::size_t Rank, typename Symmetry>
std::unordered_map<FusionTree<Rank, Symmetry>, typename Symmetry::Scalar> FusionTree<Rank, Symmetry>::permute(const util::Permutation& p) const
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
                    if(std::abs(tmp[tree2] + coeff * coeff2) < ScalarTraits<Scalar>::epsilon()) {
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

template <std::size_t Rank, typename Symmetry>
std::unordered_map<FusionTree<Rank, Symmetry>, typename Symmetry::Scalar>
FusionTree<Rank, Symmetry>::swap(const std::size_t& pos) const // swaps sites pos and pos+1
{
    assert(pos < Rank - 1 and "Invalid position for swap.");
    std::unordered_map<FusionTree<Rank, Symmetry>, typename Symmetry::Scalar> out;
    if(pos == 0) {
        if constexpr(Symmetry::ANY_HAS_MULTIPLICITIES) {
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
            if(std::abs(coeff) < ScalarTraits<typename Symmetry::Scalar>::epsilon()) { return out; }
            out.insert(std::make_pair(tree, coeff));
            return out;
        }
    }
    if constexpr(Symmetry::ANY_HAS_MULTIPLICITIES) {
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
            if(std::abs(cgc) < ScalarTraits<typename Symmetry::Scalar>::epsilon()) { continue; }
            out.insert(std::make_pair(tree, cgc));
        }
        return out;
    }
}

template <std::size_t Rank, typename Symmetry>
std::string FusionTree<Rank, Symmetry>::printTree(const std::array<std::string, Rank>& s_uncoupled,
                                                  const std::array<std::string, util::inter_dim(Rank)>& s_intermediates,
                                                  const std::string& s_coupled,
                                                  const std::array<std::string, util::mult_dim(Rank)>& s_multiplicities) const
{
    if constexpr(Rank == 4) {
        std::stringstream ss;
        if(IS_DUAL[0] and !IS_DUAL[1] and !IS_DUAL[2] and !IS_DUAL[3]) {
            ss << "  ☐" << endl;
        } else if(!IS_DUAL[0] and IS_DUAL[1] and !IS_DUAL[2] and !IS_DUAL[3]) {
            ss << "           ☐" << endl;
        } else if(!IS_DUAL[0] and !IS_DUAL[1] and IS_DUAL[2] and !IS_DUAL[3]) {
            ss << "                    ☐" << endl;
        } else if(!IS_DUAL[0] and !IS_DUAL[1] and !IS_DUAL[2] and IS_DUAL[3]) {
            ss << "                             ☐" << endl;
        }

        else if(IS_DUAL[0] and IS_DUAL[1] and !IS_DUAL[2] and !IS_DUAL[3]) {
            ss << "  ☐        ☐" << endl;
        } else if(IS_DUAL[0] and !IS_DUAL[1] and IS_DUAL[2] and !IS_DUAL[3]) {
            ss << "  ☐                 ☐" << endl;
        } else if(IS_DUAL[0] and !IS_DUAL[1] and !IS_DUAL[2] and IS_DUAL[3]) {
            ss << "  ☐                          ☐" << endl;
        } else if(!IS_DUAL[0] and IS_DUAL[1] and IS_DUAL[2] and !IS_DUAL[3]) {
            ss << "           ☐        ☐" << endl;
        } else if(!IS_DUAL[0] and IS_DUAL[1] and !IS_DUAL[2] and IS_DUAL[3]) {
            ss << "           ☐                 ☐" << endl;
        } else if(!IS_DUAL[0] and !IS_DUAL[1] and IS_DUAL[2] and IS_DUAL[3]) {
            ss << "                    ☐        ☐" << endl;
        }

        if(!IS_DUAL[0] and IS_DUAL[1] and IS_DUAL[2] and IS_DUAL[3]) {
            ss << "           ☐        ☐        ☐" << endl;
        } else if(IS_DUAL[0] and !IS_DUAL[1] and IS_DUAL[2] and IS_DUAL[3]) {
            ss << "  ☐                 ☐        ☐" << endl;
        } else if(IS_DUAL[0] and IS_DUAL[1] and !IS_DUAL[2] and IS_DUAL[3]) {
            ss << "  ☐        ☐                 ☐" << endl;
        } else if(IS_DUAL[0] and IS_DUAL[1] and IS_DUAL[2] and !IS_DUAL[3]) {
            ss << "  ☐        ☐        ☐" << endl;
        }

        else if(IS_DUAL[0] and IS_DUAL[1] and IS_DUAL[2] and IS_DUAL[3]) {
            ss << "  ☐        ☐        ☐        ☐" << endl;
        }

        ss << s_uncoupled[0] << "    " << s_uncoupled[1] << "     " << s_uncoupled[2] << "      " << s_uncoupled[3] << endl;
        ss << "   \\     /        /        /\n";
        ss << "    \\   /        /        /\n";
        ss << "     " << s_multiplicities[0] << "         /        /\n";
        ss << "       \\       /        /\n";
        ss << "        \\" << s_intermediates[0] << " /        /\n";
        ss << "         \\   /        /\n";
        ss << "          " << s_multiplicities[1] << "         /\n";
        ss << "            \\       /\n";
        ss << "             \\" << s_intermediates[1] << " /\n";
        ss << "              \\   /\n";
        ss << "               " << s_multiplicities[2] << "\n";
        ss << "                |\n";
        ss << "                |\n";
        ss << "               " << s_coupled << "\n";
        return ss.str();
    }

    if constexpr(Rank == 3) {
        std::stringstream ss;
        if(IS_DUAL[0] and !IS_DUAL[1] and !IS_DUAL[2]) {
            ss << "  ☐" << endl;
        } else if(!IS_DUAL[0] and IS_DUAL[1] and !IS_DUAL[2]) {
            ss << "           ☐" << endl;
        } else if(!IS_DUAL[0] and !IS_DUAL[1] and IS_DUAL[2]) {
            ss << "                    ☐" << endl;
        } else if(IS_DUAL[0] and IS_DUAL[1] and !IS_DUAL[2]) {
            ss << "  ☐        ☐" << endl;
        } else if(IS_DUAL[0] and !IS_DUAL[1] and IS_DUAL[2]) {
            ss << "  ☐                 ☐" << endl;
        } else if(!IS_DUAL[0] and IS_DUAL[1] and IS_DUAL[2]) {
            ss << "           ☐        ☐" << endl;
        } else if(IS_DUAL[0] and IS_DUAL[1] and IS_DUAL[2]) {
            ss << "  ☐        ☐        ☐" << endl;
        }
        ss << s_uncoupled[0] << "    " << s_uncoupled[1] << "      " << s_uncoupled[2] << endl;
        ss << "   \\     /        /\n";
        ss << "    \\   /        /\n";
        ss << "     " << s_multiplicities[0] << "         /\n";
        ss << "       \\       /\n";
        ss << "        \\" << s_intermediates[0] << " /\n";
        ss << "         \\   /\n";
        ss << "          " << s_multiplicities[1] << "\n";
        ss << "           |\n";
        ss << "           |\n";
        ss << "          " << s_coupled << "\n";
        return ss.str();
    }
    if constexpr(Rank == 2) {
        std::stringstream ss;
        if(IS_DUAL[0] and !IS_DUAL[1]) {
            ss << "  ☐" << endl;
        } else if(IS_DUAL[1] and !IS_DUAL[0]) {
            ss << "           ☐" << endl;
        } else if(IS_DUAL[0] and IS_DUAL[1]) {
            ss << "  ☐        ☐" << endl;
        }
        ss << s_uncoupled[0] << "    " << s_uncoupled[1] << endl;
        ;
        ss << "  \\     /\n";
        ss << "   \\   /\n";
        ss << "    " << s_multiplicities[0] << endl;
        ss << "     |\n";
        ss << "     |\n";
        ss << "    " << s_coupled << endl;
        return ss.str();
    }
    if constexpr(Rank == 1) {
        assert(s_uncoupled[0] == s_coupled);
        std::stringstream ss;
        if(IS_DUAL[0]) { ss << " ☐" << endl; }
        ss << s_uncoupled[0] << "\n";
        ss << " |\n";
        ss << " |\n";
        ss << " |\n";
        ss << " |\n";
        ss << s_coupled << "\n";
        return ss.str();
    }
    if constexpr(Rank == 0) {
        std::stringstream ss;
        ss << "0\n";
        ss << "|\n";
        ss << "|\n";
        ss << "|\n";
        ss << "|\n";
        ss << "0\n";
        return ss.str();
    }
}

} // namespace Xped

#if __has_include("FusionTree.gen.cpp")
#    include "FusionTree.gen.cpp"
#endif
