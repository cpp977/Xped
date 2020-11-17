#ifndef HASH_H_
#define HASH_H_

#include <boost/functional/hash.hpp>

#include "../symmetry/qarray.hpp"

//forward declaration
template<std::size_t Rank, typename Symmetry> struct FusionTree;

namespace std
{
        /**Hashes an array of quantum numbers using boost's \p hash_combine.*/
        template<size_t Nq, size_t Nlegs>
        struct hash<std::array<qarray<Nq>,Nlegs> >
        {
                inline size_t operator()(const std::array<qarray<Nq>,Nlegs> &ix) const
                {
                        size_t seed = 0;
                        for (size_t leg=0; leg<Nlegs; ++leg)
                                for (size_t q=0; q<Nq; ++q)
                                        {
                                                boost::hash_combine(seed, ix[leg][q]);
                                        }
                        return seed;
                }
        };
        
        template<size_t Nq>
        struct hash<qarray<Nq> >
        {
                inline size_t operator()(const qarray<Nq> &ix) const
                {
                        size_t seed = 0;
                        for (size_t q=0; q<Nq; ++q)
                                {
                                        boost::hash_combine(seed, ix[q]);
                                }
                        return seed;
                }
        };

        template<size_t Rank, typename Symmetry>
        struct hash<FusionTree<Rank,Symmetry> >
        {
                inline size_t operator()(const FusionTree<Rank,Symmetry> &ix) const
                {
                        size_t seed = 0;
                        for (const auto& q: ix.q_uncoupled)
                                {
                                        for (size_t nq=0; nq<Symmetry::Nq; ++nq)
                                                {
                                                        boost::hash_combine(seed, q[nq]);
                                                }
                                }
                        for (size_t nq=0; nq<Symmetry::Nq; ++nq)
                                {
                                        boost::hash_combine(seed, ix.q_coupled[nq]);
                                }
                        for (const auto& q: ix.q_intermediates)
                                {
                                        for (size_t nq=0; nq<Symmetry::Nq; ++nq)
                                                {
                                                        boost::hash_combine(seed, q[nq]);
                                                }
                                }
                        for (const auto& b: ix.IS_DUAL)
                                {
                                        boost::hash_combine(seed, b);
                                }
                        for (const auto& n: ix.multiplicities)
                                {
                                        boost::hash_combine(seed, n);
                                }
                        return seed;
                }
        };

        template<size_t Rank, size_t CoRank, typename Symmetry>
        struct hash<std::pair<FusionTree<Rank,Symmetry>, FusionTree<CoRank,Symmetry> > >
        {
                inline size_t operator()(const std::pair<FusionTree<Rank,Symmetry>, FusionTree<CoRank,Symmetry> > &ix) const
                {
                        size_t seed = 0;
                        for (const auto& q: ix.first.q_uncoupled)
                                {
                                        for (size_t nq=0; nq<Symmetry::Nq; ++nq)
                                                {
                                                        boost::hash_combine(seed, q[nq]);
                                                }
                                }
                        for (size_t nq=0; nq<Symmetry::Nq; ++nq)
                                {
                                        boost::hash_combine(seed, ix.first.q_coupled[nq]);
                                }
                        for (const auto& q: ix.first.q_intermediates)
                                {
                                        for (size_t nq=0; nq<Symmetry::Nq; ++nq)
                                                {
                                                        boost::hash_combine(seed, q[nq]);
                                                }
                                }
                        for (const auto& b: ix.first.IS_DUAL)
                                {
                                        boost::hash_combine(seed, b);
                                }
                        for (const auto& n: ix.first.multiplicities)
                                {
                                        boost::hash_combine(seed, n);
                                }
                        for (const auto& q: ix.second.q_uncoupled)
                                {
                                        for (size_t nq=0; nq<Symmetry::Nq; ++nq)
                                                {
                                                        boost::hash_combine(seed, q[nq]);
                                                }
                                }
                        for (size_t nq=0; nq<Symmetry::Nq; ++nq)
                                {
                                        boost::hash_combine(seed, ix.second.q_coupled[nq]);
                                }
                        for (const auto& q: ix.second.q_intermediates)
                                {
                                        for (size_t nq=0; nq<Symmetry::Nq; ++nq)
                                                {
                                                        boost::hash_combine(seed, q[nq]);
                                                }
                                }
                        for (const auto& b: ix.second.IS_DUAL)
                                {
                                        boost::hash_combine(seed, b);
                                }
                        for (const auto& n: ix.second.multiplicities)
                                {
                                        boost::hash_combine(seed, n);
                                }
                        return seed;
                }
        };
}        
#endif
