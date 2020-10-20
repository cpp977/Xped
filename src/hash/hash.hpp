#ifndef HASH_H_
#define HASH_H_

#include <boost/functional/hash.hpp>

#include "../symmetry/qarray.hpp"

namespace std
{
/**Hashes an array of quantum numbers using boost's \p hash_combine for the dictionaries in Biped, Multipede.*/
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

}        
#endif
