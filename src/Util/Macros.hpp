#ifndef XPED_MACROS_H
#define XPED_MACROS_H

// Get infos to the used compiler and version. Taken from Eigen (Macros.h)
#ifdef __GNUC__
#    define XPED_COMP_GNUC 1
#else
#    define XPED_COMP_GNUC 0
#endif

#if COMP_GNUC
#    define XPED_GNUC_AT_LEAST(x, y) ((__GNUC__ == x && __GNUC_MINOR__ >= y) || __GNUC__ > x)
#    define XPED_GNUC_AT_MOST(x, y) ((__GNUC__ == x && __GNUC_MINOR__ <= y) || __GNUC__ < x)
#    define XPED_GNUC_AT(x, y) (__GNUC__ == x && __GNUC_MINOR__ == y)
#else
#    define XPED_GNUC_AT_LEAST(x, y) 0
#    define XPED_GNUC_AT_MOST(x, y) 0
#    define XPED_GNUC_AT(x, y) 0
#endif

#if defined(__clang__)
#    define XPED_COMP_CLANG (__clang_major__ * 100 + __clang_minor__)
#else
#    define XPED_COMP_CLANG 0
#endif

// Get supported std of the compiler
#if defined(__cplusplus)
#    if __cplusplus == 201103L
#        define XPED_CXX11 1
#    elif __cplusplus == 201402L
#        define XPED_CXX14 1
#    elif __cplusplus >= 201703L
#        define XPED_CXX17 1
#    endif
#endif

#if __has_include("boost/functional/hash.hpp")
#    define XPED_HAS_BOOST_HASH_COMBINE 1
#endif

#ifndef XPED_EFFICIENCY_MODEL
#    define XPED_TIME_EFFICIENT
#endif
#ifdef XPED_TIME_EFFICIENT
#    pragma message("Xped tries to be time efficient.")
#elif defined(XPED_MEMORY_EFFICIENT)
#    pragma message("Xped tries to be memory efficient.")
#endif

// clang-format off
#define XPED_INIT_TREE_CACHE_VARIABLE(VARIABLE_NAME, CACHE_SIZE) \
template <std::size_t Rank, typename Symmetry> \
struct FusionTree; \
struct Permutation; \
template <int shift, std::size_t Rank, std::size_t CoRank, typename Symmetry> \
struct CacheManager \
{ \
    typedef FusionTree<CoRank, Symmetry> CoTree; \
    typedef FusionTree<CoRank + shift, Symmetry> NewCoTree; \
    typedef FusionTree<Rank, Symmetry> Tree; \
    typedef FusionTree<Rank - shift, Symmetry> NewTree; \
    typedef typename Symmetry::Scalar Scalar; \
    typedef LRU::Cache<std::tuple<Tree, CoTree, Permutation>, std::unordered_map<std::pair<NewTree, NewCoTree>, Scalar>> CacheType; \
    CacheManager(std::size_t cache_size) \
    { \
        cache = CacheType(cache_size); \
        cache.monitor(); \
    } \
    CacheType cache; \
}; \
template <int shift, std::size_t Rank, std::size_t CoRank, typename Symmetry> \
CacheManager<shift, Rank, CoRank, Symmetry> VARIABLE_NAME(CACHE_SIZE);

// clang-format on
#ifdef XPED_USE_OPENMPI
#    define XPED_MPI_BARRIER(comm) MPI_Barrier(comm);
#else
#    define XPED_MPI_BARRIER(comm)
#endif

#endif
