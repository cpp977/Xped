#ifndef XPED_MACROS_H
#define XPED_MACROS_H

#include <boost/predef.h>
#include <string>

// Get infos to the used compiler and version. Taken from Eigen (Macros.h)
#if BOOST_COMP_CLANG
const std::string XPED_COMPILER_STR = "clang++";
#elif BOOST_COMP_INTEL
const std::string XPED_COMPILER_STR = "icpc";
#elif BOOST_COMP_GNUC
const std::string XPED_COMPILER_STR = "g++";
#elif __INTEL_LLVM_COMPILER
#    define BOOST_COMP_INTEL_LLVM __INTEL_LLVM_COMPILER
const std::string XPED_COMPILER_STR = "icpx";
#elif BOOST_COMP_MSVC
const std::string XPED_COMPILER_STR = "msvc";
#else
#    ifdef BOOST_COMP_MSVC
#        error "Unsupported compiler"
#    else
#        pragma error "Unsupported compiler."
#    endif
#endif

#if defined(XPED_USE_MKL)
const std::string XPED_BLAS_STR = "MKL";
#elif defined(XPED_USE_BLAS)
const std::string XPED_BLAS_STR = "OpenBLAS";
#else
const std::string XPED_BLAS_STR = "None";
#endif

// Get supported std of the compiler
#if defined(__cplusplus)
#    if __cplusplus == 201103L
#        define XPED_CXX11 1
#    elif __cplusplus == 201402L
#        define XPED_CXX14 1
#    elif __cplusplus == 201703L
#        define XPED_CXX17 1
#    elif __cplusplus >= 202002L
#        define XPED_CXX20 1
#    endif
#endif

#if BOOST_COMP_GNUC
#    if __GNUC__ > 9
#        define XPED_HAS_NTTP 1
#    else
#        define XPED_HAS_NTTP 0
#    endif
#elif BOOST_COMP_CLANG
#    if __clang_major__ > 11
#        define XPED_HAS_NTTP 1
#    else
#        define XPED_HAS_NTTP 0
#    endif
#elif BOOST_COMP_INTEL_LLVM
#    if __INTEL_LLVM_COMPILER >= 20210400
#        define XPED_HAS_NTTP 1
#    else
#        define XPED_HAS_NTTP 0
#    endif
#elif BOOST_COMP_INTEL
#    define XPED_HAS_NTTP 0
#elif BOOST_COMP_MSVC
#    if _MSC_VER >= 1930
#        define XPED_HAS_NTTP 1
#    else
#        define XPED_HAS_NTTP 0
#    endif
#endif

#ifndef XPED_HAS_NTTP
#    error "Compiler has no non-type-template-parameter support"
#endif

#if __has_include("boost/functional/hash.hpp")
#    define XPED_HAS_BOOST_HASH_COMBINE 1
#endif

#ifndef XPED_EFFICIENCY_MODEL
#    define XPED_TIME_EFFICIENT
#endif

#ifndef SPDLOG_ACTIVE_LEVEL
#    define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_OFF
#endif

// clang-format off
#define XPED_INIT_TREE_CACHE_VARIABLE(VARIABLE_NAME, CACHE_SIZE) \
namespace Xped {\
template <std::size_t Rank, typename Symmetry> \
struct FusionTree; \
} \
template <int shift, std::size_t Rank, std::size_t CoRank, typename Symmetry> \
struct CacheManager \
{ \
 typedef Xped::FusionTree<CoRank, Symmetry> CoTree;         \
    typedef Xped::FusionTree<CoRank + shift, Symmetry> NewCoTree; \
    typedef Xped::FusionTree<Rank, Symmetry> Tree; \
    typedef Xped::FusionTree<Rank - shift, Symmetry> NewTree; \
    typedef typename Symmetry::Scalar Scalar; \
    typedef LRU::Cache<std::tuple<Tree, CoTree, Xped::util::Permutation>, std::unordered_map<std::pair<NewTree, NewCoTree>, Scalar>> CacheType; \
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
#ifdef XPED_USE_MPI
#    define XPED_MPI_BARRIER(comm) MPI_Barrier(comm);
#else
#    define XPED_MPI_BARRIER(comm)
#endif

#if defined XPED_USE_EIGEN_TENSOR_LIB && defined XPED_USE_EIGEN_MATRIX_LIB && defined XPED_USE_EIGEN_VECTOR_LIB
#    define XPED_CONST const
#elif defined XPED_USE_ARRAY_TENSOR_LIB && defined XPED_USE_EIGEN_MATRIX_LIB && defined XPED_USE_EIGEN_VECTOR_LIB
#    define XPED_CONST const
#elif defined XPED_USE_CYCLOPS_TENSOR_LIB && defined XPED_USE_CYCLOPS_MATRIX_LIB && defined XPED_USE_CYCLOPS_VECTOR_LIB
#    define XPED_CONST
#else
#    error "You specified an invalid combination of plain matrix library, plain tensor library and plain vector library."
#endif

#endif
