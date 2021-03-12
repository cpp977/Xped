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

#endif
