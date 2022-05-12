#ifndef XPED_RANDOM_H__
#define XPED_RANDOM_H__

#include <complex>
#include <random>
#include <thread>

namespace Xped::random {

template <typename Scalar, typename RealScalar>
inline Scalar threadSafeRandUniform(RealScalar, RealScalar, bool = false){};

template <>
inline double threadSafeRandUniform<double, double>(double min, double max, bool FIXED_SEED)
{
    static thread_local std::mt19937 generatorUniformReal(std::random_device{}());
    if(FIXED_SEED) generatorUniformReal.seed(std::time(0));
    std::uniform_real_distribution<double> distribution(min, max);
    return distribution(generatorUniformReal);
}

template <>
inline std::complex<double> threadSafeRandUniform<std::complex<double>, double>(double min, double max, bool FIXED_SEED)
{
    static thread_local std::mt19937 generatorUniformComplex(std::random_device{}());
    if(FIXED_SEED) generatorUniformComplex.seed(std::time(0));
    std::uniform_real_distribution<double> distribution(min, max);
    return std::complex<double>(distribution(generatorUniformComplex), distribution(generatorUniformComplex));
}

template <>
inline int threadSafeRandUniform<int, int>(int min, int max, bool FIXED_SEED)
{
    static thread_local std::mt19937 generatorUniformInt(std::random_device{}());
    if(FIXED_SEED) generatorUniformInt.seed(std::time(0));
    std::uniform_int_distribution<int> distribution(min, max);
    return distribution(generatorUniformInt);
}
} // namespace Xped::random
#endif
