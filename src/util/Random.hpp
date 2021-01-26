#ifndef RANDOM_H__
#define RANDOM_H__

#include <random>
#include <thread>
#include <complex>

namespace util::random {
        template<typename Scalar, typename RealScalar> Scalar threadSafeRandUniform (RealScalar, RealScalar, bool=false) {};

        template<>
        double threadSafeRandUniform<double,double> (double min, double max, bool SEED)
        {
                static thread_local std::mt19937 generatorUniformReal(std::random_device{}());
                if (SEED) generatorUniformReal.seed(std::time(0));
                std::uniform_real_distribution<double> distribution(min, max);
                return distribution(generatorUniformReal);
        }

        template<>
        std::complex<double> threadSafeRandUniform<std::complex<double>, double> (double min, double max, bool FIXED_SEED)
        {
                static thread_local std::mt19937 generatorUniformComplex(std::random_device{}());
                if (FIXED_SEED) generatorUniformComplex.seed(std::time(0));
                std::uniform_real_distribution<double> distribution(min, max);
                return std::complex<double>(distribution(generatorUniformComplex), distribution(generatorUniformComplex));
        }

        template<>
        int threadSafeRandUniform<int,int> (int min, int max, bool SEED)
        {
                static thread_local std::mt19937 generatorUniformInt(std::random_device{}());
                if (SEED) generatorUniformInt.seed(std::time(0));
                std::uniform_int_distribution<int> distribution(min, max);
                return distribution(generatorUniformInt);
        }
}
#endif
