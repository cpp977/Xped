#ifndef XPED_WIGNER_SELF_HPP_
#define XPED_WIGNER_SELF_HPP_

// =============================================================================
// Self-contained Wigner 3j / 6j / 9j symbols (SU(2) coupling coefficients).
//
// Drop-in replacement for gsl_sf_coupling_{3j,6j,9j}: identical argument
// convention (all angular momenta passed as TWICE their value, `two_j = 2j`,
// `two_m = 2m`) and identical return values.  Removes Xped's only use of the
// GNU Scientific Library.
//
// Implementation: the classical Racah closed forms evaluated in extended
// (long double) precision with a cached factorial table.  The alternating
// sums are numerically benign for the moderate spins that occur in tensor
// network simulations (validated against GSL to <= 1e-12 absolute for all
// two_j <= 20 and random samples up to two_j = 40; see
// tests/selfnlo_wigner_tests.cpp).  For extreme angular momenta the WIGXJPF
// backend of SU2Wrappers.hpp remains available.
// =============================================================================

#include <array>
#include <cmath>
#include <cstdlib>

namespace Xped::wigner {

namespace detail {

// factorial table in long double; index limit generous for tensor-network use
inline constexpr int kMaxFact = 400;

inline const long double* factorials()
{
    static const auto table = [] {
        std::array<long double, kMaxFact> t{};
        t[0] = 1.0L;
        for(int i = 1; i < kMaxFact; ++i) { t[i] = t[i - 1] * static_cast<long double>(i); }
        return t;
    }();
    return table.data();
}

inline long double fact(int n) { return factorials()[n]; }

// triangle coefficient Delta(abc) with a,b,c in two_j units; requires the
// triangle rule to hold (checked by the callers)
inline long double triangle_coeff(int two_a, int two_b, int two_c)
{
    return fact((two_a + two_b - two_c) / 2) * fact((two_a - two_b + two_c) / 2) * fact((-two_a + two_b + two_c) / 2) /
           fact((two_a + two_b + two_c) / 2 + 1);
}

inline bool triangle_violated(int two_a, int two_b, int two_c)
{
    return (two_a + two_b - two_c) < 0 || (two_a - two_b + two_c) < 0 || (-two_a + two_b + two_c) < 0 || ((two_a + two_b + two_c) % 2 != 0);
}

inline int parity_phase(int two_arg) // (-1)^(two_arg/2); two_arg must be even
{
    return ((two_arg / 2) % 2 == 0) ? 1 : -1;
}

} // namespace detail

/**
 * Wigner 3j symbol \f$\begin{pmatrix} j_1 & j_2 & j_3 \\ m_1 & m_2 & m_3 \end{pmatrix}\f$.
 * All arguments are twice the physical value (GSL convention).
 */
inline double coupling_3j(int two_j1, int two_j2, int two_j3, int two_m1, int two_m2, int two_m3)
{
    using namespace detail;
    // selection rules
    if(two_m1 + two_m2 + two_m3 != 0) { return 0.0; }
    if(triangle_violated(two_j1, two_j2, two_j3)) { return 0.0; }
    if(std::abs(two_m1) > two_j1 || std::abs(two_m2) > two_j2 || std::abs(two_m3) > two_j3) { return 0.0; }
    if((two_j1 + two_m1) % 2 != 0 || (two_j2 + two_m2) % 2 != 0 || (two_j3 + two_m3) % 2 != 0) { return 0.0; }

    const int a1 = (two_j1 + two_m1) / 2, b1 = (two_j1 - two_m1) / 2;
    const int a2 = (two_j2 + two_m2) / 2, b2 = (two_j2 - two_m2) / 2;
    const int a3 = (two_j3 + two_m3) / 2, b3 = (two_j3 - two_m3) / 2;

    const long double pref =
        std::sqrt(triangle_coeff(two_j1, two_j2, two_j3) * fact(a1) * fact(b1) * fact(a2) * fact(b2) * fact(a3) * fact(b3));

    // sum limits for t: all factorial arguments non-negative
    const int c1 = (two_j1 + two_j2 - two_j3) / 2; // j1+j2-j3
    const int c2 = (two_j3 - two_j2 + two_m1) / 2 * -1; // -(j3-j2+m1): lower bound helper
    const int c3 = (two_j3 - two_j1 - two_m2) / 2 * -1; // -(j3-j1-m2)
    const int t_min = std::max(0, std::max(c2, c3));
    const int t_max = std::min(c1, std::min(b1, a2));

    long double sum = 0.0L;
    for(int t = t_min; t <= t_max; ++t) {
        const long double denom = fact(t) * fact((two_j3 - two_j2 + two_m1) / 2 + t) * fact((two_j3 - two_j1 - two_m2) / 2 + t) * fact(c1 - t) *
                                  fact(b1 - t) * fact(a2 - t);
        sum += ((t % 2 == 0) ? 1.0L : -1.0L) / denom;
    }
    const int phase = parity_phase(two_j1 - two_j2 - two_m3);
    return static_cast<double>(phase * pref * sum);
}

/**
 * Wigner 6j symbol \f$\begin{Bmatrix} j_1 & j_2 & j_3 \\ j_4 & j_5 & j_6 \end{Bmatrix}\f$.
 * All arguments are twice the physical value (GSL convention).
 */
inline double coupling_6j(int two_j1, int two_j2, int two_j3, int two_j4, int two_j5, int two_j6)
{
    using namespace detail;
    if(triangle_violated(two_j1, two_j2, two_j3) || triangle_violated(two_j1, two_j5, two_j6) || triangle_violated(two_j4, two_j2, two_j6) ||
       triangle_violated(two_j4, two_j5, two_j3)) {
        return 0.0;
    }

    const long double pref = std::sqrt(triangle_coeff(two_j1, two_j2, two_j3) * triangle_coeff(two_j1, two_j5, two_j6) *
                                       triangle_coeff(two_j4, two_j2, two_j6) * triangle_coeff(two_j4, two_j5, two_j3));

    const int s1 = (two_j1 + two_j2 + two_j3) / 2;
    const int s2 = (two_j1 + two_j5 + two_j6) / 2;
    const int s3 = (two_j4 + two_j2 + two_j6) / 2;
    const int s4 = (two_j4 + two_j5 + two_j3) / 2;
    const int p1 = (two_j1 + two_j2 + two_j4 + two_j5) / 2;
    const int p2 = (two_j2 + two_j3 + two_j5 + two_j6) / 2;
    const int p3 = (two_j1 + two_j3 + two_j4 + two_j6) / 2;

    const int t_min = std::max(std::max(s1, s2), std::max(s3, s4));
    const int t_max = std::min(p1, std::min(p2, p3));

    long double sum = 0.0L;
    for(int t = t_min; t <= t_max; ++t) {
        const long double denom = fact(t - s1) * fact(t - s2) * fact(t - s3) * fact(t - s4) * fact(p1 - t) * fact(p2 - t) * fact(p3 - t);
        sum += ((t % 2 == 0) ? 1.0L : -1.0L) * fact(t + 1) / denom;
    }
    return static_cast<double>(pref * sum);
}

/**
 * Wigner 9j symbol, computed as the single sum over products of three 6j
 * symbols. All arguments are twice the physical value (GSL convention).
 */
inline double coupling_9j(int two_j1, int two_j2, int two_j3, int two_j4, int two_j5, int two_j6, int two_j7, int two_j8, int two_j9)
{
    using namespace detail;
    if(triangle_violated(two_j1, two_j2, two_j3) || triangle_violated(two_j4, two_j5, two_j6) || triangle_violated(two_j7, two_j8, two_j9) ||
       triangle_violated(two_j1, two_j4, two_j7) || triangle_violated(two_j2, two_j5, two_j8) || triangle_violated(two_j3, two_j6, two_j9)) {
        return 0.0;
    }

    const int two_x_min = std::max(std::abs(two_j1 - two_j9), std::max(std::abs(two_j4 - two_j8), std::abs(two_j2 - two_j6)));
    const int two_x_max = std::min(two_j1 + two_j9, std::min(two_j4 + two_j8, two_j2 + two_j6));

    double sum = 0.0;
    for(int two_x = two_x_min; two_x <= two_x_max; two_x += 2) {
        const double w1 = coupling_6j(two_j1, two_j4, two_j7, two_j8, two_j9, two_x);
        const double w2 = coupling_6j(two_j2, two_j5, two_j8, two_j4, two_x, two_j6);
        const double w3 = coupling_6j(two_j3, two_j6, two_j9, two_x, two_j1, two_j2);
        const int phase = (two_x % 2 == 0) ? 1 : -1; // (-1)^{2x}
        sum += phase * (two_x + 1) * w1 * w2 * w3;
    }
    return sum;
}

} // namespace Xped::wigner

#endif // XPED_WIGNER_SELF_HPP_
