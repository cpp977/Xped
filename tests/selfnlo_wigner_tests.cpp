// Validate the self-contained Wigner 3j/6j/9j symbols (Xped/Symmetry/WignerSelf.hpp)
// against the GNU Scientific Library:
//   g++ -std=c++20 -I include tests/selfnlo_wigner_tests.cpp -lgsl -lgslcblas -o wigner_tests
//
// Exhaustive scan over small angular momenta plus random samples at larger j.

#include <cmath>
#include <cstdio>
#include <random>

#include <gsl/gsl_sf_coupling.h>

#include "Xped/Symmetry/WignerSelf.hpp"

static int g_failures = 0;
static double g_max3 = 0, g_max6 = 0, g_max9 = 0;

static void check3(int a, int b, int c, int ma, int mb, int mc)
{
    const double ref = gsl_sf_coupling_3j(a, b, c, ma, mb, mc);
    const double mine = Xped::wigner::coupling_3j(a, b, c, ma, mb, mc);
    const double d = std::fabs(ref - mine);
    g_max3 = std::max(g_max3, d);
    if(d > 1e-10) {
        std::printf("3j mismatch (%d %d %d | %d %d %d): gsl=%.15e self=%.15e\n", a, b, c, ma, mb, mc, ref, mine);
        ++g_failures;
    }
}

static void check6(int a, int b, int c, int d, int e, int f)
{
    const double ref = gsl_sf_coupling_6j(a, b, c, d, e, f);
    const double mine = Xped::wigner::coupling_6j(a, b, c, d, e, f);
    const double diff = std::fabs(ref - mine);
    g_max6 = std::max(g_max6, diff);
    if(diff > 1e-10) {
        std::printf("6j mismatch {%d %d %d; %d %d %d}: gsl=%.15e self=%.15e\n", a, b, c, d, e, f, ref, mine);
        ++g_failures;
    }
}

static void check9(int a, int b, int c, int d, int e, int f, int g, int h, int i)
{
    const double ref = gsl_sf_coupling_9j(a, b, c, d, e, f, g, h, i);
    const double mine = Xped::wigner::coupling_9j(a, b, c, d, e, f, g, h, i);
    const double diff = std::fabs(ref - mine);
    g_max9 = std::max(g_max9, diff);
    if(diff > 1e-10) {
        std::printf("9j mismatch: gsl=%.15e self=%.15e (%d %d %d; %d %d %d; %d %d %d)\n", ref, mine, a, b, c, d, e, f, g, h, i);
        ++g_failures;
    }
}

int main()
{
    // ---- 3j exhaustive: all two_j <= 12, all valid m ----
    long n3 = 0;
    for(int a = 0; a <= 12; ++a)
        for(int b = 0; b <= 12; ++b)
            for(int c = std::abs(a - b); c <= std::min(12, a + b); c += 2)
                for(int ma = -a; ma <= a; ma += 2)
                    for(int mb = -b; mb <= b; mb += 2) {
                        check3(a, b, c, ma, mb, -ma - mb);
                        ++n3;
                    }

    // ---- 6j exhaustive: all two_j <= 10 (with triangle prefilter on one triad) ----
    long n6 = 0;
    for(int a = 0; a <= 10; ++a)
        for(int b = 0; b <= 10; ++b)
            for(int c = std::abs(a - b); c <= std::min(10, a + b); c += 2)
                for(int d = 0; d <= 10; ++d)
                    for(int e = 0; e <= 10; ++e)
                        for(int f = 0; f <= 10; ++f) {
                            check6(a, b, c, d, e, f);
                            ++n6;
                        }

    // ---- 9j exhaustive small + random larger ----
    long n9 = 0;
    for(int a = 0; a <= 4; ++a)
        for(int b = 0; b <= 4; ++b)
            for(int c = 0; c <= 4; ++c)
                for(int d = 0; d <= 4; ++d)
                    for(int e = 0; e <= 4; ++e)
                        for(int f = 0; f <= 4; ++f)
                            for(int g = 0; g <= 4; ++g)
                                for(int h = 0; h <= 4; ++h)
                                    for(int i = 0; i <= 4; ++i) {
                                        check9(a, b, c, d, e, f, g, h, i);
                                        ++n9;
                                    }

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> J(0, 40), J9(0, 20);
    for(int k = 0; k < 20000; ++k) {
        const int a = J(rng), b = J(rng), c = J(rng);
        std::uniform_int_distribution<int> Ma(0, a), Mb(0, b);
        const int ma = -a + 2 * Ma(rng), mb = -b + 2 * Mb(rng);
        if((a + b + c) % 2 == 0) { check3(a, b, c, ma, mb, -ma - mb); }
        check6(J(rng), J(rng), J(rng), J(rng), J(rng), J(rng));
        check9(J9(rng), J9(rng), J9(rng), J9(rng), J9(rng), J9(rng), J9(rng), J9(rng), J9(rng));
    }

    std::printf("3j: %ld checks, max |diff| = %.3e\n", n3, g_max3);
    std::printf("6j: %ld checks, max |diff| = %.3e\n", n6, g_max6);
    std::printf("9j: %ld checks (+random), max |diff| = %.3e\n", n9, g_max9);
    if(g_failures == 0) {
        std::printf("\nWIGNER SELF-CONTAINED TESTS PASSED\n");
        return 0;
    }
    std::printf("\n%d FAILURES\n", g_failures);
    return 1;
}
