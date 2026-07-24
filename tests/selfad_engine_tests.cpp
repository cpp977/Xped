// Standalone verification of the self-contained AD engine (include/selfad).
//
// Compiles WITHOUT any external dependency:
//   g++ -std=c++20 -I include -I include/selfad tests/selfad_engine_tests.cpp -o selfad_tests
//
// It deliberately re-creates, in miniature, every pattern Xped uses on top of
// the stan API:
//   1. scalar var arithmetic + elementary functions, gradients vs analytic
//   2. Xped's real complex_var/complex_vari headers (compiled verbatim)
//   3. a heap-owning tensor vari (vari_base + chainable_alloc dual
//      inheritance, exactly like Xped/AD/vari_value.hpp), with leak counting
//   4. Xped::reverse_pass_callback_alloc (real header) for closures that
//      capture heap-owning objects
//   5. the CTM checkpointing pattern: a nested tape opened *inside* a chain()
//      call during the outer reverse sweep
//   6. arena_allocator + finite_diff_gradient_auto

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "stan/math/rev.hpp"

#include "Xped/AD/complex_var.hpp"
#include "Xped/AD/reverse_pass_callback_alloc.hpp"

static int g_failures = 0;
#define CHECK(cond)                                                                                                                                  \
    do {                                                                                                                                             \
        if(!(cond)) {                                                                                                                                \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                                                              \
            ++g_failures;                                                                                                                            \
        }                                                                                                                                            \
    } while(0)

static bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol * (1.0 + std::fabs(a) + std::fabs(b)); }

// --------------------------------------------------------------------------
// 1. scalar reverse mode
// --------------------------------------------------------------------------
static void test_scalar()
{
    using stan::math::var;
    stan::math::nested_rev_autodiff nested;

    var x = 1.7;
    var y = 0.9;
    // f = x*y + sqrt(x)/y + exp(y) - x^3 + 5/x - 2
    auto f = x * y + sqrt(x) / y + exp(y) - pow(x, 3) + 5.0 / x - 2.0;
    stan::math::grad(f.vi_);

    const double xd = 1.7, yd = 0.9;
    const double fx = xd * yd + std::sqrt(xd) / yd + std::exp(yd) - std::pow(xd, 3) + 5.0 / xd - 2.0;
    const double dfdx = yd + 0.5 / (std::sqrt(xd) * yd) - 3 * xd * xd - 5.0 / (xd * xd);
    const double dfdy = xd - std::sqrt(xd) / (yd * yd) + std::exp(yd);

    CHECK(near(f.val(), fx, 1e-14));
    CHECK(near(x.adj(), dfdx, 1e-13));
    CHECK(near(y.adj(), dfdy, 1e-13));

    // compound ops and accumulate (TMatrix::sum pattern)
    std::vector<var> vs = {var(1.0), var(2.0), var(3.5)};
    var total = std::accumulate(vs.begin(), vs.end(), var(0.));
    total *= 2.0;
    total /= 4.0;
    total -= 0.25;
    total += vs[0];
    nested.set_zero_all_adjoints();
    stan::math::grad(total.vi_);
    CHECK(near(total.val(), (1.0 + 2.0 + 3.5) / 2.0 - 0.25 + 1.0, 1e-14));
    CHECK(near(vs[0].adj(), 1.5, 1e-14));
    CHECK(near(vs[1].adj(), 0.5, 1e-14));
    CHECK(near(vs[2].adj(), 0.5, 1e-14));

    // comparisons
    CHECK(vs[2] > vs[1]);
    CHECK(vs[1] < 3.0);
    CHECK(2.0 == vs[1]);
    std::printf("scalar reverse mode                ok\n");
}

// --------------------------------------------------------------------------
// 2. Xped's complex var (real headers compiled against the engine)
// --------------------------------------------------------------------------
static void test_complex()
{
    using cvar = stan::math::var_value<std::complex<double>>;
    stan::math::nested_rev_autodiff nested;

    const std::complex<double> z1v(1.5, -0.5), z2v(0.7, 2.1);
    cvar z1 = z1v;
    cvar z2 = z2v;
    cvar w = z1 / z2; // Xped's operator/ via make_callback_var
    auto r = std::real(w); // Xped's std::real overload -> var_value<double>
    stan::math::grad(r.vi_);

    CHECK(near(w.val().real(), (z1v / z2v).real(), 1e-14));
    // Xped convention: z1.adj += r.adj / z2.val  (no conjugation)
    const std::complex<double> expect_z1 = 1.0 / z2v;
    const std::complex<double> expect_z2 = -z1v / (z2v * z2v);
    CHECK(near(z1.adj().real(), expect_z1.real(), 1e-13));
    CHECK(near(z1.adj().imag(), expect_z1.imag(), 1e-13));
    CHECK(near(z2.adj().real(), expect_z2.real(), 1e-13));
    CHECK(near(z2.adj().imag(), expect_z2.imag(), 1e-13));
    std::printf("complex var (Xped headers)         ok\n");
}

// --------------------------------------------------------------------------
// 3.-5. heap-owning tensor vari + callbacks + checkpointing
// --------------------------------------------------------------------------
struct DenseTensor
{
    static inline int alive = 0;
    std::vector<double> d;

    DenseTensor() { ++alive; }
    explicit DenseTensor(std::size_t n, double v = 0.0)
        : d(n, v)
    {
        ++alive;
    }
    DenseTensor(const DenseTensor& o)
        : d(o.d)
    {
        ++alive;
    }
    DenseTensor(DenseTensor&& o) noexcept
        : d(std::move(o.d))
    {
        ++alive;
    }
    DenseTensor& operator=(const DenseTensor&) = default;
    DenseTensor& operator=(DenseTensor&&) = default;
    ~DenseTensor() { --alive; }

    void setZero()
    {
        for(auto& v : d) { v = 0.0; }
    }
};

template <typename T>
struct is_dtensor : std::false_type
{};
template <>
struct is_dtensor<DenseTensor> : std::true_type
{};
template <typename T>
using require_dtensor_v = stan::require_t<stan::bool_constant<is_dtensor<T>::value>>;

namespace stan::math {
// mirrors Xped/AD/vari_value.hpp: heap-owning value/adjoint pair, node in the
// arena, destructor registered through chainable_alloc
template <typename T>
class vari_value<T, require_dtensor_v<T>> : public vari_base, public chainable_alloc
{
public:
    T val_;
    T adj_;

    template <typename S>
    explicit vari_value(const S& x, bool stacked)
        : val_(x)
        , adj_(x.d.size(), 0.0)
    {
        if(stacked) {
            ChainableStack::instance_->var_stack_.push_back(this);
        } else {
            ChainableStack::instance_->var_nochain_stack_.push_back(this);
        }
    }

    inline const T& val() const noexcept { return val_; }
    inline T& adj() noexcept { return adj_; }

    virtual void chain() {}
    inline void init_dependent()
    {
        for(auto& v : adj_.d) { v = 1.0; }
    }
    inline void set_zero_adjoint() final { adj_.setZero(); }
};
} // namespace stan::math

// mirrors Tensor<..., true>: a plain handle around the vari
struct ADT
{
    using vari_type = stan::math::vari_value<DenseTensor>;
    vari_type* vi_;

    ADT()
        : vi_(nullptr)
    {}
    explicit ADT(const DenseTensor& x)
        : vi_(new vari_type(x, false))
    {}
    inline const DenseTensor& val() const noexcept { return vi_->val_; }
    inline DenseTensor& adj() const noexcept { return vi_->adj_; }
};

// elementwise multiplication by a *plain* tensor: the closure captures a
// heap-owning DenseTensor, so it must go through reverse_pass_callback_alloc
static ADT mul_plain(const DenseTensor& w, const ADT& t)
{
    DenseTensor out(t.val().d.size());
    for(std::size_t i = 0; i < out.d.size(); ++i) { out.d[i] = w.d[i] * t.val().d[i]; }
    ADT res(out);
    Xped::reverse_pass_callback_alloc([w, t, res]() mutable {
        for(std::size_t i = 0; i < w.d.size(); ++i) { t.adj().d[i] += w.d[i] * res.adj().d[i]; }
    });
    return res;
}

// elementwise square, tracked (plain reverse_pass_callback: captures only handles)
static ADT square_tracked(const ADT& t)
{
    DenseTensor out(t.val().d.size());
    for(std::size_t i = 0; i < out.d.size(); ++i) { out.d[i] = t.val().d[i] * t.val().d[i]; }
    ADT res(out);
    stan::math::reverse_pass_callback([t, res]() mutable {
        for(std::size_t i = 0; i < res.adj().d.size(); ++i) { t.adj().d[i] += 2.0 * t.val().d[i] * res.adj().d[i]; }
    });
    return res;
}

// elementwise square with checkpointing, exactly the CTM::grow_all pattern:
// untracked forward now, nested re-forward + nested grad() in the reverse pass
static ADT square_checkpointed(const ADT& t)
{
    DenseTensor out(t.val().d.size());
    for(std::size_t i = 0; i < out.d.size(); ++i) { out.d[i] = t.val().d[i] * t.val().d[i]; }
    ADT res(out);
    Xped::reverse_pass_callback_alloc([t, res]() mutable {
        stan::math::nested_rev_autodiff nested;
        ADT inner = square_tracked(t);
        inner.adj() = res.adj();
        stan::math::grad();
    });
    return res;
}

// dot(a, a) -> scalar var (the trace/norm pattern)
static stan::math::var self_dot(const ADT& a)
{
    double tmp = 0;
    for(double v : a.val().d) { tmp += v * v; }
    stan::math::var res(tmp);
    stan::math::reverse_pass_callback([a, res]() mutable {
        for(std::size_t i = 0; i < a.val().d.size(); ++i) { a.adj().d[i] += 2.0 * a.val().d[i] * res.adj(); }
    });
    return res;
}

static void test_tensor_tape(bool with_checkpoint)
{
    const std::size_t n = 5;
    DenseTensor x0(n), w(n);
    for(std::size_t i = 0; i < n; ++i) {
        x0.d[i] = 0.3 + 0.2 * static_cast<double>(i);
        w.d[i] = 1.0 - 0.1 * static_cast<double>(i);
    }

    const int alive_before = DenseTensor::alive;
    std::vector<double> grad_ad(n);
    double eval = 0;
    {
        stan::math::nested_rev_autodiff nested;
        ADT x(x0);
        ADT s = with_checkpoint ? square_checkpointed(x) : square_tracked(x);
        ADT m = mul_plain(w, s);
        auto r = self_dot(m);
        auto e = sqrt(r) + 3.0 * r; // scalar var tail
        eval = e.val();
        stan::math::grad(e.vi_);
        grad_ad = x.adj().d;
    } // ~nested_rev_autodiff unwinds the tape and destroys every DenseTensor

    CHECK(DenseTensor::alive == alive_before);
    CHECK(stan::math::ChainableStack::instance_->var_stack_.empty());
    CHECK(stan::math::ChainableStack::instance_->var_alloc_stack_.empty());

    // analytic: r = sum_i w_i^2 x_i^4,  e = sqrt(r) + 3 r
    double r = 0;
    for(std::size_t i = 0; i < n; ++i) { r += w.d[i] * w.d[i] * std::pow(x0.d[i], 4); }
    CHECK(near(eval, std::sqrt(r) + 3 * r, 1e-13));
    const double dedr = 0.5 / std::sqrt(r) + 3.0;
    for(std::size_t i = 0; i < n; ++i) {
        const double expect = dedr * 4.0 * w.d[i] * w.d[i] * std::pow(x0.d[i], 3);
        CHECK(near(grad_ad[i], expect, 1e-12));
    }
    std::printf("tensor tape (%s)      ok\n", with_checkpoint ? "checkpointed " : "plain        ");
}

// --------------------------------------------------------------------------
// 6. arena allocator + finite differences
// --------------------------------------------------------------------------
struct MiniVec
{
    std::vector<double> d;
    MiniVec() = default;
    explicit MiniVec(int n)
        : d(static_cast<std::size_t>(n), 0.0)
    {}
    double& operator()(int i) { return d[static_cast<std::size_t>(i)]; }
    double operator()(int i) const { return d[static_cast<std::size_t>(i)]; }
    int size() const { return static_cast<int>(d.size()); }
    void resize(int n) { d.resize(static_cast<std::size_t>(n)); }
};

static void test_arena_and_fd()
{
    {
        stan::math::nested_rev_autodiff nested;
        std::vector<double, stan::math::arena_allocator<double>> v;
        for(int i = 0; i < 1000; ++i) { v.push_back(1.0 * i); }
        CHECK(near(v[999], 999.0, 1e-14));
    }

    auto f = [](const MiniVec& x) { return x(0) * x(0) * x(1) + std::sin(x(1)); };
    MiniVec x(2);
    x(0) = 1.2;
    x(1) = 0.7;
    double fx = 0;
    MiniVec g;
    stan::math::finite_diff_gradient_auto(f, x, fx, g);
    CHECK(near(fx, 1.2 * 1.2 * 0.7 + std::sin(0.7), 1e-14));
    CHECK(near(g(0), 2 * 1.2 * 0.7, 1e-9));
    CHECK(near(g(1), 1.2 * 1.2 + std::cos(0.7), 1e-9));

    // finite-difference vs AD on the tensor pipeline
    const std::size_t n = 4;
    DenseTensor x0(n), w(n);
    for(std::size_t i = 0; i < n; ++i) {
        x0.d[i] = 0.4 + 0.15 * static_cast<double>(i);
        w.d[i] = 0.8 + 0.05 * static_cast<double>(i);
    }
    auto f_plain = [&](const MiniVec& xs) {
        double r = 0;
        for(std::size_t i = 0; i < n; ++i) {
            const double m = w.d[i] * xs(static_cast<int>(i)) * xs(static_cast<int>(i));
            r += m * m;
        }
        return std::sqrt(r) + 3 * r;
    };
    MiniVec xs(static_cast<int>(n));
    for(std::size_t i = 0; i < n; ++i) { xs(static_cast<int>(i)) = x0.d[i]; }
    double fd_val = 0;
    MiniVec fd_grad;
    stan::math::finite_diff_gradient_auto(f_plain, xs, fd_val, fd_grad);

    std::vector<double> grad_ad(n);
    {
        stan::math::nested_rev_autodiff nested;
        ADT x(x0);
        ADT m = mul_plain(w, square_checkpointed(x));
        auto e = sqrt(self_dot(m)) + 3.0 * self_dot(m);
        stan::math::grad(e.vi_);
        grad_ad = x.adj().d;
    }
    for(std::size_t i = 0; i < n; ++i) { CHECK(near(grad_ad[i], fd_grad(static_cast<int>(i)), 1e-7)); }
    std::printf("arena allocator + finite diff      ok\n");
}

// --------------------------------------------------------------------------
// tape reuse across passes (optimizer loop pattern) + print_stack smoke
// --------------------------------------------------------------------------
static void test_reuse()
{
    for(int pass = 0; pass < 3; ++pass) {
        stan::math::nested_rev_autodiff nested;
        stan::math::var x = 2.0 + pass;
        auto y = x * x + 1.0 / x;
        stan::math::grad(y.vi_);
        const double xd = 2.0 + pass;
        CHECK(near(x.adj(), 2 * xd - 1.0 / (xd * xd), 1e-13));
    }
    CHECK(stan::math::ChainableStack::instance_->var_stack_.empty());
    stan::math::recover_memory();
    std::printf("tape reuse across passes           ok\n");
}

int main()
{
    test_scalar();
    test_complex();
    test_tensor_tape(false);
    test_tensor_tape(true);
    test_arena_and_fd();
    test_reuse();

    if(g_failures == 0) {
        std::printf("\nALL SELF-CONTAINED AD ENGINE TESTS PASSED\n");
        return 0;
    }
    std::printf("\n%d FAILURES\n", g_failures);
    return 1;
}
