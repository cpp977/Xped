#ifndef XPED_SELFAD_CORE_HPP_
#define XPED_SELFAD_CORE_HPP_

// =============================================================================
// Xped self-contained reverse-mode automatic differentiation engine.
//
// This is a clean-room, header-only implementation of the small subset of the
// stan-math reverse-mode API that Xped uses.  It keeps the `stan` / `stan::math`
// namespaces and the include paths (`stan/math/rev.hpp`, ...) so that the rest
// of Xped compiles completely unchanged, but it has **zero external
// dependencies** (no stan-math, no TBB, no SUNDIALS, no Boost).
//
// The design follows the same architecture as stan-math's reverse mode:
//
//   * `stack_alloc`            -- bump-pointer arena for vari nodes
//   * `vari_base`              -- tape node interface (chain / set_zero_adjoint),
//                                 arena-allocated via class operator new
//   * `chainable_alloc`        -- registry base for nodes that own heap memory
//                                 and therefore need their destructor run when
//                                 the tape is unwound (Xped's block-sparse
//                                 tensor varis inherit from this)
//   * `ChainableStack`         -- global tape: var_stack_ (chained nodes),
//                                 var_nochain_stack_ (leaf nodes),
//                                 var_alloc_stack_ (destructor registry)
//   * `vari_value<T>/var_value<T>` -- value/adjoint node and its handle for
//                                 floating-point scalars; Xped provides its own
//                                 specialisations for tensors and complex
//   * `reverse_pass_callback`  -- record a closure on the tape
//   * `nested_rev_autodiff`    -- scoped nested tape (used for the outer
//                                 gradient pass and for checkpointing)
//   * `grad()`                 -- reverse sweep over the innermost nested
//                                 tape segment
//
// All tensor-level adjoint rules (SVD, contraction, permutation, ...) live in
// Xped itself (Xped/AD/ADTensor.hpp) and are untouched.
// =============================================================================

#include <complex>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace stan {

// ---- minimal type-trait helpers (API-compatible subset) ---------------------

template <bool B>
using bool_constant = std::integral_constant<bool, B>;

template <typename Check>
using require_t = std::enable_if_t<Check::value>;

template <typename Check>
using require_not_t = std::enable_if_t<!Check::value>;

template <typename From, typename To>
using require_convertible_t = std::enable_if_t<std::is_convertible<From, To>::value>;

namespace math {

class vari_base;
class chainable_alloc;

// ---- arena allocator (bump pointer over malloc'd blocks) --------------------

class stack_alloc
{
public:
    static constexpr std::size_t DEFAULT_INITIAL_NBYTES = 1u << 16; // 64 KiB
    static constexpr std::size_t ALIGNMENT = 32;                    // safe for all scalar types

private:
    std::vector<char*> blocks_;
    std::vector<std::size_t> sizes_;
    std::size_t cur_block_{0};
    char* next_loc_{nullptr};
    char* cur_block_end_{nullptr};

    // saved positions for nested tapes
    std::vector<std::size_t> nested_cur_blocks_;
    std::vector<char*> nested_next_locs_;
    std::vector<char*> nested_cur_block_ends_;

    static std::size_t round_up(std::size_t len) { return (len + ALIGNMENT - 1) & ~(ALIGNMENT - 1); }

    char* alloc_block(std::size_t min_size)
    {
        std::size_t size = sizes_.empty() ? DEFAULT_INITIAL_NBYTES : 2 * sizes_.back();
        if(size < min_size) { size = round_up(min_size); }
        char* block = static_cast<char*>(std::malloc(size));
        if(block == nullptr) { throw std::bad_alloc(); }
        blocks_.push_back(block);
        sizes_.push_back(size);
        return block;
    }

    char* move_to_next_block(std::size_t len)
    {
        ++cur_block_;
        // find an existing block that fits, else allocate a new one at the end
        while(cur_block_ < blocks_.size() && sizes_[cur_block_] < len) { ++cur_block_; }
        if(cur_block_ >= blocks_.size()) {
            alloc_block(len);
            cur_block_ = blocks_.size() - 1;
        }
        char* result = blocks_[cur_block_];
        next_loc_ = result + len;
        cur_block_end_ = blocks_[cur_block_] + sizes_[cur_block_];
        return result;
    }

public:
    stack_alloc()
    {
        alloc_block(DEFAULT_INITIAL_NBYTES);
        next_loc_ = blocks_[0];
        cur_block_end_ = blocks_[0] + sizes_[0];
    }

    stack_alloc(const stack_alloc&) = delete;
    stack_alloc& operator=(const stack_alloc&) = delete;

    ~stack_alloc()
    {
        for(char* block : blocks_) { std::free(block); }
    }

    inline void* alloc(std::size_t len)
    {
        len = round_up(len);
        char* result = next_loc_;
        next_loc_ += len;
        if(next_loc_ > cur_block_end_) { result = move_to_next_block(len); }
        return result;
    }

    template <typename T>
    inline T* alloc_array(std::size_t n)
    {
        return static_cast<T*>(alloc(n * sizeof(T)));
    }

    // release all memory back to position zero (blocks are kept for reuse)
    inline void recover_all()
    {
        cur_block_ = 0;
        next_loc_ = blocks_[0];
        cur_block_end_ = blocks_[0] + sizes_[0];
    }

    inline void start_nested()
    {
        nested_cur_blocks_.push_back(cur_block_);
        nested_next_locs_.push_back(next_loc_);
        nested_cur_block_ends_.push_back(cur_block_end_);
    }

    inline void recover_nested()
    {
        if(nested_cur_blocks_.empty()) {
            recover_all();
            return;
        }
        cur_block_ = nested_cur_blocks_.back();
        nested_cur_blocks_.pop_back();
        next_loc_ = nested_next_locs_.back();
        nested_next_locs_.pop_back();
        cur_block_end_ = nested_cur_block_ends_.back();
        nested_cur_block_ends_.pop_back();
    }

    // free every block except the first and reset
    inline void free_all()
    {
        for(std::size_t i = 1; i < blocks_.size(); ++i) { std::free(blocks_[i]); }
        blocks_.resize(1);
        sizes_.resize(1);
        recover_all();
    }

    inline std::size_t bytes_allocated() const
    {
        std::size_t used = 0;
        for(std::size_t i = 0; i <= cur_block_ && i < sizes_.size(); ++i) { used += sizes_[i]; }
        return used;
    }
};

// ---- global tape ------------------------------------------------------------

struct AutodiffStackStorage
{
    std::vector<vari_base*> var_stack_;
    std::vector<vari_base*> var_nochain_stack_;
    std::vector<chainable_alloc*> var_alloc_stack_;
    stack_alloc memalloc_;

    // sizes recorded at each start_nested()
    std::vector<std::size_t> nested_var_stack_sizes_;
    std::vector<std::size_t> nested_var_nochain_stack_sizes_;
    std::vector<std::size_t> nested_var_alloc_stack_starts_;
};

namespace internal {
inline AutodiffStackStorage& autodiff_stack_singleton()
{
    static AutodiffStackStorage storage;
    return storage;
}
} // namespace internal

struct ChainableStack
{
    using AutodiffStackStorage = stan::math::AutodiffStackStorage;
    static inline AutodiffStackStorage* instance_ = &internal::autodiff_stack_singleton();
};

// ---- tape node bases --------------------------------------------------------

class vari_base
{
public:
    virtual void chain() {}
    virtual void set_zero_adjoint() = 0;

    // vari nodes live in the arena: allocation is a bump of the arena pointer,
    // deallocation is a no-op (memory is reclaimed wholesale on recover)
    static inline void* operator new(std::size_t nbytes) { return ChainableStack::instance_->memalloc_.alloc(nbytes); }
    static inline void operator delete(void* /* mem */) noexcept { /* no-op: arena memory */ }
};

/**
 * Base class for tape nodes that own non-arena resources (e.g. heap-allocated
 * tensor storage).  Registering here guarantees the destructor is called when
 * the (nested) tape is recovered.  Xped's tensor vari_value inherits from both
 * vari_base and chainable_alloc: the object itself sits in the arena
 * (vari_base::operator new), while `delete` through the chainable_alloc
 * registry runs the destructor and hits the no-op vari_base::operator delete.
 */
class chainable_alloc
{
public:
    chainable_alloc() { ChainableStack::instance_->var_alloc_stack_.push_back(this); }
    virtual ~chainable_alloc() = default;
};

// ---- nested-tape bookkeeping ------------------------------------------------

inline bool empty_nested() { return ChainableStack::instance_->nested_var_stack_sizes_.empty(); }

inline std::size_t nested_size() { return ChainableStack::instance_->var_stack_.size() - ChainableStack::instance_->nested_var_stack_sizes_.back(); }

inline void start_nested()
{
    AutodiffStackStorage* i = ChainableStack::instance_;
    i->nested_var_stack_sizes_.push_back(i->var_stack_.size());
    i->nested_var_nochain_stack_sizes_.push_back(i->var_nochain_stack_.size());
    i->nested_var_alloc_stack_starts_.push_back(i->var_alloc_stack_.size());
    i->memalloc_.start_nested();
}

inline void recover_nested()
{
    AutodiffStackStorage* i = ChainableStack::instance_;
    if(i->nested_var_stack_sizes_.empty()) { throw std::logic_error("recover_nested() called outside of a nested tape"); }

    i->var_stack_.resize(i->nested_var_stack_sizes_.back());
    i->nested_var_stack_sizes_.pop_back();

    i->var_nochain_stack_.resize(i->nested_var_nochain_stack_sizes_.back());
    i->nested_var_nochain_stack_sizes_.pop_back();

    // run destructors of heap-owning nodes created inside the nested tape
    const std::size_t alloc_start = i->nested_var_alloc_stack_starts_.back();
    i->nested_var_alloc_stack_starts_.pop_back();
    for(std::size_t k = i->var_alloc_stack_.size(); k-- > alloc_start;) { delete i->var_alloc_stack_[k]; }
    i->var_alloc_stack_.resize(alloc_start);

    i->memalloc_.recover_nested();
}

inline void recover_memory()
{
    AutodiffStackStorage* i = ChainableStack::instance_;
    if(!empty_nested()) { throw std::logic_error("recover_memory() called inside a nested tape"); }
    i->var_stack_.clear();
    i->var_nochain_stack_.clear();
    for(std::size_t k = i->var_alloc_stack_.size(); k-- > 0;) { delete i->var_alloc_stack_[k]; }
    i->var_alloc_stack_.clear();
    i->memalloc_.recover_all();
}

// ---- reverse sweep ----------------------------------------------------------

/**
 * Propagate adjoints through the innermost nested tape segment (or the whole
 * tape if no nested tape is active).  Index-based iteration is essential:
 * a chain() call may itself open a nested tape (checkpointing) and push/pop
 * nodes above the current position.
 */
inline void grad()
{
    AutodiffStackStorage* inst = ChainableStack::instance_;
    const std::size_t end = inst->var_stack_.size();
    const std::size_t beginning = empty_nested() ? 0 : end - nested_size();
    for(std::size_t i = end; i-- > beginning;) { inst->var_stack_[i]->chain(); }
}

/** Seed the adjoint of `vi` with one and run the reverse sweep. */
template <typename Vari>
inline void grad(Vari* vi)
{
    vi->init_dependent();
    grad();
}

inline void set_zero_all_adjoints()
{
    for(vari_base* v : ChainableStack::instance_->var_stack_) { v->set_zero_adjoint(); }
    for(vari_base* v : ChainableStack::instance_->var_nochain_stack_) { v->set_zero_adjoint(); }
}

inline void set_zero_all_adjoints_nested()
{
    if(empty_nested()) { throw std::logic_error("set_zero_all_adjoints_nested() called outside of a nested tape"); }
    AutodiffStackStorage* i = ChainableStack::instance_;
    for(std::size_t k = i->nested_var_stack_sizes_.back(); k < i->var_stack_.size(); ++k) { i->var_stack_[k]->set_zero_adjoint(); }
    for(std::size_t k = i->nested_var_nochain_stack_sizes_.back(); k < i->var_nochain_stack_.size(); ++k) {
        i->var_nochain_stack_[k]->set_zero_adjoint();
    }
}

/** Scoped nested tape.  Construction opens the nest, destruction unwinds it. */
class nested_rev_autodiff
{
public:
    nested_rev_autodiff() { start_nested(); }
    ~nested_rev_autodiff() { recover_nested(); }

    nested_rev_autodiff(const nested_rev_autodiff&) = delete;
    nested_rev_autodiff& operator=(const nested_rev_autodiff&) = delete;
    static void* operator new(std::size_t) = delete; // stack-scoped only

    void set_zero_all_adjoints() { set_zero_all_adjoints_nested(); }
};

inline void print_stack(std::ostream& o)
{
    AutodiffStackStorage* i = ChainableStack::instance_;
    o << "STACK, size=" << i->var_stack_.size() << std::endl;
    for(std::size_t k = 0; k < i->var_stack_.size(); ++k) { o << k << "  " << static_cast<const void*>(i->var_stack_[k]) << std::endl; }
}

// ---- scalar vari / var ------------------------------------------------------

template <typename T, typename Enable = void>
class vari_value;

template <typename T>
class vari_value<T, require_t<std::is_floating_point<T>>> : public vari_base
{
public:
    using value_type = T;

    const T val_;
    T adj_{0.0};

    template <typename S, require_convertible_t<S&, T>* = nullptr>
    vari_value(S x) noexcept // NOLINT
        : val_(x)
    {
        ChainableStack::instance_->var_stack_.push_back(this);
    }

    template <typename S, require_convertible_t<S&, T>* = nullptr>
    vari_value(S x, bool stacked) noexcept
        : val_(x)
    {
        if(stacked) {
            ChainableStack::instance_->var_stack_.push_back(this);
        } else {
            ChainableStack::instance_->var_nochain_stack_.push_back(this);
        }
    }

    inline const auto& val() const { return val_; }
    inline auto& adj() const { return adj_; }
    inline auto& adj() { return adj_; }

    void chain() override {}

    inline void init_dependent() noexcept { adj_ = 1.0; }
    inline void set_zero_adjoint() noexcept final { adj_ = 0.0; }

    friend std::ostream& operator<<(std::ostream& os, const vari_value<T>* v) { return os << v->val_ << ":" << v->adj_; }
};

using vari = vari_value<double>;

template <typename T, typename Enable = void>
class var_value;

template <typename T>
class var_value<T, require_t<std::is_floating_point<T>>>
{
public:
    using value_type = T;
    using vari_type = vari_value<T>;

    vari_type* vi_;

    inline bool is_uninitialized() const noexcept { return (vi_ == nullptr); }

    var_value()
        : vi_(nullptr)
    {}

    template <typename S, require_convertible_t<S&, T>* = nullptr>
    var_value(S x) // NOLINT
        : vi_(new vari_type(x, false))
    {}

    var_value(vari_type* vi) // NOLINT
        : vi_(vi)
    {}

    inline const auto& val() const noexcept { return vi_->val(); }
    inline auto& adj() const noexcept { return vi_->adj_; }
    inline auto& adj() noexcept { return vi_->adj_; }

    /** Reverse sweep seeded at this variable (does not recover memory). */
    void grad()
    {
        vi_->init_dependent();
        stan::math::grad();
    }

    inline vari_type& operator*() { return *vi_; }
    inline vari_type* operator->() { return vi_; }

    // bodies are dependent expressions, so the binary operators defined below
    // the class are found at instantiation time
    inline var_value<T>& operator+=(const var_value<T>& b)
    {
        *this = *this + b;
        return *this;
    }
    inline var_value<T>& operator+=(T b)
    {
        *this = *this + b;
        return *this;
    }
    inline var_value<T>& operator-=(const var_value<T>& b)
    {
        *this = *this - b;
        return *this;
    }
    inline var_value<T>& operator-=(T b)
    {
        *this = *this - b;
        return *this;
    }
    inline var_value<T>& operator*=(const var_value<T>& b)
    {
        *this = *this * b;
        return *this;
    }
    inline var_value<T>& operator*=(T b)
    {
        *this = *this * b;
        return *this;
    }
    inline var_value<T>& operator/=(const var_value<T>& b)
    {
        *this = *this / b;
        return *this;
    }
    inline var_value<T>& operator/=(T b)
    {
        *this = *this / b;
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const var_value<T>& v)
    {
        if(v.vi_ == nullptr) { return os << "uninitialized"; }
        return os << v.val();
    }
};

using var = var_value<double>;

// ---- callback nodes ---------------------------------------------------------

template <typename T, typename F>
struct callback_vari : public vari_value<T>
{
    F rev_functor_;

    explicit callback_vari(T value, F&& rev_functor)
        : vari_value<T>(std::move(value)) // chained: pushed onto var_stack_
        , rev_functor_(std::forward<F>(rev_functor))
    {}

    inline void chain() final { rev_functor_(*this); }
};

/**
 * Create a var whose reverse pass invokes `functor(vi)`, where `vi` is the
 * newly created vari (so the functor reads `vi.adj()`).
 * The functor must only capture trivially destructible handles (vars,
 * AD tensors); it is stored in the arena and never destructed.
 */
template <typename T, typename F>
inline auto make_callback_var(T value, F&& functor)
{
    using T_plain = std::decay_t<T>;
    return var_value<T_plain>(new callback_vari<T_plain, std::decay_t<F>>(std::move(value), std::forward<F>(functor)));
}

namespace internal {
template <typename F>
struct reverse_pass_callback_vari : public vari_base
{
    F rev_functor_;

    explicit reverse_pass_callback_vari(F&& rev_functor)
        : rev_functor_(std::forward<F>(rev_functor))
    {
        ChainableStack::instance_->var_stack_.push_back(this);
    }

    inline void chain() final { rev_functor_(); }
    inline void set_zero_adjoint() final {}
};
} // namespace internal

/**
 * Record a closure on the tape; it is invoked (without arguments) during the
 * reverse pass.  Captures must be trivially destructible -- for closures that
 * capture heap-owning objects use Xped::reverse_pass_callback_alloc.
 */
template <typename F>
inline void reverse_pass_callback(F&& functor)
{
    new internal::reverse_pass_callback_vari<std::decay_t<F>>(std::forward<F>(functor));
}

// ---- scalar arithmetic on var ----------------------------------------------

inline double value_of(double x) { return x; }
template <typename T>
inline T value_of(const var_value<T>& v)
{
    return v.val();
}

// addition
template <typename T>
inline var_value<T> operator+(const var_value<T>& a, const var_value<T>& b)
{
    return make_callback_var(a.val() + b.val(), [a, b](const auto& vi) mutable {
        a.adj() += vi.adj_;
        b.adj() += vi.adj_;
    });
}
template <typename T, typename Arith, require_t<std::is_arithmetic<Arith>>* = nullptr>
inline var_value<T> operator+(const var_value<T>& a, Arith b)
{
    return make_callback_var(a.val() + b, [a](const auto& vi) mutable { a.adj() += vi.adj_; });
}
template <typename T, typename Arith, require_t<std::is_arithmetic<Arith>>* = nullptr>
inline var_value<T> operator+(Arith a, const var_value<T>& b)
{
    return b + a;
}

// subtraction
template <typename T>
inline var_value<T> operator-(const var_value<T>& a, const var_value<T>& b)
{
    return make_callback_var(a.val() - b.val(), [a, b](const auto& vi) mutable {
        a.adj() += vi.adj_;
        b.adj() -= vi.adj_;
    });
}
template <typename T, typename Arith, require_t<std::is_arithmetic<Arith>>* = nullptr>
inline var_value<T> operator-(const var_value<T>& a, Arith b)
{
    return make_callback_var(a.val() - b, [a](const auto& vi) mutable { a.adj() += vi.adj_; });
}
template <typename T, typename Arith, require_t<std::is_arithmetic<Arith>>* = nullptr>
inline var_value<T> operator-(Arith a, const var_value<T>& b)
{
    return make_callback_var(a - b.val(), [b](const auto& vi) mutable { b.adj() -= vi.adj_; });
}
// unary minus
template <typename T>
inline var_value<T> operator-(const var_value<T>& a)
{
    return make_callback_var(-a.val(), [a](const auto& vi) mutable { a.adj() -= vi.adj_; });
}
// unary plus
template <typename T>
inline var_value<T> operator+(const var_value<T>& a)
{
    return a;
}

// multiplication
template <typename T>
inline var_value<T> operator*(const var_value<T>& a, const var_value<T>& b)
{
    return make_callback_var(a.val() * b.val(), [a, b](const auto& vi) mutable {
        a.adj() += vi.adj_ * b.val();
        b.adj() += vi.adj_ * a.val();
    });
}
template <typename T, typename Arith, require_t<std::is_arithmetic<Arith>>* = nullptr>
inline var_value<T> operator*(const var_value<T>& a, Arith b)
{
    return make_callback_var(a.val() * b, [a, b](const auto& vi) mutable { a.adj() += vi.adj_ * b; });
}
template <typename T, typename Arith, require_t<std::is_arithmetic<Arith>>* = nullptr>
inline var_value<T> operator*(Arith a, const var_value<T>& b)
{
    return b * a;
}

// division
template <typename T>
inline var_value<T> operator/(const var_value<T>& a, const var_value<T>& b)
{
    return make_callback_var(a.val() / b.val(), [a, b](const auto& vi) mutable {
        a.adj() += vi.adj_ / b.val();
        b.adj() -= vi.adj_ * a.val() / (b.val() * b.val());
    });
}
template <typename T, typename Arith, require_t<std::is_arithmetic<Arith>>* = nullptr>
inline var_value<T> operator/(const var_value<T>& a, Arith b)
{
    return make_callback_var(a.val() / b, [a, b](const auto& vi) mutable { a.adj() += vi.adj_ / b; });
}
template <typename T, typename Arith, require_t<std::is_arithmetic<Arith>>* = nullptr>
inline var_value<T> operator/(Arith a, const var_value<T>& b)
{
    return make_callback_var(a / b.val(), [a, b](const auto& vi) mutable { b.adj() -= vi.adj_ * a / (b.val() * b.val()); });
}

// comparisons (on values)
#define XPED_SELFAD_COMPARISON(op)                                                                                                                   \
    template <typename T>                                                                                                                            \
    inline bool operator op(const var_value<T>& a, const var_value<T>& b)                                                                            \
    {                                                                                                                                                \
        return a.val() op b.val();                                                                                                                   \
    }                                                                                                                                                \
    template <typename T, typename Arith, require_t<std::is_arithmetic<Arith>>* = nullptr>                                                           \
    inline bool operator op(const var_value<T>& a, Arith b)                                                                                          \
    {                                                                                                                                                \
        return a.val() op b;                                                                                                                         \
    }                                                                                                                                                \
    template <typename T, typename Arith, require_t<std::is_arithmetic<Arith>>* = nullptr>                                                           \
    inline bool operator op(Arith a, const var_value<T>& b)                                                                                          \
    {                                                                                                                                                \
        return a op b.val();                                                                                                                         \
    }

XPED_SELFAD_COMPARISON(==)
XPED_SELFAD_COMPARISON(!=)
XPED_SELFAD_COMPARISON(<)
XPED_SELFAD_COMPARISON(<=)
XPED_SELFAD_COMPARISON(>)
XPED_SELFAD_COMPARISON(>=)
#undef XPED_SELFAD_COMPARISON

// elementary functions
template <typename T>
inline var_value<T> sqrt(const var_value<T>& a)
{
    return make_callback_var(std::sqrt(a.val()), [a](const auto& vi) mutable { a.adj() += vi.adj_ / (2.0 * vi.val_); });
}
template <typename T>
inline var_value<T> exp(const var_value<T>& a)
{
    return make_callback_var(std::exp(a.val()), [a](const auto& vi) mutable { a.adj() += vi.adj_ * vi.val_; });
}
template <typename T>
inline var_value<T> log(const var_value<T>& a)
{
    return make_callback_var(std::log(a.val()), [a](const auto& vi) mutable { a.adj() += vi.adj_ / a.val(); });
}
template <typename T>
inline var_value<T> fabs(const var_value<T>& a)
{
    return make_callback_var(std::fabs(a.val()), [a](const auto& vi) mutable { a.adj() += (a.val() < 0 ? -vi.adj_ : vi.adj_); });
}
template <typename T>
inline var_value<T> abs(const var_value<T>& a)
{
    return fabs(a);
}
template <typename T, typename Arith, require_t<std::is_arithmetic<Arith>>* = nullptr>
inline var_value<T> pow(const var_value<T>& a, Arith b)
{
    return make_callback_var(std::pow(a.val(), b), [a, b](const auto& vi) mutable { a.adj() += vi.adj_ * b * std::pow(a.val(), b - 1); });
}
template <typename T>
inline var_value<T> pow(const var_value<T>& a, const var_value<T>& b)
{
    return make_callback_var(std::pow(a.val(), b.val()), [a, b](const auto& vi) mutable {
        a.adj() += vi.adj_ * b.val() * std::pow(a.val(), b.val() - 1);
        if(a.val() > 0) { b.adj() += vi.adj_ * vi.val_ * std::log(a.val()); }
    });
}
template <typename T, typename Arith, require_t<std::is_arithmetic<Arith>>* = nullptr>
inline var_value<T> pow(Arith a, const var_value<T>& b)
{
    return make_callback_var(std::pow(a, b.val()), [a, b](const auto& vi) mutable {
        if(a > 0) { b.adj() += vi.adj_ * vi.val_ * std::log(static_cast<double>(a)); }
    });
}
template <typename T>
inline var_value<T> sin(const var_value<T>& a)
{
    return make_callback_var(std::sin(a.val()), [a](const auto& vi) mutable { a.adj() += vi.adj_ * std::cos(a.val()); });
}
template <typename T>
inline var_value<T> cos(const var_value<T>& a)
{
    return make_callback_var(std::cos(a.val()), [a](const auto& vi) mutable { a.adj() -= vi.adj_ * std::sin(a.val()); });
}
template <typename T>
inline var_value<T> tan(const var_value<T>& a)
{
    return make_callback_var(std::tan(a.val()), [a](const auto& vi) mutable { a.adj() += vi.adj_ * (1.0 + vi.val_ * vi.val_); });
}
template <typename T>
inline var_value<T> tanh(const var_value<T>& a)
{
    return make_callback_var(std::tanh(a.val()), [a](const auto& vi) mutable { a.adj() += vi.adj_ * (1.0 - vi.val_ * vi.val_); });
}
template <typename T>
inline var_value<T> cosh(const var_value<T>& a)
{
    return make_callback_var(std::cosh(a.val()), [a](const auto& vi) mutable { a.adj() += vi.adj_ * std::sinh(a.val()); });
}
template <typename T>
inline var_value<T> sinh(const var_value<T>& a)
{
    return make_callback_var(std::sinh(a.val()), [a](const auto& vi) mutable { a.adj() += vi.adj_ * std::cosh(a.val()); });
}
template <typename T>
inline var_value<T> atan(const var_value<T>& a)
{
    return make_callback_var(std::atan(a.val()), [a](const auto& vi) mutable { a.adj() += vi.adj_ / (1.0 + a.val() * a.val()); });
}
template <typename T>
inline var_value<T> asin(const var_value<T>& a)
{
    return make_callback_var(std::asin(a.val()), [a](const auto& vi) mutable { a.adj() += vi.adj_ / std::sqrt(1.0 - a.val() * a.val()); });
}
template <typename T>
inline var_value<T> acos(const var_value<T>& a)
{
    return make_callback_var(std::acos(a.val()), [a](const auto& vi) mutable { a.adj() -= vi.adj_ / std::sqrt(1.0 - a.val() * a.val()); });
}
template <typename T>
inline bool isnan(const var_value<T>& a)
{
    return std::isnan(a.val());
}
template <typename T>
inline bool isfinite(const var_value<T>& a)
{
    return std::isfinite(a.val());
}

// ---- STL arena allocator ----------------------------------------------------

/**
 * std-compatible allocator drawing from the AD arena.  Deallocation is a no-op;
 * memory is reclaimed when the tape is recovered.
 */
template <typename T>
struct arena_allocator
{
    using value_type = T;

    arena_allocator() noexcept = default;
    template <typename U>
    arena_allocator(const arena_allocator<U>&) noexcept
    {}

    T* allocate(std::size_t n) { return ChainableStack::instance_->memalloc_.alloc_array<T>(n); }
    void deallocate(T* /*p*/, std::size_t /*n*/) noexcept {}

    template <typename U>
    bool operator==(const arena_allocator<U>&) const noexcept
    {
        return true;
    }
    template <typename U>
    bool operator!=(const arena_allocator<U>&) const noexcept
    {
        return false;
    }
};

// ---- finite differences (for gradient checks) -------------------------------

inline double finite_diff_stepsize(double u)
{
    static const double cbrt_epsilon = std::cbrt(2.2204460492503131e-16);
    return cbrt_epsilon * std::fmax(1.0, std::fabs(u));
}

/**
 * Sixth-order central finite-difference gradient of `f` at `x`.
 * `VectorT` needs `size()`, `resize(n)` and `operator()(i)` (Eigen vectors and
 * similar types work).  Matches the semantics of stan-math's
 * finite_diff_gradient_auto for real arguments.
 */
template <typename F, typename VectorT, typename ScalarT>
void finite_diff_gradient_auto(const F& f, const VectorT& x, ScalarT& fx, VectorT& grad_fx)
{
    VectorT x_temp(x);
    fx = f(x);
    grad_fx.resize(x.size());
    for(decltype(x.size()) i = 0; i < x.size(); ++i) {
        const double h = finite_diff_stepsize(static_cast<double>(x(i)));
        double delta_f = 0;
        x_temp(i) = x(i) + 3 * h;
        delta_f += static_cast<double>(f(x_temp));
        x_temp(i) = x(i) + 2 * h;
        delta_f -= 9 * static_cast<double>(f(x_temp));
        x_temp(i) = x(i) + h;
        delta_f += 45 * static_cast<double>(f(x_temp));
        x_temp(i) = x(i) - h;
        delta_f -= 45 * static_cast<double>(f(x_temp));
        x_temp(i) = x(i) - 2 * h;
        delta_f += 9 * static_cast<double>(f(x_temp));
        x_temp(i) = x(i) - 3 * h;
        delta_f -= static_cast<double>(f(x_temp));
        x_temp(i) = x(i);
        grad_fx(i) = delta_f / (60 * h);
    }
}

} // namespace math
} // namespace stan

#endif // XPED_SELFAD_CORE_HPP_
