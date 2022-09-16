#ifndef XPED_CTM_H_
#define XPED_CTM_H_

#include <memory>

#include "Xped/Core/Tensor.hpp"
#include "Xped/PEPS/CTMOpts.hpp"
#include "Xped/PEPS/LinearAlgebra.hpp"
#include "Xped/PEPS/TMatrix.hpp"
#include "Xped/PEPS/UnitCell.hpp"
#include "Xped/PEPS/iPEPS.hpp"

namespace Xped {

template <typename Symmetry>
struct OneSiteObservable;
template <typename Symmetry>
struct TwoSiteObservable;

/**
 * C1 -- T1 -- C2
 *  |    ||    |
 * T4 == A  == T2
 *  |    ||    |
 * C4 -- T3 -- C3
 *
 * C1 --> 1
 * |
 * ▼
 * 0
 *
 * 0 --> C2
 *       |
 *       ▽
 *       1
 *
 *       0
 *       |
 *       ▽
 * 1 --> C3
 *
 * 0
 * |
 * ▽
 * C4 --> 1

 */

/**
 * Checkpoint move (l,r,t,b).
 * Checkpoint computeRDM_h/v.
 * Checkpoint contractCorner.
 * Checkpoint get_projectors.
 * Checkpoint renormalize (l,r,t,b)
 */
template <typename Scalar_, typename Symmetry_, std::size_t TRank = 2, bool ENABLE_AD = false, Opts::CTMCheckpoint CPOpts = Opts::CTMCheckpoint{}>
class CTM
{
    template <typename Scalar__, typename Symmetry__, std::size_t TRank__, bool ENABLE_AD__, Opts::CTMCheckpoint CPOpts__>
    friend std::pair<TMatrix<std::conditional_t<ENABLE_AD__, stan::math::var, Scalar__>>,
                     TMatrix<std::conditional_t<ENABLE_AD__, stan::math::var, Scalar__>>>
    avg(XPED_CONST CTM<Scalar__, Symmetry__, TRank__, ENABLE_AD__, CPOpts__>& env, XPED_CONST Tensor<Scalar__, 2, 2, Symmetry__, false>& op);

    template <typename Scalar__, typename Symmetry__, std::size_t TRank__, bool ENABLE_AD__, Opts::CTMCheckpoint CPOpts__>
    friend TMatrix<std::conditional_t<ENABLE_AD__, stan::math::var, Scalar__>>
    avg(XPED_CONST CTM<Scalar__, Symmetry__, TRank__, ENABLE_AD__, CPOpts__>& env, OneSiteObservable<Symmetry__>& op);

    template <typename Scalar__, typename Symmetry__, std::size_t TRank__, bool ENABLE_AD__, Opts::CTMCheckpoint CPOpts__>
    friend std::array<TMatrix<std::conditional_t<ENABLE_AD__, stan::math::var, Scalar__>>, 4>
    avg(XPED_CONST CTM<Scalar__, Symmetry__, TRank__, ENABLE_AD__, CPOpts__>& env, TwoSiteObservable<Symmetry__>& op);

    template <typename, typename, std::size_t, bool, Opts::CTMCheckpoint>
    friend class CTM;

public:
    typedef Symmetry_ Symmetry;
    typedef Scalar_ Scalar;

    template <std::size_t N>
    using cdims = std::array<int, N>;

    CTM() = default;

    explicit CTM(std::size_t chi, Opts::CTM_INIT init = Opts::CTM_INIT::FROM_A)
        : chi(chi)
        , init_m(init)
    {}

    CTM(std::size_t chi, const UnitCell& cell, Opts::CTM_INIT init = Opts::CTM_INIT::FROM_A)
        : cell_(cell)
        , chi(chi)
        , init_m(init)
    {}

    CTM(std::shared_ptr<iPEPS<Scalar, Symmetry, ENABLE_AD>> A, std::size_t chi, const Opts::CTM_INIT init = Opts::CTM_INIT::FROM_A)
        : A(A)
        , cell_(A->cell())
        , chi(chi)
        , init_m(init)
    {}

    CTM(const CTM<Scalar, Symmetry, TRank, false>& other);

    void set_A(std::shared_ptr<iPEPS<Scalar, Symmetry, ENABLE_AD>> A_in)
    {
        A = A_in;
        cell_ = A_in->cell();
    }

    template <bool TRACK = ENABLE_AD>
    void solve(std::size_t max_steps);

    template <bool TRACK = ENABLE_AD, bool CP = CPOpts.GROW_ALL>
    void grow_all();

    void init();

    template <bool TRACK = ENABLE_AD, bool CP = CPOpts.MOVE>
    void left_move();
    template <bool TRACK = ENABLE_AD, bool CP = CPOpts.MOVE>
    void right_move();
    template <bool TRACK = ENABLE_AD, bool CP = CPOpts.MOVE>
    void top_move();
    template <bool TRACK = ENABLE_AD, bool CP = CPOpts.MOVE>
    void bottom_move();
    template <bool TRACK = ENABLE_AD, bool CP = CPOpts.MOVE>
    void symmetric_move();

    template <bool TRACK = ENABLE_AD>
    void computeRDM();
    bool RDM_COMPUTED() const { return HAS_RDM; }

    void info() const;

    const UnitCell& cell() const { return cell_; }
    const std::shared_ptr<iPEPS<Scalar, Symmetry, ENABLE_AD>>& Psi() const { return A; }

private:
    std::shared_ptr<iPEPS<Scalar, Symmetry, ENABLE_AD>> A;
    UnitCell cell_;
    std::size_t chi;
    Opts::CTM_INIT init_m = Opts::CTM_INIT::FROM_A;
    Opts::PROJECTION proj_m = Opts::PROJECTION::CORNER;
    bool HAS_RDM = false;

    TMatrix<Tensor<Scalar, 0, 2, Symmetry, ENABLE_AD>> C1s;
    TMatrix<Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>> C2s;
    TMatrix<Tensor<Scalar, 2, 0, Symmetry, ENABLE_AD>> C3s;
    TMatrix<Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>> C4s;
    TMatrix<Tensor<Scalar, 1, TRank + 1, Symmetry, ENABLE_AD>> T1s;
    TMatrix<Tensor<Scalar, TRank + 1, 1, Symmetry, ENABLE_AD>> T2s;
    TMatrix<Tensor<Scalar, TRank + 1, 1, Symmetry, ENABLE_AD>> T3s;
    TMatrix<Tensor<Scalar, 1, TRank + 1, Symmetry, ENABLE_AD>> T4s;

    TMatrix<Tensor<Scalar, 2, 2, Symmetry, ENABLE_AD>> Ms;

    TMatrix<Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>> Svs;

    TMatrix<Tensor<Scalar, 2, 2, Symmetry, ENABLE_AD>> rho_h;
    TMatrix<Tensor<Scalar, 2, 2, Symmetry, ENABLE_AD>> rho_v;
    TMatrix<Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>> rho1_h;
    TMatrix<Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>> rho1_v;

    // template <bool TRACK = ENABLE_AD>
    // std::pair<Tensor<Scalar, 3, 3, Symmetry, TRACK>, Tensor<Scalar, 1, 1, Symmetry, TRACK>> get_projectors_left();

    template <bool TRACK = ENABLE_AD>
    void computeMs();

    template <bool TRACK = ENABLE_AD>
    void computeRDM_h();
    template <bool TRACK = ENABLE_AD>
    void computeRDM_v();

    template <bool TRACK = ENABLE_AD, bool CP = CPOpts.CORNER>
    Tensor<Scalar, TRank + 1, TRank + 1, Symmetry, TRACK> contractCorner(const int x, const int y, const Opts::CORNER corner) XPED_CONST;

    template <bool TRACK = ENABLE_AD, bool CP = CPOpts.PROJECTORS>
    std::pair<Tensor<Scalar, 1, TRank + 1, Symmetry, TRACK>, Tensor<Scalar, TRank + 1, 1, Symmetry, TRACK>>
    get_projectors(const int x, const int y, const Opts::DIRECTION dir) XPED_CONST;

    template <bool TRACK = ENABLE_AD, bool CP = CPOpts.RENORMALIZE>
    std::tuple<Tensor<Scalar, 0, 2, Symmetry, TRACK>, Tensor<Scalar, 1, TRank + 1, Symmetry, TRACK>, Tensor<Scalar, 1, 1, Symmetry, TRACK>>
    renormalize_left(const int x,
                     const int y,
                     XPED_CONST TMatrix<Tensor<Scalar, 1, TRank + 1, Symmetry, TRACK>>& P1,
                     XPED_CONST TMatrix<Tensor<Scalar, TRank + 1, 1, Symmetry, TRACK>>& P2,
                     bool NORMALIZE = true) XPED_CONST;

    template <bool TRACK = ENABLE_AD, bool CP = CPOpts.RENORMALIZE>
    std::tuple<Tensor<Scalar, 1, 1, Symmetry, TRACK>, Tensor<Scalar, TRank + 1, 1, Symmetry, TRACK>, Tensor<Scalar, 2, 0, Symmetry, TRACK>>
    renormalize_right(const int x,
                      const int y,
                      XPED_CONST TMatrix<Tensor<Scalar, 1, TRank + 1, Symmetry, TRACK>>& P1,
                      XPED_CONST TMatrix<Tensor<Scalar, TRank + 1, 1, Symmetry, TRACK>>& P2,
                      bool NORMALIZE = true) XPED_CONST;

    template <bool TRACK = ENABLE_AD, bool CP = CPOpts.RENORMALIZE>
    std::tuple<Tensor<Scalar, 0, 2, Symmetry, TRACK>, Tensor<Scalar, 1, TRank + 1, Symmetry, TRACK>, Tensor<Scalar, 1, 1, Symmetry, TRACK>>
    renormalize_top(const int x,
                    const int y,
                    XPED_CONST TMatrix<Tensor<Scalar, 1, TRank + 1, Symmetry, TRACK>>& P1,
                    XPED_CONST TMatrix<Tensor<Scalar, TRank + 1, 1, Symmetry, TRACK>>& P2,
                    bool NORMALIZE = true) XPED_CONST;

    template <bool TRACK = ENABLE_AD, bool CP = CPOpts.RENORMALIZE>
    std::tuple<Tensor<Scalar, 1, 1, Symmetry, TRACK>, Tensor<Scalar, TRank + 1, 1, Symmetry, TRACK>, Tensor<Scalar, 2, 0, Symmetry, TRACK>>
    renormalize_bottom(const int x,
                       const int y,
                       XPED_CONST TMatrix<Tensor<Scalar, 1, TRank + 1, Symmetry, TRACK>>& P1,
                       XPED_CONST TMatrix<Tensor<Scalar, TRank + 1, 1, Symmetry, TRACK>>& P2,
                       bool NORMALIZE = true) XPED_CONST;

    bool checkConvergence(typename ScalarTraits<Scalar>::Real epsilon);
};

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "PEPS/CTM.cpp"
#endif

#endif
