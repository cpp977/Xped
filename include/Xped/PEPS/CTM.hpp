#ifndef XPED_CTM_H_
#define XPED_CTM_H_

#include <filesystem>
#include <memory>

#include "yas/serialize.hpp"
#include "yas/std_types.hpp"

#include "Xped/AD/ADTensor.hpp"
#include "Xped/Core/Tensor.hpp"
#include "Xped/PEPS/CTMOpts.hpp"
#include "Xped/PEPS/TMatrix.hpp"
#include "Xped/PEPS/UnitCell.hpp"
#include "Xped/PEPS/iPEPS.hpp"

namespace Xped {

template <typename, typename, bool>
struct OneSiteObservable;
template <typename, typename, bool>
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
template <typename Scalar_,
          typename Symmetry_,
          std::size_t TRank = 2,
          bool ALL_OUT_LEGS = false,
          bool ENABLE_AD = false,
          Opts::CTMCheckpoint CPOpts = Opts::CTMCheckpoint{}>
class CTM
{
    template <typename Scalar__,
              typename Symmetry__,
              std::size_t TRank__,
              bool ALL_OUT_LEGS__,
              bool ENABLE_AD__,
              Opts::CTMCheckpoint CPOpts__,
              typename OpScalar__,
              bool HERMITIAN__>
    friend TMatrix<std::conditional_t<ENABLE_AD__, stan::math::var, typename OneSiteObservable<OpScalar__, Symmetry__, HERMITIAN__>::ObsScalar>>
    avg(XPED_CONST CTM<Scalar__, Symmetry__, TRank__, ALL_OUT_LEGS__, ENABLE_AD__, CPOpts__>& env,
        OneSiteObservable<OpScalar__, Symmetry__, HERMITIAN__>& op);

    template <typename Scalar__,
              typename Symmetry__,
              std::size_t TRank__,
              bool ALL_OUT_LEGS__,
              bool ENABLE_AD__,
              Opts::CTMCheckpoint CPOpts__,
              typename OpScalar__,
              bool HERMITIAN__>
    friend std::array<
        TMatrix<std::conditional_t<ENABLE_AD__, stan::math::var, typename TwoSiteObservable<OpScalar__, Symmetry__, HERMITIAN__>::ObsScalar>>,
        4>
    avg(XPED_CONST CTM<Scalar__, Symmetry__, TRank__, ALL_OUT_LEGS__, ENABLE_AD__, CPOpts__>& env,
        TwoSiteObservable<OpScalar__, Symmetry__, HERMITIAN__>& op);

    template <typename, typename, std::size_t, bool, bool, Opts::CTMCheckpoint>
    friend class CTM;

public:
    typedef Symmetry_ Symmetry;
    typedef Scalar_ Scalar;

    template <std::size_t N>
    using cdims = std::array<int, N>;

    CTM() = default;

    explicit CTM(std::size_t chi, Opts::CTM_INIT init = Opts::CTM_INIT::FROM_A)
        : chi_(chi)
        , init_m(init)
    {}

    CTM(std::size_t chi, const UnitCell& cell, Opts::CTM_INIT init = Opts::CTM_INIT::FROM_A)
        : cell_(cell)
        , chi_(chi)
        , init_m(init)
    {}

    CTM(std::shared_ptr<iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>> A, std::size_t chi, const Opts::CTM_INIT init = Opts::CTM_INIT::FROM_A)
        : A(A)
        , cell_(A->cell())
        , chi_(chi)
        , init_m(init)
    {}

    // This is a copy constructor for non-AD CTM (CTM<Scalar, Symmetry, TRank, false>) so it prohibits implicitly declared move operations in this
    // case...
    CTM(const CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, false>& other);

    void set_A(std::shared_ptr<iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>> A_in)
    {
        A = A_in;
        cell_ = A_in->cell();
        if constexpr(TRank == 1) {
            Ms.resize(cell_.pattern);
            computeMs();
        }
    }

    std::size_t fullChi() const { return CA1s[0].uncoupledCodomain()[0].fullDim(); }

    template <bool TRACK = ENABLE_AD>
    void solve(std::size_t max_steps);

    template <bool TRACK = ENABLE_AD, bool CP = CPOpts.GROW_ALL>
    void grow_all();

    void init();
    void loadFromMatlab(const std::filesystem::path& p, const std::string& root_name, int qn_scale = 1);

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

    void checkHermiticity() const;

    auto info() const;

    const UnitCell& cell() const { return cell_; }
    const std::shared_ptr<iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>>& Psi() const { return A; }

    template <typename Ar>
    void serialize(Ar& ar)
    {
        ar& YAS_OBJECT_NVP("CTM",
                           ("cell", cell_),
                           ("chi", chi_),
                           ("HAS_RDM", HAS_RDM),
                           ("CA1s", CA1s),
                           ("CA2s", CA2s),
                           ("CA3s", CA3s),
                           ("CA4s", CA4s),
                           ("TA1s", TA1s),
                           ("TA2s", TA2s),
                           ("TA3s", TA3s),
                           ("TA4s", TA4s),
                           // ("Ms", Ms),
                           ("rho_h", rho_h),
                           ("rho_v", rho_v),
                           ("rho1_h", rho1_h),
                           ("rho1_v", rho1_v));
    }

    std::size_t chi() const { return chi_; }
    std::size_t updateChi(std::size_t chi) { return chi_ = chi; }

    Opts::CTM_INIT const init_mode() { return init_m; }

private:
    std::shared_ptr<iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>> A;
    UnitCell cell_;
    std::size_t chi_;
    Opts::CTM_INIT init_m = Opts::CTM_INIT::FROM_A;
    Opts::PROJECTION proj_m = Opts::PROJECTION::CORNER;
    bool HAS_RDM = false;

    static constexpr auto getRankCA1() { return ALL_OUT_LEGS ? 2 : 0; }
    static constexpr auto getRankCA2() { return ALL_OUT_LEGS ? 2 : 1; }
    static constexpr auto getRankCA3() { return ALL_OUT_LEGS ? 2 : 2; }
    static constexpr auto getRankCA4() { return ALL_OUT_LEGS ? 2 : 1; }

    static constexpr auto getCoRankCA1() { return ALL_OUT_LEGS ? 0 : 2; }
    static constexpr auto getCoRankCA2() { return ALL_OUT_LEGS ? 0 : 1; }
    static constexpr auto getCoRankCA3() { return ALL_OUT_LEGS ? 0 : 0; }
    static constexpr auto getCoRankCA4() { return ALL_OUT_LEGS ? 0 : 1; }

    static constexpr auto getRankTA1() { return ALL_OUT_LEGS ? TRank + 2 : 1; }
    static constexpr auto getRankTA2() { return ALL_OUT_LEGS ? TRank + 2 : TRank + 1; }
    static constexpr auto getRankTA3() { return ALL_OUT_LEGS ? TRank + 2 : TRank + 1; }
    static constexpr auto getRankTA4() { return ALL_OUT_LEGS ? TRank + 2 : 1; }

    static constexpr auto getCoRankTA1() { return ALL_OUT_LEGS ? 0 : TRank + 1; }
    static constexpr auto getCoRankTA2() { return ALL_OUT_LEGS ? 0 : 1; }
    static constexpr auto getCoRankTA3() { return ALL_OUT_LEGS ? 0 : 1; }
    static constexpr auto getCoRankTA4() { return ALL_OUT_LEGS ? 0 : TRank + 1; }

    TMatrix<Tensor<Scalar, getRankCA1(), getCoRankCA1(), Symmetry, ENABLE_AD>> CA1s;
    TMatrix<Tensor<Scalar, getRankCA2(), getCoRankCA2(), Symmetry, ENABLE_AD>> CA2s;
    TMatrix<Tensor<Scalar, getRankCA3(), getCoRankCA3(), Symmetry, ENABLE_AD>> CA3s;
    TMatrix<Tensor<Scalar, getRankCA4(), getCoRankCA4(), Symmetry, ENABLE_AD>> CA4s;

    TMatrix<Tensor<Scalar, getRankTA1(), getCoRankTA1(), Symmetry, ENABLE_AD>> TA1s;
    TMatrix<Tensor<Scalar, getRankTA2(), getCoRankTA2(), Symmetry, ENABLE_AD>> TA2s;
    TMatrix<Tensor<Scalar, getRankTA3(), getCoRankTA3(), Symmetry, ENABLE_AD>> TA3s;
    TMatrix<Tensor<Scalar, getRankTA4(), getCoRankTA4(), Symmetry, ENABLE_AD>> TA4s;

    TMatrix<Tensor<Scalar, 2, 2, Symmetry, ENABLE_AD>> Ms;

    TMatrix<Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>> Svs;

    TMatrix<Tensor<Scalar, 2, 2, Symmetry, ENABLE_AD>> rho_h;
    TMatrix<Tensor<Scalar, 2, 2, Symmetry, ENABLE_AD>> rho_v;
    TMatrix<Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>> rho1_h;
    TMatrix<Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>> rho1_v;
    TMatrix<double> rho_h_hermitian_check;
    TMatrix<double> rho_v_hermitian_check;
    TMatrix<double> rho1_h_hermitian_check;
    TMatrix<double> rho1_v_hermitian_check;
    // template <bool TRACK = ENABLE_AD>
    // std::pair<Tensor<Scalar, 3, 3, Symmetry, TRACK>, Tensor<Scalar, 1, 1, Symmetry, TRACK>> get_projectors_left();

    template <bool TRACK = ENABLE_AD>
    void computeMs();

    template <bool TRACK = ENABLE_AD, bool CP = CPOpts.RDM>
    void computeRDM_h();
    template <bool TRACK = ENABLE_AD, bool CP = CPOpts.RDM>
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
