#ifndef XPED_IPEPS_H_
#define XPED_IPEPS_H_

#include "yas/serialize.hpp"
#include "yas/std_types.hpp"

#include "Xped/Core/Qbasis.hpp"
#include "Xped/Core/Tensor.hpp"
#include "Xped/PEPS/CTMOpts.hpp"
#include "Xped/PEPS/TMatrix.hpp"
#include "Xped/PEPS/UnitCell.hpp"
#include "Xped/PEPS/iPEPSIterator.hpp"
#include "Xped/PEPS/iPEPSOpts.hpp"

namespace Xped {

template <typename, typename, bool>
struct OneSiteObservable;
template <typename, typename, bool>
struct TwoSiteObservable;

template <typename, typename, std::size_t, bool, bool, Opts::CTMCheckpoint>
class CTM;

/**                   p(4)
 *             u(1)   /
 *              |    /
 *              |   ^
 *              v  /
 *           □□□□□/□
 * l(0) ---> □   / □ -->- r(2)
 *           □□□□□□□
 *              |
 *              |
 *              v
 *             d(3)
 */
template <typename Scalar_, typename Symmetry_, bool ALL_OUT_LEGS_ = false, bool ENABLE_AD_ = false>
class iPEPS
{
    static constexpr auto getRankA() { return ALL_OUT_LEGS_ ? 4ul : 2ul; }
    static constexpr auto getRankB() { return ALL_OUT_LEGS_ ? 0ul : 2ul; }
    static constexpr auto getCoRankA() { return ALL_OUT_LEGS_ ? 1ul : 3ul; }
    static constexpr auto getCoRankB() { return ALL_OUT_LEGS_ ? 5ul : 3ul; }

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

    template <typename, typename, typename, typename>
    friend class TimePropagator;

    friend class iPEPS<Scalar_, Symmetry_, ALL_OUT_LEGS_, true>;

public:
    typedef Symmetry_ Symmetry;
    typedef Scalar_ Scalar;
    static constexpr bool ENABLE_AD = ENABLE_AD_;
    static constexpr bool ALL_OUT_LEGS = ALL_OUT_LEGS_;
    typedef typename ScalarTraits<Scalar>::Real RealScalar;
    typedef typename Symmetry::qType qType;

    iPEPS() = default;

    iPEPS(const UnitCell& cell,
          std::size_t D,
          const Qbasis<Symmetry, 1>& auxBasis,
          const Qbasis<Symmetry, 1>& physBasis,
          Opts::DiscreteSym sym = Opts::DiscreteSym::None);

    iPEPS(const UnitCell& cell,
          std::size_t D,
          const TMatrix<Qbasis<Symmetry, 1>>& leftBasis,
          const TMatrix<Qbasis<Symmetry, 1>>& topBasis,
          const TMatrix<Qbasis<Symmetry, 1>>& physBasis,
          Opts::DiscreteSym sym = Opts::DiscreteSym::None);

    iPEPS(const UnitCell& cell,
          std::size_t D,
          const TMatrix<Qbasis<Symmetry, 1>>& leftBasis,
          const TMatrix<Qbasis<Symmetry, 1>>& topBasis,
          const TMatrix<Qbasis<Symmetry, 1>>& physBasis,
          const TMatrix<qType>& charges,
          Opts::DiscreteSym sym = Opts::DiscreteSym::None);

    iPEPS(const iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, false>& other);

    void setRandom(std::size_t seed = 0ul);
    void setZero();
    void normalize();

    // void set_As(const std::vector<Tensor<Scalar, getRankA(), getCoRankA(), Symmetry, ENABLE_AD>>& As_in)
    // {
    //     As.fill(As_in);
    //     for(auto i = 0ul; i < As.size(); ++i) { Adags[i] = As[i].adjoint().eval().template permute<0, 3, 4, 2, 0, 1>(); }
    // }

    Qbasis<Symmetry, 1> ketBasis(const int x, const int y, const Opts::Leg leg) const;
    Qbasis<Symmetry, 1> braBasis(const int x, const int y, const Opts::Leg leg) const;

    std::string info() const;
    void debug_info() const;

    std::vector<Scalar> data();

    std::vector<Scalar> graddata();

    void set_data(const Scalar* data, bool NORMALIZE = true);

    std::size_t plainSize() const;

    auto beginA()
    {
        iPEPSIterator<getRankA(), getCoRankA(), Scalar, Symmetry, ENABLE_AD> out(&As, /*ITER_GRAD=*/false);
        return out;
    }
    auto endA()
    {
        iPEPSIterator<getRankA(), getCoRankA(), Scalar, Symmetry, ENABLE_AD> out(&As, /*ITER_GRAD=*/false, As.size());
        return out;
    }

    auto gradbeginA()
    {
        iPEPSIterator<getRankA(), getCoRankA(), Scalar, Symmetry, ENABLE_AD> out(&As, /*ITER_GRAD=*/true);
        return out;
    }
    auto gradendA()
    {
        iPEPSIterator<getRankA(), getCoRankA(), Scalar, Symmetry, ENABLE_AD> out(&As, /*ITER_GRAD=*/true, As.size());
        return out;
    }

    auto beginB()
    {
        iPEPSIterator<getRankB(), getCoRankB(), Scalar, Symmetry, ENABLE_AD> out(&Bs, /*ITER_GRAD=*/false);
        return out;
    }
    auto endB()
    {
        iPEPSIterator<getRankB(), getCoRankB(), Scalar, Symmetry, ENABLE_AD> out(&Bs, /*ITER_GRAD=*/false, Bs.size());
        return out;
    }

    auto gradbeginB()
    {
        iPEPSIterator<getRankB(), getCoRankB(), Scalar, Symmetry, ENABLE_AD> out(&Bs, /*ITER_GRAD=*/true);
        return out;
    }
    auto gradendB()
    {
        iPEPSIterator<getRankB(), getCoRankB(), Scalar, Symmetry, ENABLE_AD> out(&Bs, /*ITER_GRAD=*/true, Bs.size());
        return out;
    }

    const UnitCell& cell() const { return cell_; }

    const TMatrix<qType>& charges() const { return charges_; }

    Opts::DiscreteSym sym() const { return sym_; }

    template <typename Ar>
    void serialize(Ar& ar)
    {
        ar& YAS_OBJECT_NVP("iPEPS", ("D", D), ("cell", cell_), ("As", As), ("Adags", Adags), ("Bs", Bs), ("Bdags", Bdags), ("charges", charges_));
    }

    void loadFromMatlab(const std::filesystem::path& p, const std::string& root_name, int qn_scale = 1);

    void loadFromJson(const std::filesystem::path& p);

    bool checkConsistency() const;
    bool checkSym() const;

    void computeMs();

    void initWeightTensors();
    void updateAtensors();

    Tensor<Scalar, 1, 1, Symmetry> Id_weight_h(int x, int y) const;
    Tensor<Scalar, 1, 1, Symmetry> Id_weight_v(int x, int y) const;

    std::tuple<std::size_t, std::size_t, double, double> calc_Ds() const;

    std::size_t D;

    // private:
    void
    init(const TMatrix<Qbasis<Symmetry, 1>>& leftBasis, const TMatrix<Qbasis<Symmetry, 1>>& topBasis, const TMatrix<Qbasis<Symmetry, 1>>& physBasis);

    void initSymMap();

    void updateAdags();

    UnitCell cell_;
    TMatrix<Tensor<Scalar, getRankA(), getCoRankA(), Symmetry, ENABLE_AD>> As;
    TMatrix<Tensor<Scalar, getRankB(), getCoRankB(), Symmetry, ENABLE_AD>> Bs;
    TMatrix<Tensor<Scalar, getRankA(), getCoRankA(), Symmetry, ENABLE_AD>> Ms;
    TMatrix<Tensor<Scalar, getRankB(), getCoRankB(), Symmetry, ENABLE_AD>> Ns;
    TMatrix<Tensor<Scalar, getRankA(), getCoRankA(), Symmetry, ENABLE_AD>> Gs;
    TMatrix<Tensor<Scalar, getRankB(), getCoRankB(), Symmetry, ENABLE_AD>> Hs;
    TMatrix<Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>> whs;
    TMatrix<Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>> wvs;

    TMatrix<Tensor<Scalar, getRankA() + 1ul, getCoRankA() - 1ul, Symmetry, ENABLE_AD>> Adags;
    TMatrix<Tensor<Scalar, getRankB() + 1ul, getCoRankB() - 1ul, Symmetry, ENABLE_AD>> Bdags;
    TMatrix<qType> charges_;

    Opts::DiscreteSym sym_ = Opts::DiscreteSym::None;

    std::pair<std::size_t, std::vector<std::size_t>> sym_map_A;
    std::pair<std::size_t, std::vector<std::size_t>> sym_map_B;
};

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "PEPS/iPEPS.cpp"
#endif

#endif
