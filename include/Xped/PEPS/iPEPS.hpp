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

template <typename Symmetry>
struct OneSiteObservable;
template <typename Symmetry>
struct TwoSiteObservable;

template <typename, typename, std::size_t, bool, Opts::CTMCheckpoint>
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
template <typename Scalar_, typename Symmetry_, bool ENABLE_AD_ = false>
class iPEPS
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

    template <typename, typename, typename>
    friend class TimePropagator;

    friend class iPEPS<Scalar_, Symmetry_, true>;

public:
    typedef Symmetry_ Symmetry;
    typedef Scalar_ Scalar;
    static constexpr bool ENABLE_AD = ENABLE_AD_;
    typedef typename ScalarTraits<Scalar>::Real RealScalar;
    typedef typename Symmetry::qType qType;

    iPEPS() = default;

    iPEPS(const UnitCell& cell, std::size_t D, const Qbasis<Symmetry, 1>& auxBasis, const Qbasis<Symmetry, 1>& physBasis);

    iPEPS(const UnitCell& cell,
          std::size_t D,
          const TMatrix<Qbasis<Symmetry, 1>>& leftBasis,
          const TMatrix<Qbasis<Symmetry, 1>>& topBasis,
          const TMatrix<Qbasis<Symmetry, 1>>& physBasis);

    iPEPS(const UnitCell& cell,
          std::size_t D,
          const TMatrix<Qbasis<Symmetry, 1>>& leftBasis,
          const TMatrix<Qbasis<Symmetry, 1>>& topBasis,
          const TMatrix<Qbasis<Symmetry, 1>>& physBasis,
          const TMatrix<qType>& charges);

    iPEPS(const iPEPS<Scalar, Symmetry, false>& other);

    void setRandom(std::size_t seed = 0ul);
    void setZero();

    void set_As(const std::vector<Tensor<Scalar, 2, 3, Symmetry, ENABLE_AD>>& As_in)
    {
        As.fill(As_in);
        for(auto i = 0ul; i < As.size(); ++i) { Adags[i] = As[i].adjoint().eval().template permute<0, 3, 4, 2, 0, 1>(); }
    }

    Qbasis<Symmetry, 1> ketBasis(const int x, const int y, const Opts::LEG leg) const;
    Qbasis<Symmetry, 1> braBasis(const int x, const int y, const Opts::LEG leg) const;

    std::string info() const;
    void debug_info() const;

    std::vector<Scalar> data();

    void set_data(const Scalar* data, bool NORMALIZE = true);

    std::size_t plainSize() const;

    iPEPSIterator<Scalar, Symmetry, ENABLE_AD> begin()
    {
        iPEPSIterator<Scalar, Symmetry, ENABLE_AD> out(&As, /*ITER_GRAD=*/false);
        return out;
    }
    iPEPSIterator<Scalar, Symmetry, ENABLE_AD> end()
    {
        iPEPSIterator<Scalar, Symmetry, ENABLE_AD> out(&As, /*ITER_GRAD=*/false, As.size());
        return out;
    }

    iPEPSIterator<Scalar, Symmetry, ENABLE_AD> gradbegin()
    {
        iPEPSIterator<Scalar, Symmetry, ENABLE_AD> out(&As, /*ITER_GRAD=*/true);
        return out;
    }
    iPEPSIterator<Scalar, Symmetry, ENABLE_AD> gradend()
    {
        iPEPSIterator<Scalar, Symmetry, ENABLE_AD> out(&As, /*ITER_GRAD=*/true, As.size());
        return out;
    }

    const UnitCell& cell() const { return cell_; }

    const TMatrix<qType>& charges() const { return charges_; }

    template <typename Ar>
    void serialize(Ar& ar)
    {
        ar& YAS_OBJECT_NVP("iPEPS", ("D", D), ("cell", cell_), ("As", As), ("Adags", Adags), ("charges", charges_));
    }

    void loadFromMatlab(const std::filesystem::path& p, const std::string& root_name, int qn_scale = 1);

    bool checkConsistency() const;

    void initWeightTensors();
    void updateAtensors();

    Tensor<Scalar, 1, 1, Symmetry> Id_weight_h(int x, int y) const;
    Tensor<Scalar, 1, 1, Symmetry> Id_weight_v(int x, int y) const;

    std::tuple<std::size_t, std::size_t, double, double> calc_Ds() const;

    std::size_t D;

private:
    void
    init(const TMatrix<Qbasis<Symmetry, 1>>& leftBasis, const TMatrix<Qbasis<Symmetry, 1>>& topBasis, const TMatrix<Qbasis<Symmetry, 1>>& physBasis);

    UnitCell cell_;
    TMatrix<Tensor<Scalar, 2, 3, Symmetry, ENABLE_AD>> As;
    TMatrix<Tensor<Scalar, 2, 3, Symmetry, ENABLE_AD>> Gs;
    TMatrix<Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>> whs;
    TMatrix<Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>> wvs;

    TMatrix<Tensor<Scalar, 3, 2, Symmetry, ENABLE_AD>> Adags;
    TMatrix<qType> charges_;
};

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "PEPS/iPEPS.cpp"
#endif

#endif
