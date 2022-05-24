#ifndef XPED_IPEPS_H_
#define XPED_IPEPS_H_

#include "Xped/Core/Qbasis.hpp"
#include "Xped/Core/Tensor.hpp"
#include "Xped/PEPS/TMatrix.hpp"
#include "Xped/PEPS/UnitCell.hpp"
#include "Xped/PEPS/iPEPSIterator.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
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
    friend class CTM<Scalar_, Symmetry_, ENABLE_AD_>;
    friend class iPEPS<Scalar_, Symmetry_, true>;

public:
    typedef Symmetry_ Symmetry;
    typedef Scalar_ Scalar;
    static constexpr bool ENABLE_AD = ENABLE_AD_;
    typedef typename ScalarTraits<Scalar>::Real RealScalar;
    typedef typename Symmetry::qType qType;

    enum class LEG
    {
        LEFT,
        UP,
        RIGHT,
        DOWN,
        PHYS
    };

    iPEPS(const UnitCell& cell, const Qbasis<Symmetry, 1>& auxBasis, const Qbasis<Symmetry, 1>& physBasis);

    iPEPS(const iPEPS<Scalar, Symmetry, false>& other);

    void setRandom();
    void setZero();

    void set_As(const std::vector<Tensor<Scalar, 2, 3, Symmetry, ENABLE_AD>>& As_in)
    {
        As.fill(As_in);
        for(auto i = 0ul; i < As.size(); ++i) { Adags[i] = As[i].adjoint().eval().template permute<0, 3, 4, 2, 0, 1>(); }
    }

    Qbasis<Symmetry, 1> ketBasis(const int x, const int y, const LEG leg) const;
    Qbasis<Symmetry, 1> braBasis(const int x, const int y, const LEG leg) const;
    void info() const;

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

private:
    std::size_t D;

    UnitCell cell_;
    TMatrix<Tensor<Scalar, 2, 3, Symmetry, ENABLE_AD>> As;
    TMatrix<Tensor<Scalar, 3, 2, Symmetry, ENABLE_AD>> Adags;
};

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "PEPS/iPEPS.cpp"
#endif

#endif
