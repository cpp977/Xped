#ifndef XPED_IPEPS_H_
#define XPED_IPEPS_H_

#include "Xped/Core/Qbasis.hpp"
#include "Xped/Core/Tensor.hpp"
#include "Xped/PEPS/TMatrix.hpp"
#include "Xped/PEPS/UnitCell.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry>
class CTM;
/**                   p(2)
 *             u(1)   /
 *              |    /
 *              |   v
 *              v  /
 *           □□□□□/□
 * l(0) ---> □   / □ <--- r(3)
 *           □□□□□□□
 *              |
 *              |
 *              v
 *             d(4)
 */
template <typename Scalar_, typename Symmetry_>
class iPEPS
{
    friend class CTM<Scalar_, Symmetry_>;

public:
    typedef Symmetry_ Symmetry;
    typedef Scalar_ Scalar;
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

    Qbasis<Symmetry, 1> ketBasis(const int x, const int y, const LEG leg) const;
    Qbasis<Symmetry, 1> braBasis(const int x, const int y, const LEG leg) const;
    void info() const;

private:
    std::size_t D;

    UnitCell cell;
    TMatrix<Tensor<Scalar, 2, 3, Symmetry>> As;
    TMatrix<Tensor<Scalar, 3, 2, Symmetry>> Adags;
};

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "PEPS/iPEPS.cpp"
#endif

#endif
