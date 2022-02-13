#ifndef XPED_CTM_H_
#define XPED_CTM_H_

#include "Xped/Core/Tensor.hpp"
#include "Xped/PEPS/TMatrix.hpp"
#include "Xped/PEPS/UnitCell.hpp"
#include "Xped/PEPS/iPEPS.hpp"

namespace Xped {

/**
 * C1 -- T1 -- C2
 *  |    ||    |
 * T4 == A  == T2
 *  |    ||    |
 * C4 -- T3 -- C3
 */
template <typename Scalar_, typename Symmetry_>
class CTM
{
public:
    typedef Symmetry_ Symmetry;
    typedef Scalar_ Scalar;

    enum class DIRECTION
    {
        LEFT,
        RIGHT,
        TOP,
        BOTTOM
    };
    enum class CORNER
    {
        UPPER_LEFT,
        UPPER_RIGHT,
        LOWER_LEFT,
        LOWER_RIGHT
    };
    enum class PROJECTION
    {
        CORNER,
        HALF,
        FULL
    };
    enum class INIT
    {
        FROM_TRIVIAL,
        FROM_A
    };
    struct Options
    {
        std::size_t max_iter = 20;
    };

    CTM(const UnitCell& cell, std::size_t chi, const INIT init = INIT::FROM_TRIVIAL)
        : cell(cell)
        , chi(chi)
        , init_m(init)
    {}

    // void setAs(std::shared_ptr<const iPEPS<Symmetry, Scalar>>& As);
    void solve(XPED_CONST iPEPS<Scalar, Symmetry>& A);
    void init(const iPEPS<Scalar, Symmetry>& A);

    void info() const;

private:
    UnitCell cell;
    std::size_t chi;
    INIT init_m;
    PROJECTION proj_m = PROJECTION::CORNER;
    Options opts{};
    // std::shared_ptr<const iPEPS<Symmetry, Scalar>> As;

    TMatrix<Tensor<Scalar, 0, 2, Symmetry>> C1s;
    TMatrix<Tensor<Scalar, 1, 1, Symmetry>> C2s;
    TMatrix<Tensor<Scalar, 2, 0, Symmetry>> C3s;
    TMatrix<Tensor<Scalar, 1, 1, Symmetry>> C4s;
    TMatrix<Tensor<Scalar, 1, 3, Symmetry>> T1s;
    TMatrix<Tensor<Scalar, 3, 1, Symmetry>> T2s;
    TMatrix<Tensor<Scalar, 3, 1, Symmetry>> T3s;
    TMatrix<Tensor<Scalar, 1, 3, Symmetry>> T4s;

    TMatrix<Tensor<Scalar, 1, 1, Symmetry>> Svs;

    std::pair<Tensor<Scalar, 3, 3, Symmetry>, Tensor<Scalar, 1, 1, Symmetry>> get_projectors_left(const iPEPS<Scalar, Symmetry>& A);

    void left_move(XPED_CONST iPEPS<Scalar, Symmetry>& A);
    void right_move(XPED_CONST iPEPS<Scalar, Symmetry>& A);
    void top_move(XPED_CONST iPEPS<Scalar, Symmetry>& A);
    void bottom_move(XPED_CONST iPEPS<Scalar, Symmetry>& A);
    void symmetric_move(XPED_CONST iPEPS<Scalar, Symmetry>& A);

    Tensor<Scalar, 3, 3, Symmetry> contractCorner(const int x, const int y, XPED_CONST iPEPS<Scalar, Symmetry>& A, const CORNER corner) XPED_CONST;

    std::pair<Tensor<Scalar, 1, 3, Symmetry>, Tensor<Scalar, 3, 1, Symmetry>>
    get_projectors(const int x, const int y, XPED_CONST iPEPS<Scalar, Symmetry>& A, const DIRECTION dir) XPED_CONST;

    std::tuple<Tensor<Scalar, 0, 2, Symmetry>, Tensor<Scalar, 1, 3, Symmetry>, Tensor<Scalar, 1, 1, Symmetry>>
    renormalize_left(const int x,
                     const int y,
                     XPED_CONST iPEPS<Scalar, Symmetry>& A,
                     XPED_CONST TMatrix<Tensor<Scalar, 1, 3, Symmetry>>& P1,
                     XPED_CONST TMatrix<Tensor<Scalar, 3, 1, Symmetry>>& P2) XPED_CONST;

    std::tuple<Tensor<Scalar, 1, 1, Symmetry>, Tensor<Scalar, 3, 1, Symmetry>, Tensor<Scalar, 2, 0, Symmetry>>
    renormalize_right(const int x,
                      const int y,
                      XPED_CONST iPEPS<Scalar, Symmetry>& A,
                      XPED_CONST TMatrix<Tensor<Scalar, 1, 3, Symmetry>>& P1,
                      XPED_CONST TMatrix<Tensor<Scalar, 3, 1, Symmetry>>& P2) XPED_CONST;

    std::tuple<Tensor<Scalar, 0, 2, Symmetry>, Tensor<Scalar, 1, 3, Symmetry>, Tensor<Scalar, 1, 1, Symmetry>>
    renormalize_top(const int x,
                    const int y,
                    XPED_CONST iPEPS<Scalar, Symmetry>& A,
                    XPED_CONST TMatrix<Tensor<Scalar, 1, 3, Symmetry>>& P1,
                    XPED_CONST TMatrix<Tensor<Scalar, 3, 1, Symmetry>>& P2) XPED_CONST;

    std::tuple<Tensor<Scalar, 1, 1, Symmetry>, Tensor<Scalar, 3, 1, Symmetry>, Tensor<Scalar, 2, 0, Symmetry>>
    renormalize_bottom(const int x,
                       const int y,
                       XPED_CONST iPEPS<Scalar, Symmetry>& A,
                       XPED_CONST TMatrix<Tensor<Scalar, 1, 3, Symmetry>>& P1,
                       XPED_CONST TMatrix<Tensor<Scalar, 3, 1, Symmetry>>& P2) XPED_CONST;
};

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "PEPS/CTM.cpp"
#endif

#endif
