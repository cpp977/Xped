#ifndef MATRIX_MULTIPLICATION_H_
#define MATRIX_MULTIPLICATION_H_

namespace internal {
/**Cost to multiply 2 matrices.*/
template <typename IndexTypeA, typename IndexTypeB>
static std::size_t mult_cost(const std::array<IndexTypeA, 2>& dimsA, const std::array<IndexTypeB, 2>& dimsB)
{
    return dimsA[0] * dimsA[1] * dimsB[1];
}

/**Cost to multiply 3 matrices in 2 possible ways.*/
template <typename IndexTypeA, typename IndexTypeB, typename IndexTypeC>
static std::vector<std::size_t>
mult_cost(const std::array<IndexTypeA, 2>& dimsA, const std::array<IndexTypeB, 2>& dimsB, const std::array<IndexTypeC, 2>& dimsC)
{
    std::vector<std::size_t> out(2);
    // (AB)C
    out[0] = mult_cost(dimsA, dimsB) + dimsA[0] * dimsC[0] * dimsC[1];

    // A(BC)
    out[1] = mult_cost(dimsB, dimsC) + dimsA[0] * dimsA[1] * dimsC[1];

    return out;
}

// /**Cost to multiply 4 matrices in 5 possible ways.*/
// template <typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC, typename MatrixTypeD>
// std::vector<size_t> mult_cost(const MatrixTypeA& A, const MatrixTypeB& B, const MatrixTypeC& C, const MatrixTypeD& D)
// {
//     std::vector<size_t> out(5);
//     // (AB)(CD)
//     out[0] = mult_cost(A, B) + mult_cost(C, D) + A.rows() * B.cols() * C.cols();

//     // ((AB)C)D
//     out[1] = mult_cost(A, B) + A.rows() * C.rows() * C.cols() + A.rows() * D.rows() * D.cols();

//     // (A(BC))D
//     out[2] = mult_cost(B, C) + A.rows() * A.cols() * C.cols() + A.rows() * D.rows() * D.cols();

//     // A((BC)D)
//     out[3] = mult_cost(B, C) + B.rows() * D.rows() * D.cols() + A.rows() * A.cols() * D.cols();

//     // A(B(CD))
//     out[4] = mult_cost(C, D) + B.rows() * B.cols() * D.cols() + A.rows() * A.cols() * D.cols();
//     return out;
// }
} // namespace internal
#endif
