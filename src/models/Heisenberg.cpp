#include "Heisenberg.h"

namespace Models {

typedef double Scalar;
typedef Eigen::SparseMatrix<Scalar> MatrixType;

MatrixType Heisenberg::Sz(const std::size_t& site) const
{
    std::vector<Eigen::Triplet<Scalar>> tripletList;
    for(size_t i = 0; i < basis_.dim(); i++) { tripletList.push_back(Eigen::Triplet<Scalar>(i, i, basis_.sz_vector(i)[site])); }
    MatrixType out(basis_.dim(), basis_.dim());
    out.setFromTriplets(tripletList.begin(), tripletList.end());
    return out;
}

MatrixType Heisenberg::Sp(const std::size_t& site) const
{
    std::vector<Eigen::Triplet<Scalar>> tripletList;
    for(size_t i = 0; i < basis_.dim(); i++) {
        auto state = basis_.state_vector(i);
        if(state[site] == basis_.D(site) - 1ul) { continue; }
        Scalar sz = state[site] - basis_.S(site);
        state[site]++;

        tripletList.push_back(Eigen::Triplet<Scalar>(i, basis_.number(state), std::sqrt(basis_.S(site) * (basis_.S(site) + 1) - sz * (sz + 1))));
    }
    MatrixType out(basis_.dim(), basis_.dim());
    out.setFromTriplets(tripletList.begin(), tripletList.end());
    return out;
}

MatrixType Heisenberg::H(const Eigen::Matrix<Scalar, -1, -1>& J) const
{
    assert(J.rows() == L() and J.cols() == L());
    MatrixType out(basis_.dim(), basis_.dim());
    out.setZero();
    for(std::size_t l1 = 0; l1 < L(); l1++)
        for(std::size_t l2 = 0; l2 < L(); l2++) {
            if(J(l1, l2) == 0.) { continue; }
            out += J(l1, l2) * (Sz(l1) * Sz(l2) + 0.5 * (Sp(l1) * Sm(l2) + Sm(l1) * Sp(l2)));
        }
    return out;
}

} // namespace Models
