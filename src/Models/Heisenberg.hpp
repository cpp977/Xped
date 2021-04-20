#ifndef HEISENBERG_H_
#define HEISENBERG_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include </home/mpeschke/__LIBS__/eigen/Eigen/Core>
#include </home/mpeschke/__LIBS__/eigen/Eigen/Sparse>

#include "/home/mpeschke/Nextcloud/postdoc/code/ED/src/Basis.h"

namespace Models {

class Heisenberg
{
public:
    typedef double Scalar;
    typedef Eigen::SparseMatrix<Scalar> MatrixType;

    Heisenberg(){};
    Heisenberg(const std::size_t& L_in, const std::size_t& D_in = 3)
        : basis_(L_in, D_in){};
    Heisenberg(const std::vector<std::size_t>& Ds_in)
        : basis_(Ds_in){};

    std::size_t L() const { return basis_.L(); }
    std::size_t dim() const { return basis_.dim(); }

    MatrixType Sz(const std::size_t& site) const;
    MatrixType Sp(const std::size_t& site) const;
    MatrixType Sm(const std::size_t& site) const { return Sp(site).adjoint(); }
    MatrixType Sx(const std::size_t& site) const { return 0.5 * (Sp(site) + Sm(site)); }

    MatrixType H(const Eigen::Matrix<Scalar, -1, -1>& J) const;

private:
    Basis basis_;
};

} // namespace Models
#endif
