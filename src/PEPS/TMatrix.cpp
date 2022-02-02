#include "Xped/PEPS/TMatrix.hpp"

template <typename Ttype>
Ttype& TMatrix<Ttype>::operator()(const std::size_t row, const std::size_t col)
{
    return tensors[pat.uniqueIndex(row, col)];
}

template <typename Ttype>
const Ttype& TMatrix<Ttype>::operator()(const std::size_t row, const std::size_t col) const
{
    return tensors[pat.uniqueIndex(row, col)];
}
