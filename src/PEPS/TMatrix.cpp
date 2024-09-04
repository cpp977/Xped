#include "Xped/PEPS/TMatrix.hpp"

namespace Xped {

template <typename Ttype>
Ttype& TMatrix<Ttype>::operator()(int row, int col)
{
    is_changed[pat.uniqueIndex(row, col)] = true;
    return tensors[pat.uniqueIndex(row, col)];
}

template <typename Ttype>
const Ttype& TMatrix<Ttype>::operator()(int row, int col) const
{
    return tensors[pat.uniqueIndex(row, col)];
}

} // namespace Xped

#if __has_include("TMatrix.gen.cpp")
#    include "TMatrix.gen.cpp"
#endif
