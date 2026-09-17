#include "Xped/PEPS/TMatrix.hpp"

#include "Xped/Symmetry/qarray.hpp"
#include "Xped/Core/Tensor.hpp"
#include "Xped/AD/ADTensor.hpp"

#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"
#include "Xped/Symmetry/ZN.hpp"
#include "Xped/Symmetry/S1xS2.hpp"
#include "Xped/Symmetry/CombSym.hpp"

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

#if __has_include("TMatrix.gen.cpp") && XPED_COMPILED_LIB
#    include "TMatrix.gen.cpp"
#endif
