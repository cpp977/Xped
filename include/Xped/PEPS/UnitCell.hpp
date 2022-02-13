#ifndef XPED_UNIT_CELL_H_
#define XPED_UNIT_CELL_H_

#include <vector>

#include "Xped/PEPS/Pattern.hpp"

namespace Xped {

struct UnitCell
{
    explicit UnitCell(const Pattern& pat)
        : Lx(pat.Lx)
        , Ly(pat.Ly)
        , pattern(pat)
    {}

    UnitCell(const std::size_t Lx = 1, const std::size_t Ly = 1);

    std::size_t Lx;
    std::size_t Ly;

    Pattern pattern;

    inline std::size_t uniqueSize() const { return pattern.uniqueSize(); }
    inline std::size_t size() const { return Lx * Ly; }
};

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "PEPS/UnitCell.cpp"
#endif

#endif
