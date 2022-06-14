#include <cassert>
#include <string>
#include <vector>

#include "Xped/PEPS/UnitCell.hpp"

namespace Xped {

UnitCell::UnitCell(const std::size_t Lx, const std::size_t Ly)
    : Lx(Lx)
    , Ly(Ly)
{
    const std::vector<std::string> alphabet = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                                               "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"};
    assert(Lx * Ly < alphabet.size());
    std::vector<std::vector<std::string>> pat(Lx);
    for(auto& row : pat) { row.resize(Ly); }
    for(std::size_t x = 0; x < Lx; x++) {
        for(std::size_t y = 0; y < Ly; y++) { pat[x][y] = alphabet[x + y * Lx]; }
    }
    pattern = Pattern(pat);
}

} // namespace Xped
