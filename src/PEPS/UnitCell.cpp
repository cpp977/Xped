#include "Xped/PEPS/UnitCell.hpp"

namespace Xped {

UnitCell::UnitCell(const std::size_t Lx, const std::size_t Ly)
    : Lx(Lx)
    , Ly(Ly)
{
    string alphabet = "abcdefghijklmnopqrstuvwxyz";
    assert(Lx * Ly < alphabet.size());
    std::vector<std::vector<char>> pat(Lx);
    for(auto& row : pat) { row.resize(Ly); }
    for(std::size_t x = 0; x < Lx; x++) {
        for(std::size_t y = 0; y < Ly; y++) { pat[x][y] = alphabet[x + y * Lx]; }
    }
    pattern = Pattern(pat);
}

} // namespace Xped
