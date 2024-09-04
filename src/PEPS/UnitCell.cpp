#include <cassert>
#include <string>
#include <vector>

#include <highfive/H5File.hpp>

#include "Xped/PEPS/UnitCell.hpp"

namespace Xped {

UnitCell::UnitCell(const std::size_t Lx, const std::size_t Ly)
    : Lx(Lx)
    , Ly(Ly)
{
    // const std::vector<std::string> alphabet = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    //                                            "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"};
    // assert(Lx * Ly < alphabet.size());
    std::vector<std::vector<std::size_t>> pat(Lx);
    for(auto& row : pat) { row.resize(Ly); }
    for(std::size_t x = 0; x < Lx; x++) {
        for(std::size_t y = 0; y < Ly; y++) { pat[x][y] = x + y * Lx; }
    }
    pattern = Pattern(pat);
}

void UnitCell::loadFromMatlab(const std::filesystem::path& p, const std::string& root_name)
{
    HighFive::File file(p.string(), HighFive::File::ReadOnly);
    auto root = file.getGroup(root_name);

    std::vector<std::vector<double>> raw_pat;
    HighFive::DataSet pat_data = root.getDataSet("meta/pattern");
    pat_data.read(raw_pat);
    std::vector<std::vector<std::size_t>> pat_vec(raw_pat.size());
    for(auto i = 0ul; i < raw_pat.size(); ++i) {
        pat_vec[i].resize(raw_pat[i].size());
        for(auto j = 0ul; j < raw_pat[i].size(); ++j) { pat_vec[i][j] = static_cast<std::size_t>(std::round(raw_pat[i][j])); }
    }
    pattern = Pattern(pat_vec, true);
}
} // namespace Xped

#if __has_include("UnitCell.gen.cpp")
#    include "UnitCell.gen.cpp"
#endif
