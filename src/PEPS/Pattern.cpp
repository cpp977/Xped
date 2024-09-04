#include <assert.hpp>
#include <numeric>

#include "Xped/PEPS/Pattern.hpp"

namespace Xped {

void Pattern::init()
{
    DEBUG_ASSERT(data.size() > 0);
    std::vector<std::size_t> flattened_pat;
    for(auto const& row : data) { flattened_pat.insert(flattened_pat.end(), row.begin(), row.end()); }
    std::sort(flattened_pat.begin(), flattened_pat.end());
    auto last = std::unique(flattened_pat.begin(), flattened_pat.end());
    flattened_pat.erase(last, flattened_pat.end());
    auto offset = flattened_pat[0];
    [[maybe_unused]] std::vector<std::size_t> check(flattened_pat.size());
    std::iota(check.begin(), check.end(), offset);
    DEBUG_ASSERT(flattened_pat == check);
    for(auto& row : data) {
        for(auto& val : row) { val = val - offset; }
    }
    Lx = data.size();
    DEBUG_ASSERT(Lx > 0);
    Ly = data[0].size();
    for([[maybe_unused]] const auto& row : data) { DEBUG_ASSERT(row.size() == Ly); }

    for(int x = 0; x < Lx; x++) {
        for(int y = 0; y < Ly; y++) {
            index2unique[index(x, y)] = data[x][y];
            label2index[data[x][y]] = data[x][y];
            sites_of_label[data[x][y]].push_back(index(x, y));
        }
    }
    for(auto i = 0ul; i < uniqueSize(); ++i) {
        DEBUG_ASSERT(i == index(coords(i).first, coords(i).second), "Index flattening in pattern is broken.");
    }
}

Pattern Pattern::col(int y) const
{
    std::vector<std::vector<std::size_t>> col_data(1);
    col_data[0] = data[y];
    return Pattern(col_data);
}

Pattern Pattern::row(int x) const
{
    std::vector<std::vector<std::size_t>> row_data(data.size());
    for(std::size_t y = 0; y < Ly; y++) { row_data[y] = {data[y][x]}; }
    return Pattern(row_data);
}

std::size_t Pattern::index(const int x, const int y) const
{
    // Apply periodic boundary conditions
    int x_copy = x;
    int y_copy = y;
    while(x_copy < 0) { x_copy += static_cast<int>(Lx); }
    while(x_copy >= Lx) { x_copy -= static_cast<int>(Lx); }
    while(y_copy < 0) { y_copy += static_cast<int>(Ly); }
    while(y_copy >= Ly) { y_copy -= static_cast<int>(Ly); }
    // return static_cast<std::size_t>(y_copy + x_copy * Ly);
    return static_cast<std::size_t>(x_copy + y_copy * Lx);
}

std::size_t Pattern::uniqueIndex(const int x, const int y) const { return uniqueIndex(index(x, y)); }

std::size_t Pattern::uniqueIndex(const std::size_t index) const { return index2unique.at(index); }

std::pair<int, int> Pattern::coords(std::size_t index) const { return std::make_pair(static_cast<int>(index % Lx), static_cast<int>(index / Lx)); }
// std::pair<int, int> Pattern::coords(std::size_t index) const { return std::make_pair(static_cast<int>(index / Ly), static_cast<int>(index % Ly)); }

// bool Pattern::isUnique(const int x, const int y) const { return (uniqueIndex(x, y) == index(x, y)); }
bool Pattern::isUnique(const int x, const int y) const { return index(x, y) < uniqueSize(); }

tabulate::Table Pattern::print() const
{
    tabulate::Table outer;
    outer.add_row({"Pattern"});
    tabulate::Table pat_table;
    using Row_t = std::vector<std::variant<std::string, const char*, tabulate::Table>>;
    for(const auto& row : data) {
        Row_t t_row(row.size());
        for(auto i = 0; i < row.size(); i++) { t_row[i] = std::to_string(row[i]); }
        pat_table.add_row(t_row);
    }
    // pat_table.format().font_style({tabulate::FontStyle::bold}).border_top(" ").border_bottom(" ").border_left(" ").border_right(" ").corner(" ");
    outer.add_row({pat_table});
    outer.format().font_style({tabulate::FontStyle::bold}).border_top(" ").border_bottom(" ").border_left(" ").border_right(" ").corner(" ");
    return outer;
}

std::ostream& operator<<(std::ostream& os, const Pattern& pat)
{
    os << pat.print();
    return os;
}

} // namespace Xped

#if __has_include("Pattern.gen.cpp")
#    include "Pattern.gen.cpp"
#endif
