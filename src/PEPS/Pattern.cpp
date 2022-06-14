#include <cassert>

#include "Xped/PEPS/Pattern.hpp"

namespace Xped {

void Pattern::init()
{
    Lx = data.size();
    assert(Lx > 0);
    Ly = data[0].size();
    for([[maybe_unused]] const auto& row : data) { assert(row.size() == Ly); }

    for(int x = 0; x < Lx; x++) {
        for(int y = 0; y < Ly; y++) {
            auto it = label2index.find(data[x][y]);
            if(it == label2index.end()) {
                label2index[data[x][y]] = index(x, y);
                index2unique[index(x, y)] = index2unique.size();
            } else {
                index2unique[index(x, y)] = index2unique.at(label2index.at(data[x][y]));
            }
            sites_of_label[data[x][y]].push_back(index(x, y));
        }
    }
}

Pattern Pattern::row(int x) const
{
    std::vector<std::vector<std::string>> row_data(1);
    row_data[0] = data[x];
    return Pattern(row_data);
}

Pattern Pattern::col(int y) const
{
    std::vector<std::vector<std::string>> col_data(data.size());
    for(std::size_t x = 0; x < Lx; x++) { col_data[x] = {data[x][y]}; }
    return Pattern(col_data);
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
    return static_cast<std::size_t>(y_copy + x_copy * Ly);
}

std::size_t Pattern::uniqueIndex(const int x, const int y) const { return uniqueIndex(index(x, y)); }

std::size_t Pattern::uniqueIndex(const std::size_t index) const { return index2unique.at(index); }

std::pair<int, int> Pattern::coords(std::size_t index) const { return std::make_pair(static_cast<int>(index % Lx), static_cast<int>(index / Lx)); }

bool Pattern::isUnique(const int x, const int y) const { return (uniqueIndex(x, y) == index(x, y)); }

tabulate::Table Pattern::print() const
{
    tabulate::Table outer;
    outer.add_row({"Pattern"});
    tabulate::Table pat_table;
    using Row_t = std::vector<std::variant<std::string, const char*, tabulate::Table>>;
    for(const auto& row : data) {
        Row_t t_row(row.size());
        for(auto i = 0; i < row.size(); i++) { t_row[i] = row[i]; }
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
