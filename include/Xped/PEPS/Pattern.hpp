#ifndef XPED_PATTERN_H_
#define XPED_PATTERN_H_

#include <map>
#include <string>
#include <vector>

#include "tabulate/table.hpp"

namespace Xped {

struct UnitCell;

struct Pattern
{
    friend struct UnitCell;

    Pattern(const std::vector<std::vector<std::string>>& pat = {{"a"}})
        : data(pat)
    {
        init();
    }

    void init();

    Pattern row(int x) const;
    Pattern col(int y) const;

    inline std::size_t uniqueSize() const { return label2index.size(); }
    inline std::size_t size() const { return Lx * Ly; }
    std::size_t index(const int x, const int y) const;
    std::size_t uniqueIndex(const int x, const int y) const;
    std::size_t uniqueIndex(const std::size_t index) const;
    std::pair<int, int> coords(const std::size_t index) const;
    bool isUnique(const int x, const int y) const;

    tabulate::Table print() const;

    std::map<std::string, std::size_t> label2index;
    std::map<std::size_t, std::size_t> index2unique;
    std::vector<std::vector<std::string>> data;
    std::map<std::string, std::vector<std::size_t>> sites_of_label;
    std::size_t Lx = 1, Ly = 1;
};

std::ostream& operator<<(std::ostream& os, const Pattern& pat);

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "PEPS/Pattern.cpp"
#endif

#endif
