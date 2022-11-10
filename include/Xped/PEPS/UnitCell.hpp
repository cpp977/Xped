#ifndef XPED_UNIT_CELL_H_
#define XPED_UNIT_CELL_H_

#include <vector>

#include "yas/serialize.hpp"
#include "yas/std_types.hpp"

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

    std::size_t Lx = 1;
    std::size_t Ly = 1;

    Pattern pattern{};

    template <typename Ar>
    void serialize(Ar& ar)
    {
        ar& YAS_OBJECT_NVP("UnitCell", ("Lx", Lx), ("Ly", Ly), ("pattern", pattern));
    }

    void loadFromMatlab(const std::filesystem::path& p, const std::string& root_name);

    inline std::size_t uniqueSize() const { return pattern.uniqueSize(); }
    inline std::size_t size() const { return Lx * Ly; }

    inline std::size_t rows() const { return Lx; }
    inline std::size_t cols() const { return Ly; }
};

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "PEPS/UnitCell.cpp"
#endif

#endif
