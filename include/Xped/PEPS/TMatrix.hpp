#ifndef XPED_TMATRIX_H_
#define XPED_TMATRIX_H_

#include "Xped/PEPS/Pattern.hpp"

namespace Xped {

template <typename Ttype>
struct TMatrix
{
    TMatrix() = default;

    TMatrix(const Pattern& pat, const std::string name = "")
        : pat(pat)
        , name(name)
    {
        tensors.resize(pat.uniqueSize());
        is_changed.resize(pat.uniqueSize());
        std::fill(is_changed.begin(), is_changed.end(), false);
    }

    inline std::size_t rows() const { return pat.Lx; }
    inline std::size_t cols() const { return pat.Lx; }

    inline Ttype& operator()(const int row, const int col)
    {
        is_changed[pat.uniqueIndex(row, col)] = true;
        return tensors[pat.uniqueIndex(row, col)];
    }
    inline const Ttype& operator()(const int row, const int col) const { return tensors[pat.uniqueIndex(row, col)]; }

    inline Ttype& operator[](const std::size_t index)
    {
        is_changed[index] = true;
        return tensors[index];
    }
    inline const Ttype& operator[](const std::size_t index) const { return tensors[index]; }

    inline bool isChanged(const int row, const int col) const { return is_changed[pat.uniqueIndex(row, col)]; }
    inline void resetChange() { std::fill(is_changed.begin(), is_changed.end(), false); }

    inline void resize(const Pattern& pattern)
    {
        pat = pattern;
        tensors.clear();
        tensors.resize(pat.uniqueSize());
        is_changed.resize(pat.uniqueSize());
        std::fill(is_changed.begin(), is_changed.end(), false);
    }

private:
    std::vector<Ttype> tensors;
    std::vector<bool> is_changed;
    Pattern pat;
    std::string name = "";
};

} // namespace Xped
#endif
