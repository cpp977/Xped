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

    template <typename OtherTtype>
    TMatrix(const TMatrix<OtherTtype>& other)
    {
        pat = other.pat;
        is_changed.resize(pat.uniqueSize());
        std::fill(is_changed.begin(), is_changed.end(), false);
        tensors.resize(other.size());
        for(auto it = other.cbegin(); it != other.cend(); ++it) { tensors[std::distance(other.cbegin(), it)] = *it; }
    }

    inline std::size_t rows() const { return pat.Lx; }
    inline std::size_t cols() const { return pat.Ly; }

    inline std::size_t size() const { return tensors.size(); }

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
    inline const Ttype& at(const std::size_t index) const { return tensors.at(index); }

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

    auto begin() { return tensors.begin(); }
    auto end() { return tensors.end(); }

    auto cbegin() const { return tensors.cbegin(); }
    auto cend() const { return tensors.cend(); }

    void fill(const std::vector<Ttype>& tensors_in) { tensors = tensors_in; }

    void setConstant(const Ttype& val) { std::fill(tensors.begin(), tensors.end(), val); }

    Ttype sum() const { return std::accumulate(tensors.begin(), tensors.end(), Ttype(0.)); }

    Pattern pat;

private:
    std::vector<Ttype> tensors;
    std::vector<bool> is_changed;
    std::string name = "";
};

} // namespace Xped
#endif
