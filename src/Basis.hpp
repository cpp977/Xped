#ifndef BASIS_H_
#define BASIS_H_

/// \cond
#include "Eigen/Core"

#include "tabulate/tabulate.hpp"
/// \endcond

// forward declaration
template <typename Symmetry, std::size_t depth>
class Qbasis;

/** \class Basis
 *
 * Class for a plain basis without symmetries. A basis with symmetries is a collection of plain bases for each Symmetry entry
 *
 */
class Basis
{
    template <typename Symmetry_, std::size_t depth>
    friend class Qbasis;

public:
    /**Does nothing.*/
    Basis(){};

    Basis(std::size_t dim_in)
        : dim_(dim_in){};

    Eigen::Index dim() const { return dim_; }

    /**Adds to bases together.*/
    Basis add(const Basis& other) const;

    /**
     * Returns the tensor product basis.
     * This function also saves the history of the combination process for later use. See leftAmount() and rightAmount().
     */
    Basis combine(const Basis& other) const;

    /**Prints the basis.*/
    auto print() const;

    /**Prints the history of the basis.*/
    auto printHistory() const;

    bool operator==(const Basis& other) const { return (this->dim() == other.dim()); }

    friend std::ostream& operator<<(std::ostream& os, const Basis& basis);

private:
    std::size_t dim_;

    struct fuseData
    {
        Eigen::Index dim1, dim2;
        std::array<Eigen::Index, 2> source(Eigen::Index combined_num) const
        {
            Eigen::Index tmp = combined_num / dim1;
            return {{combined_num - dim1 * tmp, tmp}};
        };
    };
    Basis::fuseData history;
};

Basis Basis::add(const Basis& other) const
{
    Basis out(dim() + other.dim());
    return out;
}

Basis Basis::combine(const Basis& other) const
{
    Basis out(this->dim() * other.dim());
    out.history.dim1 = this->dim();
    out.history.dim2 = other.dim();

    return out;
}

auto Basis::print() const
{
    tabulate::Table t;
    // TextTable t('-', '|', '+');
    t.add_row({"num"});
    // t.add("num");
    // t.endOfRow();
    for(std::size_t i = 0; i < dim_; i++) {
        std::stringstream ss;
        ss << i;
        t.add_row({ss.str()});
        // t.add(ss.str());
        // t.endOfRow();
    }
    t.format().font_style({tabulate::FontStyle::bold}).border_top("_").border_bottom("_").border_left("|").border_right("|").corner("+");
    t[0].format()
        .padding_top(1)
        .padding_bottom(1)
        .font_align(tabulate::FontAlign::center)
        .font_style({tabulate::FontStyle::underline})
        .font_background_color(tabulate::Color::red);
    return t;
}

auto Basis::printHistory() const
{
    tabulate::Table t;
    t.add_row({"num", "source"});
    for(std::size_t i = 0; i < dim_; i++) {
        std::stringstream ss, tt;
        ss << i;
        tt << "<-- " << history.source(i)[0] << "," << history.source(i)[1];
        t.add_row({ss.str(), tt.str()});
    }
    t.format().font_style({tabulate::FontStyle::bold}).border_top("_").border_bottom("_").border_left("|").border_right("|").corner("+");
    t[0].format()
        .padding_top(1)
        .padding_bottom(1)
        .font_align(tabulate::FontAlign::center)
        .font_style({tabulate::FontStyle::underline})
        .font_background_color(tabulate::Color::red);
    t[0][1].format().font_background_color(tabulate::Color::blue).font_color(tabulate::Color::white);
    return t;
}

std::ostream& operator<<(std::ostream& os, const Basis& basis)
{
    os << basis.print();
    return os;
}
#endif
