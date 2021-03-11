#ifndef BASIS_H_
#define BASIS_H_

#include "TextTable.h"

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
    std::string print() const;

    /**Prints the history of the basis.*/
    std::string printHistory() const;

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

std::string Basis::print() const
{
    std::stringstream out;
    TextTable t('-', '|', '+');
    t.add("num");
    t.endOfRow();
    for(std::size_t i = 0; i < dim_; i++) {
        std::stringstream ss;
        ss << i;
        t.add(ss.str());
        t.endOfRow();
    }
    out << t;
    return out.str();
}

std::string Basis::printHistory() const
{
    std::stringstream out;
    TextTable t('-', '|', '+');
    t.add("num");
    t.add("source");
    t.endOfRow();
    for(std::size_t i = 0; i < dim_; i++) {
        std::stringstream ss, tt;
        ss << i;
        tt << "<-- " << history.source(i)[0] << "," << history.source(i)[1];
        t.add(ss.str());
        t.endOfRow();
    }
    out << t;
    return out.str();
}

std::ostream& operator<<(std::ostream& os, const Basis& basis)
{
    os << basis.print();
    return os;
}
#endif
