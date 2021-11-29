#ifndef BASIS_H_
#define BASIS_H_

/// \cond
#include "yas/serialize.hpp"
#include "yas/std_types.hpp"
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

    inline const std::size_t dim() const { return dim_; }

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

    inline bool operator==(const Basis& other) const { return (this->dim() == other.dim()); }

    friend std::ostream& operator<<(std::ostream& os, const Basis& basis);

    template <typename Ar>
    void serialize(Ar& ar)
    {
        ar& YAS_OBJECT_NVP("Basis", ("dim", dim_), ("history", history));
    }

private:
    std::size_t dim_;

    struct fuseData
    {
        std::size_t dim1, dim2;
        std::array<std::size_t, 2> source(std::size_t combined_num) const
        {
            std::size_t tmp = combined_num / dim1;
            return {{combined_num - dim1 * tmp, tmp}};
        };
        template <typename Ar>
        void serialize(Ar& ar)
        {
            ar& YAS_OBJECT_NVP("fuseData", ("dim1", dim1), ("dim2", dim2));
        }
    };

    Basis::fuseData history;
};

#ifndef XPED_COMPILED_LIB
#    include "Core/Basis.cpp"
#endif

#endif
