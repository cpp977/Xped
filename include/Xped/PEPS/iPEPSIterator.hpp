#ifndef XPED_IPEPS_ITERATOR_HPP_
#define XPED_IPEPS_ITERATOR_HPP_

#include <iterator>

#include <boost/iterator/iterator_facade.hpp>

#include "Xped/Core/Qbasis.hpp"
#include "Xped/Interfaces/PlainInterface.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
class iPEPSIterator : public boost::iterator_facade<iPEPSIterator<Scalar, Symmetry, ENABLE_AD>, Scalar*, boost::forward_traversal_tag, Scalar>
{
public:
    iPEPSIterator()
        : data(nullptr)
    {}

    iPEPSIterator(TMatrix<Tensor<Scalar, 2, 3, Symmetry, ENABLE_AD>>* data_in,
                  bool ITER_GRAD = false,
                  std::size_t outer_num = 0,
                  std::size_t elem_num = 0)
        : data(data_in)
        , ITER_GRAD(ITER_GRAD)
        , outer_num(outer_num)
        , elem_num(elem_num)
    {
        if constexpr(not ENABLE_AD) { assert(ITER_GRAD == false); }
    }

private:
    friend class boost::iterator_core_access;

    void increment()
    {
        if(outer_num == data->size() - 1 and elem_num == data->at(outer_num).plainSize() - 1) {
            outer_num = data->size();
            elem_num = 0;
            return;
        }
        if(elem_num < data->at(outer_num).plainSize() - 1)
            ++elem_num;
        else {
            ++outer_num;
            elem_num = 0;
        }
    }

    bool equal(iPEPSIterator<Scalar, Symmetry, ENABLE_AD> const& other) const
    {
        return (this->outer_num == other.outer_num and this->elem_num == other.elem_num);
    }

    Scalar dereference() const
    {
        if constexpr(ENABLE_AD) {
            auto beg = ITER_GRAD ? data->at(outer_num).cgradbegin() : data->at(outer_num).cbegin();
            std::advance(beg, elem_num);
            return *beg;
        } else {
            auto beg = data->at(outer_num).cbegin();
            std::advance(beg, elem_num);
            return *beg;
        }
    }

    TMatrix<Tensor<Scalar, 2, 3, Symmetry, ENABLE_AD>>* data;
    bool ITER_GRAD = false;
    std::size_t outer_num = 0;
    std::size_t elem_num = 0;
};

} // namespace Xped
#endif
