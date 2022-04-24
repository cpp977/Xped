#ifndef XPED_VECOFMAT_ITERATOR_HPP_
#define XPED_VECOFMAT_ITERATOR_HPP_

#include <boost/iterator/iterator_facade.hpp>

#include "Xped/Core/Qbasis.hpp"
#include "Xped/Interfaces/PlainInterface.hpp"

namespace Xped {

template <typename Scalar>
class VecOfMatIterator : public boost::iterator_facade<VecOfMatIterator<Scalar>, Scalar*, boost::forward_traversal_tag, Scalar>
{
public:
    VecOfMatIterator()
        : data(nullptr)
    {}

    VecOfMatIterator(std::vector<PlainInterface::MType<Scalar>>* o_val, std::size_t block_num = 0, std::size_t elem_num = 0)
        : data(o_val)
        , block_num(block_num)
        , elem_num(elem_num)
    {}

private:
    friend class boost::iterator_core_access;

    void increment()
    {
        if(block_num == data->size() - 1 and elem_num == data->at(block_num).size() - 1) {
            block_num = data->size();
            elem_num = 0;
            return;
        }
        if(elem_num < data->at(block_num).size() - 1)
            ++elem_num;
        else {
            ++block_num;
            elem_num = 0;
        }
    }

    bool equal(VecOfMatIterator<Scalar> const& other) const { return (this->block_num == other.block_num and this->elem_num == other.elem_num); }

    Scalar dereference() const { return *(PlainInterface::get_raw_data(data->at(block_num)) + elem_num); }

    std::vector<PlainInterface::MType<Scalar>>* data;
    std::size_t block_num = 0;
    std::size_t elem_num = 0;
};

} // namespace Xped
#endif
