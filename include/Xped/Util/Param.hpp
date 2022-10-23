#ifndef XPED_PARAM_HPP_
#define XPED_PARAM_HPP_

#include <any>

namespace Xped {

struct Param
{
    std::any value;
    template <typename Scalar>
    Scalar get() const
    {
        return std::any_cast<Scalar>(value);
    }
};

} // namespace Xped
#endif
