#ifndef XPED_PARAM_HPP_
#define XPED_PARAM_HPP_

#include <any>

#include "fmt/core.h"

namespace Xped {

struct Param
{
    std::any value;
    template <typename Scalar>
    Scalar get() const
    {
        Scalar res;
        try {
            res = std::any_cast<Scalar>(value);
        } catch(const std::bad_any_cast& e) {
            fmt::print(stderr, "Conversion failed for value type={}\nError: {}\n", value.type().name(), e.what());
            std::terminate();
        }
        return res;
    }
};

} // namespace Xped
#endif
