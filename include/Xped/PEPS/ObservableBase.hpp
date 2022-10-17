#ifndef XPED_OBSERVABLE_BASE_HPP
#define XPED_OBSERVABLE_BASE_HPP

#include <string>

namespace Xped {
struct ObservableBase
{
    virtual std::string name() const { return "Op"; };
    bool MEASURE = true;

    virtual ~ObservableBase() = default;
};

} // namespace Xped
#endif
