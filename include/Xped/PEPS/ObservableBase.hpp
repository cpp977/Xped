#ifndef XPED_OBSERVABLE_BASE_HPP
#define XPED_OBSERVABLE_BASE_HPP

#include <string>

namespace Xped {
struct ObservableBase
{
    std::string name = "Op";
    bool MEASURE = true;
};

} // namespace Xped
#endif
