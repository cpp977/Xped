#ifndef XPED_OBSERVABLE_BASE_HPP
#define XPED_OBSERVABLE_BASE_HPP

#include <string>

#include <highfive/H5File.hpp>

namespace Xped {

struct ObservableBase
{
    explicit ObservableBase(const std::string& name_in = "", bool MEASURE_IN = true)
        : name(name_in)
        , MEASURE(MEASURE_IN)
    {}

    virtual std::string file_name() const { return "Op"; };
    virtual std::string format() const { return "Op"; };

    virtual std::string getResString(const std::string& offset) const = 0;

    virtual void toFile(HighFive::File& file, const std::string&) const = 0;

    std::string name = "Op";
    bool MEASURE = true;

    virtual ~ObservableBase() = default;
};

} // namespace Xped
#endif
