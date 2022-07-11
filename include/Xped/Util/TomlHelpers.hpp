#ifndef XPED_TOML_HELPERS_HPP_
#define XPED_TOML_HELPERS_HPP_

#include <any>
#include <exception>
#include <map>
#include <string>

#include "toml.hpp"

#include "Xped/PEPS/Bonds.hpp"

namespace Xped::util {

template <typename T, typename = typename std::enable_if_t<boost::describe::has_describe_enumerators<T>::value>>
T enum_from_toml(const toml::value& t)
{
    T out;
    bool success = boost::describe::enum_from_string(std::string(t.as_string()).c_str(), out);
    if(not success) { throw std::invalid_argument("Bad conversion from toml input to enum."); }
    return out;
}

std::map<std::string, std::any> params_from_toml(const toml::value& t)
{
    std::map<std::string, std::any> params;
    for(const auto& [k, v] : t.as_table()) {
        if(v.is_floating()) {
            params[k] = static_cast<double>(v.as_floating());
        } else if(v.is_array()) {
            if(v.at(0).is_floating()) {
                params[k] = toml::get<std::vector<double>>(v);
            } else if(v.at(0).is_array()) {
                params[k] = toml::get<std::vector<std::vector<double>>>(v);
            }
        } else {
            throw std::invalid_argument("Bad model parameters.");
        }
    }
    return params;
}
} // namespace Xped::util

#endif
