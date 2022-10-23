#ifndef XPED_MODELS_HELPERS_HPP_
#define XPED_MODELS_HELPERS_HPP_

#include <map>
#include <string>
#include <vector>

#include "Xped/Util/Param.hpp"

namespace Xped::internal {

std::string format_params(const std::string& name, const std::map<std::string, Param>& params, const std::vector<std::string>& used_params)
{
    std::string inner;
    for(const auto& p : used_params) {
        if(p != used_params.back()) {
            fmt::format_to(std::back_inserter(inner), "{}={:.2f}, ", p, params.at(p).get<double>());
        } else {
            fmt::format_to(std::back_inserter(inner), "{}={:.2f}", p, params.at(p).get<double>());
        }
    }
    std::string res = fmt::format("{}({})", name, inner);
    return res;
}

std::string create_filename(const std::string& name, const std::map<std::string, Param>& params, const std::vector<std::string>& used_params)
{
    std::string res = name + "_";
    for(const auto& p : used_params) {
        if(p != used_params.back()) {
            fmt::format_to(std::back_inserter(res), "{}={:.2f}_", p, params.at(p).get<double>());
        } else {
            fmt::format_to(std::back_inserter(res), "{}={:.2f}", p, params.at(p).get<double>());
        }
    }
    return res;
}

} // namespace Xped::internal
#endif
