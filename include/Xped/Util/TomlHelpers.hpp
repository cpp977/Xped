#ifndef XPED_TOML_HELPERS_HPP_
#define XPED_TOML_HELPERS_HPP_

#include <cassert>
#include <complex>
#include <map>
#include <string>

#include "toml.hpp"

#include "Xped/PEPS/Bonds.hpp"
#include "Xped/PEPS/TMatrix.hpp"
#include "Xped/PEPS/UnitCell.hpp"
#include "Xped/Util/Param.hpp"

namespace Xped::util {

template <typename T, typename = typename std::enable_if_t<boost::describe::has_describe_enumerators<T>::value>>
T enum_from_toml(const toml::value& t)
{
    T out;
    bool success = boost::describe::enum_from_string(std::string(t.as_string()).c_str(), out);
    if(not success) { throw std::invalid_argument("Bad conversion from toml input to enum."); }
    return out;
}

inline std::map<std::string, Param> params_from_toml(const toml::value& t, const UnitCell& cell)
{
    std::map<std::string, Param> params;
    for(const auto& [k, v] : t.as_table()) {
        if(v.is_floating()) {
            params[k] = Param{.value = static_cast<double>(v.as_floating())};
        } else if(v.is_array()) {
            if(v.at(0).is_floating()) {
                params[k] = Param{.value = toml::get<std::vector<double>>(v)};
            } else if(v.at(0).at(0).is_floating()) {
                TMatrix<double> p(cell.pattern);
                auto p_vec = toml::get<std::vector<std::vector<double>>>(v);
                assert(v.size() == cell.Lx);
                assert(v[0].size() == cell.Ly);
                for(int x = 0; x < v.size(); ++x) {
                    for(int y = 0; y < v.size(); ++y) { p(x, y) = p_vec[x][y]; }
                }
                params[k] = Param{.value = p};
            } else if(v.at(0).at(0).is_array()) {
                TMatrix<std::complex<double>> p(cell.pattern);
                auto p_vec = toml::get<std::vector<std::vector<std::pair<double, double>>>>(v);
                assert(v.size() == cell.Lx);
                assert(v[0].size() == cell.Ly);
                using namespace std::literals;
                for(int x = 0; x < v.size(); ++x) {
                    for(int y = 0; y < v.size(); ++y) { p(x, y) = p_vec[x][y].first + 1i * p_vec[x][y].second; }
                }
                params[k] = Param{.value = p};
            }
        } else {
            throw std::invalid_argument("Bad model parameters.");
        }
    }
    return params;
}
} // namespace Xped::util

#endif
