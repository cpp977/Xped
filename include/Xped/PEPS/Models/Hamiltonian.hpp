#ifndef XPED_HAMILTONIAN_HPP_
#define XPED_HAMILTONIAN_HPP_

#include <vector>

#include "Xped/PEPS/ObservableBase.hpp"
#include "Xped/PEPS/TwoSiteObservable.hpp"
#include "Xped/Util/Param.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry>
struct Hamiltonian
{
    Hamiltonian() = default;

    Hamiltonian(std::map<std::string, Param>& params_in, const Pattern& pat_in, Opts::Bond bond_in, const std::string& name_in = "H")
        : params(params_in)
        , pat(pat_in)
        , bond(bond_in)
        , name(name_in)
    {
        if((bond & Opts::Bond::H) == Opts::Bond::H) { data_h = TMatrix<Tensor<Scalar, 2, 2, Symmetry, false>>(pat); }
        if((bond & Opts::Bond::V) == Opts::Bond::V) { data_v = TMatrix<Tensor<Scalar, 2, 2, Symmetry, false>>(pat); }
        if((bond & Opts::Bond::D1) == Opts::Bond::D1) { data_d1 = TMatrix<Tensor<Scalar, 2, 2, Symmetry, false>>(pat); }
        if((bond & Opts::Bond::D2) == Opts::Bond::D2) { data_d2 = TMatrix<Tensor<Scalar, 2, 2, Symmetry, false>>(pat); }
    }

    virtual ~Hamiltonian() = default;

    TwoSiteObservable<Scalar, Symmetry, true> asObservable() const
    {
        TwoSiteObservable<Scalar, Symmetry, true> res(pat, bond);
        res.data_h = data_h;
        res.data_v = data_v;
        res.data_d1 = data_d1;
        res.data_d2 = data_d2;
        return res;
    }

    std::string getObsString(const std::string& offset) const
    {
        std::string out;
        for(const auto& ob : obs) {
            out.append(ob->getResString(offset));
            if(&ob != &obs.back()) { out.push_back('\n'); }
        }
        return out;
    }

    void obsToFile(HighFive::File& file, const std::string& root = "/") const
    {
        for(const auto& ob : obs) { ob->toFile(file, root); }
    }

    virtual void setDefaultObs() {}

    std::string file_name() const
    {
        std::string res = name + "_";
        for(const auto& p : used_params) {
            if(p != used_params.back()) {
                if(typeid(double).hash_code() == params.at(p).value.type().hash_code()) {
                    fmt::format_to(std::back_inserter(res), "{}={:.2f}_", p, params.at(p).get<double>());
                } else if(typeid(TMatrix<double>).hash_code() == params.at(p).value.type().hash_code()) {
                    fmt::format_to(std::back_inserter(res), "{}={}_", p, params.at(p).get<TMatrix<double>>().uncompressedVector());
                }
            } else {
                if(typeid(double).hash_code() == params.at(p).value.type().hash_code()) {
                    fmt::format_to(std::back_inserter(res), "{}={:.2f}", p, params.at(p).get<double>());
                } else if(typeid(TMatrix<double>).hash_code() == params.at(p).value.type().hash_code()) {
                    fmt::format_to(std::back_inserter(res), "{}={}", p, params.at(p).get<TMatrix<double>>().uncompressedVector());
                }
            }
        }
        return res;
    }

    std::string format() const
    {
        std::string prefix = fmt::format("{}[sym={}]", name, Symmetry::name());
        std::string inner;
        for(const auto& p : used_params) {
            if(p != used_params.back()) {
                if(typeid(double).hash_code() == params.at(p).value.type().hash_code()) {
                    fmt::format_to(std::back_inserter(inner), "{}={:.2f},", p, params.at(p).get<double>());
                } else if(typeid(TMatrix<double>).hash_code() == params.at(p).value.type().hash_code()) {
                    fmt::format_to(std::back_inserter(inner), "{}={},", p, params.at(p).get<TMatrix<double>>().uncompressedVector());
                }
            } else {
                if(typeid(double).hash_code() == params.at(p).value.type().hash_code()) {
                    fmt::format_to(std::back_inserter(inner), "{}={:.2f}", p, params.at(p).get<double>());
                } else if(typeid(TMatrix<double>).hash_code() == params.at(p).value.type().hash_code()) {
                    fmt::format_to(std::back_inserter(inner), "{}={}", p, params.at(p).get<TMatrix<double>>().uncompressedVector());
                }
            }
        }
        std::string res = fmt::format("{}({})", prefix, inner);
        return res;
    }

    TMatrix<Tensor<Scalar, 2, 2, Symmetry, false>> data_h;
    TMatrix<Tensor<Scalar, 2, 2, Symmetry, false>> data_v;
    TMatrix<Tensor<Scalar, 2, 2, Symmetry, false>> data_d1;
    TMatrix<Tensor<Scalar, 2, 2, Symmetry, false>> data_d2;

    std::map<std::string, Param> params;
    Pattern pat;
    Opts::Bond bond;
    std::string name;
    std::vector<std::string> used_params;
    std::vector<std::unique_ptr<ObservableBase>> obs;
    Opts::DiscreteSym sym_ = Opts::DiscreteSym::None;
};

} // namespace Xped

#endif
