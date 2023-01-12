#ifndef XPED_ONE_SITE_OBSERVABLE_HPP_
#define XPED_ONE_SITE_OBSERVABLE_HPP_

#include "fmt/core.h"

#include <highfive/H5DataSpace.hpp>

#include "Xped/Core/Tensor.hpp"
#include "Xped/PEPS/CTMOpts.hpp"
#include "Xped/PEPS/ObservableBase.hpp"
#include "Xped/PEPS/TMatrix.hpp"

namespace Xped {

template <typename, typename, std::size_t, bool, Opts::CTMCheckpoint>
class CTM;

template <typename Symmetry>
struct OneSiteObservable : public ObservableBase
{
    explicit OneSiteObservable(const Pattern& pat, const std::string& name_in = "")
        : ObservableBase(name_in)
        , data(pat)
        , obs(pat)
    {}

    OneSiteObservable<Symmetry> shiftQN(const TMatrix<typename Symmetry::qType>& charges)
    {
        if(std::all_of(charges.cbegin(), charges.cend(), [](auto c) { return c == Symmetry::qvacuum(); })) { return *this; }
        OneSiteObservable<Symmetry> out(data.pat);
        for(int x = 0; x < data.pat.Lx; ++x) {
            for(int y = 0; y < data.pat.Ly; ++y) {
                if(not data.pat.isUnique(x, y)) { continue; }
                out.data(x, y) = data(x, y).template shiftQN<0, 1>(std::array{charges(x, y), charges(x, y)});
            }
        }
        return out;
    }

    virtual std::string getResString(const std::string& offset) const override
    {
        std::string res;
        fmt::format_to(
            std::back_inserter(res), "{}{:<10}: avg={:+.2f}, vals={::+.4f}", offset, this->name, obs.sum() / obs.size(), obs.uncompressedVector());
        return res;
    }

    virtual void toFile(HighFive::File& file, const std::string& root = "/") const override
    {
        if(not file.exist(root + this->name)) {
            HighFive::DataSpace dataspace = HighFive::DataSpace({0, 0}, {HighFive::DataSpace::UNLIMITED, HighFive::DataSpace::UNLIMITED});

            // Use chunking
            HighFive::DataSetCreateProps props;
            props.add(HighFive::Chunking(std::vector<hsize_t>{2, 2}));

            // Create the dataset
            HighFive::DataSet dataset = file.createDataSet(root + this->name, dataspace, HighFive::create_datatype<double>(), props);
        }
        auto d = file.getDataSet(root + this->name);
        std::vector<std::vector<double>> data;
        data.push_back(obs.uncompressedVector());
        std::size_t curr_size = d.getDimensions()[0];
        d.resize({curr_size + 1, data[0].size()});
        d.select({curr_size, 0}, {1, data[0].size()}).write(data);
    }

    TMatrix<Tensor<double, 1, 1, Symmetry, false>> data;
    TMatrix<double> obs;
};

} // namespace Xped

#endif
