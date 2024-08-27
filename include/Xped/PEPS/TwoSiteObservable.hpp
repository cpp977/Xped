#ifndef XPED_TWO_SITE_OBSERVABLE_HPP_
#define XPED_TWO_SITE_OBSERVABLE_HPP_

#include <filesystem>

#include "fmt/core.h"

#include <highfive/H5DataSpace.hpp>

#include "Xped/Core/Tensor.hpp"
#include "Xped/PEPS/Bonds.hpp"
#include "Xped/PEPS/UnitCell.hpp"
#include "Xped/PEPS/ObservableBase.hpp"
#include "Xped/PEPS/TMatrix.hpp"

#include "Xped/IO/Matlab.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry, bool HERMITIAN = true>
struct TwoSiteObservable : public ObservableBase
{
    using ObsScalar = std::conditional_t<HERMITIAN, typename ScalarTraits<Scalar>::Real, typename ScalarTraits<Scalar>::Comp>;

    TwoSiteObservable() = default;

    /*
      x,y: O_h(x,y; x+1,y)
      x,y: O_v(x,y; x,y+1)
      x,y: O_d1(x,y; x+1,y+1)
      x,y: O_d2(x,y; x-1,y+1)
     */
    TwoSiteObservable(const Pattern& pat, Opts::Bond bond, const std::string& name_in = "")
        : ObservableBase(name_in)
        , bond(bond)
    {
        if((bond & Opts::Bond::H) == Opts::Bond::H) {
            data_h = TMatrix<Tensor<Scalar, 2, 2, Symmetry, false>>(pat);
            obs_h = TMatrix<ObsScalar>(pat);
        }
        if((bond & Opts::Bond::V) == Opts::Bond::V) {
            data_v = TMatrix<Tensor<Scalar, 2, 2, Symmetry, false>>(pat);
            obs_v = TMatrix<ObsScalar>(pat);
        }
        if((bond & Opts::Bond::D1) == Opts::Bond::D1) {
            data_d1 = TMatrix<Tensor<Scalar, 2, 2, Symmetry, false>>(pat);
            obs_d1 = TMatrix<ObsScalar>(pat);
        }
        if((bond & Opts::Bond::D2) == Opts::Bond::D2) {
            data_d2 = TMatrix<Tensor<Scalar, 2, 2, Symmetry, false>>(pat);
            obs_d2 = TMatrix<ObsScalar>(pat);
        }
    }

    TwoSiteObservable<Scalar, Symmetry, HERMITIAN> shiftQN(const TMatrix<typename Symmetry::qType>& charges) const
    {
        if(std::all_of(charges.cbegin(), charges.cend(), [](auto c) { return c == Symmetry::qvacuum(); })) { return *this; }
        TwoSiteObservable<Scalar, Symmetry, HERMITIAN> out(data_h.pat, bond);
        for(int x = 0; x < data_h.pat.Lx; ++x) {
            for(int y = 0; y < data_h.pat.Ly; ++y) {
                if(not data_h.pat.isUnique(x, y)) { continue; }
                if((bond & Opts::Bond::H) == Opts::Bond::H) {
                    // out.data_h(x, y) = data_h(x, y).template shiftQN<0, 2>(charges(x, y)).template shiftQN<1, 3>(charges(x + 1, y));
                    // out.data_h(x, y) =
                    //     data_h(x, y).template shiftQN<0, 1, 2, 3>(std::array{charges(x, y), charges(x + 1, y), charges(x, y), charges(x + 1, y)});
                    Qbasis<Symmetry, 1> c1;
                    c1.push_back(charges(x, y), 1);
                    Qbasis<Symmetry, 1> c2;
                    c2.push_back(charges(x + 1, y), 1);
                    Tensor<Scalar, 2, 2, Symmetry> id({{c1, c2}}, {{c1, c2}});
                    id.setIdentity();
                    auto tmp = data_h(x, y).template contract<std::array{-1, -3, -5, -7}, std::array{-2, -4, -6, -8}, 4>(id);
                    Tensor<Scalar, 2, 1, Symmetry> fuser1({{data_h(x, y).uncoupledCodomain()[0], c1}},
                                                          {{data_h(x, y).uncoupledCodomain()[0].combine(c1).forgetHistory()}});
                    fuser1.setIdentity();
                    Tensor<Scalar, 2, 1, Symmetry> fuser2({{data_h(x, y).uncoupledCodomain()[1], c2}},
                                                          {{data_h(x, y).uncoupledCodomain()[1].combine(c2).forgetHistory()}});
                    fuser2.setIdentity();
                    out.data_h(x, y) =
                        tmp.template contract<std::array{-1, -2, -3, -4, 1, 2, -6, -7}, std::array{1, 2, -5}, 4>(fuser1)
                            .template contract<std::array{-1, -2, -3, -4, -5, 1, 2}, std::array{1, 2, -6}, 4>(fuser2)
                            .template contract<std::array{1, 2, -2, -3, -4, -5}, std::array{-1, 1, 2}, 3>(fuser1.adjoint().eval().twist(1).twist(2))
                            .template contract<std::array{-1, 1, 2, -3, -4}, std::array{-2, 1, 2}, 2>(fuser2.adjoint().eval().twist(1).twist(2));
                }
                if((bond & Opts::Bond::V) == Opts::Bond::V) {
                    // out.data_v(x, y) = data_v(x, y).template shiftQN<0, 2>(charges(x, y)).template shiftQN<1, 3>(charges(x, y + 1));
                    // out.data_v(x, y) =
                    //     data_v(x, y).template shiftQN<0, 1, 2, 3>(std::array{charges(x, y), charges(x, y + 1), charges(x, y), charges(x, y + 1)});
                    Qbasis<Symmetry, 1> c1;
                    c1.push_back(charges(x, y), 1);
                    Qbasis<Symmetry, 1> c2;
                    c2.push_back(charges(x, y + 1), 1);
                    Tensor<Scalar, 2, 2, Symmetry> id({{c1, c2}}, {{c1, c2}});
                    id.setIdentity();
                    auto tmp = data_v(x, y).template contract<std::array{-1, -3, -5, -7}, std::array{-2, -4, -6, -8}, 4>(id);
                    Tensor<Scalar, 2, 1, Symmetry> fuser1({{data_v(x, y).uncoupledCodomain()[0], c1}},
                                                          {{data_v(x, y).uncoupledCodomain()[0].combine(c1).forgetHistory()}});
                    fuser1.setIdentity();
                    Tensor<Scalar, 2, 1, Symmetry> fuser2({{data_v(x, y).uncoupledCodomain()[1], c2}},
                                                          {{data_v(x, y).uncoupledCodomain()[1].combine(c2).forgetHistory()}});
                    fuser2.setIdentity();
                    out.data_v(x, y) =
                        tmp.template contract<std::array{-1, -2, -3, -4, 1, 2, -6, -7}, std::array{1, 2, -5}, 4>(fuser1)
                            .template contract<std::array{-1, -2, -3, -4, -5, 1, 2}, std::array{1, 2, -6}, 4>(fuser2)
                            .template contract<std::array{1, 2, -2, -3, -4, -5}, std::array{-1, 1, 2}, 3>(fuser1.adjoint().eval().twist(1).twist(2))
                            .template contract<std::array{-1, 1, 2, -3, -4}, std::array{-2, 1, 2}, 2>(fuser2.adjoint().eval().twist(1).twist(2));
                }
                if((bond & Opts::Bond::D1) == Opts::Bond::D1) {
                    // out.data_d1(x,y) = data_d1(x, y).template shiftQN<0, 1, 2, 3>(
                    //     std::array{charges(x, y), charges(x + 1, y + 1), charges(x, y), charges(x + 1, y + 1)});
                    Qbasis<Symmetry, 1> c1;
                    c1.push_back(charges(x, y), 1);
                    Qbasis<Symmetry, 1> c2;
                    c2.push_back(charges(x + 1, y + 1), 1);
                    Tensor<Scalar, 2, 2, Symmetry> id({{c1, c2}}, {{c1, c2}});
                    id.setIdentity();
                    auto tmp = data_d1(x, y).template contract<std::array{-1, -3, -5, -7}, std::array{-2, -4, -6, -8}, 4>(id);
                    Tensor<Scalar, 2, 1, Symmetry> fuser1({{data_d1(x, y).uncoupledCodomain()[0], c1}},
                                                          {{data_d1(x, y).uncoupledCodomain()[0].combine(c1).forgetHistory()}});
                    fuser1.setIdentity();
                    Tensor<Scalar, 2, 1, Symmetry> fuser2({{data_d1(x, y).uncoupledCodomain()[1], c2}},
                                                          {{data_d1(x, y).uncoupledCodomain()[1].combine(c2).forgetHistory()}});
                    fuser2.setIdentity();
                    out.data_d1(x, y) =
                        tmp.template contract<std::array{-1, -2, -3, -4, 1, 2, -6, -7}, std::array{1, 2, -5}, 4>(fuser1)
                            .template contract<std::array{-1, -2, -3, -4, -5, 1, 2}, std::array{1, 2, -6}, 4>(fuser2)
                            .template contract<std::array{1, 2, -2, -3, -4, -5}, std::array{-1, 1, 2}, 3>(fuser1.adjoint().eval().twist(1).twist(2))
                            .template contract<std::array{-1, 1, 2, -3, -4}, std::array{-2, 1, 2}, 2>(fuser2.adjoint().eval().twist(1).twist(2));
                }
                if((bond & Opts::Bond::D2) == Opts::Bond::D2) {
                    // out.data_d2(x, y) = data_d2(x, y).template shiftQN<0, 1, 2, 3>(
                    //     std::array{charges(x, y), charges(x - 1, y + 1), charges(x, y), charges(x - 1, y + 1)});
                    Qbasis<Symmetry, 1> c1;
                    c1.push_back(charges(x, y), 1);
                    Qbasis<Symmetry, 1> c2;
                    c2.push_back(charges(x - 1, y + 1), 1);
                    Tensor<Scalar, 2, 2, Symmetry> id({{c1, c2}}, {{c1, c2}});
                    id.setIdentity();
                    auto tmp = data_d2(x, y).template contract<std::array{-1, -3, -5, -7}, std::array{-2, -4, -6, -8}, 4>(id);
                    Tensor<Scalar, 2, 1, Symmetry> fuser1({{data_d2(x, y).uncoupledCodomain()[0], c1}},
                                                          {{data_d2(x, y).uncoupledCodomain()[0].combine(c1).forgetHistory()}});
                    fuser1.setIdentity();
                    Tensor<Scalar, 2, 1, Symmetry> fuser2({{data_d2(x, y).uncoupledCodomain()[1], c2}},
                                                          {{data_d2(x, y).uncoupledCodomain()[1].combine(c2).forgetHistory()}});
                    fuser2.setIdentity();
                    out.data_d2(x, y) =
                        tmp.template contract<std::array{-1, -2, -3, -4, 1, 2, -6, -7}, std::array{1, 2, -5}, 4>(fuser1)
                            .template contract<std::array{-1, -2, -3, -4, -5, 1, 2}, std::array{1, 2, -6}, 4>(fuser2)
                            .template contract<std::array{1, 2, -2, -3, -4, -5}, std::array{-1, 1, 2}, 3>(fuser1.adjoint().eval().twist(1).twist(2))
                            .template contract<std::array{-1, 1, 2, -3, -4}, std::array{-2, 1, 2}, 2>(fuser2.adjoint().eval().twist(1).twist(2));
                }
            }
        }
        return out;
    }

    void loadFromMatlab(const std::filesystem::path& p, const std::string& root_name, int qn_scale = 1)
    {
        HighFive::File file(p.string(), HighFive::File::ReadOnly);
        auto root = file.getGroup(root_name);

        UnitCell cell;
        cell.loadFromMatlab(p, root_name);

        // if((bond & Opts::Bond::H) == Opts::Bond::H) {
        data_h = TMatrix<Tensor<Scalar, 2, 2, Symmetry, false>>(cell.pattern);
        obs_h = TMatrix<ObsScalar>(cell.pattern);
        // }
        // if((bond & Opts::Bond::V) == Opts::Bond::V) {
        data_v = TMatrix<Tensor<Scalar, 2, 2, Symmetry, false>>(cell.pattern);
        obs_v = TMatrix<ObsScalar>(cell.pattern);
        // }
        // if((bond & Opts::Bond::D1) == Opts::Bond::D1) {
        //     data_d1 = TMatrix<Tensor<double, 2, 2, Symmetry, false>>(pat);
        //     obs_d1 = TMatrix<double>(pat);
        // }
        // if((bond & Opts::Bond::D2) == Opts::Bond::D2) {
        //     data_d2 = TMatrix<Tensor<double, 2, 2, Symmetry, false>>(pat);
        //     obs_d2 = TMatrix<double>(pat);
        // }

        for(std::size_t i = 0; i < cell.pattern.uniqueSize(); ++i) {
            auto H_ref = root.getDataSet("H");
            std::vector<HighFive::Reference> H;
            H_ref.read(H);
            auto g_Hh = H[2 * i].template dereference<HighFive::Group>(root);
            auto g_Hv = H[2 * i + 1].template dereference<HighFive::Group>(root);
            data_h[i] =
                Xped::IO::loadMatlabTensor<Scalar, 4, 0, Symmetry, Xped::HeapPolicy>(g_Hh, root, std::array{false, false, true, true}, qn_scale)
                    .template permute<2, 0, 1, 2, 3>();
            data_v[i] =
                Xped::IO::loadMatlabTensor<Scalar, 4, 0, Symmetry, Xped::HeapPolicy>(g_Hv, root, std::array{false, false, true, true}, qn_scale)
                    .template permute<2, 0, 1, 2, 3>();
        }
        fmt::print("Loaded hamiltonian from matlab.\n");
    }

    virtual std::string getResString(const std::string& offset) const override
    {
        std::string res;
        if((bond & Opts::Bond::H) == Opts::Bond::H) {
            fmt::format_to(std::back_inserter(res),
                           "{}{:<10}: avg_h={:+.2f}, vals_h={::+.4f}\n",
                           offset,
                           this->name,
                           obs_h.sum() / static_cast<ObsScalar>(obs_h.size()),
                           obs_h.uncompressedVector());
        }
        if((bond & Opts::Bond::V) == Opts::Bond::V) {
            fmt::format_to(std::back_inserter(res),
                           "{}{:<10}: avg_v={:+.2f}, vals_v={::+.4f}\n",
                           offset,
                           this->name,
                           obs_v.sum() / static_cast<ObsScalar>(obs_v.size()),
                           obs_v.uncompressedVector());
        }
        if((bond & Opts::Bond::D1) == Opts::Bond::D1) {
            fmt::format_to(std::back_inserter(res),
                           "{}{:<10}: avg_d1={:+.2f}, vals_d1={::+.4f}\n",
                           offset,
                           this->name,
                           obs_d1.sum() / static_cast<ObsScalar>(obs_d1.size()),
                           obs_d1.uncompressedVector());
        }
        if((bond & Opts::Bond::D2) == Opts::Bond::D2) {
            fmt::format_to(std::back_inserter(res),
                           "{}{:<10}: avg_d2={:+.2f}, vals_d2={::+.4f}",
                           offset,
                           this->name,
                           obs_d2.sum() / static_cast<ObsScalar>(obs_d2.size()),
                           obs_d2.uncompressedVector());
        }
        return res;
    }

    virtual void toFile(HighFive::File& file, const std::string& root = "/") const override
    {
        auto write_component = [&file, &root](std::string name, const auto& o) {
            if(not file.exist(root + name)) {
                HighFive::DataSpace dataspace = HighFive::DataSpace({0, 0}, {HighFive::DataSpace::UNLIMITED, HighFive::DataSpace::UNLIMITED});

                // Use chunking
                HighFive::DataSetCreateProps props;
                props.add(HighFive::Chunking(std::vector<hsize_t>{2, 2}));

                // Create the dataset
                HighFive::DataSet dataset = file.createDataSet(root + name, dataspace, HighFive::create_datatype<ObsScalar>(), props);
            }
            auto d = file.getDataSet(root + name);
            std::vector<std::vector<ObsScalar>> data;
            data.push_back(o.uncompressedVector());
            std::size_t curr_size = d.getDimensions()[0];
            d.resize({curr_size + 1, data[0].size()});
            d.select({curr_size, 0}, {1, data[0].size()}).write(data);
        };
        if((bond & Opts::Bond::H) == Opts::Bond::H) { write_component(this->name + "_h", obs_h); }
        if((bond & Opts::Bond::V) == Opts::Bond::V) { write_component(this->name + "_v", obs_v); }
        if((bond & Opts::Bond::D1) == Opts::Bond::D1) { write_component(this->name + "_d1", obs_d1); }
        if((bond & Opts::Bond::D2) == Opts::Bond::D2) { write_component(this->name + "_d2", obs_d2); }
    }

    template <typename Ar>
    void serialize(Ar& ar)
    {
        ar& YAS_OBJECT_NVP("TwoSiteobservable",
                           ("data_h", data_h),
                           ("data_v", data_v),
                           ("data_d1", data_d1),
                           ("data_d2", data_d2),
                           ("obs_h", obs_h),
                           ("obs_v", obs_v),
                           ("obs_d1", obs_d1),
                           ("obs_d2", obs_d2),
                           ("bond", bond));
    }

    TMatrix<Tensor<Scalar, 2, 2, Symmetry, false>> data_h;
    TMatrix<Tensor<Scalar, 2, 2, Symmetry, false>> data_v;
    TMatrix<Tensor<Scalar, 2, 2, Symmetry, false>> data_d1;
    TMatrix<Tensor<Scalar, 2, 2, Symmetry, false>> data_d2;
    TMatrix<ObsScalar> obs_h;
    TMatrix<ObsScalar> obs_v;
    TMatrix<ObsScalar> obs_d1;
    TMatrix<ObsScalar> obs_d2;
    Opts::Bond bond;
};

} // namespace Xped

#endif
