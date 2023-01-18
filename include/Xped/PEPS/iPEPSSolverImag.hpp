#ifndef XPED_IPEPS_SOLVER_IMAG_H_
#define XPED_IPEPS_SOLVER_IMAG_H_

#include "Xped/Util/Stopwatch.hpp"

#include "Xped/PEPS/CTMSolver.hpp"
#include "Xped/PEPS/ImagOpts.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry>
struct iPEPSSolverImag
{
    template <typename Sym>
    using Hamiltonian = TwoSiteObservable<Sym>;

    iPEPSSolverImag() = delete;

    iPEPSSolverImag(Opts::Imag imag_opts, Opts::CTM ctm_opts, std::shared_ptr<iPEPS<Scalar, Symmetry>> Psi_in, Hamiltonian<Symmetry>& H_in)
        : imag_opts(imag_opts)
        , Psi(Psi_in)
        , H(H_in)
    {
        std::filesystem::create_directories(imag_opts.working_directory / imag_opts.obs_directory);
        Jack = CTMSolver<Scalar, Symmetry>(ctm_opts);
        if(imag_opts.load != "") {
            switch(imag_opts.load_format) {
            case Opts::LoadFormat::MATLAB: {
                Psi->loadFromMatlab(std::filesystem::path(imag_opts.load), "cpp", imag_opts.qn_scale);
                break;
            }
            case Opts::LoadFormat::NATIVE: {
                constexpr std::size_t flags = yas::file /*IO type*/ | yas::binary; /*IO format*/
                iPEPS<Scalar, Symmetry> tmp_Psi;
                try {
                    yas::load<flags>((imag_opts.working_directory.string() + "/" + imag_opts.load).c_str(), tmp_Psi);
                } catch(const yas::serialization_exception& se) {
                    fmt::print(
                        "Error while deserializing file ({}) with initial wavefunction.\nThis might be because of incompatible symmetries between this simulation and the loaded wavefunction.",
                        imag_opts.working_directory.string() + "/" + imag_opts.load);
                    std::cout << std::flush;
                    throw;
                } catch(const yas::io_exception& ie) {
                    fmt::print("Error while loading file ({}) with initial wavefunction.\n",
                               imag_opts.working_directory.string() + "/" + imag_opts.load);
                    std::cout << std::flush;
                    throw;
                } catch(const std::exception& e) {
                    fmt::print("Unknown error while loading file ({}) with initial wavefunction.\n",
                               imag_opts.working_directory.string() + "/" + imag_opts.load);
                    std::cout << std::flush;
                    throw;
                }
                Psi = std::make_shared<iPEPS<Scalar, Symmetry>>(std::move(tmp_Psi));
                break;
            }
            }
            assert(Psi->cell().pattern == H.data_h.pat);
        }
        if(not imag_opts.obs_directory.empty()) {
            std::filesystem::create_directories(imag_opts.working_directory / imag_opts.obs_directory);
            try {
                HighFive::File file((imag_opts.working_directory / imag_opts.obs_directory).string() + "/" + H.file_name() + ".h5",
                                    imag_opts.resume ? HighFive::File::ReadWrite : HighFive::File::Excl);
            } catch(const std::exception& e) {
                fmt::print(fg(fmt::color::red),
                           "There already exists an observable file for this simulation:{}.\n",
                           (imag_opts.working_directory / imag_opts.obs_directory).string() + "/" + this->H.file_name() + ".h5");
                std::cout << std::flush;
                throw;
            }
        }
    }

    void solve()
    {
        Xped::TimePropagator<Scalar, double, Symmetry> Jim(H, Psi, imag_opts.update);

        Log::on_entry(imag_opts.verbosity,
                      "{}: Model={}(Bonds: V:{}, H:{}, D1: {}, D2: {})",
                      fmt::styled("iPEPSSolverImag", fmt::emphasis::bold),
                      H.format(),
                      (H.bond & Opts::Bond::V) == Opts::Bond::V,
                      (H.bond & Opts::Bond::H) == Opts::Bond::H,
                      (H.bond & Opts::Bond::D1) == Opts::Bond::D1,
                      (H.bond & Opts::Bond::D2) == Opts::Bond::D2);
        Log::on_entry(imag_opts.verbosity, "{}", imag_opts.info());
        Log::on_entry(imag_opts.verbosity, "{}", Jack.opts.info());

        std::vector<std::vector<double>> Es = std::vector(imag_opts.chis.size(), std::vector(imag_opts.chis[0].size(), 0.));
        util::Stopwatch<> total_t;
        std::chrono::seconds evol_time{0}, ctm_time{0};
        for(auto iD = 0; auto D : imag_opts.Ds) {
            util::Stopwatch<> evol_t;
            Psi->D = D;
            for(std::size_t i = 0; i < imag_opts.t_steps.size(); ++i) {
                util::Stopwatch<> step_t;
                TMatrix<Tensor<Scalar, 1, 1, Symmetry>> conv_h;
                TMatrix<Tensor<Scalar, 1, 1, Symmetry>> conv_v;
                double diff;
                std::size_t steps = 0ul;
                for(auto step = 0ul; step < imag_opts.t_steps[i]; ++step) {
                    ++steps;
                    Jim.t_step(imag_opts.dts[i]);
                    if(step == 0) {
                        conv_h = Jim.spectrum_h;
                        conv_v = Jim.spectrum_v;
                    } else {
                        TMatrix<double> diff_h(Psi->cell().pattern);
                        TMatrix<double> diff_v(Psi->cell().pattern);
                        for(auto i = 0ul; i < conv_h.size(); ++i) {
                            if(conv_h[i].coupledDomain() != Jim.spectrum_h[i].coupledDomain() or
                               conv_h[i].coupledCodomain() != Jim.spectrum_h[i].coupledCodomain()) {
                                diff_h[i] = std::nan("1");
                            } else {
                                diff_h[i] = (conv_h[i] - Jim.spectrum_h[i]).norm();
                            }
                            if(conv_v[i].coupledDomain() != Jim.spectrum_v[i].coupledDomain() or
                               conv_v[i].coupledCodomain() != Jim.spectrum_v[i].coupledCodomain()) {
                                diff_h[i] = std::nan("1");
                            } else {
                                diff_v[i] = (conv_v[i] - Jim.spectrum_v[i]).norm();
                            }
                        }
                        diff = std::max(*std::max_element(diff_h.begin(), diff_h.end()), *std::max_element(diff_v.begin(), diff_v.end()));
                        if(diff < imag_opts.tol * imag_opts.dts[i] * imag_opts.dts[i]) { break; }
                        conv_h = Jim.spectrum_h;
                        conv_v = Jim.spectrum_v;
                    }
                }
                Log::per_iteration(imag_opts.verbosity,
                                   "  ImagSteps(D={}: Nτ={:^4d}, dτ={:.1e}): runtime={}, conv={:2.3g}",
                                   D,
                                   steps,
                                   imag_opts.dts[i],
                                   step_t.time_string(),
                                   diff);
                constexpr std::size_t flags = yas::file /*IO type*/ | yas::binary; /*IO format*/
                yas::file_ostream ofs((imag_opts.working_directory.string() + "/" + H.file_name() + fmt::format("_D{}.psi", D)).c_str(), /*trunc*/ 1);
                yas::save<flags>(ofs, *Psi);
                // Psi->info();
            }
            Log::per_iteration(imag_opts.verbosity, "  {}", Psi->info());
            evol_time += evol_t.time();
            util::Stopwatch<> ctm_t;
            for(auto ichi = 0; auto chi : imag_opts.chis[iD]) {
                Jack.opts.chi = chi;
                Es[iD][ichi] = Jack.template solve<double>(Psi, nullptr, H, false);

                if(imag_opts.display_obs or not imag_opts.obs_directory.empty()) { H.computeObs(Jack.getCTM()); }
                if(imag_opts.display_obs) { Log::per_iteration(imag_opts.verbosity, "  Observables:\n{}", H.getObsString("    ")); }
                if(not imag_opts.obs_directory.empty()) {
                    std::string e_name = fmt::format("/{}/{}/energy", D, chi);
                    HighFive::File file((imag_opts.working_directory / imag_opts.obs_directory).string() + "/" + H.file_name() + ".h5",
                                        HighFive::File::OpenOrCreate);
                    if(not file.exist(e_name)) {
                        HighFive::DataSpace dataspace = HighFive::DataSpace({0, 0}, {HighFive::DataSpace::UNLIMITED, HighFive::DataSpace::UNLIMITED});

                        // Use chunking
                        HighFive::DataSetCreateProps props;
                        props.add(HighFive::Chunking(std::vector<hsize_t>{2, 2}));

                        // Create the dataset
                        HighFive::DataSet dataset = file.createDataSet(e_name, dataspace, HighFive::create_datatype<double>(), props);
                    }
                    {
                        auto d = file.getDataSet(e_name);
                        std::vector<std::vector<double>> data;
                        data.push_back(std::vector<double>(1, Es[iD][ichi]));
                        std::size_t curr_size = d.getDimensions()[0];
                        d.resize({curr_size + 1, data[0].size()});
                        d.select({curr_size, 0}, {1, data[0].size()}).write(data);
                    }
                    H.obsToFile(file, fmt::format("/{}/{}/", D, chi));
                }
                ++ichi;
            }

            ctm_time += ctm_t.time();
            ++iD;
        }
        Log::on_exit(imag_opts.verbosity,
                     "{}(runtime={} [evol={}, ctm={}]) energies:",
                     fmt::styled("iPEPSSolverImag", fmt::emphasis::bold),
                     total_t.time_string(),
                     util::format_secs(evol_time),
                     util::format_secs(ctm_time));
        for(auto i = 0ul; i < Es.size(); ++i) {
            for(auto j = 0; j < Es[i].size(); ++j) {
                Log::on_exit(imag_opts.verbosity, "{:^4}D={:^2d}, χ={:^3d}: E={:.8f}", "↳", imag_opts.Ds[i], imag_opts.chis[i][j], Es[i][j]);
            }
        }
    }

    CTMSolver<Scalar, Symmetry> Jack;
    Opts::Imag imag_opts;
    std::shared_ptr<iPEPS<Scalar, Symmetry>> Psi;
    Hamiltonian<Symmetry>& H;
};

} // namespace Xped

#endif
