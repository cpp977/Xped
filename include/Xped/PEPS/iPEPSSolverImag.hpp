#ifndef XPED_IPEPS_SOLVER_IMAG_H_
#define XPED_IPEPS_SOLVER_IMAG_H_

#include "Xped/Util/Stopwatch.hpp"

#include "Xped/PEPS/CTMSolver.hpp"
#include "Xped/PEPS/ImagOpts.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry, typename HamScalar = double>
struct iPEPSSolverImag
{
    iPEPSSolverImag() = delete;

    iPEPSSolverImag(Opts::Imag imag_opts, Opts::CTM ctm_opts, std::shared_ptr<iPEPS<Scalar, Symmetry>> Psi_in, Hamiltonian<HamScalar, Symmetry>& H_in)
        : imag_opts(imag_opts)
        , Psi(Psi_in)
        , H(H_in)
    {
        std::filesystem::create_directories(imag_opts.working_directory / imag_opts.obs_directory);
        Jack = CTMSolver<Scalar, Symmetry, HamScalar>(ctm_opts);
        if(not imag_opts.obs_directory.empty()) {
            std::filesystem::create_directories(imag_opts.working_directory / imag_opts.obs_directory);
            if(std::filesystem::exists(imag_opts.working_directory / imag_opts.obs_directory /
                                       std::filesystem::path(this->H.file_name() + fmt::format("_seed={}_id={}.h5", imag_opts.seed, imag_opts.id)))) {
                HighFive::File file((imag_opts.working_directory / imag_opts.obs_directory).string() + "/" + this->H.file_name() +
                                        fmt::format("_seed={}_id={}.h5", imag_opts.seed, imag_opts.id),
                                    imag_opts.resume ? HighFive::File::ReadWrite : HighFive::File::ReadOnly);
                for(auto iD = 0ul; auto D : imag_opts.Ds) {
                    for(auto chi : imag_opts.chis[iD]) {
                        if(file.exist(fmt::format("/{}/{}", D, chi))) {
                            throw std::runtime_error(fmt::format(fg(fmt::color::red),
                                                                 "There already exists data in observable file for D={}, chi={}.\n Filename:{}.\n",
                                                                 D,
                                                                 chi,
                                                                 (imag_opts.working_directory / imag_opts.obs_directory).string() + "/" +
                                                                     this->H.file_name() +
                                                                     fmt::format("_seed={}_id={}.h5", imag_opts.seed, imag_opts.id)));
                        }
                        ++iD;
                    }
                }
            } else {
                try {
                    HighFive::File file((imag_opts.working_directory / imag_opts.obs_directory).string() + "/" + this->H.file_name() +
                                            fmt::format("_seed={}_id={}.h5", imag_opts.seed, imag_opts.id),
                                        imag_opts.resume ? HighFive::File::ReadWrite : HighFive::File::Excl);
                } catch(const std::exception& e) {
                    throw std::runtime_error(fmt::format(fg(fmt::color::red),
                                                         "Error while creating the observable file for this simulation:{}.\n",
                                                         (imag_opts.working_directory / imag_opts.obs_directory).string() + "/" +
                                                             this->H.file_name() + fmt::format("_seed={}_id={}.h5", imag_opts.seed, imag_opts.id)));
                }
            }
        }
    }

    void solve()
    {
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

        init_psi();

        std::vector<std::vector<double>> Es(imag_opts.chis.size());
        for(auto iD = 0ul; iD < Es.size(); ++iD) { Es[iD].resize(imag_opts.chis[iD].size()); }
        util::Stopwatch<> total_t;
        std::chrono::seconds evol_time{0}, ctm_time{0};
        for(auto iD = 0; auto D : imag_opts.Ds) {
            util::Stopwatch<> evol_t;
            Psi->D = D;
            for(std::size_t i = 0; i < imag_opts.t_steps.size(); ++i) {
                util::Stopwatch<> step_t;
                TMatrix<Tensor<Scalar, 1, 1, Symmetry>> conv_h;
                TMatrix<Tensor<Scalar, 1, 1, Symmetry>> conv_v;
                double diff = 0.;
                std::size_t steps = 0ul;
                TimePropagator<Scalar, double, HamScalar, Symmetry> Jim(H, imag_opts.dts[i], imag_opts.update, Psi->charges());
                for(auto step = 0ul; step < imag_opts.t_steps[i]; ++step) {
                    ++steps;
                    Jim.t_step(*Psi);
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
                                diff_v[i] = std::nan("1");
                            } else {
                                diff_v[i] = (conv_v[i] - Jim.spectrum_v[i]).norm();
                            }
                        }
                        diff = std::max(*std::max_element(diff_h.begin(), diff_h.end()), *std::max_element(diff_v.begin(), diff_v.end()));
                        if(diff < imag_opts.tol * imag_opts.dts[i] * imag_opts.dts[i] and step > imag_opts.min_steps) { break; }
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
                Psi->updateAtensors();
                constexpr std::size_t flags = yas::file /*IO type*/ | yas::binary; /*IO format*/
                yas::file_ostream ofs(
                    (imag_opts.working_directory.string() + "/" + H.file_name() + fmt::format("_D={}_id={}.psi", D, imag_opts.id)).c_str(),
                    /*trunc*/ 1);
                yas::save<flags>(ofs, *Psi);
                // Psi->info();
            }
            Log::per_iteration(imag_opts.verbosity, "  {}", Psi->info());
            evol_time += evol_t.time();
            util::Stopwatch<> ctm_t;
            for(auto ichi = 0; auto chi : imag_opts.chis[iD]) {
                Jack.opts.chi = chi;
                Es[iD][ichi] = Jack.template solve<false>(Psi, nullptr, H);
                Jack.REINIT_ENV = false;
                if(imag_opts.display_obs or not imag_opts.obs_directory.empty()) {
                    for(auto& ob : H.obs) {
                        if(auto* one = dynamic_cast<OneSiteObservable<double, Symmetry>*>(ob.get()); one != nullptr) { avg(Jack.getCTM(), *one); }
                        if(auto* one_c = dynamic_cast<OneSiteObservable<std::complex<double>, Symmetry>*>(ob.get()); one_c != nullptr) {
                            avg(Jack.getCTM(), *one_c);
                        }
                        if(auto* two = dynamic_cast<TwoSiteObservable<double, Symmetry>*>(ob.get()); two != nullptr) { avg(Jack.getCTM(), *two); }
                        if(auto* two_c = dynamic_cast<TwoSiteObservable<std::complex<double>, Symmetry>*>(ob.get()); two_c != nullptr) {
                            avg(Jack.getCTM(), *two_c);
                        }
                    }
                }
                if(imag_opts.display_obs) { Log::per_iteration(imag_opts.verbosity, "  Observables:\n{}", H.getObsString("    ")); }
                if(not imag_opts.obs_directory.empty()) {
                    std::string e_name = fmt::format("/{}/{}/energy", D, chi);
                    HighFive::File file((imag_opts.working_directory / imag_opts.obs_directory).string() + "/" + H.file_name() +
                                            fmt::format("_seed={}_id={}.h5", imag_opts.seed, imag_opts.id),
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
                if(Jack.opts.EXPORT_CTM_SPECTRA) {
                    Jack.getCTM().save_spectrum((imag_opts.working_directory / imag_opts.obs_directory).string() + "/" + H.file_name() +
                                                fmt::format("_seed={}_id={}.h5", imag_opts.seed, imag_opts.id));
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

    void init_psi()
    {
        if(imag_opts.load != "") {
            std::filesystem::path load_p(imag_opts.load);
            if(load_p.is_relative()) { load_p = imag_opts.working_directory / load_p; }
            switch(imag_opts.load_format) {
            case Opts::LoadFormat::MATLAB: {
                Log::on_entry(imag_opts.verbosity,
                              "Load initial iPEPS from matlab file {}.\nScale the quantum numbers from matlab by {}.",
                              imag_opts.load,
                              imag_opts.qn_scale);
                Psi->loadFromMatlab(load_p, "cpp", imag_opts.qn_scale);
                break;
            }
            case Opts::LoadFormat::JSON: {
                if constexpr(not Symmetry::ANY_IS_FERMIONIC) {
                    Psi->loadFromJson(load_p);
                    Psi->debug_info();
                } else {
                    throw std::invalid_argument("Cannot load from JSON format for fermionic symmetries.");
                }
                break;
            }
            case Opts::LoadFormat::JSON_SU2: {
                if constexpr(not Symmetry::ANY_IS_FERMIONIC) {
                    Psi->loadFromJsonSU2(load_p);
                    Psi->debug_info();
                } else {
                    throw std::invalid_argument("Cannot load from JSON format for fermionic symmetries.");
                }
                break;
            }
            case Opts::LoadFormat::Native: {
                Log::on_entry(imag_opts.verbosity, "Load initial iPEPS from native file {}.", load_p.string());
                constexpr std::size_t flags = yas::file /*IO type*/ | yas::binary; /*IO format*/
                iPEPS<Scalar, Symmetry> tmp_Psi;
                try {
                    yas::load<flags>(load_p.string().c_str(), tmp_Psi);
                } catch(const yas::serialization_exception& se) {
                    throw std::runtime_error(fmt::format(
                        "Error while deserializing file ({}) with initial wavefunction.\nThis might be because of incompatible symmetries between this simulation and the loaded wavefunction.",
                        load_p.string()));
                } catch(const yas::io_exception& ie) {
                    throw std::runtime_error(fmt::format("Error while loading file ({}) with initial wavefunction.\n", load_p.string()));
                } catch(const std::exception& e) {
                    throw std::runtime_error(fmt::format("Unknown error while loading file ({}) with initial wavefunction.\n", load_p.string()));
                }
                Psi = std::make_shared<iPEPS<Scalar, Symmetry>>(std::move(tmp_Psi));
                break;
            }
            }
            Psi->initWeightTensors();
            assert(Psi->cell().pattern == H.data_h.pat);
        } else {
            if(imag_opts.multi_init) {
                Log::on_entry(imag_opts.verbosity, "Try multiple seeds to find a good random initial state.");
                Log::on_entry(imag_opts.verbosity, "Used protocol:");
                Log::on_entry(imag_opts.verbosity, "  {:<10} {}", "• seeds:", imag_opts.init_seeds);
                Log::on_entry(imag_opts.verbosity, "  {:<10} {}", "• Ds:", imag_opts.init_Ds);
                Log::on_entry(imag_opts.verbosity, "  {:<10} {}", "• chi:", imag_opts.init_chi);
                Log::on_entry(imag_opts.verbosity, "  {:<10} {}", "• t_steps:", imag_opts.init_t_steps);
                Log::on_entry(imag_opts.verbosity, "  {:<10} {}", "• dts:", imag_opts.init_dts);
                std::vector<std::shared_ptr<iPEPS<Scalar, Symmetry>>> init_Psis(imag_opts.init_seeds.size());
                for(auto& init_Psi : init_Psis) { init_Psi = std::make_shared<iPEPS<Scalar, Symmetry>>(*Psi); }
                std::vector<double> init_Es(imag_opts.init_seeds.size(), 0.);
                for(auto i = 0ul; i < imag_opts.init_seeds.size(); ++i) {
                    init_Psis[i]->setRandom(imag_opts.init_seeds[i]);
                    init_Psis[i]->initWeightTensors();
                    for(auto D : imag_opts.init_Ds) {
                        init_Psis[i]->D = D;
                        for(std::size_t j = 0; j < imag_opts.init_t_steps.size(); ++j) {
                            util::Stopwatch<> step_t;
                            TimePropagator<Scalar, double, HamScalar, Symmetry> Jim(
                                H, imag_opts.init_dts[j], imag_opts.update, init_Psis[i]->charges());
                            for(auto step = 0ul; step < imag_opts.init_t_steps[j]; ++step) { Jim.t_step(*init_Psis[i]); }
                            Log::per_iteration(imag_opts.verbosity,
                                               "  MultiInit(D={}, seed={}: Nτ={:^4d}, dτ={:.1e}): runtime={}",
                                               D,
                                               imag_opts.init_seeds[i],
                                               imag_opts.init_t_steps[j],
                                               imag_opts.init_dts[j],
                                               step_t.time_string());
                        }
                        init_Psis[i]->updateAtensors();
                        Log::per_iteration(imag_opts.verbosity, "  {}", init_Psis[i]->info());
                    }
                    Jack.opts.chi = imag_opts.init_chi;
                    init_Es[i] = Jack.template solve<false>(init_Psis[i], nullptr, H);
                }
                std::size_t min_index = std::distance(init_Es.begin(), std::min_element(init_Es.begin(), init_Es.end()));
                Log::on_entry(imag_opts.verbosity, "  Initialization with #{} seeds:", imag_opts.init_seeds.size());
                for(auto i = 0ul; i < imag_opts.init_seeds.size(); ++i) {
                    Log::on_entry(imag_opts.verbosity, "    seed={:<3d} --> energy={:.8f}", imag_opts.init_seeds[i], init_Es[i]);
                }
                Log::on_entry(imag_opts.verbosity, "  Chose seed={:<3d} with energy={:.8f}", imag_opts.init_seeds[min_index], init_Es[min_index]);
                Psi = std::move(init_Psis[min_index]);
            } else {
                Log::on_entry(imag_opts.verbosity, "Start from random iPEPS with seed={}.", imag_opts.seed);
                Psi->setRandom(imag_opts.seed);
                Psi->initWeightTensors();
            }
        }
    }

    CTMSolver<Scalar, Symmetry, HamScalar> Jack;
    Opts::Imag imag_opts;
    std::shared_ptr<iPEPS<Scalar, Symmetry>> Psi;
    Hamiltonian<HamScalar, Symmetry>& H;
};

} // namespace Xped

#endif
