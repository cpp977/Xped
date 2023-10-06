#include <iostream>

#include <highfive/H5File.hpp>

#include <assert.hpp>

#include "Xped/PEPS/iPEPS.hpp"

#include "Xped/Util/Bool.hpp"

#include "Xped/Core/AdjointOp.hpp"

#include "Xped/IO/Json.hpp"
#include "Xped/IO/Matlab.hpp"

#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::iPEPS(const UnitCell& cell,
                                                        std::size_t D,
                                                        const Qbasis<Symmetry, 1>& auxBasis,
                                                        const Qbasis<Symmetry, 1>& physBasis,
                                                        Opts::DiscreteSym sym)
    : D(D)
    , cell_(cell)
    , sym_(sym)

{
    charges_ = TMatrix<qType>(cell_.pattern);
    charges_.setConstant(Symmetry::qvacuum());

    TMatrix<Qbasis<Symmetry, 1>> aux;
    aux.setConstant(auxBasis);

    TMatrix<Qbasis<Symmetry, 1>> phys;
    phys.setConstant(physBasis);

    init(aux, aux, phys);
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::iPEPS(const UnitCell& cell,
                                                        std::size_t D,
                                                        const TMatrix<Qbasis<Symmetry, 1>>& leftBasis,
                                                        const TMatrix<Qbasis<Symmetry, 1>>& topBasis,
                                                        const TMatrix<Qbasis<Symmetry, 1>>& physBasis,
                                                        Opts::DiscreteSym sym)
    : D(D)
    , cell_(cell)
    , sym_(sym)
{
    charges_ = TMatrix<qType>(cell_.pattern);
    charges_.setConstant(Symmetry::qvacuum());
    init(leftBasis, topBasis, physBasis);
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::iPEPS(const UnitCell& cell,
                                                        std::size_t D,
                                                        const TMatrix<Qbasis<Symmetry, 1>>& leftBasis,
                                                        const TMatrix<Qbasis<Symmetry, 1>>& topBasis,
                                                        const TMatrix<Qbasis<Symmetry, 1>>& physBasis,
                                                        const TMatrix<qType>& charges,
                                                        Opts::DiscreteSym sym)
    : D(D)
    , cell_(cell)
    , charges_(charges)
    , sym_(sym)
{
    init(leftBasis, topBasis, physBasis);
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
void iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::init(const TMatrix<Qbasis<Symmetry, 1>>& leftBasis,
                                                            const TMatrix<Qbasis<Symmetry, 1>>& topBasis,
                                                            const TMatrix<Qbasis<Symmetry, 1>>& physBasis)
{
    auto init_basis = [this](const Qbasis<Symmetry, 1>& phys_basis) {
        Qbasis<Symmetry, 1> out;
        std::size_t dim_per_charge = D < phys_basis.Nq() ? 1ul : D / phys_basis.Nq();
        std::size_t dim_for_vac = D < phys_basis.Nq() ? 1ul : dim_per_charge + D % phys_basis.Nq();
        if(phys_basis.Nq() == 1) {
            dim_per_charge = D / 2;
            dim_for_vac = D / 2 + D % 2;
        }
        // fmt::print("D={}, dim_p_c={}, dim_p_v={}\n", D, dim_per_charge, dim_for_vac);
        auto inserted_states = 0ul;
        auto phys_qs = phys_basis.unordered_qs();
        out.push_back(Symmetry::qvacuum(), dim_for_vac);
        inserted_states += dim_for_vac;
        if(inserted_states == D) { return out; }
        for(auto Q : Symmetry::lowest_qs()) {
            if(phys_qs.contains(Q)) {
                out.push_back(Q, dim_per_charge);
                inserted_states += dim_per_charge;
            }
            if(inserted_states == D) { return out; }
        }
        for(auto Q : phys_qs) {
            if(out.IS_PRESENT(Q)) { continue; }
            out.push_back(Q, dim_per_charge);
            inserted_states += dim_per_charge;
            if(inserted_states == D) { break; }
        }
        VERIFY(inserted_states == D, "Failed to initialize quantum numbers for iPEPS A-tensor.");

        out.sort();
        return out;
    };

    As.resize(cell().pattern);
    Adags.resize(cell().pattern);

    Ms.resize(cell().pattern);

    for(int x = 0; x < cell().Lx; x++) {
        for(int y = 0; y < cell().Ly; y++) {
            if(not cell().pattern.isUnique(x, y)) { continue; }
            // auto pos = cell().pattern.uniqueIndex(x, y);
            auto [dummy_xy, shifted_physBasis_xy] = physBasis(x, y).shift(charges_(x, y));
            auto [dummy_xp1y, shifted_physBasis_xp1y] = physBasis(x, y).shift(charges_(x + 1, y));
            auto [dummy_xyp1, shifted_physBasis_xyp1] = physBasis(x, y).shift(charges_(x, y + 1));
            auto left_basis_xy = leftBasis(x, y).dim() == 0 ? init_basis(shifted_physBasis_xy) : leftBasis(x, y);
            auto left_basis_xp1y = leftBasis(x + 1, y).dim() == 0 ? init_basis(shifted_physBasis_xp1y) : leftBasis(x + 1, y);
            auto top_basis_xy = topBasis(x, y).dim() == 0 ? init_basis(shifted_physBasis_xy) : topBasis(x, y);
            auto top_basis_xyp1 = topBasis(x, y + 1).dim() == 0 ? init_basis(shifted_physBasis_xyp1) : topBasis(x, y + 1);

            // fmt::print("x={}, y={}, original basis which will be shifted by {}:\n", x, y, Sym::format<Symmetry>(charges_[pos]));
            // std::cout << physBasis[pos] << std::endl;
            // std::cout << "shifted:\n" << shifted_physBasis << std::endl;
            if constexpr(ALL_OUT_LEGS) {
                As(x, y) = Tensor<Scalar, 4, 1, Symmetry, ENABLE_AD>({{left_basis_xy, top_basis_xy, left_basis_xp1y, top_basis_xyp1}},
                                                                     {{shifted_physBasis_xy}});
            } else {
                As(x, y) = Tensor<Scalar, 2, 3, Symmetry, ENABLE_AD>({{left_basis_xy, top_basis_xy}},
                                                                     {{left_basis_xp1y, top_basis_xyp1, shifted_physBasis_xy}});
            }
            // As[pos].setZero();
            // As(x, y).print(std::cout, false);
            // std::cout << std::endl;
            // std::cout << fmt::format("A({},{}): ", x, y) << As(x, y).coupledDomain() << std::endl << As(x, y).coupledCodomain() << std::endl;
            // fmt::print("{}\n{}\n", As(x, y).coupledDomain().printTrees(), As(x, y).coupledCodomain().printTrees());
            VERIFY(As(x, y).coupledDomain().forgetHistory().intersection(As(x, y).coupledCodomain().forgetHistory()).dim() > 0 and
                   "Bases of the A tensor have no fused blocks.");
        }
    }
    if(sym() == Opts::DiscreteSym::C4v) { VERIFY(cell().Lx == 1 and cell().Ly == 1); }
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
void iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::initSymMap()
{
    switch(sym()) {
    case Opts::DiscreteSym::None: {
        return;
    }
    case Opts::DiscreteSym::C4v: {
        VERIFY(cell().Lx == 1 and cell().Ly == 1);
        VERIFY(checkSym());
        VERIFY(ALL_OUT_LEGS);
        auto computeMap = [](auto A) {
            auto comp = [](Scalar s1, Scalar s2) {
                if(std::abs(s1 - s2) < 1.e-12) { return false; }
                if constexpr(ScalarTraits<Scalar>::IS_COMPLEX()) {
                    return std::array{s1.real(), s1.imag()} < std::array{s2.real(), s2.imag()};
                } else {
                    return s1 < s2;
                }
            };
            std::map<Scalar, std::size_t, decltype(comp)> unique(comp);
            // std::unordered_map<Scalar, std::size_t> unique;
            std::size_t count_tot = 0ul;
            std::size_t count_unique = 0ul;
            std::pair<std::size_t, std::vector<std::size_t>> res_map;
            res_map.second.resize(A.plainSize());
            for(auto it = A.begin(); it != A.end(); ++it) {
                if(auto it_unique = unique.find(*it); it_unique == unique.end()) {
                    res_map.second[count_tot] = count_unique;
                    unique.insert(std::make_pair(*it, count_unique));
                    ++count_unique;
                } else {
                    res_map.second[count_tot] = it_unique->second;
                }
                ++count_tot;
            }
            res_map.first = count_unique;
            fmt::print("#unique elements={}, res_map:\n{}\n", res_map.first, res_map.second);
            return res_map;
        };
        sym_map_A = computeMap(As[0]);
        return;
    }
    }
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
void iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::updateAdags()
{
    for(int x = 0; x < cell_.Lx; x++) {
        for(int y = 0; y < cell_.Ly; y++) {
            if(not cell_.pattern.isUnique(x, y)) { continue; }
            auto pos = cell_.pattern.uniqueIndex(x, y);
            if constexpr(ALL_OUT_LEGS) {
                Adags[pos] = As[pos].adjoint().eval().template permute<-4, 1, 2, 3, 4, 0>(Bool<ENABLE_AD>{});
            } else {
                Adags[pos] = As[pos].adjoint().eval().template permute<0, 3, 4, 2, 0, 1>(Bool<ENABLE_AD>{});
            }
        }
    }
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
void iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::loadFromMatlab(const std::filesystem::path& p, const std::string& root_name, int qn_scale)
{
    if constexpr(ALL_OUT_LEGS) {
        return;
    } else {
        HighFive::File file(p.string(), HighFive::File::ReadOnly);
        auto root = file.getGroup(root_name);

        cell_.loadFromMatlab(p, root_name);

        As.resize(cell().pattern);
        Adags.resize(cell().pattern);

        for(std::size_t i = 0; i < cell_.pattern.uniqueSize(); ++i) {
            auto A_ref = root.getDataSet("A");
            std::vector<HighFive::Reference> A;
            A_ref.read(A);
            auto g_A = A[i].template dereference<HighFive::Group>(root);
            As[i] =
                Xped::IO::loadMatlabTensor<double, 5, 0, Symmetry, Xped::HeapPolicy>(g_A, root, std::array{true, true, false, false, true}, qn_scale)
                    .twist(4)
                    .template permute<3, 3, 2, 1, 4, 0>();
        }
        updateAdags();
    }
    // for(int x = 0; x < cell().Lx; ++x) {
    //     for(int y = 0; y < cell().Ly; ++y) {
    //         fmt::print("Site: {},{}: \n", x, y);
    //         std::cout << "left: " << ketBasis(x, y, Opts::Leg::Left) << std::endl;
    //         std::cout << "top: " << ketBasis(x, y, Opts::Leg::Top) << std::endl;
    //         std::cout << "right: " << ketBasis(x, y, Opts::Leg::Right) << std::endl;
    //         std::cout << "bottom: " << ketBasis(x, y, Opts::Leg::Bottom) << std::endl;
    //         std::cout << "phys: " << ketBasis(x, y, Opts::Leg::Phys) << std::endl << std::endl;
    //     }
    // }
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
void iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::loadFromJson(const std::filesystem::path& p)
{
    if constexpr(ALL_OUT_LEGS) {
        cell_ = UnitCell(1, 1);
        As.resize(cell().pattern);
        Adags.resize(cell().pattern);
        // As[0] = Xped::IO::loadSU2JsonTensor<Symmetry>(p);
        As[0] = Xped::IO::loadU0JsonTensor<Symmetry>(p);
        updateAdags();
        auto check = checkSym();
        Log::debug("Symmetry check after loadFromJson(): {}", check);
        debug_info();
        return;
    } else {
        return;
    }
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::iPEPS(const iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, false>& other)
{
    D = other.D;
    cell_ = other.cell();
    sym_ = other.sym();
    charges_ = other.charges();
    As = other.As;
    Adags = other.Adags;
    Ms = other.Ms;
    sym_map_A = other.sym_map_A;
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
bool iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::checkConsistency() const
{
    bool out = true;
    for(int x = 0; x < cell().Lx; ++x) {
        for(int y = 0; y < cell().Ly; ++y) {
            if(ketBasis(x, y, Opts::Leg::Left) != ketBasis(x - 1, y, Opts::Leg::Right)) {
                fmt::print("Site ({},{}): left basis does not match other right basis.\n", x, y);
                out = false;
            }
            if(ketBasis(x, y, Opts::Leg::Right) != ketBasis(x + 1, y, Opts::Leg::Left)) {
                fmt::print("Site ({},{}): right basis does not match other left basis.\n", x, y);
                out = false;
            }
            if(ketBasis(x, y, Opts::Leg::Top) != ketBasis(x, y - 1, Opts::Leg::Bottom)) {
                fmt::print("Site ({},{}): up basis does not match other down basis.\n", x, y);
                out = false;
            }
            if(ketBasis(x, y, Opts::Leg::Bottom) != ketBasis(x, y + 1, Opts::Leg::Top)) {
                fmt::print("Site ({},{}): down basis does not match other up basis.\n", x, y);
                out = false;
            }
        }
    }
    return out;
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
void iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::setZero()
{
    for(int x = 0; x < cell_.Lx; x++) {
        for(int y = 0; y < cell_.Ly; y++) {
            if(not cell_.pattern.isUnique(x, y)) { continue; }
            auto pos = cell_.pattern.uniqueIndex(x, y);
            As[pos].setZero();
            // Adags[pos] = As[pos].adjoint().eval().template permute<0, 3, 4, 2, 0, 1>();
        }
    }
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
void iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::setRandom(std::size_t seed)
{
    static thread_local std::mt19937 engine(std::random_device{}());
    engine.seed(seed);
    for(int x = 0; x < cell_.Lx; x++) {
        for(int y = 0; y < cell_.Ly; y++) {
            if(not cell_.pattern.isUnique(x, y)) { continue; }
            auto pos = cell_.pattern.uniqueIndex(x, y);
            As[pos].setRandom(engine);
            if(sym() == Opts::DiscreteSym::C4v) {
                As[pos] = 0.5 * (As[pos] + As[pos].template permute<0, 0, 3, 2, 1, 4>()); // U-D reflection
                As[pos] = 0.5 * (As[pos] + As[pos].template permute<0, 2, 1, 0, 3, 4>()); // L-R reflection
                As[pos] = 0.5 * (As[pos] + As[pos].template permute<0, 1, 2, 3, 0, 4>()); // 90deg CCW rotation
                As[pos] = 0.5 * (As[pos] + As[pos].template permute<0, 3, 0, 1, 2, 4>()); // 90deg CW rotation
            }
        }
    }
    updateAdags();
    initSymMap();
    // debug_info();
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
bool iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::checkSym() const
{
    switch(sym()) {
    case Opts::DiscreteSym::None: {
        return true;
    }
    case Opts::DiscreteSym::C4v: {
        if((As[0] - As[0].template permute<0, 0, 3, 2, 1, 4>()).norm() > 1.e-10) { return false; } // U-D reflection
        if((As[0] - As[0].template permute<0, 2, 1, 0, 3, 4>()).norm() > 1.e-10) { return false; } // L-R reflection
        if((As[0] - As[0].template permute<0, 1, 2, 3, 0, 4>()).norm() > 1.e-10) { return false; } // 90deg CCW rotation
        if((As[0] - As[0].template permute<0, 3, 0, 1, 2, 4>()).norm() > 1.e-10) { return false; } // 90deg CW rotation
    }
    }
    return true;
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
void iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::normalize()
{
    for(auto& A : As) {
        auto tmp = (A * (1. / A.maxNorm())).eval();
        A = tmp;
    }
    updateAdags();
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
std::size_t iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::plainSize() const
{
    std::size_t res = 0;
    switch(sym()) {
    case Opts::DiscreteSym::None: {
        for(auto it = As.cbegin(); it != As.cend(); ++it) { res += it->plainSize(); }
        break;
    }
    case Opts::DiscreteSym::C4v: {
        res = sym_map_A.first;
    }
    }
    return res;
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
std::vector<Scalar> iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::data()
{
    std::vector<Scalar> out(plainSize());
    std::size_t count = 0;
    switch(sym()) {
    case Opts::DiscreteSym::None: {
        for(auto it = beginA(); it != endA(); ++it) { out[count++] = *it; }
        break;
    }
    case Opts::DiscreteSym::C4v: {
        std::vector<Scalar> full_data_A(sym_map_A.second.size());
        for(auto it = beginA(); it != endA(); ++it) { full_data_A[count++] = *it; }
        for(auto i = 0ul; i < full_data_A.size(); ++i) { out[sym_map_A.second[i]] = full_data_A[i]; }
        break;
    }
    }
    return out;
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
std::vector<Scalar> iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::graddata()
{
    VERIFY(ENABLE_AD);
    std::vector<Scalar> out(plainSize());
    std::size_t count = 0;
    switch(sym()) {
    case Opts::DiscreteSym::None: {
        for(auto it = gradbeginA(); it != gradendA(); ++it) { out[count++] = *it; }
        break;
    }
    case Opts::DiscreteSym::C4v: {
        std::vector<Scalar> full_data_A(sym_map_A.second.size());
        for(auto it = gradbeginA(); it != gradendA(); ++it) { full_data_A[count++] = *it; }
        for(auto i = 0ul; i < full_data_A.size(); ++i) { out[sym_map_A.second[i]] = full_data_A[i]; }
        break;
    }
    }
    return out;
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
void iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::set_data(const Scalar* data, bool NORMALIZE)
{
    switch(sym()) {
    case Opts::DiscreteSym::None: {
        std::size_t count = 0;
        for(int x = 0; x < cell_.Lx; x++) {
            for(int y = 0; y < cell_.Ly; y++) {
                if(not cell_.pattern.isUnique(x, y)) { continue; }
                As(x, y).set_data(data + count, As(x, y).plainSize());
                count += As(x, y).plainSize();
            }
        }
        break;
    }
    case Opts::DiscreteSym::C4v: {
        std::vector<Scalar> full_data_A(sym_map_A.second.size());
        for(auto i = 0ul; i < full_data_A.size(); ++i) { full_data_A[i] = data[sym_map_A.second[i]]; }
        As[0].set_data(full_data_A.data(), As[0].plainSize());
        break;
    }
    }
    updateAdags();
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
std::tuple<std::size_t, std::size_t, double, double> iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::calc_Ds() const
{
    std::vector<std::size_t> allDs;
    allDs.reserve(4 * As.size());
    for(auto i = 0ul; i < As.size(); ++i) {
        auto [x, y] = cell().pattern.coords(i);
        allDs.push_back(ketBasis(x, y, Opts::Leg::Left).fullDim());
        allDs.push_back(ketBasis(x, y, Opts::Leg::Top).fullDim());
        allDs.push_back(ketBasis(x, y, Opts::Leg::Right).fullDim());
        allDs.push_back(ketBasis(x, y, Opts::Leg::Bottom).fullDim());
    }
    auto Dmin = *std::min_element(allDs.begin(), allDs.end());
    auto Dmax = *std::max_element(allDs.begin(), allDs.end());
    double Dsum = std::accumulate(allDs.begin(), allDs.end(), 0.0);
    double Dmean = Dsum / allDs.size();

    std::vector<double> diff(allDs.size());
    std::transform(allDs.begin(), allDs.end(), diff.begin(), [Dmean](double x) { return x - Dmean; });
    double Dsq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    double Dsigma = std::sqrt(Dsq_sum / allDs.size());

    return std::make_tuple(Dmin, Dmax, Dmean, Dsigma);
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
Qbasis<Symmetry, 1> iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::ketBasis(const int x, const int y, const Opts::Leg leg) const
{
    if constexpr(ALL_OUT_LEGS) {
        switch(leg) {
        case Opts::Leg::Left: return As(x, y).uncoupledDomain()[0]; break;
        case Opts::Leg::Top: return As(x, y).uncoupledDomain()[1]; break;
        case Opts::Leg::Right: return As(x, y).uncoupledDomain()[2]; break;
        case Opts::Leg::Bottom: return As(x, y).uncoupledDomain()[3]; break;
        case Opts::Leg::Phys: return As(x, y).uncoupledCodomain()[0]; break;
        default: std::terminate();
        }
    } else {
        switch(leg) {
        case Opts::Leg::Left: return As(x, y).uncoupledDomain()[0]; break;
        case Opts::Leg::Top: return As(x, y).uncoupledDomain()[1]; break;
        case Opts::Leg::Right: return As(x, y).uncoupledCodomain()[0]; break;
        case Opts::Leg::Bottom: return As(x, y).uncoupledCodomain()[1]; break;
        case Opts::Leg::Phys: return As(x, y).uncoupledCodomain()[2]; break;
        default: std::terminate();
        }
    }
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
Qbasis<Symmetry, 1> iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::braBasis(const int x, const int y, const Opts::Leg leg) const
{
    if constexpr(ALL_OUT_LEGS) {
        switch(leg) {
        case Opts::Leg::Left: return Adags(x, y).uncoupledDomain()[0]; break;
        case Opts::Leg::Top: return Adags(x, y).uncoupledDomain()[1]; break;
        case Opts::Leg::Right: return Adags(x, y).uncoupledDomain()[2]; break;
        case Opts::Leg::Bottom: return Adags(x, y).uncoupledDomain()[3]; break;
        case Opts::Leg::Phys: return Adags(x, y).uncoupledDomain()[4]; break;
        default: std::terminate();
        }
    } else {
        switch(leg) {
        case Opts::Leg::Left: return Adags(x, y).uncoupledDomain()[0]; break;
        case Opts::Leg::Top: return Adags(x, y).uncoupledDomain()[1]; break;
        case Opts::Leg::Right: return Adags(x, y).uncoupledCodomain()[0]; break;
        case Opts::Leg::Bottom: return Adags(x, y).uncoupledCodomain()[1]; break;
        case Opts::Leg::Phys: return Adags(x, y).uncoupledDomain()[2]; break;
        default: std::terminate();
        }
    }
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
void iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::computeMs()
{
    for(int x = 0; x < cell().Lx; x++) {
        for(int y = 0; y < cell().Ly; y++) {
            if(not cell().pattern.isUnique(x, y)) { continue; }
            auto pos = cell().pattern.uniqueIndex(x, y);
            Ms[pos] = contractAAdag(As[pos], Adags[pos]);
        }
    }
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
void iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::initWeightTensors()
{
    Gs.resize(cell().pattern);
    whs.resize(cell().pattern);
    wvs.resize(cell().pattern);
    for(int x = 0; x < cell_.Lx; x++) {
        for(int y = 0; y < cell_.Ly; y++) {
            if(not cell_.pattern.isUnique(x, y)) { continue; }
            Gs(x, y) = As(x, y);
            whs(x, y) = Tensor<Scalar, 1, 1, Symmetry>::Identity(
                {{ketBasis(x, y, Opts::Leg::Right)}}, {{ketBasis(x, y, Opts::Leg::Right)}}, Gs(x, y).world());
            wvs(x, y) =
                Tensor<Scalar, 1, 1, Symmetry>::Identity({{ketBasis(x, y, Opts::Leg::Top)}}, {{ketBasis(x, y, Opts::Leg::Top)}}, Gs(x, y).world());
        }
    }
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
void iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::updateAtensors()
{
    for(int x = 0; x < cell_.Lx; ++x) {
        for(int y = 0; y < cell_.Ly; ++y) {
            if(not cell_.pattern.isUnique(x, y)) { continue; }
            As(x, y) = applyWeights(Gs(x, y), whs(x - 1, y).diag_sqrt(), wvs(x, y).diag_sqrt(), whs(x, y).diag_sqrt(), wvs(x, y + 1).diag_sqrt());
        }
    }
    updateAdags();
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
Tensor<Scalar, 1, 1, Symmetry> iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::Id_weight_h(int x, int y) const
{
    return Tensor<Scalar, 1, 1, Symmetry>::Identity(whs(x, y).uncoupledDomain(), whs(x, y).uncoupledCodomain(), whs(x, y).world());
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
Tensor<Scalar, 1, 1, Symmetry> iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::Id_weight_v(int x, int y) const
{
    return Tensor<Scalar, 1, 1, Symmetry>::Identity(wvs(x, y).uncoupledDomain(), wvs(x, y).uncoupledCodomain(), wvs(x, y).world());
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
std::string iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::info() const
{
    std::string res;
    auto [Dmin, Dmax, Davg, Dsigma] = calc_Ds();
    fmt::format_to(std::back_inserter(res),
                   "iPEPS(D*={}, {}): UnitCell=({}x{}), Sym={}, Dmin={}, Dmax={}, Davg={:.1f}, Dσ={:.1f}",
                   D,
                   Symmetry::name(),
                   cell().Lx,
                   cell().Ly,
                   fmt::streamed(sym()),
                   Dmin,
                   Dmax,
                   Davg,
                   Dsigma);
    return res;
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
void iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::debug_info() const
{
    std::cout << "Tensors:" << std::endl;
    for(int x = 0; x < cell_.Lx; x++) {
        for(int y = 0; y < cell_.Ly; y++) {
            if(not cell_.pattern.isUnique(x, y)) {
                std::cout << "Cell site: (" << x << "," << y << "): not unique." << std::endl;
                continue;
            }
            std::cout << "Cell site: (" << x << "," << y << "), A:" << std::endl;
            if constexpr(ENABLE_AD) {
                As(x, y).val().print(std::cout, true);
            } else {
                As(x, y).print(std::cout, true);
            }
            std::cout << std::endl;
        }
    }

    // std::cout << "Tensors:" << std::endl;
    // for(int x = 0; x < cell_.Lx; x++) {
    //     for(int y = 0; y < cell_.Ly; y++) {
    //         if(not cell_.pattern.isUnique(x, y)) {
    //             std::cout << "Cell site: (" << x << "," << y << "): not unique." << std::endl;
    //             continue;
    //         }
    //         std::cout << "Cell site: (" << x << "," << y << "), A:" << std::endl << As(x, y) << std::endl;
    //     }
    // }
}

template <typename Scalar, typename Symmetry, bool ALL_OUT_LEGS, bool ENABLE_AD>
void iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, ENABLE_AD>::grad_info() const
{
    std::cout << "Tensors:" << std::endl;
    for(int x = 0; x < cell_.Lx; x++) {
        for(int y = 0; y < cell_.Ly; y++) {
            if(not cell_.pattern.isUnique(x, y)) {
                std::cout << "Cell site: (" << x << "," << y << "): not unique." << std::endl;
                continue;
            }
            std::cout << "Cell site: (" << x << "," << y << "), A:" << std::endl;
            if constexpr(ENABLE_AD) {
                As(x, y).adj().print(std::cout, true);
            } else {
                std::cout << "No gradient data.";
            }
            std::cout << std::endl;
        }
    }
}

} // namespace Xped
