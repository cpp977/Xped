#include <iostream>

#include <highfive/H5File.hpp>

#include <assert.hpp>

#include "Xped/PEPS/iPEPS.hpp"

#include "Xped/Util/Bool.hpp"

#include "Xped/Core/AdjointOp.hpp"

#include "Xped/IO/Matlab.hpp"

#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
iPEPS<Scalar, Symmetry, ENABLE_AD>::iPEPS(const UnitCell& cell,
                                          std::size_t D,
                                          const Qbasis<Symmetry, 1>& auxBasis,
                                          const Qbasis<Symmetry, 1>& physBasis)
    : D(D)
    , cell_(cell)

{
    charges_ = TMatrix<qType>(cell_.pattern);
    charges_.setConstant(Symmetry::qvacuum());

    TMatrix<Qbasis<Symmetry, 1>> aux;
    aux.setConstant(auxBasis);

    TMatrix<Qbasis<Symmetry, 1>> phys;
    phys.setConstant(physBasis);

    init(aux, aux, phys);
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
iPEPS<Scalar, Symmetry, ENABLE_AD>::iPEPS(const UnitCell& cell,
                                          std::size_t D,
                                          const TMatrix<Qbasis<Symmetry, 1>>& leftBasis,
                                          const TMatrix<Qbasis<Symmetry, 1>>& topBasis,
                                          const TMatrix<Qbasis<Symmetry, 1>>& physBasis)
    : D(D)
    , cell_(cell)

{
    charges_ = TMatrix<qType>(cell_.pattern);
    charges_.setConstant(Symmetry::qvacuum());
    init(leftBasis, topBasis, physBasis);
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
iPEPS<Scalar, Symmetry, ENABLE_AD>::iPEPS(const UnitCell& cell,
                                          std::size_t D,
                                          const TMatrix<Qbasis<Symmetry, 1>>& leftBasis,
                                          const TMatrix<Qbasis<Symmetry, 1>>& topBasis,
                                          const TMatrix<Qbasis<Symmetry, 1>>& physBasis,
                                          const TMatrix<qType>& charges)
    : D(D)
    , cell_(cell)
    , charges_(charges)
{
    init(leftBasis, topBasis, physBasis);
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
void iPEPS<Scalar, Symmetry, ENABLE_AD>::init(const TMatrix<Qbasis<Symmetry, 1>>& leftBasis,
                                              const TMatrix<Qbasis<Symmetry, 1>>& topBasis,
                                              const TMatrix<Qbasis<Symmetry, 1>>& physBasis)
{
    auto init_basis = [this](const Qbasis<Symmetry, 1>& phys_basis) {
        Qbasis<Symmetry, 1> out;
        std::size_t dim_per_charge = D < phys_basis.Nq() ? 1ul : D / phys_basis.Nq();
        std::size_t dim_for_vac = D < phys_basis.Nq() ? 1ul : dim_per_charge + D % phys_basis.Nq();
        auto inserted_states = 0ul;
        auto phys_qs = phys_basis.unordered_qs();
        bool HAS_VACUUM = true;
        if(phys_qs.contains(Symmetry::qvacuum())) {
            out.push_back(Symmetry::qvacuum(), dim_for_vac);
            inserted_states += dim_for_vac;
        } else {
            HAS_VACUUM = false;
        }
        if(inserted_states == D) { return out; }
        bool FIRST_INSERTION = true;
        for(auto Q : Symmetry::lowest_qs()) {
            if(phys_qs.contains(Q)) {
                if(FIRST_INSERTION and not HAS_VACUUM) {
                    out.push_back(Q, dim_for_vac);
                    inserted_states += dim_for_vac;
                    FIRST_INSERTION = false;
                } else {
                    out.push_back(Q, dim_per_charge);
                    inserted_states += dim_per_charge;
                }
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
        return out;
    };

    As.resize(cell().pattern);
    Adags.resize(cell().pattern);

    for(int x = 0; x < cell().Lx; x++) {
        for(int y = 0; y < cell().Ly; y++) {
            if(not cell().pattern.isUnique(x, y)) { continue; }
            auto pos = cell().pattern.uniqueIndex(x, y);
            auto [dummy, shifted_physBasis] = physBasis[pos].shift(charges_[pos]);
            auto left_basis_xy = leftBasis(x, y).dim() == 0 ? init_basis(shifted_physBasis) : leftBasis(x, y);
            auto left_basis_xp1y = leftBasis(x + 1, y).dim() == 0 ? init_basis(shifted_physBasis) : leftBasis(x + 1, y);
            auto top_basis_xy = topBasis(x, y).dim() == 0 ? init_basis(shifted_physBasis) : topBasis(x, y);
            auto top_basis_xyp1 = topBasis(x, y + 1).dim() == 0 ? init_basis(shifted_physBasis) : topBasis(x, y + 1);
            // fmt::print("x={}, y={}, original basis which will be shifted by {}:\n", x, y, Sym::format<Symmetry>(charges_[pos]));
            // std::cout << physBasis[pos] << std::endl;
            // std::cout << "shifted:\n" << shifted_physBasis << std::endl;
            As[pos] =
                Tensor<Scalar, 2, 3, Symmetry, ENABLE_AD>({{left_basis_xy, top_basis_xy}}, {{left_basis_xp1y, top_basis_xyp1, shifted_physBasis}});
            // As[pos].setZero();
            // std::cout << fmt::format("A({},{}): ", x, y) << As[pos].coupledDomain() << std::endl << As[pos].coupledCodomain() << std::endl;
            VERIFY(As[pos].coupledDomain().dim() > 0 and "Bases of the A tensor have no fused blocks.");
        }
    }
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
void iPEPS<Scalar, Symmetry, ENABLE_AD>::loadFromMatlab(const std::filesystem::path& p, const std::string& root_name, int qn_scale)
{
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
        As[i] = Xped::IO::loadMatlabTensor<double, 5, 0, Symmetry, Xped::HeapPolicy>(g_A, root, std::array{true, true, false, false, true}, qn_scale)
                    .twist(4)
                    .template permute<3, 3, 2, 1, 4, 0>();
        Adags[i] = As[i].adjoint().eval().template permute<0, 3, 4, 2, 0, 1>(Bool<ENABLE_AD>{});
    }
    // for(int x = 0; x < cell().Lx; ++x) {
    //     for(int y = 0; y < cell().Ly; ++y) {
    //         fmt::print("Site: {},{}: \n", x, y);
    //         std::cout << "left: " << ketBasis(x, y, Opts::LEG::LEFT) << std::endl;
    //         std::cout << "top: " << ketBasis(x, y, Opts::LEG::UP) << std::endl;
    //         std::cout << "right: " << ketBasis(x, y, Opts::LEG::RIGHT) << std::endl;
    //         std::cout << "bottom: " << ketBasis(x, y, Opts::LEG::DOWN) << std::endl;
    //         std::cout << "phys: " << ketBasis(x, y, Opts::LEG::PHYS) << std::endl << std::endl;
    //     }
    // }
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
iPEPS<Scalar, Symmetry, ENABLE_AD>::iPEPS(const iPEPS<Scalar, Symmetry, false>& other)
{
    D = other.D;
    cell_ = other.cell();
    charges_ = other.charges();
    As = other.As;
    Adags = other.Adags;
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
bool iPEPS<Scalar, Symmetry, ENABLE_AD>::checkConsistency() const
{
    bool out = true;
    for(int x = 0; x < cell().Lx; ++x) {
        for(int y = 0; y < cell().Ly; ++y) {
            if(ketBasis(x, y, Opts::LEG::LEFT) != ketBasis(x - 1, y, Opts::LEG::RIGHT)) {
                fmt::print("Site ({},{}): left basis does not match other right basis.\n", x, y);
                out = false;
            }
            if(ketBasis(x, y, Opts::LEG::RIGHT) != ketBasis(x + 1, y, Opts::LEG::LEFT)) {
                fmt::print("Site ({},{}): right basis does not match other left basis.\n", x, y);
                out = false;
            }
            if(ketBasis(x, y, Opts::LEG::UP) != ketBasis(x, y - 1, Opts::LEG::DOWN)) {
                fmt::print("Site ({},{}): up basis does not match other down basis.\n", x, y);
                out = false;
            }
            if(ketBasis(x, y, Opts::LEG::DOWN) != ketBasis(x, y + 1, Opts::LEG::UP)) {
                fmt::print("Site ({},{}): down basis does not match other up basis.\n", x, y);
                out = false;
            }
        }
    }
    return out;
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
void iPEPS<Scalar, Symmetry, ENABLE_AD>::setZero()
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

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
void iPEPS<Scalar, Symmetry, ENABLE_AD>::setRandom(std::size_t seed)
{
    static thread_local std::mt19937 engine(std::random_device{}());
    engine.seed(seed);
    for(int x = 0; x < cell_.Lx; x++) {
        for(int y = 0; y < cell_.Ly; y++) {
            if(not cell_.pattern.isUnique(x, y)) { continue; }
            auto pos = cell_.pattern.uniqueIndex(x, y);
            As[pos].setRandom(engine);
            Adags[pos] = As[pos].adjoint().eval().template permute<0, 3, 4, 2, 0, 1>(Bool<ENABLE_AD>{});
        }
    }
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
std::size_t iPEPS<Scalar, Symmetry, ENABLE_AD>::plainSize() const
{
    std::size_t res = 0;
    for(auto it = As.cbegin(); it != As.cend(); ++it) { res += it->plainSize(); }
    return res;
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
std::vector<Scalar> iPEPS<Scalar, Symmetry, ENABLE_AD>::data()
{
    std::vector<Scalar> out(plainSize());
    std::size_t count = 0;
    for(auto it = begin(); it != end(); ++it) { out[count++] = *it; }
    return out;
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
void iPEPS<Scalar, Symmetry, ENABLE_AD>::set_data(const Scalar* data, bool NORMALIZE)
{
    std::size_t count = 0;
    for(auto& A : As) {
        A.set_data(data + count, A.plainSize());
        if(NORMALIZE) {
            auto tmp = (A * (1. / A.maxNorm())).eval();
            A = tmp;
        }
        count += A.plainSize();
    }
    for(int x = 0; x < cell_.Lx; x++) {
        for(int y = 0; y < cell_.Ly; y++) {
            if(not cell_.pattern.isUnique(x, y)) { continue; }
            auto pos = cell_.pattern.uniqueIndex(x, y);

            Adags[pos] = As[pos].adjoint().eval().template permute<0, 3, 4, 2, 0, 1>(Bool<ENABLE_AD>{});
        }
    }
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
std::tuple<std::size_t, std::size_t, double, double> iPEPS<Scalar, Symmetry, ENABLE_AD>::calc_Ds() const
{
    std::vector<std::size_t> allDs;
    allDs.reserve(4 * As.size());
    for(auto i = 0ul; i < As.size(); ++i) {
        auto [x, y] = cell().pattern.coords(i);
        allDs.push_back(ketBasis(x, y, Opts::LEG::LEFT).fullDim());
        allDs.push_back(ketBasis(x, y, Opts::LEG::UP).fullDim());
        allDs.push_back(ketBasis(x, y, Opts::LEG::RIGHT).fullDim());
        allDs.push_back(ketBasis(x, y, Opts::LEG::DOWN).fullDim());
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

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
Qbasis<Symmetry, 1> iPEPS<Scalar, Symmetry, ENABLE_AD>::ketBasis(const int x, const int y, const Opts::LEG leg) const
{
    switch(leg) {
    case Opts::LEG::LEFT: return As(x, y).uncoupledDomain()[0]; break;
    case Opts::LEG::UP: return As(x, y).uncoupledDomain()[1]; break;
    case Opts::LEG::RIGHT: return As(x, y).uncoupledCodomain()[0]; break;
    case Opts::LEG::DOWN: return As(x, y).uncoupledCodomain()[1]; break;
    case Opts::LEG::PHYS: return As(x, y).uncoupledCodomain()[2]; break;
    default: std::terminate();
    }
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
Qbasis<Symmetry, 1> iPEPS<Scalar, Symmetry, ENABLE_AD>::braBasis(const int x, const int y, const Opts::LEG leg) const
{
    switch(leg) {
    case Opts::LEG::LEFT: return Adags(x, y).uncoupledDomain()[0]; break;
    case Opts::LEG::UP: return Adags(x, y).uncoupledDomain()[1]; break;
    case Opts::LEG::RIGHT: return Adags(x, y).uncoupledCodomain()[0]; break;
    case Opts::LEG::DOWN: return Adags(x, y).uncoupledCodomain()[1]; break;
    case Opts::LEG::PHYS: return Adags(x, y).uncoupledDomain()[2]; break;
    default: std::terminate();
    }
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
void iPEPS<Scalar, Symmetry, ENABLE_AD>::initWeightTensors()
{
    Gs.resize(cell().pattern);
    whs.resize(cell().pattern);
    wvs.resize(cell().pattern);
    for(int x = 0; x < cell_.Lx; x++) {
        for(int y = 0; y < cell_.Ly; y++) {
            if(not cell_.pattern.isUnique(x, y)) { continue; }
            Gs(x, y) = As(x, y);
            whs(x, y) = Tensor<Scalar, 1, 1, Symmetry>::Identity(
                {{ketBasis(x, y, Opts::LEG::RIGHT)}}, {{ketBasis(x, y, Opts::LEG::RIGHT)}}, Gs(x, y).world());
            wvs(x, y) =
                Tensor<Scalar, 1, 1, Symmetry>::Identity({{ketBasis(x, y, Opts::LEG::UP)}}, {{ketBasis(x, y, Opts::LEG::UP)}}, Gs(x, y).world());
        }
    }
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
void iPEPS<Scalar, Symmetry, ENABLE_AD>::updateAtensors()
{
    for(int x = 0; x < cell_.Lx; ++x) {
        for(int y = 0; y < cell_.Ly; ++y) {
            if(not cell_.pattern.isUnique(x, y)) { continue; }
            As(x, y) = applyWeights(Gs(x, y), whs(x - 1, y).diag_sqrt(), wvs(x, y).diag_sqrt(), whs(x, y).diag_sqrt(), wvs(x, y + 1).diag_sqrt());
            Adags(x, y) = As(x, y).adjoint().eval().template permute<0, 3, 4, 2, 0, 1>(Bool<ENABLE_AD>{});
        }
    }
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
Tensor<Scalar, 1, 1, Symmetry> iPEPS<Scalar, Symmetry, ENABLE_AD>::Id_weight_h(int x, int y) const
{
    return Tensor<Scalar, 1, 1, Symmetry>::Identity(whs(x, y).uncoupledDomain(), whs(x, y).uncoupledCodomain(), whs(x, y).world());
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
Tensor<Scalar, 1, 1, Symmetry> iPEPS<Scalar, Symmetry, ENABLE_AD>::Id_weight_v(int x, int y) const
{
    return Tensor<Scalar, 1, 1, Symmetry>::Identity(wvs(x, y).uncoupledDomain(), wvs(x, y).uncoupledCodomain(), wvs(x, y).world());
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
std::string iPEPS<Scalar, Symmetry, ENABLE_AD>::info() const
{
    std::string res;
    auto [Dmin, Dmax, Davg, Dsigma] = calc_Ds();
    fmt::format_to(std::back_inserter(res),
                   "iPEPS(D*={}): UnitCell=({}x{}), Dmin={}, Dmax={}, Davg={:.1f}, Dσ={:.1f}",
                   D,
                   cell().Lx,
                   cell().Ly,
                   Dmin,
                   Dmax,
                   Davg,
                   Dsigma);
    return res;
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
void iPEPS<Scalar, Symmetry, ENABLE_AD>::debug_info() const
{
    std::cout << "Tensors:" << std::endl;
    for(int x = 0; x < cell_.Lx; x++) {
        for(int y = 0; y < cell_.Ly; y++) {
            if(not cell_.pattern.isUnique(x, y)) {
                std::cout << "Cell site: (" << x << "," << y << "): not unique." << std::endl;
                continue;
            }
            std::cout << "Cell site: (" << x << "," << y << "), A:" << std::endl << As(x, y) << std::endl << std::endl;
            std::cout << "Cell site: (" << x << "," << y << "), A†:" << std::endl << Adags(x, y) << std::endl;
        }
    }
}

} // namespace Xped
