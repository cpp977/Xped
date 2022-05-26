#include <iostream>

#include "Xped/PEPS/iPEPS.hpp"

#include "Xped/Core/AdjointOp.hpp"

#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
iPEPS<Scalar, Symmetry, ENABLE_AD>::iPEPS(const UnitCell& cell, const Qbasis<Symmetry, 1>& auxBasis, const Qbasis<Symmetry, 1>& physBasis)
    : cell_(cell)
{
    D = auxBasis.fullDim();
    As.resize(cell.pattern);
    Adags.resize(cell.pattern);
    for(int x = 0; x < cell.Lx; x++) {
        for(int y = 0; y < cell.Ly; y++) {
            if(not cell.pattern.isUnique(x, y)) { continue; }
            auto pos = cell.pattern.uniqueIndex(x, y);
            As[pos] = Tensor<Scalar, 2, 3, Symmetry, ENABLE_AD>({{auxBasis, auxBasis}}, {{auxBasis, auxBasis, physBasis}});
            // As[pos].setZero();
            assert(As[pos].coupledDomain().dim() > 0 and "Bases of the A tensor have no fused blocks.");
        }
    }
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
iPEPS<Scalar, Symmetry, ENABLE_AD>::iPEPS(const iPEPS<Scalar, Symmetry, false>& other)
{
    D = other.D;
    cell_ = other.cell();
    As = other.As;
    Adags = other.Adags;
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
void iPEPS<Scalar, Symmetry, ENABLE_AD>::setRandom()
{
    for(int x = 0; x < cell_.Lx; x++) {
        for(int y = 0; y < cell_.Ly; y++) {
            if(not cell_.pattern.isUnique(x, y)) { continue; }
            auto pos = cell_.pattern.uniqueIndex(x, y);
            As[pos].setRandom();
            Adags[pos] = As[pos].adjoint().eval().template permute<0, 3, 4, 2, 0, 1>();
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
        if(NORMALIZE) { A = A * (1. / A.norm()); }
        count += A.plainSize();
    }

    for(int x = 0; x < cell_.Lx; x++) {
        for(int y = 0; y < cell_.Ly; y++) {
            if(not cell_.pattern.isUnique(x, y)) { continue; }
            auto pos = cell_.pattern.uniqueIndex(x, y);

            Adags[pos] = As[pos].adjoint().eval().template permute<0, 3, 4, 2, 0, 1>();
        }
    }
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
Qbasis<Symmetry, 1> iPEPS<Scalar, Symmetry, ENABLE_AD>::ketBasis(const int x, const int y, const LEG leg) const
{
    switch(leg) {
    case LEG::LEFT: return As(x, y).uncoupledDomain()[0]; break;
    case LEG::UP: return As(x, y).uncoupledDomain()[1]; break;
    case LEG::RIGHT: return As(x, y).uncoupledCodomain()[0]; break;
    case LEG::DOWN: return As(x, y).uncoupledCodomain()[1]; break;
    case LEG::PHYS: return As(x, y).uncoupledCodomain()[2]; break;
    default: std::terminate();
    }
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
Qbasis<Symmetry, 1> iPEPS<Scalar, Symmetry, ENABLE_AD>::braBasis(const int x, const int y, const LEG leg) const
{
    switch(leg) {
    case LEG::LEFT: return Adags(x, y).uncoupledDomain()[0]; break;
    case LEG::UP: return Adags(x, y).uncoupledDomain()[1]; break;
    case LEG::RIGHT: return Adags(x, y).uncoupledCodomain()[0]; break;
    case LEG::DOWN: return Adags(x, y).uncoupledCodomain()[1]; break;
    case LEG::PHYS: return Adags(x, y).uncoupledCodomain()[2]; break;
    default: std::terminate();
    }
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD>
void iPEPS<Scalar, Symmetry, ENABLE_AD>::info() const
{
    std::cout << "iPEPS(D=" << D << "): UnitCell=(" << cell_.Lx << "x" << cell_.Ly << ")" << std::endl;
    std::cout << "Tensors:" << std::endl;
    for(int x = 0; x < cell_.Lx; x++) {
        for(int y = 0; y < cell_.Lx; y++) {
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
