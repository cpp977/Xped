#include <iostream>

#include "Xped/PEPS/iPEPS.hpp"

#include "Xped/Core/AdjointOp.hpp"

#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry>
iPEPS<Scalar, Symmetry>::iPEPS(const UnitCell& cell, const Qbasis<Symmetry, 1>& auxBasis, const Qbasis<Symmetry, 1>& physBasis)
    : cell(cell)
{
    D = auxBasis.fullDim();
    As.resize(cell.pattern);
    Adags.resize(cell.pattern);
    for(int x = 0; x < cell.Lx; x++) {
        for(int y = 0; y < cell.Ly; y++) {
            if(not cell.pattern.isUnique(x, y)) { continue; }
            auto pos = cell.pattern.uniqueIndex(x, y);
            As[pos] = Tensor<Scalar, 2, 3, Symmetry>({{auxBasis, auxBasis}}, {{auxBasis, auxBasis, physBasis}});
            As[pos].setRandom();
            assert(As[pos].sector().size() > 0 and "Bases of the A tensor have no fused blocks.");
            Adags[pos] = As[pos].adjoint().eval().template permute<0, 3, 4, 2, 0, 1>();
        }
    }
}

template <typename Scalar, typename Symmetry>
Qbasis<Symmetry, 1> iPEPS<Scalar, Symmetry>::ketBasis(const int x, const int y, const LEG leg) const
{
    switch(leg) {
    case LEG::LEFT: return As(x, y).uncoupledDomain()[0]; break;
    case LEG::UP: return As(x, y).uncoupledDomain()[1]; break;
    case LEG::RIGHT: return As(x, y).uncoupledCodomain()[0]; break;
    case LEG::DOWN: return As(x, y).uncoupledCodomain()[1]; break;
    case LEG::PHYS: return As(x, y).uncoupledDomain()[2]; break;
    default: std::terminate();
    }
}

template <typename Scalar, typename Symmetry>
Qbasis<Symmetry, 1> iPEPS<Scalar, Symmetry>::braBasis(const int x, const int y, const LEG leg) const
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

template <typename Scalar, typename Symmetry>
void iPEPS<Scalar, Symmetry>::info() const
{
    std::cout << "iPEPS(D=" << D << "): UnitCell=(" << cell.Lx << "x" << cell.Ly << ")" << std::endl;
    std::cout << "Tensors:" << std::endl;
    for(int x = 0; x < cell.Lx; x++) {
        for(int y = 0; y < cell.Lx; y++) {
            if(not cell.pattern.isUnique(x, y)) {
                std::cout << "Cell site: (" << x << "," << y << "): not unique." << std::endl;
                continue;
            }
            std::cout << "Cell site: (" << x << "," << y << "), A:" << std::endl << As(x, y) << std::endl << std::endl;
            std::cout << "Cell site: (" << x << "," << y << "), A†:" << std::endl << Adags(x, y) << std::endl;
        }
    }
}

} // namespace Xped
