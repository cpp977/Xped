#include "Xped/MPS/Mps.hpp"
#include "Xped/MPS/MpsAlgebra.hpp"

int main()
{
    // Alias for the symmetry
    using Symmetry = Xped::Sym::SU2<Xped::Sym::SpinSU2>;

    // Number of sites in the MPS
    std::size_t L = 100;

    // Amount of quantum number blocks
    std::size_t Qinit = 10;

    // Amount of basis states per quantum number block
    std::size_t Minit = 100;

    // Target quantum number of the MPS
    Symmetry::qType Qtarget = Symmetry::qvacuum();

    // local (physical) basis of the MPS. In this case one spin 1/2
    Xped::Qbasis<Symmetry> qloc;
    qloc.push_back({2}, 1);

    // Contruct the MPS (with random entries)
    Xped::Mps<double, Symmetry> Psi(L, qloc, Qtot, Minit, Qinit);

    // Compute the norm
    auto norm = Xped::dot(Psi, Psi);

    // Sweep from left to put all A-tensors into right-canonical form
    for(std::size_t l = 0; l < L; ++l) { Psi.rightSweepStep(l, Xped::DMRG::BROOM::SVD); }
}
