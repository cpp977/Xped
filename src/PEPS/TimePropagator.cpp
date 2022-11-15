#include "Xped/PEPS/TimePropagator.hpp"

#include "Xped/PEPS/PEPSContractions.hpp"

namespace Xped {

template <typename Scalar, typename TimeScalar, typename Symmetry, typename Update>
void TimePropagator<Scalar, TimeScalar, Symmetry, Update>::t_step(TimeScalar dt)
{
    for(auto i = 0ul; i < cell_.uniqueSize(); ++i) {
        auto [x, y] = cell_.pattern.coords(i);
        // fmt::print("horizontal i={}: x,y={},{}\n", i, x, y);
        t_step_h(x, y, dt);
    }
    for(auto i = 0ul; i < cell_.uniqueSize(); ++i) {
        auto [x, y] = cell_.pattern.coords(i);
        // fmt::print("vertical i={}: x,y={},{}\n", i, x, y);
        t_step_v(x, y, dt);
    }

    for(auto i = cell_.uniqueSize() - 1; i != std::numeric_limits<std::size_t>::max(); i--) {
        auto [x, y] = cell_.pattern.coords(i);
        // fmt::print("vertical i={}: x,y={},{}\n", i, x, y);
        t_step_v(x, y, dt);
    }
    for(auto i = cell_.uniqueSize() - 1; i != std::numeric_limits<std::size_t>::max(); i--) {
        auto [x, y] = cell_.pattern.coords(i);
        // fmt::print("horizontal i={}: x,y={},{}\n", i, x, y);
        t_step_h(x, y, dt);
    }
}

template <typename Scalar, typename TimeScalar, typename Symmetry, typename Update>
void TimePropagator<Scalar, TimeScalar, Symmetry, Update>::t_step_h(int x, int y, TimeScalar dt)
{
    auto left_full = applyWeights(Psi->Gs(x, y), Psi->whs(x - 1, y), Psi->wvs(x, y), Psi->whs(x, y).diag_sqrt(), Psi->wvs(x, y + 1));
    auto right_full = applyWeights(Psi->Gs(x + 1, y), Psi->whs(x, y).diag_sqrt(), Psi->wvs(x + 1, y), Psi->whs(x + 1, y), Psi->wvs(x + 1, y + 1));
    // auto bond = left_full.template contract<std::array{-1, -2, 1, -3, -4}, std::array{1, -5, -6, -7, -8}, 4>(right_full);
    // auto shifted_ham = H.shiftQN(Psi->charges());
    // auto enlarged_bond =
    //     shifted_ham.data_h(x, y).mexp(-dt).eval().template contract<std::array{1, 2, -4, -8}, std::array{-1, -2, -3, 1, -5, -6,
    //     -7, 2}, 4>(bond);
    // [[maybe_unused]] double dumb;
    // auto [left_p, weight, right_p] = enlarged_bond.tSVD(Psi->D, 0., dumb, false);
    // Psi->whs(x, y) = weight * (1. / weight.maxNorm());
    // auto Gtmp = applyWeights(left_p.template permute<2, 0, 1, 4, 2, 3>(),
    //                          Psi->whs(x - 1, y).diag_inv(),
    //                          Psi->wvs(x, y).diag_inv(),
    //                          Psi->Id_weight_h(x, y),
    //                          Psi->wvs(x, y + 1).diag_inv());
    // Psi->Gs(x, y) = Gtmp * (1. / Gtmp.maxNorm());
    // auto Gtmp2 = applyWeights(right_p.template permute<-1, 0, 1, 2, 3, 4>(),
    //                           Psi->Id_weight_h(x, y),
    //                           Psi->wvs(x + 1, y).diag_inv(),
    //                           Psi->whs(x + 1, y).diag_inv(),
    //                           Psi->wvs(x + 1, y + 1).diag_inv());
    // Psi->Gs(x + 1, y) = Gtmp2 * (1. / Gtmp2.maxNorm());

    // fmt::print("left_full\n");
    // left_full.print(std::cout, true);
    // std::cout << std::endl;
    // fmt::print("right_full\n");
    // right_full.print(std::cout, true);
    // std::cout << std::endl;
    [[maybe_unused]] double dumb;
    auto [left, tmp_l, bond_l] = left_full.template permute<-1, 0, 1, 3, 2, 4>().tSVD(std::numeric_limits<std::size_t>::max(), 0., dumb, false);
    bond_l = tmp_l * bond_l;
    // fmt::print("left\n");
    // left.print(std::cout, true);
    // std::cout << std::endl;
    // fmt::print("bond_l\n");
    // bond_l.print(std::cout, true);
    // std::cout << std::endl;
    auto [bond_r, tmp_r, right] = right_full.template permute<0, 0, 4, 1, 2, 3>().tSVD(std::numeric_limits<std::size_t>::max(), 0., dumb, false);
    bond_r = bond_r * tmp_r;

    // auto [left, bond_l] = left_full.template permute<-1, 0, 1, 3, 2, 4>().tQR();
    // auto [bond_r, right] = right_full.template permute<0, 0, 4, 1, 2, 3>().tQR();

    // fmt::print("right\n");
    // right.print(std::cout, true);
    // std::cout << std::endl;
    // fmt::print("bond_r\n");
    // bond_r.print(std::cout, true);
    // std::cout << std::endl;

    auto bond = bond_l.template contract<std::array{-1, 1, -2}, std::array{1, -3, -4}, 2>(bond_r);

    // fmt::print("bond\n");
    // bond.print(std::cout, true);
    // std::cout << std::endl;
    auto shifted_ham = H.shiftQN(Psi->charges());
    auto enlarged_bond = shifted_ham.data_h(x, y).mexp(-dt).eval().template contract<std::array{1, 2, -1, -3}, std::array{-2, 1, 2, -4}, 2>(bond);
    // fmt::print("enlarged bond\n");
    // enlarged_bond.print(std::cout, true);
    // std::cout << std::endl;
    auto [bond_lp, weight, bond_rp] = updater.renormalize(enlarged_bond, left, right, Psi->D);
    Psi->whs(x, y) = weight * (1. / weight.maxNorm());
    // fmt::print("weight_h for site {},{}:\n", x, y);
    // weight.print(std::cout, true);
    // std::cout << std::endl;
    auto Gtmp1 = left.template contract<std::array{-1, -2, -4, 1}, std::array{-5, 1, -3}, 2>(bond_lp); // * weights;
    auto Gtmp2 = applyWeights(Gtmp1, Psi->whs(x - 1, y).diag_inv(), Psi->wvs(x, y).diag_inv(), Psi->Id_weight_h(x, y), Psi->wvs(x, y + 1).diag_inv());
    Psi->Gs(x, y) = Gtmp2; // * (1. / Gtmp2.maxNorm());
    Gtmp1 = bond_rp.template contract<std::array{-1, -5, 1}, std::array{1, -2, -3, -4}, 2>(right);
    Gtmp2 =
        applyWeights(Gtmp1, Psi->Id_weight_h(x, y), Psi->wvs(x + 1, y).diag_inv(), Psi->whs(x + 1, y).diag_inv(), Psi->wvs(x + 1, y + 1).diag_inv());
    Psi->Gs(x + 1, y) = Gtmp2; // * (1. / Gtmp2.maxNorm());
}

template <typename Scalar, typename TimeScalar, typename Symmetry, typename Update>
void TimePropagator<Scalar, TimeScalar, Symmetry, Update>::t_step_v(int x, int y, TimeScalar dt)
{
    auto top_full = applyWeights(Psi->Gs(x, y - 1), Psi->whs(x - 1, y - 1), Psi->wvs(x, y - 1), Psi->whs(x, y - 1), Psi->wvs(x, y).diag_sqrt());
    auto bottom_full = applyWeights(Psi->Gs(x, y), Psi->whs(x - 1, y), Psi->wvs(x, y).diag_sqrt(), Psi->whs(x, y), Psi->wvs(x, y + 1));
    [[maybe_unused]] double dumb;
    auto [top, tmp_t, bond_t] = top_full.template permute<-1, 0, 1, 2, 3, 4>().tSVD(std::numeric_limits<std::size_t>::max(), 0., dumb, false);
    bond_t = tmp_t * bond_t;
    auto [bond_b, tmp_b, bottom] = bottom_full.template permute<0, 1, 4, 0, 2, 3>().tSVD(std::numeric_limits<std::size_t>::max(), 0., dumb, false);
    bond_b = bond_b * tmp_b;
    // auto [top, bond_t] = top_full.template permute<-1, 0, 1, 2, 3, 4>().tQR();
    // auto [bond_b, bottom] = bottom_full.template permute<0, 1, 4, 0, 2, 3>().tQR();
    auto bond = bond_t.template contract<std::array{-1, 1, -2}, std::array{1, -3, -4}, 2>(bond_b);
    auto shifted_ham = H.shiftQN(Psi->charges());
    auto enlarged_bond = shifted_ham.data_v(x, y - 1).mexp(-dt).eval().template contract<std::array{1, 2, -1, -3}, std::array{-2, 1, 2, -4}, 2>(bond);
    auto [bond_tp, weight, bond_bp] = updater.renormalize(enlarged_bond, top, bottom, Psi->D);
    Psi->wvs(x, y) = weight * (1. / weight.maxNorm());
    // fmt::print("weight_v for site {},{}:\n", x, y);
    // weight.print(std::cout, true);
    // std::cout << std::endl;
    auto Gtmp1 = top.template contract<std::array{-1, -2, -3, 1}, std::array{-5, 1, -4}, 2>(bond_tp);
    auto Gtmp2 =
        applyWeights(Gtmp1, Psi->whs(x - 1, y - 1).diag_inv(), Psi->wvs(x, y - 1).diag_inv(), Psi->whs(x, y - 1).diag_inv(), Psi->Id_weight_v(x, y));
    Psi->Gs(x, y - 1) = Gtmp2; // * (1. / Gtmp2.maxNorm());
    Gtmp1 = bond_bp.template contract<std::array{-2, -5, 1}, std::array{1, -1, -3, -4}, 2>(bottom);
    Gtmp2 = applyWeights(Gtmp1, Psi->whs(x - 1, y).diag_inv(), Psi->Id_weight_v(x, y), Psi->whs(x, y).diag_inv(), Psi->wvs(x, y + 1).diag_inv());
    Psi->Gs(x, y) = Gtmp2; // * (1. / Gtmp2.maxNorm());
}

} // namespace Xped
