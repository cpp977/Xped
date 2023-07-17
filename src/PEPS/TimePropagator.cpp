#include "Xped/PEPS/TimePropagator.hpp"

#include "Xped/PEPS/PEPSContractions.hpp"

namespace Xped {

template <typename Scalar, typename TimeScalar, typename HamScalar, typename Symmetry>
void TimePropagator<Scalar, TimeScalar, HamScalar, Symmetry>::t_step(iPEPS<Scalar, Symmetry>& Psi)
{
    if(H.data_d1.size() > 0 and H.data_d2.size() == 0) {
        for(auto i = 0ul; i < cell_.uniqueSize(); ++i) {
            auto [x, y] = cell_.pattern.coords(i);
            assert(cell_.pattern.uniqueIndex(x, y) != cell_.pattern.uniqueIndex(x + 1, y) and
                   "Unit cell pattern is inconsistent with Hamiltonian terms");
            assert(cell_.pattern.uniqueIndex(x + 1, y) != cell_.pattern.uniqueIndex(x + 1, y + 1) and
                   "Unit cell pattern is inconsistent with Hamiltonian terms");
            assert(cell_.pattern.uniqueIndex(x, y) != cell_.pattern.uniqueIndex(x + 1, y + 1) and
                   "Unit cell pattern is inconsistent with Hamiltonian terms");
            // fmt::print("diagonal i={}: x,y={},{}\n", i, x, y);
            t_step_d1(Psi, x, y, Opts::GATE_ORDER::VDH);
        }

        for(auto i = cell_.uniqueSize() - 1; i != std::numeric_limits<std::size_t>::max(); i--) {
            auto [x, y] = cell_.pattern.coords(i);
            // fmt::print("reverse diagonal i={}: x,y={},{}\n", i, x, y);
            t_step_d1(Psi, x, y, Opts::GATE_ORDER::HDV);
        }
    } else if(H.data_d2.size() > 0 and H.data_d1.size() == 0) {
        for(auto i = 0ul; i < cell_.uniqueSize(); ++i) {
            auto [x, y] = cell_.pattern.coords(i);
            // fmt::print("diagonal 2 update for ({},{})\n", x, y);
            assert(cell_.pattern.uniqueIndex(x, y - 1) != cell_.pattern.uniqueIndex(x + 1, y - 1) and
                   "Unit cell pattern is inconsistent with Hamiltonian terms");
            assert(cell_.pattern.uniqueIndex(x, y) != cell_.pattern.uniqueIndex(x, y - 1) and
                   "Unit cell pattern is inconsistent with Hamiltonian terms");
            assert(cell_.pattern.uniqueIndex(x, y) != cell_.pattern.uniqueIndex(x + 1, y - 1) and
                   "Unit cell pattern is inconsistent with Hamiltonian terms");
            // fmt::print("vertical i={}: x,y={},{}\n", i, x, y);
            t_step_d2(Psi, x, y, Opts::GATE_ORDER::VDH);
        }

        for(auto i = cell_.uniqueSize() - 1; i != std::numeric_limits<std::size_t>::max(); i--) {
            auto [x, y] = cell_.pattern.coords(i);
            // fmt::print("horizontal i={}: x,y={},{}\n", i, x, y);
            t_step_d2(Psi, x, y, Opts::GATE_ORDER::HDV);
        }

    } else if(H.data_d2.size() > 0 and H.data_d1.size() > 0) {
        for(auto i = 0ul; i < cell_.uniqueSize(); ++i) {
            auto [x, y] = cell_.pattern.coords(i);
            assert(cell_.pattern.uniqueIndex(x, y) != cell_.pattern.uniqueIndex(x + 1, y) and
                   "Unit cell pattern is inconsistent with Hamiltonian terms");
            assert(cell_.pattern.uniqueIndex(x, y) != cell_.pattern.uniqueIndex(x, y + 1) and
                   "Unit cell pattern is inconsistent with Hamiltonian terms");
            assert(cell_.pattern.uniqueIndex(x + 1, y) != cell_.pattern.uniqueIndex(x, y + 1) and
                   "Unit cell pattern is inconsistent with Hamiltonian terms");
            // fmt::print("vertical i={}: x,y={},{}\n", i, x, y);
            t_step_d2(Psi, x, y, Opts::GATE_ORDER::HDV, /*UPDATE_BOTH_DIAGONALS=*/true);
        }
        for(auto i = 0ul; i < cell_.uniqueSize(); ++i) {
            auto [x, y] = cell_.pattern.coords(i);
            assert(cell_.pattern.uniqueIndex(x, y) != cell_.pattern.uniqueIndex(x + 1, y) and
                   "Unit cell pattern is inconsistent with Hamiltonian terms");
            assert(cell_.pattern.uniqueIndex(x + 1, y) != cell_.pattern.uniqueIndex(x + 1, y + 1) and
                   "Unit cell pattern is inconsistent with Hamiltonian terms");
            assert(cell_.pattern.uniqueIndex(x, y) != cell_.pattern.uniqueIndex(x + 1, y + 1) and
                   "Unit cell pattern is inconsistent with Hamiltonian terms");
            // fmt::print("vertical i={}: x,y={},{}\n", i, x, y);
            t_step_d1(Psi, x, y, Opts::GATE_ORDER::HDV, /*UPDATE_BOTH_DIAGONALS=*/true);
        }

        for(auto i = cell_.uniqueSize() - 1; i != std::numeric_limits<std::size_t>::max(); i--) {
            auto [x, y] = cell_.pattern.coords(i);
            // fmt::print("horizontal i={}: x,y={},{}\n", i, x, y);
            t_step_d1(Psi, x, y, Opts::GATE_ORDER::VDH, /*UPDATE_BOTH_DIAGONALS=*/true);
        }

        for(auto i = cell_.uniqueSize() - 1; i != std::numeric_limits<std::size_t>::max(); i--) {
            auto [x, y] = cell_.pattern.coords(i);
            // fmt::print("horizontal i={}: x,y={},{}\n", i, x, y);
            t_step_d2(Psi, x, y, Opts::GATE_ORDER::VDH, /*UPDATE_BOTH_DIAGONALS=*/true);
        }
    } else {
        for(auto i = 0ul; i < cell_.uniqueSize(); ++i) {
            auto [x, y] = cell_.pattern.coords(i);
            assert(cell_.pattern.uniqueIndex(x, y) != cell_.pattern.uniqueIndex(x + 1, y) and
                   "Unit cell pattern is inconsistent with Hamiltonian terms");
            // fmt::print("horizontal i={}: x,y={},{}\n", i, x, y);
            t_step_h(Psi, x, y);
        }
        for(auto i = 0ul; i < cell_.uniqueSize(); ++i) {
            auto [x, y] = cell_.pattern.coords(i);
            assert(cell_.pattern.uniqueIndex(x, y - 1) != cell_.pattern.uniqueIndex(x, y) and
                   "Unit cell pattern is inconsistent with Hamiltonian terms");
            // fmt::print("vertical i={}: x,y={},{}\n", i, x, y);
            t_step_v(Psi, x, y);
        }

        for(auto i = cell_.uniqueSize() - 1; i != std::numeric_limits<std::size_t>::max(); i--) {
            auto [x, y] = cell_.pattern.coords(i);
            // fmt::print("vertical i={}: x,y={},{}\n", i, x, y);
            t_step_v(Psi, x, y);
        }
        for(auto i = cell_.uniqueSize() - 1; i != std::numeric_limits<std::size_t>::max(); i--) {
            auto [x, y] = cell_.pattern.coords(i);
            // fmt::print("horizontal i={}: x,y={},{}\n", i, x, y);
            t_step_h(Psi, x, y);
        }
    }
}

template <typename Scalar, typename TimeScalar, typename HamScalar, typename Symmetry>
void TimePropagator<Scalar, TimeScalar, HamScalar, Symmetry>::t_step_h(iPEPS<Scalar, Symmetry>& Psi, int x, int y)
{
    auto left_full = applyWeights(Psi.Gs(x, y), Psi.whs(x - 1, y), Psi.wvs(x, y), Psi.whs(x, y).diag_sqrt(), Psi.wvs(x, y + 1));
    auto right_full = applyWeights(Psi.Gs(x + 1, y), Psi.whs(x, y).diag_sqrt(), Psi.wvs(x + 1, y), Psi.whs(x + 1, y), Psi.wvs(x + 1, y + 1));
    // auto bond = left_full.template contract<std::array{-1, -2, 1, -3, -4}, std::array{1, -5, -6, -7, -8}, 4>(right_full);
    // auto shifted_ham = H.shiftQN(Psi.charges());
    // auto enlarged_bond =
    //     shifted_ham.data_h(x, y).mexp(-dt).eval().template contract<std::array{1, 2, -4, -8}, std::array{-1, -2, -3, 1, -5, -6,
    //     -7, 2}, 4>(bond);
    // [[maybe_unused]] double dumb;
    // auto [left_p, weight, right_p] = enlarged_bond.tSVD(Psi.D, 0., dumb, false);
    // Psi.whs(x, y) = weight * (1. / weight.maxNorm());
    // auto Gtmp = applyWeights(left_p.template permute<2, 0, 1, 4, 2, 3>(),
    //                          Psi.whs(x - 1, y).diag_inv(),
    //                          Psi.wvs(x, y).diag_inv(),
    //                          Psi.Id_weight_h(x, y),
    //                          Psi.wvs(x, y + 1).diag_inv());
    // Psi.Gs(x, y) = Gtmp * (1. / Gtmp.maxNorm());
    // auto Gtmp2 = applyWeights(right_p.template permute<-1, 0, 1, 2, 3, 4>(),
    //                           Psi.Id_weight_h(x, y),
    //                           Psi.wvs(x + 1, y).diag_inv(),
    //                           Psi.whs(x + 1, y).diag_inv(),
    //                           Psi.wvs(x + 1, y + 1).diag_inv());
    // Psi.Gs(x + 1, y) = Gtmp2 * (1. / Gtmp2.maxNorm());

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
    auto enlarged_bond = U.data_h(x, y).twist(0).twist(1).template contract<std::array{1, 2, -1, -3}, std::array{-2, 1, 2, -4}, 2>(bond);
    // fmt::print("enlarged bond\n");
    // enlarged_bond.print(std::cout, true);
    // std::cout << std::endl;
    auto [bond_lp, weight, bond_rp] = renormalize(enlarged_bond, left, right, Psi.D);
    spectrum_h(x, y) = weight; // * (1. / weight.trace());
    Psi.whs(x, y) = weight * (1. / weight.maxNorm());
    // fmt::print("weight_h for site {},{}:\n", x, y);
    // weight.print(std::cout, true);
    // std::cout << std::endl;
    auto Gtmp1 = left.template contract<std::array{-1, -2, -4, 1}, std::array{-5, 1, -3}, 2>(bond_lp); // * weights;
    auto Gtmp2 = applyWeights(Gtmp1, Psi.whs(x - 1, y).diag_inv(), Psi.wvs(x, y).diag_inv(), Psi.Id_weight_h(x, y), Psi.wvs(x, y + 1).diag_inv());
    Psi.Gs(x, y) = Gtmp2; // * (1. / Gtmp2.maxNorm());
    Gtmp1 = bond_rp.template contract<std::array{-1, -5, 1}, std::array{1, -2, -3, -4}, 2>(right);
    Gtmp2 = applyWeights(Gtmp1, Psi.Id_weight_h(x, y), Psi.wvs(x + 1, y).diag_inv(), Psi.whs(x + 1, y).diag_inv(), Psi.wvs(x + 1, y + 1).diag_inv());
    Psi.Gs(x + 1, y) = Gtmp2; // * (1. / Gtmp2.maxNorm());
}

template <typename Scalar, typename TimeScalar, typename HamScalar, typename Symmetry>
void TimePropagator<Scalar, TimeScalar, HamScalar, Symmetry>::t_step_v(iPEPS<Scalar, Symmetry>& Psi, int x, int y)
{
    auto top_full = applyWeights(Psi.Gs(x, y - 1), Psi.whs(x - 1, y - 1), Psi.wvs(x, y - 1), Psi.whs(x, y - 1), Psi.wvs(x, y).diag_sqrt());
    auto bottom_full = applyWeights(Psi.Gs(x, y), Psi.whs(x - 1, y), Psi.wvs(x, y).diag_sqrt(), Psi.whs(x, y), Psi.wvs(x, y + 1));
    [[maybe_unused]] double dumb;
    auto [top, tmp_t, bond_t] = top_full.template permute<-1, 0, 1, 2, 3, 4>().tSVD(std::numeric_limits<std::size_t>::max(), 0., dumb, false);
    bond_t = tmp_t * bond_t;
    auto [bond_b, tmp_b, bottom] = bottom_full.template permute<0, 1, 4, 0, 2, 3>().tSVD(std::numeric_limits<std::size_t>::max(), 0., dumb, false);
    bond_b = bond_b * tmp_b;
    // auto [top, bond_t] = top_full.template permute<-1, 0, 1, 2, 3, 4>().tQR();
    // auto [bond_b, bottom] = bottom_full.template permute<0, 1, 4, 0, 2, 3>().tQR();
    auto bond = bond_t.template contract<std::array{-1, 1, -2}, std::array{1, -3, -4}, 2>(bond_b);
    auto enlarged_bond = U.data_v(x, y - 1).twist(0).twist(1).template contract<std::array{1, 2, -1, -3}, std::array{-2, 1, 2, -4}, 2>(bond);
    auto [bond_tp, weight, bond_bp] = renormalize(enlarged_bond, top, bottom, Psi.D);
    spectrum_v(x, y) = weight; // * (1. / weight.trace());
    Psi.wvs(x, y) = weight * (1. / weight.maxNorm());
    // fmt::print("weight_v for site {},{}:\n", x, y);
    // weight.print(std::cout, true);
    // std::cout << std::endl;
    auto Gtmp1 = top.template contract<std::array{-1, -2, -3, 1}, std::array{-5, 1, -4}, 2>(bond_tp);
    auto Gtmp2 =
        applyWeights(Gtmp1, Psi.whs(x - 1, y - 1).diag_inv(), Psi.wvs(x, y - 1).diag_inv(), Psi.whs(x, y - 1).diag_inv(), Psi.Id_weight_v(x, y));
    Psi.Gs(x, y - 1) = Gtmp2; // * (1. / Gtmp2.maxNorm());
    Gtmp1 = bond_bp.template contract<std::array{-2, -5, 1}, std::array{1, -1, -3, -4}, 2>(bottom);
    Gtmp2 = applyWeights(Gtmp1, Psi.whs(x - 1, y).diag_inv(), Psi.Id_weight_v(x, y), Psi.whs(x, y).diag_inv(), Psi.wvs(x, y + 1).diag_inv());
    Psi.Gs(x, y) = Gtmp2; // * (1. / Gtmp2.maxNorm());
}

/*
  via upper right:
  A(x,y)--A
        \ |
          A

  via lower left:
  A(x,y)
  |     \
  A------A
*/
template <typename Scalar, typename TimeScalar, typename HamScalar, typename Symmetry>
void TimePropagator<Scalar, TimeScalar, HamScalar, Symmetry>::t_step_d1(iPEPS<Scalar, Symmetry>& Psi,
                                                                        int x,
                                                                        int y,
                                                                        Opts::GATE_ORDER gate_order,
                                                                        bool UPDATE_BOTH_DIAGONALS)
{
    auto tl_full = applyWeights(Psi.Gs(x, y), Psi.whs(x - 1, y), Psi.wvs(x, y), Psi.whs(x, y).diag_sqrt(), Psi.wvs(x, y + 1));
    auto br_full =
        applyWeights(Psi.Gs(x + 1, y + 1), Psi.whs(x, y + 1), Psi.wvs(x + 1, y + 1).diag_sqrt(), Psi.whs(x + 1, y + 1), Psi.wvs(x + 1, y + 2));
    auto tr_full = applyWeights(Psi.Gs(x + 1, y), Psi.whs(x, y).diag_sqrt(), Psi.wvs(x + 1, y), Psi.whs(x + 1, y), Psi.wvs(x + 1, y + 1).diag_sqrt());
    [[maybe_unused]] double dumb;
    auto [tl, tmp_tl, bond_tl] = tl_full.template permute<-1, 0, 1, 3, 2, 4>().tSVD(std::numeric_limits<std::size_t>::max(), 0., dumb, false);
    bond_tl = tmp_tl * bond_tl;
    auto [bond_br, tmp_br, br] = br_full.template permute<0, 1, 4, 0, 2, 3>().tSVD(std::numeric_limits<std::size_t>::max(), 0., dumb, false);
    bond_br = bond_br * tmp_br;
    auto tmp = bond_tl.template contract<std::array{-1, 1, -2}, std::array{1, -3, -4, -5, -6}, 2>(tr_full);
    auto bond = tmp.template contract<std::array{-1, -2, -3, -4, 1, -5}, std::array{1, -6, -7}, 5>(bond_br);
    Tensor<Scalar, 2, 5, Symmetry> enlarged_bond;
    switch(gate_order) {
    case Opts::GATE_ORDER::HDV: {
        TwoSiteObservable<HamScalar, Symmetry> U_used = UPDATE_BOTH_DIAGONALS ? Usq : U;
        enlarged_bond = U.data_h(x, y).twist(0).twist(1).template contract<std::array{1, 2, -2, -5}, std::array{-1, 1, -3, -4, 2, -6, -7}, 2>(bond);
        enlarged_bond = U_used.data_d1(x, y).twist(0).twist(1).template contract<std::array{1, 2, -2, -6}, std::array{-1, 1, -3, -4, -5, 2, -7}, 2>(
            enlarged_bond);
        enlarged_bond =
            U.data_v(x + 1, y).twist(0).twist(1).template contract<std::array{1, 2, -5, -6}, std::array{-1, -2, -3, -4, 1, 2, -7}, 2>(enlarged_bond);
        break;
    }
    case Opts::GATE_ORDER::VDH: {
        TwoSiteObservable<HamScalar, Symmetry> U_used = UPDATE_BOTH_DIAGONALS ? Usq : U;
        enlarged_bond =
            U.data_v(x + 1, y).twist(0).twist(1).template contract<std::array{1, 2, -5, -6}, std::array{-1, -2, -3, -4, 1, 2, -7}, 2>(bond);
        enlarged_bond = U_used.data_d1(x, y).twist(0).twist(1).template contract<std::array{1, 2, -2, -6}, std::array{-1, 1, -3, -4, -5, 2, -7}, 2>(
            enlarged_bond);
        enlarged_bond =
            U.data_h(x, y).twist(0).twist(1).template contract<std::array{1, 2, -2, -5}, std::array{-1, 1, -3, -4, 2, -6, -7}, 2>(enlarged_bond);
        break;
    }
    }
    auto [bond_tlp, weight_tl, tmp_trp, weight_br, bond_brp] = renormalize_d1(enlarged_bond, Psi.D);
    auto trp = tmp_trp.template permute<2, 3, 0, 1, 4, 2>();

    spectrum_h(x, y) = weight_tl; // * (1. / weight.trace());
    spectrum_v(x + 1, y + 1) = weight_br; // * (1. / weight.trace());
    Psi.whs(x, y) = weight_tl * (1. / weight_tl.maxNorm());
    Psi.wvs(x + 1, y + 1) = weight_br * (1. / weight_br.maxNorm());

    auto Gtmp1 = tl.template contract<std::array{-1, -2, -4, 1}, std::array{1, -5, -3}, 2>(bond_tlp);
    auto Gtmp2 = applyWeights(Gtmp1, Psi.whs(x - 1, y).diag_inv(), Psi.wvs(x, y).diag_inv(), Psi.Id_weight_h(x, y), Psi.wvs(x, y + 1).diag_inv());
    Psi.Gs(x, y) = Gtmp2; // * (1. / Gtmp2.maxNorm());

    Gtmp1 = bond_brp.template contract<std::array{-2, -5, 1}, std::array{1, -1, -3, -4}, 2>(br);
    Gtmp2 = applyWeights(
        Gtmp1, Psi.whs(x, y + 1).diag_inv(), Psi.Id_weight_v(x + 1, y + 1), Psi.whs(x + 1, y + 1).diag_inv(), Psi.wvs(x + 1, y + 2).diag_inv());
    Psi.Gs(x + 1, y + 1) = Gtmp2; // * (1. / Gtmp2.maxNorm());

    Psi.Gs(x + 1, y) =
        applyWeights(trp, Psi.whs(x, y).diag_inv(), Psi.wvs(x + 1, y).diag_inv(), Psi.whs(x + 1, y).diag_inv(), Psi.Id_weight_v(x + 1, y + 1));
}

/*
  via upper left:
  A--A(x+1,y)
  | /
  A

  via lower right:
     A(x+1,y)
   / |
  A--A
*/
template <typename Scalar, typename TimeScalar, typename HamScalar, typename Symmetry>
void TimePropagator<Scalar, TimeScalar, HamScalar, Symmetry>::t_step_d2(iPEPS<Scalar, Symmetry>& Psi,
                                                                        int x,
                                                                        int y,
                                                                        Opts::GATE_ORDER gate_order,
                                                                        bool UPDATE_BOTH_DIAGONALS)
{
    auto tl_full = applyWeights(Psi.Gs(x, y), Psi.whs(x - 1, y), Psi.wvs(x, y), Psi.whs(x, y).diag_sqrt(), Psi.wvs(x, y + 1).diag_sqrt());
    auto bl_full = applyWeights(Psi.Gs(x, y + 1), Psi.whs(x - 1, y + 1), Psi.wvs(x, y + 1).diag_sqrt(), Psi.whs(x, y + 1), Psi.wvs(x, y + 2));
    auto tr_full = applyWeights(Psi.Gs(x + 1, y), Psi.whs(x, y).diag_sqrt(), Psi.wvs(x + 1, y), Psi.whs(x + 1, y), Psi.wvs(x + 1, y + 1));
    [[maybe_unused]] double dumb;
    auto [bond_tr, tmp_tr, tr] = tr_full.template permute<0, 0, 4, 1, 2, 3>().tSVD(std::numeric_limits<std::size_t>::max(), 0., dumb, false);
    bond_tr = bond_tr * tmp_tr;
    auto [bond_bl, tmp_bl, bl] = bl_full.template permute<0, 1, 4, 0, 2, 3>().tSVD(std::numeric_limits<std::size_t>::max(), 0., dumb, false);
    bond_bl = bond_bl * tmp_bl;
    auto tmp = bond_bl.twist(0).template contract<std::array{1, -1, -2}, std::array{-3, -4, -5, 1, -6}, 2>(tl_full);
    auto bond = tmp.template contract<std::array{-1, -2, -3, -4, 1, -5}, std::array{1, -6, -7}, 5>(bond_tr);
    auto shifted_ham = H.shiftQN(Psi.charges());
    Tensor<Scalar, 2, 5, Symmetry> enlarged_bond;
    switch(gate_order) {
    case Opts::GATE_ORDER::HDV: {
        TwoSiteObservable<HamScalar, Symmetry> U_used = UPDATE_BOTH_DIAGONALS ? Usq : U;
        enlarged_bond = U.data_h(x, y).twist(0).twist(1).template contract<std::array{1, 2, -5, -6}, std::array{-1, -2, -3, -4, 1, 2, -7}, 2>(bond);
        enlarged_bond =
            U_used.data_d2(x + 1, y).twist(0).twist(1).template contract<std::array{1, 2, -1, -6}, std::array{1, -2, -3, -4, -5, 2, -7}, 2>(
                enlarged_bond);
        enlarged_bond =
            U.data_v(x, y).twist(0).twist(1).template contract<std::array{1, 2, -5, -1}, std::array{2, -2, -3, -4, 1, -6, -7}, 2>(enlarged_bond);
        break;
    }
    case Opts::GATE_ORDER::VDH: {
        TwoSiteObservable<HamScalar, Symmetry> U_used = UPDATE_BOTH_DIAGONALS ? Usq : U;
        enlarged_bond = U.data_v(x, y).twist(0).twist(1).template contract<std::array{1, 2, -5, -1}, std::array{2, -2, -3, -4, 1, -6, -7}, 2>(bond);
        enlarged_bond =
            U_used.data_d2(x + 1, y).twist(0).twist(1).template contract<std::array{1, 2, -1, -6}, std::array{1, -2, -3, -4, -5, 2, -7}, 2>(
                enlarged_bond);
        enlarged_bond =
            U.data_h(x, y).twist(0).twist(1).template contract<std::array{1, 2, -5, -6}, std::array{-1, -2, -3, -4, 1, 2, -7}, 2>(enlarged_bond);
        break;
    }
    }
    auto [bond_blp, weight_bl, tmp_tlp, weight_tr, bond_trp] = renormalize_d2(enlarged_bond, Psi.D);
    auto tlp = tmp_tlp.template permute<2, 0, 1, 4, 3, 2>();
    spectrum_h(x, y) = weight_tr; // * (1. / weight.trace());
    spectrum_v(x, y + 1) = weight_bl; // * (1. / weight.trace());
    Psi.whs(x, y) = weight_tr * (1. / weight_tr.maxNorm());
    Psi.wvs(x, y + 1) = weight_bl * (1. / weight_bl.maxNorm());

    auto Gtmp1 = bond_blp.template contract<std::array{-2, -5, 1}, std::array{1, -1, -3, -4}, 2>(bl);
    auto Gtmp2 =
        applyWeights(Gtmp1, Psi.whs(x - 1, y + 1).diag_inv(), Psi.Id_weight_v(x, y + 1), Psi.whs(x, y + 1).diag_inv(), Psi.wvs(x, y + 2).diag_inv());
    Psi.Gs(x, y + 1) = Gtmp2; // * (1. / Gtmp2.maxNorm());

    Gtmp1 = bond_trp.template contract<std::array{-1, -5, 1}, std::array{1, -2, -3, -4}, 2>(tr);
    Gtmp2 = applyWeights(Gtmp1, Psi.Id_weight_h(x, y), Psi.wvs(x + 1, y).diag_inv(), Psi.whs(x + 1, y).diag_inv(), Psi.wvs(x + 1, y + 1).diag_inv());
    Psi.Gs(x + 1, y) = Gtmp2; // * (1. / Gtmp2.maxNorm());

    Psi.Gs(x, y) = applyWeights(tlp, Psi.whs(x - 1, y).diag_inv(), Psi.wvs(x, y).diag_inv(), Psi.Id_weight_h(x, y), Psi.wvs(x, y + 1).diag_inv());
}

template <typename Scalar, typename TimeScalar, typename HamScalar, typename Symmetry>
std::tuple<Tensor<Scalar, 2, 1, Symmetry>, Tensor<Scalar, 1, 1, Symmetry>, Tensor<Scalar, 1, 2, Symmetry>>
TimePropagator<Scalar, TimeScalar, HamScalar, Symmetry>::renormalize(const Tensor<Scalar, 2, 2, Symmetry>& bond,
                                                                     const Tensor<Scalar, 3, 1, Symmetry>& left,
                                                                     const Tensor<Scalar, 1, 3, Symmetry>& right,
                                                                     std::size_t max_keep) const
{
    std::tuple<Tensor<Scalar, 2, 1, Symmetry>, Tensor<Scalar, 1, 1, Symmetry>, Tensor<Scalar, 1, 2, Symmetry>> res;
    switch(update) {
    case Opts::Update::SIMPLE: {
        double dummy;
        res = bond.tSVD(max_keep, 1.e-14, dummy, false);
        break;
    }
    case Opts::Update::FULL: {
        assert(false and "Full update not implemented");
    }
    case Opts::Update::CLUSTER: {
        assert(false and "Cluster update not implemented");
    }
    }
    return res;
}

template <typename Scalar, typename TimeScalar, typename HamScalar, typename Symmetry>
std::tuple<Tensor<Scalar, 1, 2, Symmetry>,
           Tensor<Scalar, 1, 1, Symmetry>,
           Tensor<Scalar, 4, 1, Symmetry>,
           Tensor<Scalar, 1, 1, Symmetry>,
           Tensor<Scalar, 1, 2, Symmetry>>
TimePropagator<Scalar, TimeScalar, HamScalar, Symmetry>::renormalize_d2(const Tensor<Scalar, 2, 5, Symmetry>& bond, std::size_t max_keep) const
{
    std::tuple<Tensor<Scalar, 1, 2, Symmetry>,
               Tensor<Scalar, 1, 1, Symmetry>,
               Tensor<Scalar, 4, 1, Symmetry>,
               Tensor<Scalar, 1, 1, Symmetry>,
               Tensor<Scalar, 1, 2, Symmetry>>
        res;
    switch(update) {
    case Opts::Update::SIMPLE: {
        double dummy;
        Tensor<Scalar, 5, 1, Symmetry> tmp;
        std::tie(tmp, std::get<1>(res), std::get<0>(res)) = bond.template permute<-3, 2, 3, 4, 5, 6, 0, 1>().tSVD(max_keep, 1.e-14, dummy, false);
        Tensor<Scalar, 4, 1, Symmetry> tmp2;
        std::tie(std::get<2>(res), std::get<3>(res), std::get<4>(res)) =
            (tmp * std::get<1>(res)).template permute<1, 0, 1, 2, 5, 3, 4>().tSVD(max_keep, 1.e-14, dummy, false);
        break;
    }
    case Opts::Update::FULL: {
        assert(false and "Full update not implemented");
    }
    case Opts::Update::CLUSTER: {
        assert(false and "Cluster update not implemented");
    }
    }
    return res;
}

template <typename Scalar, typename TimeScalar, typename HamScalar, typename Symmetry>
std::tuple<Tensor<Scalar, 2, 1, Symmetry>,
           Tensor<Scalar, 1, 1, Symmetry>,
           Tensor<Scalar, 4, 1, Symmetry>,
           Tensor<Scalar, 1, 1, Symmetry>,
           Tensor<Scalar, 1, 2, Symmetry>>
TimePropagator<Scalar, TimeScalar, HamScalar, Symmetry>::renormalize_d1(const Tensor<Scalar, 2, 5, Symmetry>& bond, std::size_t max_keep) const
{
    std::tuple<Tensor<Scalar, 2, 1, Symmetry>,
               Tensor<Scalar, 1, 1, Symmetry>,
               Tensor<Scalar, 4, 1, Symmetry>,
               Tensor<Scalar, 1, 1, Symmetry>,
               Tensor<Scalar, 1, 2, Symmetry>>
        res;
    switch(update) {
    case Opts::Update::SIMPLE: {
        double dummy;
        Tensor<Scalar, 1, 5, Symmetry> tmp;
        std::tie(std::get<0>(res), std::get<1>(res), tmp) = bond.tSVD(max_keep, 1.e-14, dummy, false);
        Tensor<Scalar, 4, 1, Symmetry> tmp2;
        std::tie(std::get<2>(res), std::get<3>(res), std::get<4>(res)) =
            (std::get<1>(res) * tmp).template permute<-3, 1, 2, 3, 0, 4, 5>().tSVD(max_keep, 1.e-14, dummy, false);
        break;
    }
    case Opts::Update::FULL: {
        assert(false and "Full update not implemented");
    }
    case Opts::Update::CLUSTER: {
        assert(false and "Cluster update not implemented");
    }
    }
    return res;
}

template <typename Scalar, typename TimeScalar, typename HamScalar, typename Symmetry>
void TimePropagator<Scalar, TimeScalar, HamScalar, Symmetry>::initU()
{
    auto shifted_ham = H.shiftQN(charges);
    U = TwoSiteObservable<HamScalar, Symmetry>(H.data_h.pat, H.bond);
    Usqrt = TwoSiteObservable<HamScalar, Symmetry>(H.data_h.pat, H.bond);
    Usq = TwoSiteObservable<HamScalar, Symmetry>(H.data_h.pat, H.bond);
    if(H.data_h.size() > 0) {
        for(auto i = 0ul; i < cell_.uniqueSize(); ++i) {
            U.data_h[i] = shifted_ham.data_h[i].mexp(-dt).eval();
            Usqrt.data_h[i] = shifted_ham.data_h[i].mexp(-0.5 * dt).eval();
            Usq.data_h[i] = shifted_ham.data_h[i].mexp(-2. * dt).eval();
        }
    }
    if(H.data_v.size() > 0) {
        for(auto i = 0ul; i < cell_.uniqueSize(); ++i) {
            U.data_v[i] = shifted_ham.data_v[i].mexp(-dt).eval();
            Usqrt.data_v[i] = shifted_ham.data_v[i].mexp(-0.5 * dt).eval();
            Usq.data_v[i] = shifted_ham.data_v[i].mexp(-2. * dt).eval();
        }
    }
    if(H.data_d1.size() > 0) {
        for(auto i = 0ul; i < cell_.uniqueSize(); ++i) {
            U.data_d1[i] = shifted_ham.data_d1[i].mexp(-dt).eval();
            Usqrt.data_d1[i] = shifted_ham.data_d1[i].mexp(-0.5 * dt).eval();
            Usq.data_d1[i] = shifted_ham.data_d1[i].mexp(-2. * dt).eval();
        }
    }
    if(H.data_d2.size() > 0) {
        for(auto i = 0ul; i < cell_.uniqueSize(); ++i) {
            U.data_d2[i] = shifted_ham.data_d2[i].mexp(-dt).eval();
            Usqrt.data_d2[i] = shifted_ham.data_d2[i].mexp(-0.5 * dt).eval();
            Usq.data_d2[i] = shifted_ham.data_d2[i].mexp(-2. * dt).eval();
        }
    }
}

} // namespace Xped
