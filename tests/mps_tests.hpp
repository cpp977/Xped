template <typename Symmetry>
void perform_mps_contraction(std::size_t basis_size)
{
    Qbasis<Symmetry, 1> in, loc;
    in.setRandom(basis_size);
    loc.setRandom(8);
    auto out = in.combine(loc).forgetHistory();
    Xped<double, 2, 1, Symmetry> T1({{in, loc}}, {{out}});
    T1.setRandom();
    Xped<double, 2, 1, Symmetry> T2({{in, loc}}, {{out}});
    T2.setRandom();

    Xped<double, 1, 1, Symmetry> Bright({{out}}, {{out}});
    Bright.setRandom();
    Xped<double, 1, 1, Symmetry> Brightn;
    contract_R(Bright, T1, T2, Brightn);
    auto Bcheck = (T2 * Bright).template permute<+1, 0, 1, 2>() * (T1.template permute<+1, 0, 1, 2>().adjoint());
    CHECK((Brightn - Bcheck).norm() == doctest::Approx(0.));

    Xped<double, 1, 1, Symmetry> Bleft({{in}}, {{in}});
    Bleft.setRandom();
    Xped<double, 1, 1, Symmetry> Bleftn;
    contract_L(Bleft, T1, T2, Bleftn);
    Bcheck.clear();
    Bcheck = (T1.template permute<+1, 0, 1, 2>().adjoint() * Bleft).template permute<+1, 1, 2, 0>() * T2;
    CHECK((Bleftn - Bcheck).norm() == doctest::Approx(0.));
}
