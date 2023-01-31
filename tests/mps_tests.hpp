template <typename Symmetry>
void perform_mps_contraction(std::size_t basis_size, mpi::XpedWorld& world)
{
    Qbasis<Symmetry, 1> in, loc;
    if(world.rank == 0) {
        in.setRandom(basis_size);
        loc.setRandom(8);
    }
    mpi::broadcast(in, world.rank, 0, world);
    mpi::broadcast(loc, world.rank, 0, world);

    auto out = in.combine(loc).forgetHistory();
    static thread_local std::mt19937 engine(std::random_device{}());
    Tensor<double, 2, 1, Symmetry> T1({{in, loc}}, {{out}}, world);
    T1.setRandom(engine);
    Tensor<double, 2, 1, Symmetry> T2({{in, loc}}, {{out}}, world);
    T2.setRandom(engine);

    Tensor<double, 1, 1, Symmetry> Bright({{out}}, {{out}}, world);
    Bright.setRandom(engine);
    SPDLOG_INFO("Mps tests: Set tensors to random.");
    Tensor<double, 1, 1, Symmetry> Brightn;
    contract_R(Bright, T1, T2, Brightn);
    auto Bcheck = (T2 * Bright).template permute<+1, 0, 1, 2>() * (T1.template permute<+1, 0, 1, 2>().adjoint());
    CHECK((Brightn - Bcheck).norm() == doctest::Approx(0.));
    SPDLOG_INFO("Mps tests: right check done.");
    Tensor<double, 1, 1, Symmetry> Bleft({{in}}, {{in}}, world);
    Bleft.setRandom(engine);
    Tensor<double, 1, 1, Symmetry> Bleftn;
    contract_L(Bleft, T1, T2, Bleftn);
    Bcheck.clear();
    Bcheck = (T1.template permute<+1, 0, 1, 2>().adjoint() * Bleft).template permute<+1, 1, 2, 0>() * T2;
    CHECK((Bleftn - Bcheck).norm() == doctest::Approx(0.));
    SPDLOG_INFO("Mps tests: left check done.");
}
