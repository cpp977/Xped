template <typename Symmetry, typename Scalar, int shift, std::size_t... per>
void perform_tensor_permute(const std::size_t& size, mpi::XpedWorld& world)
{
    // CTF::World world(comm);
    SPDLOG_WARN("Permute: Number of processes in tensor-test #={}", world.np);
    SPDLOG_WARN("Permute: I am process number #={}", world.rank);
    Qbasis<Symmetry, 1> B, C;
    if(world.rank == 0) {
        B.setRandom(size);
        C.setRandom(size);
    }

    XPED_MPI_BARRIER(world.comm)
    mpi::broadcast(B, world.rank, 0, world);
    mpi::broadcast(C, world.rank, 0, world);

    [[maybe_unused]] std::array<std::size_t, 4> perm{per...};
    SPDLOG_CRITICAL("Permutation: {} {} {} {}. Shift={}", perm[0], perm[1], perm[2], perm[3], shift);

    static thread_local std::mt19937 engine(std::random_device{}());

    Tensor<Scalar, 2, 2, Symmetry> t({{B, C}}, {{B, C}}, world);
    t.setRandom(engine);
    XPED_MPI_BARRIER(world.comm)
    auto tplain = t.plainTensor();
    XPED_MPI_BARRIER(world.comm)
    auto tp = t.template permute<shift, per...>();
    XPED_MPI_BARRIER(world.comm)
    PlainInterface::TType<Scalar, 4> tplainshuffle = PlainInterface::shuffle<Scalar, 4, per...>(tplain);
    XPED_MPI_BARRIER(world.comm)
    auto tplainp = tp.plainTensor();
    XPED_MPI_BARRIER(world.comm)
#ifdef XPED_USE_ARRAY_TENSOR_LIB
    auto check = nda::make_ein_sum<Scalar, 0, 1, 2, 3>(nda::ein<0, 1, 2, 3>(tplainp) - nda::ein<0, 1, 2, 3>(tplainshuffle));
#elif defined(XPED_USE_CYCLOPS_TENSOR_LIB)
    auto dims = PlainInterface::dimensions<Scalar, 4>(tplainshuffle);
    auto check = PlainInterface::construct<Scalar>(dims, world);
    check["ijkl"] = tplainshuffle["ijkl"] - tplainp["ijkl"];
#else
    Eigen::Tensor<Scalar, 4> check = tplainshuffle - tplainp;
#endif
    XPED_MPI_BARRIER(world.comm)
    auto zero_ = PlainInterface::contract<Scalar, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3>(check, check);
    XPED_MPI_BARRIER(world.comm)
    Scalar zero = PlainInterface::getVal<Scalar, 0>(zero_, {{}});
    SPDLOG_INFO("zero={}.", zero);
    CHECK(std::abs(zero) == doctest::Approx(0.));
}

template <typename Symmetry, typename Scalar, int shift>
void test_tensor_permute(const std::size_t& size, mpi::XpedWorld& world = mpi::getUniverse())
{
    perform_tensor_permute<Symmetry, Scalar, shift, 0, 1, 2, 3>(size, world);
    perform_tensor_permute<Symmetry, Scalar, shift, 0, 1, 3, 2>(size, world);
    perform_tensor_permute<Symmetry, Scalar, shift, 0, 3, 1, 2>(size, world);
    perform_tensor_permute<Symmetry, Scalar, shift, 0, 2, 1, 3>(size, world);
    perform_tensor_permute<Symmetry, Scalar, shift, 0, 2, 3, 1>(size, world);
    perform_tensor_permute<Symmetry, Scalar, shift, 0, 3, 2, 1>(size, world);
    perform_tensor_permute<Symmetry, Scalar, shift, 1, 0, 2, 3>(size, world);
    perform_tensor_permute<Symmetry, Scalar, shift, 1, 0, 3, 2>(size, world);
    perform_tensor_permute<Symmetry, Scalar, shift, 3, 0, 1, 2>(size, world);
    perform_tensor_permute<Symmetry, Scalar, shift, 2, 0, 1, 3>(size, world);
    perform_tensor_permute<Symmetry, Scalar, shift, 2, 0, 3, 1>(size, world);
    perform_tensor_permute<Symmetry, Scalar, shift, 3, 0, 2, 1>(size, world);
    perform_tensor_permute<Symmetry, Scalar, shift, 1, 2, 0, 3>(size, world);
    perform_tensor_permute<Symmetry, Scalar, shift, 1, 3, 0, 2>(size, world);
    perform_tensor_permute<Symmetry, Scalar, shift, 3, 1, 0, 2>(size, world);
    perform_tensor_permute<Symmetry, Scalar, shift, 2, 1, 0, 3>(size, world);
    perform_tensor_permute<Symmetry, Scalar, shift, 2, 3, 0, 1>(size, world);
    perform_tensor_permute<Symmetry, Scalar, shift, 3, 2, 0, 1>(size, world);
    perform_tensor_permute<Symmetry, Scalar, shift, 1, 2, 3, 0>(size, world);
    perform_tensor_permute<Symmetry, Scalar, shift, 1, 3, 2, 0>(size, world);
    perform_tensor_permute<Symmetry, Scalar, shift, 3, 1, 2, 0>(size, world);
    perform_tensor_permute<Symmetry, Scalar, shift, 2, 1, 3, 0>(size, world);
    perform_tensor_permute<Symmetry, Scalar, shift, 2, 3, 1, 0>(size, world);
    perform_tensor_permute<Symmetry, Scalar, shift, 3, 2, 1, 0>(size, world);

    // for (const auto& p : Permutation::all(4)) {
    //         auto tp = t.template permute<shift>(p);
    //         auto tplainp = tp.plainTensor();
    //         Eigen::Tensor<Scalar,4> tplainshuffle = tplain.shuffle(p.template pi_as_index<Eigen::Index>());
    //         auto check = tplainshuffle - tplainp;
    //         Eigen::Tensor<Scalar,0> zero = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>,
    //         4>{{Eigen::IndexPair<Eigen::Index>(0,0),
    //                                                                                                                      Eigen::IndexPair<Eigen::Index>(1,1),
    //                                                                                                                      Eigen::IndexPair<Eigen::Index>(2,2),
    //                                                                                                                      Eigen::IndexPair<Eigen::Index>(3,3)}});
    //         CHECK(zero() == doctest::Approx(0.));
    // }
    // std::cout << "shift=-2, total hits=" << tree_cache<-2,2,2,Symmetry>.cache.stats().total_hits() << endl; // Hits for any key
    // std::cout << "shift=-2, total misses=" << tree_cache<-2,2,2,Symmetry>.cache.stats().total_misses() << endl; // Misses for any key
    // std::cout << "shift=-2, hit rate=" << tree_cache<-2,2,2,Symmetry>.cache.stats().hit_rate() << endl; // Hit rate in [0, 1]

    // std::cout << "shift=-1, total hits=" << tree_cache<-1,2,2,Symmetry>.cache.stats().total_hits() << endl; // Hits for any key
    // std::cout << "shift=-1, total misses=" << tree_cache<-1,2,2,Symmetry>.cache.stats().total_misses() << endl; // Misses for any key
    // std::cout << "shift=-1, hit rate=" << tree_cache<-1,2,2,Symmetry>.cache.stats().hit_rate() << endl; // Hit rate in [0, 1]

    // std::cout << "shift=0, total hits=" << tree_cache<0,2,2,Symmetry>.cache.stats().total_hits() << endl; // Hits for any key
    // std::cout << "shift=0, total misses=" << tree_cache<0,2,2,Symmetry>.cache.stats().total_misses() << endl; // Misses for any key
    // std::cout << "shift=0, hit rate=" << tree_cache<0,2,2,Symmetry>.cache.stats().hit_rate() << endl; // Hit rate in [0, 1]

    // std::cout << "shift=1, total hits=" << tree_cache<1,2,2,Symmetry>.cache.stats().total_hits() << endl; // Hits for any key
    // std::cout << "shift=1, total misses=" << tree_cache<1,2,2,Symmetry>.cache.stats().total_misses() << endl; // Misses for any key
    // std::cout << "shift=1, hit rate=" << tree_cache<1,2,2,Symmetry>.cache.stats().hit_rate() << endl; // Hit rate in [0, 1]

    // std::cout << "shift=2, total hits=" << tree_cache<2,2,2,Symmetry>.cache.stats().total_hits() << endl; // Hits for any key
    // std::cout << "shift=2, total misses=" << tree_cache<2,2,2,Symmetry>.cache.stats().total_misses() << endl; // Misses for any key
    // std::cout << "shift=2, hit rate=" << tree_cache<2,2,2,Symmetry>.cache.stats().hit_rate() << endl; // Hit rate in [0, 1]
};

template <typename Symmetry, typename Scalar, std::size_t... per>
void perform_tensor_permute_intern(const std::size_t size, mpi::XpedWorld& world = mpi::getUniverse())
{
    // CTF::World world(comm);
    SPDLOG_WARN("Permute intern: Number of processes in tensor-test #={}", world.np);
    SPDLOG_WARN("Permute intern: I am process number #={}", world.rank);
    // SPDLOG_WARN("Permute intern: World #={}", world.comm);
    Qbasis<Symmetry, 1> B, C, D, E;
    if(world.rank == 0) {
        B.setRandom(size);
        C.setRandom(size);
        D.setRandom(size);
        E.setRandom(size);
        // std::cout << B << std::endl;
        // std::cout << C << std::endl;
        // std::cout << D << std::endl;
        // std::cout << E << std::endl;
    }
    XPED_MPI_BARRIER(world.comm)
    mpi::broadcast(B, world.rank, 0, world);
    mpi::broadcast(C, world.rank, 0, world);
    mpi::broadcast(D, world.rank, 0, world);
    mpi::broadcast(E, world.rank, 0, world);
    // if(world.rank == 1) {
    //     std::cout << "process " << world.rank << std::endl;
    //     std::cout << B << std::endl;
    //     std::cout << C << std::endl;
    //     std::cout << D << std::endl;
    //     std::cout << E << std::endl;
    // }
    // XPED_MPI_BARRIER(world.comm)
    // if(world.rank == 2) {
    //     std::cout << "process " << world.rank << std::endl;
    //     std::cout << B << std::endl;
    //     std::cout << C << std::endl;
    //     std::cout << D << std::endl;
    //     std::cout << E << std::endl;
    // }
    // XPED_MPI_BARRIER(world.comm)
    // if(world.rank == 3) {
    //     std::cout << "process " << world.rank << std::endl;
    //     std::cout << B << std::endl;
    //     std::cout << C << std::endl;
    //     std::cout << D << std::endl;
    //     std::cout << E << std::endl;
    // }

    [[maybe_unused]] std::array<std::size_t, 4> perm{per...};
    SPDLOG_CRITICAL("Permutation: {} {} {} {}", perm[0], perm[1], perm[2], perm[3]);
    XPED_MPI_BARRIER(world.comm)
    static thread_local std::mt19937 engine(std::random_device{}());
    Tensor<Scalar, 4, 0, Symmetry> t({{B, B, B, B}}, {{}}, world);
    // if(world.rank == 0) { std::cout << t << std::endl; }
    t.setRandom(engine);
    SPDLOG_WARN("Tensor t set to Random.");
    XPED_MPI_BARRIER(world.comm)
    auto tplain = t.plainTensor();
    SPDLOG_WARN("Computed plain tensor.");
    XPED_MPI_BARRIER(world.comm)
    auto tp = t.template permute<0, per...>();
    SPDLOG_WARN("Computed permutation of tensor.");
    XPED_MPI_BARRIER(world.comm)

    PlainInterface::TType<Scalar, 4> tplainshuffle = PlainInterface::shuffle<Scalar, 4, per...>(tplain);
    SPDLOG_WARN("Computed plain shuffle of tensor.");
    XPED_MPI_BARRIER(world.comm)
    auto tplainp = tp.plainTensor();
    SPDLOG_WARN("Computed plain tensor of permuted tensor.");
    XPED_MPI_BARRIER(world.comm)
#ifdef XPED_USE_ARRAY_TENSOR_LIB
    auto check = nda::make_ein_sum<Scalar, 0, 1, 2, 3>(nda::ein<0, 1, 2, 3>(tplainp) - nda::ein<0, 1, 2, 3>(tplainshuffle));
#elif defined(XPED_USE_CYCLOPS_TENSOR_LIB)
    auto dims = PlainInterface::dimensions<Scalar, 4>(tplainshuffle);
    auto check = PlainInterface::construct<Scalar>(dims, world);
    check["ijkl"] = tplainshuffle["ijkl"] - tplainp["ijkl"];
#else
    Eigen::Tensor<Scalar, 4> check = tplainshuffle - tplainp;
#endif
    auto zero_ = PlainInterface::contract<Scalar, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3>(check, check);
    // Eigen::Tensor<Scalar,4> tplainshuffle = tplain.shuffle(p);
    // auto tplainp = tp.plainTensor();
    // auto check = tplainshuffle - tplainp;
    // Eigen::Tensor<Scalar,0> zero_ = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
    //                                                                                                               Eigen::IndexPair<Eigen::Index>(1,1),
    //                                                                                                               Eigen::IndexPair<Eigen::Index>(2,2),
    //                                                                                                               Eigen::IndexPair<Eigen::Index>(3,3)}});
    Scalar zero = PlainInterface::getVal<Scalar, 0>(zero_, {{}});
    CHECK(std::abs(zero) == doctest::Approx(0.));
}

template <typename Symmetry, typename Scalar>
void test_tensor_permute_within_codomain(const std::size_t size, mpi::XpedWorld& world)
{
    // Qbasis<Symmetry,1> F; F.setRandom(50);
    // Tensor<0,3,Symmetry> three({{}},{{F,F,F}}); three.setRandom();
    // auto threep = three.plainTensor();
    // auto tp=three.template permute<0,2,0,1>();
    // PlainInterface::TType<Scalar,3> tplainshuffle = PlainInterface<M_MATRIXLIB,
    // M_TENSORLIB>::shuffle<Scalar,3,2,0,1>(threep); auto tplainp = tp.plainTensor(); auto check =
    // nda::make_ein_sum<Scalar,0,1,2,3>(nda::ein<0,1,2,3>(tplainshuffle) - nda::ein<0,1,2,3>(tplainshuffle));

    // Tensor<Scalar, 0, 4, Symmetry> t({{}}, {{B, C, D, E}});
    // t.setRandom();
    // auto tplain = t.plainTensor();
    // Permutation ptriv(std::array<std::size_t,0>{{}});

    perform_tensor_permute_intern<Symmetry, Scalar, 0, 1, 2, 3>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 0, 1, 3, 2>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 0, 3, 1, 2>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 0, 2, 1, 3>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 0, 2, 3, 1>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 0, 3, 2, 1>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 1, 0, 2, 3>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 1, 0, 3, 2>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 3, 0, 1, 2>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 2, 0, 1, 3>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 2, 0, 3, 1>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 3, 0, 2, 1>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 1, 2, 0, 3>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 1, 3, 0, 2>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 3, 1, 0, 2>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 2, 1, 0, 3>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 2, 3, 0, 1>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 3, 2, 0, 1>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 1, 2, 3, 0>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 1, 3, 2, 0>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 3, 1, 2, 0>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 2, 1, 3, 0>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 2, 3, 1, 0>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 3, 2, 1, 0>(size, world);

    // for (const auto& p : Permutation::all(4)) {
    //         auto test = t.template permute<0,2,1,3>();
    //         auto tp = t.permute(ptriv,p);
    //         Eigen::Tensor<Scalar,4> tplainshuffle = tplain.shuffle(p.pi_as_index<Eigen::Index>());
    //         auto tplainp = tp.plainTensor();
    //         auto check = tplainshuffle - tplainp;
    //         Eigen::Tensor<Scalar,0> zero_ = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>,
    //         4>{{Eigen::IndexPair<Eigen::Index>(0,0),
    //                                                                                                                       Eigen::IndexPair<Eigen::Index>(1,1),
    //                                                                                                                       Eigen::IndexPair<Eigen::Index>(2,2),
    //                                                                                                                       Eigen::IndexPair<Eigen::Index>(3,3)}});
    //         Scalar zero = zero_();
    //         CHECK(zero == doctest::Approx(0.));
    // }
}

template <typename Symmetry, typename Scalar>
void test_tensor_permute_within_domain(const std::size_t size, mpi::XpedWorld& world)
{
    // Tensor<Scalar, 4, 0, Symmetry> t({{B, C, D, E}}, {{}});
    // t.setRandom();
    // auto tplain = t.plainTensor();
    perform_tensor_permute_intern<Symmetry, Scalar, 0, 1, 2, 3>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 0, 1, 3, 2>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 0, 3, 1, 2>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 0, 2, 1, 3>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 0, 2, 3, 1>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 0, 3, 2, 1>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 1, 0, 2, 3>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 1, 0, 3, 2>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 3, 0, 1, 2>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 2, 0, 1, 3>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 2, 0, 3, 1>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 3, 0, 2, 1>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 1, 2, 0, 3>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 1, 3, 0, 2>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 3, 1, 0, 2>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 2, 1, 0, 3>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 2, 3, 0, 1>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 3, 2, 0, 1>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 1, 2, 3, 0>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 1, 3, 2, 0>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 3, 1, 2, 0>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 2, 1, 3, 0>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 2, 3, 1, 0>(size, world);
    perform_tensor_permute_intern<Symmetry, Scalar, 3, 2, 1, 0>(size, world);
    // Permutation ptriv(std::array<std::size_t,0>{{}});

    // for (const auto& p : Permutation::all(4)) {
    //         auto tp = t.permute(p,ptriv);
    //         Eigen::Tensor<Scalar,4> tplainshuffle = tplain.shuffle(p.pi_as_index<Eigen::Index>());
    //         auto tplainp = tp.plainTensor();
    //         auto check = tplainshuffle - tplainp;
    //         Eigen::Tensor<Scalar,0> zero_ = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>,
    //         4>{{Eigen::IndexPair<Eigen::Index>(0,0),
    //                                                                                                                       Eigen::IndexPair<Eigen::Index>(1,1),
    //                                                                                                                       Eigen::IndexPair<Eigen::Index>(2,2),
    //                                                                                                                       Eigen::IndexPair<Eigen::Index>(3,3)}});
    //         Scalar zero = zero_();
    //         CHECK(zero == doctest::Approx(0.));
    // }
}

template <typename Scalar, typename Symmetry>
void test_tensor_transformation_to_plain(const Qbasis<Symmetry, 1>& B, const Qbasis<Symmetry, 1>& C, mpi::XpedWorld& world)
{
    // CTF::World world(comm);
    SPDLOG_CRITICAL("Number of processes in tensor-test #={}", world.np);
    SPDLOG_CRITICAL("I am process number #={}", world.rank);
    // SPDLOG_INFO("basis B");
    // for(const auto& [q, pos, plain] : B.data_) { SPDLOG_INFO("QN: {}, deg={}", q.data[0], plain.dim()); }
    // SPDLOG_INFO("basis C");
    // for(const auto& [q, pos, plain] : C.data_) { SPDLOG_INFO("QN: {}, deg={}", q.data[0], plain.dim()); }

    static thread_local std::mt19937 engine(std::random_device{}());
    Tensor<Scalar, 2, 2, Symmetry> t({{B, C}}, {{B, C}}, world);
    t.setRandom(engine);

    // if(world.rank == 0) { std::cout << t << std::endl; }
    // t.print(std::cout, true);
    // std::cout << std::endl;

    // SPDLOG_INFO(t);
    auto tplain = t.plainTensor();
    // std::cout << tplain << std::endl;
    // tplain.print(std::cout, true);
    // std::cout << std::endl;
    auto norm_ = PlainInterface::contract<Scalar, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3>(tplain, tplain.conjugate());
    // auto norm_ = PlainInterface::contract<Scalar, 2, 2, 0, 1, 1, 0>(tplain, tplain.conjugate());
    Scalar norm = PlainInterface::getVal<Scalar, 0>(norm_, {{}});
    CHECK(t.squaredNorm() == doctest::Approx(norm.real()));
}
