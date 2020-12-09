template<int shift, typename Symmetry>
void test_tensor_permute(const Qbasis<Symmetry,1>& B, const Qbasis<Symmetry,1>& C)
{
        Tensor<2,2,Symmetry> t({{B,C}},{{B,C}}); t.setRandom();
        auto tplain = t.plainTensor();

        for (const auto& p : Permutation<4>::all()) {
                auto tp = t.template permute<shift>(p);
                auto tplainp = tp.plainTensor();
                Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(p.template pi_as_index<Eigen::Index>());
                auto check = tplainshuffle - tplainp;
                Eigen::Tensor<double,0> zero = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                             Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                             Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                             Eigen::IndexPair<Eigen::Index>(3,3)}});
                CHECK(zero() == doctest::Approx(0.));
        }
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

template<typename Symmetry>
void test_tensor_permute_within_codomain(const Qbasis<Symmetry,1>& B, const Qbasis<Symmetry,1>& C, const Qbasis<Symmetry,1>& D, const Qbasis<Symmetry,1>& E)
{
        Tensor<0,4,Symmetry> t({{}},{{B,C,D,E}}); t.setRandom();
        auto tplain = t.plainTensor();
        Permutation<0> ptriv(std::array<std::size_t,0>{{}});

        for (const auto& p : Permutation<4>::all()) {
                auto tp = t.permute(ptriv,p);
                Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(p.pi_as_index<Eigen::Index>());
                auto tplainp = tp.plainTensor();
                auto check = tplainshuffle - tplainp;
                Eigen::Tensor<double,0> zero_ = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                              Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                              Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                              Eigen::IndexPair<Eigen::Index>(3,3)}});
                double zero = zero_();
                CHECK(zero == doctest::Approx(0.));
        }
}

template<typename Symmetry>
void test_tensor_permute_within_domain(const Qbasis<Symmetry,1>& B, const Qbasis<Symmetry,1>& C, const Qbasis<Symmetry,1>& D, const Qbasis<Symmetry,1>& E)
{
        Tensor<4,0,Symmetry> t({{B,C,D,E}},{{}}); t.setRandom();
        auto tplain = t.plainTensor();
        Permutation<0> ptriv(std::array<std::size_t,0>{{}});

        for (const auto& p : Permutation<4>::all()) {
                auto tp = t.permute(p,ptriv);
                Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(p.pi_as_index<Eigen::Index>());
                auto tplainp = tp.plainTensor();
                auto check = tplainshuffle - tplainp;
                Eigen::Tensor<double,0> zero_ = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                              Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                              Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                              Eigen::IndexPair<Eigen::Index>(3,3)}});
                double zero = zero_();
                CHECK(zero == doctest::Approx(0.));
        }
}

template<typename Symmetry>
void test_tensor_transformation_to_plain(const Qbasis<Symmetry,1>& B, const Qbasis<Symmetry,1>& C)
{
        Tensor<2,2,Symmetry> t({{B,C}},{{B,C}}); t.setRandom();
        auto tplain = t.plainTensor();
        Eigen::Tensor<double,0> norm_ = tplain.contract(tplain,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
                                                                                                                        Eigen::IndexPair<Eigen::Index>(1,1),
                                                                                                                        Eigen::IndexPair<Eigen::Index>(2,2),
                                                                                                                        Eigen::IndexPair<Eigen::Index>(3,3)}});
        double norm = norm_();
        CHECK(t.squaredNorm() == doctest::Approx(norm));
}
