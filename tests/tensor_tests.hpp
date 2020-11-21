template<int shift, typename Symmetry>
void test_tensor_permute(const Qbasis<Symmetry,1>& B, const Qbasis<Symmetry,1>& C)
{
        // std::array<std::size_t,4> pi{};
        std::array<std::size_t,4> pi{};
        Permutation p(pi);

        Tensor<2,2,Symmetry> t({{B,C}},{{B,C}}); t.setRandom();
        // auto tp = t.permute<1>(p);
        auto tplain = t.plainTensor();

        // for (const auto& p : Permutation<4>::all()) {
        //         auto tp = t.permute<shift>(p);
        //         auto tplainp = tp.plainTensor();
        //         Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(p.pi);
        //         auto check = tplainshuffle - tplainp;
        //         Eigen::Tensor<double,0> zero = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
        //                                                                                                                      Eigen::IndexPair<Eigen::Index>(1,1),
        //                                                                                                                      Eigen::IndexPair<Eigen::Index>(2,2),
        //                                                                                                                      Eigen::IndexPair<Eigen::Index>(3,3)}});
        //         CHECK(zero() == doctest::Approx(0.));
        // }
};
