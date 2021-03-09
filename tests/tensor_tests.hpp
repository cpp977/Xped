template<typename TensorT, int shift, std::size_t... per>
void perform_tensor_permute(const TensorT& t, tensortraits<M_TENSORLIB>::cTtype<double,4>& tplain)
{
        // std::array<Eigen::Index,4> p = {per...};
        // std::cout << "permutation: "; for (const auto& elem:p) {std::cout << elem << " ";} std::cout << ", shift=" << shift << std::endl;
        
        auto tp=t.template permute<shift,per...>();
        tensortraits<M_TENSORLIB>::cTtype<double,4> tplainshuffle = tensortraits<M_TENSORLIB>::shuffle<double,4,per...>(tplain);
        auto tplainp = tp.plainTensor();
#ifdef XPED_USE_ARRAY_TENSOR_LIB
        auto check = nda::make_ein_sum<double,0,1,2,3>(nda::ein<0,1,2,3>(tplainp) - nda::ein<0,1,2,3>(tplainshuffle));
#else
        Eigen::Tensor<double,4> check = tplainshuffle - tplainp;
#endif
        auto zero_ = tensortraits<M_TENSORLIB>::contract<double,4,4,0,0,1,1,2,2,3,3>(check,check);
        // auto check = tplainshuffle - tplainp;
        // Eigen::Tensor<double,0> zero_ = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
        //                                                                                                               Eigen::IndexPair<Eigen::Index>(1,1),
        //                                                                                                               Eigen::IndexPair<Eigen::Index>(2,2),
        //                                                                                                               Eigen::IndexPair<Eigen::Index>(3,3)}});
        double zero = zero_();
        CHECK(zero == doctest::Approx(0.));
}

template<int shift, typename Symmetry>
void test_tensor_permute(const Qbasis<Symmetry,1>& B, const Qbasis<Symmetry,1>& C)
{
        Tensor<2,2,Symmetry> t({{B,C}},{{B,C}}); t.setRandom();
        auto tplain = t.plainTensor();

        perform_tensor_permute<Tensor<2,2,Symmetry>,shift,0,1,2,3>(t,tplain);
        perform_tensor_permute<Tensor<2,2,Symmetry>,shift,0,1,3,2>(t,tplain);
        perform_tensor_permute<Tensor<2,2,Symmetry>,shift,0,3,1,2>(t,tplain);
        perform_tensor_permute<Tensor<2,2,Symmetry>,shift,0,2,1,3>(t,tplain);
        perform_tensor_permute<Tensor<2,2,Symmetry>,shift,0,2,3,1>(t,tplain);
        perform_tensor_permute<Tensor<2,2,Symmetry>,shift,0,3,2,1>(t,tplain);
        perform_tensor_permute<Tensor<2,2,Symmetry>,shift,1,0,2,3>(t,tplain);
        perform_tensor_permute<Tensor<2,2,Symmetry>,shift,1,0,3,2>(t,tplain);
        perform_tensor_permute<Tensor<2,2,Symmetry>,shift,3,0,1,2>(t,tplain);
        perform_tensor_permute<Tensor<2,2,Symmetry>,shift,2,0,1,3>(t,tplain);
        perform_tensor_permute<Tensor<2,2,Symmetry>,shift,2,0,3,1>(t,tplain);
        perform_tensor_permute<Tensor<2,2,Symmetry>,shift,3,0,2,1>(t,tplain);
        perform_tensor_permute<Tensor<2,2,Symmetry>,shift,1,2,0,3>(t,tplain);
        perform_tensor_permute<Tensor<2,2,Symmetry>,shift,1,3,0,2>(t,tplain);
        perform_tensor_permute<Tensor<2,2,Symmetry>,shift,3,1,0,2>(t,tplain);
        perform_tensor_permute<Tensor<2,2,Symmetry>,shift,2,1,0,3>(t,tplain);
        perform_tensor_permute<Tensor<2,2,Symmetry>,shift,2,3,0,1>(t,tplain);
        perform_tensor_permute<Tensor<2,2,Symmetry>,shift,3,2,0,1>(t,tplain);
        perform_tensor_permute<Tensor<2,2,Symmetry>,shift,1,2,3,0>(t,tplain);
        perform_tensor_permute<Tensor<2,2,Symmetry>,shift,1,3,2,0>(t,tplain);
        perform_tensor_permute<Tensor<2,2,Symmetry>,shift,3,1,2,0>(t,tplain);
        perform_tensor_permute<Tensor<2,2,Symmetry>,shift,2,1,3,0>(t,tplain);
        perform_tensor_permute<Tensor<2,2,Symmetry>,shift,2,3,1,0>(t,tplain);
        perform_tensor_permute<Tensor<2,2,Symmetry>,shift,3,2,1,0>(t,tplain);
        
        // for (const auto& p : Permutation<4>::all()) {
        //         auto tp = t.template permute<shift>(p);
        //         auto tplainp = tp.plainTensor();
        //         Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(p.template pi_as_index<Eigen::Index>());
        //         auto check = tplainshuffle - tplainp;
        //         Eigen::Tensor<double,0> zero = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
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

template<typename TensorT, std::size_t... per>
void perform_tensor_permute_intern(const TensorT& t, tensortraits<M_TENSORLIB>::cTtype<double,4>& tplain)
{
        // std::array<Eigen::Index,4> p = {per...};
        // std::cout << "permutation: "; for (const auto& elem:p) {std::cout << elem << " ";} std::cout << std::endl;

        auto tp=t.template permute<0,per...>();
        tensortraits<M_TENSORLIB>::Ttype<double,4> tplainshuffle = tensortraits<M_TENSORLIB>::shuffle<double,4,per...>(tplain);
        auto tplainp = tp.plainTensor();
#ifdef XPED_USE_ARRAY_TENSOR_LIB
        auto check = nda::make_ein_sum<double,0,1,2,3>(nda::ein<0,1,2,3>(tplainp) - nda::ein<0,1,2,3>(tplainshuffle));
#else
        Eigen::Tensor<double,4> check = tplainshuffle - tplainp;
#endif
        auto zero_ = tensortraits<M_TENSORLIB>::contract<double,4,4,0,0,1,1,2,2,3,3>(check,check);
        
        // Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(p);
        // auto tplainp = tp.plainTensor();
        // auto check = tplainshuffle - tplainp;
        // Eigen::Tensor<double,0> zero_ = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
        //                                                                                                               Eigen::IndexPair<Eigen::Index>(1,1),
        //                                                                                                               Eigen::IndexPair<Eigen::Index>(2,2),
        //                                                                                                               Eigen::IndexPair<Eigen::Index>(3,3)}});
        double zero = zero_();
        CHECK(zero == doctest::Approx(0.));
}

template<typename Symmetry>
void test_tensor_permute_within_codomain(const Qbasis<Symmetry,1>& B, const Qbasis<Symmetry,1>& C, const Qbasis<Symmetry,1>& D, const Qbasis<Symmetry,1>& E)
{
        // Qbasis<Symmetry,1> F; F.setRandom(50);
        // Tensor<0,3,Symmetry> three({{}},{{F,F,F}}); three.setRandom();
        // auto threep = three.plainTensor();
        // auto tp=three.template permute<0,2,0,1>();
        // tensortraits<M_TENSORLIB>::Ttype<double,3> tplainshuffle = tensortraits<M_TENSORLIB>::shuffle<double,3,2,0,1>(threep);
        // auto tplainp = tp.plainTensor();
        // auto check = nda::make_ein_sum<double,0,1,2,3>(nda::ein<0,1,2,3>(tplainshuffle) - nda::ein<0,1,2,3>(tplainshuffle));
        
        Tensor<0,4,Symmetry> t({{}},{{B,C,D,E}}); t.setRandom();
        auto tplain = t.plainTensor();
        // Permutation<0> ptriv(std::array<std::size_t,0>{{}});
        
        perform_tensor_permute_intern<Tensor<0,4,Symmetry>,0,1,2,3>(t,tplain);
        perform_tensor_permute_intern<Tensor<0,4,Symmetry>,0,1,3,2>(t,tplain);
        perform_tensor_permute_intern<Tensor<0,4,Symmetry>,0,3,1,2>(t,tplain);
        perform_tensor_permute_intern<Tensor<0,4,Symmetry>,0,2,1,3>(t,tplain);
        perform_tensor_permute_intern<Tensor<0,4,Symmetry>,0,2,3,1>(t,tplain);
        perform_tensor_permute_intern<Tensor<0,4,Symmetry>,0,3,2,1>(t,tplain);
        perform_tensor_permute_intern<Tensor<0,4,Symmetry>,1,0,2,3>(t,tplain);
        perform_tensor_permute_intern<Tensor<0,4,Symmetry>,1,0,3,2>(t,tplain);
        perform_tensor_permute_intern<Tensor<0,4,Symmetry>,3,0,1,2>(t,tplain);
        perform_tensor_permute_intern<Tensor<0,4,Symmetry>,2,0,1,3>(t,tplain);
        perform_tensor_permute_intern<Tensor<0,4,Symmetry>,2,0,3,1>(t,tplain);
        perform_tensor_permute_intern<Tensor<0,4,Symmetry>,3,0,2,1>(t,tplain);
        perform_tensor_permute_intern<Tensor<0,4,Symmetry>,1,2,0,3>(t,tplain);
        perform_tensor_permute_intern<Tensor<0,4,Symmetry>,1,3,0,2>(t,tplain);
        perform_tensor_permute_intern<Tensor<0,4,Symmetry>,3,1,0,2>(t,tplain);
        perform_tensor_permute_intern<Tensor<0,4,Symmetry>,2,1,0,3>(t,tplain);
        perform_tensor_permute_intern<Tensor<0,4,Symmetry>,2,3,0,1>(t,tplain);
        perform_tensor_permute_intern<Tensor<0,4,Symmetry>,3,2,0,1>(t,tplain);
        perform_tensor_permute_intern<Tensor<0,4,Symmetry>,1,2,3,0>(t,tplain);
        perform_tensor_permute_intern<Tensor<0,4,Symmetry>,1,3,2,0>(t,tplain);
        perform_tensor_permute_intern<Tensor<0,4,Symmetry>,3,1,2,0>(t,tplain);
        perform_tensor_permute_intern<Tensor<0,4,Symmetry>,2,1,3,0>(t,tplain);
        perform_tensor_permute_intern<Tensor<0,4,Symmetry>,2,3,1,0>(t,tplain);
        perform_tensor_permute_intern<Tensor<0,4,Symmetry>,3,2,1,0>(t,tplain);
        
        // for (const auto& p : Permutation<4>::all()) {
        //         auto test = t.template permute<0,2,1,3>();
        //         auto tp = t.permute(ptriv,p);
        //         Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(p.pi_as_index<Eigen::Index>());
        //         auto tplainp = tp.plainTensor();
        //         auto check = tplainshuffle - tplainp;
        //         Eigen::Tensor<double,0> zero_ = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
        //                                                                                                                       Eigen::IndexPair<Eigen::Index>(1,1),
        //                                                                                                                       Eigen::IndexPair<Eigen::Index>(2,2),
        //                                                                                                                       Eigen::IndexPair<Eigen::Index>(3,3)}});
        //         double zero = zero_();
        //         CHECK(zero == doctest::Approx(0.));
        // }
}

template<typename Symmetry>
void test_tensor_permute_within_domain(const Qbasis<Symmetry,1>& B, const Qbasis<Symmetry,1>& C, const Qbasis<Symmetry,1>& D, const Qbasis<Symmetry,1>& E)
{
        Tensor<4,0,Symmetry> t({{B,C,D,E}},{{}}); t.setRandom();
        auto tplain = t.plainTensor();

        perform_tensor_permute_intern<Tensor<4,0,Symmetry>,0,1,2,3>(t,tplain);
        perform_tensor_permute_intern<Tensor<4,0,Symmetry>,0,1,3,2>(t,tplain);
        perform_tensor_permute_intern<Tensor<4,0,Symmetry>,0,3,1,2>(t,tplain);
        perform_tensor_permute_intern<Tensor<4,0,Symmetry>,0,2,1,3>(t,tplain);
        perform_tensor_permute_intern<Tensor<4,0,Symmetry>,0,2,3,1>(t,tplain);
        perform_tensor_permute_intern<Tensor<4,0,Symmetry>,0,3,2,1>(t,tplain);
        perform_tensor_permute_intern<Tensor<4,0,Symmetry>,1,0,2,3>(t,tplain);
        perform_tensor_permute_intern<Tensor<4,0,Symmetry>,1,0,3,2>(t,tplain);
        perform_tensor_permute_intern<Tensor<4,0,Symmetry>,3,0,1,2>(t,tplain);
        perform_tensor_permute_intern<Tensor<4,0,Symmetry>,2,0,1,3>(t,tplain);
        perform_tensor_permute_intern<Tensor<4,0,Symmetry>,2,0,3,1>(t,tplain);
        perform_tensor_permute_intern<Tensor<4,0,Symmetry>,3,0,2,1>(t,tplain);
        perform_tensor_permute_intern<Tensor<4,0,Symmetry>,1,2,0,3>(t,tplain);
        perform_tensor_permute_intern<Tensor<4,0,Symmetry>,1,3,0,2>(t,tplain);
        perform_tensor_permute_intern<Tensor<4,0,Symmetry>,3,1,0,2>(t,tplain);
        perform_tensor_permute_intern<Tensor<4,0,Symmetry>,2,1,0,3>(t,tplain);
        perform_tensor_permute_intern<Tensor<4,0,Symmetry>,2,3,0,1>(t,tplain);
        perform_tensor_permute_intern<Tensor<4,0,Symmetry>,3,2,0,1>(t,tplain);
        perform_tensor_permute_intern<Tensor<4,0,Symmetry>,1,2,3,0>(t,tplain);
        perform_tensor_permute_intern<Tensor<4,0,Symmetry>,1,3,2,0>(t,tplain);
        perform_tensor_permute_intern<Tensor<4,0,Symmetry>,3,1,2,0>(t,tplain);
        perform_tensor_permute_intern<Tensor<4,0,Symmetry>,2,1,3,0>(t,tplain);
        perform_tensor_permute_intern<Tensor<4,0,Symmetry>,2,3,1,0>(t,tplain);
        perform_tensor_permute_intern<Tensor<4,0,Symmetry>,3,2,1,0>(t,tplain);
        // Permutation<0> ptriv(std::array<std::size_t,0>{{}});

        // for (const auto& p : Permutation<4>::all()) {
        //         auto tp = t.permute(p,ptriv);
        //         Eigen::Tensor<double,4> tplainshuffle = tplain.shuffle(p.pi_as_index<Eigen::Index>());
        //         auto tplainp = tp.plainTensor();
        //         auto check = tplainshuffle - tplainp;
        //         Eigen::Tensor<double,0> zero_ = check.contract(check,Eigen::array<Eigen::IndexPair<Eigen::Index>, 4>{{Eigen::IndexPair<Eigen::Index>(0,0),
        //                                                                                                                       Eigen::IndexPair<Eigen::Index>(1,1),
        //                                                                                                                       Eigen::IndexPair<Eigen::Index>(2,2),
        //                                                                                                                       Eigen::IndexPair<Eigen::Index>(3,3)}});
        //         double zero = zero_();
        //         CHECK(zero == doctest::Approx(0.));
        // }
}

template<typename Symmetry>
void test_tensor_transformation_to_plain(const Qbasis<Symmetry,1>& B, const Qbasis<Symmetry,1>& C)
{
        Tensor<2,2,Symmetry> t({{B,C}},{{B,C}}); t.setRandom();
        auto tplain = t.plainTensor();
        auto norm_ = tensortraits<M_TENSORLIB>::contract<double,4,4,0,0,1,1,2,2,3,3>(tplain,tplain);
        double norm = norm_();
        CHECK(t.squaredNorm() == doctest::Approx(norm));
}
