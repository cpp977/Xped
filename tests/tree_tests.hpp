template <typename Symmetry>
void test_tree_swap(const Qbasis<Symmetry, 1>& B, const Qbasis<Symmetry, 1>& C, const Qbasis<Symmetry, 1>& D, const Qbasis<Symmetry, 1>& E)
{
    auto BC = B.combine(C);
    auto BCD = BC.combine(D);
    auto BCDE = BCD.combine(E);

    for(std::size_t pos = 0ul; pos < 3; pos++) {
        for(const auto [q, num, plain] : BCDE) {
            for(const auto& tree : BCDE.tree(q)) {
                std::unordered_map<FusionTree<4, Symmetry>, typename Symmetry::Scalar> check;
                auto transformed = tree.swap(pos);
                for(const auto& [tree_p, coeff] : transformed) {
                    auto inv = tree_p.swap(pos);
                    for(const auto& [tree_inv, coeff2] : inv) {
                        auto it = check.find(tree_inv);
                        if(it == check.end()) {
                            check.insert(std::make_pair(tree_inv, coeff * coeff2));
                        } else
                            (check[tree_inv] += coeff * coeff2);
                    }
                }
                for(const auto& [tree_check, coeff] : check) {
                    if(tree_check == tree) {
                        CHECK(coeff == doctest::Approx(1.));
                    } else {
                        CHECK(coeff == doctest::Approx(0.));
                    }
                }
            }
        }
    }
};

template <typename Symmetry>
void test_tree_permute(const Qbasis<Symmetry, 1>& B, const Qbasis<Symmetry, 1>& C, const Qbasis<Symmetry, 1>& D, const Qbasis<Symmetry, 1>& E)
{
    auto BC = B.combine(C);
    auto BCD = BC.combine(D);
    auto BCDE = BCD.combine(E);

    for(const auto& p : Permutation<4>::all()) {
        for(const auto& [q, num, plain] : BCDE) {
            for(const auto& tree : BCDE.tree(q)) {
                std::unordered_map<FusionTree<4, Symmetry>, typename Symmetry::Scalar> check;
                auto transformed = tree.permute(p);
                for(const auto& [tree_p, coeff] : transformed) {
                    auto inv = tree_p.permute(p.inverse());
                    for(const auto& [tree_inv, coeff2] : inv) {
                        auto it = check.find(tree_inv);
                        if(it == check.end()) {
                            check.insert(std::make_pair(tree_inv, coeff * coeff2));
                        } else
                            (check[tree_inv] += coeff * coeff2);
                    }
                }
                for(const auto& [tree_check, coeff] : check) {
                    if(tree_check == tree) {
                        CHECK(coeff == doctest::Approx(1.));
                    } else {
                        CHECK(coeff == doctest::Approx(0.));
                    }
                }
            }
        }
    }
};

template <int shift, typename Symmetry>
void test_tree_pair_turn(const Qbasis<Symmetry, 1>& B, const Qbasis<Symmetry, 1>& C)
{
    auto BC = B.combine(C);

    for(const auto& [q, num, plain] : BC) {
        for(const auto& t1 : BC.tree({q})) {
            for(const auto& t2 : BC.tree({q})) {
                std::unordered_map<std::pair<FusionTree<2, Symmetry>, FusionTree<2, Symmetry>>, typename Symmetry::Scalar> check;
                for(const auto& [trees, coeff] : treepair::turn<shift>(t1, t2)) {
                    auto [t1p, t2p] = trees;
                    for(const auto& [retrees, recoeff] : treepair::turn<-shift>(t1p, t2p)) {
                        // auto [t1pp,t2pp] = retrees;
                        auto it = check.find(retrees);
                        if(it == check.end()) {
                            check.insert(std::make_pair(retrees, coeff * recoeff));
                        } else
                            (check[retrees] += coeff * recoeff);
                    }
                }
                for(const auto& [tree_check, coeff] : check) {
                    auto [t1_check, t2_check] = tree_check;
                    if(t1_check == t1 and t2_check == t2) {
                        CHECK(coeff == doctest::Approx(1.));
                    } else {
                        CHECK(coeff == doctest::Approx(0.));
                    }
                }
            }
        }
    }
};

template <int shift, typename Symmetry>
void test_tree_pair_permute(const Qbasis<Symmetry, 1>& B, const Qbasis<Symmetry, 1>& C)
{
    auto BC = B.combine(C);
    for(const auto& p : Permutation<4>::all()) {
        for(const auto& [q, num, plain] : BC) {
            for(const auto& t1 : BC.tree({q})) {
                for(const auto& t2 : BC.tree({q})) {
                    std::unordered_map<std::pair<FusionTree<2, Symmetry>, FusionTree<2, Symmetry>>, typename Symmetry::Scalar> check;
                    for(const auto& [trees, coeff] : treepair::permute<shift>(t1, t2, p)) {
                        auto [t1p, t2p] = trees;
                        for(const auto& [retrees, recoeff] : treepair::permute<-shift>(t1p, t2p, p.inverse())) {
                            // auto [t1pp,t2pp] = retrees;
                            auto it = check.find(retrees);
                            if(it == check.end()) {
                                check.insert(std::make_pair(retrees, coeff * recoeff));
                            } else
                                (check[retrees] += coeff * recoeff);
                        }
                    }
                    for(const auto& [tree_check, coeff] : check) {
                        auto [t1_check, t2_check] = tree_check;
                        if(t1_check == t1 and t2_check == t2) {
                            CHECK(coeff == doctest::Approx(1.));
                        } else {
                            CHECK(coeff == doctest::Approx(0.));
                        }
                    }
                }
            }
        }
    }

#ifdef XPED_CACHE_PERMUTE_OUTPUT
    std::cout << "shift=-2, total hits=" << tree_cache<-2, 2, 2, Symmetry>.cache.stats().total_hits() << endl; // Hits for any key
    std::cout << "shift=-2, total misses=" << tree_cache<-2, 2, 2, Symmetry>.cache.stats().total_misses() << endl; // Misses for any key
    std::cout << "shift=-2, hit rate=" << tree_cache<-2, 2, 2, Symmetry>.cache.stats().hit_rate() << endl; // Hit rate in [0, 1]

    std::cout << "shift=-1, total hits=" << tree_cache<-1, 2, 2, Symmetry>.cache.stats().total_hits() << endl; // Hits for any key
    std::cout << "shift=-1, total misses=" << tree_cache<-1, 2, 2, Symmetry>.cache.stats().total_misses() << endl; // Misses for any key
    std::cout << "shift=-1, hit rate=" << tree_cache<-1, 2, 2, Symmetry>.cache.stats().hit_rate() << endl; // Hit rate in [0, 1]

    std::cout << "shift=0, total hits=" << tree_cache<0, 2, 2, Symmetry>.cache.stats().total_hits() << endl; // Hits for any key
    std::cout << "shift=0, total misses=" << tree_cache<0, 2, 2, Symmetry>.cache.stats().total_misses() << endl; // Misses for any key
    std::cout << "shift=0, hit rate=" << tree_cache<0, 2, 2, Symmetry>.cache.stats().hit_rate() << endl; // Hit rate in [0, 1]

    std::cout << "shift=1, total hits=" << tree_cache<1, 2, 2, Symmetry>.cache.stats().total_hits() << endl; // Hits for any key
    std::cout << "shift=1, total misses=" << tree_cache<1, 2, 2, Symmetry>.cache.stats().total_misses() << endl; // Misses for any key
    std::cout << "shift=1, hit rate=" << tree_cache<1, 2, 2, Symmetry>.cache.stats().hit_rate() << endl; // Hit rate in [0, 1]

    std::cout << "shift=2, total hits=" << tree_cache<2, 2, 2, Symmetry>.cache.stats().total_hits() << endl; // Hits for any key
    std::cout << "shift=2, total misses=" << tree_cache<2, 2, 2, Symmetry>.cache.stats().total_misses() << endl; // Misses for any key
    std::cout << "shift=2, hit rate=" << tree_cache<2, 2, 2, Symmetry>.cache.stats().hit_rate() << endl; // Hit rate in [0, 1]
#endif
};
