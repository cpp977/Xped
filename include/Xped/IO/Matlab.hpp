#ifndef XPED_MATLAB_HPP_
#define XPED_MATLAB_HPP_

#include <highfive/H5File.hpp>

#include "Xped/Core/Tensor.hpp"

namespace Xped::IO {
template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>
loadMatlabTensor(const HighFive::Group& t,
                 const HighFive::Group& base,
                 std::array<bool, Rank + CoRank> conj = std::array<bool, Rank + CoRank>{},
                 int qn_scale = 1)
{
    constexpr std::size_t full_rank = Rank + CoRank;
    auto get_indices = [](auto combined_index, std::array<std::size_t, full_rank> dims) -> auto
    {
        std::array<std::size_t, dims.size()> out{};
        std::size_t divisor = combined_index;
        std::size_t count = 0;
        while(divisor > 0) {
            std::size_t tmp = divisor / dims[count];
            std::size_t remainder = divisor % dims[count];
            assert(count < dims.size());
            out[count] = remainder;
            count++;
            divisor = tmp;
        }
        return out;
    };

    std::array<Qbasis<Symmetry, 1>, full_rank> basis;
    std::array<std::size_t, full_rank> dims;

    auto charges_ref_dat = t.getDataSet("charges");
    std::vector<HighFive::Reference> charges_ref;
    charges_ref_dat.read(charges_ref);
    std::array<std::vector<std::vector<double>>, full_rank> charges;
    for(std::size_t r = 0; r < full_rank; ++r) {
        auto ch_dat = charges_ref[r].template dereference<HighFive::DataSet>(base);
        std::vector<std::vector<double>> ch;
        ch_dat.read(ch);
        charges[r] = ch;
        for(std::size_t j = 0; j < ch[0].size(); ++j) {
            basis[r].push_back({(qn_scale * static_cast<int>(std::round(ch[0][j]))) % Symmetry::MOD_N[0]}, ch[1][j]);
        }
        if(conj[r]) {
            basis[r].SET_CONJ();
            for(auto& [q, trees] : basis[r].allTrees()) {
                for(auto& tree : trees) { tree.IS_DUAL[0] = true; }
            }
        }
        basis[r].sort();
        dims[r] = ch[0].size();
    }

    auto raw_dat = t.getDataSet("vdat");
    std::vector<double> raw;
    raw_dat.read(raw);

    auto off_dat = t.getDataSet("voffs");
    std::vector<std::vector<Scalar>> off;
    off_dat.read(off);

    Tensor<Scalar, full_rank, 0, Symmetry, false, AllocationPolicy> tmp(basis, {{}});
    tmp.setZero();
    for(std::size_t i = 0; i < off.size(); ++i) {
        auto indices = get_indices(off[i][0], dims);
        FusionTree<full_rank, Symmetry> tree;
        tree.q_coupled = Symmetry::qvacuum();
        for(std::size_t r = 0; r < full_rank; ++r) {
            tree.q_uncoupled[r] = {(qn_scale * static_cast<int>(std::round(charges[r][0][indices[r]]))) % Symmetry::MOD_N[0]};
            tree.dims[r] = charges[r][1][indices[r]];
            if(conj[r]) { tree.IS_DUAL[r] = true; }
        }
        tree.computeDim();
        tree.computeIntermediates();
        // if constexpr(full_rank > 2) {
        //     tree.q_intermediates[0] = Symmetry::reduceSilent(tree.q_uncoupled[0], tree.q_uncoupled[1])[0];
        //     for(std::size_t intermediate = 1; intermediate < full_rank - 2; ++intermediate) {
        //         tree.q_intermediates[intermediate] =
        //             Symmetry::reduceSilent(tree.q_intermediates[intermediate - 1], tree.q_uncoupled[intermediate + 1])[0];
        //     }
        // }
        FusionTree<0, Symmetry> trivial;
        trivial.dim = 1;
        trivial.q_coupled = Symmetry::qvacuum();

        auto submatrix = tmp.subMatrix(tree, trivial);
        std::size_t begin = off[i][1];
        std::size_t end = (i == off.size() - 1) ? raw.size() : off[i + 1][1];
        for(auto i = begin; i < end; ++i) { submatrix(i - begin, 0) = raw[i]; }
    }
    return tmp.template permute<full_rank - Rank>(seq::make<std::size_t, full_rank>{});
}

} // namespace Xped::IO
#endif
