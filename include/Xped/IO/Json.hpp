#ifndef XPED_JSON_HPP_
#define XPED_JSON_HPP_

#include <fstream>

#include <nlohmann/json.hpp>

#include "fmt/ranges.h"

#include "Xped/Core/Tensor.hpp"
#include "Xped/Symmetry/functions.hpp"

namespace Xped::IO {

template <typename Symmetry>
auto loadSU2JsonBaseTensor(const std::filesystem::path& p, std::size_t elem)
{
    // fmt::print("Reading json: {}\n", p.string());
    constexpr std::size_t Rank = 1;
    constexpr std::size_t CoRank = 4;
    using Scalar = double;

    std::ifstream jsonf(p);
    auto data = nlohmann::json::parse(jsonf);
    constexpr std::size_t full_rank = Rank + CoRank;
    constexpr int sfull_rank = static_cast<int>(full_rank);
    auto locdim = data["su2_tensors"][elem]["physDim"].template get<PlainInterface::Indextype>();
    auto auxdim = data["su2_tensors"][elem]["auxDim"].template get<PlainInterface::Indextype>();
    // fmt::print("locdim={}, auxdim={}\n", locdim, auxdim);
    typename PlainInterface::TType<Scalar, full_rank> T =
        PlainInterface::construct<Scalar>(std::array<PlainInterface::Indextype, full_rank>{locdim, auxdim, auxdim, auxdim, auxdim});
    PlainInterface::setZero(T);
    // std::cout << std::setw(4) << data["su2_tensors"][0] << std::endl;
    auto entries = data["su2_tensors"][elem]["entries"].template get<std::vector<std::string>>();
    // fmt::print("{}\n", entries);
    std::vector<PlainInterface::Indextype> perm(auxdim, 0);
    for(PlainInterface::Indextype i = 1; i < auxdim; ++i) { perm[i - 1] = i; }
    for(const auto& elem : entries) {
        std::vector<std::string> vals;
        boost::split(vals, elem, [](char c) { return c == ' '; });
        PlainInterface::setVal<double, sfull_rank>(T,
                                                   std::array<PlainInterface::Indextype, sfull_rank>{(std::stoi(vals[0]) + 0) % locdim,
                                                                                                     perm[std::stoi(vals[1])],
                                                                                                     perm[std::stoi(vals[2])],
                                                                                                     perm[std::stoi(vals[3])],
                                                                                                     perm[std::stoi(vals[4])]},
                                                   std::stod(vals[5]));
    }
    auto irrep_string = data["su2_tensors"][elem]["meta"]["irreps"].template get<std::string>();
    std::vector<std::string> irreps;
    boost::split(irreps, irrep_string.substr(1, irrep_string.size() - 2), [](char c) { return c == ','; });
    Xped::Qbasis<Symmetry, 1> aux;
    if constexpr(Symmetry::ALL_IS_TRIVIAL) {
        aux.push_back(Symmetry::qvacuum(), auxdim);
    } else {
        for(auto& irrep : irreps) {
            if constexpr(Symmetry::ALL_ABELIAN) {
                if(std::stoi(irrep) == 0) {
                    aux.push_back(Symmetry::qvacuum(), 1);
                    continue;
                }
                aux.push_back({std::stoi(irrep)}, 1);
                aux.push_back({34 + std::stoi(irrep)}, 1);
            } else {
                aux.push_back({std::stoi(irrep) + 1}, 1);
            }
        }
    }
    aux.sort();
    Xped::Qbasis<Symmetry, 1> aux_l = aux;
    aux_l.SET_CONJ();
    Xped::Qbasis<Symmetry, 1> aux_t = aux;
    aux_t.SET_CONJ();
    Xped::Qbasis<Symmetry, 1> aux_r = aux;
    aux_r.SET_CONJ();
    Xped::Qbasis<Symmetry, 1> aux_b = aux;
    aux_b.SET_CONJ();
    Xped::Qbasis<Symmetry, 1> loc;
    if constexpr(Symmetry::ALL_IS_TRIVIAL) {
        loc.push_back(Symmetry::qvacuum(), 2);
    } else if constexpr(Symmetry::ALL_ABELIAN) {
        loc.push_back({+1}, 1);
        loc.push_back({+35}, 1);
    } else {
        loc.push_back({2}, 1);
    }
    loc.sort();
    loc.SET_CONJ();

    Xped::Tensor<Scalar, 1, 4, Symmetry> res({{loc}}, {{aux_t, aux_l, aux_b, aux_r}});
    // Xped::Tensor<Scalar, 5, 0, Symmetry> res({{aux_t, aux_l, aux_b, aux_r, loc}}, {{}});
    res.setZero();
    fmt::print("Domain: {}, Codomain: {}\n", res.coupledDomain().print(), res.coupledCodomain().print());
    auto U = res.unitaryDomain();
    fmt::print("U(rank={}): ({}x{})\n", U.NumDimensions, U.dimension(0), U.dimension(0));
    auto V = res.unitaryCodomain();
    fmt::print("V(rank={}): ({}x{}x{}x{}x{})\n", V.NumDimensions, V.dimension(0), V.dimension(1), V.dimension(2), V.dimension(3), V.dimension(4));

    // std::cout << std::setprecision(3) << U << std::endl;
    // {
    //     auto check = PlainInterface::contract<Scalar, 5 + 1, 5 + 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4>(V, V);
    //     Eigen::Tensor<double, 2> Id = check;
    //     Id.setZero();
    //     for(auto i = 0; i < check.dimension(0); ++i) Id(i, i) = 1.;
    //     std::cout << "codomain: " << (check - Id).sum() << std::endl;
    // }
    // {
    //     auto check = PlainInterface::contract<Scalar, 0 + 1, 0 + 1>(U, U);
    //     Eigen::Tensor<double, 2> Id = check;
    //     Id.setZero();
    //     for(auto i = 0; i < check.dimension(0); ++i) Id(i, i) = 1.;
    //     std::cout << "domain: " << (check - Id).sum() << std::endl;
    // }
    auto tmp = PlainInterface::contract<Scalar, 1 + 1, 5, 0, 0>(U, T);
    // auto tmp = PlainInterface::contract<Scalar, 5 + 1, 4 + 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4>(U, T);
    auto Tmat = PlainInterface::contract<Scalar, 4 + 1, 4 + 1, 1, 0, 2, 1, 3, 2, 4, 3>(tmp, V);
    std::cout << Tmat.dimension(0) << "x" << Tmat.dimension(1) << std::endl;
    // std::cout << "Tmat: " << Tmat.dimension(0) << "x" << Tmat.dimension(1) << "\n" << Tmat << std::endl;
    // Eigen::Tensor<double, 2> Tmatcheck = T.reshape(std::array<int, 2>{locdim, auxdim * auxdim * auxdim * auxdim});
    // std::cout << "Tmatcheck: " << (Tmatcheck - Tmat).sum() << std::endl;
    Eigen::Map<Eigen::MatrixXd> M(Tmat.data(), Tmat.dimension(0), Tmat.dimension(1));

    for(auto i = 0ul; i < res.sector().size(); ++i) {
        // fmt::print("q={}\n", Sym::format<Symmetry>(res.sector(i)));
        // fmt::print("Take block at ({},{}) with size={}x{}\n",
        //            res.coupledDomain().full_outer_num(res.sector(i)),
        //            res.coupledCodomain().full_outer_num(res.sector(i)),
        //            Symmetry::degeneracy(res.sector(i)) * PlainInterface::rows(res.block(i)),
        //            Symmetry::degeneracy(res.sector(i)) * PlainInterface::cols(res.block(i)));
        Eigen::MatrixXd sub = M.block(res.coupledDomain().full_outer_num(res.sector(i)),
                                      res.coupledCodomain().full_outer_num(res.sector(i)),
                                      Symmetry::degeneracy(res.sector(i)) * PlainInterface::rows(res.block(i)),
                                      Symmetry::degeneracy(res.sector(i)) * PlainInterface::cols(res.block(i)));
        // std::cout << std::setprecision(2) << sub << std::endl;
        for(auto row = 0; row < PlainInterface::rows(res.block(i)); ++row) {
            for(auto col = 0; col < PlainInterface::cols(res.block(i)); ++col) {
                res.block(i)(row, col) = sub(row * Symmetry::degeneracy(res.sector(i)), col * Symmetry::degeneracy(res.sector(i)));
            }
        }
    }
    // res.print(std::cout, true);
    // std::cout << std::endl;
    std::cout << "T-T_constructed: " << (T - res.plainTensor()).sum() << std::endl;
    return res;
}

template <typename Symmetry>
Tensor<double, 4, 1, Symmetry> loadSU2JsonTensor(const std::filesystem::path& p)
{
    std::ifstream jsonf(p);
    auto data = nlohmann::json::parse(jsonf);
    auto num_tensors = data["coeffs"][0]["numEntries"].template get<std::size_t>();
    auto coeffs_string = data["coeffs"][0]["entries"].template get<std::vector<std::string>>();
    assert(coeffs_string.size() == num_tensors);
    std::vector<double> coeffs(num_tensors);
    for(const auto& coeff : coeffs_string) {
        std::vector<std::string> vals;
        boost::split(vals, coeff, [](char c) { return c == ' '; });
        coeffs[std::stoi(vals[0])] = std::stod(vals[1]);
    }
    auto tmp = (coeffs[0] * loadSU2JsonBaseTensor<Symmetry>(p, 0)).eval();
    for(auto i = 1ul; i < num_tensors; ++i) { tmp += coeffs[i] * loadSU2JsonBaseTensor<Symmetry>(p, i); }
    return tmp.template permute<-3, 2, 1, 4, 3, 0>();
}

template <typename Symmetry>
auto loadU0JsonTensor(const std::filesystem::path& p)
{
    // fmt::print("Reading json: {}\n", p.string());
    constexpr std::size_t Rank = 1;
    constexpr std::size_t CoRank = 4;
    using Scalar = double;

    std::ifstream jsonf(p);
    auto data = nlohmann::json::parse(jsonf);
    constexpr std::size_t full_rank = Rank + CoRank;
    constexpr int sfull_rank = static_cast<int>(full_rank);
    auto locdim = data["sites"][0]["physDim"].template get<PlainInterface::Indextype>();
    auto auxdim = data["sites"][0]["auxDim"].template get<PlainInterface::Indextype>();
    typename PlainInterface::TType<Scalar, full_rank> T =
        PlainInterface::construct<Scalar>(std::array<PlainInterface::Indextype, full_rank>{locdim, auxdim, auxdim, auxdim, auxdim});
    PlainInterface::setZero(T);
    // std::cout << std::setw(4) << data[0]["su2_tensors"][0] << std::endl;
    auto entries = data["sites"][0]["entries"].template get<std::vector<std::string>>();
    fmt::print("{}\n", entries);
    // std::vector<PlainInterface::Indextype> perm = {1, 2, 0};
    for(const auto& elem : entries) {
        std::vector<std::string> vals;
        boost::split(vals, elem, [](char c) { return c == ' '; });
        fmt::print("vals={}\n", vals);
        PlainInterface::setVal<double, sfull_rank>(
            T,
            std::array<PlainInterface::Indextype, sfull_rank>{
                (std::stoi(vals[0]) + 0) % locdim, std::stoi(vals[1]), std::stoi(vals[2]), std::stoi(vals[3]), std::stoi(vals[4])},
            std::stod(vals[5]));
    }
    std::cout << "T:\n" << T << std::endl;
    Xped::Qbasis<Symmetry, 1> aux;
    aux.push_back(Symmetry::qvacuum(), auxdim);
    aux.sort();
    Xped::Qbasis<Symmetry, 1> aux_l = aux;
    aux_l.SET_CONJ();
    Xped::Qbasis<Symmetry, 1> aux_t = aux;
    aux_t.SET_CONJ();
    Xped::Qbasis<Symmetry, 1> aux_r = aux;
    aux_r.SET_CONJ();
    Xped::Qbasis<Symmetry, 1> aux_b = aux;
    aux_b.SET_CONJ();
    Xped::Qbasis<Symmetry, 1> loc;
    if constexpr(Symmetry::ALL_IS_TRIVIAL) {
        loc.push_back(Symmetry::qvacuum(), 2);
    } else if constexpr(Symmetry::ALL_ABELIAN) {
        loc.push_back({+1}, 1);
        loc.push_back({+35}, 1);
    } else {
        loc.push_back({2}, 1);
    }
    loc.sort();
    loc.SET_CONJ();

    Xped::Tensor<Scalar, 1, 4, Symmetry> res({{loc}}, {{aux_t, aux_l, aux_b, aux_r}});
    // Xped::Tensor<Scalar, 5, 0, Symmetry> res({{aux_t, aux_l, aux_b, aux_r, loc}}, {{}});
    res.setZero();
    fmt::print("Domain: {}, Codomain: {}\n", res.coupledDomain().print(), res.coupledCodomain().print());
    auto U = res.unitaryDomain();
    fmt::print("U(rank={}): ({}x{})\n", U.NumDimensions, U.dimension(0), U.dimension(0));
    auto V = res.unitaryCodomain();
    fmt::print("V(rank={}): ({}x{}x{}x{}x{})\n", V.NumDimensions, V.dimension(0), V.dimension(1), V.dimension(2), V.dimension(3), V.dimension(4));

    // std::cout << std::setprecision(3) << U << std::endl;
    // {
    //     auto check = PlainInterface::contract<Scalar, 5 + 1, 5 + 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4>(V, V);
    //     Eigen::Tensor<double, 2> Id = check;
    //     Id.setZero();
    //     for(auto i = 0; i < check.dimension(0); ++i) Id(i, i) = 1.;
    //     std::cout << "codomain: " << (check - Id).sum() << std::endl;
    // }
    // {
    //     auto check = PlainInterface::contract<Scalar, 0 + 1, 0 + 1>(U, U);
    //     Eigen::Tensor<double, 2> Id = check;
    //     Id.setZero();
    //     for(auto i = 0; i < check.dimension(0); ++i) Id(i, i) = 1.;
    //     std::cout << "domain: " << (check - Id).sum() << std::endl;
    // }
    auto tmp = PlainInterface::contract<Scalar, 1 + 1, 5, 0, 0>(U, T);
    // auto tmp = PlainInterface::contract<Scalar, 5 + 1, 4 + 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4>(U, T);
    auto Tmat = PlainInterface::contract<Scalar, 4 + 1, 4 + 1, 1, 0, 2, 1, 3, 2, 4, 3>(tmp, V);
    std::cout << Tmat.dimension(0) << "x" << Tmat.dimension(1) << std::endl;
    std::cout << "Tmat: " << Tmat.dimension(0) << "x" << Tmat.dimension(1) << "\n" << Tmat << std::endl;
    // Eigen::Tensor<double, 2> Tmatcheck = T.reshape(std::array<int, 2>{locdim, auxdim * auxdim * auxdim * auxdim});
    // std::cout << "Tmatcheck: " << (Tmatcheck - Tmat).sum() << std::endl;
    Eigen::Map<Eigen::MatrixXd> M(Tmat.data(), Tmat.dimension(0), Tmat.dimension(1));

    for(auto i = 0ul; i < res.sector().size(); ++i) {
        fmt::print("q={}\n", Sym::format<Symmetry>(res.sector(i)));
        fmt::print("Take block at ({},{}) with size={}x{}\n",
                   res.coupledDomain().full_outer_num(res.sector(i)),
                   res.coupledCodomain().full_outer_num(res.sector(i)),
                   Symmetry::degeneracy(res.sector(i)) * PlainInterface::rows(res.block(i)),
                   Symmetry::degeneracy(res.sector(i)) * PlainInterface::cols(res.block(i)));
        Eigen::MatrixXd sub = M.block(res.coupledDomain().full_outer_num(res.sector(i)),
                                      res.coupledCodomain().full_outer_num(res.sector(i)),
                                      Symmetry::degeneracy(res.sector(i)) * PlainInterface::rows(res.block(i)),
                                      Symmetry::degeneracy(res.sector(i)) * PlainInterface::cols(res.block(i)));
        std::cout << std::setprecision(2) << sub << std::endl;
        for(auto row = 0; row < PlainInterface::rows(res.block(i)); ++row) {
            for(auto col = 0; col < PlainInterface::cols(res.block(i)); ++col) {
                res.block(i)(row, col) = sub(row * Symmetry::degeneracy(res.sector(i)), col * Symmetry::degeneracy(res.sector(i)));
            }
        }
    }
    res.print(std::cout, true);
    std::cout << std::endl;
    std::cout << "T-T_constructed: " << (T - res.plainTensor()).sum() << std::endl;
    return res.template permute<-3, 2, 1, 4, 3, 0>();
}

} // namespace Xped::IO
#endif
