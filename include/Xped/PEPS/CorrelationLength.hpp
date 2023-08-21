#ifndef XPED_CORRELATION_LENGTH_HPP_
#define XPED_CORRELATION_LENGTH_HPP_

#include "ALGS/ArnoldiSolver.h"

#include "Xped/PEPS/CTM.hpp"
#include "Xped/PEPS/iPEPS.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry>
void normalize(Xped::Tensor<Scalar, 3, 0, Symmetry>& V)
{
    V = V / V.norm();
}

template <typename Scalar, typename Symmetry>
Scalar norm(const Xped::Tensor<Scalar, 3, 0, Symmetry>& V)
{
    return V.norm();
}

template <typename Scalar, typename Symmetry>
struct TransferMatrix2
{
    std::vector<Tensor<Scalar, 1, 3, Symmetry>> T_top;
    std::vector<Tensor<Scalar, 3, 1, Symmetry>> T_bot;
    std::size_t size() const { return T_top.size(); }
    auto asMatrix()
    {
        Tensor<Scalar, 2, 2, Symmetry> tmp0 = T_top[0].template contract<std::array{-1, -3, 1, 2}, std::array{1, 2, -2, -4}, 2>(T_bot[0]);
        Tensor<Scalar, 2, 2, Symmetry> tmp1 = T_top[1].template contract<std::array{-1, -3, 1, 2}, std::array{1, 2, -2, -4}, 2>(T_bot[1]);
        return tmp0 * tmp1;
        // for(auto pos = size() - 2; pos < size(); --pos) {
        //     auto tmp1 = T_top[pos].template contract<std::array{-1, 1, -2, -3}, std::array{1, -4, -5, -6}, 3>(tmp);
        //     tmp = tmp1.template contract<std::array{-1, 1, 2, 3, -3, -4}, std::array{1, 2, -2, 3}, 2>(T_bot[pos].twist(3));
        // }
        // return tmp;
    }
};

template <typename Scalar, typename Symmetry>
using TransferVector2 = Tensor<Scalar, 3, 0, Symmetry>;

template <typename Scalar, typename Symmetry>
Scalar dot(const TransferVector2<Scalar, Symmetry>& V1, const TransferVector2<Scalar, Symmetry>& V2)
{
    return (V1.adjoint() * V2).trace();
}

template <typename Scalar, typename Symmetry>
void addScale(Scalar scale, const TransferVector2<Scalar, Symmetry>& Vin, TransferVector2<Scalar, Symmetry>& Vout)
{
    Vout += scale * Vin;
}

template <typename Scalar, typename Symmetry>
std::size_t dim(const TransferVector2<Scalar, Symmetry>& v)
{
    return v.plainSize();
}

template <typename Scalar, typename Symmetry>
std::size_t dim(const TransferMatrix2<Scalar, Symmetry>&)
{
    return 0ul;
}

template <typename OtherScalar, typename Scalar, typename Symmetry>
void HxV(const TransferMatrix2<Scalar, Symmetry>& H,
         const TransferVector2<OtherScalar, Symmetry>& v_in,
         TransferVector2<OtherScalar, Symmetry>& v_out)
{
    TransferVector2<OtherScalar, Symmetry> tmp = v_in;
    for(auto pos = H.size() - 1; pos < H.size(); --pos) {
        auto tmp1 = H.T_top[pos].template contract<std::array{-1, 1, -2, -3}, std::array{1, -4, -5}, 3>(tmp);
        tmp = tmp1.template contract<std::array{-1, 1, 2, -2, 3}, std::array{1, 2, -3, 3}, 3>(H.T_bot[pos].twist(3));
    }
    v_out = tmp;
}

template <typename Scalar, typename Symmetry>
auto correlation_length(const CTM<Scalar, Symmetry>& env,
                        Opts::Orientation orientation = Opts::Orientation::H,
                        int x = 0,
                        int y = 0,
                        const std::vector<typename Symmetry::qType>& Qlist = {})
{
    const std::size_t Neig = 4;
    auto size = (orientation == Opts::Orientation::H) ? env.cell().Lx : env.cell().Ly;
    std::vector<Tensor<Scalar, 1, 3, Symmetry>> T_top(size);
    if(orientation == Opts::Orientation::H) {
        for(int ix = 0; ix < env.cell().Lx; ++ix) { T_top[ix] = env.T1s(x + ix, y); }
    } else {
        for(int iy = 0; iy < env.cell().Ly; ++iy) { T_top[iy] = env.T4s(x, y + iy); }
    }
    std::vector<Tensor<Scalar, 3, 1, Symmetry>> T_bot(size);
    if(orientation == Opts::Orientation::H) {
        for(int ix = 0; ix < env.cell().Lx; ++ix) { T_bot[ix] = env.T3s(x + ix, y + 1); }
    } else {
        for(int iy = 0; iy < env.cell().Ly; ++iy) { T_bot[iy] = env.T2s(x + 1, y + iy); }
    }
    TransferMatrix2<Scalar, Symmetry> T{T_top, T_bot};
    // auto mat = T.asMatrix();
    // mat.print(std::cout, true);
    // std::cout << std::endl;
    auto first_b = T.T_top[T.size() - 1].uncoupledCodomain()[0];
    auto second_b = T.T_bot[T.size() - 1].uncoupledCodomain()[0];
    auto combined = first_b.combine(second_b).forgetHistory();
    std::vector<std::pair<typename Symmetry::qType, typename ScalarTraits<Scalar>::Comp>> eigs;
    for(const auto Q : combined.qs()) {
        if(Qlist.size() > 0) {
            if(auto it = std::find(Qlist.begin(), Qlist.end(), Q); it == Qlist.end()) { continue; }
        }
        Qbasis<Symmetry> sector;
        sector.push_back(Q, 1);

        TransferVector2<typename ScalarTraits<Scalar>::Comp, Symmetry> V({{first_b, sector, second_b}}, {{}});
        static thread_local std::mt19937 engine(std::random_device{}());
        V.setRandom(engine);
        // TransferVector2<typename ScalarTraits<Scalar>::Comp, Symmetry> V;
        // if(orientation == Opts::Orientation::H) {
        //     V = 1i * env.C2s(x, y).template contract<std::array{-1, 1}, std::array{1, -2}, 2>(env.C3s(x, y + 1));
        // } else {
        //     V = 1i * env.C4s(x, y).template contract<std::array{-1, 1}, std::array{-2, 1}, 2>(env.C3s(x, y + 1));
        // }
        ArnoldiSolver<TransferMatrix2<Scalar, Symmetry>, TransferVector2<typename ScalarTraits<Scalar>::Comp, Symmetry>> Lucy(Neig);
        Lucy.calc_dominant(T, V);
        for(auto i = 0ul; i < Neig; ++i) { eigs.push_back(make_pair(Q, Lucy.get_lambda(i))); }
        // Log::on_exit("{}", Lucy.info());
    }
    std::sort(eigs.begin(), eigs.end(), [](auto eig1, auto eig2) { return std::abs(eig1.second) > std::abs(eig2.second); });
    Log::on_exit("All eigs:");
    // for(auto [Q, eig] : eigs) { Log::on_exit("Q={}:{}, ", Sym::format<Symmetry>(Q), eig / std::abs(eigs[0].second)); }
    for(auto [Q, eig] : eigs) { Log::on_exit("Q={}:{}, ", Sym::format<Symmetry>(Q), std::abs(eig / std::abs(eigs[0].second))); }
    Log::on_exit("corr length={}", size * (-1. / std::log(std::abs(eigs[1].second) / std::abs(eigs[0].second))));
    return std::make_pair(size * (-1. / std::log(std::abs(eigs[1].second) / std::abs(eigs[0].second))), eigs);
}

template <typename Scalar, typename Symmetry>
void normalize(Xped::Tensor<Scalar, 4, 0, Symmetry>& V)
{
    V = V / V.norm();
}

template <typename Scalar, typename Symmetry>
Scalar norm(const Xped::Tensor<Scalar, 4, 0, Symmetry>& V)
{
    return V.norm();
}

template <typename Scalar, typename Symmetry>
struct TransferMatrix3
{
    std::vector<Tensor<Scalar, 1, 3, Symmetry>> T_top;
    std::vector<Tensor<Scalar, 3, 1, Symmetry>> T_bot;
    std::vector<Tensor<Scalar, 2, 3, Symmetry>> A;
    std::vector<Tensor<Scalar, 3, 2, Symmetry>> Ad;
    std::size_t size() const { return T_top.size(); }
};

template <typename Scalar, typename Symmetry>
using TransferVector3 = Tensor<Scalar, 4, 0, Symmetry>;

template <typename Scalar, typename Symmetry>
Scalar dot(const TransferVector3<Scalar, Symmetry>& V1, const TransferVector3<Scalar, Symmetry>& V2)
{
    return (V1.adjoint() * V2).trace();
}

template <typename Scalar, typename Symmetry>
void addScale(Scalar scale, const TransferVector3<Scalar, Symmetry>& Vin, TransferVector3<Scalar, Symmetry>& Vout)
{
    Vout += scale * Vin;
}

template <typename Scalar, typename Symmetry>
std::size_t dim(const TransferVector3<Scalar, Symmetry>& v)
{
    return v.plainSize();
}

template <typename Scalar, typename Symmetry>
std::size_t dim(const TransferMatrix3<Scalar, Symmetry>& v)
{
    return 0ul;
}

template <typename Scalar, typename Symmetry>
void HxV(const TransferMatrix3<Scalar, Symmetry>& H, const TransferVector3<Scalar, Symmetry>& v_in, TransferVector3<Scalar, Symmetry>& v_out)
{
    TransferVector3<Scalar, Symmetry> tmp = v_in;
    for(auto pos = H.size() - 1; pos < H.size(); --pos) {
        auto tmp1 = H.T_top[pos].template contract<std::array{-1, 1, -2, -3}, std::array{1, -4, -5, -6}, 3>(tmp);
        auto tmp2 = tmp1.template contract<std::array{-1, 1, -2, 2, -3, -4}, std::array{-5, 1, 2, -6, -7}, 4>(H.A[pos].twist(3));
        auto tmp3 = tmp2.template contract<std::array{-1, 1, 2, -2, -3, -4, 3}, std::array{-5, 1, 3, 2, -6}, 4>(H.Ad[pos]);
        tmp = tmp3.template contract<std::array{-1, 1, -2, 2, -3, 3}, std::array{2, 3, -4, 1}, 4>(H.T_bot[pos].twist(1).twist(3));
    }
    v_out = tmp;
}

template <typename Scalar, typename Symmetry>
Scalar correlation_length(const iPEPS<Scalar, Symmetry>& Psi, const CTM<Scalar, Symmetry>& env)
{
    std::vector<Tensor<Scalar, 1, 3, Symmetry>> T_top(env.cell().Lx);
    for(auto x = 0ul; x < env.cell().Lx; ++x) { T_top[x] = env.T1s(x, -1); }
    std::vector<Tensor<Scalar, 3, 1, Symmetry>> T_bot(env.cell().Lx);
    for(auto x = 0ul; x < env.cell().Lx; ++x) { T_bot[x] = env.T3s(x, 1); }
    std::vector<Tensor<Scalar, 2, 3, Symmetry>> A(env.cell().Lx);
    for(auto x = 0ul; x < env.cell().Lx; ++x) { A[x] = Psi.As(x, 0); }
    std::vector<Tensor<Scalar, 3, 2, Symmetry>> Adag(env.cell().Lx);
    for(auto x = 0ul; x < env.cell().Lx; ++x) { Adag[x] = Psi.Adags(x, 0); }
    TransferMatrix3<Scalar, Symmetry> T{T_top, T_bot, A, Adag};
    TransferVector3<Scalar, Symmetry> x({{env.T1s(0, -1).uncoupledCodomain()[0],
                                          Psi.ketBasis(0, 0, Opts::Leg::Right),
                                          Psi.braBasis(0, 0, Opts::Leg::Right),
                                          env.T3s(0, 1).uncoupledCodomain()[0]}},
                                        {{}});
    static thread_local std::mt19937 engine(std::random_device{}());
    x.setRandom(engine);
    ArnoldiSolver<TransferMatrix3<Scalar, Symmetry>, TransferVector3<Scalar, Symmetry>> Lucy(3);
    Lucy.calc_dominant(T, x);
    Log::on_exit("{}", Lucy.info());
    Log::on_exit("corr length={}", env.cell().Lx * (-1. / std::log(std::abs(Lucy.get_lambda(1) / Lucy.get_lambda(0)))));
    return env.cell().Lx * (-1. / std::log(std::abs(Lucy.get_lambda(1) / Lucy.get_lambda(0))));
}

} // namespace Xped

#endif
