#ifndef XPED_STORAGE_TYPE_VEC_OF_MAT_HPP_
#define XPED_STORAGE_TYPE_VEC_OF_MAT_HPP_

#include "Xped/Core/Qbasis.hpp"
#include "Xped/Core/TensorHelper.hpp"
#include "Xped/Core/VecOfMatIterator.hpp"

namespace Xped {

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
struct StorageType
{
    using MatrixType = PlainInterface::MType<Scalar>;
    using MapMatrixType = PlainInterface::MapMType<Scalar>;
    using cMapMatrixType = PlainInterface::cMapMType<Scalar>;
    using qType = typename Symmetry::qType;

    using MatrixReturnType = MatrixType&;
    using ConstMatrixReturnType = const MatrixType&;

private:
    using DictType = std::unordered_map<qType,
                                        std::size_t,
                                        std::hash<qType>,
                                        std::equal_to<qType>,
                                        typename AllocationPolicy::template Allocator<std::pair<const qType, std::size_t>>>;

public:
    StorageType() = default;

    StorageType(const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, Rank> basis_domain,
                const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, CoRank> basis_codomain,
                const mpi::XpedWorld& world = mpi::getUniverse())
        : m_uncoupled_domain(basis_domain)
        , m_uncoupled_codomain(basis_codomain)
        , m_world(world)
    {
        m_domain = internal::build_FusionTree(m_uncoupled_domain);
        m_codomain = internal::build_FusionTree(m_uncoupled_codomain);
        // uninitialized_resize();
    }

    StorageType(const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, Rank> basis_domain,
                const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, CoRank> basis_codomain,
                const Scalar* data,
                std::size_t size,
                const mpi::XpedWorld& world = mpi::getUniverse())
        : m_uncoupled_domain(basis_domain)
        , m_uncoupled_codomain(basis_codomain)
        , m_world(world)
    {
        m_domain = internal::build_FusionTree(m_uncoupled_domain);
        m_codomain = internal::build_FusionTree(m_uncoupled_codomain);
        initialized_resize(data, size);
    }

    StorageType(const StorageType<Scalar, Rank, CoRank, Symmetry, AllocationPolicy>& other) = default;
    // {
    //     sector_ = other.derived().sector();
    //     // block_.resize(sector_.size());
    //     storage_ = other.storage();
    //     // for(std::size_t i = 0; i < sector_.size(); i++) { block_[i] = other.derived().block(i); }
    //     dict_ = other.derived().dict();
    //     world_ = other.derived().world();
    //     uncoupled_domain = other.derived().uncoupledDomain();
    //     uncoupled_codomain = other.derived().uncoupledCodomain();
    //     domain = other.derived().coupledDomain();
    //     codomain = other.derived().coupledCodomain();
    // }

    // template <template <typename> typename OtherAllocator>
    // StorageType(const StorageType<Scalar, Rank, CoRank, Symmetry, OtherAllocator>& other);

    static constexpr bool IS_CONTIGUOUS() { return false; }

    std::size_t plainSize() const
    {
        return std::accumulate(m_data.cbegin(), m_data.cend(), 0, [](std::size_t cur, const MatrixType& mat) { return cur + mat.size(); });
    }

    void resize()
    {
        m_data.reserve(std::max(m_domain.Nq(), m_codomain.Nq()));

        for(const auto& [q, dim, plain] : m_domain) {
            if(m_codomain.IS_PRESENT(q)) {
                m_sector.push_back(q);
                m_dict.insert(std::make_pair(q, m_sector.size() - 1));
                MatrixType Mtmp = PlainInterface::construct<Scalar>(m_domain.inner_dim(q), m_codomain.inner_dim(q), m_world);
                m_data.push_back(Mtmp);
                // m_data.emplace_back(m_domain.inner_dim(q), m_codomain.inner_dim(q));
            }
        }

        m_data.shrink_to_fit();
    }

    const MatrixType& block(std::size_t i) const
    {
        assert(i < m_data.size());
        return m_data[i];
    }

    MatrixType& block(std::size_t i)
    {
        assert(i < m_data.size());
        return m_data[i];
    }

    const MatrixType& block(qType q) const
    {
        auto it = m_dict.find(q);
        assert(it != m_dict.end());
        return m_data[it->second];
    }

    MatrixType& block(qType q)
    {
        auto it = m_dict.find(q);
        assert(it != m_dict.end());
        return m_data[it->second];
    }

    const DictType& dict() const { return m_dict; }
    DictType& dict() { return m_dict; }

    const std::vector<qType, typename AllocationPolicy::template Allocator<qType>>& sector() const { return m_sector; }
    qType sector(std::size_t i) const { return m_sector[i]; }

    const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, Rank>& uncoupledDomain() const { return m_uncoupled_domain; }
    const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, CoRank>& uncoupledCodomain() const { return m_uncoupled_codomain; }

    const Qbasis<Symmetry, Rank, AllocationPolicy>& coupledDomain() const { return m_domain; }
    const Qbasis<Symmetry, CoRank, AllocationPolicy>& coupledCodomain() const { return m_codomain; }

    void push_back(const qType& q, const MatrixType& M)
    {
        m_sector.push_back(q);
        m_dict.insert(std::make_pair(q, m_sector.size() - 1));
        m_data.push_back(M);
    }

    void reserve(std::size_t size) { m_data.reserve(size); }

    void clear()
    {
        m_data.clear();
        m_sector.clear();
    }

    VecOfMatIterator<Scalar> begin()
    {
        VecOfMatIterator<Scalar> out(&m_data);
        return out;
    }
    VecOfMatIterator<Scalar> end()
    {
        VecOfMatIterator<Scalar> out(&m_data, m_data.size());
        return out;
    }

    const mpi::XpedWorld& world() const { return m_world; }

private:
    std::vector<MatrixType, typename AllocationPolicy::template Allocator<MatrixType>> m_data;

    DictType m_dict; // sector --> number
    std::vector<qType, typename AllocationPolicy::template Allocator<qType>> m_sector;

    std::array<Qbasis<Symmetry, 1, AllocationPolicy>, Rank> m_uncoupled_domain;
    std::array<Qbasis<Symmetry, 1, AllocationPolicy>, CoRank> m_uncoupled_codomain;
    Qbasis<Symmetry, Rank, AllocationPolicy> m_domain;
    Qbasis<Symmetry, CoRank, AllocationPolicy> m_codomain;

    mpi::XpedWorld m_world;

    void initialized_resize(const Scalar* data, std::size_t size)
    {
        m_data.reserve(std::min(m_domain.Nq(), m_codomain.Nq()));

        std::size_t current_dim = 0;
        for(const auto& [q, dim, plain] : m_domain) {
            if(m_codomain.IS_PRESENT(q)) {
                m_sector.push_back(q);
                m_dict.insert(std::make_pair(q, m_sector.size() - 1));
                cMapMatrixType tmp(data + current_dim, m_domain.inner_dim(q), m_codomain.inner_dim(q));
                m_data.emplace_back(tmp);
                current_dim += m_domain.inner_dim(q) * m_codomain.inner_dim(q);
            }
        }
        assert(current_dim == size and "You specified an incompatible data array for this tensor.");

        m_data.shrink_to_fit();
    }
};

} // namespace Xped

#endif
