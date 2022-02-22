#ifndef XPED_STORAGE_TYPE_CONTIGOUS_HPP_
#define XPED_STORAGE_TYPE_CONTIGOUS_HPP_

namespace Xped {

template <typename Scalar, template <typename> typename Allocator>
struct StorageType
{
    StorageType()
        : block_size(0)
        , m_rows(0)
        , m_cols(0)
        , block_dim(0)
    {}

    StorageType(std::size_t block_size, std::size_t rows, std::size_t cols)
        : block_size(block_size)
        , m_rows(rows)
        , m_cols(cols)
    {
        block_dim = m_rows * m_cols;
        m_data.resize(block_size * block_dim);
    }

    StorageType(std::size_t block_size, std::size_t rows, std::size_t cols, const Scalar* data)
        : block_size(block_size)
        , m_rows(rows)
        , m_cols(cols)
    {
        block_dim = m_rows * m_cols;
        m_data = std::vector<Scalar, Allocator<Scalar>>(data, data + block_size * block_dim);
    }

    // template <template <typename> typename OtherAllocator>
    // StorageType(const StorageType<Scalar, OtherAllocator>& other)
    //     : block_size(other.blockSize())
    //     , m_rows(other.rows())
    //     , m_cols(other.cols())
    //     , block_dim(other.blockDim())
    // {
    //     m_data = stan::math::to_arena(other.data());
    // };

    Eigen::Map<const Eigen::Matrix<Scalar, -1, -1>> block(std::size_t i) const
    {
        assert(i < block_size);
        return Eigen::Map<const Eigen::Matrix<Scalar, -1, -1>>(m_data.data() + block_dim * i, m_rows, m_cols);
    }

    Eigen::Map<Eigen::Matrix<Scalar, -1, -1>> block(std::size_t i)
    {
        assert(i < block_size);
        return Eigen::Map<Eigen::Matrix<Scalar, -1, -1>>(m_data.data() + block_dim * i, m_rows, m_cols);
    }

    void resize(std::size_t new_size, std::size_t new_rows, std::size_t new_cols)
    {
        block_size = new_size;
        m_rows = new_rows;
        m_cols = new_cols;
        block_dim = m_rows * m_cols;
        m_data.resize(new_size * block_dim);
    }

    std::size_t blockSize() const { return block_size; }
    std::size_t blockDim() const { return block_dim; }
    std::size_t rows() const { return m_rows; }
    std::size_t cols() const { return m_cols; }

    const auto& data() const { return m_data; }
    auto& data() { return m_data; }

private:
    std::size_t block_size;
    std::size_t block_dim;
    std::size_t m_rows, m_cols;

    std::vector<Scalar, Allocator<Scalar>> m_data;
};

} // namespace Xped
#endif
