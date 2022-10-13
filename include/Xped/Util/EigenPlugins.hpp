#ifndef XPED_EIGEN_DENSE_BASE_ADDONS_H_
#define XPED_EIGEN_DENSE_BASE_ADDONS_H_

template <typename Ar>
void serialize(Ar& ar) const
{
    derived().eval();
    const std::size_t rows = derived().rows(), cols = derived().cols();
    std::vector<double> data(derived().data(), derived().data() + derived().size());
    ar& YAS_OBJECT_NVP("Matrix", ("data", data), ("rows", rows), ("cols", cols));
}

template <typename Ar>
void serialize(Ar& ar)
{
    // derived().eval();
    std::size_t rows, cols;
    std::vector<double> data;
    ar& YAS_OBJECT_NVP("Matrix", ("data", data), ("rows", rows), ("cols", cols));
    derived().resize(rows, cols);
    Eigen::Map<Derived> m(data.data(), rows, cols);
    derived() = m;
}

#endif
