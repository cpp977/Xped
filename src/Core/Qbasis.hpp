#ifndef QBASIS_H_
#define QBASIS_H_

/// \cond
#include <numeric>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "tabulate/tabulate.hpp"
/// \endcond

#include "Core/Basis.hpp"
#include "Core/FusionTree.hpp"
#include "Hash/hash.hpp"
#include "Symmetry/functions.hpp"
#include "Util/Random.hpp"

/** \class Qbasis
 *
 * \ingroup Tensors
 * \describe_Symmetry
 *
 * This class is a container like class for a basis of a Hilbert space in which global symmetries are present.
 * For each irreducible representation (irrep) of the global symmetry (for each quantum number),
 * the states of the Hilbert states that transforms under that irrep are collected together in a plain Basis object.
 *
 * One central function is the combine() method, which combine two instances of Qbasis to the tensor product basis,
 * already proper sorted into irreps.
 *
 */
template <typename Symmetry, std::size_t depth>
class Qbasis
{
    template <typename Symmetry_, std::size_t depth_>
    friend class Qbasis;

    typedef typename Symmetry::qType qType;

public:
    /**Does nothing.*/
    Qbasis(){};

    /**
     * Inserts all quantum numbers in the Container \p qins with constant dimension \p dim into the basis.
     */
    template <typename Container>
    Qbasis(const Container& qins, const size_t& dim)
    {
        for(const auto& qin : qins) { push_back(qin, dim); }
    }

    ///\{
    /**Returns the number of (reduced) basis states.*/
    std::size_t dim() const
    {
        std::size_t out = 0;
        for(const auto& [qVal, num, plain] : data_) { out += plain.dim(); }
        return out;
    }

    /**
     * Returns the full number of basis states.
     * If irreps has internal states, the basis states transforming under this irreps are multiplied by this degeneracy.
     */
    std::size_t fullDim() const
    {
        std::size_t out = 0;
        for(const auto& [qVal, num, plain] : data_) { out += plain.dim() * Symmetry::degeneracy(qVal); }
        return out;
    }

    /**Returns the largest state sector.*/
    std::size_t largestSector() const;

    /**Returns the number of quantum numbers (irreps) contained in this basis.*/
    std::size_t Nq() const { return data_.size(); }
    ///\}

    /**
     * Returns the quantum number of the state with number \p num.
     */
    qType getQ(const size_t& num) const;

    ///\{
    std::vector<qType> qs() const
    {
        std::vector<qType> out;
        for(auto triple : data_) { out.push_back(std::get<0>(triple)); }
        return out;
    }
    std::unordered_set<qType> unordered_qs() const
    {
        std::set<qType> out;
        for(auto triple : data_) { out.insert(std::get<0>(triple)); }
        return out;
    }
    ///\}

    ///\{
    std::vector<std::size_t> dims() const
    {
        std::vector<std::size_t> out;
        for(auto triple : data_) { out.push_back(std::get<2>(triple).dim()); }
        return out;
    }
    ///\}

    ///\{
    /**Checks whether states with quantum number \p q are in the basis. Returns true if the state is present.*/
    bool IS_PRESENT(const qType& q) const { return cfind(q) != data_.end(); }
    /**Checks whether states with quantum number \p q are not in the basis. Returns true if the state is absent.*/
    bool NOT_PRESENT(const qType& q) const { return !IS_PRESENT(q); }
    ///\}

    ///\{
    /**Computes the number within the sector from the total number of the state*/
    size_t inner_num(const size_t& outer_num) const;
    /**Computes the sector size of a state with number \p num_in.*/
    size_t inner_dim(const size_t& num_in) const;
    /**Computes the sector size of a state with quantum number \p q.*/
    size_t inner_dim(const qType& q) const;
    /**Computes the total number with which the sector with quantum number \p q begins.*/
    size_t outer_num(const qType& q) const;
    /**Computes the total number with which the sector with quantum number \p q begins and takes into account the dimension of each irrep.*/
    size_t full_outer_num(const qType& q) const;
    ///\}

    void remove(const qType& q);

    qType maxQ() const
    {
        auto qs = this->qs();
        return *std::max_element(qs.begin(), qs.end());
    }
    qType minQ() const
    {
        auto qs = this->qs();
        return *std::min_element(qs.begin(), qs.end());
    }
    // template<typename Iterator>
    // void remove(Iterator& begin, Iterator& end);

    void truncate(const std::unordered_set<qType>& qs, const std::size_t& M);

    size_t leftOffset(const FusionTree<depth, Symmetry>& tree, const std::array<size_t, depth>& plain = std::array<std::size_t, depth>()) const;
    size_t rightOffset(const FusionTree<depth, Symmetry>& tree, const std::array<size_t, depth>& plain = std::array<std::size_t, depth>()) const;

    /**Insert the quantum number \p q with dimension \p inner_dim into the basis.*/
    void push_back(const qType& q, const size_t& inner_dim);

    void push_back(const qType& q, const size_t& inner_dim, const std::vector<FusionTree<depth, Symmetry>>& tree);

    void setRandom(const std::size_t& fullSize, const std::size_t& max_sectorSize = 5ul);

    /**Completely clear the basis.*/
    void clear()
    {
        data_.clear();
        trees.clear();
        curr_dim = 0;
    }

    /**
     * Returns the tensor product basis, already properly sorted with respect to the resulting irreps.
     * This function also saves the history of the combination process for later use. See leftOffset() and rightOffset().
     */
    Qbasis<Symmetry, depth + 1> combine(const Qbasis<Symmetry, 1>& other, bool CONJ = false) const;

    /**Adds two bases together.*/
    Qbasis<Symmetry, depth> add(const Qbasis<Symmetry, depth>& other) const;

    /**Returns the intersection of this and \p other.*/
    Qbasis<Symmetry, depth> intersection(const Qbasis<Symmetry, depth>& other) const;

    Qbasis<Symmetry, depth> conj() const;

    Qbasis<Symmetry, 1> forgetHistory() const;

    /**Prints the basis.*/
    auto print() const;

    /**Prints the trees.*/
    std::string printTrees() const;

    bool operator==(const Qbasis<Symmetry, depth>& other) const;

    typename std::vector<std::tuple<qType, size_t, Basis>>::iterator begin() { return data_.begin(); }
    typename std::vector<std::tuple<qType, size_t, Basis>>::iterator end() { return data_.end(); }

    typename std::vector<std::tuple<qType, size_t, Basis>>::const_iterator cbegin() const { return data_.cbegin(); }
    typename std::vector<std::tuple<qType, size_t, Basis>>::const_iterator cend() const { return data_.cend(); }

    /**Swaps with another Qbasis.*/
    // void swap (Qbasis<Symmetry,depth> &other) { this->data.swap(other.data()); }

    void sort();

    bool IS_SORTED() const { return IS_SORTED_; }

    std::vector<FusionTree<depth, Symmetry>> tree(const qType& q) const
    {
        auto it = trees.find(q);
        if(it == trees.end()) { assert(false); }
        return it->second;
    }

private:
    typename std::vector<std::tuple<qType, size_t, Basis>>::const_iterator cfind(const qType& q) const;

    typename std::vector<std::tuple<qType, size_t, Basis>>::iterator find(const qType& q);

    // vector with entry: {Quantumnumber (QN), state number of the first plain state for this QN, all plain states for this QN in a Basis object.}
    //[{q1,0,plain_q1}, {q2,dim(plain_q1),plain_q2}, {q3,dim(plain_q1)+dim(plain_q2),plain_q3}, ..., {qi, sum_j^(i-1)dim(plain_qj), plain_qi}]
    std::vector<std::tuple<qType, size_t, Basis>> data_;
    size_t curr_dim = 0;
    std::unordered_map<qType, std::vector<FusionTree<depth, Symmetry>>> trees;

    bool IS_SORTED_ = false;
    bool CONJ = false;
};

template <typename Symmetry, std::size_t depth>
void Qbasis<Symmetry, depth>::push_back(const qType& q, const size_t& inner_dim)
{
    static_assert(depth == 1 or depth == 0);
    auto it = find(q);
    if(it == data_.end()) // insert quantum number if it is not there (sorting is lost then)
    {
        Basis plain_basis(inner_dim);
        auto entry = std::make_tuple(q, curr_dim, plain_basis);
        if constexpr(depth == 1) {
            FusionTree<1, Symmetry> trivial;
            trivial.q_uncoupled[0] = q;
            trivial.q_coupled = q;
            trivial.IS_DUAL[0] = false;
            trivial.dims[0] = inner_dim;
            trivial.dim = inner_dim;
            std::vector<FusionTree<1, Symmetry>> tree(1, trivial);
            trees.insert(std::make_pair(q, tree));
        } else {
            FusionTree<0, Symmetry> trivial;
            trivial.dim = 1;
            trivial.q_coupled = q;
            std::vector<FusionTree<0, Symmetry>> tree(1, trivial);
            trees.insert(std::make_pair(q, tree));
        }
        data_.push_back(entry);
        IS_SORTED_ = false;
    } else // append to quantumnumber if it is there
    {
        std::get<2>(*it) = std::get<2>(*it).add(Basis(inner_dim));
        for(auto loop = it++; loop == data_.end(); loop++) { std::get<1>(*loop) += inner_dim; }
        for(auto& [p, tree] : trees) {
            if(p != q) { continue; }
            tree[0].dim += inner_dim;
            tree[0].dims[0] += inner_dim;
        }
    }
    curr_dim += inner_dim;
}

template <typename Symmetry, std::size_t depth>
void Qbasis<Symmetry, depth>::push_back(const qType& q, const size_t& inner_dim, const std::vector<FusionTree<depth, Symmetry>>& tree)
{
    auto it = find(q);
    if(it == data_.end()) // insert quantum number if it is not there (sorting is lost then)
    {
        Basis plain_basis(inner_dim);
        auto entry = std::make_tuple(q, curr_dim, plain_basis);
        trees.insert(std::make_pair(q, tree));
        data_.push_back(entry);
        IS_SORTED_ = false;
    } else // append to quantumnumber if it is there
    {
        assert(false);
    }
    curr_dim += inner_dim;
}

template <typename Symmetry, std::size_t depth>
void Qbasis<Symmetry, depth>::setRandom(const std::size_t& fullSize, const std::size_t& max_sectorSize)
{
    clear();
    static_assert(depth == 1);
    while(fullDim() < fullSize) {
        qType q = Symmetry::random_q();
        std::size_t inner_dim =
            static_cast<std::size_t>(util::random::threadSafeRandUniform<int, int>(1, std::min(max_sectorSize, fullSize - fullDim())));
        if((fullDim() + Symmetry::degeneracy(q) * inner_dim) <= fullSize) { push_back(q, inner_dim); }
    }
}

template <typename Symmetry, std::size_t depth>
std::size_t Qbasis<Symmetry, depth>::largestSector() const
{
    std::size_t out = 0;
    for(const auto& entry : data_) {
        auto [qVal, num, plain] = entry;
        if(plain.dim() > out) { out = plain.dim(); }
    }
    return out;
}

template <typename Symmetry, std::size_t depth>
typename std::vector<std::tuple<typename Symmetry::qType, size_t, Basis>>::const_iterator Qbasis<Symmetry, depth>::cfind(const qType& q) const
{
    auto it = std::find_if(data_.cbegin(), data_.cend(), [&q](const std::tuple<qType, size_t, Basis>& entry) { return std::get<0>(entry) == q; });
    return it;
}

template <typename Symmetry, std::size_t depth>
typename std::vector<std::tuple<typename Symmetry::qType, size_t, Basis>>::iterator Qbasis<Symmetry, depth>::find(const qType& q)
{
    auto it = std::find_if(data_.begin(), data_.end(), [&q](const std::tuple<qType, size_t, Basis>& entry) { return std::get<0>(entry) == q; });
    return it;
}

template <typename Symmetry, std::size_t depth>
typename Symmetry::qType Qbasis<Symmetry, depth>::getQ(const size_t& num_in) const
{
    assert(num_in < dim() and "The number is larger than the size of this basis.");
    for(const auto& [qVal, num, basis] : data_) {
        if(num_in < num + basis.dim()) { return qVal; }
    }
    assert(false and "Something went wrong in Qbasis::find(size_t num_in)");
}

template <typename Symmetry, std::size_t depth>
size_t Qbasis<Symmetry, depth>::inner_num(const size_t& outer_num) const
{
    assert(outer_num < dim() and "The number is larger than the size of this basis.");
    for(const auto& [qVal, num, plain] : data_) {
        if(outer_num >= num and outer_num < num + plain.size()) { return outer_num - num; }
    }
    assert(false and "Something went wrong in Qbasis::inner_num");
}

template <typename Symmetry, std::size_t depth>
size_t Qbasis<Symmetry, depth>::inner_dim(const qType& q) const
{
    if(auto it = cfind(q); it == data_.end()) {
        return 0;
    } else {
        return std::get<2>(*it).dim();
    }
}

template <typename Symmetry, std::size_t depth>
size_t Qbasis<Symmetry, depth>::outer_num(const qType& q) const
{
    if(auto it = cfind(q); it == data_.end()) {
        assert(false and "The quantum number is not in the basis");
    } else {
        return std::get<1>(*it);
    }
}

template <typename Symmetry, std::size_t depth>
size_t Qbasis<Symmetry, depth>::full_outer_num(const qType& q) const
{
    size_t out = 0ul;
    for(const auto& [p, num, plain] : data_) {
        if(q == p) {
            return out;
        } else {
            out += plain.dim() * Symmetry::degeneracy(p);
        }
    }
    return out;
}

template <typename Symmetry, std::size_t depth>
size_t Qbasis<Symmetry, depth>::inner_dim(const size_t& num_in) const
{
    for(const auto& elem : data_) {
        auto [qVal, num, plain] = elem;
        if(num == num_in) { return plain.dim(); }
    }
    assert(1 != 1 and "This number is not in the basis");
}

template <typename Symmetry, std::size_t depth>
size_t Qbasis<Symmetry, depth>::leftOffset(const FusionTree<depth, Symmetry>& tree, const std::array<size_t, depth>& plain) const
{
    assert(trees.size() == data_.size() and "The history for this basis is not defined properly");
    auto it = trees.find(tree.q_coupled);
    assert(it != trees.end() and "The history for this basis is not defined properly");
    size_t out = 0;

    for(const auto& i : it->second) {
        if(i != tree) { out += i.dim; }
        if(i == tree) {
            plain.size(); // supresses gcc warning
            // maybe some code for plain
            break;
        }
    }
    return out;
}

template <typename Symmetry, std::size_t depth>
size_t Qbasis<Symmetry, depth>::rightOffset(const FusionTree<depth, Symmetry>& tree, const std::array<size_t, depth>& plain) const
{
    assert(trees.size() == data_.size() and "The history for this basis is not defined properly");

    auto it = trees.find(tree.q_coupled);
    assert(it != trees.end() and "The history for this basis is not defined properly");

    size_t out = 0;
    bool SWITCH = false;

    for(const auto& i : it->second) {
        if(i != tree and SWITCH == true) { out += i.dim; }
        if(i == tree) {
            plain.size(); // supresses gcc warning
            // some code for plain size
            SWITCH = true;
        }
    }
    return out;
}

template <typename Symmetry, std::size_t depth>
void Qbasis<Symmetry, depth>::remove(const qType& Q)
{
    auto it = find(Q);
    if(it != data_.end()) {
        auto dimQ = std::get<2>(*it).dim();
        data_.erase(it);
        for(auto loop = it++; loop == data_.end(); loop++) { std::get<1>(*loop) -= dimQ; }
    }
    trees.erase(Q);
    // std::get<2>(*it) = std::get<2>(*it).add(Basis(inner_dim));
    // for (auto& [p,tree] : trees) {
    //         if (p != q) {continue;}
    //         tree[0].dim += inner_dim;
    //         tree[0].dims[0] += inner_dim;
    // }
}

// template<typename Iterator>
// template<typename Symmetry, std::size_t depth>
// void Qbasis<Symmetry,depth>::
// remove (Iterator& begin, Iterator& end)
// {
//         for(auto it=begin, it != end; it++) {
//                 remove(*it);
//         }
// }

template <typename Symmetry, std::size_t depth>
void Qbasis<Symmetry, depth>::truncate(const std::unordered_set<qType>& qs, const std::size_t& M)
{
    static_assert(depth == 1, "Cannot truncate a Qbasis with a depth != 1.");
    if(qs.size() < Nq()) {
        for(const auto& q : this->qs()) {
            if(auto it = qs.find(q); it == qs.end()) { remove(q); }
        }
    }
    if(M < dim()) {
        assert(M >= Nq() and "Cannot truncate basis down and keep at least one state per block");
        size_t D = M / Nq();
        size_t D_remainder = M % Nq();

        for(size_t i = 0; i < this->Nq(); i++) {
            std::size_t Deff = (std::get<0>(data_[i]) == Symmetry::qvacuum()) ? D + D_remainder : D;
            if(std::get<2>(data_[i]).dim() > Deff) {
                std::get<2>(data_[i]) = Basis(Deff);
                auto it = trees.find(std::get<0>(data_[i]));
                assert(it->second.size() == 1);
                it->second[0].dims[0] = Deff;
                it->second[0].dim = Deff;
            }
            auto dims = this->dims();
            std::get<1>(data_[i]) = std::accumulate(dims.begin(), dims.begin() + i, 0ul);
        }
    }
}

template <typename Symmetry, std::size_t depth>
void Qbasis<Symmetry, depth>::sort()
{
    std::vector<std::size_t> index_sort(data_.size());
    std::iota(index_sort.begin(), index_sort.end(), 0);
    std::sort(index_sort.begin(), index_sort.end(), [&](std::size_t n1, std::size_t n2) {
        qarray<Symmetry::Nq> q1 = std::get<0>(data_[n1]);
        qarray<Symmetry::Nq> q2 = std::get<0>(data_[n2]);
        if(CONJ) {
            return Symmetry::compare(Symmetry::conj(q1), Symmetry::conj(q2));
        } else {
            return Symmetry::compare(q1, q2);
        }
    });
    auto new_data_ = data_;
    for(std::size_t i = 0; i < data_.size(); i++) { new_data_[i] = data_[index_sort[i]]; }
    std::get<1>(new_data_[0]) = 0;
    for(std::size_t i = 1; i < data_.size(); i++) {
        std::get<1>(new_data_[i]) = 0;
        for(std::size_t j = 0; j < i; j++) { std::get<1>(new_data_[i]) += std::get<2>(new_data_[j]).dim(); }
    }
    data_ = new_data_;
    IS_SORTED_ = true;
}

template <typename Symmetry, std::size_t depth>
bool Qbasis<Symmetry, depth>::operator==(const Qbasis<Symmetry, depth>& other) const
{
    return (this->data_ == other.data_) and (this->trees == other.trees);
}

template <typename Symmetry, std::size_t depth>
Qbasis<Symmetry, depth> Qbasis<Symmetry, depth>::add(const Qbasis<Symmetry, depth>& other) const
{
    std::unordered_set<qType> uniqueController;
    Qbasis out;
    for(const auto& [q1, num1, plain1] : this->data_) {
        auto it_other = std::find_if(
            other.data_.begin(), other.data_.end(), [q1 = q1](std::tuple<qType, size_t, Basis> entry) { return std::get<0>(entry) == q1; });
        if(it_other != other.data_.end()) {
            out.push_back(q1, plain1.add(std::get<2>(*it_other)).dim());
        } else {
            out.push_back(q1, plain1.dim());
        }
    }

    for(const auto& [q2, num2, plain2] : other.data_) {
        auto it_this =
            std::find_if(data_.begin(), data_.end(), [q2 = q2](std::tuple<qType, size_t, Basis> entry) { return std::get<0>(entry) == q2; });
        if(it_this == data_.end()) { out.push_back(q2, plain2.dim()); }
    }
    out.sort();
    return out;
}

template <typename Symmetry, std::size_t depth>
Qbasis<Symmetry, depth> Qbasis<Symmetry, depth>::intersection(const Qbasis<Symmetry, depth>& other) const
{
    Qbasis out;
    for(const auto& triple : this->data_) {
        if(auto it = other.cfind(std::get<0>(triple)); it != other.cend()) {
            auto dim = std::min(std::get<2>(triple).dim(), std::get<2>(*it).dim());
            out.push_back(std::get<0>(triple), dim);
        }
    }
    return out;
}

template <typename Symmetry, std::size_t depth>
Qbasis<Symmetry, 1> Qbasis<Symmetry, depth>::forgetHistory() const
{
    Qbasis<Symmetry, 1> out;
    for(const auto& [q, num, plain] : data_) { out.push_back(q, plain.dim()); }
    return out;
}

template <typename Symmetry, std::size_t depth>
Qbasis<Symmetry, depth> Qbasis<Symmetry, depth>::conj() const
{
    static_assert(depth == 1);
    Qbasis<Symmetry, depth> out;
    out.CONJ = !this->CONJ;
    for(const auto& [q, num, plain] : data_) {
        auto oldtrees = tree(q);
        assert(oldtrees.size() == 1);
        std::vector<FusionTree<depth, Symmetry>> trees;
        FusionTree<depth, Symmetry> tree;
        tree.q_uncoupled[0] = Symmetry::conj(q);
        tree.q_coupled = Symmetry::conj(q);
        tree.IS_DUAL[0] = !oldtrees[0].IS_DUAL[0];
        tree.dims[0] = oldtrees[0].dims[0];
        tree.dim = oldtrees[0].dim;
        trees.push_back(tree);
        out.push_back(Symmetry::conj(q), plain.dim(), trees);
    }
    return out;
}

template <typename Symmetry, std::size_t depth>
Qbasis<Symmetry, depth + 1> Qbasis<Symmetry, depth>::combine(const Qbasis<Symmetry, 1>& other, bool CONJ) const
{
    Qbasis<Symmetry, depth + 1> out;
    // build the history of the combination. Data is relevant for MultipedeQ contractions which include a fuse of two leg.
    for(const auto& elem1 : this->data_) {
        auto [q1, num1, plain1] = elem1;
        for(const auto& elem2 : other.data_) {
            auto [q2, num2, plain2] = elem2;
            if(CONJ) { q2 = Symmetry::conj(q2); }
            // auto plain = plain1.combine(plain2);
            auto qVec = Symmetry::reduceSilent(q1, q2);
            for(const auto& q : qVec) {
                for(auto& thisTree : this->tree(q1))
                    for(const auto& otherTree : other.tree(q2)) {
                        auto totalTree = thisTree.enlarge(otherTree);
                        totalTree.q_coupled = q;
                        auto it = out.trees.find(q);
                        if(it == out.trees.end()) {
                            std::vector<FusionTree<depth + 1, Symmetry>> tree(1, totalTree);
                            out.trees.insert(std::make_pair(q, tree));
                        } else {
                            auto it_tree = std::find(it->second.begin(), it->second.end(), totalTree);
                            if(it_tree == it->second.end()) {
                                it->second.push_back(totalTree);
                            } else {
                                continue;
                            }
                        }
                    }
            }
        }
    }

    // sort the trees on the basis of Symmetry::compare()
    for(auto it = out.trees.begin(); it != out.trees.end(); it++) {
        std::vector<std::size_t> index_sort((it->second).size());
        std::iota(index_sort.begin(), index_sort.end(), 0ul);
        std::sort(index_sort.begin(), index_sort.end(), [&](std::size_t n1, std::size_t n2) {
            // return Symmetry::compare((it->second)[n1].source,(it->second)[n2].source);
            return (it->second)[n1] < (it->second)[n2];
        });
        std::vector<FusionTree<depth + 1, Symmetry>> tree = it->second;
        for(std::size_t i = 0; i < tree.size(); i++) { (it->second)[i] = tree[index_sort[i]]; }
    }

    // build up the new basis
    for(auto it = out.trees.begin(); it != out.trees.end(); it++) {
        size_t inner_dim = 0;
        for(const auto& i : it->second) { inner_dim += i.dim; }
        out.push_back(it->first, inner_dim, it->second);
    }
    out.sort();
    return out;
}

template <typename Symmetry, std::size_t depth>
auto Qbasis<Symmetry, depth>::print() const
{
    tabulate::Table t;
    // TextTable t('-', '|', '+');
    t.add_row({"Q", "Dim(Q)", "num"});
    // t.add("Q");
    // t.add("Dim(Q)");
    // t.add("num");
    // t.endOfRow();
    for(const auto& entry : data_) {
        auto [q_Phys, curr_num, plain] = entry;
        std::stringstream ss, tt, uu;
        ss << Sym::format<Symmetry>(q_Phys);

        tt << plain.dim();

        uu << curr_num;
        t.add_row({ss.str(), tt.str(), uu.str()});
        // t.add(ss.str());
        // t.add(tt.str());
        // t.add(uu.str());
        // t.endOfRow();
    }
    t.format()
        .font_align(tabulate::FontAlign::center)
        .font_style({tabulate::FontStyle::bold})
        .border_top(" ")
        .border_bottom(" ")
        .border_left(" ")
        .border_right(" ")
        .corner(" ");
    t[0].format().padding_top(1).padding_bottom(1).font_style({tabulate::FontStyle::underline}).font_background_color(tabulate::Color::red);
    t[0][1].format().font_background_color(tabulate::Color::blue);
    return t;
}

template <typename Symmetry, std::size_t depth>
std::string Qbasis<Symmetry, depth>::printTrees() const
{
    std::stringstream out;
    for(auto it = trees.begin(); it != trees.end(); it++) {
        out << it->second.size() << " Fusion trees for Q=" << it->first << std::endl;
        for(const auto& i : it->second) { out << i.draw() << std::endl; }
        out << std::endl << "******************************************************************************************" << std::endl;
        ;
    }
    return out.str();
}

template <typename Symmetry, std::size_t depth>
std::ostream& operator<<(std::ostream& os, const Qbasis<Symmetry, depth>& basis)
{
    os << basis.print();
    return os;
}

#endif