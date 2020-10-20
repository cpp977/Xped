#ifndef FUSIONTREE_H_
#define FUSIONTREE_H_

namespace util
{
        constexpr std::size_t inter_dim(std::size_t Rank) {return (Rank == 1 or Rank == 0) ? 0 : Rank-2; }
        constexpr std::size_t mult_dim(std::size_t Rank) {return (Rank == 0) ? 0 : Rank-1; }
}

template<std::size_t Rank, typename Symmetry>
struct FusionTree
{
        typedef typename Symmetry::qType qType;

        std::array<qType, Rank> q_uncoupled;
        qType q_coupled;
        std::size_t dim;
        std::array<qType, util::inter_dim(Rank)> q_intermediates;
        std::array<size_t, util::mult_dim(Rank)> multiplicities; //only for non-Abelian symmetries with outermultiplicity.
        std::array<bool, Rank> IS_DUAL;

        bool operator< (const FusionTree<Rank,Symmetry>& other) const
        {
                if (Symmetry::compare(q_uncoupled,other.q_uncoupled)) return true;
                if (Symmetry::compare(q_intermediates,other.q_intermediates)) return true;
                if (multiplicities < other.multiplicities) return true;
                if (q_coupled < other.q_coupled) return true;
                return false;
        }

        bool operator== (const FusionTree<Rank,Symmetry>& other) const
        {
                return 
                q_uncoupled == other.q_uncoupled and
                        q_intermediates == other.q_intermediates and
                        multiplicities == other.multiplicities and
                        q_coupled == other.q_coupled;
        }

        bool operator!= (const FusionTree<Rank,Symmetry>& other) const {return !this->operator==(other);}
                        
        std::string print() const
        {
                std::stringstream ss;
                ss << "Fusiontree with dim=" << dim << std::endl;
                ss << "uncoupled sectors: "; for (const auto& q : q_uncoupled) {ss << q << " ";} ss << std::endl;
                ss << "intermediate sectors: "; for (const auto& q : q_intermediates) {ss << q << " ";} ss << std::endl;
                ss << "coupled sector: " << q_coupled << std::endl;
                return ss.str();
        }
        
        FusionTree<Rank+1, Symmetry> enlarge(const FusionTree<1, Symmetry>& other) const
        {
                FusionTree<Rank+1, Symmetry> out;
                out.dim = this->dim * other.dim;
                std::copy(this->q_uncoupled.begin(), this->q_uncoupled.end(), out.q_uncoupled.begin());
                out.q_uncoupled[Rank] = other.q_uncoupled[0];

                std::copy(this->q_intermediates.begin(), this->q_intermediates.end(), out.q_intermediates.begin());
                if (out.q_intermediates.size() > 0) {out.q_intermediates[out.q_intermediates.size()-1] = this->q_coupled;}


                std::copy(this->multiplicities.begin(), this->multiplicities.end(), out.multiplicities.begin());
                return out;
        }
};

template<std::size_t depth, typename Symmetry>
std::ostream& operator<<(std::ostream& os, const FusionTree<depth,Symmetry> &tree)
{
	os << tree.print();
	return os;
}
#endif
