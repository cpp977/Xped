#ifdef FUSER_H_
#define FUSER_H_

template<std::size_t Rank, typename Symmetry>
class Fuser
{
public:
        using std::size_t;
        typedef typename Symmetry::qType qType;
private:
        typedef Qbasis<Symmetry> BasisT;
public:
        Fuser() {};
        Fuser(const std::array<BasisT, Rank>& uncoupled_in) : uncoupled(uncoupled_in) {fuse();}

        void fuse();
private:
        BasisT elementaryFuse(const BasisT& left, const BasisT& left);
        
        std::unordered_map<qType, std::vector<FusionTree<Rank,Symmetry> > > trees;
        BasisT coupled;
        std::array<BasisT, Rank> uncoupled;
};

template<std::size_t Rank, typename Symmetry>
Qbasis<Symmetry> Fuser<Rank,Symmetry>::
elementaryFuse(const BasisT& left, const BasisT& right)
{
        return left.combine(right);
}

template<std::size_t Rank, typename Symmetry>
void Fuser<Rank,Symmetry>::
fuse()
{        
        BasisT intermediate = uncoupled[0];
        for (size_t i=1; i<Rank; i++)
                {
                        intermediate = elementaryFuse(intermediate,uncoupled[i]);
                }
        coupled = intermediate;
        trees = coupled.trees;
        return;
}
#endif
