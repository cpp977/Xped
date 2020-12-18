#ifndef SYM_BASE_H_
#define SYM_BASE_H_

namespace Sym {
template <typename Derived> struct SymTraits;

template<typename Derived>
struct SymBase
{
        typedef typename SymTraits<Derived>::qType qType;
        typedef typename SymTraits<Derived>::Scalar Scalar;
        
        static std::vector<qType> reduceSilent( const qType& ql, const qType& qr) {
                return Derived::basis_combine(ql,qr);
        }
        
        static std::vector<qType> reduceSilent( const std::vector<qType>& ql, const qType& qr) {
        	std::vector<qType> vout;
                for (const auto& q: ql) {
                        for (const auto& qmerge:Derived::basis_combine(q,qr)) {
                                vout.push_back(qmerge);
                        }
                }
                return vout;        
        }

        static std::vector<qType> reduceSilent( const std::vector<qType>& ql, const std::vector<qType>& qr) {
        	std::vector<qType> vout;
                for (const auto& q: ql) {
                        for (const auto& p: qr) {
                                for (const auto& qmerge:Derived::basis_combine(q,p)) {
                                        vout.push_back(qmerge);
                                }
                        }
                }
                return vout;        
        }
        
        static std::vector<qType> reduceSilent(const qType& ql, const qType& qm, const qType& qr) {
                auto qtmp = Derived::basis_combine(ql,qm);
                return reduceSilent(qtmp,qr);
        }

        static std::unordered_set<qType> reduceSilent(const std::unordered_set<qType>& ql, const std::vector<qType>& qr) {
                std::unordered_set<qType> out;
                for (const auto q : ql) {
                        for (const auto& p : qr) {
                                for (const auto qp : Derived::basis_combine(q,p)) {
                                        out.insert(qp);
                                }
                        }
                }
                return out;
        }
        
        static Scalar coeff_FS(const qType& q) {return std::signbit(Derived::coeff_recouple(q, Derived::conj(q), q, q, Derived::qvacuum(), Derived::qvacuum())) ? Scalar(-1) : Scalar(1.);}
        
        static Scalar coeff_turn(const qType& ql, const qType& qr, const qType& qf) {
                return std::sqrt(Derived::degeneracy(qr)) * Derived::coeff_recouple(ql, qr, Derived::conj(qr),
                                                                                    ql, qf, Derived::qvacuum());
        }

        template<std::size_t M>
	static bool compare (const std::array<qType,M>& q1, const std::array<qType,M>& q2) {
                for (std::size_t m=0; m<M; m++) {
                        if (q1[m] > q2[m]) { return false; }
                        else if (q1[m] < q2[m]) {return true; }
                }
                return false;
        }

        static bool compare (const qType& q1, const qType& q2) {
                return q1 < q2;
        }

        static bool triangle (const qType& q1, const qType& q2, const qType& q3) {
                auto q12 = reduceSilent(q1, q2);
                for (const auto& q:q12) {
                        if (q == q3) {return true;}
                }
                return false;
        }        
};
        
} // end namespace Sym
#endif
