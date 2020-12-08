#ifndef U1_H_
#define U1_H_

//include <array>
//include <cstddef>
/// \cond
#include <unordered_set>
/// \endcond

#include <unsupported/Eigen/CXX11/Tensor>

#include "../util/Random.hpp"

namespace Sym{

/** \class U1
  * \ingroup Symmetry
  *
  * Class for handling a U(1) symmetry of a Hamiltonian.
  *
  * \describe_Scalar
  */
template<typename Kind, typename Scalar_=double>
class U1
{
public:
	typedef Scalar_ Scalar;
	
	static constexpr size_t Nq=1;
	
        static constexpr bool HAS_MULTIPLICITIES = false;
	static constexpr bool NON_ABELIAN = false;
	static constexpr bool ABELIAN = true;
	static constexpr bool IS_TRIVIAL = false;
	static constexpr bool IS_MODULAR = false;
	static constexpr int MOD_N = 0;
	
	static constexpr bool IS_CHARGE_SU2() { return false; }
	static constexpr bool IS_SPIN_SU2() { return false; }
	
	static constexpr bool IS_SPIN_U1() { if constexpr (U1<Kind,Scalar>::kind()[0] == KIND::M) {return true;} return false; }
	
	static constexpr bool NO_SPIN_SYM() { if (U1<Kind,Scalar>::kind()[0] != KIND::M and U1<Kind,Scalar>::kind()[0] != KIND::Nup and U1<Kind,Scalar>::kind()[0] != KIND::Ndn) {return true;} return false;}
	static constexpr bool NO_CHARGE_SYM() { if (U1<Kind,Scalar>::kind()[0] != KIND::N and U1<Kind,Scalar>::kind()[0] != KIND::Nup and U1<Kind,Scalar>::kind()[0] != KIND::Ndn) {return true;} return false;}
	
	typedef qarray<Nq> qType;

	U1() {};

	inline static constexpr qType qvacuum() { return {0}; }
	inline static constexpr std::array<qType,2> lowest_qs() { return std::array<qType,2> {{ qarray<1>( std::array<int,1>{{-1}}), qarray<1>(std::array<int,1>{{+1}}) }}; }
	
	inline static std::string name() { return "U1"; }
	inline static constexpr std::array<KIND,Nq> kind() { return {Kind::name}; }

	inline static qType conj( const qType& q ) { return {-q[0]}; }
	inline static int degeneracy( const qType& ) { return 1; }
	
	inline static int spinorFactor() { return +1; }

        inline static qType random_q() { int qval = util::random::threadSafeRandUniform<int,int>(-20,20,false); qType out = {qval}; return out; }
	///@{
	/**
	 * Calculate the irreps of the tensor product of \p ql and \p qr.
	 */
	static std::vector<qType> reduceSilent( const qType& ql, const qType& qr);
	/**
	 * Calculate the irreps of the tensor product of \p ql, \p qm and \p qr.
	 * \note This is independent of the order the quantumnumbers.
	 */
	static std::vector<qType> reduceSilent( const qType& ql, const qType& qm, const qType& qr);
	/**
	 * Calculate the irreps of the tensor product of all entries of \p ql and \p qr.
	 */
	static std::vector<qType> reduceSilent( const std::vector<qType>& ql, const qType& qr);
	/**
	 * Calculate the irreps of the tensor product of all entries of \p ql with all entries of \p qr.
	 */
	static std::vector<qType> reduceSilent( const std::vector<qType>& ql, const std::vector<qType>& qr, bool UNIQUE = false);
	
	static std::vector<std::tuple<qarray<1>,size_t,qarray<1>,size_t,qarray<1> > > tensorProd ( const std::vector<qType>& ql, const std::vector<qType>& qr );
	///@}

        std::size_t multiplicity (const qType& q1, const qType& q2, const qType& q3) {return triangle(q1,q2,q3) ? 1ul : 0ul;}
        
	///@{
	/**
	 * Various coeffecients, all resulting from contractions or traces of the Clebsch-Gordon coefficients.
	 * \note All coefficients are trivial for U(1) and could be represented by a bunch of Kronecker deltas.
	 *       Here we return simply 1, because the algorithm only allows valid combinations of quantumnumbers,
	 *       for which the Kronecker deltas are not necessary.  
	 */
	inline static Scalar coeff_dot(const qType& q1);

        static Scalar coeff_FS(const qType&) {return 1.;}

        static Eigen::Tensor<Scalar_, 2> one_j_tensor(const qType&);
        
	inline static Scalar coeff_3j(const qType& q1, const qType& q2, const qType& q3,
                                      int        q1_z, int        q2_z,        int q3_z);
        
        static Eigen::Tensor<Scalar_, 3> CGC(const qType& q1, const qType& q2, const qType& q3, const std::size_t);

        static Scalar coeff_turn(const qType& ql, const qType& qr, const qType& qf) {return triangle(ql,qr,qf) ? Scalar(1.) : Scalar(0.);}
        
	inline static Scalar coeff_6j(const qType& q1, const qType& q2, const qType& q3,
                                      const qType& q4, const qType& q5, const qType& q6);
	
	inline static Scalar coeff_9j(const qType& q1, const qType& q2, const qType& q3,
                                      const qType& q4, const qType& q5, const qType& q6,
                                      const qType& q7, const qType& q8, const qType& q9);

        static Scalar coeff_swap(const qType& ql, const qType& qr, const qType& qf) {return triangle(ql,qr,qf) ? Scalar(1.) : Scalar(0.);};
        static Scalar coeff_recouple(const qType& q1, const qType& q2, const qType& q3, const qType& Q,
                                     const qType& Q12, const qType& Q23) {return coeff_6j(q1,q2,Q12,
                                                                                          q3,Q,Q23);};
	///@}

	/** 
	 * This function defines a strict order for arrays of quantum-numbers.
	 * \note The implementation is arbritary, as long as it defines a strict order.
	 */
	template<std::size_t M>
	static bool compare ( const std::array<qType,M>& q1, const std::array<qType,M>& q2 );

	/** 
	 * This function checks if the array \p qs contains quantum-numbers which match together, with respect to the flow equations.
	 * \todo2 Write multiple functions, for different sizes of the array and rename them, to have a more clear interface.
	 *        Example: For 3-array: triangular(...) or something similar.
	 */
	template<std::size_t M>
	static bool validate( const std::array<qType,M>& qs );

	static bool triangle( const std::array<qType,3>& qs );
        static bool triangle( const qType& q1, const qType& q2, const qType& q3 ) {return triangle({{q1,q2,q3}});}
	static bool pair( const std::array<qType,2>& qs );

};
	
template<typename Kind, typename Scalar_>
std::vector<typename U1<Kind,Scalar_>::qType> U1<Kind,Scalar_>::
reduceSilent( const qType& ql, const qType& qr )
{
	std::vector<qType> vout;
	vout.push_back({ql[0]+qr[0]});
	return vout;
}

template<typename Kind, typename Scalar_>
std::vector<typename U1<Kind,Scalar_>::qType> U1<Kind,Scalar_>::
reduceSilent( const qType& ql, const qType& qm, const qType& qr )
{
	std::vector<qType> vout;
	vout.push_back({ql[0]+qm[0]+qr[0]});
	return vout;
}

template<typename Kind, typename Scalar_>
std::vector<typename U1<Kind,Scalar_>::qType> U1<Kind,Scalar_>::
reduceSilent( const std::vector<qType>& ql, const qType& qr )
{
	std::vector<typename U1<Kind,Scalar_>::qType> vout;
	for (std::size_t q=0; q<ql.size(); q++)
	{
		vout.push_back({ql[q][0]+qr[0]});
	}
	return vout;
}

template<typename Kind, typename Scalar_>
std::vector<typename U1<Kind,Scalar_>::qType> U1<Kind,Scalar_>::
reduceSilent( const std::vector<qType>& ql, const std::vector<qType>& qr, bool UNIQUE )
{
	if (UNIQUE)
	{
		std::unordered_set<qType> uniqueControl;
		std::vector<qType> vout;
		for (std::size_t q=0; q<ql.size(); q++)
		for (std::size_t p=0; p<qr.size(); p++)
		{
			int i = ql[q][0]+qr[p][0];
			if( auto it = uniqueControl.find({i}) == uniqueControl.end() ) { uniqueControl.insert({i}); vout.push_back({i}); }
		}
		return vout;
	}
	else
	{
		std::vector<qType> vout;

		for (std::size_t q=0; q<ql.size(); q++)
		for (std::size_t p=0; p<qr.size(); p++)
		{
			vout.push_back({ql[q][0]+qr[p][0]});
		}
		return vout;
	}
}

template<typename Kind, typename Scalar_>
Scalar_ U1<Kind,Scalar_>::
coeff_dot(const qType& q1)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, typename Scalar_>
Eigen::Tensor<Scalar_, 2> U1<Kind,Scalar_>::
one_j_tensor(const qType&)
{
        Eigen::Tensor<Scalar, 2> T(1,1); T(0,0) = 1;
        return T;
}

template<typename Kind, typename Scalar_>
Eigen::Tensor<Scalar_, 3> U1<Kind,Scalar_>::
CGC(const qType& q1, const qType& q2, const qType& q3, const std::size_t)
{
        Eigen::Tensor<Scalar, 3> T(1,1,1);
        if (triangle(q1,q2,q3)) {T(0,0,0) = 1.;}
        else {T(0,0,0) = 0.;}
        return T;
}

template<typename Kind, typename Scalar_>
Scalar_ U1<Kind,Scalar_>::
coeff_3j(const qType& q1, const qType& q2, const qType& q3,
          int        q1_z, int        q2_z,        int q3_z)
{
        if (triangle(q1,q2,q3)) {return Scalar(1.);}
        else {return Scalar(0.);}
}

template<typename Kind, typename Scalar_>
Scalar_ U1<Kind,Scalar_>::
coeff_6j(const qType& q1, const qType& q2, const qType& q3,
         const qType& q4, const qType& q5, const qType& q6)
{
        if (triangle(q1,q2,q3) and triangle(q1,q6,q5) and triangle(q2,q4,q6) and triangle(q3,q4,q5)) {return Scalar(1.);}
        return Scalar(0.);
}

template<typename Kind, typename Scalar_>
Scalar_ U1<Kind,Scalar_>::
coeff_9j(const qType& q1, const qType& q2, const qType& q3,
		 const qType& q4, const qType& q5, const qType& q6,
		 const qType& q7, const qType& q8, const qType& q9)
{
	// std::cout << "q1=" << q1 << " q2=" << q2 << " q3=" << q3 << " q4=" << q4 << " q5=" << q5 << " q6=" << q6 << " q7=" << q7 << " q8=" << q8 << " q9=" << q9 << std::endl;
	// if (q1[0] + q4[0] - q7[0] != 0) {return 0.;}
	// if (q2[0] + q5[0] - q8[0] != 0) {return 0.;}
	// if (q3[0] + q6[0] - q9[0] != 0) {return 0.;}
	// if (q4[0] + q5[0] - q6[0] != 0) {return 0.;}
	// if (q7[0] + q8[0] - q9[0] != 0) {return 0.;}
	return Scalar(1.);
}

template<typename Kind, typename Scalar_>
template<std::size_t M>
bool U1<Kind,Scalar_>::
compare ( const std::array<U1<Kind,Scalar>::qType,M>& q1, const std::array<U1<Kind,Scalar>::qType,M>& q2 )
{
        for (std::size_t m=0; m<M; m++)
	{
		if (q1[m][0] > q2[m][0]) { return false; }
		else if (q1[m][0] < q2[m][0]) {return true; }
	}
	return false;
	// for (std::size_t m=0; m<M; m++)
	// {
	// 	if (std::abs(q1[m][0]) > std::abs(q2[m][0])) {return false;}
	// 	else if (std::abs(q1[m][0] < q2[m][0])) {return true;}
        //         else if (q1[m][0] < 0) {return false;}
	// }
	// return false;
}

template<typename Kind, typename Scalar_>
bool U1<Kind,Scalar_>::
triangle ( const std::array<U1<Kind,Scalar>::qType,3>& qs )
{
	//check the triangle rule for U1 quantum numbers
	if (qs[0][0] + qs[1][0] == qs[2][0]) {return true;}
	return false;
}

template<typename Kind, typename Scalar_>
bool U1<Kind,Scalar_>::
pair ( const std::array<U1<Kind,Scalar>::qType,2>& qs )
{
	//check if two quantum numbers fulfill the flow equations: simply qin = qout
	if (qs[0] == qs[1]) {return true;}
	return false;
}

template<typename Kind, typename Scalar_>
template<std::size_t M>
bool U1<Kind,Scalar_>::
validate ( const std::array<U1<Kind,Scalar>::qType,M>& qs )
{
	if constexpr( M == 1 ) { return true; }
	else if constexpr( M == 2 ) { return U1<Kind,Scalar>::pair(qs); }
	else if constexpr( M==3 ) { return U1<Kind,Scalar>::triangle(qs); }
	else { cout << "This should not be printed out!" << endl; return true; }
}

} //end namespace Sym

#ifndef STREAM_OPERATOR_ARR_1_INT
#define STREAM_OPERATOR_ARR_1_INT
std::ostream& operator<< (std::ostream& os, const typename Sym::U1<double>::qType &q)
{
	os << q[0];
	return os;
}
#endif

#endif
