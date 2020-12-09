#ifndef U0_H_
#define U0_H_

/// \cond
#include <cstddef>
/// \endcond

#include <unsupported/Eigen/CXX11/Tensor>

#include "qarray.hpp"

/**Dummies for models without symmetries.*/
const std::array<qarray<0>,1> qloc1dummy {qarray<0>{}};
const std::array<qarray<0>,2> qloc2dummy {qarray<0>{}, qarray<0>{}};
const std::array<qarray<0>,3> qloc3dummy {qarray<0>{}, qarray<0>{}, qarray<0>{}};
const std::array<qarray<0>,4> qloc4dummy {qarray<0>{}, qarray<0>{}, qarray<0>{}, qarray<0>{}};
const std::array<qarray<0>,8> qloc8dummy {qarray<0>{}, qarray<0>{}, qarray<0>{}, qarray<0>{}, qarray<0>{}, qarray<0>{}, qarray<0>{}, qarray<0>{}};
const std::array<string,0>    labeldummy{};

namespace Sym{
	
/** \class U0
 * \ingroup Symmetry
 *
 * Dummy class for no symmetry.
 *
 */
	class U0
	{
	public:
                typedef double Scalar;
		typedef qarray<0> qType;
		
		U0() {};
		
		static std::string name() { return "noSymmetry"; }
		
		static constexpr std::size_t Nq=0;

                static constexpr bool HAS_MULTIPLICITIES = false;
		static constexpr bool NON_ABELIAN = false;
		static constexpr bool ABELIAN = true;
		static constexpr bool IS_TRIVIAL = true;
		static constexpr bool IS_MODULAR = false;
		static constexpr int MOD_N = 0;

		static constexpr bool IS_CHARGE_SU2() { return false; }
		static constexpr bool IS_SPIN_SU2() { return false; }

		static constexpr bool IS_SPIN_U1() { return false; }

		static constexpr bool NO_SPIN_SYM() { return true; }
		static constexpr bool NO_CHARGE_SYM() { return true; }
		
		inline static constexpr std::array<KIND,Nq> kind() { return {}; }

		inline static constexpr qType qvacuum() {return {};}
		inline static constexpr std::array<qType,1> lowest_qs() { return std::array<qType,1> {{ qarray<0>(std::array<int,0>{{}}) }}; }

		inline static qType conj( const qType& ) { return {}; }
		inline static int degeneracy( const qType& ) { return 1; }

		inline static int spinorFactor() { return +1; }

                inline static qType random_q() { return {}; }
                
		inline static std::vector<qType> reduceSilent( const qType&, const qType&) { return {{}}; }
		inline static std::vector<qType> reduceSilent( const qType&, const qType&, const qType&) { return {{}}; }
		inline static std::vector<qType> reduceSilent( const std::vector<qType>&, const qType&) { return {{}}; }
		inline static std::vector<qType> reduceSilent( const std::vector<qType>&, const std::vector<qType>&, bool = false) { return {{}}; }
				
		template<std::size_t M>
		static bool compare ( const std::array<qType,M>&, const std::array<qType,M>& )
		{
			return false;
		}
		
		inline static double coeff_dot(const qType&) { return 1.; }

                static Scalar coeff_FS(const qType&) {return 1.;}

                static Eigen::Tensor<Scalar, 2> one_j_tensor(const qType&) {Eigen::Tensor<Scalar,2> T(1,1); T(0,0) = 1.; return T;}
                
                inline static Scalar coeff_3j(const qType&, const qType&, const qType&,
                                              int         , int         ,        int) {return 1.;}
                
                static Eigen::Tensor<Scalar, 3> CGC(const qType&, const qType&, const qType&, const std::size_t) {Eigen::Tensor<Scalar,3> T(1,1,1); T(0,0,0) = 1.; return T;}
                static Scalar coeff_turn(const qType&, const qType&, const qType&) {return Scalar(1.);}
                
		inline static double coeff_6j(const qType&, const qType&, const qType&,
                                              const qType&, const qType&, const qType&) { return 1.; }
		
		inline static double coeff_9j(const qType&, const qType&, const qType&,
                                              const qType&, const qType&, const qType&,
                                              const qType&, const qType&, const qType&) { return 1.; }

                static Scalar coeff_swap(const qType& ql, const qType& qr, const qType& qf) {return triangle(ql,qr,qf) ? Scalar(1.) : Scalar(0.);};
                static Scalar coeff_recouple(const qType& q1, const qType& q2, const qType& q3, const qType& Q,
                                             const qType& Q12, const qType& Q23) {return coeff_6j(q1,q2,Q12,
                                                                                                  q3,Q,Q23);};
                
		inline static bool triangle( const std::array<qType,3>& ) { return true; }
                inline static bool triangle( const qType& q1, const qType& q2, const qType& q3) {return triangle({{q1,q2,q3}});}
                        
		inline static bool pair( const std::array<qType,2>&) { return true; }

		template<std::size_t M> inline static bool validate( const std::array<qType,M>& ) { return true; }
	};

} //end namespace Sym
#endif

