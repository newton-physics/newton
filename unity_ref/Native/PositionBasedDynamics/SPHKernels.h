#ifndef SPHKERNELS_H
#define SPHKERNELS_H

#define _USE_MATH_DEFINES
#include <math.h>
#include "Common/Common.h"
#include <algorithm>

#define NO_DISTANCE_TEST

namespace PBD
{
	class CubicKernel
	{
	protected:
		static Real m_radius;
		static Real m_k;
		static Real m_l;
		static Real m_W_zero;
	public:
		static Real getRadius() { return m_radius; }
		static void setRadius(Real val)
		{
			m_radius = val;
			static const Real pi = static_cast<Real>(M_PI);

			const Real h3 = m_radius*m_radius*m_radius;
			m_k = static_cast<Real>(8.0) / (pi*h3);
			m_l = static_cast<Real>(48.0) / (pi*h3);
			m_W_zero = W(Vector3r(0.0, 0.0, 0.0));
		}

	public:
		//static unsigned int counter;
		static Real W(const Vector3r &r)
		{
			//counter++;
			Real res = 0.0;
			const Real rl = r.norm();
			const Real q = rl/m_radius;
#ifndef NO_DISTANCE_TEST
			if (q <= 1.0)
#endif
			{
				if (q <= 0.5)
				{
					const Real q2 = q*q;
					const Real q3 = q2*q;
					res = m_k * (static_cast<Real>(6.0)*q3- static_cast<Real>(6.0)*q2+ static_cast<Real>(1.0));
				}
				else
				{
					res = m_k * (static_cast<Real>(2.0)*pow(static_cast<Real>(1.0)-q,3));
				}
			}
			return res;
		}

		static Vector3r gradW(const Vector3r &r)
		{
			Vector3r res;
			const Real rl = r.norm();
			const Real q = rl / m_radius;
#ifndef NO_DISTANCE_TEST
			if (q <= 1.0)
#endif
			{
				if (rl > 1.0e-6)
				{
					const Vector3r gradq = r * ((Real) 1.0 / (rl*m_radius));
					if (q <= 0.5)
					{
						res = m_l*q*((Real) 3.0*q - (Real) 2.0)*gradq;
					}
					else
					{
						const Real factor = static_cast<Real>(1.0) - q;
						res = m_l*(-factor*factor)*gradq;
					}
				}
			}
#ifndef NO_DISTANCE_TEST
 			else
 				res.zero();
#endif

			return res;
		}

		static Real W_zero()
		{
			return m_W_zero;
		}
	};
}

#endif
