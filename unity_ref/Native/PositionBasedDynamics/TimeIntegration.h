#ifndef TIMEINTEGRATION_H
#define TIMEINTEGRATION_H

#include "Common/Common.h"

// ------------------------------------------------------------------------------------
namespace PBD
{
	class TimeIntegration
	{
	public:	
		/** Perform an integration step for a particle using the semi-implicit Euler 
		 * (symplectic Euler) method:
		 * \f{align*}{
		 * \mathbf{v}(t+h) &= \mathbf{v}(t) + \mathbf{a}(t) h\\
		 * \mathbf{x}(t+h) &= \mathbf{x}(t) + \mathbf{v}(t+h) h
		 * \f}
		 *
		 * @param  h time step size
		 * @param  mass mass of the particle
		 * @param  position position of the particle
		 * @param  velocity velocity of the particle
		 * @param  acceleration acceleration of the particle
		 */		
		static void semiImplicitEuler(
			const Real h,
			const Real mass,
			Vector3r &position,
			Vector3r &velocity,
			const Vector3r &acceleration);

		// -------------- semi-implicit Euler (symplectic Euler) for rotational part of a rigid body -----------------
		static void semiImplicitEulerRotation(
			const Real h,
			const Real mass,
			const Matrix3r &invInertiaW,
			Quaternionr &rotation,
			Vector3r &angularVelocity,
			const Vector3r &torque);


		// -------------- velocity update (first order) -----------------------------------------------------
		/** Perform a velocity update (first order) for the linear velocity:
		 * \f{equation*}{
		 * \mathbf{v}(t+h) = \frac{1}{h} (\mathbf{p}(t+h) - \mathbf{p}(t)
		 * \f}
		 *
		 * @param  h time step size
		 * @param  mass mass of the particle
		 * @param  position new position \f$\mathbf{p}(t+h)\f$ of the particle
		 * @param  oldPosition position \f$\mathbf{p}(t)\f$ of the particle before the time step
		 * @param  velocity resulting velocity of the particle
		 */		
		static void velocityUpdateFirstOrder(
			const Real h,
			const Real mass,
			const Vector3r &position,				// position after constraint projection	at time t+h
			const Vector3r &oldPosition,				// position before constraint projection at time t
			Vector3r &velocity);

		// -------------- angular velocity update (first order)  ------------------------------------------------
		static void angularVelocityUpdateFirstOrder(
			const Real h,
			const Real mass,
			const Quaternionr &rotation,				// rotation after constraint projection	at time t+h
			const Quaternionr &oldRotation,			// rotation before constraint projection at time t
			Vector3r &angularVelocity);


		// -------------- velocity update (second order) -----------------------------------------------------
		static void velocityUpdateSecondOrder(
			const Real h,
			const Real mass,
			const Vector3r &position,				// position after constraint projection	at time t+h
			const Vector3r &oldPosition,				// position before constraint projection at time t
			const Vector3r &positionOfLastStep,		// position of last simulation step at time t-h
			Vector3r &velocity);

		// -------------- angular velocity update (second order)  ------------------------------------------------
		static void angularVelocityUpdateSecondOrder(
			const Real h,
			const Real mass,
			const Quaternionr &rotation,				// rotation after constraint projection	at time t+h
			const Quaternionr &oldRotation,			// rotation before constraint projection at time t
			const Quaternionr &rotationOfLastStep,	// rotation of last simulation step at time t-h
			Vector3r &angularVelocity);

	};
}

#endif