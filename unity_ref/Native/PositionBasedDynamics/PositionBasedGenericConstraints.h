#ifndef POSITIONBASEDGENERICCONSTRAINTS_H
#define POSITIONBASEDGENERICCONSTRAINTS_H

#include "Common/Common.h"

// ------------------------------------------------------------------------------------
namespace PBD
{
	/** Generic constraints provide the possibility to experiment with new constraint
	* functions very easily. Only a callback for the constraint function and optionally
	* for the gradient of the constraint must be provided.\n\n
	*
	* Examples and more details can be found here: \ref pageGenericConstraints \n
	*
	*/
	class PositionBasedGenericConstraints
	{
	public:
		/** Determine the position corrections for a constraint function
		 * with a known gradient function.\n\n
		 * More information can be found here: \ref secGenericConstraintsCGFct
		 *
		 * @param  invMass inverse mass of constrained particles
		 * @param  x positions of constrained particles
		 * @param  userData	user data which is required in the callback functions
		 * @param  constraintFct constraint callback
		 * @param  gradientFct gradient callback
		 * @param  corr_x position corrections of constrained particles
		 */		
		template<unsigned int numberOfParticles, unsigned int dim>
		static bool solve_GenericConstraint(
			const Real invMass[numberOfParticles],							// inverse mass is zero if particle is static
			const Vector3r x[numberOfParticles],						// positions of particles
			void *userData,

			void(*constraintFct)(
				const unsigned int numParticles,
				const Real invMass[],							// inverse mass is zero if particle is static
				const Vector3r x[],						// positions of particles
				void *userData,
				Eigen::Matrix<Real, dim, 1> &constraintValue),

			void(*gradientFct)(
				const unsigned int i,							// compute dC / dxi
				const unsigned int numParticles,
				const Real invMass[],							// inverse mass is zero if particle is static
				const Vector3r x[],						// positions of particles
				void *userData,
				Eigen::Matrix<Real, dim, 3> &jacobian),

			Vector3r corr_x[numberOfParticles]);		


		/** Determine the position corrections for a constraint function.
		* This method approximates the gradient using finite differences.
		* Note that it is faster to define the gradient function explicitly.\n\n
		* More information can be found here: \ref secGenericConstraintsCFct
		*
		* @param  invMass inverse mass of constrained particles
		* @param  x positions of constrained particles
		* @param  userData	user data which is required in the callback functions
		* @param  constraintFct constraint callback
		* @param  corr_x position corrections of constrained particles
		*/
		template<unsigned int numberOfParticles, unsigned int dim>
		static bool solve_GenericConstraint(
			const Real invMass[numberOfParticles],							// inverse mass is zero if particle is static
			const Vector3r x[numberOfParticles],						// positions of particles
			void *userData,

			void(*constraintFct)(
				const unsigned int numParticles,
				const Real invMass[],							// inverse mass is zero if particle is static
				const Vector3r x[],						// positions of particles
				void *userData,
				Eigen::Matrix<Real, dim, 1> &constraintValue),

			Vector3r corr_x[numberOfParticles]);

		/** Determines the Jacobian of the i-th particle with finite differences.
		*
		* @param  i index of particle for which the gradient should be computed
		* @param  invMass inverse mass of constrained particles
		* @param  x positions of constrained particles
		* @param  userData	user data which is required in the callback functions
		* @param  constraintFct constraint callback
		* @param  jacobian Jacobian of the i-th particle
		*/
		template<unsigned int numberOfParticles, unsigned int dim>
		static void approximateGradient(
			const unsigned int i,							// compute dC / dxi
			const Real invMass[],							// inverse mass is zero if particle is static
			const Vector3r x[],						// positions of particles
			void *userData,

			void(*constraintFct)(
				const unsigned int numParticles,
				const Real invMass[],							// inverse mass is zero if particle is static
				const Vector3r x[],						// positions of particles
				void *userData,
				Eigen::Matrix<Real, dim, 1> &constraintValue),

			Eigen::Matrix<Real, dim, 3> &jacobian);


		/** Determine the position corrections for a constraint function
		 * with a known gradient function.\n\n
		 * More information can be found here: \ref secGenericConstraintsCGFct
		 *
		 * @param  invMass inverse mass of constrained rigid bodies
		 * @param  x positions of constrained rigid bodies
		 * @param  inertiaInverseW inverse inertia tensor in world coordinates of the rigid bodies
		 * @param  q rotation of the rigid bodies
		 * @param  userData	user data which is required in the callback functions
		 * @param  constraintFct constraint callback
		 * @param  gradientFct gradient callback
		 * @param  corr_x position corrections of the constrained rigid bodies
		 * @param  corr_q rotation corrections of the constrained rigid bodies
		 */		
		template<unsigned int numberOfRigidBodies, unsigned int dim>
		static bool solve_GenericConstraint(
			const Real invMass[numberOfRigidBodies],					// inverse mass is zero if body is static
			const Vector3r x[numberOfRigidBodies],						// positions of bodies
			const Matrix3r inertiaInverseW[numberOfRigidBodies],		// inverse inertia tensor (world space) of bodies
			const Quaternionr q[numberOfRigidBodies],					// rotation of bodies
			void *userData,

			void(*constraintFct)(
				const unsigned int numRigidBodies,
				const Real invMass[numberOfRigidBodies],					// inverse mass is zero if body is static
				const Vector3r x[numberOfRigidBodies],						// positions of bodies
				const Matrix3r inertiaInverseW[numberOfRigidBodies],		// inverse inertia tensor (world space) of bodies
				const Quaternionr q[numberOfRigidBodies],					// rotation of bodies
				void *userData,
				Eigen::Matrix<Real, dim, 1> &constraintValue),

			void(*gradientFct)(
				const unsigned int i,										// compute dC / dxi
				const unsigned int numRigidBodies,
				const Real invMass[numberOfRigidBodies],					// inverse mass is zero if body is static
				const Vector3r x[numberOfRigidBodies],						// positions of bodies
				const Matrix3r inertiaInverseW[numberOfRigidBodies],		// inverse inertia tensor (world space) of bodies
				const Quaternionr q[numberOfRigidBodies],					// rotation of bodies
				void *userData,
				Eigen::Matrix<Real, dim, 6> &jacobian),

			Vector3r corr_x[numberOfRigidBodies],
			Quaternionr corr_q[numberOfRigidBodies]);


		/** Determine the position corrections for a constraint function.
		* This method approximates the gradient using finite differences.
		* Note that it is faster to define the gradient function explicitly.\n\n
		* More information can be found here: \ref secGenericConstraintsCFct
		*
		* @param  invMass inverse mass of constrained rigid bodies
		* @param  x positions of constrained rigid bodies
		* @param  inertiaInverseW inverse inertia tensor in world coordinates of the rigid bodies
		* @param  q rotation of the rigid bodies
		* @param  userData	user data which is required in the callback functions
		* @param  constraintFct constraint callback
		* @param  corr_x position corrections of the constrained rigid bodies
		* @param  corr_q rotation corrections of the constrained rigid bodies
		*/
		template<unsigned int numberOfRigidBodies, unsigned int dim>
		static bool solve_GenericConstraint(
			const Real invMass[numberOfRigidBodies],					// inverse mass is zero if body is static
			const Vector3r x[numberOfRigidBodies],						// positions of bodies
			const Matrix3r inertiaInverseW[numberOfRigidBodies],		// inverse inertia tensor (world space) of bodies
			const Quaternionr q[numberOfRigidBodies],
			void *userData,

			void(*constraintFct)(
				const unsigned int numParticles,
				const Real invMass[numberOfRigidBodies],					// inverse mass is zero if body is static
				const Vector3r x[numberOfRigidBodies],						// positions of bodies
				const Matrix3r inertiaInverseW[numberOfRigidBodies],		// inverse inertia tensor (world space) of bodies
				const Quaternionr q[numberOfRigidBodies],
				void *userData,
				Eigen::Matrix<Real, dim, 1> &constraintValue),

			Vector3r corr_x[numberOfRigidBodies],
			Quaternionr corr_q[numberOfRigidBodies]);

		/** Determines the Jacobian of the i-th rigid body with finite differences.
		*
		* @param  i index of rigid body for which the gradient should be computed
		* @param  invMass inverse mass of constrained rigid bodies
		* @param  x positions of constrained rigid bodies
		* @param  inertiaInverseW inverse inertia tensor in world coordinates of the rigid bodies
		* @param  q rotation of the rigid bodies
		* @param  userData	user data which is required in the callback functions
		* @param  constraintFct constraint callback
		* @param  jacobian Jacobian of the i-th rigid body
		*/
		template<unsigned int numberOfRigidBodies, unsigned int dim>
		static void approximateGradient(
			const unsigned int i,										// compute dC / dxi
			const Real invMass[numberOfRigidBodies],					// inverse mass is zero if body is static
			const Vector3r x[numberOfRigidBodies],						// positions of bodies
			const Matrix3r inertiaInverseW[numberOfRigidBodies],		// inverse inertia tensor (world space) of bodies
			const Quaternionr q[numberOfRigidBodies],
			void *userData,

			void(*constraintFct)(
				const unsigned int numParticles,
				const Real invMass[numberOfRigidBodies],					// inverse mass is zero if body is static
				const Vector3r x[numberOfRigidBodies],						// positions of bodies
				const Matrix3r inertiaInverseW[numberOfRigidBodies],		// inverse inertia tensor (world space) of bodies
				const Quaternionr q[numberOfRigidBodies],
				void *userData,
				Eigen::Matrix<Real, dim, 1> &constraintValue),

			Eigen::Matrix<Real, dim, 6> &jacobian);

		/** Compute matrix that is required to transform quaternion in 
		 * a 3D representation. */
		static void computeMatrixG(const Quaternionr &q, Eigen::Matrix<Real, 4, 3> &G);

	};



	template<unsigned int numberOfParticles, unsigned int dim>
	bool PositionBasedGenericConstraints::solve_GenericConstraint(
		const Real invMass[numberOfParticles],							// inverse mass is zero if particle is static
		const Vector3r x[numberOfParticles],						// positions of particles
		void *userData,

		void(*constraintFct)(
			const unsigned int numParticles,
			const Real invMass[],							// inverse mass is zero if particle is static
			const Vector3r x[],						// positions of particles
			void *userData,
			Eigen::Matrix<Real, dim, 1> &constraintValue),

		void(*gradientFct)(
			const unsigned int i,							// compute dC / dxi
			const unsigned int numParticles,
			const Real invMass[],							// inverse mass is zero if particle is static
			const Vector3r x[],						// positions of particles
			void *userData,
			Eigen::Matrix<Real, dim, 3> &jacobian),

		Vector3r corr_x[numberOfParticles])
	{
		// evaluate constraint function
		Eigen::Matrix<Real, dim, 1> C;
		constraintFct(numberOfParticles, invMass, x, userData, C);

		Eigen::Matrix<Real, dim, dim> K;
		K.setZero();

		Eigen::Matrix<Real, dim, 3> gradients[numberOfParticles];
		for (unsigned int i = 0u; i < numberOfParticles; i++)
		{
			if (invMass[i] != 0.0)
			{
				// compute gradient
				gradientFct(i, numberOfParticles, invMass, x, userData, gradients[i]);
				K += invMass[i] * gradients[i] * gradients[i].transpose();
			}
		}

		// compute Kinv
		if (K.determinant() < 1.0e-6)
			return false;

		Eigen::Matrix<Real, dim, dim> Kinv = K.inverse();

		Eigen::Matrix<Real, dim, 1> lambda = -Kinv * C;

		for (unsigned int i = 0u; i < numberOfParticles; i++)
		{
			if (invMass[i] != 0.0)
			{
				// compute position correction
				corr_x[i] = invMass[i] * gradients[i].transpose() * lambda;
			}
			else
				corr_x[i].setZero();
		}

		return true;
	}

	template<unsigned int numberOfParticles, unsigned int dim>
	bool PositionBasedGenericConstraints::solve_GenericConstraint(
		const Real invMass[numberOfParticles],							// inverse mass is zero if particle is static
		const Vector3r x[numberOfParticles],						// positions of particles
		void *userData,

		void(*constraintFct)(
			const unsigned int numParticles,
			const Real invMass[],							// inverse mass is zero if particle is static
			const Vector3r x[],						// positions of particles
			void *userData,
			Eigen::Matrix<Real, dim, 1> &constraintValue),

		Vector3r corr_x[numberOfParticles])
	{
		// evaluate constraint function
		Eigen::Matrix<Real, dim, 1> Cd;
		Vector3r xTemp[numberOfParticles];
		for (unsigned int j = 0; j < numberOfParticles; j++)
			xTemp[j] = x[j].template cast<Real>();

		constraintFct(numberOfParticles, invMass, xTemp, userData, Cd);
		Eigen::Matrix<Real, dim, 1> C = Cd.template cast<Real>();

		Eigen::Matrix<Real, dim, dim> K;
		K.setZero();

		Eigen::Matrix<Real, dim, 3> gradients[numberOfParticles];
		for (unsigned int i = 0u; i < numberOfParticles; i++)
		{
			if (invMass[i] != 0.0)
			{
				// compute gradient
				approximateGradient<numberOfParticles, dim>(i, invMass, x, userData, constraintFct, gradients[i]);
				K += invMass[i] * gradients[i] * gradients[i].transpose();
			}
		}

		// compute Kinv
		if (K.determinant() < 1.0e-6)
			return false;

		Eigen::Matrix<Real, dim, dim> Kinv = K.inverse();

		Eigen::Matrix<Real, dim, 1> lambda = -Kinv * C;

		for (unsigned int i = 0u; i < numberOfParticles; i++)
		{
			if (invMass[i] != 0.0)
			{
				// compute position correction
				corr_x[i] = invMass[i] * gradients[i].transpose() * lambda;
			}
			else
				corr_x[i].setZero();
		}

		return true;
	}

	template<unsigned int numberOfParticles, unsigned int dim>
	void PositionBasedGenericConstraints::approximateGradient(
		const unsigned int i,							// compute dC / dxi
		const Real invMass[],							// inverse mass is zero if particle is static
		const Vector3r x[],						// positions of particles
		void *userData,

		void(*constraintFct)(
			const unsigned int numParticles,
			const Real invMass[],							// inverse mass is zero if particle is static
			const Vector3r x[],						// positions of particles
			void *userData,
			Eigen::Matrix<Real, dim, 1> &constraintValue),

		Eigen::Matrix<Real, dim, 3> &jacobian)
	{
		Vector3r xTemp[numberOfParticles];
		for (unsigned int j = 0; j < numberOfParticles; j++)
			xTemp[j] = x[j].template cast<Real>();

		Real eps = static_cast<Real>(1.e-6);
		jacobian.setZero();
		Vector3r x_i = xTemp[i];
		for (unsigned int j = 0; j < 3; j++)
		{
			xTemp[i][j] += eps;

			Eigen::Matrix<Real, dim, 1> e_p, e_m;
			constraintFct(numberOfParticles, invMass, xTemp, userData, e_p);
			xTemp[i][j] = x_i[j] - eps;
			constraintFct(numberOfParticles, invMass, xTemp, userData, e_m);
			xTemp[i][j] = x_i[j];

			Eigen::Matrix<Real, dim, 1> res = (e_p - e_m) * (1.0 / (2.0*eps));

			jacobian.template block<dim, 1>(0, j) = res.template cast<Real>();
		}
	}


	template<unsigned int numberOfRigidBodies, unsigned int dim>
	bool PositionBasedGenericConstraints::solve_GenericConstraint(
		const Real invMass[numberOfRigidBodies],					// inverse mass is zero if body is static
		const Vector3r x[numberOfRigidBodies],						// positions of bodies
		const Matrix3r inertiaInverseW[numberOfRigidBodies],		// inverse inertia tensor (world space) of bodies
		const Quaternionr q[numberOfRigidBodies],
		void *userData,

		void(*constraintFct)(
			const unsigned int numRigidBodies,
			const Real invMass[numberOfRigidBodies],					// inverse mass is zero if body is static
			const Vector3r x[numberOfRigidBodies],						// positions of bodies
			const Matrix3r inertiaInverseW[numberOfRigidBodies],		// inverse inertia tensor (world space) of bodies
			const Quaternionr q[numberOfRigidBodies],
			void *userData,
			Eigen::Matrix<Real, dim, 1> &constraintValue),

		void(*gradientFct)(
			const unsigned int i,										// compute dC / dxi
			const unsigned int numRigidBodies,
			const Real invMass[numberOfRigidBodies],					// inverse mass is zero if body is static
			const Vector3r x[numberOfRigidBodies],						// positions of bodies
			const Matrix3r inertiaInverseW[numberOfRigidBodies],		// inverse inertia tensor (world space) of bodies
			const Quaternionr q[numberOfRigidBodies],
			void *userData,
			Eigen::Matrix<Real, dim, 6> &jacobian),

		Vector3r corr_x[numberOfRigidBodies],
		Quaternionr corr_q[numberOfRigidBodies])
	{
		// evaluate constraint function
		Eigen::Matrix<Real, dim, 1> C;
		constraintFct(numberOfRigidBodies, invMass, x, inertiaInverseW, q, userData, C);

		Eigen::Matrix<Real, dim, dim> K;
		K.setZero();

		Eigen::Matrix<Real, dim, 6> gradients[numberOfRigidBodies];
		for (unsigned int i = 0u; i < numberOfRigidBodies; i++)
		{
			if (invMass[i] != 0.0)
			{
				// compute gradient
				gradientFct(i, numberOfRigidBodies, invMass, x, inertiaInverseW, q, userData, gradients[i]);

				// inverse mass matrix
				Eigen::Matrix<Real, 6, 6> Minv;
				Minv.setZero();
				Minv(0, 0) = invMass[i];
				Minv(1, 1) = invMass[i];
				Minv(2, 2) = invMass[i];
				Minv.block<3, 3>(3, 3) = inertiaInverseW[i];

				K += gradients[i] * Minv * gradients[i].transpose();
			}
		}

		Eigen::Matrix<Real, dim, dim> Kinv = K.inverse();

		Eigen::Matrix<Real, dim, 1> lambda = -Kinv * C;

		for (unsigned int i = 0u; i < numberOfRigidBodies; i++)
		{
			if (invMass[i] != 0.0)
			{
				// compute position correction
				const Vector6r pt = gradients[i].transpose() * lambda;
				corr_x[i] = invMass[i] * pt.block<3, 1>(0, 0);
				const Vector3r ot = (inertiaInverseW[i] * pt.block<3, 1>(3, 0));
				const Quaternionr otQ(0.0, ot[0], ot[1], ot[2]);
				corr_q[i].coeffs() = 0.5 *(otQ*q[i]).coeffs();
			}
			else
			{
				corr_x[i].setZero();
				corr_q[i].coeffs().setZero();
			}
		}

		return true;
	}

	template<unsigned int numberOfRigidBodies, unsigned int dim>
	bool PositionBasedGenericConstraints::solve_GenericConstraint(
		const Real invMass[numberOfRigidBodies],					// inverse mass is zero if body is static
		const Vector3r x[numberOfRigidBodies],						// positions of bodies
		const Matrix3r inertiaInverseW[numberOfRigidBodies],		// inverse inertia tensor (world space) of bodies
		const Quaternionr q[numberOfRigidBodies],
		void *userData,

		void(*constraintFct)(
			const unsigned int numParticles,
			const Real invMass[numberOfRigidBodies],					// inverse mass is zero if body is static
			const Vector3r x[numberOfRigidBodies],						// positions of bodies
			const Matrix3r inertiaInverseW[numberOfRigidBodies],		// inverse inertia tensor (world space) of bodies
			const Quaternionr q[numberOfRigidBodies],
			void *userData,
			Eigen::Matrix<Real, dim, 1> &constraintValue),

		Vector3r corr_x[numberOfRigidBodies],
		Quaternionr corr_q[numberOfRigidBodies])
	{
		// evaluate constraint function
		Eigen::Matrix<Real, dim, 1> C;
		constraintFct(numberOfRigidBodies, invMass, x, inertiaInverseW, q, userData, C);

		Eigen::Matrix<Real, dim, dim> K;
		K.setZero();

		Eigen::Matrix<Real, dim, 6> gradients[numberOfRigidBodies];
		for (unsigned int i = 0u; i < numberOfRigidBodies; i++)
		{
			if (invMass[i] != 0.0)
			{
				// compute gradient
				approximateGradient<numberOfRigidBodies, dim>(i, invMass, x, inertiaInverseW, q, userData, constraintFct, gradients[i]);
				
				// inverse mass matrix
				Eigen::Matrix<Real, 6, 6> Minv;
				Minv.setZero();
				Minv(0, 0) = invMass[i];
				Minv(1, 1) = invMass[i];
				Minv(2, 2) = invMass[i];
				Minv.block<3, 3>(3, 3) = inertiaInverseW[i];

				K += gradients[i] * Minv * gradients[i].transpose();
			}
		}

		Eigen::Matrix<Real, dim, dim> Kinv = K.inverse();

		Eigen::Matrix<Real, dim, 1> lambda = -Kinv * C;

		for (unsigned int i = 0u; i < numberOfRigidBodies; i++)
		{
			if (invMass[i] != 0.0)
			{
				// compute position correction
				const Vector6r pt = gradients[i].transpose() * lambda;
				corr_x[i] = invMass[i] * pt.block<3, 1>(0, 0);
				const Vector3r ot = (inertiaInverseW[i] * pt.block<3, 1>(3, 0));
				const Quaternionr otQ(0.0, ot[0], ot[1], ot[2]);
				corr_q[i].coeffs() = 0.5 *(otQ*q[i]).coeffs();
			}
			else
			{
				corr_x[i].setZero();
				corr_q[i].coeffs().setZero();
			}
		}

		return true;
	}

	void PositionBasedGenericConstraints::computeMatrixG(const Quaternionr &q, Eigen::Matrix<Real, 4, 3> &G)
	{
		G(0, 0) = -static_cast<Real>(0.5)*q.x();
		G(0, 1) = -static_cast<Real>(0.5)*q.y();
		G(0, 2) = -static_cast<Real>(0.5)*q.z();

		G(1, 0) = static_cast<Real>(0.5)*q.w();
		G(1, 1) = static_cast<Real>(0.5)*q.z();
		G(1, 2) = -static_cast<Real>(0.5)*q.y();

		G(2, 0) = -static_cast<Real>(0.5)*q.z();
		G(2, 1) =  static_cast<Real>(0.5)*q.w();
		G(2, 2) =  static_cast<Real>(0.5)*q.x();

		G(3, 0) =  static_cast<Real>(0.5)*q.y();
		G(3, 1) = -static_cast<Real>(0.5)*q.x();
		G(3, 2) =  static_cast<Real>(0.5)*q.w();
	}

	template<unsigned int numberOfRigidBodies, unsigned int dim>
	void PositionBasedGenericConstraints::approximateGradient(
		const unsigned int i,										// compute dC / dxi
		const Real invMass[numberOfRigidBodies],					// inverse mass is zero if body is static
		const Vector3r x[numberOfRigidBodies],						// positions of bodies
		const Matrix3r inertiaInverseW[numberOfRigidBodies],		// inverse inertia tensor (world space) of bodies
		const Quaternionr q[numberOfRigidBodies],
		void *userData,

		void(*constraintFct)(
			const unsigned int numParticles,
			const Real invMass[numberOfRigidBodies],					// inverse mass is zero if body is static
			const Vector3r x[numberOfRigidBodies],						// positions of bodies
			const Matrix3r inertiaInverseW[numberOfRigidBodies],		// inverse inertia tensor (world space) of bodies
			const Quaternionr q[numberOfRigidBodies],
			void *userData,
			Eigen::Matrix<Real, dim, 1> &constraintValue),

		Eigen::Matrix<Real, dim, 6> &jacobian)
	{
		Vector3r xTemp[numberOfRigidBodies];
		Quaternionr qTemp[numberOfRigidBodies];
		for (unsigned int j = 0; j < numberOfRigidBodies; j++)
		{
			xTemp[j] = x[j];
			qTemp[j] = q[j];
		}

		Real eps = static_cast<Real>(1.e-6);
		jacobian.setZero();
		Vector3r x_i = xTemp[i];

		// Translation
		for (unsigned int j = 0; j < 3; j++)
		{
			xTemp[i][j] += eps;

			Eigen::Matrix<Real, dim, 1> e_p, e_m;
			constraintFct(numberOfRigidBodies, invMass, xTemp, inertiaInverseW, q, userData, e_p);
			xTemp[i][j] = x_i[j] - eps;
			constraintFct(numberOfRigidBodies, invMass, xTemp, inertiaInverseW, q, userData, e_m);
			xTemp[i][j] = x_i[j];

			jacobian.template block<dim, 1>(0, j) = (e_p - e_m) * (1.0 / (2.0*eps));
		}

		// Rotation
		Quaternionr q_i = qTemp[i];
		Eigen::Matrix<Real, dim, 1> jacobian4[4];
		for (unsigned int j = 0; j < 4; j++)
		{
			qTemp[i].coeffs()[j] += eps;

			Eigen::Matrix<Real, dim, 1> e_p, e_m;
			constraintFct(numberOfRigidBodies, invMass, x, inertiaInverseW, qTemp, userData, e_p);
			qTemp[i].coeffs()[j] = q_i.coeffs()[j] - eps;
			constraintFct(numberOfRigidBodies, invMass, x, inertiaInverseW, qTemp, userData, e_m);
			qTemp[i].coeffs()[j] = q_i.coeffs()[j];

			jacobian4[j] = (e_p - e_m) * (1.0 / (2.0*eps));
		}

		Eigen::Matrix<Real, 4, 3> G;
		computeMatrixG(q[i], G);
		// Bring in correct order
		Eigen::Matrix<Real, dim, 4> jh;
		jh.template block<dim, 1>(0, 0) = jacobian4[3];
		jh.template block<dim, 1>(0, 1) = jacobian4[0];
		jh.template block<dim, 1>(0, 2) = jacobian4[1];
		jh.template block<dim, 1>(0, 3) = jacobian4[2];
		jacobian.template block<dim, 3>(0, 3) = jh * G;
	}

}

#endif
