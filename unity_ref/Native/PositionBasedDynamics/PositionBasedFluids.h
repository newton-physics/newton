#ifndef POSITION_BASED_FLUIDS_H
#define POSITION_BASED_FLUIDS_H

#include "Common/Common.h"

// ------------------------------------------------------------------------------------
namespace PBD
{
	class PositionBasedFluids
	{
	public:
		// -------------- Position Based Fluids  -----------------------------------------------------
		
		/** Perform an SPH computation of the density of a fluid particle: 
		* \f{equation*}{
		* \rho_i = \sum_j m_j W(\mathbf{x}_i-\mathbf{x}_j).		
		* \f}
		* An additional term is added for neighboring boundary particles 
		* according to \cite Akinci:2012 in order to perform boundary handling.\n\n
		* Remark: A neighboring particle with an index >= numberOfParticles is
		* handled as boundary particle.\n\n
		*
		* More information can be found in the following papers: \cite Macklin:2013:PBF, \cite BMOTM2014, \cite BMM2015
		*
		* @param particleIndex	index of current fluid particle
		* @param numberOfParticles	number of fluid particles
		* @param x	array of all particle positions
		* @param mass array of all particle masses
		* @param boundaryX array of all boundary particles
		* @param boundaryPsi array of all boundary psi values (see \cite Akinci:2012)
		* @param numNeighbors number of neighbors
		* @param neighbors array with indices of all neighbors (indices larger than numberOfParticles are boundary particles)
		* @param density0 rest density
		* @param boundaryHandling perform boundary handling (see \cite Akinci:2012)
		* @param density_err returns the clamped density error (can be used for enforcing a maximal global density error)
		* @param density return the density
		*/	
		static bool computePBFDensity(
			const unsigned int particleIndex,				// current fluid particle	
			const unsigned int numberOfParticles,			// number of fluid particles 
			const Vector3r x[],						// array of all particle positions
			const Real mass[],								// array of all particle masses
			const Vector3r boundaryX[],				// array of all boundary particles
			const Real boundaryPsi[],						// array of all boundary psi values (Akinci2012)
			const unsigned int numNeighbors,				// number of neighbors 
			const unsigned int neighbors[],					// array with indices of all neighbors (indices larger than numberOfParticles are boundary particles)
			const Real density0,							// rest density
			const bool boundaryHandling,					// perform boundary handling (Akinci2012)
			Real &density_err,								// returns the clamped density error (can be used for enforcing a maximal global density error)
			Real &density);								// return the density

		/**
		 * Compute Lagrange multiplier \f$\lambda_i\f$ for a fluid particle which is required by
		 * the solver step:
		 * \f{equation*}{
		 * \lambda_i = -\frac{C_i(\mathbf{x}_1,...,\mathbf{x}_n)}{\sum_k \|\nabla_{\mathbf{x}_k} C_i\|^2 + \varepsilon}
		 * \f}
		 * with the constraint gradient:
		 * \f{equation*}{
		 * \nabla_{\mathbf{x}_k}C_i = \frac{m_j}{\rho_0}
		 * \begin{cases}
		 * \sum\limits_j \nabla_{\mathbf{x}_k} W(\mathbf{x}_i-\mathbf{x}_j, h) & \text{if }  k = i \\
		 * -\nabla_{\mathbf{x}_k} W(\mathbf{x}_i-\mathbf{x}_j, h)  & \text{if } k = j.
		 * \end{cases}
		 * \f}
		 * \n
		 * Remark: The computation of the gradient is extended for neighboring boundary particles
		 * according to \cite Akinci:2012 to perform a boundary handling. A neighboring 
		 * particle with an index >= numberOfParticles is handled as boundary particle.\n\n
		 *
		 * More information can be found in the following papers: \cite Macklin:2013:PBF, \cite BMOTM2014, \cite BMM2015
		 *
		 * @param particleIndex	index of current fluid particle
		 * @param numberOfParticles	number of fluid particles
		 * @param x	array of all particle positions
		 * @param mass array of all particle masses
		 * @param boundaryX array of all boundary particles
		 * @param boundaryPsi array of all boundary psi values (see \cite Akinci:2012)
		 * @param density density of current fluid particle
		 * @param numNeighbors number of neighbors
		 * @param neighbors array with indices of all neighbors (indices larger than numberOfParticles are boundary particles)
		 * @param density0 rest density
		 * @param boundaryHandling perform boundary handling (see \cite Akinci:2012)
		 * @param lambda returns the Lagrange multiplier
		 */		
		static bool computePBFLagrangeMultiplier(
			const unsigned int particleIndex,				// current fluid particle	
			const unsigned int numberOfParticles,			// number of fluid particles 
			const Vector3r x[],						// array of all particle positions
			const Real mass[],								// array of all particle masses
			const Vector3r boundaryX[],				// array of all boundary particles
			const Real boundaryPsi[],						// array of all boundary psi values (Akinci2012)
			const Real density,							// density of current fluid particle
			const unsigned int numNeighbors,				// number of neighbors 
			const unsigned int neighbors[],					// array with indices of all neighbors
			const Real density0,							// rest density
			const bool boundaryHandling,					// perform boundary handling (Akinci2012)
			Real &lambda);									// returns the Lagrange multiplier

		/** Perform a solver step for a fluid particle:
		*
		* \f{equation*}{
		* \Delta\mathbf{x}_{i} = \frac{m_j}{\rho_0}\sum\limits_j{\left(\lambda_i + \lambda_j\right)\nabla W(\mathbf{x}_i-\mathbf{x}_j, h)},
		* \f}
		* where \f$h\f$ is the smoothing length of the kernel function \f$W\f$.\n
		*\n
		* Remark: The computation of the position correction is extended for neighboring boundary particles
		* according to \cite Akinci:2012 to perform a boundary handling. A neighboring
		* particle with an index >= numberOfParticles is handled as boundary particle.\n\n
		*
		* More information can be found in the following papers: \cite Macklin:2013:PBF, \cite BMOTM2014, \cite BMM2015
		*
		* @param particleIndex	index of current fluid particle
		* @param numberOfParticles	number of fluid particles
		* @param x	array of all particle positions
		* @param mass array of all particle masses
		* @param boundaryX array of all boundary particles
		* @param boundaryPsi array of all boundary psi values (see \cite Akinci:2012)
		* @param numNeighbors number of neighbors
		* @param neighbors array with indices of all neighbors (indices larger than numberOfParticles are boundary particles)
		* @param density0 rest density
		* @param boundaryHandling perform boundary handling (see \cite Akinci:2012)
		* @param lambda Lagrange multipliers
		* @param corr returns the position correction for the current fluid particle
		*/
		static bool solveDensityConstraint(
			const unsigned int particleIndex,				// current fluid particle	
			const unsigned int numberOfParticles,			// number of fluid particles 
			const Vector3r x[],						// array of all particle positions
			const Real mass[],								// array of all particle masses
			const Vector3r boundaryX[],				// array of all boundary particles
			const Real boundaryPsi[],						// array of all boundary psi values (Akinci2012)
			const unsigned int numNeighbors,				// number of neighbors 
			const unsigned int neighbors[],					// array with indices of all neighbors
			const Real density0,							// rest density
			const bool boundaryHandling,					// perform boundary handling (Akinci2012)
			const Real lambda[],							// Lagrange multiplier
			Vector3r &corr);							// returns the position correction for the current fluid particle
	};
}

#endif