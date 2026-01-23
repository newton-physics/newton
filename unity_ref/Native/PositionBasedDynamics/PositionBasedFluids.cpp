#include "PositionBasedFluids.h"
#include <cfloat>
#include "SPHKernels.h"

using namespace PBD;

// ----------------------------------------------------------------------------------------------
bool PositionBasedFluids::computePBFDensity(
	const unsigned int particleIndex,
	const unsigned int numberOfParticles,
	const Vector3r x[],
	const Real mass[],
	const Vector3r boundaryX[],
	const Real boundaryPsi[],
	const unsigned int numNeighbors,
	const unsigned int neighbors[],
	const Real density0,
	const bool boundaryHandling,
	Real &density_err,
	Real &density)
{
	// Compute current density for particle i
	density = mass[particleIndex] * CubicKernel::W_zero();
	for (unsigned int j = 0; j < numNeighbors; j++)
	{
		const unsigned int neighborIndex = neighbors[j];
		if (neighborIndex < numberOfParticles)		// Test if fluid particle
		{
			density += mass[neighborIndex] * CubicKernel::W(x[particleIndex] - x[neighborIndex]);
		}
		else if (boundaryHandling)
		{
			// Boundary: Akinci2012
			density += boundaryPsi[neighborIndex - numberOfParticles] * CubicKernel::W(x[particleIndex] - boundaryX[neighborIndex - numberOfParticles]);
		}
	}

	density_err = std::max(density, density0) - density0;
	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedFluids::computePBFLagrangeMultiplier(
	const unsigned int particleIndex,
	const unsigned int numberOfParticles,
	const Vector3r x[],	
	const Real mass[],
	const Vector3r boundaryX[],
	const Real boundaryPsi[],
	const Real density,
	const unsigned int numNeighbors,
	const unsigned int neighbors[],
	const Real density0,
	const bool boundaryHandling,
	Real &lambda)
{
	const Real eps = static_cast<Real>(1.0e-6);

	// Evaluate constraint function
	const Real C = std::max(density / density0 - static_cast<Real>(1.0), static_cast<Real>(0.0));			// clamp to prevent particle clumping at surface

	if (C != 0.0)
	{
		// Compute gradients dC/dx_j 
		Real sum_grad_C2 = 0.0;
		Vector3r gradC_i(0.0, 0.0, 0.0);

		for (unsigned int j = 0; j < numNeighbors; j++)
		{
			const unsigned int neighborIndex = neighbors[j];
			if (neighborIndex < numberOfParticles)		// Test if fluid particle
			{
				const Vector3r gradC_j = -mass[neighborIndex] / density0 * CubicKernel::gradW(x[particleIndex] - x[neighborIndex]);
				sum_grad_C2 += gradC_j.squaredNorm();
				gradC_i -= gradC_j;
			}
			else if (boundaryHandling)
			{
				// Boundary: Akinci2012
				const Vector3r gradC_j = -boundaryPsi[neighborIndex - numberOfParticles] / density0 * CubicKernel::gradW(x[particleIndex] - boundaryX[neighborIndex - numberOfParticles]);
				sum_grad_C2 += gradC_j.squaredNorm();
				gradC_i -= gradC_j;
			}
		}

		sum_grad_C2 += gradC_i.squaredNorm();

		// Compute lambda
		lambda = -C / (sum_grad_C2 + eps);
	}
	else
		lambda = 0.0;

	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedFluids::solveDensityConstraint(
	const unsigned int particleIndex,
	const unsigned int numberOfParticles,
	const Vector3r x[],	
	const Real mass[],
	const Vector3r boundaryX[],
	const Real boundaryPsi[],
	const unsigned int numNeighbors,
	const unsigned int neighbors[],
	const Real density0,
	const bool boundaryHandling,
	const Real lambda[],
	Vector3r &corr)
{
	// Compute position correction
	corr.setZero();
	for (unsigned int j = 0; j < numNeighbors; j++)
	{
		const unsigned int neighborIndex = neighbors[j];
		if (neighborIndex < numberOfParticles)		// Test if fluid particle
		{
			const Vector3r gradC_j = -mass[neighborIndex] / density0 * CubicKernel::gradW(x[particleIndex] - x[neighborIndex]);
			corr -= (lambda[particleIndex] + lambda[neighborIndex]) * gradC_j;
		}
		else if (boundaryHandling)
		{
			// Boundary: Akinci2012
			const Vector3r gradC_j = -boundaryPsi[neighborIndex - numberOfParticles] / density0 * CubicKernel::gradW(x[particleIndex] - boundaryX[neighborIndex - numberOfParticles]);
			corr -= (lambda[particleIndex]) * gradC_j;
		}
	}

	return true;
}
