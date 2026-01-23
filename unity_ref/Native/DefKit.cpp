#include "DefKit.h"
#include "Constraints.h"

extern "C" {




	void EXPORT_API PredictPositions_native(float dt, float damping, int pointsCount, btVector3* positions, btVector3* predictedPositions, btVector3* velocities, btVector3* forces, float* invMasses, btVector3* gravity)
	{
		float damp = (1.0f - damping);

		//#pragma omp for
		for (int i = 0; i < pointsCount; i++)
		{

			if (invMasses[i] != 0)
			{
				//  body.forces[i] -= body.velocities[i] * (damping * mass);
		
				velocities[i] += (forces[i] * invMasses[i] + gravity[0]) * dt;
				velocities[i] *= damp;
				predictedPositions[i] = positions[i] + velocities[i] * dt;
			}
			else
			{
				velocities[i].setZero();
				predictedPositions[i] = positions[i];
			}

		}

	}

	void EXPORT_API Integrate_native(float dt, int pointsCount, btVector3* positions, btVector3* predictedPositions, btVector3* velocities, float* invMasses)
	{
		float dtInv = 1.0f / dt;

		//#pragma omp for
		for (int i = 0; i < pointsCount; i++)
		{
			if (invMasses[i] != 0)
			{
				velocities[i] = (predictedPositions[i] - positions[i]) * dtInv;
				positions[i] = predictedPositions[i];
			}
		}
	}

	void EXPORT_API PredictRotationsPBD(float dt, float damping, int pointsCount, btQuaternion* orientations, btQuaternion* predictedOrientations, btVector3* angVelocities, btVector3* torques, float* quatInvMass)
	{

		float damp = (1.0f - damping);
		float halfDt = dt * 0.5f;
		for (int i = 0; i < pointsCount; i++)
		{

			if (quatInvMass[i] != 0)
			{
				// simple form without nutation effect
				angVelocities[i] += torques[i] * quatInvMass[i] * dt;
				angVelocities[i] *= damp;

				btQuaternion angVelQ(angVelocities[i].x(), angVelocities[i].y(), angVelocities[i].z(), 0);
				//predictedOrientations[i] = orientations[i] + (orientations[i] * btQuaternion(angVelocities[i].x(), angVelocities[i].y(), angVelocities[i].z(), 0)) * dt * 0.5f;
				predictedOrientations[i] = orientations[i] + (angVelQ * orientations[i]) * halfDt;
				predictedOrientations[i].normalize();

			}
			else
			{
				angVelocities[i].setZero();
				predictedOrientations[i] = orientations[i];
			}

		}

	}

	void EXPORT_API PredictRotationsPBD2(float dt, float damping, int pointsCount, btQuaternion* orientations, btQuaternion* predictedOrientations, btVector3* angVelocities, btVector3* torques, float* quatInvMass, btVector3* inertiaTensor)
	{
		btVector3 inertiaInv(1.0f / inertiaTensor->x(), 1.0f / inertiaTensor->y(), 1.0f / inertiaTensor->z());
		float damp = (1.0f - damping);

		for (int i = 0; i < pointsCount; i++)
		{

			if (quatInvMass[i] != 0)
			{
				// simple form without nutation effect
				angVelocities[i] += inertiaInv * torques[i] * dt;
				angVelocities[i] *= damp;

				//btQuaternion q = orientations[i];
				//float qX = (q.w() * angVelocities[i].x() + q.z() * angVelocities[i].y() - q.y() * angVelocities[i].z()) * 0.5f * dt;
				//float qY = (-q.z() * angVelocities[i].x() + q.w() * angVelocities[i].y() + q.x() * angVelocities[i].z()) * 0.5f * dt;
				//float qZ = (q.y() * angVelocities[i].x() - q.x() * angVelocities[i].y() + q.w() * angVelocities[i].z()) * 0.5f * dt;
				//float qW = (-q.x() * angVelocities[i].x() - q.y() * angVelocities[i].y() - q.z() * angVelocities[i].z()) * 0.5f * dt;

				predictedOrientations[i] = orientations[i] + orientations[i] * angVelocities[i] * dt * 0.5f;
				predictedOrientations[i].normalize();
			}
			else
			{
				angVelocities[i].setZero();
				predictedOrientations[i] = orientations[i];
			}

		}

	}


	void EXPORT_API IntegrateRotationsPBD(float dt, int pointsCount, btQuaternion* orientations, btQuaternion* predictedOrientations, btQuaternion* prevOrientations, btVector3* angVelocities, float* quatInvMass)
	{
		float dtInv2 = 2.0f / dt;

		for (int i = 0; i < pointsCount; i++)
		{
			//const Quaternionr relRot = (rotation * oldRotation.conjugate());
			//angularVelocity = relRot.vec() *(2.0 / h);

			if (quatInvMass[i] != 0) 
			{
				//btQuaternion relRot = orientations[i].inverse() * predictedOrientations[i];
				btQuaternion relRot = predictedOrientations[i] * orientations[i].inverse();
				angVelocities[i] = btVector3(relRot[0], relRot[1], relRot[2]) * dtInv2;

				prevOrientations[i] = orientations[i];
				orientations[i] = predictedOrientations[i];
			}
		}
	}




	



}