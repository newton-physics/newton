#pragma once

#include "LinearMath/btScalar.h"
#include "LinearMath/btVector3.h"
#include "LinearMath/btQuaternion.h"
#include "LinearMath/btMatrix3x3.h"
#include "LinearMath/btTransform.h"
#include "LinearMath/btTransformUtil.h"

#include "LinearMath/btDbvt.h"
#include "LinearMath/btDbvtAabbMm.h"

#include <vector>

#define EXPORT_API	__declspec(dllexport) 
#define EPSILON		0.0000001f
#define MAX_COLCOUNT 4096

extern "C" 
{
	
	void EXPORT_API Initialize();

	void EXPORT_API Destroy();

	void EXPORT_API PredictPositions_native(float dt, float damping, int pointsCount, btVector3* positions, btVector3* predictedPositions, btVector3* velocities, btVector3* forces, float* invMasses, btVector3* gravity);

	void EXPORT_API Integrate_native(float dt, int pointsCount, btVector3* positions, btVector3* predictedPositions, btVector3* velocities, float* invMasses);


}


struct Edge
{
	int idA;
	int idB;
	float restLength;
	int type;
};

struct Triangle
{
	int id;
	int pointAid;
	int pointBid;
	int pointCid;

};

struct Tetrahedron
{
	int idA;
	int idB;
	int idC;
	int idD;
};

struct TetNeighbours
{
	int idA;
	int idB;
	int idC;
	int idD;
};

struct CollisionResult
{
	int triangleIndex;
	float distance;
	int isBehind;
	btVector3 contactPoint;
	btVector3 normal;
	btVector3 contactBar;

};

