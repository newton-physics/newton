#ifndef _CubicSDFCollisionDetection_H
#define _CubicSDFCollisionDetection_H

#include "Common/Common.h"
#include "Simulation/DistanceFieldCollisionDetection.h"
#include <memory>

#include "Discregrid/All"

namespace PBD
{
	/** Collision detection based on cubic signed distance fields. 
	*/
	class CubicSDFCollisionDetection : public DistanceFieldCollisionDetection
	{
	public:
		using Grid = Discregrid::CubicLagrangeDiscreteGrid;
		using GridPtr = std::shared_ptr<Discregrid::CubicLagrangeDiscreteGrid>;

		struct CubicSDFCollisionObject : public DistanceFieldCollisionDetection::DistanceFieldCollisionObject
		{
			std::string m_sdfFile;
			Vector3r m_scale;
			GridPtr m_sdf;
			static int TYPE_ID;

			CubicSDFCollisionObject();
			virtual ~CubicSDFCollisionObject();
			virtual int &getTypeId() const { return TYPE_ID; }
			virtual bool collisionTest(const Vector3r &x, const Real tolerance, Vector3r &cp, Vector3r &n, Real &dist, const Real maxDist = 0.0);
			virtual double distance(const Eigen::Vector3d &x, const Real tolerance);
		};

	public:
		CubicSDFCollisionDetection();
		virtual ~CubicSDFCollisionDetection();

		virtual bool isDistanceFieldCollisionObject(CollisionObject *co) const;

		void addCubicSDFCollisionObject(const unsigned int bodyIndex, const unsigned int bodyType, const Vector3r *vertices, const unsigned int numVertices, const std::string &sdfFile, const Vector3r &scale, const bool testMesh = true, const bool invertSDF = false);
		void addCubicSDFCollisionObject(const unsigned int bodyIndex, const unsigned int bodyType, const Vector3r *vertices, const unsigned int numVertices, GridPtr sdf, const Vector3r &scale, const bool testMesh = true, const bool invertSDF = false);
	};
}

#endif
