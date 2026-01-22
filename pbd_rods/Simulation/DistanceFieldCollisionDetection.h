#ifndef _SIMPLECOLLISIONDETECTION_H
#define _SIMPLECOLLISIONDETECTION_H

#include "Common/Common.h"
#include "Simulation/CollisionDetection.h"
#include "AABB.h"
#include "BoundingSphereHierarchy.h"

namespace PBD
{
	/** Distance field collision detection. */
	class DistanceFieldCollisionDetection : public CollisionDetection
	{
	public:
		struct DistanceFieldCollisionObject : public CollisionObject
		{		
			bool m_testMesh;
			Real m_invertSDF;
			PointCloudBSH m_bvh;
			TetMeshBSH m_bvhTets;
			TetMeshBSH m_bvhTets0;

			DistanceFieldCollisionObject() { m_testMesh = true; m_invertSDF = 1.0; }
			virtual ~DistanceFieldCollisionObject() {}
			virtual bool collisionTest(const Vector3r &x, const Real tolerance, Vector3r &cp, Vector3r &n, Real &dist, const Real maxDist = 0.0);
			virtual void approximateNormal(const Eigen::Vector3d &x, const Real tolerance, Vector3r &n);

			virtual double distance(const Eigen::Vector3d &x, const Real tolerance) = 0;
			void initTetBVH(const Vector3r *vertices, const unsigned int numVertices, const unsigned int *indices, const unsigned int numTets, const Real tolerance);
		};

		struct DistanceFieldCollisionObjectWithoutGeometry : public DistanceFieldCollisionObject
		{
			static int TYPE_ID;

			virtual ~DistanceFieldCollisionObjectWithoutGeometry() {}
			virtual int &getTypeId() const { return TYPE_ID; }
			virtual bool collisionTest(const Vector3r &x, const Real tolerance, Vector3r &cp, Vector3r &n, Real &dist, const Real maxDist = 0.0) { return false; }
			virtual double distance(const Eigen::Vector3d &x, const Real tolerance) { return 0.0; }
		};

		struct DistanceFieldCollisionBox : public DistanceFieldCollisionObject
		{
			Vector3r m_box;
			static int TYPE_ID;

			virtual ~DistanceFieldCollisionBox() {}
			virtual int &getTypeId() const { return TYPE_ID; }
			virtual double distance(const Eigen::Vector3d &x, const Real tolerance);
		};

		struct DistanceFieldCollisionSphere : public DistanceFieldCollisionObject
		{
			Real m_radius;
			static int TYPE_ID;

			virtual ~DistanceFieldCollisionSphere() {}
			virtual int &getTypeId() const { return TYPE_ID; }
			virtual bool collisionTest(const Vector3r &x, const Real tolerance, Vector3r &cp, Vector3r &n, Real &dist, const Real maxDist = 0.0);
			virtual double distance(const Eigen::Vector3d &x, const Real tolerance);
		};

		struct DistanceFieldCollisionTorus : public DistanceFieldCollisionObject
		{
			Vector2r m_radii;
			static int TYPE_ID;

			virtual ~DistanceFieldCollisionTorus() {}
			virtual int &getTypeId() const { return TYPE_ID; }
			virtual double distance(const Eigen::Vector3d &x, const Real tolerance);
		};

		struct DistanceFieldCollisionCylinder : public DistanceFieldCollisionObject
		{
			Vector2r m_dim;
			static int TYPE_ID;

			virtual ~DistanceFieldCollisionCylinder() {}
			virtual int &getTypeId() const { return TYPE_ID; }
			virtual double distance(const Eigen::Vector3d &x, const Real tolerance);
		};

		struct DistanceFieldCollisionHollowSphere : public DistanceFieldCollisionObject
		{
			Real m_radius;
			Real m_thickness;
			static int TYPE_ID;

			virtual ~DistanceFieldCollisionHollowSphere() {}
			virtual int &getTypeId() const { return TYPE_ID; }
			virtual bool collisionTest(const Vector3r &x, const Real tolerance, Vector3r &cp, Vector3r &n, Real &dist, const Real maxDist = 0.0);
			virtual double distance(const Eigen::Vector3d &x, const Real tolerance);
		};

		struct DistanceFieldCollisionHollowBox : public DistanceFieldCollisionObject
		{
			Vector3r m_box;
			Real m_thickness;
			static int TYPE_ID;

			virtual ~DistanceFieldCollisionHollowBox() {}
			virtual int &getTypeId() const { return TYPE_ID; }
			virtual double distance(const Eigen::Vector3d &x, const Real tolerance);
		};

		struct ContactData
		{
			char m_type;
			unsigned int m_index1;
			unsigned int m_index2;
			Vector3r m_cp1;
			Vector3r m_cp2;
			Vector3r m_normal;
			Real m_dist;
			Real m_restitution;
			Real m_friction;

			// Test
			unsigned int m_elementIndex1;
			unsigned int m_elementIndex2;
			Vector3r m_bary1;
			Vector3r m_bary2;
		};

	protected:
		void collisionDetectionRigidBodies(RigidBody *rb1, DistanceFieldCollisionObject *co1, RigidBody *rb2, DistanceFieldCollisionObject *co2,
			const Real restitutionCoeff, const Real frictionCoeff
			, std::vector<std::vector<ContactData> > &contacts_mt
			);
		void collisionDetectionRBSolid(const ParticleData &pd, const unsigned int offset, const unsigned int numVert, 
			DistanceFieldCollisionObject *co1, RigidBody *rb2, DistanceFieldCollisionObject *co2, 
			const Real restitutionCoeff, const Real frictionCoeff
			, std::vector<std::vector<ContactData> > &contacts_mt
			);

		void collisionDetectionSolidSolid(const ParticleData &pd, const unsigned int offset, const unsigned int numVert,
			DistanceFieldCollisionObject *co1, TetModel *tm2, DistanceFieldCollisionObject *co2,
			const Real restitutionCoeff, const Real frictionCoeff
			, std::vector<std::vector<ContactData> > &contacts_mt
		);

		bool findRefTetAt(const ParticleData &pd, TetModel *tm, const DistanceFieldCollisionDetection::DistanceFieldCollisionObject *co, const Vector3r &X, 
			unsigned int &tetIndex, Vector3r &barycentricCoordinates);


	public:
		DistanceFieldCollisionDetection();
		virtual ~DistanceFieldCollisionDetection();

		virtual void collisionDetection(SimulationModel &model);

		virtual bool isDistanceFieldCollisionObject(CollisionObject *co) const;

		void addCollisionBox(const unsigned int bodyIndex, const unsigned int bodyType, const Vector3r *vertices, const unsigned int numVertices, const Vector3r &box, const bool testMesh = true, const bool invertSDF = false);
		void addCollisionSphere(const unsigned int bodyIndex, const unsigned int bodyType, const Vector3r *vertices, const unsigned int numVertices, const Real radius, const bool testMesh = true, const bool invertSDF = false);
		void addCollisionTorus(const unsigned int bodyIndex, const unsigned int bodyType, const Vector3r *vertices, const unsigned int numVertices, const Vector2r &radii, const bool testMesh = true, const bool invertSDF = false);
		void addCollisionObjectWithoutGeometry(const unsigned int bodyIndex, const unsigned int bodyType, const Vector3r *vertices, const unsigned int numVertices, const bool testMesh);
		void addCollisionHollowSphere(const unsigned int bodyIndex, const unsigned int bodyType, const Vector3r *vertices, const unsigned int numVertices, const Real radius, const Real thickness, const bool testMesh = true, const bool invertSDF = false);
		void addCollisionHollowBox(const unsigned int bodyIndex, const unsigned int bodyType, const Vector3r *vertices, const unsigned int numVertices, const Vector3r &box, const Real thickness, const bool testMesh = true, const bool invertSDF = false);

		/** Add collision cylinder
		 *
		 * @param  bodyIndex index of corresponding body
		 * @param  bodyType type of corresponding body
		 * @param  dim (radius, height) of cylinder
		 */		
		void addCollisionCylinder(const unsigned int bodyIndex, const unsigned int bodyType, const Vector3r *vertices, const unsigned int numVertices, const Vector2r &dim, const bool testMesh = true, const bool invertSDF = false);
	};
}

#endif
