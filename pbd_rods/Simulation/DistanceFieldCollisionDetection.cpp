#include "DistanceFieldCollisionDetection.h"
#include "Simulation/IDFactory.h"
#include "omp.h"

using namespace PBD;
using namespace Utilities;

int DistanceFieldCollisionDetection::DistanceFieldCollisionBox::TYPE_ID = IDFactory::getId();
int DistanceFieldCollisionDetection::DistanceFieldCollisionSphere::TYPE_ID = IDFactory::getId();
int DistanceFieldCollisionDetection::DistanceFieldCollisionTorus::TYPE_ID = IDFactory::getId();
int DistanceFieldCollisionDetection::DistanceFieldCollisionCylinder::TYPE_ID = IDFactory::getId();
int DistanceFieldCollisionDetection::DistanceFieldCollisionHollowSphere::TYPE_ID = IDFactory::getId();
int DistanceFieldCollisionDetection::DistanceFieldCollisionHollowBox::TYPE_ID = IDFactory::getId();
int DistanceFieldCollisionDetection::DistanceFieldCollisionObjectWithoutGeometry::TYPE_ID = IDFactory::getId();


DistanceFieldCollisionDetection::DistanceFieldCollisionDetection() :
	CollisionDetection()
{
}

DistanceFieldCollisionDetection::~DistanceFieldCollisionDetection()
{
}

void DistanceFieldCollisionDetection::collisionDetection(SimulationModel &model)
{
	model.resetContacts();
	const SimulationModel::RigidBodyVector &rigidBodies = model.getRigidBodies();
	const SimulationModel::TriangleModelVector &triModels = model.getTriangleModels();
	const SimulationModel::TetModelVector &tetModels = model.getTetModels();
	const ParticleData &pd = model.getParticles();

	std::vector < std::pair<unsigned int, unsigned int>> coPairs;
	for (unsigned int i = 0; i < m_collisionObjects.size(); i++)
	{
		CollisionDetection::CollisionObject *co1 = m_collisionObjects[i];
		for (unsigned int k = 0; k < m_collisionObjects.size(); k++)
		{
			CollisionDetection::CollisionObject *co2 = m_collisionObjects[k];
			if ((i != k))
			{
				// ToDo: self collisions for deformables
				coPairs.push_back({ i, k });
			}
		}
	}

	//omp_set_num_threads(1);
	std::vector<std::vector<ContactData> > contacts_mt;	
#ifdef _DEBUG
	const unsigned int maxThreads = 1;
#else
	const unsigned int maxThreads = omp_get_max_threads();
#endif
	contacts_mt.resize(maxThreads);

	#pragma omp parallel default(shared)
	{
		// Update BVHs
		#pragma omp for schedule(static)  
		for (int i = 0; i < (int)m_collisionObjects.size(); i++)
		{
			CollisionDetection::CollisionObject *co = m_collisionObjects[i];
			updateAABB(model, co);
			if (isDistanceFieldCollisionObject(co))
			{
				if (co->m_bodyType == CollisionDetection::CollisionObject::TriangleModelCollisionObjectType) 
				{
					DistanceFieldCollisionObject *sco = (DistanceFieldCollisionObject*)co;

					TriangleModel* tm = triModels[co->m_bodyIndex];
					const unsigned int offset = tm->getIndexOffset();
					const IndexedFaceMesh& mesh = tm->getParticleMesh();
					const unsigned int numVert = mesh.numVertices();
					sco->m_bvh.init(&pd.getPosition(offset), numVert);
					sco->m_bvh.update();
				}
				else if (co->m_bodyType == CollisionDetection::CollisionObject::TetModelCollisionObjectType)
				{
					TetModel* tm = tetModels[co->m_bodyIndex];
					const unsigned int offset = tm->getIndexOffset();
					const IndexedTetMesh& mesh = tm->getParticleMesh();
					const unsigned int numVert = mesh.numVertices();

					DistanceFieldCollisionObject *sco = (DistanceFieldCollisionObject*)co;
					sco->m_bvh.init(&pd.getPosition(offset), numVert);
					sco->m_bvhTets.updateVertices(&pd.getPosition(offset));
					sco->m_bvhTets0.updateVertices(&pd.getPosition(offset));

					sco->m_bvh.update();
					sco->m_bvhTets.update();
				}
			}
		}

		#pragma omp for schedule(static)
		for (int i = 0; i < (int)coPairs.size(); i++)
		{
			std::pair<unsigned int, unsigned int> &coPair = coPairs[i];
			CollisionDetection::CollisionObject *co1 = m_collisionObjects[coPair.first];
			CollisionDetection::CollisionObject *co2 = m_collisionObjects[coPair.second];

			if (((co2->m_bodyType != CollisionDetection::CollisionObject::RigidBodyCollisionObjectType) &&
				(co2->m_bodyType != CollisionDetection::CollisionObject::TetModelCollisionObjectType)) ||
				!isDistanceFieldCollisionObject(co1) ||
				!isDistanceFieldCollisionObject(co2) ||
				!AABB::intersection(co1->m_aabb, co2->m_aabb))
				continue;


			if ((co1->m_bodyType == CollisionDetection::CollisionObject::RigidBodyCollisionObjectType) &&
				(co2->m_bodyType == CollisionDetection::CollisionObject::RigidBodyCollisionObjectType) &&
				((DistanceFieldCollisionObject*) co1)->m_testMesh)
			{
				RigidBody *rb1 = rigidBodies[co1->m_bodyIndex];				
				RigidBody *rb2 = rigidBodies[co2->m_bodyIndex];
				const Real restitutionCoeff = rb1->getRestitutionCoeff() * rb2->getRestitutionCoeff();
				const Real frictionCoeff = rb1->getFrictionCoeff() + rb2->getFrictionCoeff();
				collisionDetectionRigidBodies(rb1, (DistanceFieldCollisionObject*)co1, rb2, (DistanceFieldCollisionObject*)co2,
					restitutionCoeff, frictionCoeff
					, contacts_mt
					);
			}
			else if ((co1->m_bodyType == CollisionDetection::CollisionObject::TriangleModelCollisionObjectType) &&
					(co2->m_bodyType == CollisionDetection::CollisionObject::RigidBodyCollisionObjectType) &&
					((DistanceFieldCollisionObject*)co1)->m_testMesh)
			{
				TriangleModel *tm = triModels[co1->m_bodyIndex];
				RigidBody *rb2 = rigidBodies[co2->m_bodyIndex];
				const unsigned int offset = tm->getIndexOffset();
				const IndexedFaceMesh &mesh = tm->getParticleMesh();
				const unsigned int numVert = mesh.numVertices();
				const Real restitutionCoeff = tm->getRestitutionCoeff() * rb2->getRestitutionCoeff();
				const Real frictionCoeff = tm->getFrictionCoeff() + rb2->getFrictionCoeff();
				collisionDetectionRBSolid(pd, offset, numVert, (DistanceFieldCollisionObject*)co1, rb2, (DistanceFieldCollisionObject*)co2,
					restitutionCoeff, frictionCoeff
					, contacts_mt
					);
			}
			else if ((co1->m_bodyType == CollisionDetection::CollisionObject::TetModelCollisionObjectType) && 
					(co2->m_bodyType == CollisionDetection::CollisionObject::RigidBodyCollisionObjectType) &&
					((DistanceFieldCollisionObject*)co1)->m_testMesh)
			{
				TetModel *tm = tetModels[co1->m_bodyIndex];
				RigidBody *rb2 = rigidBodies[co2->m_bodyIndex];
				const unsigned int offset = tm->getIndexOffset();
				const IndexedTetMesh &mesh = tm->getParticleMesh();
				const unsigned int numVert = mesh.numVertices();
				const Real restitutionCoeff = tm->getRestitutionCoeff() * rb2->getRestitutionCoeff();
				const Real frictionCoeff = tm->getFrictionCoeff() + rb2->getFrictionCoeff();
				collisionDetectionRBSolid(pd, offset, numVert, (DistanceFieldCollisionObject*)co1, rb2, (DistanceFieldCollisionObject*)co2,
					restitutionCoeff, frictionCoeff
					, contacts_mt
					);
			}
 			else if ((co1->m_bodyType == CollisionDetection::CollisionObject::TetModelCollisionObjectType) &&
 				(co2->m_bodyType == CollisionDetection::CollisionObject::TetModelCollisionObjectType) &&
 				((DistanceFieldCollisionObject*)co1)->m_testMesh)
 			{
 				TetModel *tm1 = tetModels[co1->m_bodyIndex];
 				TetModel *tm2 = tetModels[co2->m_bodyIndex];
 				const unsigned int offset = tm1->getIndexOffset();
 				const IndexedTetMesh &mesh = tm1->getParticleMesh();
 				const unsigned int numVert = mesh.numVertices();
 				const Real restitutionCoeff = tm1->getRestitutionCoeff() * tm2->getRestitutionCoeff();
 				const Real frictionCoeff = tm1->getFrictionCoeff() + tm2->getFrictionCoeff();
 				collisionDetectionSolidSolid(pd, offset, numVert, (DistanceFieldCollisionObject*)co1, tm2, (DistanceFieldCollisionObject*)co2,
 					restitutionCoeff, frictionCoeff
 					, contacts_mt
 				);
 			}
		}
	}

	for (unsigned int i = 0; i < contacts_mt.size(); i++)
	{
		for (unsigned int j = 0; j < contacts_mt[i].size(); j++)
		{
			if (contacts_mt[i][j].m_type == 1)
				addParticleRigidBodyContact(contacts_mt[i][j].m_index1, contacts_mt[i][j].m_index2,
					contacts_mt[i][j].m_cp1, contacts_mt[i][j].m_cp2, contacts_mt[i][j].m_normal,
					contacts_mt[i][j].m_dist, contacts_mt[i][j].m_restitution, contacts_mt[i][j].m_friction);
			else if (contacts_mt[i][j].m_type == 0)
				addRigidBodyContact(contacts_mt[i][j].m_index1, contacts_mt[i][j].m_index2,
					contacts_mt[i][j].m_cp1, contacts_mt[i][j].m_cp2, contacts_mt[i][j].m_normal,
					contacts_mt[i][j].m_dist, contacts_mt[i][j].m_restitution, contacts_mt[i][j].m_friction);
			else if (contacts_mt[i][j].m_type == 2)
			{
				addParticleSolidContact(contacts_mt[i][j].m_index1, contacts_mt[i][j].m_index2,
					contacts_mt[i][j].m_elementIndex2, contacts_mt[i][j].m_bary2,
					contacts_mt[i][j].m_cp1, contacts_mt[i][j].m_cp2, contacts_mt[i][j].m_normal,					
					contacts_mt[i][j].m_dist, contacts_mt[i][j].m_restitution, contacts_mt[i][j].m_friction);
			}
		}
	}
}

void DistanceFieldCollisionDetection::collisionDetectionRigidBodies(RigidBody *rb1, DistanceFieldCollisionObject *co1, RigidBody *rb2, DistanceFieldCollisionObject *co2, 
	const Real restitutionCoeff, const Real frictionCoeff
	, std::vector<std::vector<ContactData> > &contacts_mt
	)
{
	if ((rb1->getMass() == 0.0) && (rb2->getMass() == 0.0))
		return;

	const VertexData &vd = rb1->getGeometry().getVertexData();

	const Vector3r &com2 = rb2->getPosition();

	// remove the rotation of the main axis transformation that is performed
	// to get a diagonal inertia tensor since the distance function is 
	// evaluated in local coordinates
	//
	// transformation world to local:
	// p_local = R_initial^T ( R_MAT R^T (p_world - x) - x_initial + x_MAT)
	// 
	// transformation local to:
	// p_world = R R_MAT^T (R_initial p_local + x_initial - x_MAT) + x
	//
	const Matrix3r &R = rb2->getTransformationR();
	const Vector3r &v1 = rb2->getTransformationV1();
	const Vector3r &v2 = rb2->getTransformationV2();

	const PointCloudBSH &bvh = ((DistanceFieldCollisionDetection::DistanceFieldCollisionObject*) co1)->m_bvh;
	std::function<bool(unsigned int, unsigned int)> predicate = [&](unsigned int node_index, unsigned int depth)
	{
		const BoundingSphere &bs = bvh.hull(node_index);
		const Vector3r &sphere_x = bs.x();
		const Vector3r sphere_x_w = rb1->getRotation() * sphere_x + rb1->getPosition();

		AlignedBox3r box3f;
		box3f.extend(co2->m_aabb.m_p[0]);
		box3f.extend(co2->m_aabb.m_p[1]);
		const Real dist = box3f.exteriorDistance(sphere_x_w);

		// Test if center of bounding sphere intersects AABB
		if (dist < bs.r())
		{
			// Test if distance of center of bounding sphere to collision object is smaller than the radius
			const Vector3r x = R * (sphere_x_w - com2) + v1;
			const double dist2 = co2->distance(x.template cast<double>(), m_tolerance);
			if (dist2 == std::numeric_limits<double>::max())
				return true;
			if (dist2 < bs.r())
				return true;
		}
		return false;
	};
	std::function<void(unsigned int, unsigned int)> cb = [&](unsigned int node_index, unsigned int depth)
	{
		auto const& node = bvh.node(node_index);
		if (!node.is_leaf())
			return;

		for (auto i = node.begin; i < node.begin + node.n; ++i)
		{
			unsigned int index = bvh.entity(i);
			const Vector3r &x_w = vd.getPosition(index);
			const Vector3r x = R * (x_w - com2) + v1;
			Vector3r cp, n;
			Real dist;
			if (co2->collisionTest(x, m_tolerance, cp, n, dist))
			{
				const Vector3r cp_w = R.transpose() * cp + v2;
				const Vector3r n_w = R.transpose() * n;

#ifdef _DEBUG
				int tid = 0;
#else
				int tid = omp_get_thread_num();
#endif			

				contacts_mt[tid].push_back({ 0, co1->m_bodyIndex, co2->m_bodyIndex, x_w, cp_w, n_w, dist, restitutionCoeff, frictionCoeff });
			}
		}
	};
	bvh.traverse_depth_first(predicate, cb);
}


void DistanceFieldCollisionDetection::collisionDetectionRBSolid(const ParticleData &pd, const unsigned int offset, const unsigned int numVert,
	DistanceFieldCollisionObject *co1, RigidBody *rb2, DistanceFieldCollisionObject *co2, 
	const Real restitutionCoeff, const Real frictionCoeff
	, std::vector<std::vector<ContactData> > &contacts_mt
	)
{
	const Vector3r &com2 = rb2->getPosition();

	// remove the rotation of the main axis transformation that is performed
	// to get a diagonal inertia tensor since the distance function is 
	// evaluated in local coordinates
	//
	// transformation world to local:
	// p_local = R_initial^T ( R_MAT R^T (p_world - x) - x_initial + x_MAT)
	// 
	// transformation local to:
	// p_world = R R_MAT^T (R_initial p_local + x_initial - x_MAT) + x
	//
	const Matrix3r &R = rb2->getTransformationR();
	const Vector3r &v1 = rb2->getTransformationV1();
	const Vector3r &v2 = rb2->getTransformationV2();

	const PointCloudBSH &bvh = ((DistanceFieldCollisionDetection::DistanceFieldCollisionObject*) co1)->m_bvh;

	std::function<bool(unsigned int, unsigned int)> predicate = [&](unsigned int node_index, unsigned int depth)
	{
		const BoundingSphere &bs = bvh.hull(node_index);
		const Vector3r &sphere_x_w = bs.x();

		AlignedBox3r box3f;
		box3f.extend(co2->m_aabb.m_p[0]);
		box3f.extend(co2->m_aabb.m_p[1]);
		const Real dist = box3f.exteriorDistance(sphere_x_w);

		// Test if center of bounding sphere intersects AABB
		if (dist < bs.r())
		{
			// Test if distance of center of bounding sphere to collision object is smaller than the radius
			const Vector3r x = R * (sphere_x_w - com2) + v1;
			const double dist2 = co2->distance(x.template cast<double>(), m_tolerance);
			if (dist2 == std::numeric_limits<double>::max())
				return true;
			if (dist2 < bs.r())
				return true;
		}
		return false;
	};

	std::function<void(unsigned int, unsigned int)> cb = [&](unsigned int node_index, unsigned int depth)
	{
		auto const& node = bvh.node(node_index);
		if (!node.is_leaf())
			return;

		for (auto i = node.begin; i < node.begin + node.n; ++i)
		{
			unsigned int index = bvh.entity(i) + offset;
			const Vector3r &x_w = pd.getPosition(index);
			const Vector3r x = R * (x_w - com2) + v1;
			Vector3r cp, n;
			Real dist;
			if (co2->collisionTest(x, m_tolerance, cp, n, dist))
			{
				const Vector3r cp_w = R.transpose() * cp + v2;
				const Vector3r n_w = R.transpose() * n;

#ifdef _DEBUG
				int tid = 0;
#else
				int tid = omp_get_thread_num();
#endif			
				contacts_mt[tid].push_back({ 1, index, co2->m_bodyIndex, x_w, cp_w, n_w, dist, restitutionCoeff, frictionCoeff });
			}
		}
	};

	bvh.traverse_depth_first(predicate, cb);
}

void DistanceFieldCollisionDetection::collisionDetectionSolidSolid(const ParticleData &pd, const unsigned int offset, const unsigned int numVert,
	DistanceFieldCollisionObject *co1, TetModel *tm2, DistanceFieldCollisionObject *co2,
	const Real restitutionCoeff, const Real frictionCoeff
	, std::vector<std::vector<ContactData> > &contacts_mt
)
{
	const PointCloudBSH &bvh1 = ((DistanceFieldCollisionDetection::DistanceFieldCollisionObject*) co1)->m_bvh;
	const TetMeshBSH &bvh2 = ((DistanceFieldCollisionDetection::DistanceFieldCollisionObject*) co2)->m_bvhTets;
	const unsigned int *indices = tm2->getParticleMesh().getTets().data();
	const unsigned int offset2 = tm2->getIndexOffset();


	// callback function for BVH which is called if a leaf node in the point cloud BVH
	// has a collision with a leaf node in the tet BVH
	std::function<void(unsigned int, unsigned int)> cb = [&](unsigned int node_index1, unsigned int node_index2)
	{
		auto const& node1 = bvh1.node(node_index1);
		auto const& node2 = bvh2.node(node_index2);

		// loop over all primitives (points, tets) in the leaf nodes
		for (auto i = node1.begin; i < node1.begin + node1.n; ++i)
		{
			for (auto j = node2.begin; j < node2.begin + node2.n; ++j)
			{
				// Get sample point
				unsigned int index = bvh1.entity(i) + offset;
				const Vector3r &x_w = pd.getPosition(index);

				// Get tet
				const unsigned int tetIndex = bvh2.entity(j);
				const Vector3r &x0 = pd.getPosition(indices[4 * tetIndex] + offset2);
				const Vector3r &x1 = pd.getPosition(indices[4 * tetIndex + 1] + offset2);
				const Vector3r &x2 = pd.getPosition(indices[4 * tetIndex + 2] + offset2);
				const Vector3r &x3 = pd.getPosition(indices[4 * tetIndex + 3] + offset2);

				// Compute barycentric coordinates of point in tet
				Matrix3r A;
				A.col(0) = x1 - x0;
				A.col(1) = x2 - x0;
				A.col(2) = x3 - x0;
				Vector3r bary = A.inverse() * (x_w - x0);

				// check if point lies in tet using barycentric coordinates
 				if ((bary[0] >= 0.0) && (bary[1] >= 0.0) && (bary[2] >= 0.0) && 
 					(bary[0] + bary[1] + bary[2] <= 1.0))
				{
					// use barycentric coordinates to determine position of the point in the reference space of the tet
 					const Vector3r &X0 = pd.getPosition0(indices[4 * tetIndex] + offset2);
					const Vector3r &X1 = pd.getPosition0(indices[4 * tetIndex + 1] + offset2);
					const Vector3r &X2 = pd.getPosition0(indices[4 * tetIndex + 2] + offset2);
					const Vector3r &X3 = pd.getPosition0(indices[4 * tetIndex + 3] + offset2);

					Matrix3r A0;
					A0.col(0) = X1 - X0;
					A0.col(1) = X2 - X0;
					A0.col(2) = X3 - X0;

					// point in reference space of the tet
					const Vector3r X = X0 + A0 * bary;

					Vector3r cp_l, n_l;
					Real dist;

					// apply inverse initial transform to transform the point in the space of the 
					// signed distance field
					const Vector3r X_l = (tm2->getInitialR().transpose() * (X - tm2->getInitialX()));

					// perform collision test with distance field to get closest point on surface
					//if (co2->collisionTest(X_l, m_tolerance, cp_l, n_l, dist))
					if (co2->collisionTest(X_l, 0.0, cp_l, n_l, dist))
					{
						unsigned int cp_tetIndex; 
						Vector3r cp_bary;

						// transform closest point on surface back to the reference space of the tet model
						const Vector3r cp0 = (tm2->getInitialR() * cp_l + tm2->getInitialX());

						// find the tet which contains the resulting point
						if (findRefTetAt(pd, tm2, co2, cp0, cp_tetIndex, cp_bary))
						{
							// if we are in another tet, update matrix A
							Vector3r cp_w;
							if (cp_tetIndex != tetIndex)
							{
								const Vector3r &x0 = pd.getPosition(indices[4 * cp_tetIndex] + offset2);
								const Vector3r &x1 = pd.getPosition(indices[4 * cp_tetIndex + 1] + offset2);
								const Vector3r &x2 = pd.getPosition(indices[4 * cp_tetIndex + 2] + offset2);
								const Vector3r &x3 = pd.getPosition(indices[4 * cp_tetIndex + 3] + offset2);
								A.col(0) = x1 - x0;
								A.col(1) = x2 - x0;
								A.col(2) = x3 - x0;
								
								// compute world space contact point in body 2	
								cp_w = x0 + A * cp_bary;
							}
							else 
								// compute world space contact point in body 2	
								cp_w = x0 + A * cp_bary;							

#ifdef _DEBUG
							int tid = 0;
#else
							int tid = omp_get_thread_num();
#endif	

							Vector3r n_w = cp_w - x_w;

							// normalize normal vector
	 						const Real dist = (x_w - cp_w).norm();
							if (dist > 1.0e-6)
								n_w /= dist;

	 						contacts_mt[tid].push_back({ 2, index, co2->m_bodyIndex, x_w, cp_w, n_w, dist, restitutionCoeff, frictionCoeff, tetIndex, cp_tetIndex, bary, cp_bary });
						}
					}
				}
			}
		}
	};

	BVHTest::traverse(bvh1, bvh2, cb);

}

bool DistanceFieldCollisionDetection::isDistanceFieldCollisionObject(CollisionObject *co) const
{
	return (co->getTypeId() == DistanceFieldCollisionDetection::DistanceFieldCollisionBox::TYPE_ID) ||
		(co->getTypeId() == DistanceFieldCollisionDetection::DistanceFieldCollisionSphere::TYPE_ID) ||
		(co->getTypeId() == DistanceFieldCollisionDetection::DistanceFieldCollisionTorus::TYPE_ID) ||
		(co->getTypeId() == DistanceFieldCollisionDetection::DistanceFieldCollisionCylinder::TYPE_ID) ||
		(co->getTypeId() == DistanceFieldCollisionDetection::DistanceFieldCollisionHollowSphere::TYPE_ID) ||
		(co->getTypeId() == DistanceFieldCollisionDetection::DistanceFieldCollisionHollowBox::TYPE_ID) ||
		(co->getTypeId() == DistanceFieldCollisionDetection::DistanceFieldCollisionObjectWithoutGeometry::TYPE_ID);
}

void DistanceFieldCollisionDetection::addCollisionBox(const unsigned int bodyIndex, const unsigned int bodyType, const Vector3r *vertices, const unsigned int numVertices, const Vector3r &box, const bool testMesh, const bool invertSDF)
{
	DistanceFieldCollisionDetection::DistanceFieldCollisionBox *cf = new DistanceFieldCollisionDetection::DistanceFieldCollisionBox();
	cf->m_bodyIndex = bodyIndex;
	cf->m_bodyType = bodyType;
	// distance function requires 0.5*box 
	cf->m_box = 0.5*box;
	cf->m_bvh.init(vertices, numVertices);
	cf->m_bvh.construct();
	cf->m_testMesh = testMesh;
	if (invertSDF)
		cf->m_invertSDF = -1.0;
	m_collisionObjects.push_back(cf);
}

void DistanceFieldCollisionDetection::addCollisionSphere(const unsigned int bodyIndex, const unsigned int bodyType, const Vector3r *vertices, const unsigned int numVertices, const Real radius, const bool testMesh, const bool invertSDF)
{
	DistanceFieldCollisionDetection::DistanceFieldCollisionSphere *cs = new DistanceFieldCollisionDetection::DistanceFieldCollisionSphere();
	cs->m_bodyIndex = bodyIndex;
	cs->m_bodyType = bodyType;
	cs->m_radius = radius;
	cs->m_bvh.init(vertices, numVertices);
	cs->m_bvh.construct();
	cs->m_testMesh = testMesh;
	if (invertSDF)
		cs->m_invertSDF = -1.0;
	m_collisionObjects.push_back(cs);
}

void DistanceFieldCollisionDetection::addCollisionTorus(const unsigned int bodyIndex, const unsigned int bodyType, const Vector3r *vertices, const unsigned int numVertices, const Vector2r &radii, const bool testMesh, const bool invertSDF)
{
	DistanceFieldCollisionDetection::DistanceFieldCollisionTorus *ct = new DistanceFieldCollisionDetection::DistanceFieldCollisionTorus();
	ct->m_bodyIndex = bodyIndex;
	ct->m_bodyType = bodyType;
	ct->m_radii = radii;
	ct->m_bvh.init(vertices, numVertices);
	ct->m_bvh.construct();
	ct->m_testMesh = testMesh;
	if (invertSDF)
		ct->m_invertSDF = -1.0;
	m_collisionObjects.push_back(ct);
}

void DistanceFieldCollisionDetection::addCollisionCylinder(const unsigned int bodyIndex, const unsigned int bodyType, const Vector3r *vertices, const unsigned int numVertices, const Vector2r &dim, const bool testMesh, const bool invertSDF)
{
	DistanceFieldCollisionDetection::DistanceFieldCollisionCylinder *ct = new DistanceFieldCollisionDetection::DistanceFieldCollisionCylinder();
	ct->m_bodyIndex = bodyIndex;
	ct->m_bodyType = bodyType;
	ct->m_dim = dim;
	// distance function uses height/2
	ct->m_dim[1] *= 0.5;
	ct->m_bvh.init(vertices, numVertices);
	ct->m_bvh.construct();
	ct->m_testMesh = testMesh;
	if (invertSDF)
		ct->m_invertSDF = -1.0;
	m_collisionObjects.push_back(ct);
}

void DistanceFieldCollisionDetection::addCollisionHollowSphere(const unsigned int bodyIndex, const unsigned int bodyType, const Vector3r *vertices, const unsigned int numVertices, const Real radius, const Real thickness, const bool testMesh, const bool invertSDF)
{
	DistanceFieldCollisionDetection::DistanceFieldCollisionHollowSphere *cs = new DistanceFieldCollisionDetection::DistanceFieldCollisionHollowSphere();
	cs->m_bodyIndex = bodyIndex;
	cs->m_bodyType = bodyType;
	cs->m_radius = radius;
	cs->m_thickness = thickness;
	cs->m_bvh.init(vertices, numVertices);
	cs->m_bvh.construct();
	cs->m_testMesh = testMesh;
	if (invertSDF)
		cs->m_invertSDF = -1.0;
	m_collisionObjects.push_back(cs);
}

void DistanceFieldCollisionDetection::addCollisionHollowBox(const unsigned int bodyIndex, const unsigned int bodyType, const Vector3r *vertices, const unsigned int numVertices, const Vector3r &box, const Real thickness, const bool testMesh, const bool invertSDF)
{
	DistanceFieldCollisionDetection::DistanceFieldCollisionHollowBox *cf = new DistanceFieldCollisionDetection::DistanceFieldCollisionHollowBox();
	cf->m_bodyIndex = bodyIndex;
	cf->m_bodyType = bodyType;
	// distance function requires 0.5*box 
	cf->m_box = 0.5*box;
	cf->m_thickness = thickness;
	cf->m_bvh.init(vertices, numVertices);
	cf->m_bvh.construct();
	cf->m_testMesh = testMesh;
	if (invertSDF)
		cf->m_invertSDF = -1.0;
	m_collisionObjects.push_back(cf);
}

void DistanceFieldCollisionDetection::addCollisionObjectWithoutGeometry(const unsigned int bodyIndex, const unsigned int bodyType, const Vector3r *vertices, const unsigned int numVertices, const bool testMesh)
{
	DistanceFieldCollisionObjectWithoutGeometry *co = new DistanceFieldCollisionObjectWithoutGeometry();
	co->m_bodyIndex = bodyIndex;
	co->m_bodyType = bodyType;
	co->m_bvh.init(vertices, numVertices);
	co->m_bvh.construct();
	co->m_testMesh = testMesh;
	co->m_invertSDF = 1.0;
	m_collisionObjects.push_back(co);
}

double DistanceFieldCollisionDetection::DistanceFieldCollisionBox::distance(const Eigen::Vector3d &x, const Real tolerance)
{
	const Eigen::Vector3d box_d = m_box.template cast<double>();
	const Eigen::Vector3d x_d = x.template cast<double>();
	const Eigen::Vector3d d(fabs(x_d.x()) - box_d.x(), fabs(x_d.y()) - box_d.y(), fabs(x_d.z()) - box_d.z());
	const Eigen::Vector3d max_d(std::max(d.x(), 0.0), std::max(d.y(), 0.0), std::max(d.z(), 0.0));
	return m_invertSDF*(std::min(std::max(d.x(), std::max(d.y(), d.z())), 0.0) + max_d.norm()) - static_cast<double>(tolerance);
}

double DistanceFieldCollisionDetection::DistanceFieldCollisionSphere::distance(const Eigen::Vector3d &x, const Real tolerance)
{
	const Eigen::Vector3d d = x.template cast<double>();
	const double dl = d.norm();
	return m_invertSDF*(dl - static_cast<double>(m_radius)) - static_cast<double>(tolerance);
}

bool DistanceFieldCollisionDetection::DistanceFieldCollisionSphere::collisionTest(const Vector3r &x, const Real tolerance, Vector3r &cp, Vector3r &n, Real &dist, const Real maxDist)
{
	const Vector3r d = x;
	const Real dl = d.norm();
	dist = m_invertSDF*(dl - m_radius) - tolerance;
	if (dist < maxDist)
	{
		if (dl < 1.e-6)
			n.setZero();
		else
			n = m_invertSDF * d / dl;

		cp = ((m_radius+tolerance) * n);
		return true;
	}
	return false;
}

double DistanceFieldCollisionDetection::DistanceFieldCollisionTorus::distance(const Eigen::Vector3d &x, const Real tolerance)
{
	const Eigen::Vector2d radii_d = m_radii.template cast<double>();
	const Eigen::Vector2d q(Vector2r(x.x(), x.z()).norm() - radii_d.x(), x.y());
	return m_invertSDF*(q.norm() - radii_d.y()) - tolerance;
}

double DistanceFieldCollisionDetection::DistanceFieldCollisionCylinder::distance(const Eigen::Vector3d &x, const Real tolerance)
{
	const double l = sqrt(x.x()*x.x() + x.z()*x.z());
	const Eigen::Vector2d d = Eigen::Vector2d(fabs(l), fabs(x.y())) - m_dim.template cast<double>();
	const Eigen::Vector2d max_d(std::max(d.x(), 0.0), std::max(d.y(), 0.0));
	return m_invertSDF*(std::min(std::max(d.x(), d.y()), 0.0) + max_d.norm()) - static_cast<double>(tolerance);
}


double DistanceFieldCollisionDetection::DistanceFieldCollisionHollowSphere::distance(const Eigen::Vector3d &x, const Real tolerance)
{
	const Eigen::Vector3d d = x.template cast<double>();
	const double dl = d.norm();
	return m_invertSDF*(fabs(dl - static_cast<double>(m_radius)) - static_cast<double>(m_thickness)) - static_cast<double>(tolerance);
}

bool DistanceFieldCollisionDetection::DistanceFieldCollisionHollowSphere::collisionTest(const Vector3r &x, const Real tolerance, Vector3r &cp, Vector3r &n, Real &dist, const Real maxDist)
{
	const Vector3r d = x;
	const Real dl = d.norm();
	dist = m_invertSDF*(fabs(dl - m_radius) - m_thickness) - tolerance;
	if (dist < maxDist)
	{
		if (dl < 1.e-6)
			n.setZero();
		else if (dl < m_radius)
			n = -m_invertSDF*d / dl;
		else
			n = m_invertSDF*d / dl;

		cp = x - dist * n;
		return true;
	}
	return false;
}

double DistanceFieldCollisionDetection::DistanceFieldCollisionHollowBox::distance(const Eigen::Vector3d &x, const Real tolerance)
{
	const Eigen::Vector3d box_d = m_box.template cast<double>();
	const Eigen::Vector3d x_d = x.template cast<double>();
	const Eigen::Vector3d d = x_d.cwiseAbs() - box_d;
	const Eigen::Vector3d max_d = d.cwiseMax(Eigen::Vector3d(0.0, 0.0, 0.0));
	return m_invertSDF * (fabs(std::min(d.maxCoeff(), 0.0) + max_d.norm()) - m_thickness) - static_cast<double>(tolerance);
}

void DistanceFieldCollisionDetection::DistanceFieldCollisionObject::approximateNormal(const Eigen::Vector3d &x, const Real tolerance, Vector3r &n)
{
	// approximate gradient
	double eps = 1.e-6;
	n.setZero();
	Eigen::Vector3d xTmp = x;
	for (unsigned int j = 0; j < 3; j++)
	{
		xTmp[j] += eps;

		double e_p, e_m;
		e_p = distance(xTmp, tolerance);
		xTmp[j] = x[j] - eps;
		e_m = distance(xTmp, tolerance);
		xTmp[j] = x[j];

		double res = (e_p - e_m) * (1.0 / (2.0*eps));

		n[j] = static_cast<Real>(res);
	}

	const Real norm2 = n.squaredNorm();
	if (norm2 < 1.e-6)
		n.setZero();
	else
		n = n / sqrt(norm2);
}


bool DistanceFieldCollisionDetection::DistanceFieldCollisionObject::collisionTest(const Vector3r &x, const Real tolerance, Vector3r &cp, Vector3r &n, Real &dist, const Real maxDist)
{
	const Real t_d = static_cast<Real>(tolerance);
	dist = static_cast<Real>(distance(x.template cast<double>(), t_d));
	if (dist < maxDist)
	{
		// approximate gradient
		const Eigen::Vector3d x_d = x.template cast<double>();

		approximateNormal(x_d, t_d, n);

		cp = (x - dist * n);
		return true;
	}
	return false;
}

void DistanceFieldCollisionDetection::DistanceFieldCollisionObject::initTetBVH(const Vector3r *vertices, const unsigned int numVertices, const unsigned int *indices, const unsigned int numTets, const Real tolerance)
{
	if (m_bodyType == CollisionDetection::CollisionObject::TetModelCollisionObjectType)
	{
		m_bvhTets.init(vertices, numVertices, indices, numTets, tolerance);
		m_bvhTets.construct();

		// ToDo: copy constructor
		m_bvhTets0.init(vertices, numVertices, indices, numTets, 0.0);
		m_bvhTets0.construct();

	}
}

bool DistanceFieldCollisionDetection::findRefTetAt(const ParticleData &pd, TetModel *tm, const DistanceFieldCollisionDetection::DistanceFieldCollisionObject *co, const Vector3r &X,
	unsigned int &tetIndex, Vector3r &barycentricCoordinates)
{
 	const TetMeshBSH &bvh0 = ((DistanceFieldCollisionDetection::DistanceFieldCollisionObject*) co)->m_bvhTets0;
 	const unsigned int *indices = tm->getParticleMesh().getTets().data();
 	const unsigned int offset = tm->getIndexOffset();
	std::vector<Vector3r> bary;
	std::vector<unsigned int> tets;
	bary.reserve(100);
	tets.reserve(100);

	std::function<bool(unsigned int, unsigned int)> predicate = [&](unsigned int node_index, unsigned int depth)
 	{
 		const BoundingSphere &bs = bvh0.hull(node_index);
		return bs.contains(X);
 	};
 	std::function<void(unsigned int, unsigned int)> cb = [&](unsigned int node_index, unsigned int depth)
 	{
 		auto const& node = bvh0.node(node_index);
 		if (!node.is_leaf())
 			return;
 
 		for (auto i = node.begin; i < node.begin + node.n; ++i)
 		{
			const unsigned int tetIndex = bvh0.entity(i);

 			// use barycentric coordinates to determine position in reference space
 			const Vector3r &X0 = pd.getPosition0(indices[4 * tetIndex] + offset);
 			const Vector3r &X1 = pd.getPosition0(indices[4 * tetIndex + 1] + offset);
 			const Vector3r &X2 = pd.getPosition0(indices[4 * tetIndex + 2] + offset);
 			const Vector3r &X3 = pd.getPosition0(indices[4 * tetIndex + 3] + offset);

 			// Compute barycentric coordinates of point in tet
 			Matrix3r A;
			A.col(0) = X1 - X0;
			A.col(1) = X2 - X0;
			A.col(2) = X3 - X0;
 			bary.push_back(A.inverse() * (X - X0));
			tets.push_back(tetIndex);
 		}
 	};

 	bvh0.traverse_depth_first(predicate, cb);

	if (bary.size() == 0)
		return false;

	// find best set of barycentric coordinates
	unsigned int index = 0;
	Real minError = REAL_MAX;
	for (unsigned int i = 0; i < bary.size(); i++)
	{
		// Determine if barycentric coordinates are negative and add distance to 0 as error
		Real error = std::max(static_cast<Real>(0.0), -bary[i][0]);
		error += std::max(static_cast<Real>(0.0), -bary[i][1]);
		error += std::max(static_cast<Real>(0.0), -bary[i][2]);

		// Determine if sum of barycentric coordinates is larger than one and add distance to 1 as error
		error += std::max(static_cast<Real>(0.0), bary[i][0] + bary[i][1] + bary[i][2] - static_cast<Real>(1.0));

		if (error < minError)
		{
			minError = error;
			index = i;
		}
	}
	barycentricCoordinates = bary[index];
	tetIndex = tets[index];
	return true;
}

