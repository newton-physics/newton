#include "DefKitAdv.h"
#include "DirectElasticRod.h"

#ifdef  VMPROTECT
#include "..\VMProtectSDK.h"
#endif

#include <iostream>
#include <set>
#include <map>

int DirectPositionBasedSolverForStiffRodsConstraint::TYPE_ID = 0;



//bool DirectPositionBasedSolverForStiffRodsConstraint::initConstraint(
//	const std::vector<std::pair<unsigned int, unsigned int>> & constraintSegmentIndices,
//	const std::vector<Vector3r> &constraintPositions,
//	const std::vector<Real> &averageRadii,
//	const std::vector<Real> &averageSegmentLengths,
//	const std::vector<Real> &youngsModuli,
//	const std::vector<Real> &torsionModuli
//)
//{
//	// create unique segment indices from joints
//
//	std::set<unsigned int> uniqueSegmentIndices;
//	for (auto &idxPair : constraintSegmentIndices)
//	{
//		uniqueSegmentIndices.insert(idxPair.first);
//		uniqueSegmentIndices.insert(idxPair.second);
//	}
//
//	delete[] m_bodies;
//	m_numberOfBodies = (unsigned int)uniqueSegmentIndices.size();
//	m_bodies = new unsigned int[m_numberOfBodies];
//
//	// initialize m_bodies for constraint colouring algorithm of multi threading implementation
//
//	size_t segmentIdx(0);
//
//	for (auto idx : uniqueSegmentIndices)
//	{
//		m_bodies[segmentIdx] = idx;
//		++segmentIdx;
//	}
//
//	// create RodSegment instances and map simulation model body indices to RodSegment indices
//
//	std::map<unsigned int, unsigned int> idxMap;
//	unsigned int idx(0);
//
//	m_Segments.reserve(uniqueSegmentIndices.size());
//	m_rodSegments.reserve(uniqueSegmentIndices.size());
//	int z = 0;
//	for (auto bodyIdx : uniqueSegmentIndices)
//	{
//		idx = (unsigned int)m_Segments.size();
//		idxMap[bodyIdx] = idx;
//
//		RodSegmentImpl seg = RodSegmentImpl(bodyIdx);
//		seg.m_inertiaTensor = Vector3r(0.1f, 0.1f, 0.1f);
//		seg.m_position = Vector3r(0, 0, z++); //TODO
//		seg.m_rotation = Quaternionr(1, 0, 0, 0);
//		seg.m_mass = 1;
//
//		m_Segments.push_back(seg);
//		m_rodSegments.push_back(&m_Segments.back());
//	}
//
//	// create rod constraints
//
//	m_Constraints.resize(constraintPositions.size());
//	m_rodConstraints.resize(constraintPositions.size());
//
//	for (size_t idx(0); idx < constraintPositions.size(); ++idx)
//	{
//		const std::pair<unsigned int, unsigned int> &bodyIndices(constraintSegmentIndices[idx]);
//		unsigned int firstSegmentIndex(idxMap.find(bodyIndices.first)->second);
//		unsigned int secondSegmentIndex(idxMap.find(bodyIndices.second)->second);
//
//		m_Constraints[idx].m_segments.push_back(firstSegmentIndex);
//		m_Constraints[idx].m_segments.push_back(secondSegmentIndex);
//		m_Constraints[idx].m_averageSegmentLength = averageSegmentLengths[idx];
//		m_rodConstraints[idx] = &m_Constraints[idx];
//	}
//
//	// initialize data of the sparse direct solver
//	deleteNodes();
//	DirectPositionBasedSolverForStiffRods::init_DirectPositionBasedSolverForStiffRodsConstraint(
//		m_rodConstraints, m_rodSegments, intervals, numberOfIntervals, forward, backward, root,
//		constraintPositions, averageRadii, youngsModuli, torsionModuli,
//		m_rightHandSide, m_lambdaSums, m_bendingAndTorsionJacobians, m_corr_x, m_corr_q);
//
//	return true;
//}

bool DirectPositionBasedSolverForStiffRodsConstraint::initConstraint(
	const std::vector<Vector3r>& segmentPositions,
	const std::vector<Quaternionr>& segmentRotations,
	const std::vector<Vector3r>& constraintPositions,
	const std::vector<Real>& averageRadii,
	const std::vector<Real>& averageSegmentLengths,
	const std::vector<Real>& youngsModuli,
	const std::vector<Real>& torsionModuli
)
{


	m_Segments.reserve(segmentPositions.size());
	m_rodSegments.reserve(segmentPositions.size());



#ifdef VMPROTECT
	float lambda = (float)VMProtectGetSerialNumberState();
	Real zeta = 1.000001f - lambda;
#else
	Real zeta = 1.0f;
#endif // VMPROTECT

	for (unsigned int i = 0; i < segmentPositions.size(); i++)
	{
		RodSegmentImpl seg = RodSegmentImpl(i);
		seg.m_inertiaTensor = Vector3r(0.1f, 0.1f, 0.1f);
		seg.m_position = segmentPositions[i];
		seg.m_rotation = segmentRotations[i];
		seg.m_mass = 1;
		seg.m_zeta = zeta;

		m_Segments.push_back(seg);
		m_rodSegments.push_back(&m_Segments.back());
	}

	m_Constraints.resize(constraintPositions.size());
	m_rodConstraints.resize(constraintPositions.size());

#ifdef VMPROTECT
	float ypsilon = 1.0000001f * zeta;
	Vector3r theta = Vector3r(ypsilon, ypsilon, ypsilon);
#else
	Vector3r theta = Vector3r(1.0f, 1.0f, 1.0f);
#endif // VMPROTECT

	for (unsigned int idx(0); idx < constraintPositions.size(); ++idx)
	{
		m_Constraints[idx].m_segments.push_back(idx);
		m_Constraints[idx].m_segments.push_back(idx + 1);
		m_Constraints[idx].m_thetaVector = theta;
		m_Constraints[idx].m_averageSegmentLength = averageSegmentLengths[idx];
		m_rodConstraints[idx] = &m_Constraints[idx];

	}

	m_bandedSolverNodesForward.resize(constraintPositions.size());
	m_bandedSolverNodesBackward.resize(constraintPositions.size());
	for (unsigned int idx(0); idx < constraintPositions.size(); ++idx)
	{
		BandedNode* constraintNode = new BandedNode();
		constraintNode->index = idx;
		constraintNode->object = NULL;
		constraintNode->M.setZero();
		constraintNode->D.setZero();
		constraintNode->Dinv.setZero();
		constraintNode->J.setZero();
		constraintNode->JMJT.setZero();
		constraintNode->soln.setZero();
		m_bandedSolverNodesForward[idx] = constraintNode;

		BandedNode* constraintNode2 = new BandedNode();
		constraintNode2->index = idx;
		constraintNode2->object = NULL;
		constraintNode2->M.setZero();
		constraintNode2->D.setZero();
		constraintNode2->Dinv.setZero();
		constraintNode2->J.setZero();
		constraintNode2->JMJT.setZero();
		constraintNode2->soln.setZero();
		m_bandedSolverNodesBackward[idx] = constraintNode2;

	}

	m_JMJT = MatrixXr(constraintPositions.size() * 6, constraintPositions.size() * 6);
	m_JMJT.setZero();

	m_RHS = VectorXr(constraintPositions.size() * 6);
	m_RHS.setZero();

	//m_JMJT_Band = MatrixXr(1 + 2 * 11 + 11, constraintPositions.size()* 6);
	
	//this was working fine with LAPACK's SPBSV
	//m_JMJT_Band = MatrixXr(1 + 11, constraintPositions.size() * 6);

	//this is for custom solver LDAB = 34
	m_JMJT_Band = MatrixXr(34, constraintPositions.size() * 6);
	m_JMJT_Band.setZero();



	//// initialize data of the sparse direct solver
	deleteNodes();

	DirectPositionBasedSolverForStiffRods::init_DirectPositionBasedSolverForStiffRodsConstraint(
		m_rodConstraints, m_rodSegments, intervals, numberOfIntervals, forward, backward, root,
		constraintPositions, averageRadii, youngsModuli, torsionModuli,
		m_rightHandSide, m_lambdaSums, m_bendingAndTorsionJacobians, m_corr_x, m_corr_q);

	return true;
}

void DirectPositionBasedSolverForStiffRodsConstraint::deleteNodes()
{
	std::list<PBD::Node*>::iterator nodeIter;
	for (int i = 0; i < numberOfIntervals; i++)
	{
		for (nodeIter = forward[i].begin(); nodeIter != forward[i].end(); nodeIter++)
		{
			PBD::Node* node = *nodeIter;

			// Root node does not have to be deleted
			if (node->parent != NULL)
				delete node;
		}
	}
}

bool DirectPositionBasedSolverForStiffRodsConstraint::initConstraintBeforeProjection(float dt)
{

	DirectPositionBasedSolverForStiffRods::initBeforeProjection_DirectPositionBasedSolverForStiffRodsConstraint(m_rodConstraints, static_cast<Real>(1.0 / dt), m_lambdaSums);
	return true;
}


bool DirectPositionBasedSolverForStiffRodsConstraint::updateConstraint()
{
	DirectPositionBasedSolverForStiffRods::update_DirectPositionBasedSolverForStiffRodsConstraint(m_rodConstraints, m_rodSegments);
	return true;
}


bool DirectPositionBasedSolverForStiffRodsConstraint::solvePositionConstraint(std::vector<Vector3r>& m_corr_x, std::vector<Quaternionr>& m_corr_q)
{
	const bool res = DirectPositionBasedSolverForStiffRods::solve_DirectPositionBasedSolverForStiffRodsConstraint(
		m_rodConstraints, m_rodSegments, intervals, numberOfIntervals, numberOfUpdates, forward, backward,
		m_rightHandSide, m_lambdaSums, m_bendingAndTorsionJacobians, m_corr_x, m_corr_q
	);

	return res;
}

bool DirectPositionBasedSolverForStiffRodsConstraint::solvePositionConstraintBanded(std::vector<Vector3r>& m_corr_x, std::vector<Quaternionr>& m_corr_q)
{
	const bool res = DirectPositionBasedSolverForStiffRods::solve_DirectPositionBasedSolverForStiffRodsConstraintBanded(
		m_rodConstraints, m_rodSegments, m_bandedSolverNodesForward, m_bandedSolverNodesBackward,
		m_rightHandSide, m_lambdaSums, numberOfUpdates, m_bendingAndTorsionJacobians, m_JMJT, m_JMJT_Band, m_RHS, m_corr_x, m_corr_q
	);

	return res;
}

bool DirectPositionBasedSolverForStiffRodsConstraint::computeJacobians_PositionConstraintBanded(int startId, int endId)
{
	const bool res = DirectPositionBasedSolverForStiffRods::computeJacobians_DirectPositionBasedSolverForStiffRodsConstraintBanded(
		startId, endId,
		m_rodConstraints, m_rodSegments, m_bandedSolverNodesForward, m_bandedSolverNodesBackward,
		m_rightHandSide, m_lambdaSums, numberOfUpdates, m_bendingAndTorsionJacobians, m_JMJT, m_JMJT_Band, m_RHS);

	return res;
}



bool DirectPositionBasedSolverForStiffRodsConstraint::assembleJMJT_PositionConstraintBanded(int startId, int endId)
{
	const bool res = DirectPositionBasedSolverForStiffRods::assembleJMJT_DirectPositionBasedSolverForStiffRodsConstraintBanded(
		startId, endId,
		m_rodConstraints, m_rodSegments, m_bandedSolverNodesForward, m_bandedSolverNodesBackward,
		m_rightHandSide, m_lambdaSums, numberOfUpdates, m_bendingAndTorsionJacobians, m_JMJT, m_JMJT_Band, m_RHS);

	return res;
}

bool DirectPositionBasedSolverForStiffRodsConstraint::computeJacobiansAndAssembleJMJT_PositionConstraintBanded(int startId, int endId)
{
	const bool res = DirectPositionBasedSolverForStiffRods::computeJacobiansAndAssemble_DirectPositionBasedSolverForStiffRodsConstraintBanded(
		startId, endId,
		m_rodConstraints, m_rodSegments, m_bandedSolverNodesForward, m_bandedSolverNodesBackward,
		m_rightHandSide, m_lambdaSums, numberOfUpdates, m_bendingAndTorsionJacobians, m_JMJT, m_JMJT_Band, m_RHS);

	return res;
}

bool DirectPositionBasedSolverForStiffRodsConstraint::solveJMJT_PositionConstraintBanded(std::vector<Vector3r>& m_corr_x, std::vector<Quaternionr>& m_corr_q)
{
	const bool res = DirectPositionBasedSolverForStiffRods::solveJMJT_DirectPositionBasedSolverForStiffRodsConstraintBanded(
		m_rodConstraints, m_rodSegments, m_bandedSolverNodesForward, m_bandedSolverNodesBackward,
		m_rightHandSide, m_lambdaSums, numberOfUpdates, m_bendingAndTorsionJacobians, m_JMJT, m_JMJT_Band, m_RHS, m_corr_x, m_corr_q
	);

	return res;
}



DirectPositionBasedSolverForStiffRodsConstraint::~DirectPositionBasedSolverForStiffRodsConstraint()
{
	deleteNodes();
	if (intervals != NULL)
		delete[] intervals;
	if (forward != NULL)
		delete[] forward;
	if (backward != NULL)
		delete[] backward;
	if (root != NULL)
		delete[] root;
	root = NULL;
	forward = NULL;
	backward = NULL;
	intervals = NULL;
	numberOfIntervals = 0;
}




extern "C"
{


	EXPORT_API DirectPositionBasedSolverForStiffRodsConstraint* InitDirectElasticRod(int pointsCount, btVector3* positions, btQuaternion* orientations, float radius, float* restLengths, float youngModulus, float torsionModulus)
	{
		DirectPositionBasedSolverForStiffRodsConstraint* c = new DirectPositionBasedSolverForStiffRodsConstraint();



		//std::vector<std::pair<unsigned int, unsigned int>>  constraintSegmentIndices;
		std::vector<Vector3r> segmentPositions;
		std::vector<Quaternionr> segmentRotations;
		std::vector<Vector3r> constraintPositions;
		std::vector<Real> averageSegmentLengths;
		std::vector<Real> averageRadii(pointsCount, radius);
		//std::vector<Real> averageSegmentLengths(pointsCount, restLength);
		std::vector<Real> youngsModuli(pointsCount, youngModulus);
		std::vector<Real> torsionModuli(pointsCount, torsionModulus);

		for (size_t i = 0; i < pointsCount; i++)
		{
			Vector3r v = Vector3r(positions[i].x(), positions[i].y(), positions[i].z());
			segmentPositions.push_back(v);

			//Quaternionr q(1, 0, 0, 0);
			Quaternionr q(orientations[i].w(), orientations[i].x(), orientations[i].y(), orientations[i].z());


			segmentRotations.push_back(q);
		}


		for (size_t i = 0; i < pointsCount - 1; i++)
		{
			//constraintSegmentIndices.push_back(std::pair<unsigned int, unsigned int>(i, i+1));


			constraintPositions.push_back((segmentPositions[i] + segmentPositions[i + 1]) * 0.5f);

			averageSegmentLengths.push_back(restLengths[i]);
		}


		const bool res = c->initConstraint(
			//constraintSegmentIndices,
			segmentPositions,
			segmentRotations,
			constraintPositions,
			averageRadii,
			averageSegmentLengths,
			youngsModuli,
			torsionModuli
		);

		return c;
	}


	EXPORT_API void  PrepareDirectElasticRodConstraints(DirectPositionBasedSolverForStiffRodsConstraint* rod, int pointsCount, float dt, btVector3* bendStiff, btVector3* restDarboux, float* restLengths, float youngModulusMult, float torsionModulusMult)
	{
		for (size_t i = 0; i < pointsCount; i++)
		{
			rod->m_Constraints[i].m_stiffnessCoefficientK = Vector3r(bendStiff[i].x() * youngModulusMult, bendStiff[i].y() * youngModulusMult, bendStiff[i].z() * torsionModulusMult);
			rod->m_Constraints[i].m_restDarbouxVector = Vector3r(restDarboux[i].x(), restDarboux[i].y(), restDarboux[i].z());
			rod->m_Constraints[i].m_averageSegmentLength = restLengths[i];
		}

		//for (size_t i = 1; i < pointsCount; i++)
		//{
		////	rod->m_Constraints[i].m_stiffnessCoefficientK = Vector3r(bendStiff[i].x() * youngModulusMult, bendStiff[i].y() * youngModulusMult, bendStiff[i].z() * torsionModulusMult);
		//	rod->m_Constraints[i].m_restDarbouxVector = Vector3r(restDarboux[i-1].x(), restDarboux[i-1].y(), restDarboux[i-1].z());
		////	rod->m_Constraints[i].m_averageSegmentLength = restLengths[i];
		//}

		rod->initConstraintBeforeProjection(dt);
	}


	EXPORT_API void  UpdateDirectElasticRodConstraints(DirectPositionBasedSolverForStiffRodsConstraint* rod, int pointsCount, btVector3* positions, btQuaternion* orientations, float* invMasses)
	{
		for (size_t i = 0; i < pointsCount; i++)
		{
			rod->m_Segments[i].m_position = Vector3r(positions[i].x(), positions[i].y(), positions[i].z());
			rod->m_Segments[i].m_rotation = Quaternionr(orientations[i].w(), orientations[i].x(), orientations[i].y(), orientations[i].z());
			rod->m_Segments[i].m_mass = invMasses[i]; //TODO!!! 

		}

		rod->updateConstraint();
	}


	EXPORT_API void  ProjectDirectElasticRodConstraints(DirectPositionBasedSolverForStiffRodsConstraint* rod, int pointsCount, btVector3* positions, btQuaternion* orientations, float* invMasses, btVector3* posCorr, btQuaternion* rotCorr)
	{
		for (size_t i = 0; i < pointsCount; i++)
		{
			//TODO segment position should be (posA + posB) / 2 ???
			rod->m_Segments[i].m_position = Vector3r(positions[i].x(), positions[i].y(), positions[i].z());
			rod->m_Segments[i].m_rotation = Quaternionr(orientations[i].w(), orientations[i].x(), orientations[i].y(), orientations[i].z());
			rod->m_Segments[i].m_mass = invMasses[i]; //TODO!!! 

		}

		rod->updateConstraint();

		//for (size_t cIdx(0); cIdx < pointsCount-1; ++cIdx)
		//{
		//	RodConstraint * constraint(rod->m_rodConstraints[cIdx]);

		//	Vector3r p0 = Vector3r(positions[cIdx].x(), positions[cIdx].y(), positions[cIdx].z());
		//	Vector3r p1 = Vector3r(positions[cIdx+1].x(), positions[cIdx+1].y(), positions[cIdx+1].z());
		//	
		//	Quaternionr q0 =  Quaternionr(orientations[cIdx].w(), orientations[cIdx].x(), orientations[cIdx].y(), orientations[cIdx].z());
		//	Quaternionr q1 = Quaternionr(orientations[cIdx+1].w(), orientations[cIdx+1].x(), orientations[cIdx+1].y(), orientations[cIdx+1].z());

		//	PBD::DirectPositionBasedSolverForStiffRods::update_StretchBendingTwistingConstraint(
		//		p0, q0, p1, q1,
		//		constraint->getConstraintInfo());
		//}


		std::vector<Vector3r> corr_x(pointsCount, Vector3r(0, 0, 0));
		std::vector<Quaternionr> corr_q(pointsCount, Quaternionr(0, 0, 0, 0));



		rod->solvePositionConstraint(corr_x, corr_q);

		for (size_t i = 0; i < pointsCount; i++)
		{

			if (invMasses[i] != 0)
			{
				btVector3 dP(corr_x[i].x(), corr_x[i].y(), corr_x[i].z());
				positions[i] += dP;

				Quaternionr Q(orientations[i].w(), orientations[i].x(), orientations[i].y(), orientations[i].z());
				//btQuaternion dQ = btQuaternion(corr_q[i].x(), corr_q[i].y(), corr_q[i].z(), corr_q[i].w());
				//orientations[i] += dQ; //TODO ???

				Q.coeffs() += corr_q[i].coeffs();
				orientations[i] = btQuaternion(Q.x(), Q.y(), Q.z(), Q.w());
				orientations[i].normalize();
			}

		}

	}


	EXPORT_API void  ProjectDirectElasticRodConstraintsBanded(DirectPositionBasedSolverForStiffRodsConstraint* rod, int pointsCount, btVector3* positions, btQuaternion* orientations, float* invMasses, btVector3* posCorr, btQuaternion* rotCorr)
	{
		for (size_t i = 0; i < pointsCount; i++)
		{
			//TODO segment position should be (posA + posB) / 2 ???
			rod->m_Segments[i].m_position = Vector3r(positions[i].x(), positions[i].y(), positions[i].z());
			rod->m_Segments[i].m_rotation = Quaternionr(orientations[i].w(), orientations[i].x(), orientations[i].y(), orientations[i].z());
			rod->m_Segments[i].m_mass = invMasses[i]; //TODO!!! 

		}

		rod->updateConstraint();


		std::vector<Vector3r> corr_x(pointsCount, Vector3r(0, 0, 0));
		std::vector<Quaternionr> corr_q(pointsCount, Quaternionr(0, 0, 0, 0));



		bool res = rod->solvePositionConstraintBanded(corr_x, corr_q);

		if (res)
		{
			for (size_t i = 0; i < pointsCount; i++)
			{

				if (invMasses[i] != 0)
				{
					btVector3 dP(corr_x[i].x(), corr_x[i].y(), corr_x[i].z());
					positions[i] += dP;

					Quaternionr Q(orientations[i].w(), orientations[i].x(), orientations[i].y(), orientations[i].z());
					//btQuaternion dQ = btQuaternion(corr_q[i].x(), corr_q[i].y(), corr_q[i].z(), corr_q[i].w());
					//orientations[i] += dQ; //TODO ???

					Q.coeffs() += corr_q[i].coeffs();
					orientations[i] = btQuaternion(Q.x(), Q.y(), Q.z(), Q.w());
					orientations[i].normalize();
				}

			}
		}
	}

	EXPORT_API void  UpdateConstraints_DirectElasticRodConstraintsBanded(DirectPositionBasedSolverForStiffRodsConstraint* rod, int pointsCount, btVector3* positions, btQuaternion* orientations, float* invMasses)
	{
#ifdef  _PERF_TIMER
		plf::nanotimer timer;
		timer.start();
#endif //  _PERF_TIMER

		for (size_t i = 0; i < pointsCount; i++)
		{
			//TODO segment position should be (posA + posB) / 2 ???
			rod->m_Segments[i].m_position = Vector3r(positions[i].x(), positions[i].y(), positions[i].z());
			rod->m_Segments[i].m_rotation = Quaternionr(orientations[i].w(), orientations[i].x(), orientations[i].y(), orientations[i].z());
			rod->m_Segments[i].m_mass = invMasses[i]; //TODO!!! 

		}

		rod->updateConstraint();


#ifdef  _PERF_TIMER
		std::cout << "[UPDATE CNSTRS]: " << timer.get_elapsed_ms() << " ms." << std::endl;
#endif

	}


	EXPORT_API void  ComputeJacobians_DirectElasticRodConstraintsBanded(DirectPositionBasedSolverForStiffRodsConstraint* rod, int startId, int pointsCount, btVector3* positions, btQuaternion* orientations, float* invMasses)
	{
#ifdef  _PERF_TIMER
		plf::nanotimer timer;
		timer.start();
#endif //  _PERF_TIMER


		bool res = rod->computeJacobians_PositionConstraintBanded(startId, pointsCount);


#ifdef  _PERF_TIMER
		std::cout << "[JACOBIANS]: " << timer.get_elapsed_ms() << " ms." << std::endl;
#endif

	}


	EXPORT_API void  AssembleJMJT_DirectElasticRodConstraintsBanded(DirectPositionBasedSolverForStiffRodsConstraint* rod, int startId, int count, btVector3* positions, btQuaternion* orientations, float* invMasses)
	{
#ifdef  _PERF_TIMER
		plf::nanotimer timer;
		timer.start();
#endif //  _PERF_TIMER

		bool res = rod->assembleJMJT_PositionConstraintBanded(startId, count);

#ifdef  _PERF_TIMER
		std::cout << "[ASSEMBLE JMJT]: " << timer.get_elapsed_ms() << " ms." << std::endl;
#endif

	}

	EXPORT_API void  ComputeJacobiansAndAssembleJMJT_DirectElasticRodConstraintsBanded(DirectPositionBasedSolverForStiffRodsConstraint* rod, int startId, int count, btVector3* positions, btQuaternion* orientations, float* invMasses)
	{
#ifdef  _PERF_TIMER
		plf::nanotimer timer;
		timer.start();
#endif //  _PERF_TIMER

		bool res = rod->computeJacobiansAndAssembleJMJT_PositionConstraintBanded(startId, count);

#ifdef  _PERF_TIMER
		std::cout << "[ASSEMBLE JMJT]: " << timer.get_elapsed_ms() << " ms." << std::endl;
#endif

	}

	EXPORT_API void  ProjectJMJT_DirectElasticRodConstraintsBanded(DirectPositionBasedSolverForStiffRodsConstraint* rod, int pointsCount, btVector3* positions, btQuaternion* orientations, float* invMasses, btVector3* posCorr, btQuaternion* rotCorr)
	{
#ifdef  _PERF_TIMER
		plf::nanotimer timer;
		timer.start();
#endif //  _PERF_TIMER

		std::vector<Vector3r> corr_x(pointsCount, Vector3r(0, 0, 0));
		std::vector<Quaternionr> corr_q(pointsCount, Quaternionr(0, 0, 0, 0));


#ifdef  _PERF_TIMER
		std::cout << "Allocate corr arrays: " << timer.get_elapsed_ms() << " ms." << std::endl;
		timer.start();
#endif


		bool res = rod->solveJMJT_PositionConstraintBanded(corr_x, corr_q);

#ifdef  _PERF_TIMER
		std::cout << "[SOLVE JMJT] " << timer.get_elapsed_ms() << " ms." << std::endl;
		timer.start();
#endif


		if (res)
		{
			for (size_t i = 0; i < pointsCount; i++)
			{

				if (invMasses[i] != 0)
				{
					btVector3 dP(corr_x[i].x(), corr_x[i].y(), corr_x[i].z());
					positions[i] += dP;

					//Quaternionr Q(orientations[i].w(), orientations[i].x(), orientations[i].y(), orientations[i].z());
					////btQuaternion dQ = btQuaternion(corr_q[i].x(), corr_q[i].y(), corr_q[i].z(), corr_q[i].w());
					////orientations[i] += dQ; //TODO ???

					//Q.coeffs() += corr_q[i].coeffs();
					//orientations[i] = btQuaternion(Q.x(), Q.y(), Q.z(), Q.w());
					//orientations[i].normalize();
				}

			}

			for (size_t i = 0; i < pointsCount; i++)
			{

				if (invMasses[i] != 0)
				{

					Quaternionr Q(orientations[i].w(), orientations[i].x(), orientations[i].y(), orientations[i].z());
					//btQuaternion dQ = btQuaternion(corr_q[i].x(), corr_q[i].y(), corr_q[i].z(), corr_q[i].w());
					//orientations[i] += dQ; //TODO ???

					Q.coeffs() += corr_q[i].coeffs();
					orientations[i] = btQuaternion(Q.x(), Q.y(), Q.z(), Q.w());
					orientations[i].normalize();
				}

			}
		}


#ifdef  _PERF_TIMER
		std::cout << "Apply projections: " << timer.get_elapsed_ms() << " ms." << std::endl;

#endif

	}

	EXPORT_API void DestroyDirectElasticRod(DirectPositionBasedSolverForStiffRodsConstraint* rod)
	{
		delete rod;
	}

}