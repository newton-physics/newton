#pragma once
#include "DefKit.h"
#include "Utils.h"
#include "PositionBasedDynamics/PositionBasedElasticRods.h"
#include "PositionBasedDynamics/DirectPositionBasedSolverForStiffRodsInterface.h"

#define _USE_MATH_DEFINES
#include "math.h"
#include <vector>
#include <list>
#include <memory>

struct Node;
struct Interval;
class SimulationModel;
using Vector6r = Eigen::Matrix<Real, 6, 1>;

//class SimulationModel;


using namespace PBD;

class Constraint
{
public:
	unsigned int m_numberOfBodies;
	/** indices of the linked bodies */
	unsigned int* m_bodies;

	Constraint(const unsigned int numberOfBodies)
	{
		m_numberOfBodies = numberOfBodies;
		m_bodies = new unsigned int[numberOfBodies];
	}

	virtual ~Constraint() { delete[] m_bodies; };
	virtual int& getTypeId() const = 0;

	virtual bool initConstraintBeforeProjection(float dt) { return true; };
	virtual bool updateConstraint() { return true; };
	virtual bool solvePositionConstraint() { return true; };
	virtual bool solveVelocityConstraint() { return true; };
};

class DirectPositionBasedSolverForStiffRodsConstraint : public Constraint
{
	//class RodSegmentImpl : public RodSegment
	//{
	//public:
	//	RodSegmentImpl(SimulationModel &model, unsigned int idx) :
	//		m_model(model), m_segmentIdx(idx) {};

	//	virtual bool isDynamic();
	//	virtual Real Mass();
	//	virtual const Vector3r & InertiaTensor();
	//	virtual const Vector3r & Position();
	//	virtual const Quaternionr & Rotation();

	//	SimulationModel &m_model;
	//	unsigned int m_segmentIdx;
	//};


	class RodSegmentImpl : public RodSegment
	{
	public:
		RodSegmentImpl(unsigned int idx) : m_segmentIdx(idx) {};

		Real m_mass;

		Vector3r m_position;
		Quaternionr m_rotation;
		Vector3r m_inertiaTensor;

		Real m_zeta;

		unsigned int m_segmentIdx;

		virtual bool isDynamic() { return m_mass != 0; };
		virtual Real Mass() { return m_mass; };
		virtual Real GetZeta() { return m_zeta; };
		virtual void SetZeta(Real zeta) { m_zeta = zeta; };
		virtual const Vector3r& InertiaTensor() { return m_inertiaTensor; };
		virtual const Vector3r& Position() { return m_position; };
		virtual const Quaternionr& Rotation() { return m_rotation; };

	};

	class RodConstraintImpl : public RodConstraint
	{
	public:
		std::vector<unsigned int> m_segments;
		Eigen::Matrix<Real, 3, 4> m_constraintInfo;

		Real m_averageRadius;
		Real m_averageSegmentLength;

		Vector3r m_restDarbouxVector;
		Vector3r m_stiffnessCoefficientK;
		Vector3r m_stretchCompliance;
		Vector3r m_bendingAndTorsionCompliance;
		Vector3r m_thetaVector;

		virtual unsigned int segmentIndex(unsigned int i) {
			if (i < static_cast<unsigned int>(m_segments.size()))
				return m_segments[i];
			return 0u;
		};

		virtual Eigen::Matrix<Real, 3, 4>& getConstraintInfo() { return m_constraintInfo; }
		virtual Real getAverageSegmentLength() { return m_averageSegmentLength; }

		virtual Vector3r& getRestDarbouxVector() { return m_restDarbouxVector; }
		virtual Vector3r& getStiffnessCoefficientK() { return m_stiffnessCoefficientK; };
		virtual Vector3r& getThetaVector() { return m_thetaVector; }
		virtual Vector3r& getStretchCompliance() { return m_stretchCompliance; }
		virtual Vector3r& getBendingAndTorsionCompliance() { return m_bendingAndTorsionCompliance; }
	};

public:
	static int TYPE_ID;

	DirectPositionBasedSolverForStiffRodsConstraint() : Constraint(2),
		root(NULL), numberOfIntervals(0), numberOfUpdates(0), intervals(NULL), forward(NULL), backward(NULL) {}
	~DirectPositionBasedSolverForStiffRodsConstraint();

	virtual int& getTypeId() const { return TYPE_ID; }

	bool initConstraint(
		//	const std::vector<std::pair<unsigned int, unsigned int>> & constraintSegmentIndices,
		const std::vector<Vector3r>& segmentPositions,
		const std::vector<Quaternionr>& segmentRotations,
		const std::vector<Vector3r>& constraintPositions,
		const std::vector<Real>& averageRadii,
		const std::vector<Real>& averageSegmentLengths,
		const std::vector<Real>& youngsModuli,
		const std::vector<Real>& torsionModuli);

	virtual bool initConstraintBeforeProjection(float dt);
	virtual bool updateConstraint();
	virtual bool solvePositionConstraint(std::vector<Vector3r>& m_corr_x, std::vector<Quaternionr>& m_corr_q);
	virtual bool solvePositionConstraintBanded(std::vector<Vector3r>& m_corr_x, std::vector<Quaternionr>& m_corr_q);
	virtual bool computeJacobians_PositionConstraintBanded(int startId, int endId);
	virtual bool assembleJMJT_PositionConstraintBanded(int startId, int endId);
	virtual bool computeJacobiansAndAssembleJMJT_PositionConstraintBanded(int startId, int endId);
	virtual bool solveJMJT_PositionConstraintBanded(std::vector<Vector3r>& m_corr_x, std::vector<Quaternionr>& m_corr_q);

	std::vector<RodConstraintImpl> m_Constraints;
	std::vector<RodConstraint*> m_rodConstraints;

	std::vector<BandedNode*> m_bandedSolverNodesForward;
	std::vector<BandedNode*> m_bandedSolverNodesBackward;



	std::vector<RodSegmentImpl> m_Segments;
	std::vector<RodSegment*> m_rodSegments;

	MatrixXr m_JMJT;
	VectorXr m_RHS;
	MatrixXr m_JMJT_Band;



protected:

	/** root node */
	PBD::Node* root;
	/** intervals of constraints */
	PBD::Interval* intervals;
	/** number of intervals */
	int numberOfIntervals;

	int numberOfUpdates;
	/** list to process nodes with increasing row index in the system matrix H (from the leaves to the root) */
	std::list <PBD::Node*>* forward;
	/** list to process nodes starting with the highest row index to row index zero in the matrix H (from the root to the leaves) */
	std::list <PBD::Node*>* backward;



	std::vector<Vector6r> m_rightHandSide;
	std::vector<Vector6r> m_lambdaSums;
	std::vector<std::vector<Matrix3r>> m_bendingAndTorsionJacobians;
	std::vector<Vector3r> m_corr_x;
	std::vector<Quaternionr> m_corr_q;

	void deleteNodes();
};
