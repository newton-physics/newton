#ifndef POSITION_BASED_ELASTIC_RODS_H
#define POSITION_BASED_ELASTIC_RODS_H

#include <Eigen/Dense>
#include "Common/Common.h"
#include "DirectPositionBasedSolverForStiffRodsInterface.h"

#include <vector>
#include <list>

// ------------------------------------------------------------------------------------
namespace PBD
{
	// Implementation of "Direct Position-Based Solver for Stiff Rods" paper
	// (https://animation.rwth-aachen.de/publication/0557/)
	//
	//	Implemented by:
	//
	//	Crispin Deul
	//	Graduate School CE
	//	Technische Universität Darmstadt
	//
	//  deul[at] gsc.tu-darmstadt.de
	//

	using Matrix6r = Eigen::Matrix<Real, 6, 6, Eigen::DontAlign>;
	using Vector6r = Eigen::Matrix<Real, 6, 1, Eigen::DontAlign>;
	/** Node in the simulated tree structure */
	struct Node {
		Node() {
			object = NULL; D = Dinv = J = Matrix6r::Zero(); parent = NULL;
			soln.setZero(); index = 0;
		};
		bool isconstraint;
		void *object;
		Matrix6r D, Dinv, J;
		std::vector <Node*> children;
		Node *parent;
		Vector6r soln;
		int index;
		Eigen::LDLT<Matrix6r> DLDLT;
	};

	struct Interval
	{
		int start;
		int end;
	};

	class DirectPositionBasedSolverForStiffRods
	{
	private:

		static void initLists(
			int numberOfIntervals,
			std::list <Node*> * &forward,
			std::list <Node*> * &backward,
			Node* &root);

		/** Returns, whether the passed segment is connected to a constraint in the
		* passed index range of the entire constraints.
		*/
		static bool isSegmentInInterval(
			RodSegment *segment,
			int intervalIndex,
			Interval* intervals,
			std::vector<RodConstraint*> &rodConstraints,
			std::vector<RodSegment*> &rodSegments);

		/** Returns, whether the passed constraint is within the passed index range
		* of the entire constraints.
		*/
		static bool isConstraintInInterval(
			RodConstraint *constraint,
			int intervalIndex,
			Interval* intervals,
			std::vector<RodConstraint*> &rodConstraints);


		/** This method recursively visits all segment nodes and inserts constraint nodes between them.
		*/
		static void initSegmentNode(
			Node *n,
			int intervalIndex,
			std::vector<RodConstraint*> &rodConstraints,
			std::vector<RodSegment*> &rodSegments,
			std::vector <RodConstraint*> &markedConstraints,
			Interval* intervals);


		/** Sorts matrix H.
		*/
		static void orderMatrix(
			Node *n,
			int intervalIndex,
			std::list <Node*> * forward,
			std::list <Node*> * backward);

		/** Initializes the nodes.
		* The first static node is selected as the root of the tree.
		* Then, starting from this node, all edges (joints) are followed
		* and the children and the parent node are saved.
		*/
		static void initNodes(
			int intervalIndex,
			std::vector<RodSegment*> &rodSegments,
			Node* &root,
			Interval* intervals,
			std::vector<RodConstraint*> &rodConstraints,
			std::list <Node*> * forward,
			std::list <Node*> * backward,
			std::vector <RodConstraint*> &markedConstraints);


		static void initTree(
			std::vector<RodConstraint*> &rodConstraints,
			std::vector<RodSegment*> & rodSegments,
			Interval* &intervals,
			int &numberOfIntervals,
			std::list <Node*> * &forward,
			std::list <Node*> * &backward,
			Node* &root
			);


		/** Compute the discrete Darboux vector based on Equation (7)
		*/
		static bool computeDarbouxVector(
			const Quaternionr & q0,
			const Quaternionr & q1,
			const Real averageSegmentLength,
			Vector3r & darbouxVector
			);

		static bool computeBendingAndTorsionJacobians(
			const Quaternionr & q0,
			const Quaternionr & q1,
			const Real averageSegmentLength,
			Eigen::Matrix<Real, 3, 4> & jOmega0,
			Eigen::Matrix<Real, 3, 4> & jOmega1
			);

		static bool computeMatrixG(
			const Quaternionr & q,
			Eigen::Matrix<Real, 4, 3> & G
			);

		static void computeMatrixK(
			const Vector3r &connector,
			const Real invMass,
			const Vector3r &x,
			const Matrix3r &inertiaInverseW,
			Matrix3r &K);


		/** Returns the 6x6 mass matrix M of a segment
		*/
		static void getMassMatrix(RodSegment *segment, Matrix6r &M);

		/** Factorizes matrix H and computes the right hand side vector -b.
		*/
		static Real factor(
			const int intervalIndex,
			const std::vector<RodConstraint*> &rodConstraints,
			std::vector<RodSegment*> & rodSegments,
			const Interval* &intervals,
			std::list <Node*> * forward,
			std::list <Node*> * backward,
			std::vector<Vector6r> & RHS,
			std::vector<Vector6r> & lambdaSums,
			std::vector<std::vector<Matrix3r>> & bendingAndTorsionJacobians
			);


		/** Solves the system of equations with the factorized matrix H.
		*/
		static bool solve(
			int intervalIndex,
			std::list <Node*> * forward,
			std::list <Node*> * backward,
			std::vector<Vector6r> & RHS,
			std::vector<Vector6r> & lambdaSums,
			std::vector<Vector3r> & corr_x,
			std::vector<Quaternionr> & corr_q
			);
	public:

		///** Initialize the zero-stretch, bending, and torsion constraints of the rod.
		//* Computes constraint connectors in segment space, computes the diagonal stiffness matrices
		//* and the Darboux vectors of the initial state. Initializes the forward and backward lists
		//* of nodes  for the direct solver\n\n
		//*
		//* @param rodConstraints contains the combined zero-stretch, bending
		//* and torsion constraints of the rod. The set of constraints must by acyclic.
		//* @param rodSegments contains the segments of the rod
		//* @param forward list of nodes in the acyclic tree of rod segments and zero-stretch,
		//* bending and torsion constraints so that parent nodes occur later in the list than their children
		//* @param backward list of nodes in the acyclic tree of rod segments and zero-stretch,
		//* bending and torsion constraints. The reverse of the forward list.
		//* @param constraintPositions positions of the rod's constraints in world coordinates
		//* @param averageRadii the average radii at the constraint positions of the rod. Value in Meters (m)
		//* @param averageSegmentLengths vector of the average lengths of the two rod segments
		//* connected by a constraint of the rod. Value in Meters (m)
		//* @param youngsModuli vector of the Young's modulus of every constraint of the rod.
		//* The Young's modulus of the rod material measures
		//* stiffness against bending. Value in Pascal (Pa)
		//* @param torsionModuli vector of the torsion modulus of every constraint of the rod.
		//* The torsion modulus (also called shear modulus)
		//* of the rod material measures stiffness against torsion. Value in Pascal (Pa)
		//* @param RHS vector with entries for each constraint. In concatenation these entries represent the right hand side of the system of equations to be solved. (eq. 22 in the paper)
		//* @param lambdaSums contains entries of the sum of all lambda updates for
		//* each constraint in the rod during one time step which is needed by the solver to handle
		//* compliance in the correct way (cf. the right hand side of eq. 22 in the paper).
		//* @param bendingAndTorsionJacobians this vector is used to temporary
		//* save the Jacobians of the bending and torsion part of each constraint
		//* during the solution process of the system of equations. Allocating this
		//* vector outside of the solve-method avoids repeated reallocation between iterations of the solver		
		//* @param corr_x vector of position corrections for every segment of the rod (part of delta-x in eq. 22 in the paper)
		//* @param corr_q vector of rotation corrections for every segment of the rod (part of delta-x in eq. 22 in the paper)
		//*/
		static bool init_DirectPositionBasedSolverForStiffRodsConstraint(
			std::vector<RodConstraint*> &rodConstraints,
			std::vector<RodSegment*> & rodSegments,
			Interval* &intervals,
			int &numberOfIntervals,
			std::list <Node*> * &forward,
			std::list <Node*> * &backward,
			Node* &root,
			const std::vector<Vector3r> &constraintPositions,
			const std::vector<Real> &averageRadii,
			const std::vector<Real> &youngsModuli,
			const std::vector<Real> &torsionModuli,
			std::vector<Vector6r> & RHS,
			std::vector<Vector6r> & lambdaSums,
			std::vector<std::vector<Matrix3r>> & bendingAndTorsionJacobians,
			std::vector<Vector3r> & corr_x,
			std::vector<Quaternionr> & corr_q
			);

		///** Update the constraint info data. \n\n
		//*
		//* @param rodConstraints contains the combined zero-stretch, bending and torsion constraints of the rod.
		//* @param rodSegments contains the segments of the rod
		//*/
		static bool update_DirectPositionBasedSolverForStiffRodsConstraint(
			const std::vector<RodConstraint*> &rodConstraints,
			const std::vector<RodSegment*> & rodSegments
			);

		///** Initialize the constraint before the projection iterations in each time step. \n\n
		//*
		//* @param rodConstraints contains the combined zero-stretch, bending and torsion constraints of the rod.
		//* @param inverseTimeStepSize inverse of the current time step size used to compute compliance (see computation of alpha-tilde in eq. 17)		
		//* @param lambdaSums contains entries of the sum of all lambda updates for
		//* each constraint in the rod during one time step which is needed by the solver to handle
		//* compliance in the correct way (cf. the right hand side of eq. 22 in the paper).
		//*/
		static bool initBeforeProjection_DirectPositionBasedSolverForStiffRodsConstraint(
			const std::vector<RodConstraint*> &rodConstraints,
			const Real inverseTimeStepSize,
			std::vector<Vector6r> & lambdaSums
			);

		///** Determine the position and orientation corrections for all combined zero-stretch, bending and twisting constraints of the rod (eq. 22 in the paper). \n\n
		//*
		//* @param rodConstraints contains the combined zero-stretch, bending and torsion constraints of the rod. The set of constraints must by acyclic.
		//* @param rodSegments contains the segments of the rod
		//* @param forward list of nodes in the acyclic tree of rod segments and zero-stretch, bending and torsion constraints so that parent nodes occur later in the list than their children
		//* @param backward list of nodes in the acyclic tree of rod segments and zero-stretch, bending and torsion constraints. The reverse of the forward list.
		//* @param RHS vector with entries for each constraint. In concatenation these entries represent the right hand side of the system of equations to be solved. (eq. 22 in the paper)
		//* @param lambdaSums contains entries of the sum of all lambda updates for
		//* each constraint in the rod during one time step which is needed by the solver to handle
		//* compliance in the correct way.
		//* @param bendingAndTorsionJacobians this vector is used to temporary
		//* save the Jacobians of the bending and torsion part of each constraint
		//* during the solution process of the system of equations. Allocating this
		//* vector outside of the solve-method avoids repeated reallocation between iterations of the solver		
		//* @param corr_x vector of position corrections for every segment of the rod (part of delta-x in eq. 22 in the paper)
		//* @param corr_q vector of rotation corrections for every segment of the rod (part of delta-x in eq. 22 in the paper)
		//*/
		static bool solve_DirectPositionBasedSolverForStiffRodsConstraint(
			const std::vector<RodConstraint*> &rodConstraints,
			std::vector<RodSegment*> & rodSegments,
			const Interval* intervals,
			const int &numberOfIntervals,
			std::list <Node*> * forward,
			std::list <Node*> * backward,
			std::vector<Vector6r> & RHS,
			std::vector<Vector6r> & lambdaSums,
			std::vector<std::vector<Matrix3r>> & bendingAndTorsionJacobians,
			std::vector<Vector3r> & corr_x,
			std::vector<Quaternionr> & corr_q
			);

		///** Initialize the zero-stretch, bending, and torsion constraint.
		//* Computes constraint connectors in segment space, computes the diagonal stiffness matrix
		//* and the Darboux vector of the initial state. \n\n
		//*
		//* @param x0 center of mass of body 0
		//* @param q0 rotation of body 0
		//* @param x1 center of mass of body 1
		//* @param q1 rotation of body 1
		//* @param constraintPosition position of the constraint in world coordinates
		//* @param averageRadius the average radius of the two rod segments connected by the constraint. Value in Meters (m)
		//* @param averageSegmentLength the average length of the two rod segments connected by the constraint. Value in Meters (m)
		//* @param youngsModulus the Young's modulus of the rod material measures stiffness against bending. Value in Pascal (Pa)
		//* @param torsionModulus the torsion modulus (also called shear modulus) of the rod material measures stiffness against torsion. Value in Pascal (Pa)
		//* @param jointInfo joint information which is required by the solver.This
		//* information is generated in this method
		//* and updated each time the bodies change their state by update_StretchBendingTwistingConstraint().
		//* @param stiffnessCoefficientK diagonal matrix with material parameters for bending and torsion stiffness (eq. 5 in the paper)
		//* @param restDarbouxVector the rest Darboux vector computed in this method with the initial constraint configuration
		//*/
		static bool init_StretchBendingTwistingConstraint(
			const Vector3r &x0,
			const Quaternionr &q0,
			const Vector3r &x1,
			const Quaternionr &q1,
			const Vector3r &constraintPosition,
			const Real averageRadius,
			const Real averageSegmentLength,
			const Real youngsModulus,
			const Real torsionModulus,
			Eigen::Matrix<Real, 3, 4, Eigen::DontAlign> &jointInfo,
			Vector3r &stiffnessCoefficientK,
			Vector3r &restDarbouxVector
			);

		///** Initialize the constraint before the projection iterations in each time step. \n\n
		//*
		//* @param stiffnessCoefficientK diagonal matrix with material parameters for bending and torsion stiffness (eq. 5 in the paper)
		//* @param inverseTimeStepSize inverse of the current time step size used to compute compliance (see computation of alpha-tilde in eq. 17)
		//* @param bendingAndTorsionCompliance the compliance of the bending and torsion constraint part (eq. 24 in the paper)
		//* @param stretchCompliance the compliance of the stretch constraint part (eq. 24 in the paper)
		//* @param lambdaSum the sum of all lambda updates of
		//* this constraint during one time step which is needed by the solver to handle
		//* compliance in the correct way. Is set to zero. (see eq. 19 in the paper)
		//*/
		static bool initBeforeProjection_StretchBendingTwistingConstraint(
			const Vector3r &stiffnessCoefficientK,
			const Real inverseTimeStepSize,
			const Real averageSegmentLength,
			Vector3r &stretchCompliance,
			Vector3r &bendingAndTorsionCompliance,
			Vector6r &lambdaSum
			);

		///** Update the joint info data. \n\n
		//*
		//* @param x0 center of mass of body 0
		//* @param q0 rotation of body 0
		//* @param x1 center of mass of body 1
		//* @param q1 rotation of body 1
		//* @param jointInfo joint information which is required by the solver.This
		//* information is updated by calling this method.
		//*/
		static bool update_StretchBendingTwistingConstraint(
			const Vector3r &x0,
			const Quaternionr &q0,
			const Vector3r &x1,
			const Quaternionr &q1,
			Eigen::Matrix<Real, 3, 4, Eigen::DontAlign> &jointInfo
		);

		///** Determine the position and orientation corrections for the combined zero-stretch, bending and twisting constraint (eq. 23 in the paper). \n\n
		//*
		//* @param invMass0 inverse mass of first body; inverse mass is zero if body is static
		//* @param x0 center of mass of body 0
		//* @param inertiaInverseW0 inverse inertia tensor (world space) of body 0
		//* @param q0 rotation of body 0
		//* @param invMass1 inverse mass of second body; inverse mass is zero if body is static
		//* @param x1 center of mass of body 1
		//* @param inertiaInverseW1 inverse inertia tensor (world space) of body 1
		//* @param q1 rotation of body 1
		//* @param restDarbouxVector the rest Darboux vector of the initial constraint configuration
		//* @param averageSegmentLength the average length of the two rod segments connected by the constraint
		//* @param stretchCompliance the compliance of the stretch constraint part (eq. 24 in the paper)
		//* @param bendingAndTorsionCompliance the compliance of the bending and torsion constraint part (eq. 24 in the paper)
		//* @param jointInfo joint information which is required by the solver.This
		//* information must be generated in the beginning by calling init_StretchBendingTwistingConstraint()
		//* and updated each time the bodies change their state by update_StretchBendingTwistingConstraint().
		//* @param corr_x0 position correction of center of mass of first body
		//* @param corr_q0 rotation correction of first body
		//* @param corr_x1 position correction of center of mass of second body
		//* @param corr_q1 rotation correction of second body
		//* @param lambdaSum the sum of all lambda updates of
		//* this constraint during one time step which is needed by the solver to handle
		//* compliance in the correct way. Must be set to zero before the position
		//* projection iterations start at each time step by calling
		//* initBeforeProjection_StretchBendingTwistingConstraint(). (see eq. 19 in the paper)
		//*/
		static bool solve_StretchBendingTwistingConstraint(
			const Real invMass0,
			const Vector3r &x0,
			const Matrix3r &inertiaInverseW0,
			const Quaternionr &q0,
			const Real invMass1,
			const Vector3r &x1,
			const Matrix3r &inertiaInverseW1,
			const Quaternionr &q1,
			const Vector3r &restDarbouxVector,
			const Real averageSegmentLength,
			const Vector3r &stretchCompliance,
			const Vector3r &bendingAndTorsionCompliance,
			const Eigen::Matrix<Real, 3, 4, Eigen::DontAlign> &jointInfo,
			Vector3r &corr_x0, Quaternionr &corr_q0,
			Vector3r &corr_x1, Quaternionr &corr_q1,
			Vector6r &lambdaSum
		);
	};


	// Implementation of "Position And Orientation Based Cosserat Rods" paper
	// (https://animation.rwth-aachen.de/publication/0550/)
	//
	//	Implemented by:
	//
	//	Tassilo Kugelstadt
	//	Computer Animation Group
	//	RWTH Aachen University
	//
	//  kugelstadt[at] cs.rwth-aachen.de
	//
	class PositionBasedCosseratRods
	{
	public:
		/** Determine the position and orientation corrections for the stretch and shear constraint constraint (eq. 37 in the paper). \n\n
		*
		* @param p0 position of first particle
		* @param invMass0 inverse mass of first particle
		* @param p1 position of second particle
		* @param invMass1 inverse mass of second particle
		* @param q0 Quaternionr at the center of the edge
		* @param invMassq0 inverse mass of the quaternion
		* @param stretchingAndShearingKs stiffness coefficients for stretching and shearing
		* @param restLength rest edge length
		* @param corr0 position correction of first particle
		* @param corr1 position correction of second particle
		* @param corrq0 orientation correction of quaternion
		*/
		static bool solve_StretchShearConstraint(
			const Vector3r& p0, Real invMass0,
			const Vector3r& p1, Real invMass1,
			const Quaternionr& q0, Real invMassq0,
			const Vector3r& stretchingAndShearingKs,
			const Real restLength,
			Vector3r& corr0, Vector3r&  corr1, Quaternionr&  corrq0);

		/** Determine the position corrections for the bending and torsion constraint constraint (eq. 40 in the paper). \n\n
		*
		* @param q0 first quaternion
		* @param invMassq0 inverse mass of the first quaternion
		* @param q1 second quaternion
		* @param invMassq1 inverse Mass of the second quaternion
		* @param bendingAndTwistingKs stiffness coefficients for stretching and shearing
		* @param restDarbouxVector rest Darboux vector
		* @param corrq0 position correction of first particle
		* @param corrq1 position correction of second particle
		*/
		static bool solve_BendTwistConstraint(
			const Quaternionr& q0, Real invMassq0,
			const Quaternionr& q1, Real invMassq1,
			const Vector3r& bendingAndTwistingKs,
			const Quaternionr& restDarbouxVector,
			Quaternionr& corrq0, Quaternionr&  corrq1);
	};

	
}

#endif
