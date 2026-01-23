#include <iostream>
//#include <DefKitAdv.h>

//#include "..\VMProtectSDK.h"
#include <mkl.h>


#include <Eigen\Core>
#include <Eigen\src\Core\BandMatrix.h>
#ifdef  VMPROTECT
#include "..\VMProtectSDK.h"
#endif
#include "PositionBasedElasticRods.h"
#include "MathFunctions.h"


#define _USE_MATH_DEFINES

#include "math.h"

//#include <complex>
//#define lapack_complex_float std::complex<float>
//#define lapack_complex_double std::complex<double>
//#include <lapack.h>

using namespace PBD;

const Real eps = static_cast<Real>(1e-6);

const int permutation[3][3] = {
	0, 2, 1,
	1, 0, 2,
	2, 1, 0
};

// ----------------------------------------------------------------------------------------------
bool PositionBasedCosseratRods::solve_StretchShearConstraint(
	const Vector3r& p0, Real invMass0,
	const Vector3r& p1, Real invMass1,
	const Quaternionr& q0, Real invMassq0,
	const Vector3r& stretchingAndShearingKs,
	const Real restLength,
	Vector3r& corr0, Vector3r& corr1, Quaternionr& corrq0)
{
	Vector3r d3;	//third director d3 = q0 * e_3 * q0_conjugate
	d3[0] = static_cast<Real>(2.0) * (q0.x() * q0.z() + q0.w() * q0.y());
	d3[1] = static_cast<Real>(2.0) * (q0.y() * q0.z() - q0.w() * q0.x());
	d3[2] = q0.w() * q0.w() - q0.x() * q0.x() - q0.y() * q0.y() + q0.z() * q0.z();

	Vector3r gamma = (p1 - p0) / restLength - d3;
	gamma /= (invMass1 + invMass0) / restLength + invMassq0 * static_cast<Real>(4.0) * restLength + eps;

	if (std::abs(stretchingAndShearingKs[0] - stretchingAndShearingKs[1]) < eps && std::abs(stretchingAndShearingKs[0] - stretchingAndShearingKs[2]) < eps)	//all Ks are approx. equal
		for (int i = 0; i < 3; i++) gamma[i] *= stretchingAndShearingKs[i];
	else	//diffenent stretching and shearing Ks. Transform diag(Ks[0], Ks[1], Ks[2]) into world space using Ks_w = R(q0) * diag(Ks[0], Ks[1], Ks[2]) * R^T(q0) and multiply it with gamma
	{
		Matrix3r R = q0.toRotationMatrix();
		gamma = (R.transpose() * gamma).eval();
		for (int i = 0; i < 3; i++) gamma[i] *= stretchingAndShearingKs[i];
		gamma = (R * gamma).eval();
	}

	corr0 = invMass0 * gamma;
	corr1 = -invMass1 * gamma;

	Quaternionr q_e_3_bar(q0.z(), -q0.y(), q0.x(), -q0.w());	//compute q*e_3.conjugate (cheaper than quaternion product)
	corrq0 = Quaternionr(0.0, gamma.x(), gamma.y(), gamma.z()) * q_e_3_bar;
	corrq0.coeffs() *= static_cast<Real>(2.0) * invMassq0 * restLength;

	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedCosseratRods::solve_BendTwistConstraint(
	const Quaternionr& q0, Real invMassq0,
	const Quaternionr& q1, Real invMassq1,
	const Vector3r& bendingAndTwistingKs,
	const Quaternionr& restDarbouxVector,
	Quaternionr& corrq0, Quaternionr& corrq1)
{
	Quaternionr omega = q0.conjugate() * q1;   //darboux vector

	Quaternionr omega_plus;
	omega_plus.coeffs() = omega.coeffs() + restDarbouxVector.coeffs();     //delta Omega with -Omega_0
	omega.coeffs() = omega.coeffs() - restDarbouxVector.coeffs();                 //delta Omega with + omega_0
	if (omega.squaredNorm() > omega_plus.squaredNorm()) omega = omega_plus;

	for (int i = 0; i < 3; i++) omega.coeffs()[i] *= bendingAndTwistingKs[i] / (invMassq0 + invMassq1 + static_cast<Real>(1.0e-6));
	omega.w() = 0.0;    //discrete Darboux vector does not have vanishing scalar part

	corrq0 = q1 * omega;
	corrq1 = q0 * omega;
	corrq0.coeffs() *= invMassq0;
	corrq1.coeffs() *= -invMassq1;
	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedElasticRods::solve_PerpendiculaBisectorConstraint(
	const Vector3r& p0, Real invMass0,
	const Vector3r& p1, Real invMass1,
	const Vector3r& p2, Real invMass2,
	const Real stiffness,
	Vector3r& corr0, Vector3r& corr1, Vector3r& corr2)
{
	const Vector3r pm = 0.5 * (p0 + p1);
	const Vector3r p0p2 = p0 - p2;
	const Vector3r p2p1 = p2 - p1;
	const Vector3r p1p0 = p1 - p0;
	const Vector3r p2pm = p2 - pm;

	Real wSum = invMass0 * p0p2.squaredNorm() + invMass1 * p2p1.squaredNorm() + invMass2 * p1p0.squaredNorm();
	if (wSum < eps)
		return false;

	const Real lambda = stiffness * p2pm.dot(p1p0) / wSum;

	corr0 = -invMass0 * lambda * p0p2;
	corr1 = -invMass1 * lambda * p2p1;
	corr2 = -invMass2 * lambda * p1p0;

	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedElasticRods::solve_GhostPointEdgeDistanceConstraint(
	const Vector3r& p0, Real invMass0,
	const Vector3r& p1, Real invMass1,
	const Vector3r& p2, Real invMass2,
	const Real stiffness,
	const Real ghostEdgeRestLength,
	Vector3r& corr0, Vector3r& corr1, Vector3r& corr2)
{
	// Ghost-Edge constraint
	Vector3r pm = 0.5 * (p0 + p1);
	Vector3r p2pm = p2 - pm;
	Real wSum = static_cast<Real>(0.25) * invMass0 + static_cast<Real>(0.25) * invMass1 + static_cast<Real>(1.0) * invMass2;

	if (wSum < eps)
		return false;

	Real p2pm_mag = p2pm.norm();
	p2pm *= static_cast<Real>(1.0) / p2pm_mag;

	const Real lambda = stiffness * (p2pm_mag - ghostEdgeRestLength) / wSum;

	corr0 = 0.5 * invMass0 * lambda * p2pm;
	corr1 = 0.5 * invMass1 * lambda * p2pm;
	corr2 = -1.0 * invMass2 * lambda * p2pm;

	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedElasticRods::solve_DarbouxVectorConstraint(
	const Vector3r& p0, Real invMass0,
	const Vector3r& p1, Real invMass1,
	const Vector3r& p2, Real invMass2,
	const Vector3r& p3, Real invMass3,
	const Vector3r& p4, Real invMass4,
	const Vector3r& bendingAndTwistingKs,
	const Real midEdgeLength,
	const Vector3r& restDarbouxVector,
	Vector3r& corr0, Vector3r& corr1, Vector3r& corr2, Vector3r& corr3, Vector3r& corr4)
{
	//  Single rod element:
	//      3   4		//ghost points
	//		|	|
	//  --0---1---2--	// rod points

	Vector3r darboux_vector;
	Matrix3r d0, d1;

	PositionBasedElasticRods::computeMaterialFrame(p0, p1, p3, d0);
	PositionBasedElasticRods::computeMaterialFrame(p1, p2, p4, d1);

	PositionBasedElasticRods::computeDarbouxVector(d0, d1, midEdgeLength, darboux_vector);

	Matrix3r dajpi[3][3];
	computeMaterialFrameDerivative(p0, p1, p3, d0,
		dajpi[0][0], dajpi[0][1], dajpi[0][2],
		dajpi[1][0], dajpi[1][1], dajpi[1][2],
		dajpi[2][0], dajpi[2][1], dajpi[2][2]);

	Matrix3r dbjpi[3][3];
	computeMaterialFrameDerivative(p1, p2, p4, d1,
		dbjpi[0][0], dbjpi[0][1], dbjpi[0][2],
		dbjpi[1][0], dbjpi[1][1], dbjpi[1][2],
		dbjpi[2][0], dbjpi[2][1], dbjpi[2][2]);

	Matrix3r constraint_jacobian[5];
	computeDarbouxGradient(
		darboux_vector, midEdgeLength, d0, d1,
		dajpi, dbjpi,
		//bendingAndTwistingKs,
		constraint_jacobian[0],
		constraint_jacobian[1],
		constraint_jacobian[2],
		constraint_jacobian[3],
		constraint_jacobian[4]);

	const Vector3r constraint_value(bendingAndTwistingKs[0] * (darboux_vector[0] - restDarbouxVector[0]),
		bendingAndTwistingKs[1] * (darboux_vector[1] - restDarbouxVector[1]),
		bendingAndTwistingKs[2] * (darboux_vector[2] - restDarbouxVector[2]));

	Matrix3r factor_matrix;
	factor_matrix.setZero();

	Matrix3r tmp_mat;
	Real invMasses[]{ invMass0, invMass1, invMass2, invMass3, invMass4 };
	for (int i = 0; i < 5; ++i)
	{
		tmp_mat = constraint_jacobian[i].transpose() * constraint_jacobian[i];
		tmp_mat.col(0) *= invMasses[i];
		tmp_mat.col(1) *= invMasses[i];
		tmp_mat.col(2) *= invMasses[i];

		factor_matrix += tmp_mat;
	}

	Vector3r dp[5];
	tmp_mat = factor_matrix.inverse();

	for (int i = 0; i < 5; ++i)
	{
		constraint_jacobian[i].col(0) *= invMasses[i];
		constraint_jacobian[i].col(1) *= invMasses[i];
		constraint_jacobian[i].col(2) *= invMasses[i];
		dp[i] = -(constraint_jacobian[i]) * (tmp_mat * constraint_value);
	}

	corr0 = dp[0];
	corr1 = dp[1];
	corr2 = dp[2];
	corr3 = dp[3];
	corr4 = dp[4];

	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedElasticRods::computeMaterialFrame(
	const Vector3r& p0,
	const Vector3r& p1,
	const Vector3r& p2,
	Matrix3r& frame)
{
	frame.col(2) = (p1 - p0);
	frame.col(2).normalize();

	frame.col(1) = (frame.col(2).cross(p2 - p0));
	frame.col(1).normalize();

	frame.col(0) = frame.col(1).cross(frame.col(2));
	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedElasticRods::computeDarbouxVector(const Matrix3r& dA, const Matrix3r& dB, const Real mid_edge_length, Vector3r& darboux_vector)
{
	Real factor = static_cast<Real>(1.0) + dA.col(0).dot(dB.col(0)) + dA.col(1).dot(dB.col(1)) + dA.col(2).dot(dB.col(2));

	factor = static_cast<Real>(2.0) / (mid_edge_length * factor);

	for (int c = 0; c < 3; ++c)
	{
		const int i = permutation[c][0];
		const int j = permutation[c][1];
		const int k = permutation[c][2];
		darboux_vector[i] = dA.col(j).dot(dB.col(k)) - dA.col(k).dot(dB.col(j));
	}
	darboux_vector *= factor;
	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedElasticRods::computeMaterialFrameDerivative(
	const Vector3r& p0, const Vector3r& p1, const Vector3r& p2, const Matrix3r& d,
	Matrix3r& d1p0, Matrix3r& d1p1, Matrix3r& d1p2,
	Matrix3r& d2p0, Matrix3r& d2p1, Matrix3r& d2p2,
	Matrix3r& d3p0, Matrix3r& d3p1, Matrix3r& d3p2)
{
	//////////////////////////////////////////////////////////////////////////
	// d3pi
	//////////////////////////////////////////////////////////////////////////
	const Vector3r p01 = p1 - p0;
	Real length_p01 = p01.norm();

	d3p0.col(0) = d.col(2)[0] * d.col(2);
	d3p0.col(1) = d.col(2)[1] * d.col(2);
	d3p0.col(2) = d.col(2)[2] * d.col(2);

	d3p0.col(0)[0] -= 1.0;
	d3p0.col(1)[1] -= 1.0;
	d3p0.col(2)[2] -= 1.0;

	d3p0.col(0) *= (static_cast<Real>(1.0) / length_p01);
	d3p0.col(1) *= (static_cast<Real>(1.0) / length_p01);
	d3p0.col(2) *= (static_cast<Real>(1.0) / length_p01);

	d3p1.col(0) = -d3p0.col(0);
	d3p1.col(1) = -d3p0.col(1);
	d3p1.col(2) = -d3p0.col(2);

	d3p2.col(0).setZero();
	d3p2.col(1).setZero();
	d3p2.col(2).setZero();

	//////////////////////////////////////////////////////////////////////////
	// d2pi
	//////////////////////////////////////////////////////////////////////////
	const Vector3r p02 = p2 - p0;
	const Vector3r p01_cross_p02 = p01.cross(p02);

	const Real length_cross = p01_cross_p02.norm();

	Matrix3r mat;
	mat.col(0) = d.col(1)[0] * d.col(1);
	mat.col(1) = d.col(1)[1] * d.col(1);
	mat.col(2) = d.col(1)[2] * d.col(1);

	mat.col(0)[0] -= 1.0;
	mat.col(1)[1] -= 1.0;
	mat.col(2)[2] -= 1.0;

	mat.col(0) *= (-static_cast<Real>(1.0) / length_cross);
	mat.col(1) *= (-static_cast<Real>(1.0) / length_cross);
	mat.col(2) *= (-static_cast<Real>(1.0) / length_cross);

	Matrix3r product_matrix;
	MathFunctions::crossProductMatrix(p2 - p1, product_matrix);
	d2p0 = mat * product_matrix;

	MathFunctions::crossProductMatrix(p0 - p2, product_matrix);
	d2p1 = mat * product_matrix;

	MathFunctions::crossProductMatrix(p1 - p0, product_matrix);
	d2p2 = mat * product_matrix;

	//////////////////////////////////////////////////////////////////////////
	// d1pi
	//////////////////////////////////////////////////////////////////////////
	Matrix3r product_mat_d3;
	Matrix3r product_mat_d2;
	MathFunctions::crossProductMatrix(d.col(2), product_mat_d3);
	MathFunctions::crossProductMatrix(d.col(1), product_mat_d2);

	d1p0 = product_mat_d2 * d3p0 - product_mat_d3 * d2p0;
	d1p1 = product_mat_d2 * d3p1 - product_mat_d3 * d2p1;
	d1p2 = -product_mat_d3 * d2p2;
	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedElasticRods::computeDarbouxGradient(
	const Vector3r& darboux_vector, const Real length,
	const Matrix3r& da, const Matrix3r& db,
	const Matrix3r dajpi[3][3], const Matrix3r dbjpi[3][3],
	//const Vector3r& bendAndTwistKs,
	Matrix3r& omega_pa, Matrix3r& omega_pb, Matrix3r& omega_pc, Matrix3r& omega_pd, Matrix3r& omega_pe
)
{
	Real X = static_cast<Real>(1.0) + da.col(0).dot(db.col(0)) + da.col(1).dot(db.col(1)) + da.col(2).dot(db.col(2));
	X = static_cast<Real>(2.0) / (length * X);

	for (int c = 0; c < 3; ++c)
	{
		const int i = permutation[c][0];
		const int j = permutation[c][1];
		const int k = permutation[c][2];
		// pa
		{
			Vector3r term1(0, 0, 0);
			Vector3r term2(0, 0, 0);
			Vector3r tmp(0, 0, 0);

			// first term
			term1 = dajpi[j][0].transpose() * db.col(k);
			tmp = dajpi[k][0].transpose() * db.col(j);
			term1 = term1 - tmp;
			// second term
			for (int n = 0; n < 3; ++n)
			{
				tmp = dajpi[n][0].transpose() * db.col(n);
				term2 = term2 + tmp;
			}
			omega_pa.col(i) = X * (term1 - (0.5 * darboux_vector[i] * length) * term2);
			//omega_pa.col(i) *= bendAndTwistKs[i];
		}
		// pb
		{
			Vector3r term1(0, 0, 0);
			Vector3r term2(0, 0, 0);
			Vector3r tmp(0, 0, 0);
			// first term
			term1 = dajpi[j][1].transpose() * db.col(k);
			tmp = dajpi[k][1].transpose() * db.col(j);
			term1 = term1 - tmp;
			// third term
			tmp = dbjpi[j][0].transpose() * da.col(k);
			term1 = term1 - tmp;

			tmp = dbjpi[k][0].transpose() * da.col(j);
			term1 = term1 + tmp;

			// second term
			for (int n = 0; n < 3; ++n)
			{
				tmp = dajpi[n][1].transpose() * db.col(n);
				term2 = term2 + tmp;

				tmp = dbjpi[n][0].transpose() * da.col(n);
				term2 = term2 + tmp;
			}
			omega_pb.col(i) = X * (term1 - (0.5 * darboux_vector[i] * length) * term2);
			//omega_pb.col(i) *= bendAndTwistKs[i];
		}
		// pc
		{
			Vector3r term1(0, 0, 0);
			Vector3r term2(0, 0, 0);
			Vector3r tmp(0, 0, 0);

			// first term
			term1 = dbjpi[j][1].transpose() * da.col(k);
			tmp = dbjpi[k][1].transpose() * da.col(j);
			term1 = term1 - tmp;

			// second term
			for (int n = 0; n < 3; ++n)
			{
				tmp = dbjpi[n][1].transpose() * da.col(n);
				term2 = term2 + tmp;
			}
			omega_pc.col(i) = -X * (term1 + (0.5 * darboux_vector[i] * length) * term2);
			//omega_pc.col(i) *= bendAndTwistKs[i];
		}
		// pd
		{
			Vector3r term1(0, 0, 0);
			Vector3r term2(0, 0, 0);
			Vector3r tmp(0, 0, 0);
			// first term
			term1 = dajpi[j][2].transpose() * db.col(k);
			tmp = dajpi[k][2].transpose() * db.col(j);
			term1 = term1 - tmp;
			// second term
			for (int n = 0; n < 3; ++n) {
				tmp = dajpi[n][2].transpose() * db.col(n);
				term2 = term2 + tmp;
			}
			omega_pd.col(i) = X * (term1 - (0.5 * darboux_vector[i] * length) * term2);
			//omega_pd.col(i) *= bendAndTwistKs[i];
		}
		// pe
		{
			Vector3r term1(0, 0, 0);
			Vector3r term2(0, 0, 0);
			Vector3r tmp(0, 0, 0);
			// first term
			term1 = dbjpi[j][2].transpose() * da.col(k);
			tmp = dbjpi[k][2].transpose() * da.col(j);
			term1 -= tmp;

			// second term
			for (int n = 0; n < 3; ++n)
			{
				tmp = dbjpi[n][2].transpose() * da.col(n);
				term2 += tmp;
			}

			omega_pe.col(i) = -X * (term1 + (0.5 * darboux_vector[i] * length) * term2);
			//omega_pe.col(i) *= bendAndTwistKs[i];
		}
	}
	return true;
}
// ----------------------------------------------------------------------------------------------


void PBD::DirectPositionBasedSolverForStiffRods::initLists(int numberOfIntervals, std::list <Node*>*& forward, std::list <Node*>*& backward, Node*& root)
{
	if (forward != NULL)
		delete[] forward;
	if (backward != NULL)
		delete[] backward;
	if (root != NULL)
		delete[] root;
	forward = new std::list<Node*>[numberOfIntervals];
	backward = new std::list<Node*>[numberOfIntervals];
	root = new Node[numberOfIntervals];
}

bool PBD::DirectPositionBasedSolverForStiffRods::isSegmentInInterval(RodSegment* segment, int intervalIndex, Interval* intervals, std::vector<RodConstraint*>& rodConstraints, std::vector<RodSegment*>& rodSegments)
{
	for (int i = intervals[intervalIndex].start; i <= intervals[intervalIndex].end; i++)
	{
		if ((segment == rodSegments[rodConstraints[i]->segmentIndex(0)])
			|| (segment == rodSegments[rodConstraints[i]->segmentIndex(1)]))
			return true;
	}
	return false;
}

bool PBD::DirectPositionBasedSolverForStiffRods::isConstraintInInterval(RodConstraint* constraint, int intervalIndex, Interval* intervals, std::vector<RodConstraint*>& rodConstraints)
{
	for (int i = intervals[intervalIndex].start; i <= intervals[intervalIndex].end; i++)
	{
		if (constraint == rodConstraints[i])
			return true;
	}
	return false;
}

void PBD::DirectPositionBasedSolverForStiffRods::initSegmentNode(Node* n, int intervalIndex, std::vector<RodConstraint*>& rodConstraints, std::vector<RodSegment*>& rodSegments, std::vector <RodConstraint*>& markedConstraints, Interval* intervals)
{
	RodSegment* segment =
		(RodSegment*)n->object;

	std::vector<RodConstraint*> constraints;
	std::vector<int> constraintIndices;
	for (int j = 0; j < static_cast<int>(rodConstraints.size()); ++j)
	{
		RodConstraint* constraint(rodConstraints[j]);
		if (rodSegments[constraint->segmentIndex(0)] == segment
			|| rodSegments[constraint->segmentIndex(1)] == segment)
		{
			constraints.push_back(constraint);
			constraintIndices.push_back(j);
		}
	}

	for (unsigned int i = 0; i < constraints.size(); i++)
	{
		if (!isConstraintInInterval(constraints[i], intervalIndex, intervals, rodConstraints))
			continue;

		// Test whether the edge has been visited before
		bool marked = false;
		for (unsigned int j = 0; j < markedConstraints.size(); j++)
		{
			if (constraints[i] == markedConstraints[j])
			{
				marked = true;
				break;
			}
		}
		if (!marked)
		{
			Node* constraintNode = new Node();
			constraintNode->index = constraintIndices[i];
			constraintNode->object = constraints[i];
			constraintNode->isconstraint = true;
			constraintNode->parent = n;
			constraintNode->D.setZero();
			constraintNode->Dinv.setZero();
			constraintNode->J.setZero();
			constraintNode->soln.setZero();

			n->children.push_back(constraintNode);

			Node* segmentNode = new Node();
			segmentNode->isconstraint = false;
			segmentNode->parent = constraintNode;

			//	get other segment connected to constraint for new node
			if (rodSegments[constraints[i]->segmentIndex(0)] == segment)
			{
				segmentNode->object = rodSegments[constraints[i]->segmentIndex(1)];
				segmentNode->index = constraints[i]->segmentIndex(1);
			}
			else
			{
				segmentNode->object = rodSegments[constraints[i]->segmentIndex(0)];
				segmentNode->index = constraints[i]->segmentIndex(0);
			}

			segmentNode->D.setZero();
			segmentNode->Dinv.setZero();
			segmentNode->J.setZero();
			segmentNode->soln.setZero();

			constraintNode->children.push_back(segmentNode);

			// mark constraint
			markedConstraints.push_back(constraints[i]);

			initSegmentNode(segmentNode, intervalIndex, rodConstraints,
				rodSegments, markedConstraints, intervals);
		}
	}
}

void PBD::DirectPositionBasedSolverForStiffRods::orderMatrix(Node* n, int intervalIndex, std::list <Node*>* forward, std::list <Node*>* backward)
{
	for (unsigned int i = 0; i < n->children.size(); i++)
		orderMatrix(n->children[i], intervalIndex, forward, backward);
	forward[intervalIndex].push_back(n);
	backward[intervalIndex].push_front(n);
}

void PBD::DirectPositionBasedSolverForStiffRods::initNodes(int intervalIndex, std::vector<RodSegment*>& rodSegments, Node*& root, Interval* intervals, std::vector<RodConstraint*>& rodConstraints, std::list <Node*>* forward, std::list <Node*>* backward, std::vector <RodConstraint*>& markedConstraints)
{
	// find root
	for (int i = 0; i < (int)rodSegments.size(); i++)
	{
		RodSegment* rb(rodSegments[i]);
		if (!isSegmentInInterval(rb, intervalIndex, intervals, rodConstraints, rodSegments))
			continue;
		else
		{
			if (root[intervalIndex].object == NULL)
			{
				root[intervalIndex].object = rb;
				root[intervalIndex].index = i;
			}
		}

		if (!rb->isDynamic())
		{
			root[intervalIndex].object = rb;
			root[intervalIndex].index = i;
			break;
		}
	}
	root[intervalIndex].isconstraint = false;
	root[intervalIndex].parent = NULL;

	root[intervalIndex].D.setZero();
	root[intervalIndex].Dinv.setZero();
	root[intervalIndex].soln.setZero();

	initSegmentNode(&root[intervalIndex], intervalIndex, rodConstraints,
		rodSegments, markedConstraints, intervals);
	orderMatrix(&root[intervalIndex], intervalIndex, forward, backward);
}

void PBD::DirectPositionBasedSolverForStiffRods::initTree(std::vector<RodConstraint*>& rodConstraints, std::vector<RodSegment*>& rodSegments, Interval*& intervals, int& numberOfIntervals, std::list <Node*>*& forward, std::list <Node*>*& backward, Node*& root)
{
	numberOfIntervals = 1;
	intervals = new Interval[1];
	intervals[0].start = 0;
	intervals[0].end = (int)rodConstraints.size() - 1;
	initLists(numberOfIntervals, forward, backward, root);

	std::vector <RodConstraint*> markedConstraints;
	for (int i = 0; i < numberOfIntervals; i++)
	{
		initNodes(i, rodSegments, root, intervals, rodConstraints, forward, backward, markedConstraints);
		markedConstraints.clear();
	}
}

bool PBD::DirectPositionBasedSolverForStiffRods::computeDarbouxVector(const Quaternionr& q0, const Quaternionr& q1, const Real averageSegmentLength, Vector3r& darbouxVector)
{
	//darbouxVector = 2. / averageSegmentLength * (q0.conjugate() * q1).vec();
	darbouxVector = (q0.conjugate() * q1).vec();
	return true;
}

bool PBD::DirectPositionBasedSolverForStiffRods::computeBendingAndTorsionJacobians(const Quaternionr& q0, const Quaternionr& q1, const Real averageSegmentLength, Eigen::Matrix<Real, 3, 4>& jOmega0, Eigen::Matrix<Real, 3, 4>& jOmega1)
{
	jOmega0 <<
		-q1.w(), -q1.z(), q1.y(), q1.x(),
		q1.z(), -q1.w(), -q1.x(), q1.y(),
		-q1.y(), q1.x(), -q1.w(), q1.z();
	jOmega1 <<
		q0.w(), q0.z(), -q0.y(), -q0.x(),
		-q0.z(), q0.w(), q0.x(), -q0.y(),
		q0.y(), -q0.x(), q0.w(), -q0.z();
	//jOmega0 *= static_cast<Real>(2.0) / averageSegmentLength;
	//jOmega1 *= static_cast<Real>(2.0) / averageSegmentLength;
	return true;
}

bool PBD::DirectPositionBasedSolverForStiffRods::computeMatrixG(const Quaternionr& q, Eigen::Matrix<Real, 4, 3>& G)
{
	// w component at index 3
	G <<
		static_cast<Real>(0.5) * q.w(), static_cast<Real>(0.5)* q.z(), -static_cast<Real>(0.5) * q.y(),
		-static_cast<Real>(0.5) * q.z(), static_cast<Real>(0.5)* q.w(), static_cast<Real>(0.5)* q.x(),
		static_cast<Real>(0.5)* q.y(), -static_cast<Real>(0.5) * q.x(), static_cast<Real>(0.5)* q.w(),
		-static_cast<Real>(0.5) * q.x(), -static_cast<Real>(0.5) * q.y(), -static_cast<Real>(0.5) * q.z();
	return true;
}

void PBD::DirectPositionBasedSolverForStiffRods::computeMatrixK(const Vector3r& connector, const Real invMass, const Vector3r& x, const Matrix3r& inertiaInverseW, Matrix3r& K)
{
	if (invMass != 0.0)
	{
		const Vector3r v = connector - x;
		const Real a = v[0];
		const Real b = v[1];
		const Real c = v[2];

		// J is symmetric
		const Real j11 = inertiaInverseW(0, 0);
		const Real j12 = inertiaInverseW(0, 1);
		const Real j13 = inertiaInverseW(0, 2);
		const Real j22 = inertiaInverseW(1, 1);
		const Real j23 = inertiaInverseW(1, 2);
		const Real j33 = inertiaInverseW(2, 2);

		K(0, 0) = c * c * j22 - b * c * (j23 + j23) + b * b * j33 + invMass;
		K(0, 1) = -(c * c * j12) + a * c * j23 + b * c * j13 - a * b * j33;
		K(0, 2) = b * c * j12 - a * c * j22 - b * b * j13 + a * b * j23;
		K(1, 0) = K(0, 1);
		K(1, 1) = c * c * j11 - a * c * (j13 + j13) + a * a * j33 + invMass;
		K(1, 2) = -(b * c * j11) + a * c * j12 + a * b * j13 - a * a * j23;
		K(2, 0) = K(0, 2);
		K(2, 1) = K(1, 2);
		K(2, 2) = b * b * j11 - a * b * (j12 + j12) + a * a * j22 + invMass;
	}
	else
		K.setZero();
}

void PBD::DirectPositionBasedSolverForStiffRods::getMassMatrix(RodSegment* segment, Matrix6r& M)
{
	if (!segment->isDynamic())
	{
		M = Matrix6r::Identity();
		return;
	}

	const Vector3r& inertiaLocal = segment->InertiaTensor();
	Matrix3r rotationMatrix(segment->Rotation().toRotationMatrix());
	Matrix3r inertia(rotationMatrix *
		Eigen::DiagonalMatrix<Real, 3>(inertiaLocal) *
		rotationMatrix.transpose());

	Real mass = segment->Mass();

	// Upper half
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 6; j++)
			if (i == j)
				M(i, j) = mass;
			else
				M(i, j) = 0.0;

	// lower left
	for (int i = 3; i < 6; i++)
		for (int j = 0; j < 3; j++)
			M(i, j) = 0.0;

	// lower right
	for (int i = 3; i < 6; i++)
		for (int j = 3; j < 6; j++)
			M(i, j) = inertia(i - 3, j - 3);
}

Real PBD::DirectPositionBasedSolverForStiffRods::factor(const int intervalIndex, const std::vector<RodConstraint*>& rodConstraints, std::vector<RodSegment*>& rodSegments, const Interval*& intervals, std::list <Node*>* forward, std::list <Node*>* backward, std::vector<Vector6r>& RHS, std::vector<Vector6r>& lambdaSums, std::vector<std::vector<Matrix3r>>& bendingAndTorsionJacobians)
{
	Real maxError(0.);

	// compute right hand side of linear equation system
	for (size_t currentConstraintIndex = 0; currentConstraintIndex < rodConstraints.size(); ++currentConstraintIndex)
	{
		RodConstraint* currentConstraint = rodConstraints[currentConstraintIndex];

		RodSegment* segment0 = rodSegments[currentConstraint->segmentIndex(0)];
		RodSegment* segment1 = rodSegments[currentConstraint->segmentIndex(1)];

		const Quaternionr& q0 = segment0->Rotation();
		const Quaternionr& q1 = segment1->Rotation();

		const Eigen::Matrix<Real, 3, 4>& constraintInfo(currentConstraint->getConstraintInfo());
		Vector6r& rhs(RHS[currentConstraintIndex]);

		// Compute zero-stretch part of constraint violation
		const Vector3r& connector0 = constraintInfo.col(2);
		const Vector3r& connector1 = constraintInfo.col(3);
#ifdef VMPROTECT
		Vector3r stretchViolation = connector0 * segment0->GetZeta() - connector1 * segment1->GetZeta();
#else
		Vector3r stretchViolation = connector0 - connector1;
#endif // VMPROTEC



		// compute Darboux vector (Equation (7))
		Vector3r omega;
		computeDarbouxVector(q0, q1, currentConstraint->getAverageSegmentLength(), omega);

		// Compute bending and torsion part of constraint violation
		Vector3r bendingAndTorsionViolation = omega - currentConstraint->getRestDarbouxVector();

		// fill right hand side of the linear equation system
		const Vector6r& lambdaSum(lambdaSums[currentConstraintIndex]);
#ifdef VMPROTECT
		rhs.block<3, 1>(0, 0) = -stretchViolation - Vector3r(currentConstraint->getStretchCompliance().array() * lambdaSum.block<3, 1>(0, 0).array());
		rhs.block<3, 1>(3, 0) = -bendingAndTorsionViolation - Vector3r(currentConstraint->getBendingAndTorsionCompliance().array() * lambdaSum.block<3, 1>(3, 0).array() * currentConstraint->getThetaVector().array());
#else
		rhs.block<3, 1>(0, 0) = -stretchViolation - Vector3r(currentConstraint->getStretchCompliance().array() * lambdaSum.block<3, 1>(0, 0).array());
		rhs.block<3, 1>(3, 0) = -bendingAndTorsionViolation - Vector3r(currentConstraint->getBendingAndTorsionCompliance().array() * lambdaSum.block<3, 1>(3, 0).array());
#endif // VMPROTECT

		// compute max error
		for (unsigned char i(0); i < 6; ++i)
		{
			maxError = std::max(maxError, abs(rhs[i]));
		}

		// Compute a part of the Jacobian here, because the relationship
		// of the first and second segment to the constraint can be determined directly

		// compute G matrices
		Eigen::Matrix<Real, 4, 3> G0, G1;
		computeMatrixG(q0, G0);
		computeMatrixG(q1, G1);

		// compute stretching bending Jacobians (Equation (10) and Equation (11))
		Eigen::Matrix<Real, 3, 4> jOmega0, jOmega1;
		computeBendingAndTorsionJacobians(q0, q1, currentConstraint->getAverageSegmentLength(), jOmega0, jOmega1);

		bendingAndTorsionJacobians[currentConstraintIndex][0] = jOmega0 * G0;
		bendingAndTorsionJacobians[currentConstraintIndex][1] = jOmega1 * G1;
	}

	std::list<Node*>::iterator nodeIter;
	for (nodeIter = forward[intervalIndex].begin(); nodeIter != forward[intervalIndex].end(); nodeIter++)
	{
		Node* node = *nodeIter;
		// compute system matrix diagonal
		if (node->isconstraint)
		{
			RodConstraint* currentConstraint = (RodConstraint*)node->object;
			//insert compliance
			node->D.setZero();
			const Vector3r& stretchCompliance(currentConstraint->getStretchCompliance());

			node->D(0, 0) -= stretchCompliance[0];
			node->D(1, 1) -= stretchCompliance[1];
			node->D(2, 2) -= stretchCompliance[2];

			const Vector3r& bendingAndTorsionCompliance(currentConstraint->getBendingAndTorsionCompliance());
			node->D(3, 3) -= bendingAndTorsionCompliance[0];
			node->D(4, 4) -= bendingAndTorsionCompliance[1];
			node->D(5, 5) -= bendingAndTorsionCompliance[2];
		}
		else
		{
			getMassMatrix((RodSegment*)node->object, node->D);
		}

		// compute Jacobian
		if (node->parent != NULL)
		{
			if (node->isconstraint)
			{
				//compute J 
				RodConstraint* constraint = (RodConstraint*)node->object;
				RodSegment* segment = (RodSegment*)node->parent->object;

				Real sign = 1;
				int segmentIndex = 0;
				if (segment == rodSegments[constraint->segmentIndex(1)])
				{
					segmentIndex = 1;
					sign = -1;
				}

				const Eigen::Matrix<Real, 3, 4>& constraintInfo(constraint->getConstraintInfo());
				const Vector3r r = constraintInfo.col(2 + segmentIndex) - segment->Position();
				Matrix3r r_cross;
				Real crossSign(-static_cast<Real>(1.0) * sign);
				MathFunctions::crossProductMatrix(crossSign * r, r_cross);

				Eigen::DiagonalMatrix<Real, 3> upperLeft(sign, sign, sign);
				node->J.block<3, 3>(0, 0) = upperLeft;

				Matrix3r lowerLeft(Matrix3r::Zero());
				node->J.block<3, 3>(3, 0) = lowerLeft;

				node->J.block<3, 3>(0, 3) = r_cross;

				Matrix3r& lowerRight(bendingAndTorsionJacobians[node->index][segmentIndex]);
				node->J.block<3, 3>(3, 3) = lowerRight;
			}
			else
			{
				//compute JT
				RodConstraint* constraint = (RodConstraint*)node->parent->object;
				RodSegment* segment = (RodSegment*)node->object;

				Real sign = 1;
				int segmentIndex = 0;
				if (segment == rodSegments[constraint->segmentIndex(1)])
				{
					segmentIndex = 1;
					sign = -1;
				}

				const Eigen::Matrix<Real, 3, 4>& constraintInfo(constraint->getConstraintInfo());
				const Vector3r r = constraintInfo.col(2 + segmentIndex) - segment->Position();
				Matrix3r r_crossT;
				MathFunctions::crossProductMatrix(sign * r, r_crossT);

				Eigen::DiagonalMatrix<Real, 3> upperLeft(sign, sign, sign);
				node->J.block<3, 3>(0, 0) = upperLeft;

				node->J.block<3, 3>(3, 0) = r_crossT;

				Matrix3r upperRight(Matrix3r::Zero());
				node->J.block<3, 3>(0, 3) = upperRight;

				Matrix3r lowerRight(bendingAndTorsionJacobians[node->parent->index][segmentIndex].transpose());
				node->J.block<3, 3>(3, 3) = lowerRight;
			}
		}
	}

	for (nodeIter = forward[intervalIndex].begin(); nodeIter != forward[intervalIndex].end(); nodeIter++)
	{
		Node* node = *nodeIter;
		std::vector <Node*> children = node->children;
		for (size_t i = 0; i < children.size(); i++)
		{
			Matrix6r JT = (children[i]->J).transpose();
			Matrix6r& D = children[i]->D;
			Matrix6r& J = children[i]->J;
			Matrix6r JTDJ = ((JT * D) * J);
			node->D = node->D - JTDJ;
		}
		bool chk = false;
		if (!node->isconstraint)
		{
			RodSegment* segment = (RodSegment*)node->object;
			if (!segment->isDynamic())
			{
				node->Dinv.setZero();
				chk = true;
			}
		}

		node->DLDLT.compute(node->D); // result reused in solve()
		if (node->parent != NULL)
		{
			if (!chk)
			{
				node->J = node->DLDLT.solve(node->J);
			}
			else
			{
				node->J.setZero();
			}
		}
	}
	return maxError;
}

bool PBD::DirectPositionBasedSolverForStiffRods::solve(int intervalIndex, std::list <Node*>* forward, std::list <Node*>* backward, std::vector<Vector6r>& RHS, std::vector<Vector6r>& lambdaSums, std::vector<Vector3r>& corr_x, std::vector<Quaternionr>& corr_q)
{
	std::list<Node*>::iterator nodeIter;
	for (nodeIter = forward[intervalIndex].begin(); nodeIter != forward[intervalIndex].end(); nodeIter++)
	{
		Node* node = *nodeIter;
		if (node->isconstraint)
		{
			node->soln = -RHS[node->index];
		}
		else
		{
			node->soln.setZero();
		}
		std::vector <Node*>& children = node->children;
		for (size_t i = 0; i < children.size(); ++i)
		{
			Matrix6r cJT = children[i]->J.transpose();
			Vector6r& csoln = children[i]->soln;
			Vector6r v = cJT * csoln;
			node->soln = node->soln - v;
		}
	}

	for (nodeIter = backward[intervalIndex].begin(); nodeIter != backward[intervalIndex].end(); nodeIter++)
	{
		Node* node = *nodeIter;

		bool noZeroDinv(true);
		if (!node->isconstraint)
		{
			RodSegment* segment = (RodSegment*)node->object;
			noZeroDinv = segment->isDynamic();
		}
		if (noZeroDinv) // if DInv == 0 child value is 0 and node->soln is not altered
		{
			node->soln = node->DLDLT.solve(node->soln);

			if (node->parent != NULL)
			{
				node->soln -= node->J * node->parent->soln;
			}
		}
		else
		{
			node->soln.setZero(); // segment of node is not dynamic
		}

		if (node->isconstraint)
		{
			lambdaSums[node->index] += node->soln;
		}
	}

	// compute position and orientation updates
	for (nodeIter = forward[intervalIndex].begin(); nodeIter != forward[intervalIndex].end(); nodeIter++)
	{
		Node* node = *nodeIter;
		if (!node->isconstraint)
		{
			RodSegment* segment = (RodSegment*)node->object;
			if (!segment->isDynamic())
			{
				break;
			}

			const Vector6r& soln(node->soln);
			Vector3r deltaXSoln = Vector3r(-soln[0], -soln[1], -soln[2]);
			corr_x[node->index] = deltaXSoln;

			Eigen::Matrix<Real, 4, 3> G;
			computeMatrixG(segment->Rotation(), G);
			Quaternionr deltaQSoln;
			deltaQSoln.coeffs() = G * Vector3r(-soln[3], -soln[4], -soln[5]);
			corr_q[node->index] = deltaQSoln;
		}
	}
	return true;
}

bool PBD::DirectPositionBasedSolverForStiffRods::init_DirectPositionBasedSolverForStiffRodsConstraint(
	std::vector<RodConstraint*>& rodConstraints,
	std::vector<RodSegment*>& rodSegments,
	Interval*& intervals,
	int& numberOfIntervals,
	std::list <Node*>*& forward,
	std::list <Node*>*& backward,
	Node*& root,
	const std::vector<Vector3r>& constraintPositions,
	const std::vector<Real>& averageRadii,
	const std::vector<Real>& youngsModuli,
	const std::vector<Real>& torsionModuli,
	std::vector<Vector6r>& RHS,
	std::vector<Vector6r>& lambdaSums,
	std::vector<std::vector<Matrix3r>>& bendingAndTorsionJacobians,
	std::vector<Vector3r>& corr_x,
	std::vector<Quaternionr>& corr_q
)
{
	// init constraints
	for (size_t cIdx(0); cIdx < rodConstraints.size(); ++cIdx)
	{
		RodConstraint* constraint(rodConstraints[cIdx]);
		RodSegment* segment0(rodSegments[constraint->segmentIndex(0)]);
		RodSegment* segment1(rodSegments[constraint->segmentIndex(1)]);

		init_StretchBendingTwistingConstraint(
			segment0->Position(), segment0->Rotation(), segment1->Position(), segment1->Rotation(),
			constraintPositions[cIdx], averageRadii[cIdx], constraint->getAverageSegmentLength(),
			youngsModuli[cIdx], torsionModuli[cIdx], constraint->getConstraintInfo(),
			constraint->getStiffnessCoefficientK(), constraint->getRestDarbouxVector());
	}

	// compute tree data structure for direct solver
	initTree(rodConstraints, rodSegments, intervals, numberOfIntervals, forward, backward, root);

	RHS.resize(rodConstraints.size());
	std::fill(RHS.begin(), RHS.end(), Vector6r::Zero());

	lambdaSums.resize(rodConstraints.size());
	std::fill(lambdaSums.begin(), lambdaSums.end(), Vector6r::Zero());

	bendingAndTorsionJacobians.resize(rodConstraints.size());
	std::vector<Matrix3r> sampleJacobians(2);
	sampleJacobians[0].setZero();
	sampleJacobians[1].setZero();
	std::fill(bendingAndTorsionJacobians.begin(), bendingAndTorsionJacobians.end(), sampleJacobians);

	corr_x.resize(rodSegments.size());
	std::fill(corr_x.begin(), corr_x.end(), Vector3r::Zero());

	corr_q.resize(rodSegments.size());
	std::fill(corr_q.begin(), corr_q.end(), Quaternionr::Identity());

	return true;
}

bool PBD::DirectPositionBasedSolverForStiffRods::initBeforeProjection_DirectPositionBasedSolverForStiffRodsConstraint(
	const std::vector<RodConstraint*>& rodConstraints,
	const Real inverseTimeStepSize,
	std::vector<Vector6r>& lambdaSums
)
{
#ifdef VMPROTECT
	//TODO too slow to check each time
	//int stiffnessMult = VMProtectGetSerialNumberState();
	float stiffnessMult = 0.0f;
#else
	float stiffnessMult = 0.0f;
#endif

	for (size_t cIdx(0); cIdx < rodConstraints.size(); ++cIdx)
	{
		RodConstraint* constraint(rodConstraints[cIdx]);

		initBeforeProjection_StretchBendingTwistingConstraint(
			constraint->getStiffnessCoefficientK(),
			inverseTimeStepSize,
			constraint->getAverageSegmentLength(),
			constraint->getStretchCompliance(),
			constraint->getBendingAndTorsionCompliance(),
			lambdaSums[cIdx]);
	}
	return true;
}



bool PBD::DirectPositionBasedSolverForStiffRods::update_DirectPositionBasedSolverForStiffRodsConstraint(
	const std::vector<RodConstraint*>& rodConstraints,
	const std::vector<RodSegment*>& rodSegments
)
{
	// update rod constraints
	for (size_t cIdx(0); cIdx < rodConstraints.size(); ++cIdx)
	{
		RodConstraint* constraint(rodConstraints[cIdx]);
		RodSegment* segment0(rodSegments[constraint->segmentIndex(0)]);
		RodSegment* segment1(rodSegments[constraint->segmentIndex(1)]);

		update_StretchBendingTwistingConstraint(
			segment0->Position(), segment0->Rotation(), segment1->Position(), segment1->Rotation(),
			constraint->getConstraintInfo());
	}
	return true;
}



bool PBD::DirectPositionBasedSolverForStiffRods::solve_DirectPositionBasedSolverForStiffRodsConstraint(
	const std::vector<RodConstraint*>& rodConstraints,
	std::vector<RodSegment*>& rodSegments,
	const Interval* intervals,
	const int& numberOfIntervals,
	int& numberOfUpdates,
	std::list <Node*>* forward,
	std::list <Node*>* backward,
	std::vector<Vector6r>& RHS,
	std::vector<Vector6r>& lambdaSums,
	std::vector<std::vector<Matrix3r>>& bendingAndTorsionJacobians,
	std::vector<Vector3r>& corr_x,
	std::vector<Quaternionr>& corr_q
)
{
	for (int i = 0; i < numberOfIntervals; i++)
	{
		factor(i, rodConstraints, rodSegments, intervals,
			forward, backward, RHS, lambdaSums, bendingAndTorsionJacobians);
	}

	for (int i = 0; i < numberOfIntervals; i++)
	{
		solve(i, forward, backward, RHS, lambdaSums, corr_x, corr_q);
#ifdef VMPROTECT
		if (numberOfUpdates++ % 30000 == 0)
		{
			Real zeta = VMProtectGetSerialNumberState() == 0 ? 1.00001f : 0.001f;
			for (size_t i = 0; i < rodSegments.size(); i++)
			{
				RodSegment* seg = rodSegments[i];
				seg->SetZeta(zeta);
			}
		}
#endif // VMPROTECT


	}
	return true;
}

bool PBD::DirectPositionBasedSolverForStiffRods::solve_DirectPositionBasedSolverForStiffRodsConstraintBanded(
	const std::vector<RodConstraint*>& rodConstraints,
	std::vector<RodSegment*>& rodSegments,
	std::vector<BandedNode*>& nodesForward,
	std::vector<BandedNode*>& nodesBackward,
	std::vector<Vector6r>& RHS,
	std::vector<Vector6r>& lambdaSums,
	int& numberOfUpdates,
	std::vector<std::vector<Matrix3r>>& bendingAndTorsionJacobians,
	MatrixXr& JMJT_Global,
	MatrixXr& JMJT_Banded,
	VectorXr& RHS_Global,
	std::vector<Vector3r>& corr_x,
	std::vector<Quaternionr>& corr_q
)
{

	Real maxError(0.);

#ifdef  _PERF_TIMER
	plf::nanotimer timer;
	timer.start();
#endif //  _PERF_TIMER



	// compute right hand side of linear equation system
	for (size_t currentConstraintIndex = 0; currentConstraintIndex < rodConstraints.size(); ++currentConstraintIndex)
	{
		RodConstraint* currentConstraint = rodConstraints[currentConstraintIndex];

		RodSegment* segment0 = rodSegments[currentConstraint->segmentIndex(0)];
		RodSegment* segment1 = rodSegments[currentConstraint->segmentIndex(1)];

		const Quaternionr& q0 = segment0->Rotation();
		const Quaternionr& q1 = segment1->Rotation();

		const Eigen::Matrix<Real, 3, 4>& constraintInfo(currentConstraint->getConstraintInfo());
		Vector6r& rhs(RHS[currentConstraintIndex]);

		// Compute zero-stretch part of constraint violation
		const Vector3r& connector0 = constraintInfo.col(2);
		const Vector3r& connector1 = constraintInfo.col(3);

#ifdef VMPROTECT
		Vector3r stretchViolation = connector0 * segment0->GetZeta() - connector1 * segment1->GetZeta();
#else
		Vector3r stretchViolation = connector0 - connector1;
#endif // VMPROTEC

		// compute Darboux vector (Equation (7))
		Vector3r omega;
		computeDarbouxVector(q0, q1, currentConstraint->getAverageSegmentLength(), omega);

		// Compute bending and torsion part of constraint violation
		Vector3r bendingAndTorsionViolation = omega - currentConstraint->getRestDarbouxVector();

		// fill right hand side of the linear equation system
		const Vector6r& lambdaSum(lambdaSums[currentConstraintIndex]);
		//rhs.block<3, 1>(0, 0) = -stretchViolation - Vector3r(currentConstraint->getStretchCompliance().array() * lambdaSum.block<3, 1>(0, 0).array());
		//rhs.block<3, 1>(3, 0) = -bendingAndTorsionViolation - Vector3r(currentConstraint->getBendingAndTorsionCompliance().array() * lambdaSum.block<3, 1>(3, 0).array());

#ifdef VMPROTECT
		rhs.block<3, 1>(0, 0) = -stretchViolation;
		rhs.block<3, 1>(3, 0) = -bendingAndTorsionViolation * currentConstraint->getThetaVector().x();
#else
		rhs.block<3, 1>(0, 0) = -stretchViolation;
		rhs.block<3, 1>(3, 0) = -bendingAndTorsionViolation;
#endif // VMPROTECT

		// compute max error
		for (unsigned char i(0); i < 6; ++i)
		{
			maxError = std::max(maxError, abs(rhs[i]));
		}

		// Compute a part of the Jacobian here, because the relationship
		// of the first and second segment to the constraint can be determined directly

		// compute G matrices
		Eigen::Matrix<Real, 4, 3> G0, G1;
		computeMatrixG(q0, G0);
		computeMatrixG(q1, G1);

		// compute stretching bending Jacobians (Equation (10) and Equation (11))
		Eigen::Matrix<Real, 3, 4> jOmega0, jOmega1;
		computeBendingAndTorsionJacobians(q0, q1, currentConstraint->getAverageSegmentLength(), jOmega0, jOmega1);

		bendingAndTorsionJacobians[currentConstraintIndex][0] = jOmega0 * G0;
		bendingAndTorsionJacobians[currentConstraintIndex][1] = jOmega1 * G1;

		BandedNode* node = nodesForward[currentConstraintIndex];
		// compute system matrix diagonal
		//if (node->isconstraint)

		//RodConstraint* currentConstraint = (RodConstraint*)node->object;

		//insert compliance
		node->D.setZero();
		const Vector3r& stretchCompliance(currentConstraint->getStretchCompliance());

		node->D(0, 0) -= stretchCompliance[0];
		node->D(1, 1) -= stretchCompliance[1];
		node->D(2, 2) -= stretchCompliance[2];

		const Vector3r& bendingAndTorsionCompliance(currentConstraint->getBendingAndTorsionCompliance());
		node->D(3, 3) -= bendingAndTorsionCompliance[0];
		node->D(4, 4) -= bendingAndTorsionCompliance[1];
		node->D(5, 5) -= bendingAndTorsionCompliance[2];


		//getMassMatrix((RodSegment*)node->object, node->D);
		//getMassMatrix(segment0, node->M);


		// compute Jacobian


		//if (node->isconstraint)
		if (true)
		{
			//compute J 
			//RodConstraint *constraint = (RodConstraint*)node->object;
			//RodSegment *segment = (RodSegment*)node->parent->object;

			//Real sign = 1;
			//int segmentIndex = 0;
			//if (segment == rodSegments[constraint->segmentIndex(1)])
			//{
			//	segmentIndex = 1;
			//	sign = -1;
			//}



			//const Eigen::Matrix<Real, 3, 4> &constraintInfo(constraint->getConstraintInfo());
			//const Vector3r r = constraintInfo.col(2 + segmentIndex) - segment->Position();

			//Forward
			{
				BandedNode* node = nodesForward[currentConstraintIndex];
				Real sign = 1;
				int segmentIndex = 0; // CHECK THIS

				//Vector3r r = constraintInfo.col(2 + segmentIndex) - segment->Position();
				Vector3r r = constraintInfo.col(2 + segmentIndex) - segment0->Position();

				Matrix3r r_cross;
				Real crossSign(-static_cast<Real>(1.0) * sign);
				MathFunctions::crossProductMatrix(crossSign * r, r_cross);

				Eigen::DiagonalMatrix<Real, 3> upperLeft(sign, sign, sign);
				node->J.block<3, 3>(0, 0) = upperLeft;

				Matrix3r lowerLeft(Matrix3r::Zero());
				node->J.block<3, 3>(3, 0) = lowerLeft;

				node->J.block<3, 3>(0, 3) = r_cross;

				Matrix3r& lowerRight(bendingAndTorsionJacobians[node->index][segmentIndex]);
				node->J.block<3, 3>(3, 3) = lowerRight;
			}


			//Backward
			{
				BandedNode* node = nodesBackward[currentConstraintIndex];
				Real sign = -1;
				int segmentIndex = 1;// CHECK THIS

				//Vector3r r = constraintInfo.col(2 + segmentIndex) - segment->Position();
				Vector3r r = constraintInfo.col(2 + segmentIndex) - segment1->Position();

				Matrix3r r_cross;
				Real crossSign(-static_cast<Real>(1.0) * sign);
				MathFunctions::crossProductMatrix(crossSign * r, r_cross);

				Eigen::DiagonalMatrix<Real, 3> upperLeft(sign, sign, sign);
				node->J.block<3, 3>(0, 0) = upperLeft;

				Matrix3r lowerLeft(Matrix3r::Zero());
				node->J.block<3, 3>(3, 0) = lowerLeft;

				node->J.block<3, 3>(0, 3) = r_cross;

				Matrix3r& lowerRight(bendingAndTorsionJacobians[node->index][segmentIndex]);
				node->J.block<3, 3>(3, 3) = lowerRight;
			}
		}
		//else
		//{
		//	//compute JT
		//	RodConstraint *constraint = (RodConstraint*)node->parent->object;
		//	RodSegment *segment = (RodSegment*)node->object;

		//	Real sign = 1;
		//	int segmentIndex = 0;
		//	if (segment == rodSegments[constraint->segmentIndex(1)])
		//	{
		//		segmentIndex = 1;
		//		sign = -1;
		//	}

		//	const Eigen::Matrix<Real, 3, 4> &constraintInfo(constraint->getConstraintInfo());
		//	const Vector3r r = constraintInfo.col(2 + segmentIndex) - segment->Position();
		//	Matrix3r r_crossT;
		//	MathFunctions::crossProductMatrix(sign*r, r_crossT);

		//	Eigen::DiagonalMatrix<Real, 3> upperLeft(sign, sign, sign);
		//	node->J.block<3, 3>(0, 0) = upperLeft;

		//	node->J.block<3, 3>(3, 0) = r_crossT;

		//	Matrix3r upperRight(Matrix3r::Zero());
		//	node->J.block<3, 3>(0, 3) = upperRight;

		//	Matrix3r lowerRight(bendingAndTorsionJacobians[node->parent->index][segmentIndex].transpose());
		//	node->J.block<3, 3>(3, 3) = lowerRight;
		//}
	}

	// Asemble global matrices
	size_t rows = rodConstraints.size() * 6;
	size_t cols = rodSegments.size() * 6;


	//VectorXr RHS_global(rodConstraints.size() * 6);

#ifdef  _PERF_TIMER
	std::cout << "Jacobians: " << timer.get_elapsed_ms() << " ms." << std::endl;
	timer.start();
#endif

	const int N = JMJT_Global.rows();
	const int KU = 11;
	const int KL = 11;


	Matrix6r D = nodesForward[1]->D;
	for (size_t c = 0; c < rodConstraints.size(); ++c)
	{

		RodConstraint* currentConstraint = rodConstraints[c];
		BandedNode* nodeFwd = nodesForward[c];
		BandedNode* nodeBwd = nodesBackward[c];

		const int row = c * 6;
		const int col = c * 6;

		//Matrix6r M;
		//M.setZero();
		//getMassMatrix(rodSegments[c], M);


		//main diagonal
		//JMJT_Global.block<6, 6>(row, row) = nodeFwd->J * nodeFwd->J.transpose() + nodeBwd->J * nodeBwd->J.transpose();
		JMJT_Global.block<6, 6>(row, row) = nodeFwd->J * nodeFwd->J.transpose() + nodeBwd->J * nodeBwd->J.transpose();
		Matrix6r JMJT_block = nodeFwd->J * nodeFwd->J.transpose() + nodeBwd->J * nodeBwd->J.transpose();


		JMJT_block(0, 0) -= D(0, 0);
		JMJT_block(1, 1) -= D(1, 1);
		JMJT_block(2, 2) -= D(2, 2);
		JMJT_block(3, 3) -= D(3, 3);
		JMJT_block(4, 4) -= D(4, 4);
		JMJT_block(5, 5) -= D(5, 5);

		//in band:
		//return column - ku <= row && row <= column + kl;
		for (int j = col, v = 0; j < col + 6; j++, v++)
		{
			for (int i = row, u = 0; i < row + 6; i++, u++)
			{
				//we know that we are within bands
				//JMJT_Banded(KU - j + i + 11, j) = JMJT_Global(i, j);
				//JMJT_Banded(KU - j + i + 11, j) = JMJT_block(u, v);

				//Symmetric
				if (KU - j + i < JMJT_Banded.rows())
					JMJT_Banded(KU - j + i, j) = JMJT_block(u, v);
			}
		}

		if (c < rodConstraints.size() - 1)
		{
			BandedNode* nodeNext = nodesForward[c + 1];

			//off-diagonals - these two are almost symmetric (floating-point errors?)


			//sub-diagonal
			JMJT_Global.block<6, 6>(row + 6, row) = nodeNext->J * nodeBwd->J.transpose();

			JMJT_block = nodeNext->J * nodeBwd->J.transpose();

			//for (int j = col, v = 0; j < col + 6; j++, v++)
			//{
			//	for (int i = row + 6, u = 0; i < row + 12; i++, u++)
			//	{
			//		//JMJT_Banded(KU - j + i + 11, j) = JMJT_Global(i, j);
			//		//JMJT_Banded(KU - j + i + 11, j) = JMJT_block(u, v);
			//	}
			//}



			//super-diagonal
			//JMJT_Global.block<6, 6>(row, row + 6) = nodeBwd->J * nodeNext->J.transpose();
			JMJT_Global.block<6, 6>(row, row + 6) = (nodeNext->J * nodeBwd->J.transpose()).transpose();
			//JMJT_block = nodeBwd->J * nodeNext->J.transpose();
			JMJT_block.transposeInPlace();
			for (int j = col + 6, v = 0; j < col + 12; j++, v++)
			{
				for (int i = row, u = 0; i < row + 6; i++, u++)
				{
					//JMJT_Banded(KU - j + i + 11, j) = JMJT_Global(i, j);
					//JMJT_Banded(KU - j + i + 11, j) = JMJT_block(u, v);

					//Symmetric
					if (KU - j + i < JMJT_Banded.rows())
						JMJT_Banded(KU - j + i, j) = JMJT_block(u, v);
				}
			}


		}

		RHS_Global.block<6, 1>(row, 0) = RHS[c];

	}


	for (size_t i = 0; i < rodConstraints.size(); ++i)
	{
		const int row = i * 6;
		const int col = i * 6;
		Matrix6r D = nodesForward[1]->D;

		JMJT_Global(row + 0, col + 0) -= D(0, 0);
		JMJT_Global(row + 1, col + 1) -= D(1, 1);
		JMJT_Global(row + 2, col + 2) -= D(2, 2);
		JMJT_Global(row + 3, col + 3) -= D(3, 3);
		JMJT_Global(row + 4, col + 4) -= D(4, 4);
		JMJT_Global(row + 5, col + 5) -= D(5, 5);
	}


	//std::cout << "JMJT_Global---------------------------------------------------------------------------------------------" << std::endl;
	//std::cout << JMJT_Global << std::endl;

	//std::cout << "JMJT_Banded---------------------------------------------------------------------------------------------" << std::endl;
	//std::cout << JMJT_Banded.block<34, 24>(0, 0) << std::endl;
	//std::cout << JMJT_Banded << std::endl;


#ifdef  _PERF_TIMER
	std::cout << "JMJT Assemby: " << timer.get_elapsed_ms() << " ms." << std::endl;
	timer.start();
#endif

	//for (int j = 0; j < N; j++)
	//{
	//	for (int i = std::max(0, j - KU); i < std::min(N, j + KL + 1); i++)
	//	{
	//		JMJT_Banded(KU - j + i + 11, j) = JMJT_Global(i, j);
	//	}
	//}


	//for (int j = 0; j < N; j++)
	//{
	//	for (int i = std::max(0, j - KU); i < j; i++)
	//	{
	//		JMJT_Banded(KU - j + i, j) = JMJT_Global(i, j);
	//	}
	//}

	//	DO 20 J = 1, N
	//		M = K + 1 - J
	//		DO 10 I = MAX(1, J - K), J
	//		ASB(M + I, J) = A(I, J)
	//	10 CONTINUE
	//20 CONTINUE

#ifdef  _PERF_TIMER
	std::cout << "JMJT Banding: " << timer.get_elapsed_ms() << " ms." << std::endl;
	timer.start();
#endif



	const int NRHS = 1;
	const int LDA = N;
	const int LDB = N;
	const int LWORK = 8;

	int info;
	int* ipiv = new int[N];
	float* work = new float[LWORK];


	//SGESV(&N, &NRHS, JMJT_Global.data(), &LDA, ipiv, RHS_Global.data(), &LDB, &info);
	//SSYSV("U", &N, &NRHS, JMJT_Global.data(), &LDA, ipiv, RHS_Global.data(), &LDB, work, &LWORK, &info);


	const int LDAB = 34;

	//SGBSV(&N, &KL, &KU, &NRHS, JMJT_Banded.data(), &LDAB, ipiv, RHS_Global.data(), &LDB, &info);


	//MatrixXr JMJT_PBSV(JMJT_Banded.topRows(12));
	//const int PBSV_N = JMJT_PBSV.cols();
	//const int PBSV_LDA = JMJT_PBSV.rows();
	//SPBSV("U", &PBSV_N, &KU, &NRHS, JMJT_PBSV.data(), &PBSV_LDA, RHS_Global.data(), &LDB, &info);
	//std::cout << "JMJT_PBSV---------------------------------------------------------------------------------------------" << std::endl;
	//std::cout << JMJT_PBSV << std::endl;

	const int PBSV_N = JMJT_Banded.cols();
	const int PBSV_LDA = JMJT_Banded.rows();
	SPBSV("U", &PBSV_N, &KU, &NRHS, JMJT_Banded.data(), &PBSV_LDA, RHS_Global.data(), &LDB, &info);
	//LAPACK_spbsv("U", &PBSV_N, &KU, &NRHS, JMJT_Banded.data(), &PBSV_LDA, RHS_Global.data(), &LDB, &info);

	delete ipiv;
	delete work;

#ifdef  _PERF_TIMER
	std::cout << "SOLVE: " << timer.get_elapsed_ms() << " ms." << std::endl;
#endif

	//std::cout << "INFO: " << info << std::endl;
	if (info != 0)
	{
		std::cout << "LINEAR SOLVE ERROR!  INFO" << info << std::endl;
		return false;
	}

#ifdef  _PERF_TIMER
	timer.start();
#endif


	VectorXr deltaLambda = RHS_Global;

	//	std::cout << "Decomposition---------------------------------------------------------------------------------------------" << std::endl;
	//	std::cout << decomposition.matrixLDLT() << std::endl;


	//	std::cout << "DeltaLambda---------------------------------------------------------------------------------------------" << std::endl;
	//	std::cout << deltaLambda.transpose() << std::endl;

		//VectorXr JT_deltaLambda = JT * deltaLambda;

		//std::cout << "JT_DeltaLambda---------------------------------------------------------------------------------------------" << std::endl;
		//std::cout << JT_deltaLambda.transpose() << std::endl;

		//VectorXr JT_deltaLambda2(rodSegments.size() * 6);
		//JT_deltaLambda2.setZero();

	for (size_t i = 0; i < rodSegments.size(); ++i)
	{
		//Vector3r deltaLambdaStretch(JT_deltaLambda.block<3, 1>(i * 6, 0));
		//Vector3r deltaLambdaBendingAndTorsion(JT_deltaLambda.block<3, 1>( i * 6 + 3, 0));
		Vector6r JT_deltaLambda;
		if (i == 0)
		{
			//JT_deltaLambda2.block<6, 1>(0, 0) = nodesForward[i]->J.transpose() * deltaLambda.block<6, 1>(0, 0);
			JT_deltaLambda = nodesForward[i]->J.transpose() * deltaLambda.block<6, 1>(0, 0);
		}
		else if (i < rodSegments.size() - 1)
		{
			//JT_deltaLambda2.block<6, 1>(i*6, 0) = nodesBackward[i-1]->J.transpose() * deltaLambda.block<6, 1>((i-1) * 6, 0) + nodesForward[i]->J.transpose() * deltaLambda.block<6, 1>(i*6, 0);
			JT_deltaLambda = nodesBackward[i - 1]->J.transpose() * deltaLambda.block<6, 1>((i - 1) * 6, 0) + nodesForward[i]->J.transpose() * deltaLambda.block<6, 1>(i * 6, 0);
		}
		else
		{
			//JT_deltaLambda2.block<6, 1>(i * 6, 0) = nodesBackward[i - 1]->J.transpose() * deltaLambda.block<6, 1>((i - 1) * 6, 0);
			JT_deltaLambda = nodesBackward[i - 1]->J.transpose() * deltaLambda.block<6, 1>((i - 1) * 6, 0);
		}

		Vector3r deltaLambdaStretch = JT_deltaLambda.block<3, 1>(0, 0);
		Vector3r deltaLambdaBendingAndTorsion = JT_deltaLambda.block<3, 1>(3, 0);

		//std::cout << "JT_DeltaLambda_SUB---------------------------------------------------------------------------------------------" << std::endl;
		//std::cout << JT_deltaLambda_SUB << std::endl;

		RodSegment* segment = rodSegments[i];
		if (segment->Mass() != 0.0f)
		{
			corr_x[i] += segment->Mass() * deltaLambdaStretch;
			//corr_q[i].coeffs() += G0 * (inertiaInverseW0 * ra_crossT * (-1 * deltaLambdaStretch) + MInvJT0 * deltaLambdaBendingAndTorsion);

			Quaternionr deltaQSoln;
			Eigen::Matrix<Real, 4, 3> G;
			computeMatrixG(segment->Rotation(), G);
			deltaQSoln.coeffs() = G * deltaLambdaBendingAndTorsion;
			corr_q[i] = deltaQSoln;
			//corr_q[i].coeffs() += G * deltaLambdaBendingAndTorsion;

		}
	}
	//std::cout << "JT_DeltaLambda_Custom---------------------------------------------------------------------------------------------" << std::endl;
	//std::cout << JT_deltaLambda2.transpose() << std::endl;


#ifdef  _PERF_TIMER
	std::cout << "JT_MULT: " << timer.get_elapsed_ms() << " ms." << std::endl;
#endif
	return true;
}


bool PBD::DirectPositionBasedSolverForStiffRods::computeJacobians_DirectPositionBasedSolverForStiffRodsConstraintBanded(
	int startId, int count,
	const std::vector<RodConstraint*>& rodConstraints,
	std::vector<RodSegment*>& rodSegments,
	std::vector<BandedNode*>& nodesForward,
	std::vector<BandedNode*>& nodesBackward,
	std::vector<Vector6r>& RHS,
	std::vector<Vector6r>& lambdaSums,
	int& numberOfUpdates,
	std::vector<std::vector<Matrix3r>>& bendingAndTorsionJacobians,
	MatrixXr& JMJT_Global,
	MatrixXr& JMJT_Banded,
	VectorXr& RHS_Global
)
{
#ifdef  _PERF_TIMER
	plf::nanotimer timer;
	timer.start();
#endif //  _PERF_TIMER

	const int N = JMJT_Global.rows();
	const int KU = 11;
	const int KL = 11;

	const size_t rows = rodConstraints.size() * 6;
	const size_t cols = rodSegments.size() * 6;


	// compute right hand side of linear equation system
	//for (size_t currentConstraintIndex = 0; currentConstraintIndex < rodConstraints.size(); ++currentConstraintIndex)
	const int endId = startId + count;
	for (size_t currentConstraintIndex = startId; currentConstraintIndex < endId; ++currentConstraintIndex)
	{
		RodConstraint* currentConstraint = rodConstraints[currentConstraintIndex];

		RodSegment* segment0 = rodSegments[currentConstraint->segmentIndex(0)];
		RodSegment* segment1 = rodSegments[currentConstraint->segmentIndex(1)];

		const Quaternionr& q0 = segment0->Rotation();
		const Quaternionr& q1 = segment1->Rotation();

		const Eigen::Matrix<Real, 3, 4>& constraintInfo(currentConstraint->getConstraintInfo());
		Vector6r& rhs(RHS[currentConstraintIndex]);

		// Compute zero-stretch part of constraint violation
		const Vector3r& connector0 = constraintInfo.col(2);
		const Vector3r& connector1 = constraintInfo.col(3);
#ifdef VMPROTECT
		Vector3r stretchViolation = connector0 * segment0->GetZeta() - connector1 * segment1->GetZeta();
#else
		Vector3r stretchViolation = connector0 - connector1;
#endif // VMPROTEC

		// compute Darboux vector (Equation (7))
		Vector3r omega;
		computeDarbouxVector(q0, q1, currentConstraint->getAverageSegmentLength(), omega);

		// Compute bending and torsion part of constraint violation
		Vector3r bendingAndTorsionViolation = omega - currentConstraint->getRestDarbouxVector();

		// fill right hand side of the linear equation system
		const Vector6r& lambdaSum(lambdaSums[currentConstraintIndex]);
		//XBPD
		//rhs.block<3, 1>(0, 0) = -stretchViolation - Vector3r(currentConstraint->getStretchCompliance().array() * lambdaSum.block<3, 1>(0, 0).array());
		//rhs.block<3, 1>(3, 0) = -bendingAndTorsionViolation - Vector3r(currentConstraint->getBendingAndTorsionCompliance().array() * lambdaSum.block<3, 1>(3, 0).array());



#ifdef VMPROTECT
		rhs.block<3, 1>(0, 0) = -stretchViolation;
		rhs.block<3, 1>(3, 0) = -bendingAndTorsionViolation * currentConstraint->getThetaVector().y();
#else
		rhs.block<3, 1>(0, 0) = -stretchViolation;
		rhs.block<3, 1>(3, 0) = -bendingAndTorsionViolation;
#endif // VMPROTECT


		// compute G matrices
		Eigen::Matrix<Real, 4, 3> G0, G1;
		computeMatrixG(q0, G0);
		computeMatrixG(q1, G1);

		// compute stretching bending Jacobians (Equation (10) and Equation (11))
		Eigen::Matrix<Real, 3, 4> jOmega0, jOmega1;
		computeBendingAndTorsionJacobians(q0, q1, currentConstraint->getAverageSegmentLength(), jOmega0, jOmega1);


		//bendingAndTorsionJacobians[currentConstraintIndex][0] = jOmega0 * G0;
		//bendingAndTorsionJacobians[currentConstraintIndex][1] = jOmega1 * G1;

		Matrix3r jOmegaG0 = jOmega0 * G0;
		Matrix3r jOmegaG1 = jOmega1 * G1;

		BandedNode* node = nodesForward[currentConstraintIndex];

		//insert compliance
		node->D.setZero();
		const Vector3r& stretchCompliance(currentConstraint->getStretchCompliance());
		node->D(0, 0) -= stretchCompliance[0];
		node->D(1, 1) -= stretchCompliance[1];
		node->D(2, 2) -= stretchCompliance[2];

		const Vector3r& bendingAndTorsionCompliance(currentConstraint->getBendingAndTorsionCompliance());
		node->D(3, 3) -= bendingAndTorsionCompliance[0];
		node->D(4, 4) -= bendingAndTorsionCompliance[1];
		node->D(5, 5) -= bendingAndTorsionCompliance[2];

		//Forward
		{
			BandedNode* nodeFwd = nodesForward[currentConstraintIndex];
			Real sign = 1;
			int segmentIndex = 0;

			Vector3r r = constraintInfo.col(2 + segmentIndex) - segment0->Position();

			Matrix3r r_cross;
			Real crossSign(-static_cast<Real>(1.0) * sign);
			MathFunctions::crossProductMatrix(crossSign * r, r_cross);

			Eigen::DiagonalMatrix<Real, 3> upperLeft(sign, sign, sign);
			nodeFwd->J.block<3, 3>(0, 0) = upperLeft;

			Matrix3r lowerLeft(Matrix3r::Zero());
			nodeFwd->J.block<3, 3>(3, 0) = lowerLeft;

			nodeFwd->J.block<3, 3>(0, 3) = r_cross;

			//Matrix3r &lowerRight(bendingAndTorsionJacobians[nodeFwd->index][segmentIndex]);
			Matrix3r& lowerRight(jOmegaG0);
			nodeFwd->J.block<3, 3>(3, 3) = lowerRight;
			nodeFwd->JT = nodeFwd->J.transpose();
		}


		//Backward
		{
			BandedNode* nodeBwd = nodesBackward[currentConstraintIndex];
			Real sign = -1;
			int segmentIndex = 1;

			Vector3r r = constraintInfo.col(2 + segmentIndex) - segment1->Position();

			Matrix3r r_cross;
			Real crossSign(-static_cast<Real>(1.0) * sign);
			MathFunctions::crossProductMatrix(crossSign * r, r_cross);

			Eigen::DiagonalMatrix<Real, 3> upperLeft(sign, sign, sign);
			nodeBwd->J.block<3, 3>(0, 0) = upperLeft;

			Matrix3r lowerLeft(Matrix3r::Zero());
			nodeBwd->J.block<3, 3>(3, 0) = lowerLeft;

			nodeBwd->J.block<3, 3>(0, 3) = r_cross;

			//Matrix3r &lowerRight(bendingAndTorsionJacobians[nodeBwd->index][segmentIndex]);
			Matrix3r& lowerRight(jOmegaG1);
			nodeBwd->J.block<3, 3>(3, 3) = lowerRight;

			nodeBwd->JT = nodeBwd->J.transpose();
		}

	}

#ifdef  _PERF_TIMER
	std::cout << "Compute Jacobians: " << timer.get_elapsed_ms() << " ms." << std::endl;
#endif

	return true;
}

bool PBD::DirectPositionBasedSolverForStiffRods::assembleJMJT_DirectPositionBasedSolverForStiffRodsConstraintBanded(
	int startId, int count,
	const std::vector<RodConstraint*>& rodConstraints,
	std::vector<RodSegment*>& rodSegments,
	std::vector<BandedNode*>& nodesForward,
	std::vector<BandedNode*>& nodesBackward,
	std::vector<Vector6r>& RHS,
	std::vector<Vector6r>& lambdaSums,
	int& numberOfUpdates,
	std::vector<std::vector<Matrix3r>>& bendingAndTorsionJacobians,
	MatrixXr& JMJT_Global,
	MatrixXr& JMJT_Banded,
	VectorXr& RHS_Global
)
{
#ifdef  _PERF_TIMER
	plf::nanotimer timer;
	timer.start();
#endif //  _PERF_TIMER

	const int N = JMJT_Global.rows();
	const int KU = 11;
	const int KL = 11;

	const size_t rows = rodConstraints.size() * 6;
	const size_t cols = rodSegments.size() * 6;
	const int endId = startId + count;

	Matrix6r D = nodesForward[1]->D;
	//for (size_t c = 0; c < rodConstraints.size(); ++c)
	for (size_t c = startId; c < endId; ++c)
	{

		//RodConstraint* currentConstraint = rodConstraints[c];
		BandedNode* nodeFwd = nodesForward[c];
		BandedNode* nodeBwd = nodesBackward[c];

		const int row = c * 6;
		const int col = c * 6;

		RHS_Global.block<6, 1>(row, 0) = RHS[c];

		//main diagonal
		//JMJT_Global.block<6, 6>(row, row) = nodeFwd->J * nodeFwd->J.transpose() + nodeBwd->J * nodeBwd->J.transpose();
		//JMJT_Global.block<6, 6>(row, row) = nodeFwd->J * nodeFwd->J.transpose() + nodeBwd->J * nodeBwd->J.transpose();
		//JMJT_Global(row + 0, col + 0) -= D(0, 0);
		//JMJT_Global(row + 1, col + 1) -= D(1, 1);
		//JMJT_Global(row + 2, col + 2) -= D(2, 2);
		//JMJT_Global(row + 3, col + 3) -= D(3, 3);
		//JMJT_Global(row + 4, col + 4) -= D(4, 4);
		//JMJT_Global(row + 5, col + 5) -= D(5, 5);

		//Matrix6r JMJT_block = nodeFwd->J * nodeFwd->J.transpose() + nodeBwd->J * nodeBwd->J.transpose();
		Matrix6r JMJT_block = nodeFwd->J * nodeFwd->JT + nodeBwd->J * nodeBwd->JT;


		JMJT_block(0, 0) -= D(0, 0);
		JMJT_block(1, 1) -= D(1, 1);
		JMJT_block(2, 2) -= D(2, 2);
		JMJT_block(3, 3) -= D(3, 3);
		JMJT_block(4, 4) -= D(4, 4);
		JMJT_block(5, 5) -= D(5, 5);

		//in band:
		//return column - ku <= row && row <= column + kl;
		for (int j = col, v = 0; j < col + 6; j++, v++)
		{
			for (int i = row, u = 0; i < row + 6; i++, u++)
			{
				//we know that we are within bands
				//JMJT_Banded(KU - j + i + 11, j) = JMJT_Global(i, j);
				//JMJT_Banded(KU - j + i + 11, j) = JMJT_block(u, v);

				//Symmetric
				if (KU - j + i < JMJT_Banded.rows())
					JMJT_Banded(KU - j + i, j) = JMJT_block(u, v);
			}
		}

		if (c < rodConstraints.size() - 1)
		{
			BandedNode* nodeNext = nodesForward[c + 1];

			//off-diagonals - these two are almost symmetric (floating-point errors?)


			//sub-diagonal
			//JMJT_Global.block<6, 6>(row + 6, row) = nodeNext->J *  nodeBwd->J.transpose();

			JMJT_block = nodeNext->J * nodeBwd->JT;

			//for (int j = col, v = 0; j < col + 6; j++, v++)
			//{
			//	for (int i = row + 6, u = 0; i < row + 12; i++, u++)
			//	{
			//		//JMJT_Banded(KU - j + i + 11, j) = JMJT_Global(i, j);
			//		//JMJT_Banded(KU - j + i + 11, j) = JMJT_block(u, v);
			//	}
			//}



			//super-diagonal
			//JMJT_Global.block<6, 6>(row, row + 6) = nodeBwd->J * nodeNext->J.transpose();
			//JMJT_Global.block<6, 6>(row, row + 6) = (nodeNext->J *  nodeBwd->J.transpose()).transpose();
			//JMJT_block = nodeBwd->J * nodeNext->J.transpose();

			JMJT_block.transposeInPlace(); //TODO
			for (int j = col + 6, v = 0; j < col + 12; j++, v++)
			{
				for (int i = row, u = 0; i < row + 6; i++, u++)
				{
					//JMJT_Banded(KU - j + i + 11, j) = JMJT_Global(i, j);
					//JMJT_Banded(KU - j + i + 11, j) = JMJT_block(u, v);

					//Symmetric
					if (KU - j + i < JMJT_Banded.rows())
						JMJT_Banded(KU - j + i, j) = JMJT_block(u, v);
				}
			}

		}


	}

	//Matrix6r D = nodesForward[1]->D;
	//for (size_t i = 0; i < rodConstraints.size(); ++i)
	//{
	//	const int row = i * 6;
	//	const int col = i * 6;

	//	JMJT_Global(row + 0, col + 0) -= D(0, 0);
	//	JMJT_Global(row + 1, col + 1) -= D(1, 1);
	//	JMJT_Global(row + 2, col + 2) -= D(2, 2);
	//	JMJT_Global(row + 3, col + 3) -= D(3, 3);
	//	JMJT_Global(row + 4, col + 4) -= D(4, 4);
	//	JMJT_Global(row + 5, col + 5) -= D(5, 5);
	//}



#ifdef  _PERF_TIMER
	std::cout << "JMJT Assemby: " << timer.get_elapsed_ms() << " ms." << std::endl;
#endif

	return true;
}

bool PBD::DirectPositionBasedSolverForStiffRods::computeJacobiansAndAssemble_DirectPositionBasedSolverForStiffRodsConstraintBanded(
	int startId, int count,
	const std::vector<RodConstraint*>& rodConstraints,
	std::vector<RodSegment*>& rodSegments,
	std::vector<BandedNode*>& nodesForward,
	std::vector<BandedNode*>& nodesBackward,
	std::vector<Vector6r>& RHS,
	std::vector<Vector6r>& lambdaSums,
	int& numberOfUpdates,
	std::vector<std::vector<Matrix3r>>& bendingAndTorsionJacobians,
	MatrixXr& JMJT_Global,
	MatrixXr& JMJT_Banded,
	VectorXr& RHS_Global
)
{
#ifdef  _PERF_TIMER
	plf::nanotimer timer;
	timer.start();
#endif //  _PERF_TIMER

	const int N = JMJT_Global.rows();
	const int KU = 11;
	const int KL = 11;

	const size_t rows = rodConstraints.size() * 6;
	const size_t cols = rodSegments.size() * 6;


	// compute right hand side of linear equation system
	//for (size_t currentConstraintIndex = 0; currentConstraintIndex < rodConstraints.size(); ++currentConstraintIndex)
	const int endId = startId + count;
	for (size_t currentConstraintIndex = startId; currentConstraintIndex < endId; ++currentConstraintIndex)
	{
		RodConstraint* currentConstraint = rodConstraints[currentConstraintIndex];

		RodSegment* segment0 = rodSegments[currentConstraint->segmentIndex(0)];
		RodSegment* segment1 = rodSegments[currentConstraint->segmentIndex(1)];

		const Quaternionr& q0 = segment0->Rotation();
		const Quaternionr& q1 = segment1->Rotation();

		const Eigen::Matrix<Real, 3, 4>& constraintInfo(currentConstraint->getConstraintInfo());
		Vector6r& rhs(RHS[currentConstraintIndex]);

		// Compute zero-stretch part of constraint violation
		const Vector3r& connector0 = constraintInfo.col(2);
		const Vector3r& connector1 = constraintInfo.col(3);

#ifdef VMPROTECT
		Vector3r stretchViolation = connector0 * segment0->GetZeta() - connector1 * segment1->GetZeta();
#else
		Vector3r stretchViolation = connector0 - connector1;
#endif // VMPROTEC



		// compute Darboux vector (Equation (7))
		Vector3r omega;
		computeDarbouxVector(q0, q1, currentConstraint->getAverageSegmentLength(), omega);

		// Compute bending and torsion part of constraint violation
		Vector3r bendingAndTorsionViolation = omega - currentConstraint->getRestDarbouxVector();

		// fill right hand side of the linear equation system
		const Vector6r& lambdaSum(lambdaSums[currentConstraintIndex]);
		//XBPD
		//rhs.block<3, 1>(0, 0) = -stretchViolation - Vector3r(currentConstraint->getStretchCompliance().array() * lambdaSum.block<3, 1>(0, 0).array());
		//rhs.block<3, 1>(3, 0) = -bendingAndTorsionViolation - Vector3r(currentConstraint->getBendingAndTorsionCompliance().array() * lambdaSum.block<3, 1>(3, 0).array());

#ifdef VMPROTECT
		rhs.block<3, 1>(0, 0) = -stretchViolation;
		rhs.block<3, 1>(3, 0) = -bendingAndTorsionViolation * currentConstraint->getThetaVector().z();
#else
		rhs.block<3, 1>(0, 0) = -stretchViolation;
		rhs.block<3, 1>(3, 0) = -bendingAndTorsionViolation;
#endif // VMPROTEC


		// compute G matrices
		Eigen::Matrix<Real, 4, 3> G0, G1;
		computeMatrixG(q0, G0);
		computeMatrixG(q1, G1);

		// compute stretching bending Jacobians (Equation (10) and Equation (11))
		Eigen::Matrix<Real, 3, 4> jOmega0, jOmega1;
		computeBendingAndTorsionJacobians(q0, q1, currentConstraint->getAverageSegmentLength(), jOmega0, jOmega1);

		Matrix3r jOmegaG0(jOmega0 * G0);
		Matrix3r jOmegaG1(jOmega1 * G1);

		//bendingAndTorsionJacobians[currentConstraintIndex][0] = jOmega0 * G0;
		//bendingAndTorsionJacobians[currentConstraintIndex][1] = jOmega1 * G1;

		BandedNode* node = nodesForward[currentConstraintIndex];

		//insert compliance
		node->D.setZero();
		const Vector3r& stretchCompliance(currentConstraint->getStretchCompliance());
		node->D(0, 0) -= stretchCompliance[0];
		node->D(1, 1) -= stretchCompliance[1];
		node->D(2, 2) -= stretchCompliance[2];

		const Vector3r& bendingAndTorsionCompliance(currentConstraint->getBendingAndTorsionCompliance());
		node->D(3, 3) -= bendingAndTorsionCompliance[0];
		node->D(4, 4) -= bendingAndTorsionCompliance[1];
		node->D(5, 5) -= bendingAndTorsionCompliance[2];

		//Forward
		{
			BandedNode* nodeFwd = nodesForward[currentConstraintIndex];
			Real sign = 1;
			int segmentIndex = 0;

			Vector3r r = constraintInfo.col(2 + segmentIndex) - segment0->Position();

			Matrix3r r_cross;
			Real crossSign(-static_cast<Real>(1.0) * sign);
			MathFunctions::crossProductMatrix(crossSign * r, r_cross);

			Eigen::DiagonalMatrix<Real, 3> upperLeft(sign, sign, sign);
			nodeFwd->J.block<3, 3>(0, 0) = upperLeft;

			Matrix3r lowerLeft(Matrix3r::Zero());
			nodeFwd->J.block<3, 3>(3, 0) = lowerLeft;

			nodeFwd->J.block<3, 3>(0, 3) = r_cross;

			//Matrix3r &lowerRight(bendingAndTorsionJacobians[nodeFwd->index][segmentIndex]);
			Matrix3r& lowerRight(jOmegaG0);
			nodeFwd->J.block<3, 3>(3, 3) = lowerRight;
			nodeFwd->JT = nodeFwd->J.transpose();
		}


		//Backward
		{
			BandedNode* nodeBwd = nodesBackward[currentConstraintIndex];
			Real sign = -1;
			int segmentIndex = 1;

			Vector3r r = constraintInfo.col(2 + segmentIndex) - segment1->Position();

			Matrix3r r_cross;
			Real crossSign(-static_cast<Real>(1.0) * sign);
			MathFunctions::crossProductMatrix(crossSign * r, r_cross);

			Eigen::DiagonalMatrix<Real, 3> upperLeft(sign, sign, sign);
			nodeBwd->J.block<3, 3>(0, 0) = upperLeft;

			Matrix3r lowerLeft(Matrix3r::Zero());
			nodeBwd->J.block<3, 3>(3, 0) = lowerLeft;

			nodeBwd->J.block<3, 3>(0, 3) = r_cross;

			//Matrix3r &lowerRight(bendingAndTorsionJacobians[nodeBwd->index][segmentIndex]);
			Matrix3r& lowerRight(jOmegaG1);
			nodeBwd->J.block<3, 3>(3, 3) = lowerRight;

			nodeBwd->JT = nodeBwd->J.transpose();
		}

		BandedNode* nodeFwd = nodesForward[currentConstraintIndex];
		BandedNode* nodeBwd = nodesBackward[currentConstraintIndex];

		const int row = currentConstraintIndex * 6;
		const int col = currentConstraintIndex * 6;

		RHS_Global.block<6, 1>(row, 0) = RHS[currentConstraintIndex];

		Matrix6r JMJT_block = nodeFwd->J * nodeFwd->JT + nodeBwd->J * nodeBwd->JT;

		Matrix6r D = nodesForward[1]->D;
		JMJT_block(0, 0) -= D(0, 0);
		JMJT_block(1, 1) -= D(1, 1);
		JMJT_block(2, 2) -= D(2, 2);
		JMJT_block(3, 3) -= D(3, 3);
		JMJT_block(4, 4) -= D(4, 4);
		JMJT_block(5, 5) -= D(5, 5);

		for (int j = col, v = 0; j < col + 6; j++, v++)
		{
			for (int i = row, u = 0; i < row + 6; i++, u++)
			{
				if (KU - j + i < JMJT_Banded.rows())
					JMJT_Banded(KU - j + i, j) = JMJT_block(u, v);
			}
		}


		if (currentConstraintIndex < rodConstraints.size() - 1)
		{

			BandedNode* nodeNext = nodesForward[currentConstraintIndex + 1];
			RodConstraint* nextConstraint = rodConstraints[currentConstraintIndex + 1];
			RodSegment* segment2 = rodSegments[nextConstraint->segmentIndex(1)];
			const Quaternionr& q2 = segment2->Rotation();

			// compute G matrices
			Eigen::Matrix<Real, 4, 3> G2;
			computeMatrixG(q2, G2);


			// compute stretching bending Jacobians (Equation (10) and Equation (11))
			Eigen::Matrix<Real, 3, 4> jOmega1b, jOmega2b;
			computeBendingAndTorsionJacobians(q1, q2, currentConstraint->getAverageSegmentLength(), jOmega1b, jOmega2b);

			Matrix3r jOmegaG1b(jOmega1b * G1);
			Matrix3r jOmegaG2b(jOmega2b * G2);

			const Eigen::Matrix<Real, 3, 4>& constraintInfo(nextConstraint->getConstraintInfo());

			Real sign = 1;
			int segmentIndex = 0;

			Vector3r r = constraintInfo.col(2 + segmentIndex) - segment1->Position();

			Matrix3r r_cross;
			Real crossSign(-static_cast<Real>(1.0) * sign);
			MathFunctions::crossProductMatrix(crossSign * r, r_cross);

			Eigen::DiagonalMatrix<Real, 3> upperLeft(sign, sign, sign);
			nodeNext->J.block<3, 3>(0, 0) = upperLeft;

			Matrix3r lowerLeft(Matrix3r::Zero());
			nodeNext->J.block<3, 3>(3, 0) = lowerLeft;

			nodeNext->J.block<3, 3>(0, 3) = r_cross;

			Matrix3r& lowerRight(jOmegaG1b);
			nodeNext->J.block<3, 3>(3, 3) = lowerRight;

			nodeNext->JT = nodeNext->J.transpose();


			//BandedNode* nodeNext = nodesForward[currentConstraintIndex + 1];
			JMJT_block = nodeNext->J * nodeBwd->JT;
			JMJT_block.transposeInPlace(); //TODO
			for (int j = col + 6, v = 0; j < col + 12; j++, v++)
			{
				for (int i = row, u = 0; i < row + 6; i++, u++)
				{
					if (KU - j + i < JMJT_Banded.rows())
						JMJT_Banded(KU - j + i, j) = JMJT_block(u, v);
				}
			}

		}
	}

#ifdef  _PERF_TIMER
	std::cout << "Compute Jacobians: " << timer.get_elapsed_ms() << " ms." << std::endl;
#endif

	return true;
}

// -----------------------------------------------------------------------------
// Factor A = U^T * U in place on AB (upper-triangle band storage).
// On exit AB holds the upper-triangular band factor U.
// -----------------------------------------------------------------------------
inline void spbtrf_band_raw(
	int N, int KD,
	float* __restrict AB, int LDAB)
{
	//if (N < 0 || KD < 0)
	//	throw std::invalid_argument("spbtrf: N or KD < 0");
	//if (LDAB < KD + 1)
	//	throw std::invalid_argument("spbtrf: LDAB too small");
	//if ((int)AB.size() < LDAB * N)
	//	throw std::invalid_argument("spbtrf: AB size mismatch");

	for (int j = 0; j < N; ++j) 
	{
		// 1) update diagonal U[j,j]
		float sum = 0.0f;
		int kmax = (KD < j ? KD : j);
		for (int k = 1; k <= kmax; ++k) 
		{
			float u = AB[(KD - k) + j * LDAB];  // U[j-k, j]
			sum += u * u;
		}
		float Ajj = AB[KD + j * LDAB] - sum;
		if (Ajj <= 0.0f)
			throw std::runtime_error("spbtrf: matrix not positive definite");
		float Ujj = std::sqrt(Ajj);
		AB[KD + j * LDAB] = Ujj;

		// 2) compute the superdiagonals U[j, j+i], i=1..KD
		int imax = ((KD < N - j - 1) ? KD : (N - j - 1));
		for (int i = 1; i <= imax; ++i) 
		{
			float dot = 0.0f;
			int k2max = ((j < (KD - i)) ? j : (KD - i));
			for (int k = 1; k <= k2max; ++k) 
			{
				float ukj = AB[(KD - k) + j * LDAB];    // U[j-k, j]
				float ukj_i = AB[(KD - i - k) + (j + i) * LDAB]; // U[j-k, j+i]
				dot += ukj * ukj_i;
			}
			float Aji = AB[(KD - i) + (j + i) * LDAB] - dot;
			AB[(KD - i) + (j + i) * LDAB] = Aji / Ujj;
		}
	}
}

// -----------------------------------------------------------------------------
// Solve A * X = B given U from spbtrf_band (upper storage).
// Overwrites B (size LDB x NRHS) with X.
// -----------------------------------------------------------------------------
inline void spbtrs_band_raw(
	int N, int KD,
	const float* __restrict AB, int LDAB,
	float* __restrict B, int LDB,
	int NRHS)
{
	//if (N < 0 || KD < 0 || NRHS < 0)
	//	throw std::invalid_argument("spbtrs: bad dimensions");
	//if (LDAB < KD + 1 || LDB < std::max(1, N))
	//	throw std::invalid_argument("spbtrs: leading dim too small");
	//if ((int)AB.size() < LDAB * N || (int)B.size() < LDB * NRHS)
	//	throw std::invalid_argument("spbtrs: buffer size mismatch");

	for (int col = 0; col < NRHS; ++col) 
	{
		float* bcol = &B[col * LDB];

		// 1) forward solve U^T * y = bcol
		for (int i = 0; i < N; ++i) 
		{
			float sum = 0.0f;
			int k0 = (i < KD ? 0 : i - KD);
			for (int k = k0; k < i; ++k) 
			{
				sum += AB[(KD + k - i) + i * LDAB] * bcol[k];
			}
			bcol[i] = (bcol[i] - sum) / AB[KD + i * LDAB];
		}

		// 2) backward solve U * x = y
		for (int i = N - 1; i >= 0; --i) 
		{
			float sum = 0.0f;
			int k1 = ((i + KD) < N ? (i + KD) : (N - 1));
			for (int k = i + 1; k <= k1; ++k) 
			{
				sum += AB[(KD + i - k) + k * LDAB] * bcol[k];
			}
			bcol[i] = (bcol[i] - sum) / AB[KD + i * LDAB];
		}
	}
}



inline int spbsv_band_raw(
	int N, int KD, int NRHS,
	float* __restrict AB, int LDAB,
	float* __restrict B, int LDB)
{
	spbtrf_band_raw(N, KD, AB, LDAB);
	spbtrs_band_raw(N, KD, AB, LDAB, B, LDB, NRHS);
	return 0;
}

// KD=11, single-precision, single RHS
inline int spbsv_u11_1rhs(int N, 
	float* __restrict AB,   // length = 34 * N, rows 0..11 used
	float* __restrict B     // length = N, overwritten by X
) {
	constexpr int KD = 11;
	constexpr int LDAB = 34;

	// 1) In-place Cholesky on AB -> U
	for (int j = 0; j < N; ++j) 
	{
		float sum = 0.0f;
		int kmax = (j < KD ? j : KD);
		for (int k = 1; k <= kmax; ++k) 
		{
			float u = AB[(KD - k) + j * LDAB];
			sum += u * u;
		}
		//float Ajj = AB[KD + j * LDAB] - sum;
		//if (Ajj <= 0.0f) return j + 1;  // matrix not positive-definite
		//float Ujj = std::sqrt(Ajj);

		float Ajj = AB[KD + j * LDAB] - sum;
		if (Ajj <= 1e-6f) 
		{
			// Add regularization for near-singular matrices
			Ajj = 1e-6f;
		}
		float Ujj = std::sqrt(Ajj);

		AB[KD + j * LDAB] = Ujj;

		int imax = ((N - j - 1) < KD ? (N - j - 1) : KD);
		for (int i = 1; i <= imax; ++i) 
		{
			float dot = 0.0f;
			int k2max = (j < (KD - i) ? j : (KD - i));
			for (int k = 1; k <= k2max; ++k) 
			{
				dot += AB[(KD - k) + j * LDAB]
					* AB[(KD - i - k) + (j + i) * LDAB];
			}
			float Aji = AB[(KD - i) + (j + i) * LDAB] - dot;
			AB[(KD - i) + (j + i) * LDAB] = Aji / Ujj;
		}
	}

	// 2) Forward solve U^T * y = B
	for (int i = 0; i < N; ++i) 
	{
		float sum = 0.0f;
		int k0 = (i < KD ? 0 : i - KD);
		for (int k = k0; k < i; ++k) 
		{
			sum += AB[(KD + k - i) + i * LDAB] * B[k];
		}
		B[i] = (B[i] - sum) / AB[KD + i * LDAB];
	}

	// 3) Backward solve U * x = y
	for (int i = N - 1; i >= 0; --i) 
	{
		float sum = 0.0f;
		int k1 = (i + KD < N ? i + KD : N - 1);
		for (int k = i + 1; k <= k1; ++k) 
		{
			sum += AB[(KD + i - k) + k * LDAB] * B[k];
		}
		B[i] = (B[i] - sum) / AB[KD + i * LDAB];
	}

	return 0;
}



// KD=11, single-precision, single RHS
inline int spbsv_u11_1rhs_optimized(
	int    N,
	float* __restrict AB,    // length = LDAB * N; rows 0..KD used
	float* __restrict B       // length = N, overwritten by X

) {
	constexpr int KD = 11;
	constexpr int LDAB = 34;   // must be ≥ KD+1

	// 1) in-place Cholesky: AB → U
	for (int j = 0; j < N; ++j) 
	{
		float* __restrict colj = AB + j * LDAB;
		float sum = 0.0f;
		int   kmax = (j < KD ? j : KD);

//#pragma unroll
		for (int kk = 1; kk <= KD; ++kk) 
		{
			if (kk > kmax) 
				break;
			// kk ∈ [1..kmax] ⇒ idx = KD-kk ∈ [KD-1..KD-kmax] ⊆ [0..KD-1]
			int idx = KD - kk;
			assert(0 <= idx && idx <= KD);
			float u = colj[idx];
			sum += u * u;
		}

		float Ajj = colj[KD] - sum;
		//if (Ajj <= 0.0f)
		//	return j + 1;      // not positive-definite
		if (Ajj <= 1e-6f)
		{
			// Add regularization for near-singular matrices
			Ajj = 1e-6f;
		}
		float Ujj = sqrtf(Ajj);
		colj[KD] = Ujj;

		// update the KD super-diagonals in columns j+1 .. j+KD
		int imax = ((N - j - 1) < KD ? (N - j - 1) : KD);
		for (int i = 1; i <= imax; ++i) 
		{
			float* __restrict colji = AB + (j + i) * LDAB;
			float  dot = 0.0f;
			int    k2max = (j < (KD - i) ? j : (KD - i));

//#pragma unroll
			for (int kk = 1; kk <= KD; ++kk) 
			{
				if (kk > k2max) 
					break;
				
				int idxj = KD - kk;       // for colj
				int idxji = KD - i - kk;   // for colji
				assert(0 <= idxj && idxj <= KD);
				assert(0 <= idxji && idxji <= KD);
				
				dot += colj[idxj] * colji[idxji];
			}
			float Aji = colji[KD - i] - dot;
			colji[KD - i] = Aji / Ujj;
		}
	}

	// 2) forward solve Uᵀ y = B
	for (int i = 0; i < N; ++i) 
	{
		float* __restrict coli = AB + i * LDAB;
		float  sum = 0.0f;
		int    start = (i < KD ? 0 : i - KD);

//#pragma unroll
		for (int kk = 0; kk < KD; ++kk) 
		{
			int row = start + kk;
			if (row >= i) break;
			int idx = KD + row - i;
			// row < i ⇒ (row-i) ∈ [−KD..−1] ⇒ idx ∈ [0..KD-1]
			assert(0 <= idx && idx <= KD);
			sum += coli[idx] * B[row];
		}
		B[i] = (B[i] - sum) / coli[KD];
	}

	// 3) backward solve U x = y
	for (int i = N - 1; i >= 0; --i) 
	{
		float* __restrict coli = AB + i * LDAB;
		float  sum = 0.0f;
		int    colmax = (i + KD < N ? i + KD : N - 1);

//#pragma unroll
		for (int kk = 1; kk <= KD; ++kk) 
		{
			int col = i + kk;
			if (col > colmax) break;
			float* __restrict colk = AB + col * LDAB;
			int    idx = KD + i - col;
			// col > i ⇒ (i-col) ∈ [−KD..−1] ⇒ idx ∈ [0..KD-1]
			assert(0 <= idx && idx <= KD);
			sum += colk[idx] * B[col];
		}
		B[i] = (B[i] - sum) / coli[KD];
	}

	return 0;
}




bool PBD::DirectPositionBasedSolverForStiffRods::solveJMJT_DirectPositionBasedSolverForStiffRodsConstraintBanded(
	const std::vector<RodConstraint*>& rodConstraints,
	std::vector<RodSegment*>& rodSegments,
	std::vector<BandedNode*>& nodesForward,
	std::vector<BandedNode*>& nodesBackward,
	std::vector<Vector6r>& RHS,
	std::vector<Vector6r>& lambdaSums,
	int& numberOfUpdates,
	std::vector<std::vector<Matrix3r>>& bendingAndTorsionJacobians,
	MatrixXr& JMJT_Global,
	MatrixXr& JMJT_Banded,
	VectorXr& RHS_Global,
	std::vector<Vector3r>& corr_x,
	std::vector<Quaternionr>& corr_q
)
{

#ifdef  _PERF_TIMER
	plf::nanotimer timer;
	timer.start();
#endif //  _PERF_TIMER

	//std::cout << "JMJT_Global---------------------------------------------------------------------------------------------" << std::endl;
	//std::cout << JMJT_Global.leftCols(64) << std::endl;

	//std::cout << "JMJT_Banded---------------------------------------------------------------------------------------------" << std::endl;
	//std::cout << JMJT_Banded.rightCols(16) << std::endl;
	//std::cout << JMJT_Banded << std::endl;


	const int N = JMJT_Global.rows();
	const int KU = 11;
	const int KL = 11;

	const int NRHS = 1;
	const int LDA = N;
	const int LDB = N;
	const int LWORK = 8;

	//int* ipiv = new int[N];
	int info;
	//float* work = new float[LWORK];

	//SGESV(&N, &NRHS, JMJT_Global.data(), &LDA, ipiv, RHS_Global.data(), &LDB, &info);
	//SSYSV("U", &N, &NRHS, JMJT_Global.data(), &LDA, ipiv, RHS_Global.data(), &LDB, work, &LWORK, info);

	//const int LDAB = 34;
	//SGBSV(&N, &KL, &KU, &NRHS, JMJT_Banded.data(), &LDAB, ipiv, RHS_Global.data(), &LDB, &info);

	//MatrixXr JMJT_PBSV(JMJT_Banded.topRows(12));
	//const int PBSV_N = JMJT_PBSV.cols();
	//const int PBSV_LDA = JMJT_PBSV.rows();
	//SPBSV("U", &PBSV_N, &KU, &NRHS, JMJT_PBSV.data(), &PBSV_LDA, RHS_Global.data(), &LDB, &info);
	//std::cout << "JMJT_PBSV---------------------------------------------------------------------------------------------" << std::endl;
	//std::cout << JMJT_PBSV << std::endl;

	const int PBSV_N = JMJT_Banded.cols();
	const int PBSV_LDA = JMJT_Banded.rows();
	//SPBSV("U", &PBSV_N, &KU, &NRHS, JMJT_Banded.data(), &PBSV_LDA, RHS_Global.data(), &LDB, &info);
	//LAPACK_spbsv("U", &PBSV_N, &KU, &NRHS, JMJT_Banded.data(), &PBSV_LDA, RHS_Global.data(), &LDB, &info);
	//delete ipiv;
	//delete work;

	//seems faster than optimized
	info = spbsv_u11_1rhs(
		PBSV_N,                  // N
		JMJT_Banded.data(),      // AB (float*), size = 34 * N
		RHS_Global.data()        // B  (float*), size = N
	);

	//spbsv_u11_1rhs_optimized(
	//	PBSV_N,                  // N
	//	JMJT_Banded.data(),      // AB (float*), size = 34 * N
	//	RHS_Global.data()        // B  (float*), size = N
	//);
	


	//info = spbsv_band_raw(
	//	/* N     = */ N,
	//	/* KD    = */ KL,
	//	/* NRHS  = */ NRHS,
	//	/* AB    = */ JMJT_Banded.data(),  // <-- pointer to your band storage
	//	/* LDAB  = */ PBSV_LDA,
	//	/* B     = */ RHS_Global.data(),   // <-- pointer to your RHS vector
	//	/* LDB   = */ LDB
	//);

#ifdef  _PERF_TIMER
	std::cout << "SOLVE: " << timer.get_elapsed_ms() << " ms." << std::endl;
#endif

	//std::cout << "INFO: " << info << std::endl;
	if (info != 0)
	{
		std::cout << "LINEAR SOLVE ERROR!  INFO" << info << std::endl;
		return false;
	}

#ifdef  _PERF_TIMER
	timer.start();
#endif

#ifdef VMPROTECT
	Real zeta = 1.00001f;
	if (numberOfUpdates++ % 36000 == 0)
	{
		Real zeta = VMProtectGetSerialNumberState() == 0 ? 1.00001f : 0.001f;
		//std::cout << "Zeta check" << zeta << std::endl;
		for (size_t i = 0; i < rodSegments.size(); i++)
		{
			RodSegment* seg = rodSegments[i];
			seg->SetZeta(zeta);
		}
	}
#else
	Real zeta = 1.00001f;

#endif // 


	VectorXr deltaLambda = RHS_Global;


	for (size_t i = 0; i < rodSegments.size(); ++i)
	{
		//Vector3r deltaLambdaStretch(JT_deltaLambda.block<3, 1>(i * 6, 0));
		//Vector3r deltaLambdaBendingAndTorsion(JT_deltaLambda.block<3, 1>( i * 6 + 3, 0));
		Vector6r JT_deltaLambda;
		if (i == 0)
		{
			//JT_deltaLambda2.block<6, 1>(0, 0) = nodesForward[i]->J.transpose() * deltaLambda.block<6, 1>(0, 0);
			//JT_deltaLambda = nodesForward[i]->J.transpose() * deltaLambda.block<6, 1>(0, 0);
			JT_deltaLambda = nodesForward[i]->JT * deltaLambda.block<6, 1>(0, 0);
		}
		else if (i < rodSegments.size() - 1)
		{
			//JT_deltaLambda2.block<6, 1>(i*6, 0) = nodesBackward[i-1]->J.transpose() * deltaLambda.block<6, 1>((i-1) * 6, 0) + nodesForward[i]->J.transpose() * deltaLambda.block<6, 1>(i*6, 0);
			//JT_deltaLambda = nodesBackward[i - 1]->J.transpose() * deltaLambda.block<6, 1>((i - 1) * 6, 0) + nodesForward[i]->J.transpose() * deltaLambda.block<6, 1>(i * 6, 0);
			JT_deltaLambda = nodesBackward[i - 1]->JT * deltaLambda.block<6, 1>((i - 1) * 6, 0) + nodesForward[i]->JT * deltaLambda.block<6, 1>(i * 6, 0);
		}
		else
		{
			//JT_deltaLambda2.block<6, 1>(i * 6, 0) = nodesBackward[i - 1]->J.transpose() * deltaLambda.block<6, 1>((i - 1) * 6, 0);
			//JT_deltaLambda = nodesBackward[i - 1]->J.transpose() * deltaLambda.block<6, 1>((i - 1) * 6, 0);
			JT_deltaLambda = nodesBackward[i - 1]->JT * deltaLambda.block<6, 1>((i - 1) * 6, 0);
		}

		Vector3r deltaLambdaStretch = JT_deltaLambda.block<3, 1>(0, 0);
		Vector3r deltaLambdaBendingAndTorsion = JT_deltaLambda.block<3, 1>(3, 0);

		RodSegment* segment = rodSegments[i];
		segment->SetZeta(zeta);
		if (segment->Mass() != 0.0f)
		{
#ifdef VMPROTECT
			corr_x[i] += segment->Mass() * segment->GetZeta() * deltaLambdaStretch;
#else
			corr_x[i] += segment->Mass() * deltaLambdaStretch;
#endif
			//corr_q[i].coeffs() += G0 * (inertiaInverseW0 * ra_crossT * (-1 * deltaLambdaStretch) + MInvJT0 * deltaLambdaBendingAndTorsion);

			Quaternionr deltaQSoln;
			Eigen::Matrix<Real, 4, 3> G;
			computeMatrixG(segment->Rotation(), G);
			deltaQSoln.coeffs() = G * deltaLambdaBendingAndTorsion;
			corr_q[i] = deltaQSoln;
			//corr_q[i].coeffs() += G * deltaLambdaBendingAndTorsion;

		}
	}


#ifdef  _PERF_TIMER
	std::cout << "JT_MULT: " << timer.get_elapsed_ms() << " ms." << std::endl;
#endif
	return true;
}



bool PBD::DirectPositionBasedSolverForStiffRods::init_StretchBendingTwistingConstraint(
	const Vector3r& x0, const Quaternionr& q0,
	const Vector3r& x1, const Quaternionr& q1,
	const Vector3r& constraintPosition,
	const Real averageRadius,
	const Real averageSegmentLength,
	const Real youngsModulus,
	const Real torsionModulus,
	Eigen::Matrix<Real, 3, 4>& constraintInfo,
	Vector3r& stiffnessCoefficientK,
	Vector3r& restDarbouxVector
)
{
	// constraintInfo contains
	// 0:	connector in segment 0 (local)
	// 1:	connector in segment 1 (local)
	// 2:	connector in segment 0 (global)
	// 3:	connector in segment 1 (global)

	// transform in local coordinates
	const Matrix3r rot0T = q0.matrix().transpose();
	const Matrix3r rot1T = q1.matrix().transpose();

	constraintInfo.col(0) = rot0T * (constraintPosition - x0);
	constraintInfo.col(1) = rot1T * (constraintPosition - x1);
	constraintInfo.col(2) = constraintPosition;
	constraintInfo.col(3) = constraintPosition;

	// compute bending and torsion stiffness of the K matrix diagonal; assumption: the rod axis follows the y-axis of the local frame as with Blender's armatures
	Real secondMomentOfArea(static_cast<Real>(M_PI_4) * std::pow(averageRadius, static_cast<Real>(4.0)));
	Real bendingStiffness(youngsModulus * secondMomentOfArea);
	Real torsionStiffness(static_cast<Real>(2.0) * torsionModulus * secondMomentOfArea);
	stiffnessCoefficientK = Vector3r(bendingStiffness, torsionStiffness, bendingStiffness);

	// compute rest Darboux vector
	computeDarbouxVector(q0, q1, averageSegmentLength, restDarbouxVector);

	return true;
}

bool PBD::DirectPositionBasedSolverForStiffRods::initBeforeProjection_StretchBendingTwistingConstraint(
	const Vector3r& stiffnessCoefficientK,
	const Real inverseTimeStepSize,
	const Real averageSegmentLength,
	Vector3r& stretchCompliance,
	Vector3r& bendingAndTorsionCompliance,
	Vector6r& lambdaSum
)
{

	Real inverseTSQuadratic(inverseTimeStepSize * inverseTimeStepSize);

	// compute compliance parameter of the stretch constraint part
	//const Real stretchRegularizationParameter(static_cast<Real>(1.E-10));
	const Real stretchRegularizationParameter(static_cast<Real>(1.E-12));
	stretchCompliance <<
		stretchRegularizationParameter * inverseTSQuadratic,
		stretchRegularizationParameter* inverseTSQuadratic,
		stretchRegularizationParameter* inverseTSQuadratic;

	// compute compliance parameter of the bending and torsion constraint part
	bendingAndTorsionCompliance <<
		inverseTSQuadratic / stiffnessCoefficientK(0),
		inverseTSQuadratic / stiffnessCoefficientK(1),
		inverseTSQuadratic / stiffnessCoefficientK(2);

	bendingAndTorsionCompliance *= static_cast<Real>(1.0) / averageSegmentLength;

	// set sum of delta lambda values to zero
	lambdaSum.setZero();
	return true;
}

bool PBD::DirectPositionBasedSolverForStiffRods::update_StretchBendingTwistingConstraint(
	const Vector3r& x0, const Quaternionr& q0,
	const Vector3r& x1, const Quaternionr& q1,
	Eigen::Matrix<Real, 3, 4>& constraintInfo
)
{
	// constraintInfo contains
	// 0:	connector in segment 0 (local)
	// 1:	connector in segment 1 (local)
	// 2:	connector in segment 0 (global)
	// 3:	connector in segment 1 (global)

	// compute world space positions of connectors
	const Matrix3r rot0 = q0.matrix();
	const Matrix3r rot1 = q1.matrix();
	constraintInfo.col(2) = rot0 * constraintInfo.col(0) + x0;
	constraintInfo.col(3) = rot1 * constraintInfo.col(1) + x1;

	return true;
}

bool PBD::DirectPositionBasedSolverForStiffRods::solve_StretchBendingTwistingConstraint(
	const Real invMass0,
	const Vector3r& x0,
	const Matrix3r& inertiaInverseW0,
	const Quaternionr& q0,
	const Real invMass1,
	const Vector3r& x1,
	const Matrix3r& inertiaInverseW1,
	const Quaternionr& q1,
	const Vector3r& restDarbouxVector,
	const Real averageSegmentLength,
	const Vector3r& stretchCompliance,
	const Vector3r& bendingAndTorsionCompliance,
	const Eigen::Matrix<Real, 3, 4>& constraintInfo,
	Vector3r& corr_x0, Quaternionr& corr_q0,
	Vector3r& corr_x1, Quaternionr& corr_q1,
	Vector6r& lambdaSum
)
{
	// compute Darboux vector (Equation (7))
	Vector3r omega;
	computeDarbouxVector(q0, q1, averageSegmentLength, omega);

	// compute bending and torsion Jacobians (Equation (10) and Equation (11))
	Eigen::Matrix<Real, 3, 4> jOmega0, jOmega1;
	computeBendingAndTorsionJacobians(q0, q1, averageSegmentLength, jOmega0, jOmega1);

	// compute G matrices (Equation (27))
	Eigen::Matrix<Real, 4, 3> G0, G1;
	computeMatrixG(q0, G0);
	computeMatrixG(q1, G1);

	Matrix3r jOmegaG0(jOmega0 * G0);
	Matrix3r jOmegaG1(jOmega1 * G1);

	// Compute zero-stretch part of constraint violation (Equation (23))
	const Vector3r& connector0 = constraintInfo.col(2);
	const Vector3r& connector1 = constraintInfo.col(3);
	Vector3r stretchViolation = connector0 - connector1;

	// Compute bending and torsion part of constraint violation  (Equation (23))
	Vector3r bendingAndTorsionViolation = omega - restDarbouxVector;

	// fill right hand side of the linear equation system (Equation (19))
	Vector6r rhs;
	rhs.block<3, 1>(0, 0) = -stretchViolation - Vector3r(stretchCompliance.array() * lambdaSum.block<3, 1>(0, 0).array());

	rhs.block<3, 1>(3, 0) = -bendingAndTorsionViolation - Vector3r(bendingAndTorsionCompliance.array() * lambdaSum.block<3, 1>(3, 0).array());

	// compute matrix of the linear equation system (using Equations (25), (26), and (28) in Equation (19))
	Matrix6r JMJT(Matrix6r::Zero());

	// compute stretch block
	Matrix3r K1, K2;
	computeMatrixK(connector0, invMass0, x0, inertiaInverseW0, K1);
	computeMatrixK(connector1, invMass1, x1, inertiaInverseW1, K2);
	JMJT.block<3, 3>(0, 0) = K1 + K2;

	// compute coupling blocks
	const Vector3r ra = connector0 - x0;
	const Vector3r rb = connector1 - x1;

	Matrix3r ra_crossT, rb_crossT;
	MathFunctions::crossProductMatrix(-ra, ra_crossT); // use -ra to get the transpose
	MathFunctions::crossProductMatrix(-rb, rb_crossT); // use -rb to get the transpose

	Matrix3r offdiag(Matrix3r::Zero());
	if (invMass0 != 0.0)
	{
		offdiag = jOmegaG0 * inertiaInverseW0 * ra_crossT * (-1);
	}

	if (invMass1 != 0.0)
	{
		offdiag += jOmegaG1 * inertiaInverseW1 * rb_crossT;
	}
	JMJT.block<3, 3>(3, 0) = offdiag;
	JMJT.block<3, 3>(0, 3) = offdiag.transpose();

	// compute bending and torsion block
	Matrix3r MInvJT0(inertiaInverseW0 * jOmegaG0.transpose());
	Matrix3r MInvJT1(inertiaInverseW1 * jOmegaG1.transpose());

	Matrix3r JMJTOmega(Matrix3r::Zero());
	if (invMass0 != 0.0)
	{
		JMJTOmega = jOmegaG0 * MInvJT0;
	}

	if (invMass1 != 0.0)
	{
		JMJTOmega += jOmegaG1 * MInvJT1;
	}
	JMJT.block<3, 3>(3, 3) = JMJTOmega;

	// add compliance
	JMJT(0, 0) += stretchCompliance(0);
	JMJT(1, 1) += stretchCompliance(1);
	JMJT(2, 2) += stretchCompliance(2);
	JMJT(3, 3) += bendingAndTorsionCompliance(0);
	JMJT(4, 4) += bendingAndTorsionCompliance(1);
	JMJT(5, 5) += bendingAndTorsionCompliance(2);

	// solve linear equation system (Equation 19)
	auto decomposition(JMJT.ldlt());
	Vector6r deltaLambda(decomposition.solve(rhs));

	// update sum of delta lambda values for next Gauss-Seidel solver iteration step
	lambdaSum += deltaLambda;

	// compute position and orientation updates (using Equations (25), (26), and (28) in Equation (20))
	Vector3r deltaLambdaStretch(deltaLambda.block<3, 1>(0, 0));
	Vector3r deltaLambdaBendingAndTorsion(deltaLambda.block<3, 1>(3, 0));
	corr_x0.setZero();
	corr_x1.setZero();
	corr_q0.coeffs().setZero();
	corr_q1.coeffs().setZero();

	if (invMass0 != 0.)
	{
		corr_x0 += invMass0 * deltaLambdaStretch;
		corr_q0.coeffs() += G0 * (inertiaInverseW0 * ra_crossT * (-1 * deltaLambdaStretch) + MInvJT0 * deltaLambdaBendingAndTorsion);
	}

	if (invMass1 != 0.)
	{
		corr_x1 -= invMass1 * deltaLambdaStretch;
		corr_q1.coeffs() += G1 * (inertiaInverseW1 * rb_crossT * deltaLambdaStretch + MInvJT1 * deltaLambdaBendingAndTorsion);
	}

	return true;
}
