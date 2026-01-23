#include "PositionBasedRigidBodyDynamics.h"
#include "MathFunctions.h"
#include <cfloat>
#include <iostream>
#define _USE_MATH_DEFINES
#include "math.h"

using namespace PBD;

// ----------------------------------------------------------------------------------------------
void PositionBasedRigidBodyDynamics::computeMatrixK(
	const Vector3r &connector,
	const Real invMass,
	const Vector3r &x,
	const Matrix3r &inertiaInverseW,
	Matrix3r &K)
{
	if (invMass != 0.0)
	{
		const Vector3r v = connector - x;
		const Real a = v[0];
		const Real b = v[1];
		const Real c = v[2];

		// J is symmetric
		const Real j11 = inertiaInverseW(0,0);
		const Real j12 = inertiaInverseW(0,1);
		const Real j13 = inertiaInverseW(0,2);
		const Real j22 = inertiaInverseW(1,1);
		const Real j23 = inertiaInverseW(1,2);
		const Real j33 = inertiaInverseW(2,2);

		K(0,0) = c*c*j22 - b*c*(j23 + j23) + b*b*j33 + invMass;
		K(0,1) = -(c*c*j12) + a*c*j23 + b*c*j13 - a*b*j33;
		K(0,2) = b*c*j12 - a*c*j22 - b*b*j13 + a*b*j23;
		K(1,0) = K(0,1);
		K(1,1) = c*c*j11 - a*c*(j13 + j13) + a*a*j33 + invMass;
		K(1,2) = -(b*c*j11) + a*c*j12 + a*b*j13 - a*a*j23;
		K(2,0) = K(0,2);
		K(2,1) = K(1,2);
		K(2,2) = b*b*j11 - a*b*(j12 + j12) + a*a*j22 + invMass;
	}
	else
		K.setZero();
}

// ----------------------------------------------------------------------------------------------
void PositionBasedRigidBodyDynamics::computeMatrixK(
	const Vector3r &connector0,
	const Vector3r &connector1,
	const Real invMass,
	const Vector3r &x,
	const Matrix3r &inertiaInverseW,
	Matrix3r &K)
{
	if (invMass != 0.0)
	{
		const Vector3r v0 = connector0 - x;
		const Real a = v0[0];
		const Real b = v0[1];
		const Real c = v0[2];

		const Vector3r v1 = connector1 - x;
		const Real d = v1[0];
		const Real e = v1[1];
		const Real f = v1[2];

		// J is symmetric
		const Real j11 = inertiaInverseW(0, 0);
		const Real j12 = inertiaInverseW(0, 1);
		const Real j13 = inertiaInverseW(0, 2);
		const Real j22 = inertiaInverseW(1, 1);
		const Real j23 = inertiaInverseW(1, 2);
		const Real j33 = inertiaInverseW(2, 2);

		K(0, 0) = c*f*j22 - c*e*j23 - b*f*j23 + b*e*j33 + invMass;
		K(0, 1) = -(c*f*j12) + c*d*j23 + b*f*j13 - b*d*j33;
		K(0, 2) = c*e*j12 - c*d*j22 - b*e*j13 + b*d*j23;
		K(1, 0) = -(c*f*j12) + c*e*j13 + a*f*j23 - a*e*j33;
		K(1, 1) = c*f*j11 - c*d*j13 - a*f*j13 + a*d*j33 + invMass;
		K(1, 2) = -(c*e*j11) + c*d*j12 + a*e*j13 - a*d*j23;
		K(2, 0) = b*f*j12 - b*e*j13 - a*f*j22 + a*e*j23;
		K(2, 1) = -(b*f*j11) + b*d*j13 + a*f*j12 - a*d*j23;
		K(2, 2) = b*e*j11 - b*d*j12 - a*e*j12 + a*d*j22 + invMass;
	}
	else
		K.setZero();
}


// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::init_BallJoint(
	const Vector3r &x0, 						
	const Quaternionr &q0,					
	const Vector3r &x1, 						
	const Quaternionr &q1,					
	const Vector3r &ballJointPosition,		
	Eigen::Matrix<Real, 3, 4> &ballJointInfo
	)
{
	// jointInfo contains
	// 0:	connector in body 0 (local)
	// 1:	connector in body 1 (local)
	// 2:	connector in body 0 (global)
	// 3:	connector in body 1 (global)

	// transform in local coordinates
	const Matrix3r rot0T = q0.matrix().transpose();
	const Matrix3r rot1T = q1.matrix().transpose();

	ballJointInfo.col(0) = rot0T * (ballJointPosition - x0);
	ballJointInfo.col(1) = rot1T * (ballJointPosition - x1);
	ballJointInfo.col(2) = ballJointPosition;
	ballJointInfo.col(3) = ballJointPosition;

	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::update_BallJoint(
	const Vector3r &x0,
	const Quaternionr &q0,
	const Vector3r &x1,
	const Quaternionr &q1,
	Eigen::Matrix<Real, 3, 4> &ballJointInfo
	)
{
	// jointInfo contains
	// 0:	connector in body 0 (local)
	// 1:	connector in body 1 (local)
	// 2:	connector in body 0 (global)
	// 3:	connector in body 1 (global)

	// compute world space positions of connectors
	const Matrix3r rot0 = q0.matrix();
	const Matrix3r rot1 = q1.matrix();
	ballJointInfo.col(2) = rot0 * ballJointInfo.col(0) + x0;
	ballJointInfo.col(3) = rot1 * ballJointInfo.col(1) + x1;

	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::solve_BallJoint(
	const Real invMass0,
	const Vector3r &x0, 						
	const Matrix3r &inertiaInverseW0,		
	const Quaternionr &q0,					
	const Real invMass1,
	const Vector3r &x1, 						
	const Matrix3r &inertiaInverseW1,		
	const Quaternionr &q1,	
	const Eigen::Matrix<Real, 3, 4> &ballJointInfo,
	Vector3r &corr_x0, Quaternionr &corr_q0,
	Vector3r &corr_x1, Quaternionr &corr_q1)
{
	// jointInfo contains
	// 0:	connector in body 0 (local)
	// 1:	connector in body 1 (local)
	// 2:	connector in body 0 (global)
	// 3:	connector in body 1 (global)

	const Vector3r &connector0 = ballJointInfo.col(2);
	const Vector3r &connector1 = ballJointInfo.col(3);

	// Compute Kinv
	Matrix3r K1, K2;
	computeMatrixK(connector0, invMass0, x0, inertiaInverseW0, K1);
	computeMatrixK(connector1, invMass1, x1, inertiaInverseW1, K2);
	const Matrix3r Kinv = (K1 + K2).inverse();

	const Vector3r pt = Kinv * (connector1 - connector0);

	if (invMass0 != 0.0)
	{
		const Vector3r r0 = connector0 - x0;
		corr_x0 = invMass0*pt;

		const Vector3r ot = (inertiaInverseW0 * (r0.cross(pt)));
		const Quaternionr otQ(0.0, ot[0], ot[1], ot[2]);
		corr_q0.coeffs() = 0.5 *(otQ*q0).coeffs();
	}

	if (invMass1 != 0.0)
	{
		const Vector3r r1 = connector1 - x1;
		corr_x1 = -invMass1*pt;

		const Vector3r ot = (inertiaInverseW1 * (r1.cross(-pt)));
		const Quaternionr otQ(0.0, ot[0], ot[1], ot[2]);
		corr_q1.coeffs() = 0.5 *(otQ*q1).coeffs();
	}

	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::init_Spring(
	const Vector3r &x0,
	const Quaternionr &q0,
	const Vector3r &x1,
	const Quaternionr &q1,
	const Vector3r &pos0,
	const Vector3r &pos1,
	Eigen::Matrix<Real, 3, 4> &jointInfo
)
{
	// jointInfo contains
	// 0:	connector in body 0 (local)
	// 1:	connector in body 1 (local)
	// 2:	connector in body 0 (global)
	// 3:	connector in body 1 (global)

	// transform in local coordinates
	const Matrix3r rot0T = q0.matrix().transpose();
	const Matrix3r rot1T = q1.matrix().transpose();

	jointInfo.col(0) = rot0T * (pos0 - x0);
	jointInfo.col(1) = rot1T * (pos1 - x1);
	jointInfo.col(2) = pos0;
	jointInfo.col(3) = pos1;

	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::update_Spring(
	const Vector3r &x0,
	const Quaternionr &q0,
	const Vector3r &x1,
	const Quaternionr &q1,
	Eigen::Matrix<Real, 3, 4> &jointInfo
)
{
	// jointInfo contains
	// 0:	connector in body 0 (local)
	// 1:	connector in body 1 (local)
	// 2:	connector in body 0 (global)
	// 3:	connector in body 1 (global)

	// compute world space positions of connectors
	const Matrix3r rot0 = q0.matrix();
	const Matrix3r rot1 = q1.matrix();
	jointInfo.col(2) = rot0 * jointInfo.col(0) + x0;
	jointInfo.col(3) = rot1 * jointInfo.col(1) + x1;

	return true;
}


// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::solve_Spring(
	const Real invMass0,
	const Vector3r &x0,
	const Matrix3r &inertiaInverseW0,
	const Quaternionr &q0,
	const Real invMass1,
	const Vector3r &x1,
	const Matrix3r &inertiaInverseW1,
	const Quaternionr &q1,
	const Real stiffness,
	const Real restLength,
	const Eigen::Matrix<Real, 3, 4> &jointInfo,
	Vector3r &corr_x0, Quaternionr &corr_q0,
	Vector3r &corr_x1, Quaternionr &corr_q1)
{
	// jointInfo contains
	// 0:	connector in body 0 (local)
	// 1:	connector in body 1 (local)
	// 2:	connector in body 0 (global)
	// 3:	connector in body 1 (global)

	const Vector3r &connector0 = jointInfo.col(2);
	const Vector3r &connector1 = jointInfo.col(3);

	Vector3r dir = connector1 - connector0;
	Real length = dir.norm();
	dir = (1.0 / length) * dir;

	const Vector3r pt = stiffness * (length- restLength) * dir;

	corr_x0.setZero();
	corr_q0.coeffs().setZero();
	corr_x1.setZero();
	corr_q1.coeffs().setZero();

	if (invMass0 != 0.0)
	{
		const Vector3r r0 = connector0 - x0;
		corr_x0 = invMass0*pt;

		const Vector3r ot = (inertiaInverseW0 * (r0.cross(pt)));
		const Quaternionr otQ(0.0, ot[0], ot[1], ot[2]);
		corr_q0.coeffs() = 0.5 *(otQ*q0).coeffs();
	}

	if (invMass1 != 0.0)
	{
		const Vector3r r1 = connector1 - x1;
		corr_x1 = -invMass1*pt;

		const Vector3r ot = (inertiaInverseW1 * (r1.cross(-pt)));
		const Quaternionr otQ(0.0, ot[0], ot[1], ot[2]);
		corr_q1.coeffs() = 0.5 *(otQ*q1).coeffs();
	}
	return true;
}


// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::init_BallOnLineJoint(
	const Vector3r &x0,
	const Quaternionr &q0,
	const Vector3r &x1,
	const Quaternionr &q1,
	const Vector3r &position,
	const Vector3r &direction,
	Eigen::Matrix<Real, 3, 10> &jointInfo
	)
{
	// jointInfo contains
	// 0:	connector in body 0 (local)
	// 1:	connector in body 1 (local)
	// 2-4:	coordinate system of body 0 (local)
	// 5:	connector in body 0 (global)
	// 6:	connector in body 1 (global)
	// 7-9:	coordinate system of body 0 (global)

	// transform in local coordinates
	const Matrix3r rot0T = q0.matrix().transpose();
	const Matrix3r rot1T = q1.matrix().transpose();

	jointInfo.col(0) = rot0T * (position - x0);
	jointInfo.col(1) = rot1T * (position - x1);
	jointInfo.col(5) = position;
	jointInfo.col(6) = position;

	// determine constraint coordinate system
	// with direction as x-axis
	jointInfo.col(7) = direction;
	jointInfo.col(7).normalize();

	Vector3r v(1.0, 0.0, 0.0);
	// check if vectors are parallel
	if (fabs(v.dot(jointInfo.col(7))) > 0.99)
		v = Vector3r(0.0, 1.0, 0.0);

	jointInfo.col(8) = jointInfo.col(7).cross(v);
	jointInfo.col(9) = jointInfo.col(7).cross(jointInfo.col(8));
	jointInfo.col(8).normalize();
	jointInfo.col(9).normalize();

	jointInfo.block<3, 3>(0, 2) = rot0T * jointInfo.block<3, 3>(0, 7);

	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::update_BallOnLineJoint(
	const Vector3r &x0,
	const Quaternionr &q0,
	const Vector3r &x1,
	const Quaternionr &q1,
	Eigen::Matrix<Real, 3, 10> &jointInfo
	)
{
	// jointInfo contains
	// 0:	connector in body 0 (local)
	// 1:	connector in body 1 (local)
	// 2-4:	coordinate system of body 0 (local)
	// 5:	connector in body 0 (global)
	// 6:	connector in body 1 (global)
	// 7-9:	coordinate system of body 0 (global)

	// compute world space positions of connectors
	const Matrix3r rot0 = q0.matrix();
	const Matrix3r rot1 = q1.matrix();
	jointInfo.col(5) = rot0 * jointInfo.col(0) + x0;
	jointInfo.col(6) = rot1 * jointInfo.col(1) + x1;

	// transform constraint coordinate system to world space
	jointInfo.block<3, 3>(0, 7) = rot0 * jointInfo.block<3, 3>(0, 2);

	const Vector3r dir = jointInfo.col(7);
	const Vector3r p = jointInfo.col(5);
	const Vector3r s = jointInfo.col(6);
	// move the joint point of body 0 to the closest point on the line to joint point 1
	jointInfo.col(5) = p + (dir * (((s - p).dot(dir)) / dir.squaredNorm()));

	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::solve_BallOnLineJoint(
	const Real invMass0,
	const Vector3r &x0,
	const Matrix3r &inertiaInverseW0,
	const Quaternionr &q0,
	const Real invMass1,
	const Vector3r &x1,
	const Matrix3r &inertiaInverseW1,
	const Quaternionr &q1,
	const Eigen::Matrix<Real, 3, 10> &jointInfo,
	Vector3r &corr_x0, Quaternionr &corr_q0,
	Vector3r &corr_x1, Quaternionr &corr_q1)
{
	// jointInfo contains
	// 0:	connector in body 0 (local)
	// 1:	connector in body 1 (local)
	// 2-4:	coordinate system of body 0 (local)
	// 5:	connector in body 0 (global)
	// 6:	connector in body 1 (global)
	// 7-9:	coordinate system of body 0 (global)

	const Vector3r &connector0 = jointInfo.col(5);
	const Vector3r &connector1 = jointInfo.col(6);

	// Compute Kinv
	Matrix3r K1, K2;
	computeMatrixK(connector0, invMass0, x0, inertiaInverseW0, K1);
	computeMatrixK(connector1, invMass1, x1, inertiaInverseW1, K2);

	// projection 
	const Eigen::Matrix<Real, 3, 2> PT = jointInfo.block<3, 2>(0, 8);
	const Eigen::Matrix<Real, 2, 3> P = PT.transpose();

	const Matrix2r K = P * (K1 + K2) * PT;
	const Matrix2r Kinv = K.inverse();

	const Vector2r pt2D = Kinv * (P * (connector1 - connector0));
	const Vector3r pt = PT * pt2D;

	if (invMass0 != 0.0)
	{
		const Vector3r r0 = connector0 - x0;
		corr_x0 = invMass0*pt;

		const Vector3r ot = (inertiaInverseW0 * (r0.cross(pt)));
		const Quaternionr otQ(0.0, ot[0], ot[1], ot[2]);
		corr_q0.coeffs() = 0.5 *(otQ*q0).coeffs();
	}

	if (invMass1 != 0.0)
	{
		const Vector3r r1 = connector1 - x1;
		corr_x1 = -invMass1*pt;

		const Vector3r ot = (inertiaInverseW1 * (r1.cross(-pt)));
		const Quaternionr otQ(0.0, ot[0], ot[1], ot[2]);
		corr_q1.coeffs() = 0.5 *(otQ*q1).coeffs();
	}

	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::init_HingeJoint(
	const Vector3r &x0,
	const Quaternionr &q0,
	const Vector3r &x1,
	const Quaternionr &q1,
	const Vector3r &position,
	const Vector3r &direction,
	Eigen::Matrix<Real, 3, 12> &jointInfo
	)
{
	// jointInfo contains
	// 0:	connector in body 0 (local)
	// 1:	connector in body 1 (local)
	// 2-4:	coordinate system of body 0 (local)
	// 5:	joint axis in body 1 (local)
	// 6:	connector in body 0 (global)
	// 7:	connector in body 1 (global)
	// 8-10:coordinate system of body 0 (global)
	// 11:	joint axis in body 1 (global)

	// transform in local coordinates
	const Matrix3r rot0T = q0.matrix().transpose();
	const Matrix3r rot1T = q1.matrix().transpose();

	// connector in body 0 (local)
	jointInfo.col(0) = rot0T * (position - x0);
	// connector in body 1 (local)
	jointInfo.col(1) = rot1T * (position - x1);
	// connector in body 0 (global)
	jointInfo.col(6) = position;
	// connector in body 1 (global)
	jointInfo.col(7) = position;

	// determine constraint coordinate system
	// with direction as x-axis
	jointInfo.col(8) = direction;
	jointInfo.col(8).normalize();

	Vector3r v(1.0, 0.0, 0.0);
	// check if vectors are parallel
	if (fabs(v.dot(jointInfo.col(8))) > 0.99)
		v = Vector3r(0.0, 1.0, 0.0);

	jointInfo.col(9) = jointInfo.col(8).cross(v);
	jointInfo.col(10) = jointInfo.col(8).cross(jointInfo.col(9));
	jointInfo.col(9).normalize();
	jointInfo.col(10).normalize();

	// joint axis in body 1 (global)
	jointInfo.col(11) = jointInfo.col(8);

	// coordinate system of body 0 (local)
	jointInfo.block<3, 3>(0, 2) = rot0T * jointInfo.block<3, 3>(0, 8);

	// joint axis in body 1 (local)
	jointInfo.col(5) = rot1T * jointInfo.col(11);

	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::update_HingeJoint(
	const Vector3r &x0,
	const Quaternionr &q0,
	const Vector3r &x1,
	const Quaternionr &q1,
	Eigen::Matrix<Real, 3, 12> &jointInfo
	)
{
	// jointInfo contains
	// 0:	connector in body 0 (local)
	// 1:	connector in body 1 (local)
	// 2-4:	coordinate system of body 0 (local)
	// 5:	joint axis in body 1 (local)
	// 6:	connector in body 0 (global)
	// 7:	connector in body 1 (global)
	// 8-10:coordinate system of body 0 (global)
	// 11:	joint axis in body 1 (global)


	// compute world space positions of connectors
	const Matrix3r rot0 = q0.matrix();
	const Matrix3r rot1 = q1.matrix();
	jointInfo.col(6) = rot0 * jointInfo.col(0) + x0;
	jointInfo.col(7) = rot1 * jointInfo.col(1) + x1;

	// transform constraint coordinate system of body 0 to world space
	jointInfo.block<3, 3>(0, 8) = rot0 * jointInfo.block<3, 3>(0, 2);
	// transform joint axis of body 1 to world space
	jointInfo.col(11) = rot1 * jointInfo.col(5);

	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::solve_HingeJoint(
	const Real invMass0,
	const Vector3r &x0,
	const Matrix3r &inertiaInverseW0,
	const Quaternionr &q0,
	const Real invMass1,
	const Vector3r &x1,
	const Matrix3r &inertiaInverseW1,
	const Quaternionr &q1,
	const Eigen::Matrix<Real, 3, 12> &jointInfo,
	Vector3r &corr_x0, Quaternionr &corr_q0,
	Vector3r &corr_x1, Quaternionr &corr_q1)
{
	// jointInfo contains
	// 0:	connector in body 0 (local)
	// 1:	connector in body 1 (local)
	// 2-4:	coordinate system of body 0 (local)
	// 5:	joint axis in body 1 (local)
	// 6:	connector in body 0 (global)
	// 7:	connector in body 1 (global)
	// 8-10:coordinate system of body 0 (global)
	// 11:	joint axis in body 1 (global)

	const Vector3r &c0 = jointInfo.col(6);
	const Vector3r &c1 = jointInfo.col(7);
	const Vector3r &axis1 = jointInfo.col(11);
	const Vector3r &t1 = jointInfo.col(9);
	const Vector3r &t2 = jointInfo.col(10);
	const Vector3r r0 = c0 - x0;
	const Vector3r r1 = c1 - x1;
	Matrix3r r0_star, r1_star;
	MathFunctions::crossProductMatrix(r0, r0_star);
	MathFunctions::crossProductMatrix(r1, r1_star);

	const Vector3r u1 = t1.cross(axis1);
	const Vector3r u2 = t2.cross(axis1);

	Eigen::Matrix<Real, 5, 1> b;
	b.block<3, 1>(0, 0) = c1 - c0;
	b(3, 0) = -t1.dot(axis1);
	b(4, 0) = -t2.dot(axis1);

	Eigen::Matrix<Real, 5, 5> K;
	K.setZero();
	if (invMass0 != 0.0)
	{
		// Jacobian for body 0 is
		//
		// (I_3   -r0*)
		// (0     u1^T)
		// (0     u2^T)
		//
		// where I_3 is the identity matrix and r0* is the cross product matrix of r0
		//
		// J M^-1 J^T =
		// ( 1/m I_3-r0 * J0^-1 * r0*    -r0 * J0^-1 * u1      -r0 * J0^-1 * u2   )
		// ( (-r0 * J0^-1 * u1)^T         u1^T * J0^-1 * u1    u1^T * J0^-1 u2    )
		// ( (-r0 * J0^-1 * u2)^T         u2^T * J0^-1 u1      u2^T * J0^-1 * u2  )

		Matrix3r K00;
		computeMatrixK(c0, invMass0, x0, inertiaInverseW0, K00);

		K.block<3, 3>(0, 0) = K00;
		K.block<3, 1>(0, 3) = -r0_star * inertiaInverseW0 * u1;
		K.block<3, 1>(0, 4) = -r0_star * inertiaInverseW0 * u2;
		K.block<1, 3>(3, 0) = K.block<3, 1>(0, 3).transpose();
		K.block<1, 3>(4, 0) = K.block<3, 1>(0, 4).transpose();
		K(3, 3) = u1.transpose() * inertiaInverseW0 * u1;
		K(3, 4) = u1.transpose() * inertiaInverseW0 * u2;
		K(4, 3) = K(3, 4);
		K(4, 4) = u2.transpose() * inertiaInverseW0 * u2;
	}
	if (invMass1 != 0.0)
	{
		// Jacobian for body 1 is
		//
		// ( -I_3   r1*  )
		// ( 0     -u1^T )
		// ( 0     -u2^T )
		//
		// where I_3 is the identity matrix and r1* is the cross product matrix of r1
		//
		// J M^-1 J^T =
		// ( 1/m I_3-r1 * J1^-1 * r1*    -r1 * J1^-1 * u1    -r1 * J1^-1 * u2 )
		// ( (-r1* J1^-1 * u1)^T         u1^T * J1^-1 u1     u1^T * J1^-1 u2  )
		// ( (-r1* J1^-1 * u2)^T         u2^T * J1^-1 u1     u2^T * J1^-1 u2  )

		Matrix3r K11;
		computeMatrixK(c1, invMass1, x1, inertiaInverseW1, K11);

		K.block<3, 3>(0, 0) += K11;
		const Vector3r K_03 = -r1_star * inertiaInverseW1 * u1;
		const Vector3r K_04 = -r1_star * inertiaInverseW1 * u2;
		K.block<3, 1>(0, 3) += K_03;
		K.block<3, 1>(0, 4) += K_04;
		K.block<1, 3>(3, 0) += K_03.transpose();
		K.block<1, 3>(4, 0) += K_04.transpose();
		K(3, 3) += u1.transpose() * inertiaInverseW1 * u1;
		const Real K_34 = u1.transpose() * inertiaInverseW1 * u2;
		K(3, 4) += K_34;
		K(4, 3) += K_34;
		K(4, 4) += u2.transpose() * inertiaInverseW1 * u2;
	}

	const Eigen::Matrix<Real, 5, 5> Kinv = K.inverse();

	const Eigen::Matrix<Real, 5, 1> lambda = Kinv * b;
	const Vector3r pt = lambda.block<3, 1>(0, 0);

	if (invMass0 != 0.0)
	{
		corr_x0 = invMass0*pt;
		const Vector3r ot = (inertiaInverseW0 * (r0.cross(pt) + u1*lambda(3, 0) + u2*lambda(4, 0)));
		const Quaternionr otQ(0.0, ot[0], ot[1], ot[2]);
		corr_q0.coeffs() = 0.5 *(otQ*q0).coeffs();
	}

	if (invMass1 != 0.0)
	{
		corr_x1 = -invMass1*pt;
		const Vector3r ot = (inertiaInverseW1 * (r1.cross(-pt) - u1*lambda(3, 0) - u2*lambda(4, 0)));
		const Quaternionr otQ(0.0, ot[0], ot[1], ot[2]);
		corr_q1.coeffs() = 0.5 *(otQ*q1).coeffs();
	}

	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::init_UniversalJoint(
	const Vector3r &x0,
	const Quaternionr &q0,
	const Vector3r &x1,
	const Quaternionr &q1,
	const Vector3r &position,
	const Vector3r &jointAxis0,
	const Vector3r &jointAxis1,
	Eigen::Matrix<Real, 3, 8> &jointInfo
	)
{
	// jointInfo contains
	// 0:	connector in body 0 (local)
	// 1:	connector in body 1 (local)
	// 2:	constraint axis 0 in body 0 (local)
	// 3:	constraint axis 1 in body 1 (local)
	// 4:	connector in body 0 (global)
	// 5:	connector in body 1 (global)
	// 6:	constraint axis 0 in body 0 (global)	
	// 7:	constraint axis 1 in body 1 (global)

	// transform in local coordinates
	const Matrix3r rot0T = q0.matrix().transpose();
	const Matrix3r rot1T = q1.matrix().transpose();

	// connector in body 0 (local)
	jointInfo.col(0) = rot0T * (position - x0);
	// connector in body 1 (local)
	jointInfo.col(1) = rot1T * (position - x1);
	// connector in body 0 (global)
	jointInfo.col(4) = position;
	// connector in body 1 (global)
	jointInfo.col(5) = position;

	// determine constraint coordinate system
	Vector3r constraintAxis = jointAxis0.cross(jointAxis1);
	if (constraintAxis.norm() < 1.0e-3)
		return false;

	// joint axis in body 0 (global)
	jointInfo.col(6) = jointAxis0;
	jointInfo.col(6).normalize();

	// joint axis in body 1 (global)
	jointInfo.col(7) = jointAxis1;
	jointInfo.col(7).normalize();

	// correction axis in body 0 (local)
	jointInfo.col(2) = rot0T * jointInfo.col(6);

	// correction axis in body 1 (local)
	jointInfo.col(3) = rot1T * jointInfo.col(7);

	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::update_UniversalJoint(
	const Vector3r &x0,
	const Quaternionr &q0,
	const Vector3r &x1,
	const Quaternionr &q1,
	Eigen::Matrix<Real, 3, 8> &jointInfo
	)
{
	// jointInfo contains
	// 0:	connector in body 0 (local)
	// 1:	connector in body 1 (local)
	// 2:	constraint axis 0 in body 0 (local)
	// 3:	constraint axis 1 in body 1 (local)
	// 4:	connector in body 0 (global)
	// 5:	connector in body 1 (global)
	// 6:	constraint axis 0 in body 0 (global)	
	// 7:	constraint axis 1 in body 1 (global)

	// compute world space positions of connectors
	const Matrix3r rot0 = q0.matrix();
	const Matrix3r rot1 = q1.matrix();
	jointInfo.col(4) = rot0 * jointInfo.col(0) + x0;
	jointInfo.col(5) = rot1 * jointInfo.col(1) + x1;

	// transform joint axis of body 0 to world space
	jointInfo.col(6) = rot0 * jointInfo.col(2);
	// transform joint axis of body 1 to world space
	jointInfo.col(7) = rot1 * jointInfo.col(3);

	return true;
}


// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::solve_UniversalJoint(
	const Real invMass0,
	const Vector3r &x0,
	const Matrix3r &inertiaInverseW0,
	const Quaternionr &q0,
	const Real invMass1,
	const Vector3r &x1,
	const Matrix3r &inertiaInverseW1,
	const Quaternionr &q1,
	const Eigen::Matrix<Real, 3, 8> &jointInfo,
	Vector3r &corr_x0, Quaternionr &corr_q0,
	Vector3r &corr_x1, Quaternionr &corr_q1)
{
	// jointInfo contains
	// 0:	connector in body 0 (local)
	// 1:	connector in body 1 (local)
	// 2:	constraint axis 0 in body 0 (local)
	// 3:	constraint axis 1 in body 1 (local)
	// 4:	connector in body 0 (global)
	// 5:	connector in body 1 (global)
	// 6:	constraint axis 0 in body 0 (global)	
	// 7:	constraint axis 1 in body 1 (global)

	const Vector3r &c0 = jointInfo.col(4);
	const Vector3r &c1 = jointInfo.col(5);
	const Vector3r &axis0 = jointInfo.col(6);
	const Vector3r &axis1 = jointInfo.col(7);
	const Vector3r u = axis0.cross(axis1);
	const Vector3r r0 = c0 - x0;
	const Vector3r r1 = c1 - x1;
	Matrix3r r0_star, r1_star;
	MathFunctions::crossProductMatrix(r0, r0_star);
	MathFunctions::crossProductMatrix(r1, r1_star);

	Eigen::Matrix<Real, 4, 1> b;
	b.block<3, 1>(0, 0) = c1 - c0;
	b(3, 0) = -axis0.dot(axis1);

	Eigen::Matrix<Real, 4, 4> K;
	K.setZero();
	Eigen::Matrix<Real, 4, 6> J0, J1;
	if (invMass0 != 0.0)
	{
		// Jacobian for body 0 is
		//
		// (I_3   -r0*)
		// (0     u^T)
		//
		// where I_3 is the identity matrix and r0* is the cross product matrix of r0
		//
		// J M^-1 J^T =
		// ( 1/m I_3-r0 * J0^-1 * r0*    -r0 * J0^-1 * u )
		// ( (-r0 * J0^-1 * u)^T         u^T * J0^-1 * u )

		Matrix3r K00;
		computeMatrixK(c0, invMass0, x0, inertiaInverseW0, K00);

		K.block<3, 3>(0, 0) = K00;
		K.block<3, 1>(0, 3) = -r0_star * inertiaInverseW0 * u;
		K.block<1, 3>(3, 0) = K.block<3, 1>(0, 3).transpose();
		K(3, 3) = u.transpose() * inertiaInverseW0 * u;
	}
	if (invMass1 != 0.0)
	{
		// Jacobian for body 1 is
		//
		// ( -I_3   r1*  )
		// ( 0     -u^T )
		//
		// where I_3 is the identity matrix and r1* is the cross product matrix of r1
		//
		// J M^-1 J^T =
		// ( 1/m I_3-r1 * J1^-1 * r1*    -r1 * J1^-1 * u )
		// ( (-r1 * J1^-1 * u)^T         u^T * J1^-1 * u )

		Matrix3r K11;
		computeMatrixK(c1, invMass1, x1, inertiaInverseW1, K11);

		K.block<3, 3>(0, 0) += K11;
		const Vector3r K_03 = -r1_star * inertiaInverseW1 * u;
		K.block<3, 1>(0, 3) += K_03;
		K.block<1, 3>(3, 0) += K_03.transpose();
		K(3, 3) += u.transpose() * inertiaInverseW1 * u;
	}

	const Eigen::Matrix<Real, 4, 4> Kinv = K.inverse();

	const Eigen::Matrix<Real, 4, 1> lambda = Kinv * b;
	const Vector3r pt = lambda.block<3, 1>(0, 0);

	if (invMass0 != 0.0)
	{
		corr_x0 = invMass0*pt;
		const Vector3r ot = (inertiaInverseW0 * (r0.cross(pt) + u*lambda(3, 0)));
		const Quaternionr otQ(0.0, ot[0], ot[1], ot[2]);
		corr_q0.coeffs() = 0.5 *(otQ*q0).coeffs();
	}

	if (invMass1 != 0.0)
	{
		corr_x1 = -invMass1*pt;
		const Vector3r ot = (inertiaInverseW1 * (r1.cross(-pt) - u*lambda(3, 0)));
		const Quaternionr otQ(0.0, ot[0], ot[1], ot[2]);
		corr_q1.coeffs() = 0.5 *(otQ*q1).coeffs();
	}

	return true;
}


// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::init_SliderJoint(
	const Vector3r &x0,
	const Quaternionr &q0,
	const Vector3r &x1,
	const Quaternionr &q1,
	const Vector3r &position,
	const Vector3r &direction,
	Eigen::Matrix<Real, 3, 14> &jointInfo
	)
{
	// jointInfo contains
	// 0:	connector in body 0 (local)
	// 1:	connector in body 1 (local)
	// 2-4:	coordinate system of body 0 (local)
	// 5:	joint axis in body 1 (local)	
	// 6:	connector in body 0 (global)
	// 7:	connector in body 1 (global)
	// 8-10:coordinate system of body 0 (global)
	// 11:	joint axis in body 1 (global)
	// 12:	perpendicular vector on joint axis (normalized) in body 1 (local)
	// 13:	perpendicular vector on joint axis (normalized) in body 1 (global)

	// transform in local coordinates
	const Matrix3r rot0T = q0.matrix().transpose();
	const Matrix3r rot1T = q1.matrix().transpose();

	// connector in body 0 (local)
	jointInfo.col(0) = rot0T * (position - x0);
	// connector in body 1 (local)
	jointInfo.col(1) = rot1T * (position - x1);
	// connector in body 0 (global)
	jointInfo.col(6) = position;
	// connector in body 1 (global)
	jointInfo.col(7) = position;

	// determine constraint coordinate system
	// with direction as x-axis
	jointInfo.col(8) = direction;
	jointInfo.col(8).normalize();

	Vector3r v(1.0, 0.0, 0.0);
	// check if vectors are parallel
	if (fabs(v.dot(jointInfo.col(8))) > 0.99)
		v = Vector3r(0.0, 1.0, 0.0);

	jointInfo.col(9) = jointInfo.col(8).cross(v);
	jointInfo.col(10) = jointInfo.col(8).cross(jointInfo.col(9));
	jointInfo.col(9).normalize();
	jointInfo.col(10).normalize();

	// joint axis in body 1 (global)
	jointInfo.col(11) = jointInfo.col(8);

	// coordinate system of body 0 (local)
	jointInfo.block<3, 3>(0, 2) = rot0T * jointInfo.block<3, 3>(0, 8);

	// joint axis in body 1 (local)
	jointInfo.col(5) = rot1T * jointInfo.col(11);

	// perpendicular vector on joint axis (normalized) in body 1 (global)
	jointInfo.col(13) = jointInfo.col(9);

	// perpendicular vector on joint axis(normalized) in body 1 (local)
	jointInfo.col(12) = rot1T * jointInfo.col(13);

	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::update_SliderJoint(
	const Vector3r &x0,
	const Quaternionr &q0,
	const Vector3r &x1,
	const Quaternionr &q1,
	Eigen::Matrix<Real, 3, 14> &jointInfo
	)
{
	// jointInfo contains
	// 0:	connector in body 0 (local)
	// 1:	connector in body 1 (local)
	// 2-4:	coordinate system of body 0 (local)
	// 5:	joint axis in body 1 (local)	
	// 6:	connector in body 0 (global)
	// 7:	connector in body 1 (global)
	// 8-10:coordinate system of body 0 (global)
	// 11:	joint axis in body 1 (global)
	// 12:	perpendicular vector on joint axis (normalized) in body 1 (local)
	// 13:	perpendicular vector on joint axis (normalized) in body 1 (global)


	// compute world space positions of connectors
	const Matrix3r rot0 = q0.matrix();
	const Matrix3r rot1 = q1.matrix();
	jointInfo.col(6) = rot0 * jointInfo.col(0) + x0;
	jointInfo.col(7) = rot1 * jointInfo.col(1) + x1;

	// transform constraint coordinate system of body 0 to world space
	jointInfo.block<3, 3>(0, 8) = rot0 * jointInfo.block<3, 3>(0, 2);
	// transform joint axis of body 1 to world space
	jointInfo.col(11) = rot1 * jointInfo.col(5);
	// transform perpendicular vector on joint axis of body 1 to world space 
	jointInfo.col(13) = rot1 * jointInfo.col(12);

	const Vector3r dir = jointInfo.col(8);
	const Vector3r p = jointInfo.col(6);
	const Vector3r s = jointInfo.col(7);
	// move the joint point of body 0 to the closest point on the line to joint point 1
	jointInfo.col(6) = p + (dir * (((s - p).dot(dir)) / dir.squaredNorm()));

	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::solve_SliderJoint(
	const Real invMass0,
	const Vector3r &x0,
	const Matrix3r &inertiaInverseW0,
	const Quaternionr &q0,
	const Real invMass1,
	const Vector3r &x1,
	const Matrix3r &inertiaInverseW1,
	const Quaternionr &q1,
	const Eigen::Matrix<Real, 3, 14> &jointInfo,
	Vector3r &corr_x0, Quaternionr &corr_q0,
	Vector3r &corr_x1, Quaternionr &corr_q1)
{
	// jointInfo contains
	// 0:	connector in body 0 (local)
	// 1:	connector in body 1 (local)
	// 2-4:	coordinate system of body 0 (local)
	// 5:	joint axis in body 1 (local)	
	// 6:	connector in body 0 (global)
	// 7:	connector in body 1 (global)
	// 8-10:coordinate system of body 0 (global)
	// 11:	joint axis in body 1 (global)
	// 12:	perpendicular vector on joint axis (normalized) in body 1 (local)
	// 13:	perpendicular vector on joint axis (normalized) in body 1 (global)

	const Vector3r &c0 = jointInfo.col(6);
	const Vector3r &c1 = jointInfo.col(7);
	const Vector3r &axis0 = jointInfo.col(8);
	const Vector3r &axis1 = jointInfo.col(11);
	const Vector3r axis = 0.5 * (axis0 + axis1);
	const Vector3r u = axis0.cross(axis1);
	const Vector3r &t1 = jointInfo.col(9);
	const Vector3r &t2 = jointInfo.col(10);
	const Vector3r &t3 = jointInfo.col(13);
	const Vector3r r0 = c0 - x0;
	const Vector3r r1 = c1 - x1;
	Matrix3r r0_star, r1_star;
	MathFunctions::crossProductMatrix(r0, r0_star);
	MathFunctions::crossProductMatrix(r1, r1_star);

	const Vector3r u1 = t1.cross(axis1);
	const Vector3r u2 = t2.cross(axis1);

	// projection 
	const Eigen::Matrix<Real, 3, 2> PT = jointInfo.block<3, 2>(0, 9);
	const Eigen::Matrix<Real, 2, 3> P = PT.transpose();


	Eigen::Matrix<Real, 5, 1> b;
	b.block<2, 1>(0, 0) = P * (c1 - c0);
	b(2, 0) = -t1.dot(axis1);
	b(3, 0) = -t2.dot(axis1);

// 	// determine correction angle (slider axis)
// 	Real delta = 0.0;
// 	Real c = t1.dot(t3);
// 	c = std::min<Real>(1.0, c);
// 	c = std::max<Real>(-1.0, c);
// 	if ((t1.cross(t3)).dot(axis) > 0.0)
// 		delta -= acos(c);
// 	else
// 		delta += acos(c);
// 
// 	const Real pi = (Real)M_PI;
// 	if (delta < -pi)
// 		delta += 2.0 * pi;
// 	if (delta > pi)
// 		delta -= 2.0 * pi;

	//b(4, 0) = -delta;
	b(4, 0) = t2.dot(t3);
//	std::cout << -delta << ", " << t2.dot(t3) << "\n";


	Eigen::Matrix<Real, 5, 5> K;
	K.setZero();
	if (invMass0 != 0.0)
	{
		// Jacobian for body 0 is
		//
		// (P     -P r0*)
		// (0     t1^T)
		// (0     t2^T)
		// (0     axis^T)
		//
		// where I_3 is the identity matrix and r0* is the cross product matrix of r0
		//
		// J M^-1 J^T =
		// ( P (1/m I_3-r0 * J0^-1 * r0*) P^T    P (-r0 * J0^-1 * t1)    P (-r0 * J0^-1 * t2)		P (-r0 * J0^-1 * axis) )
		// ( (-r0 * J0^-1 * t1)^T P^T            t1^T * J0^-1 t1         t1^T * J0^-1 * t2		    t1^T * J0^-1 * axis    )
		// ( (-r0 * J0^-1 * t2)^T P^T            t2^T * J0^-1 t1         t2^T * J0^-1 * t2		    t2^T * J0^-1 axis      )
		// ( (-r0 * J0^-1 * axis)^T P^T          axis^T * J0^-1 t1       axis^T * J0^-1 t2		    axis^T * J0^-1 axis    )

		Matrix3r K00;
		computeMatrixK(c0, invMass0, x0, inertiaInverseW0, K00);
		const Eigen::Matrix<Real, 2, 3> neg_P_r0_star_Jinv = -P * r0_star * inertiaInverseW0;
		const Vector3r Jinv_axis = inertiaInverseW0 * axis;
		const Vector3r Jinv_u1 = inertiaInverseW0 * u1;
		const Vector3r Jinv_u2 = inertiaInverseW0 * u2;

		K.block<2, 2>(0, 0) = P * K00 * PT;
		K.block<2, 1>(0, 2) = neg_P_r0_star_Jinv * u1;
		K.block<2, 1>(0, 3) = neg_P_r0_star_Jinv * u2;
		K.block<2, 1>(0, 4) = neg_P_r0_star_Jinv * axis;
		K.block<1, 2>(2, 0) = K.block<2, 1>(0, 2).transpose();
		K.block<1, 2>(3, 0) = K.block<2, 1>(0, 3).transpose();
		K.block<1, 2>(4, 0) = K.block<2, 1>(0, 4).transpose();
		K(2, 2) = u1.transpose() * Jinv_u1;
		K(2, 3) = u1.transpose() * Jinv_u2;
		K(2, 4) = u1.transpose() * Jinv_axis;
		K(3, 2) = K(2, 3);
		K(4, 2) = K(2, 4);
		K(3, 3) = u2.transpose() * Jinv_u2;
		K(3, 4) = u2.transpose() * Jinv_axis;
		K(4, 3) = K(3, 4);
		K(4, 4) = axis.transpose() * Jinv_axis;
	}
	if (invMass1 != 0.0)
	{
		// Jacobian for body 1 is
		//
		// ( -P    P r1*  )
		// ( 0     -t1^T )
		// ( 0     -t2^T )
		// ( 0     axis^T)
		//
		// where I_3 is the identity matrix and r1* is the cross product matrix of r1
		//
		// J M^-1 J^T =
		// ( P (1/m I_3-r1 * J1^-1 * r1*) P^T    P (-r1 * J1^-1 * t1)    P (-r1 * J1^-1 * t2)		P (-r1 * J1^-1 * axis) )
		// ( (-r1 * J1^-1 * t1)^T P^T            t1^T * J1^-1 t1         t1^T * J1^-1 * t2		    t1^T * J1^-1 * axis    )
		// ( (-r1 * J1^-1 * t2)^T P^T            t2^T * J1^-1 t1         t2^T * J1^-1 * t2		    t2^T * J1^-1 axis      )
		// ( (-r1 * J1^-1 * axis)^T P^T          axis^T * J1^-1 t1       axis^T * J1^-1 t2		    axis^T * J1^-1 axis    )

		Matrix3r K11;
		computeMatrixK(c1, invMass1, x1, inertiaInverseW1, K11);
		const Eigen::Matrix<Real, 2, 3> neg_P_r1_star_Jinv = -P * r1_star * inertiaInverseW1;
		const Vector3r Jinv_axis = inertiaInverseW1 * axis;
		const Vector3r Jinv_u1 = inertiaInverseW1 * u1;
		const Vector3r Jinv_u2 = inertiaInverseW1 * u2;

		K.block<2, 2>(0, 0) += P * K11 * PT;
		const Vector2r K_02 = neg_P_r1_star_Jinv * u1;
		const Vector2r K_03 = neg_P_r1_star_Jinv * u2;
		const Vector2r K_04 = neg_P_r1_star_Jinv * axis;
		K.block<2, 1>(0, 2) += K_02;
		K.block<2, 1>(0, 3) += K_03;
		K.block<2, 1>(0, 4) += K_04;
		K.block<1, 2>(2, 0) += K_02.transpose();
		K.block<1, 2>(3, 0) += K_03.transpose();
		K.block<1, 2>(4, 0) += K_04.transpose();
		K(2, 2) += u1.transpose() * Jinv_u1;
		const Real K_23 = u1.transpose() * Jinv_u2;
		K(2, 3) += K_23;
		const Real K_24 = u1.transpose() * Jinv_axis;
		K(2, 4) += K_24;
		K(3, 2) += K_23;
		K(3, 3) += u2.transpose() * Jinv_u2;
		const Real K_34 = u2.transpose() * Jinv_axis;
		K(3, 4) += K_34;
		K(4, 2) += K_24;
		K(4, 3) += K_34;
		K(4, 4) += axis.transpose() * Jinv_axis;
	}

	const Eigen::Matrix<Real, 5, 5> Kinv = K.inverse();
	const Eigen::Matrix<Real, 5, 1> lambda = Kinv * b;
	const Vector3r pt = PT * lambda.block<2, 1>(0, 0);

	const Vector3r angMomentum = u1*lambda(2, 0) + u2*lambda(3, 0) + axis*lambda(4, 0);

	if (invMass0 != 0.0)
	{
		corr_x0 = invMass0*pt;
		const Vector3r ot = (inertiaInverseW0 * (r0.cross(pt) + angMomentum));
		const Quaternionr otQ(0.0, ot[0], ot[1], ot[2]);
		corr_q0.coeffs() = 0.5 *(otQ*q0).coeffs();
	}

	if (invMass1 != 0.0)
	{
		corr_x1 = -invMass1*pt;
		const Vector3r ot = (inertiaInverseW1 * (r1.cross(-pt) - angMomentum));
		const Quaternionr otQ(0.0, ot[0], ot[1], ot[2]);
		corr_q1.coeffs() = 0.5 *(otQ*q1).coeffs();
	}

	return true;
}


// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::init_TargetPositionMotorSliderJoint(
	const Vector3r &x0,
	const Quaternionr &q0,
	const Vector3r &x1,
	const Quaternionr &q1,
	const Vector3r &position,
	const Vector3r &direction,
	Eigen::Matrix<Real, 3, 14> &jointInfo
	)
{
	// jointInfo contains
	// 0:	connector in body 0 (local)
	// 1:	connector in body 1 (local)
	// 2-4:	coordinate system of body 0 (local)
	// 5:	joint axis in body 1 (local)	
	// 6:	connector in body 0 (global)
	// 7:	connector in body 1 (global)
	// 8-10:coordinate system of body 0 (global)
	// 11:	joint axis in body 1 (global)
	// 12:	perpendicular vector on joint axis (normalized) in body 1 (local)
	// 13:	perpendicular vector on joint axis (normalized) in body 1 (global)

	// transform in local coordinates
	const Matrix3r rot0T = q0.matrix().transpose();
	const Matrix3r rot1T = q1.matrix().transpose();

	// connector in body 0 (local)
	jointInfo.col(0) = rot0T * (position - x0);
	// connector in body 1 (local)
	jointInfo.col(1) = rot1T * (position - x1);
	// connector in body 0 (global)
	jointInfo.col(6) = position;
	// connector in body 1 (global)
	jointInfo.col(7) = position;

	// determine constraint coordinate system
	// with direction as x-axis
	jointInfo.col(8) = direction;
	jointInfo.col(8).normalize();

	Vector3r v(1.0, 0.0, 0.0);
	// check if vectors are parallel
	if (fabs(v.dot(jointInfo.col(8))) > 0.99)
		v = Vector3r(0.0, 1.0, 0.0);

	jointInfo.col(9) = jointInfo.col(8).cross(v);
	jointInfo.col(10) = jointInfo.col(8).cross(jointInfo.col(9));
	jointInfo.col(9).normalize();
	jointInfo.col(10).normalize();

	// joint axis in body 1 (global)
	jointInfo.col(11) = jointInfo.col(8);

	// coordinate system of body 0 (local)
	jointInfo.block<3, 3>(0, 2) = rot0T * jointInfo.block<3, 3>(0, 8);

	// joint axis in body 1 (local)
	jointInfo.col(5) = rot1T * jointInfo.col(11);

	// perpendicular vector on joint axis (normalized) in body 1 (global)
	jointInfo.col(13) = jointInfo.col(9);

	// perpendicular vector on joint axis(normalized) in body 1 (local)
	jointInfo.col(12) = rot1T * jointInfo.col(13);

	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::update_TargetPositionMotorSliderJoint(
	const Vector3r &x0,
	const Quaternionr &q0,
	const Vector3r &x1,
	const Quaternionr &q1,
	Eigen::Matrix<Real, 3, 14> &jointInfo
	)
{
	// jointInfo contains
	// 0:	connector in body 0 (local)
	// 1:	connector in body 1 (local)
	// 2-4:	coordinate system of body 0 (local)
	// 5:	joint axis in body 1 (local)	
	// 6:	connector in body 0 (global)
	// 7:	connector in body 1 (global)
	// 8-10:coordinate system of body 0 (global)
	// 11:	joint axis in body 1 (global)
	// 12:	perpendicular vector on joint axis (normalized) in body 1 (local)
	// 13:	perpendicular vector on joint axis (normalized) in body 1 (global)


	// compute world space positions of connectors
	const Matrix3r rot0 = q0.matrix();
	const Matrix3r rot1 = q1.matrix();
	jointInfo.col(6) = rot0 * jointInfo.col(0) + x0;
	jointInfo.col(7) = rot1 * jointInfo.col(1) + x1;

	// transform constraint coordinate system of body 0 to world space
	jointInfo.block<3, 3>(0, 8) = rot0 * jointInfo.block<3, 3>(0, 2);
	// transform joint axis of body 1 to world space
	jointInfo.col(11) = rot1 * jointInfo.col(5);
	// transform perpendicular vector on joint axis of body 1 to world space 
	jointInfo.col(13) = rot1 * jointInfo.col(12);

	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::solve_TargetPositionMotorSliderJoint(
	const Real invMass0,
	const Vector3r &x0,
	const Matrix3r &inertiaInverseW0,
	const Quaternionr &q0,
	const Real invMass1,
	const Vector3r &x1,
	const Matrix3r &inertiaInverseW1,
	const Quaternionr &q1,
	const Real targetPosition,	
	const Eigen::Matrix<Real, 3, 14> &jointInfo,
	Vector3r &corr_x0, Quaternionr &corr_q0,
	Vector3r &corr_x1, Quaternionr &corr_q1)
{
	// jointInfo contains
	// 0:	connector in body 0 (local)
	// 1:	connector in body 1 (local)
	// 2-4:	coordinate system of body 0 (local)
	// 5:	joint axis in body 1 (local)	
	// 6:	connector in body 0 (global)
	// 7:	connector in body 1 (global)
	// 8-10:coordinate system of body 0 (global)
	// 11:	joint axis in body 1 (global)
	// 12:	perpendicular vector on joint axis (normalized) in body 1 (local)
	// 13:	perpendicular vector on joint axis (normalized) in body 1 (global)

	const Vector3r &c0 = jointInfo.col(6);
	const Vector3r &c1 = jointInfo.col(7);
	const Vector3r &axis0 = jointInfo.col(8);
	const Vector3r &axis1 = jointInfo.col(11);
	const Vector3r axis = 0.5 * (axis0 + axis1);
	const Vector3r u = axis0.cross(axis1);
	const Vector3r &t1 = jointInfo.col(9);
	const Vector3r &t2 = jointInfo.col(10);
	const Vector3r &t3 = jointInfo.col(13);
	const Vector3r r0 = c0 - x0;
	const Vector3r r1 = c1 - x1;
	Matrix3r r0_star, r1_star;
	MathFunctions::crossProductMatrix(r0, r0_star);
	MathFunctions::crossProductMatrix(r1, r1_star);

	// projection 
	const Eigen::Matrix<Real, 3, 3> PT = jointInfo.block<3, 3>(0, 8);
	const Eigen::Matrix<Real, 3, 3> P = PT.transpose();

	Eigen::Matrix<Real, 6, 1> b;
	b.block<3, 1>(0, 0) = P * (c1 - c0);
	b(0, 0) = -targetPosition + b(0, 0);
	b(3, 0) = t1.dot(u);
	b(4, 0) = t2.dot(u);

	// determine correction angle (slider axis)
	Real delta = 0.0;
	Real c = t1.dot(t3);
	c = std::min<Real>(1.0, c);
	c = std::max<Real>(-1.0, c);
	if ((t1.cross(t3)).dot(axis) > 0.0)
		delta -= acos(c);
	else
		delta += acos(c);

	const Real pi = (Real)M_PI;
	if (delta < -pi)
		delta += 2.0 * pi;
	if (delta > pi)
		delta -= 2.0 * pi;

	b(5, 0) = -delta;


	Eigen::Matrix<Real, 6, 6> K;
	K.setZero();
	if (invMass0 != 0.0)
	{
		// Jacobian for body 0 is
		//
		// (P     -P r0*)
		// (0     t1^T)
		// (0     t2^T)
		// (0     axis^T)
		//
		// where I_3 is the identity matrix and r0* is the cross product matrix of r0
		//
		// J M^-1 J^T =
		// ( P (1/m I_3-r0 * J0^-1 * r0*) P^T    P (-r0 * J0^-1 * t1)    P (-r0 * J0^-1 * t2)		P (-r0 * J0^-1 * axis) )
		// ( (-r0 * J0^-1 * t1)^T P^T            t1^T * J0^-1 t1         t1^T * J0^-1 * t2		    t1^T * J0^-1 * axis    )
		// ( (-r0 * J0^-1 * t2)^T P^T            t2^T * J0^-1 t1         t2^T * J0^-1 * t2		    t2^T * J0^-1 axis      )
		// ( (-r0 * J0^-1 * axis)^T P^T          axis^T * J0^-1 t1       axis^T * J0^-1 t2		    axis^T * J0^-1 axis    )

		Matrix3r K00;
		computeMatrixK(c0, invMass0, x0, inertiaInverseW0, K00);
		const Matrix3r neg_P_r0_star_Jinv = -P * r0_star * inertiaInverseW0;
		const Vector3r Jinv_axis = inertiaInverseW0 * axis;
		const Vector3r Jinv_t1 = inertiaInverseW0 * t1;
		const Vector3r Jinv_t2 = inertiaInverseW0 * t2;

		K.block<3, 3>(0, 0) = P * K00 * PT;
		K.block<3, 1>(0, 3) = neg_P_r0_star_Jinv * t1;
		K.block<3, 1>(0, 4) = neg_P_r0_star_Jinv * t2;
		K.block<3, 1>(0, 5) = neg_P_r0_star_Jinv * axis;
		K.block<1, 3>(3, 0) = K.block<3, 1>(0, 3).transpose();
		K.block<1, 3>(4, 0) = K.block<3, 1>(0, 4).transpose();
		K.block<1, 3>(5, 0) = K.block<3, 1>(0, 5).transpose();
		K(3, 3) = t1.transpose() * Jinv_t1;
		K(3, 4) = t1.transpose() * Jinv_t2;
		K(3, 5) = t1.transpose() * Jinv_axis;
		K(4, 3) = K(3, 4);
		K(5, 3) = K(3, 5);
		K(4, 4) = t2.transpose() * Jinv_t2;
		K(4, 5) = t2.transpose() * Jinv_axis;
		K(5, 4) = K(4, 5);
		K(5, 5) = axis.transpose() * Jinv_axis;
	}
	if (invMass1 != 0.0)
	{
		// Jacobian for body 1 is
		//
		// ( -P    P r1*  )
		// ( 0     -t1^T )
		// ( 0     -t2^T )
		// ( 0     axis^T)
		//
		// where I_3 is the identity matrix and r1* is the cross product matrix of r1
		//
		// J M^-1 J^T =
		// ( P (1/m I_3-r1 * J1^-1 * r1*) P^T    P (-r1 * J1^-1 * t1)    P (-r1 * J1^-1 * t2)		P (-r1 * J1^-1 * axis) )
		// ( (-r1 * J1^-1 * t1)^T P^T            t1^T * J1^-1 t1         t1^T * J1^-1 * t2		    t1^T * J1^-1 * axis    )
		// ( (-r1 * J1^-1 * t2)^T P^T            t2^T * J1^-1 t1         t2^T * J1^-1 * t2		    t2^T * J1^-1 axis      )
		// ( (-r1 * J1^-1 * axis)^T P^T          axis^T * J1^-1 t1       axis^T * J1^-1 t2		    axis^T * J1^-1 axis    )

		Matrix3r K11;
		computeMatrixK(c1, invMass1, x1, inertiaInverseW1, K11);
		const Matrix3r neg_P_r1_star_Jinv = -P * r1_star * inertiaInverseW1;
		const Vector3r Jinv_axis = inertiaInverseW1 * axis;
		const Vector3r Jinv_t1 = inertiaInverseW1 * t1;
		const Vector3r Jinv_t2 = inertiaInverseW1 * t2;

		K.block<3, 3>(0, 0) += P * K11 * PT;
		const Vector3r K_03 = neg_P_r1_star_Jinv * t1;
		const Vector3r K_04 = neg_P_r1_star_Jinv * t2;
		const Vector3r K_05 = neg_P_r1_star_Jinv * axis;
		K.block<3, 1>(0, 3) += K_03;
		K.block<3, 1>(0, 4) += K_04;
		K.block<3, 1>(0, 5) += K_05;
		K.block<1, 3>(3, 0) += K_03.transpose();
		K.block<1, 3>(4, 0) += K_04.transpose();
		K.block<1, 3>(5, 0) += K_05.transpose();
		K(3, 3) += t1.transpose() * Jinv_t1;
		const Real K_34 = t1.transpose() * Jinv_t2;
		K(3, 4) += K_34;
		const Real K_35 = t1.transpose() * Jinv_axis;
		K(3, 5) += K_35;
		K(4, 3) += K_34;
		K(4, 4) += t2.transpose() * Jinv_t2;
		const Real K_45 = t2.transpose() * Jinv_axis;
		K(4, 5) += K_45;
		K(5, 3) += K_35;
		K(5, 4) += K_45;
		K(5, 5) += axis.transpose() * Jinv_axis;
	}

	const Eigen::Matrix<Real, 6, 6> Kinv = K.inverse();
	const Eigen::Matrix<Real, 6, 1> lambda = Kinv * b;
	const Vector3r pt = PT * lambda.block<3, 1>(0, 0);
	const Vector3r angMomentum = t1*lambda(3, 0) + t2*lambda(4, 0) + axis*lambda(5, 0);

	if (invMass0 != 0.0)
	{
		corr_x0 = invMass0*pt;
		const Vector3r ot = (inertiaInverseW0 * (r0.cross(pt) + angMomentum));
		const Quaternionr otQ(0.0, ot[0], ot[1], ot[2]);
		corr_q0.coeffs() = 0.5 *(otQ*q0).coeffs();
	}

	if (invMass1 != 0.0)
	{
		corr_x1 = -invMass1*pt;
		const Vector3r ot = (inertiaInverseW1 * (r1.cross(-pt) - angMomentum));
		const Quaternionr otQ(0.0, ot[0], ot[1], ot[2]);
		corr_q1.coeffs() = 0.5 *(otQ*q1).coeffs();
	}

	return true;
}


// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::init_TargetVelocityMotorSliderJoint(
	const Vector3r &x0,
	const Quaternionr &q0,
	const Vector3r &x1,
	const Quaternionr &q1,
	const Vector3r &position,
	const Vector3r &direction,
	Eigen::Matrix<Real, 3, 14> &jointInfo
	)
{
	return init_TargetPositionMotorSliderJoint(x0, q0, x1, q1, position, direction, jointInfo);
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::update_TargetVelocityMotorSliderJoint(
	const Vector3r &x0,
	const Quaternionr &q0,
	const Vector3r &x1,
	const Quaternionr &q1,
	Eigen::Matrix<Real, 3, 14> &jointInfo
	)
{
	return update_TargetPositionMotorSliderJoint(x0, q0, x1, q1, jointInfo);
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::solve_TargetVelocityMotorSliderJoint(
	const Real invMass0,
	const Vector3r &x0,
	const Matrix3r &inertiaInverseW0,
	const Quaternionr &q0,
	const Real invMass1,
	const Vector3r &x1,
	const Matrix3r &inertiaInverseW1,
	const Quaternionr &q1,
	const Eigen::Matrix<Real, 3, 14> &jointInfo,
	Vector3r &corr_x0, Quaternionr &corr_q0,
	Vector3r &corr_x1, Quaternionr &corr_q1)
{
	return solve_SliderJoint(invMass0, x0, inertiaInverseW0, q0,
		invMass1, x1, inertiaInverseW1, q1,
		jointInfo, corr_x0, corr_q0, corr_x1, corr_q1);
}


// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::velocitySolve_TargetVelocityMotorSliderJoint(
	const Real invMass0,
	const Vector3r &x0,
	const Vector3r &v0,
	const Matrix3r &inertiaInverseW0,
	const Vector3r &omega0,
	const Real invMass1,
	const Vector3r &x1,
	const Vector3r &v1,
	const Matrix3r &inertiaInverseW1,
	const Vector3r &omega1,
	const Real targetAngularVelocity,
	const Eigen::Matrix<Real, 3, 14> &jointInfo,
	Vector3r &corr_v0, Vector3r &corr_omega0,
	Vector3r &corr_v1, Vector3r &corr_omega1)
{
	// jointInfo contains
	// 0:	connector in body 0 (local)
	// 1:	connector in body 1 (local)
	// 2-4:	coordinate system of body 0 (local)
	// 5:	joint axis in body 1 (local)	
	// 6:	connector in body 0 (global)
	// 7:	connector in body 1 (global)
	// 8-10:coordinate system of body 0 (global)
	// 11:	joint axis in body 1 (global)
	// 12:	perpendicular vector on joint axis (normalized) in body 1 (local)
	// 13:	perpendicular vector on joint axis (normalized) in body 1 (global)

	const Vector3r &c0 = jointInfo.col(6);
	const Vector3r &c1 = jointInfo.col(7);
	const Vector3r r0 = c0 - x0;
	const Vector3r r1 = c1 - x1;
	Matrix3r r0_star, r1_star;
	MathFunctions::crossProductMatrix(r0, r0_star);
	MathFunctions::crossProductMatrix(r1, r1_star);
	Vector3r deltaOmega = omega1 - omega0;

	// projection 
	const Eigen::Matrix<Real, 3, 3> PT = jointInfo.block<3, 3>(0, 8);
	const Eigen::Matrix<Real, 3, 3> P = PT.transpose();

	Eigen::Matrix<Real, 6, 1> b;
	b.block<3, 1>(0, 0) = P * (v1 - v0);
	b(0, 0) = -targetAngularVelocity + b(0, 0);
	b.block<3, 1>(3, 0) = deltaOmega;

	Eigen::Matrix<Real, 6, 6> K;
	K.setZero();
	if (invMass0 != 0.0)
	{
		// Jacobian for body 0 is
		//
		// (P     -P r0*)
		// (0     I_3)
		//
		// where I_3 is the identity matrix and r0* is the cross product matrix of r0
		//
		// J M^-1 J^T =
		// ( P (1/m I_3-r0 * J0^-1 * r0*) P^T    P (-r0 * J0^-1) )
		// ( (-r0 * J0^-1)^T P^T				J0^-1		     )

		Matrix3r K00;
		computeMatrixK(c0, invMass0, x0, inertiaInverseW0, K00);
		const Matrix3r neg_P_r0_star_Jinv = -P * r0_star * inertiaInverseW0;
		K.block<3, 3>(0, 0) = P * K00 * PT;
		K.block<3, 3>(0, 3) = neg_P_r0_star_Jinv;
		K.block<3, 3>(3, 0) = K.block<3, 3>(0, 3).transpose();
		K.block<3, 3>(3, 3) = inertiaInverseW0;
	}
	if (invMass1 != 0.0)
	{
		// Jacobian for body 1 is
		//
		// ( -P    P r1*  )
		// ( 0     -t1^T )
		// ( 0     -t2^T )
		// ( 0     axis^T)
		//
		// where I_3 is the identity matrix and r1* is the cross product matrix of r1
		//
		// J M^-1 J^T =
		// ( P (1/m I_3-r1 * J1^-1 * r1*) P^T    P (-r1 * J1^-1) )
		// ( (-r1 * J1^-1)^T P^T				 J1^-1			 )

		Matrix3r K11;
		computeMatrixK(c1, invMass1, x1, inertiaInverseW1, K11);
		const Matrix3r neg_P_r1_star_Jinv = -P * r1_star * inertiaInverseW1;

		K.block<3, 3>(0, 0) += P * K11 * PT;
		const Matrix3r K_03 = neg_P_r1_star_Jinv;
		K.block<3, 3>(0, 3) += K_03;
		K.block<3, 3>(3, 0) += K_03.transpose();
		K.block<3, 3>(3, 3) += inertiaInverseW1;
	}

	const Eigen::Matrix<Real, 6, 6> Kinv = K.inverse();
	const Eigen::Matrix<Real, 6, 1> lambda = Kinv * b;
	const Vector3r p = PT * lambda.block<3, 1>(0, 0);
	const Vector3r angMomentum = lambda.block<3, 1>(3, 0);

	if (invMass0 != 0.0)
	{
		corr_v0 = invMass0*p;
		corr_omega0 = (inertiaInverseW0 * (r0.cross(p) + angMomentum));
	}

	if (invMass1 != 0.0)
	{
		corr_v1 = -invMass1*p;
		corr_omega1 = (inertiaInverseW1 * (r1.cross(-p) - angMomentum));
	}

	return true;
}


// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::init_TargetAngleMotorHingeJoint(
	const Vector3r &x0,
	const Quaternionr &q0,
	const Vector3r &x1,
	const Quaternionr &q1,
	const Vector3r &position,
	const Vector3r &direction,
	Eigen::Matrix<Real, 3, 14> &jointInfo
	)
{
	// jointInfo contains
	// 0:	connector in body 0 (local)
	// 1:	connector in body 1 (local)
	// 2-4:	coordinate system of body 0 (local)
	// 5:	joint axis in body 1 (local)	
	// 6:	connector in body 0 (global)
	// 7:	connector in body 1 (global)
	// 8-10:coordinate system of body 0 (global)
	// 11:	joint axis in body 1 (global)
	// 12:	perpendicular vector on joint axis (normalized) in body 1 (local)
	// 13:	perpendicular vector on joint axis (normalized) in body 1 (global)

	// transform in local coordinates
	const Matrix3r rot0T = q0.matrix().transpose();
	const Matrix3r rot1T = q1.matrix().transpose();

	// connector in body 0 (local)
	jointInfo.col(0) = rot0T * (position - x0);
	// connector in body 1 (local)
	jointInfo.col(1) = rot1T * (position - x1);
	// connector in body 0 (global)
	jointInfo.col(6) = position;
	// connector in body 1 (global)
	jointInfo.col(7) = position;

	// determine constraint coordinate system
	// with direction as x-axis
	jointInfo.col(8) = direction;
	jointInfo.col(8).normalize();

	Vector3r v(1.0, 0.0, 0.0);
	// check if vectors are parallel
	if (fabs(v.dot(jointInfo.col(8))) > 0.99)
		v = Vector3r(0.0, 1.0, 0.0);

	jointInfo.col(9) = jointInfo.col(8).cross(v);
	jointInfo.col(10) = jointInfo.col(8).cross(jointInfo.col(9));
	jointInfo.col(9).normalize();
	jointInfo.col(10).normalize();

	// joint axis in body 1 (global)
	jointInfo.col(11) = jointInfo.col(8);

	// coordinate system of body 0 (local)
	jointInfo.block<3, 3>(0, 2) = rot0T * jointInfo.block<3, 3>(0, 8);

	// joint axis in body 1 (local)
	jointInfo.col(5) = rot1T * jointInfo.col(11);

	// perpendicular vector on joint axis (normalized) in body 1 (global)
	jointInfo.col(13) = jointInfo.col(9);

	// perpendicular vector on joint axis(normalized) in body 1 (local)
	jointInfo.col(12) = rot1T * jointInfo.col(13);

	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::update_TargetAngleMotorHingeJoint(
	const Vector3r &x0,
	const Quaternionr &q0,
	const Vector3r &x1,
	const Quaternionr &q1,
	Eigen::Matrix<Real, 3, 14> &jointInfo
	)
{
	// jointInfo contains
	// 0:	connector in body 0 (local)
	// 1:	connector in body 1 (local)
	// 2-4:	coordinate system of body 0 (local)
	// 5:	joint axis in body 1 (local)	
	// 6:	connector in body 0 (global)
	// 7:	connector in body 1 (global)
	// 8-10:coordinate system of body 0 (global)
	// 11:	joint axis in body 1 (global)
	// 12:	perpendicular vector on joint axis (normalized) in body 1 (local)
	// 13:	perpendicular vector on joint axis (normalized) in body 1 (global)


	// compute world space positions of connectors
	const Matrix3r rot0 = q0.matrix();
	const Matrix3r rot1 = q1.matrix();
	jointInfo.col(6) = rot0 * jointInfo.col(0) + x0;
	jointInfo.col(7) = rot1 * jointInfo.col(1) + x1;

	// transform constraint coordinate system of body 0 to world space
	jointInfo.block<3, 3>(0, 8) = rot0 * jointInfo.block<3, 3>(0, 2);
	// transform joint axis of body 1 to world space
	jointInfo.col(11) = rot1 * jointInfo.col(5);
	// transform perpendicular vector on joint axis of body 1 to world space 
	jointInfo.col(13) = rot1 * jointInfo.col(12);

	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::solve_TargetAngleMotorHingeJoint(
	const Real invMass0,
	const Vector3r &x0,
	const Matrix3r &inertiaInverseW0,
	const Quaternionr &q0,
	const Real invMass1,
	const Vector3r &x1,
	const Matrix3r &inertiaInverseW1,
	const Quaternionr &q1,
	const Real targetAngle,
	const Eigen::Matrix<Real, 3, 14> &jointInfo,
	Vector3r &corr_x0, Quaternionr &corr_q0,
	Vector3r &corr_x1, Quaternionr &corr_q1)
{
	// jointInfo contains
	// 0:	connector in body 0 (local)
	// 1:	connector in body 1 (local)
	// 2-4:	coordinate system of body 0 (local)
	// 5:	joint axis in body 1 (local)	
	// 6:	connector in body 0 (global)
	// 7:	connector in body 1 (global)
	// 8-10:coordinate system of body 0 (global)
	// 11:	joint axis in body 1 (global)
	// 12:	perpendicular vector on joint axis (normalized) in body 1 (local)
	// 13:	perpendicular vector on joint axis (normalized) in body 1 (global)

	const Vector3r &c0 = jointInfo.col(6);
	const Vector3r &c1 = jointInfo.col(7);
	const Vector3r &axis0 = jointInfo.col(8);
	const Vector3r &axis1 = jointInfo.col(11);
	const Vector3r axis = 0.5 * (axis0 + axis1);
	const Vector3r u = axis0.cross(axis1);
	const Vector3r &t1 = jointInfo.col(9);
	const Vector3r &t2 = jointInfo.col(10);
	const Vector3r &t3 = jointInfo.col(13);
	const Vector3r r0 = c0 - x0;
	const Vector3r r1 = c1 - x1;
	Matrix3r r0_star, r1_star;
	MathFunctions::crossProductMatrix(r0, r0_star);
	MathFunctions::crossProductMatrix(r1, r1_star);

	Eigen::Matrix<Real, 6, 1> b;
	b.block<3, 1>(0, 0) = c1 - c0;
	b(3, 0) = t1.dot(u);
	b(4, 0) = t2.dot(u);

	// determine correction angle
	Real delta = targetAngle;
	Real c = t1.dot(t3);
	c = std::min<Real>(1.0, c);
	c = std::max<Real>(-1.0, c);
	if ((t1.cross(t3)).dot(axis) > 0.0)
		delta -= acos(c);
	else
		delta += acos(c);

	const Real pi = (Real)M_PI;
	if (delta < -pi)
		delta += 2.0 * pi;
	if (delta > pi)
		delta -= 2.0 * pi;

	b(5, 0) = -delta;

	Eigen::Matrix<Real, 6, 6> K;
	K.setZero();
	if (invMass0 != 0.0)
	{
		// Jacobian for body 0 is
		//
		// (I_3   -r0*)
		// (0     t1^T)
		// (0     t2^T)
		// (0     axis^T)
		//
		// where I_3 is the identity matrix and r0* is the cross product matrix of r0
		//
		// J M^-1 J^T =
		// ( 1/m I_3-r0 * J0^-1 * r0*    -r0 * J0^-1 * t1    -r0 * J0^-1 * t2		-r0 * J0^-1 * axis)
		// ( (-r0 * J0^-1 * t1)^T        t1^T * J0^-1 t1     t1^T * J0^-1 * t2		t1^T * J0^-1 * axis)
		// ( (-r0 * J0^-1 * t2)^T        t2^T * J0^-1 t1     t2^T * J0^-1 * t2		t2^T * J0^-1 axis)
		// ( (-r0 * J0^-1 * axis)^T      axis^T * J0^-1 t1   axis^T * J0^-1 t2		axis^T * J0^-1 axis)

		Matrix3r K00;
		computeMatrixK(c0, invMass0, x0, inertiaInverseW0, K00);
		const Matrix3r neg_P_r0_star_Jinv = -r0_star * inertiaInverseW0;
		const Vector3r Jinv_axis = inertiaInverseW0 * axis;
		const Vector3r Jinv_t1 = inertiaInverseW0 * t1;
		const Vector3r Jinv_t2 = inertiaInverseW0 * t2;

		K.block<3, 3>(0, 0) = K00;
		K.block<3, 1>(0, 3) = neg_P_r0_star_Jinv * t1;
		K.block<3, 1>(0, 4) = neg_P_r0_star_Jinv * t2;
		K.block<3, 1>(0, 5) = neg_P_r0_star_Jinv * axis;
		K.block<1, 3>(3, 0) = K.block<3, 1>(0, 3).transpose();
		K.block<1, 3>(4, 0) = K.block<3, 1>(0, 4).transpose();
		K.block<1, 3>(5, 0) = K.block<3, 1>(0, 5).transpose();
		K(3, 3) = t1.transpose() * Jinv_t1;
		K(3, 4) = t1.transpose() * Jinv_t2;
		K(3, 5) = t1.transpose() * Jinv_axis;
		K(4, 3) = K(3, 4);
		K(5, 3) = K(3, 5);
		K(4, 4) = t2.transpose() * Jinv_t2;
		K(4, 5) = t2.transpose() * Jinv_axis;
		K(5, 4) = K(4, 5);
		K(5, 5) = axis.transpose() * Jinv_axis;
	}
	if (invMass1 != 0.0)
	{
		// Jacobian for body 1 is
		//
		// ( -I_3   r1*  )
		// ( 0     -t1^T )
		// ( 0     -t2^T )
		// ( 0     axis^T)
		//
		// where I_3 is the identity matrix and r1* is the cross product matrix of r1
		//
		// J M^-1 J^T =
		// ( 1/m I_3-r1 * J1^-1 * r1*    -r1 * J1^-1 * t1    -r1 * J1^-1 * t2		-r1 * J1^-1 * axis)
		// ( (-r1 * J1^-1 * t1)^T        t1^T * J1^-1 t1     t1^T * J1^-1 t2		t1^T * J1^-1 * axis)
		// ( (-r1 * J1^-1 * t2)^T        t2^T * J1^-1 t1     t2^T * J1^-1 t2		t2^T * J1^-1 axis)
		// ( (-r1 * J1^-1 * axis)^T      axis^T * J1^-1 t1   axis^T * J1^-1	t2		axis^T * J1^-1 axis)

		Matrix3r K11;
		computeMatrixK(c1, invMass1, x1, inertiaInverseW1, K11);
		const Matrix3r neg_P_r1_star_Jinv = -r1_star * inertiaInverseW1;
		const Vector3r Jinv_axis = inertiaInverseW1 * axis;
		const Vector3r Jinv_t1 = inertiaInverseW1 * t1;
		const Vector3r Jinv_t2 = inertiaInverseW1 * t2;

		K.block<3, 3>(0, 0) += K11;
		const Vector3r K_03 = neg_P_r1_star_Jinv * t1;
		const Vector3r K_04 = neg_P_r1_star_Jinv * t2;
		const Vector3r K_05 = neg_P_r1_star_Jinv * axis;
		K.block<3, 1>(0, 3) += K_03;
		K.block<3, 1>(0, 4) += K_04;
		K.block<3, 1>(0, 5) += K_05;
		K.block<1, 3>(3, 0) += K_03.transpose();
		K.block<1, 3>(4, 0) += K_04.transpose();
		K.block<1, 3>(5, 0) += K_05.transpose();
		K(3, 3) += t1.transpose() * Jinv_t1;
		const Real K_34 = t1.transpose() * Jinv_t2;
		K(3, 4) += K_34;
		const Real K_35 = t1.transpose() * Jinv_axis;
		K(3, 5) += K_35;
		K(4, 3) += K_34;
		K(4, 4) += t2.transpose() * Jinv_t2;
		const Real K_45 = t2.transpose() * Jinv_axis;
		K(4, 5) += K_45;
		K(5, 3) += K_35;	
		K(5, 4) += K_45;
		K(5, 5) += axis.transpose() * Jinv_axis;
	}


	const Eigen::Matrix<Real, 6, 6> Kinv = K.inverse();

	const Eigen::Matrix<Real, 6, 1> lambda = Kinv * b;
	const Vector3r pt = lambda.block<3, 1>(0, 0);

	const Vector3r angMomentum = t1*lambda(3, 0) + t2*lambda(4, 0) + axis*lambda(5, 0);

	if (invMass0 != 0.0)
	{
		corr_x0 = invMass0*pt;
		const Vector3r ot = (inertiaInverseW0 * (r0.cross(pt) + angMomentum));
		const Quaternionr otQ(0.0, ot[0], ot[1], ot[2]);
		corr_q0.coeffs() = 0.5 *(otQ*q0).coeffs();
	}

	if (invMass1 != 0.0)
	{
		corr_x1 = -invMass1*pt;
		const Vector3r ot = (inertiaInverseW1 * (r1.cross(-pt) - angMomentum));
		const Quaternionr otQ(0.0, ot[0], ot[1], ot[2]);
		corr_q1.coeffs() = 0.5 *(otQ*q1).coeffs();
	}

	return true;
}


// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::init_TargetVelocityMotorHingeJoint(
	const Vector3r &x0,
	const Quaternionr &q0,
	const Vector3r &x1,
	const Quaternionr &q1,
	const Vector3r &position,
	const Vector3r &direction,
	Eigen::Matrix<Real, 3, 14> &jointInfo
	)
{
	return init_TargetAngleMotorHingeJoint(x0, q0, x1, q1, position, direction, jointInfo);
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::update_TargetVelocityMotorHingeJoint(
	const Vector3r &x0,
	const Quaternionr &q0,
	const Vector3r &x1,
	const Quaternionr &q1,
	Eigen::Matrix<Real, 3, 14> &jointInfo
	)
{
	return update_TargetAngleMotorHingeJoint(x0, q0, x1, q1, jointInfo);
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::solve_TargetVelocityMotorHingeJoint(
	const Real invMass0,
	const Vector3r &x0,
	const Matrix3r &inertiaInverseW0,
	const Quaternionr &q0,
	const Real invMass1,
	const Vector3r &x1,
	const Matrix3r &inertiaInverseW1,
	const Quaternionr &q1,
	const Eigen::Matrix<Real, 3, 14> &jointInfo,
	Vector3r &corr_x0, Quaternionr &corr_q0,
	Vector3r &corr_x1, Quaternionr &corr_q1)
{
	Eigen::Matrix<Real, 3, 12> hingeJointInfo = jointInfo.block<3, 12>(0, 0);
	const bool res = solve_HingeJoint(	invMass0, x0, inertiaInverseW0, q0, 
												invMass1, x1, inertiaInverseW1, q1, 
												hingeJointInfo, corr_x0, corr_q0, corr_x1, corr_q1);
	return res;
}


// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::velocitySolve_TargetVelocityMotorHingeJoint(
	const Real invMass0,
	const Vector3r &x0,
	const Vector3r &v0,
	const Matrix3r &inertiaInverseW0,	
	const Vector3r &omega0,
	const Real invMass1,
	const Vector3r &x1,
	const Vector3r &v1,
	const Matrix3r &inertiaInverseW1,
	const Vector3r &omega1,
	const Real targetAngularVelocity,
	const Eigen::Matrix<Real, 3, 14> &jointInfo,
	Vector3r &corr_v0, Vector3r &corr_omega0,
	Vector3r &corr_v1, Vector3r &corr_omega1)
{
	// jointInfo contains
	// 0:	connector in body 0 (local)
	// 1:	connector in body 1 (local)
	// 2-4:	coordinate system of body 0 (local)
	// 5:	joint axis in body 1 (local)
	// 6:	connector in body 0 (global)
	// 7:	connector in body 1 (global)
	// 8-10:coordinate system of body 0 (global)
	// 11:	joint axis in body 1 (global)
	// 12:	perpendicular vector on joint axis (normalized) in body 1 (local)
	// 13:	perpendicular vector on joint axis (normalized) in body 1 (global)

	const Vector3r &c0 = jointInfo.col(6);
	const Vector3r &c1 = jointInfo.col(7);
	const Vector3r &axis0 = jointInfo.col(8);
	const Vector3r &axis1 = jointInfo.col(11);
	const Vector3r axis = 0.5 * (axis0 + axis1);
	const Vector3r u = axis0.cross(axis1);
	const Vector3r &t1 = jointInfo.col(9);
	const Vector3r &t2 = jointInfo.col(10);
	const Vector3r &t3 = jointInfo.col(13);
	const Vector3r r0 = c0 - x0;
	const Vector3r r1 = c1 - x1;
	Matrix3r r0_star, r1_star;
	MathFunctions::crossProductMatrix(r0, r0_star);
	MathFunctions::crossProductMatrix(r1, r1_star);
	Vector3r deltaOmega = omega1 - omega0;

	Eigen::Matrix<Real, 6, 1> b;
	b.block<3, 1>(0, 0) = v1 - v0;
	b(3, 0) = t1.dot(deltaOmega);
	b(4, 0) = t2.dot(deltaOmega);

	// determine correction angle
	Real delta = targetAngularVelocity;
	delta -= axis.dot(deltaOmega);

	b(5, 0) = -delta;

	Eigen::Matrix<Real, 6, 6> K;
	K.setZero();
	if (invMass0 != 0.0)
	{
		// Jacobian for body 0 is
		//
		// (I_3   -r0*)
		// (0     t1^T)
		// (0     t2^T)
		// (0     axis^T)
		//
		// where I_3 is the identity matrix and r0* is the cross product matrix of r0
		//
		// J M^-1 J^T =
		// ( 1/m I_3-r0 * J0^-1 * r0*    -r0 * J0^-1 * t1    -r0 * J0^-1 * t2		-r0 * J0^-1 * axis)
		// ( (-r0 * J0^-1 * t1)^T        t1^T * J0^-1 t1     t1^T * J0^-1 * t2		t1^T * J0^-1 * axis)
		// ( (-r0 * J0^-1 * t2)^T        t2^T * J0^-1 t1     t2^T * J0^-1 * t2		t2^T * J0^-1 axis)
		// ( (-r0 * J0^-1 * axis)^T      axis^T * J0^-1 t1   axis^T * J0^-1 t2		axis^T * J0^-1 axis)

		Matrix3r K00;
		computeMatrixK(c0, invMass0, x0, inertiaInverseW0, K00);

		K.block<3, 3>(0, 0) = K00;
		K.block<3, 1>(0, 3) = -r0_star * inertiaInverseW0 * t1;
		K.block<3, 1>(0, 4) = -r0_star * inertiaInverseW0 * t2;
		K.block<3, 1>(0, 5) = -r0_star * inertiaInverseW0 * axis;
		K.block<1, 3>(3, 0) = K.block<3, 1>(0, 3).transpose();
		K.block<1, 3>(4, 0) = K.block<3, 1>(0, 4).transpose();
		K.block<1, 3>(5, 0) = K.block<3, 1>(0, 5).transpose();
		K(3, 3) = t1.transpose() * inertiaInverseW0 * t1;
		K(3, 4) = t1.transpose() * inertiaInverseW0 * t2;
		K(3, 5) = t1.transpose() * inertiaInverseW0 * axis;
		K(4, 3) = K(3, 4);
		K(5, 3) = K(3, 5);
		K(4, 4) = t2.transpose() * inertiaInverseW0 * t2;
		K(4, 5) = t2.transpose() * inertiaInverseW0 * axis;
		K(5, 4) = K(4, 5);
		K(5, 5) = axis.transpose() * inertiaInverseW0 * axis;
	}
	if (invMass1 != 0.0)
	{
		// Jacobian for body 1 is
		//
		// ( -I_3   r1*  )
		// ( 0     -t1^T )
		// ( 0     -t2^T )
		// ( 0     axis^T)
		//
		// where I_3 is the identity matrix and r1* is the cross product matrix of r1
		//
		// J M^-1 J^T =
		// ( 1/m I_3-r1 * J1^-1 * r1*    -r1 * J1^-1 * t1    -r1 * J1^-1 * t2		-r1 * J1^-1 * axis)
		// ( (-r1 * J1^-1 * t1)^T        t1^T * J1^-1 t1     t1^T * J1^-1 t2		t1^T * J1^-1 * axis)
		// ( (-r1 * J1^-1 * t2)^T        t2^T * J1^-1 t1     t2^T * J1^-1 t2		t2^T * J1^-1 axis)
		// ( (-r1 * J1^-1 * axis)^T      axis^T * J1^-1 t1   axis^T * J1^-1	t2		axis^T * J1^-1 axis)

		Matrix3r K11;
		computeMatrixK(c1, invMass1, x1, inertiaInverseW1, K11);

		K.block<3, 3>(0, 0) += K11;
		const Vector3r K_03 = -r1_star * inertiaInverseW1 * t1;
		const Vector3r K_04 = -r1_star * inertiaInverseW1 * t2;
		const Vector3r K_05 = -r1_star * inertiaInverseW1 * axis;
		K.block<3, 1>(0, 3) += K_03;
		K.block<3, 1>(0, 4) += K_04;
		K.block<3, 1>(0, 5) += K_05;
		K.block<1, 3>(3, 0) += K_03.transpose();
		K.block<1, 3>(4, 0) += K_04.transpose();
		K.block<1, 3>(5, 0) += K_05.transpose();
		K(3, 3) += t1.transpose() * inertiaInverseW1 * t1;
		const Real K_34 = t1.transpose() * inertiaInverseW1 * t2;
		K(3, 4) += K_34;
		const Real K_35 = t1.transpose() * inertiaInverseW1 * axis;
		K(3, 5) += K_35;
		K(4, 3) += K_34;
		K(4, 4) += t2.transpose() * inertiaInverseW1 * t2;
		const Real K_45 = t2.transpose() * inertiaInverseW1 * axis;
		K(4, 5) += K_45;
		K(5, 3) += K_35;
		K(5, 4) += K_45;
		K(5, 5) += axis.transpose() * inertiaInverseW1 * axis;
	}

	const Eigen::Matrix<Real, 6, 6> Kinv = K.inverse();

	const Eigen::Matrix<Real, 6, 1> lambda = Kinv * b;
	const Vector3r p = lambda.block<3, 1>(0, 0);

	Vector3r angMomentum = t1*lambda(3, 0) + t2*lambda(4, 0) + axis*lambda(5, 0);

	if (invMass0 != 0.0)
	{
		corr_v0 = invMass0*p;
		corr_omega0 = (inertiaInverseW0 * (r0.cross(p) + angMomentum));
	}

	if (invMass1 != 0.0)
	{
		corr_v1 = -invMass1*p;
		corr_omega1 = (inertiaInverseW1 * (r1.cross(-p) - angMomentum));
	}

	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::init_RigidBodyParticleBallJoint(
	const Vector3r &x0,
	const Quaternionr &q0,
	const Vector3r &x1,
	Eigen::Matrix<Real, 3, 2> &jointInfo
	)
{
	// jointInfo contains
	// 0:	connector in rigid body (local)
	// 1:	connector in rigid body (global)

	// transform in local coordinates
	const Matrix3r rot0T = q0.matrix().transpose();

	jointInfo.col(0) = rot0T * (x1 - x0);
	jointInfo.col(1) = x1;

	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::update_RigidBodyParticleBallJoint(
	const Vector3r &x0,
	const Quaternionr &q0,
	const Vector3r &x1,
	Eigen::Matrix<Real, 3, 2> &jointInfo
	)
{
	// jointInfo contains
	// 0:	connector in rigid body (local)
	// 1:	connector in rigid body (global)

	// compute world space position of connector
	const Matrix3r rot0 = q0.matrix();
	jointInfo.col(1) = rot0 * jointInfo.col(0) + x0;

	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::solve_RigidBodyParticleBallJoint(
	const Real invMass0,
	const Vector3r &x0,
	const Matrix3r &inertiaInverseW0,
	const Quaternionr &q0,
	const Real invMass1,
	const Vector3r &x1,
	const Eigen::Matrix<Real, 3, 2> &jointInfo,
	Vector3r &corr_x0, Quaternionr &corr_q0,
	Vector3r &corr_x1)
{
	// jointInfo contains
	// 0:	connector in rigid body (local)
	// 1:	connector in rigid body (global)

	const Vector3r &connector0 = jointInfo.col(1);

	// Compute Kinv
	Matrix3r K1, K2;
	computeMatrixK(connector0, invMass0, x0, inertiaInverseW0, K1);

	K2.setZero();
	if (invMass1 != 0.0)
	{
		K2(0, 0) = invMass1;
		K2(1, 1) = invMass1;
		K2(2, 2) = invMass1;
	}
	const Matrix3r Kinv = (K1 + K2).inverse();

	const Vector3r pt = Kinv * (x1 - connector0);

	if (invMass0 != 0.0)
	{
		const Vector3r r0 = connector0 - x0;
		corr_x0 = invMass0*pt;

		const Vector3r ot = (inertiaInverseW0 * (r0.cross(pt)));
		const Quaternionr otQ(0.0, ot[0], ot[1], ot[2]);
		corr_q0.coeffs() = 0.5 *(otQ*q0).coeffs();
	}

	if (invMass1 != 0.0)
	{
		corr_x1 = -invMass1*pt;
	}

	return true;
}


// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::init_RigidBodyContactConstraint(
	const Real invMass0,							// inverse mass is zero if body is static
	const Vector3r &x0,						// center of mass of body 0
	const Vector3r &v0,						// velocity of body 0
	const Matrix3r &inertiaInverseW0,		// inverse inertia tensor (world space) of body 0
	const Quaternionr &q0,					// rotation of body 0	
	const Vector3r &omega0,					// angular velocity of body 0
	const Real invMass1,							// inverse mass is zero if body is static
	const Vector3r &x1,						// center of mass of body 1
	const Vector3r &v1,						// velocity of body 1
	const Matrix3r &inertiaInverseW1,		// inverse inertia tensor (world space) of body 1
	const Quaternionr &q1,					// rotation of body 1
	const Vector3r &omega1,					// angular velocity of body 1
	const Vector3r &cp0,						// contact point of body 0
	const Vector3r &cp1,						// contact point of body 1
	const Vector3r &normal,					// contact normal in body 1
	const Real restitutionCoeff,					// coefficient of restitution
	Eigen::Matrix<Real, 3, 5> &constraintInfo)
{
	// constraintInfo contains
	// 0:	contact point in body 0 (global)
	// 1:	contact point in body 1 (global)
	// 2:	contact normal in body 1 (global)
	// 3:	contact tangent (global)
	// 0,4:  1.0 / normal^T * K * normal
	// 1,4: maximal impulse in tangent direction
	// 2,4: goal velocity in normal direction after collision

	// compute goal velocity in normal direction after collision
	const Vector3r r0 = cp0 - x0;
	const Vector3r r1 = cp1 - x1;

	const Vector3r u0 = v0 + omega0.cross(r0);
	const Vector3r u1 = v1 + omega1.cross(r1);
	const Vector3r u_rel = u0 - u1;
	const Real u_rel_n = normal.dot(u_rel);

	constraintInfo.col(0) = cp0;
	constraintInfo.col(1) = cp1;
	constraintInfo.col(2) = normal;

	// tangent direction
	Vector3r t = u_rel - u_rel_n*normal;
	Real tl2 = t.squaredNorm();
	if (tl2 > 1.0e-6)
		t *= static_cast<Real>(1.0) / sqrt(tl2);

	constraintInfo.col(3) = t;

	// determine K matrix
	Matrix3r K1, K2;
	computeMatrixK(cp0, invMass0, x0, inertiaInverseW0, K1);
	computeMatrixK(cp1, invMass1, x1, inertiaInverseW1, K2);
	Matrix3r K = K1 + K2;

	constraintInfo(0, 4) = static_cast<Real>(1.0) / (normal.dot(K*normal));

	// maximal impulse in tangent direction
	constraintInfo(1, 4) = static_cast<Real>(1.0) / (t.dot(K*t)) * u_rel.dot(t);

	// goal velocity in normal direction after collision
	constraintInfo(2, 4) = 0.0;
	if (u_rel_n < 0.0)
		constraintInfo(2, 4) = -restitutionCoeff * u_rel_n;

	return true;
}

//--------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::velocitySolve_RigidBodyContactConstraint(
	const Real invMass0,							// inverse mass is zero if body is static
	const Vector3r &x0, 						// center of mass of body 0
	const Vector3r &v0,						// velocity of body 0
	const Matrix3r &inertiaInverseW0,		// inverse inertia tensor (world space) of body 0
	const Vector3r &omega0,					// angular velocity of body 0
	const Real invMass1,							// inverse mass is zero if body is static
	const Vector3r &x1, 						// center of mass of body 1
	const Vector3r &v1,						// velocity of body 1
	const Matrix3r &inertiaInverseW1,		// inverse inertia tensor (world space) of body 1
	const Vector3r &omega1,					// angular velocity of body 1
	const Real stiffness,							// stiffness parameter of penalty impulse
	const Real frictionCoeff,						// friction coefficient
	Real &sum_impulses,							// sum of all impulses
	Eigen::Matrix<Real, 3, 5> &constraintInfo,		// precomputed contact info
	Vector3r &corr_v0, Vector3r &corr_omega0,
	Vector3r &corr_v1, Vector3r &corr_omega1)
{
	// constraintInfo contains
	// 0:	contact point in body 0 (global)
	// 1:	contact point in body 1 (global)
	// 2:	contact normal in body 1 (global)
	// 3:	contact tangent (global)
	// 0,4:  1.0 / normal^T * K * normal
	// 1,4: maximal impulse in tangent direction
	// 2,4: goal velocity in normal direction after collision

	if ((invMass0 == 0.0) && (invMass1 == 0.0))
		return false;

	const Vector3r &connector0 = constraintInfo.col(0);
	const Vector3r &connector1 = constraintInfo.col(1);
	const Vector3r &normal = constraintInfo.col(2);
	const Vector3r &tangent = constraintInfo.col(3);

	// 1.0 / normal^T * K * normal
	const Real nKn_inv = constraintInfo(0, 4);

	// penetration depth 
	const Real d = normal.dot(connector0 - connector1);

	// maximal impulse in tangent direction
	const Real pMax = constraintInfo(1, 4);

	// goal velocity in normal direction after collision
	const Real goal_u_rel_n = constraintInfo(2, 4);

	const Vector3r r0 = connector0 - x0;
	const Vector3r r1 = connector1 - x1;

	const Vector3r u0 = v0 + omega0.cross(r0);
	const Vector3r u1 = v1 + omega1.cross(r1);

	const Vector3r u_rel = u0-u1;
	const Real u_rel_n = u_rel.dot(normal);
	const Real delta_u_reln = goal_u_rel_n - u_rel_n;

	Real correctionMagnitude = nKn_inv * delta_u_reln;

	if (correctionMagnitude < -sum_impulses)
		correctionMagnitude = -sum_impulses;

	// add penalty impulse to counteract penetration
	if (d < 0.0)
		correctionMagnitude -= stiffness * nKn_inv * d;


	Vector3r p(correctionMagnitude * normal);
	sum_impulses += correctionMagnitude;

	// dynamic friction
	const Real pn = p.dot(normal);
	if (frictionCoeff * pn > pMax)
		p -= pMax * tangent;
	else if (frictionCoeff * pn < -pMax)
		p += pMax * tangent;
	else
		p -= frictionCoeff * pn * tangent;

	if (invMass0 != 0.0)
	{
		corr_v0 = invMass0*p;
		corr_omega0 = inertiaInverseW0 * (r0.cross(p));
	}

	if (invMass1 != 0.0)
	{
		corr_v1 = -invMass1*p;
		corr_omega1 = inertiaInverseW1 * (r1.cross(-p));
	}

	return true;
}


// ----------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::init_ParticleRigidBodyContactConstraint(
	const Real invMass0,							// inverse mass is zero if body is static
	const Vector3r &x0,						// center of mass of body 0
	const Vector3r &v0,						// velocity of body 0
	const Real invMass1,							// inverse mass is zero if body is static
	const Vector3r &x1,						// center of mass of body 1
	const Vector3r &v1,						// velocity of body 1
	const Matrix3r &inertiaInverseW1,		// inverse inertia tensor (world space) of body 1
	const Quaternionr &q1,					// rotation of body 1	
	const Vector3r &omega1,					// angular velocity of body 1
	const Vector3r &cp0,						// contact point of body 0
	const Vector3r &cp1,						// contact point of body 1
	const Vector3r &normal,					// contact normal in body 1
	const Real restitutionCoeff,					// coefficient of restitution
	Eigen::Matrix<Real, 3, 5> &constraintInfo)
{
	// constraintInfo contains
	// 0:	contact point in body 0 (global)
	// 1:	contact point in body 1 (global)
	// 2:	contact normal in body 1 (global)
	// 3:	contact tangent (global)
	// 0,4:  1.0 / normal^T * K * normal
	// 1,4: maximal impulse in tangent direction
	// 2,4: goal velocity in normal direction after collision

	// compute goal velocity in normal direction after collision
	const Vector3r r1 = cp1 - x1;

	const Vector3r u1 = v1 + omega1.cross(r1);
	const Vector3r u_rel = v0 - u1;
	const Real u_rel_n = normal.dot(u_rel);

	constraintInfo.col(0) = cp0;
	constraintInfo.col(1) = cp1;
	constraintInfo.col(2) = normal;

	// tangent direction
	Vector3r t = u_rel - u_rel_n*normal;
	Real tl2 = t.squaredNorm();
	if (tl2 > 1.0e-6)
		t *= static_cast<Real>(1.0) / sqrt(tl2);

	constraintInfo.col(3) = t;

	// determine K matrix
	Matrix3r K;
	computeMatrixK(cp1, invMass1, x1, inertiaInverseW1, K);
	if (invMass0 != 0.0)
	{
		K(0, 0) += invMass0;
		K(1, 1) += invMass0;
		K(2, 2) += invMass0;
	}

	constraintInfo(0, 4) = static_cast<Real>(1.0) / (normal.dot(K*normal));

	// maximal impulse in tangent direction
	constraintInfo(1, 4) = static_cast<Real>(1.0) / (t.dot(K*t)) * u_rel.dot(t);

	// goal velocity in normal direction after collision
	constraintInfo(2, 4) = 0.0;
	if (u_rel_n < 0.0)
		constraintInfo(2, 4) = -restitutionCoeff * u_rel_n;

	return true;
}

//--------------------------------------------------------------------------------------------
bool PositionBasedRigidBodyDynamics::velocitySolve_ParticleRigidBodyContactConstraint(
	const Real invMass0,							// inverse mass is zero if body is static
	const Vector3r &x0, 						// center of mass of body 0
	const Vector3r &v0,						// velocity of body 0
	const Real invMass1,							// inverse mass is zero if body is static
	const Vector3r &x1, 						// center of mass of body 1
	const Vector3r &v1,						// velocity of body 1
	const Matrix3r &inertiaInverseW1,		// inverse inertia tensor (world space) of body 1
	const Vector3r &omega1,					// angular velocity of body 1
	const Real stiffness,							// stiffness parameter of penalty impulse
	const Real frictionCoeff,						// friction coefficient
	Real &sum_impulses,							// sum of all impulses
	Eigen::Matrix<Real, 3, 5> &constraintInfo,		// precomputed contact info
	Vector3r &corr_v0,
	Vector3r &corr_v1, Vector3r &corr_omega1)
{
	// constraintInfo contains
	// 0:	contact point in body 0 (global)
	// 1:	contact point in body 1 (global)
	// 2:	contact normal in body 1 (global)
	// 3:	contact tangent (global)
	// 0,4:  1.0 / normal^T * K * normal
	// 1,4: maximal impulse in tangent direction
	// 2,4: goal velocity in normal direction after collision

	if ((invMass0 == 0.0) && (invMass1 == 0.0))
		return false;

	const Vector3r &connector0 = constraintInfo.col(0);
	const Vector3r &connector1 = constraintInfo.col(1);
	const Vector3r &normal = constraintInfo.col(2);
	const Vector3r &tangent = constraintInfo.col(3);

	// 1.0 / normal^T * K * normal
	const Real nKn_inv = constraintInfo(0, 4);

	// penetration depth 
	const Real d = normal.dot(connector0 - connector1);

	// maximal impulse in tangent direction
	const Real pMax = constraintInfo(1, 4);

	// goal velocity in normal direction after collision
	const Real goal_u_rel_n = constraintInfo(2, 4);

	const Vector3r r1 = connector1 - x1;
	const Vector3r u1 = v1 + omega1.cross(r1);

	const Vector3r u_rel = v0 - u1;
	const Real u_rel_n = u_rel.dot(normal);
	const Real delta_u_reln = goal_u_rel_n - u_rel_n;

	Real correctionMagnitude = nKn_inv * delta_u_reln;

	if (correctionMagnitude < -sum_impulses)
		correctionMagnitude = -sum_impulses;

	// add penalty impulse to counteract penetration
	if (d < 0.0)
		correctionMagnitude -= stiffness * nKn_inv * d;


	Vector3r p(correctionMagnitude * normal);
	sum_impulses += correctionMagnitude;

	const Real pn = p.dot(normal);
	if (frictionCoeff * pn > pMax)
		p -= pMax * tangent;
	else if (frictionCoeff * pn < -pMax)
		p += pMax * tangent;
	else
		p -= frictionCoeff * pn * tangent;

	if (invMass0 != 0.0)
	{
		corr_v0 = invMass0*p;		
	}

	if (invMass1 != 0.0)
	{
		corr_v1 = -invMass1*p;
		corr_omega1 = inertiaInverseW1 * (r1.cross(-p));
	}

	return true;
}
