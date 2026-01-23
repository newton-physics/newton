/*---------------------------------------------------------------------
* Copyright(C) 2017-2024 KRN Labs
* Dr Przemyslaw Korzeniowski <korzenio@gmail.com>
* All rights reserved.
*
* This file is part of DefKit.
* It can not be copied and/or distributed
* without the express permission of Dr Przemyslaw Korzeniowski
* -------------------------------------------------------------------*/

#include "ElasticRod.h"

//optimized computation of director d_3 (cheaper than 2 quaternion products)
SIMD_FORCE_INLINE btVector3 d3(const btQuaternion& q)
{
	btVector3 res;
	res[0] = 2.0f * (q.x() * q.z() + q.w() * q.y());
	res[1] = 2.0f * (q.y() * q.z() - q.w() * q.x());
	res[2] = q.w() * q.w() - q.x() * q.x() - q.y() * q.y() + q.z() * q.z();
	return res;
}

SIMD_FORCE_INLINE void ProjectStretchAndShearConstraint(btVector3& pA, btVector3& pB, btQuaternion& q, float wA, float wB, float wQ, float restLength, float kS)
{

	btVector3 gamma = (pB - pA) / restLength - d3(q);
	gamma /= (wA + wB) / restLength + wQ * 4.0f * restLength + 1.0e-6f;
	gamma *= kS;

	btQuaternion q_e_3_bar(-q[1], q[0], -q[3], q[2]);
	btQuaternion dq = btQuaternion(gamma[0], gamma[1], gamma[2], 0) * q_e_3_bar;

	pA += gamma * wA;
	pB -= gamma * wB;

	q += dq * wQ * 2.0f * restLength;
	q.normalize();

}

SIMD_FORCE_INLINE void ProjectStretchAndShearConstraint(const btVector3& pA, const btVector3& pB, const btQuaternion& q, float wA, float wB, float wQ, float restLength, float kS, btVector3& deltaA, btVector3& deltaB, btQuaternion& deltaQ)
{

	btVector3 gamma = (pB - pA) / restLength - d3(q);
	gamma /= (wA + wB) / restLength + wQ * 4.0f * restLength + 1.0e-6f;
	gamma *= kS;

	btQuaternion q_e_3_bar(-q[1], q[0], -q[3], q[2]);
	btQuaternion dq = btQuaternion(gamma[0], gamma[1], gamma[2], 0) * q_e_3_bar;

	deltaA += gamma * wA;
	deltaB -= gamma * wB;

	deltaQ += dq * wQ * 2.0f * restLength;
	deltaQ.normalize();

}



SIMD_FORCE_INLINE void  ProjectBendAndTwistConstraint(btQuaternion& qA, btQuaternion& qB, float wA, float wB, const btQuaternion& restDarboux, float restLength, const btVector3& bendKs)
{

	//compute darboux vector
	btQuaternion omega = qA.inverse() * qB;
	btQuaternion omega_0 = restDarboux;

	btQuaternion omega_plus = omega + omega_0;//delta Omega with -Omega_0
	omega = omega - omega_0; //delta Omega

	if (omega.length2() > omega_plus.length2())
		omega = omega_plus;

	omega[0] *= bendKs[0]; //bending stiffness factor
	omega[1] *= bendKs[1];
	omega[2] *= bendKs[2];
	omega[3] = 0.0f;    //discrete Darboux vector does not have vanishing scalar part
	omega /= (wA + wB);

	qA += qB * omega * wA;
	qB -= qA * omega * wB;

	qA.normalize();
	qB.normalize();

}

SIMD_FORCE_INLINE void  ProjectBendAndTwistConstraint(const btQuaternion& qA, const btQuaternion& qB, float wA, float wB, const btQuaternion& restDarboux, float restLength, const btVector3& bendKs, btQuaternion& deltaA, btQuaternion& deltaB)
{

	//compute darboux vector
	btQuaternion omega = qA.inverse() * qB;
	btQuaternion omega_0 = restDarboux;

	btQuaternion omega_plus = omega + omega_0;//delta Omega with -Omega_0
	omega = omega - omega_0; //delta Omega

	if (omega.length2() > omega_plus.length2())
		omega = omega_plus;

	omega[0] *= bendKs[0]; //bending stiffness factor
	omega[1] *= bendKs[1];
	omega[2] *= bendKs[2];
	omega[3] = 0.0f;    //discrete Darboux vector does not have vanishing scalar part
	omega /= (wA + wB);

	deltaA += qB * omega * wA;
	deltaB -= qA * omega * wB;

	deltaA.normalize();
	deltaB.normalize();

}

extern "C" {


	void EXPORT_API ProjectStretchAndShearConstraint(btVector3& positionA, btVector3& positionB, btQuaternion& orientation, float invMassA, float invMassB, float quatInvMass, float restLength, float stretchKs, float shearKs)
	{
		ProjectStretchAndShearConstraint(positionA, positionB, orientation, invMassA, invMassB, quatInvMass, restLength, stretchKs);
	}

	void EXPORT_API ProjectBendAndTwistConstraint(btQuaternion& qA, btQuaternion& qB, float wA, float wB, btQuaternion& restDarboux, float restLength, btVector3& bendKs, float asd)
	{
		ProjectBendAndTwistConstraint(qA, qB, wA, wB, restDarboux, restLength, bendKs);
	}



	void EXPORT_API ProjectElasticRodConstraints(int pointsCount, btVector3* positions, btQuaternion* orientations, float* invMasses, float* quatInvMasses, btQuaternion* restDarboux, btVector3*  bendAndTwistKs, float* restLength, float stretchKs, float shearKs)
	{
		int left = 0;
		int right = pointsCount - 2;

		for (size_t i = 0; i < pointsCount - 1; i++)
		{
			ProjectStretchAndShearConstraint(positions[i], positions[i + 1], orientations[i], invMasses[i], invMasses[i + 1], quatInvMasses[i], restLength[i], stretchKs);
		}

		left = 0;
		right = pointsCount - 2;
		for (; left < right; ++left, --right)
		{
			ProjectBendAndTwistConstraint(orientations[left], orientations[left + 1], quatInvMasses[left], quatInvMasses[left + 1], restDarboux[left], restLength[left], bendAndTwistKs[left] * shearKs);
			ProjectBendAndTwistConstraint(orientations[right], orientations[right + 1], quatInvMasses[right], quatInvMasses[right + 1], restDarboux[right], restLength[right], bendAndTwistKs[right] * shearKs);
		}

		if (left == right)
		{
			ProjectBendAndTwistConstraint(orientations[left], orientations[left + 1], quatInvMasses[left], quatInvMasses[left + 1], restDarboux[left], restLength[left], bendAndTwistKs[left] * shearKs);
		}
	}

	//void EXPORT_API ProjectStretchAndShearConstraints(int pointsCount, btVector3* positions, btQuaternion* orientations, float* invMasses, float* quatInvMasses, float restLength, float stretchKs, float shearKs)
	//{

	//	int left = 0;
	//	int right = pointsCount - 2;
	//	for (; left < right; ++left, --right)
	//	{
	//		ProjectDistanceConstraint(positions[left], positions[left + 1], invMasses[left], invMasses[left + 1], restLength, stretchKs);
	//		ProjectDistanceConstraint(positions[right], positions[right + 1], invMasses[right], invMasses[right + 1], restLength, stretchKs);

	//	}

	//	if (left == right)
	//	{
	//		ProjectDistanceConstraint(positions[left], positions[left + 1], invMasses[left], invMasses[left + 1], restLength, stretchKs);
	//	}


	//	left = 0;
	//	right = pointsCount - 2;
	//	for (; left < right; ++left, --right)
	//	{
	//		ProjectStretchAndShearConstraint(positions[left], positions[left + 1], orientations[left], invMasses[left], invMasses[left + 1], quatInvMasses[left], restLength, shearKs);
	//		ProjectStretchAndShearConstraint(positions[right], positions[right + 1], orientations[right], invMasses[right], invMasses[right + 1], quatInvMasses[right], restLength, shearKs);
	//	}

	//	if (left == right)
	//	{
	//		ProjectStretchAndShearConstraint(positions[left], positions[left + 1], orientations[left], invMasses[left], invMasses[left + 1], quatInvMasses[left], restLength, shearKs);
	//	}


	//}

	//void EXPORT_API ProjectBendAndTwistConstraints(int pointsCount, btQuaternion* orientations, float* quatInvMasses, float restLength, btQuaternion* restDarboux, btVector3* bendAndTwistKs)
	//{
	//	int left = 0;
	//	int right = pointsCount - 2;
	//	for (; left < right; ++left, --right)
	//	{
	//		ProjectBendAndTwistConstraint(orientations[left], orientations[left + 1], quatInvMasses[left], quatInvMasses[left + 1], restDarboux[left], restLength, bendAndTwistKs[left]);
	//		ProjectBendAndTwistConstraint(orientations[right], orientations[right + 1], quatInvMasses[right], quatInvMasses[right + 1], restDarboux[right], restLength, bendAndTwistKs[right]);
	//	}

	//	if (left == right)
	//	{
	//		ProjectBendAndTwistConstraint(orientations[left], orientations[left + 1], quatInvMasses[left], quatInvMasses[left + 1], restDarboux[left], restLength, bendAndTwistKs[left]);
	//	}
	//}
}