#include "MathFunctions.h"
#include <cfloat>

using namespace PBD;

//////////////////////////////////////////////////////////////////////////
// MathFunctions
//////////////////////////////////////////////////////////////////////////

// ----------------------------------------------------------------------------------------------
void MathFunctions::jacobiRotate(Matrix3r &A, Matrix3r &R, int p, int q)
{
	// rotates A through phi in pq-plane to set A(p,q) = 0
	// rotation stored in R whose columns are eigenvectors of A
	if (A(p, q) == 0.0)
		return;

	Real d = (A(p, p) - A(q, q)) / (static_cast<Real>(2.0)*A(p, q));
	Real t = static_cast<Real>(1.0) / (fabs(d) + sqrt(d*d + static_cast<Real>(1.0)));
	if (d < 0.0) t = -t;
	Real c = static_cast<Real>(1.0) / sqrt(t*t + 1);
	Real s = t*c;
	A(p, p) += t*A(p, q);
	A(q, q) -= t*A(p, q);
	A(p, q) = A(q, p) = 0.0;
	// transform A
	int k;
	for (k = 0; k < 3; k++) {
		if (k != p && k != q) {
			Real Akp = c*A(k, p) + s*A(k, q);
			Real Akq = -s*A(k, p) + c*A(k, q);
			A(k, p) = A(p, k) = Akp;
			A(k, q) = A(q, k) = Akq;
		}
	}
	// store rotation in R
	for (k = 0; k < 3; k++) {
		Real Rkp = c*R(k, p) + s*R(k, q);
		Real Rkq = -s*R(k, p) + c*R(k, q);
		R(k, p) = Rkp;
		R(k, q) = Rkq;
	}
}

// ----------------------------------------------------------------------------------------------
void MathFunctions::eigenDecomposition(const Matrix3r &A, Matrix3r &eigenVecs, Vector3r &eigenVals)
{
	const int numJacobiIterations = 10;
	const Real epsilon = static_cast<Real>(1e-15);

	Matrix3r D = A;

	// only for symmetric matrices!
	eigenVecs.setIdentity();	// unit matrix
	int iter = 0;
	while (iter < numJacobiIterations) {	// 3 off diagonal elements
		// find off diagonal element with maximum modulus
		int p, q;
		Real a, max;
		max = fabs(D(0, 1));
		p = 0; q = 1;
		a = fabs(D(0, 2));
		if (a > max) { p = 0; q = 2; max = a; }
		a = fabs(D(1, 2));
		if (a > max) { p = 1; q = 2; max = a; }
		// all small enough -> done
		if (max < epsilon) break;
		// rotate matrix with respect to that element
		jacobiRotate(D, eigenVecs, p, q);
		iter++;
	}
	eigenVals[0] = D(0, 0);
	eigenVals[1] = D(1, 1);
	eigenVals[2] = D(2, 2);
}

/** Perform polar decomposition A = (U D U^T) R
*/
void MathFunctions::polarDecomposition(const Matrix3r &A, Matrix3r &R, Matrix3r &U, Matrix3r &D)
{
	// A = SR, where S is symmetric and R is orthonormal
	// -> S = (A A^T)^(1/2)

	// A = U D U^T R

	Matrix3r AAT;
	AAT(0, 0) = A(0, 0)*A(0, 0) + A(0, 1)*A(0, 1) + A(0, 2)*A(0, 2);
	AAT(1, 1) = A(1, 0)*A(1, 0) + A(1, 1)*A(1, 1) + A(1, 2)*A(1, 2);
	AAT(2, 2) = A(2, 0)*A(2, 0) + A(2, 1)*A(2, 1) + A(2, 2)*A(2, 2);

	AAT(0, 1) = A(0, 0)*A(1, 0) + A(0, 1)*A(1, 1) + A(0, 2)*A(1, 2);
	AAT(0, 2) = A(0, 0)*A(2, 0) + A(0, 1)*A(2, 1) + A(0, 2)*A(2, 2);
	AAT(1, 2) = A(1, 0)*A(2, 0) + A(1, 1)*A(2, 1) + A(1, 2)*A(2, 2);

	AAT(1, 0) = AAT(0, 1);
	AAT(2, 0) = AAT(0, 2);
	AAT(2, 1) = AAT(1, 2);

	R.setIdentity();
	Vector3r eigenVals;
	eigenDecomposition(AAT, U, eigenVals);

	Real d0 = sqrt(eigenVals[0]);
	Real d1 = sqrt(eigenVals[1]);
	Real d2 = sqrt(eigenVals[2]);
	D.setZero();
	D(0, 0) = d0;
	D(1, 1) = d1;
	D(2, 2) = d2;

	const Real eps = static_cast<Real>(1e-15);

	Real l0 = eigenVals[0]; if (l0 <= eps) l0 = 0.0; else l0 = static_cast<Real>(1.0) / d0;
	Real l1 = eigenVals[1]; if (l1 <= eps) l1 = 0.0; else l1 = static_cast<Real>(1.0) / d1;
	Real l2 = eigenVals[2]; if (l2 <= eps) l2 = 0.0; else l2 = static_cast<Real>(1.0) / d2;

	Matrix3r S1;
	S1(0, 0) = l0*U(0, 0)*U(0, 0) + l1*U(0, 1)*U(0, 1) + l2*U(0, 2)*U(0, 2);
	S1(1, 1) = l0*U(1, 0)*U(1, 0) + l1*U(1, 1)*U(1, 1) + l2*U(1, 2)*U(1, 2);
	S1(2, 2) = l0*U(2, 0)*U(2, 0) + l1*U(2, 1)*U(2, 1) + l2*U(2, 2)*U(2, 2);

	S1(0, 1) = l0*U(0, 0)*U(1, 0) + l1*U(0, 1)*U(1, 1) + l2*U(0, 2)*U(1, 2);
	S1(0, 2) = l0*U(0, 0)*U(2, 0) + l1*U(0, 1)*U(2, 1) + l2*U(0, 2)*U(2, 2);
	S1(1, 2) = l0*U(1, 0)*U(2, 0) + l1*U(1, 1)*U(2, 1) + l2*U(1, 2)*U(2, 2);

	S1(1, 0) = S1(0, 1);
	S1(2, 0) = S1(0, 2);
	S1(2, 1) = S1(1, 2);

	R = S1*A;

	// stabilize
	Vector3r c0, c1, c2;
	c0 = R.col(0);
	c1 = R.col(1);
	c2 = R.col(2);

	if (c0.squaredNorm() < eps)
		c0 = c1.cross(c2);
	else if (c1.squaredNorm() < eps)
		c1 = c2.cross(c0);
	else
		c2 = c0.cross(c1);
	R.col(0) = c0;
	R.col(1) = c1;
	R.col(2) = c2;
}

/** Return the one norm of the matrix.
*/
Real MathFunctions::oneNorm(const Matrix3r &A)
{
	const Real sum1 = fabs(A(0,0)) + fabs(A(1,0)) + fabs(A(2,0));
	const Real sum2 = fabs(A(0,1)) + fabs(A(1,1)) + fabs(A(2,1));
	const Real sum3 = fabs(A(0,2)) + fabs(A(1,2)) + fabs(A(2,2));
	Real maxSum = sum1;
	if (sum2 > maxSum)
		maxSum = sum2;
	if (sum3 > maxSum)
		maxSum = sum3;
	return maxSum;
}

/** Return the inf norm of the matrix.
*/
Real MathFunctions::infNorm(const Matrix3r &A)
{
	const Real sum1 = fabs(A(0, 0)) + fabs(A(0, 1)) + fabs(A(0, 2));
	const Real sum2 = fabs(A(1, 0)) + fabs(A(1, 1)) + fabs(A(1, 2));
	const Real sum3 = fabs(A(2, 0)) + fabs(A(2, 1)) + fabs(A(2, 2));
	Real maxSum = sum1;
	if (sum2 > maxSum)
		maxSum = sum2;
	if (sum3 > maxSum)
		maxSum = sum3;
	return maxSum;
}

/** Perform a polar decomposition of matrix M and return the rotation matrix R. This method handles the degenerated cases.
*/
void MathFunctions::polarDecompositionStable(const Matrix3r &M, const Real tolerance, Matrix3r &R)
{
	Matrix3r Mt = M.transpose();
	Real Mone = oneNorm(M);
	Real Minf = infNorm(M);
	Real Eone;
	Matrix3r MadjTt, Et;
	do
	{
		MadjTt.row(0) = Mt.row(1).cross(Mt.row(2));
		MadjTt.row(1) = Mt.row(2).cross(Mt.row(0));
		MadjTt.row(2) = Mt.row(0).cross(Mt.row(1));

		Real det = Mt(0,0) * MadjTt(0,0) + Mt(0,1) * MadjTt(0,1) + Mt(0,2) * MadjTt(0,2);

		if (fabs(det) < 1.0e-12)
		{
			Vector3r len;
			unsigned int index = 0xffffffff;
			for (unsigned int i = 0; i < 3; i++)
			{
				len[i] = MadjTt.row(i).squaredNorm();
				if (len[i] > 1.0e-12)
				{
					// index of valid cross product
					// => is also the index of the vector in Mt that must be exchanged
					index = i;
					break;
				}
			}
			if (index == 0xffffffff)
			{
				R.setIdentity();
				return;
			}
			else
			{
				Mt.row(index) = Mt.row((index + 1) % 3).cross(Mt.row((index + 2) % 3));
				MadjTt.row((index + 1) % 3) = Mt.row((index + 2) % 3).cross(Mt.row(index));
				MadjTt.row((index + 2) % 3) = Mt.row(index).cross(Mt.row((index + 1) % 3));
				Matrix3r M2 = Mt.transpose();
				Mone = oneNorm(M2);
				Minf = infNorm(M2);
				det = Mt(0,0) * MadjTt(0,0) + Mt(0,1) * MadjTt(0,1) + Mt(0,2) * MadjTt(0,2);
			}
		}

		const Real MadjTone = oneNorm(MadjTt);
		const Real MadjTinf = infNorm(MadjTt);

		const Real gamma = sqrt(sqrt((MadjTone*MadjTinf) / (Mone*Minf)) / fabs(det));

		const Real g1 = gamma* static_cast<Real>(0.5);
		const Real g2 = static_cast<Real>(0.5) / (gamma*det);

		for (unsigned char i = 0; i < 3; i++)
		{
			for (unsigned char j = 0; j < 3; j++)
			{
				Et(i,j) = Mt(i,j);
				Mt(i,j) = g1*Mt(i,j) + g2*MadjTt(i,j);
				Et(i,j) -= Mt(i,j);
			}
		}

		Eone = oneNorm(Et);

		Mone = oneNorm(Mt);
		Minf = infNorm(Mt);
	} while (Eone > Mone * tolerance);

	// Q = Mt^T 
	R = Mt.transpose();
}

/** Perform a singular value decomposition of matrix A: A = U * sigma * V^T.
* This function returns two proper rotation matrices U and V^T which do not 
* contain a reflection. Reflections are corrected by the inversion handling
* proposed by Irving et al. 2004.
*/
void MathFunctions::svdWithInversionHandling(const Matrix3r &A, Vector3r &sigma, Matrix3r &U, Matrix3r &VT)
{

	Matrix3r AT_A, V;
	AT_A = A.transpose() * A;

	Vector3r S;

	// Eigen decomposition of A^T * A
	eigenDecomposition(AT_A, V, S);

	// Detect if V is a reflection .
	// Make a rotation out of it by multiplying one column with -1.
	const Real detV = V.determinant();
	if (detV < 0.0)
	{
		Real minLambda = REAL_MAX;
		unsigned char pos = 0;
		for (unsigned char l = 0; l < 3; l++)
		{
			if (S[l] < minLambda)
			{
				pos = l;
				minLambda = S[l];
			}
		}
		V(0, pos) = -V(0, pos);
		V(1, pos) = -V(1, pos);
		V(2, pos) = -V(2, pos);
	}

	if (S[0] < 0.0) S[0] = 0.0;		// safety for sqrt
	if (S[1] < 0.0) S[1] = 0.0;
	if (S[2] < 0.0) S[2] = 0.0;

	sigma[0] = sqrt(S[0]);
	sigma[1] = sqrt(S[1]);
	sigma[2] = sqrt(S[2]);

	VT = V.transpose();

	//
	// Check for values of hatF near zero
	//
	unsigned char chk = 0;
	unsigned char pos = 0;
	for (unsigned char l = 0; l < 3; l++)
	{
		if (fabs(sigma[l]) < 1.0e-4)
		{
			pos = l;
			chk++;
		}
	}

	if (chk > 0)
	{
		if (chk > 1)
		{
			U.setIdentity();
		}
		else
		{
			U = A * V;
			for (unsigned char l = 0; l < 3; l++)
			{
				if (l != pos)
				{
					for (unsigned char m = 0; m < 3; m++)
					{
						U(m, l) *= static_cast<Real>(1.0) / sigma[l];
					}
				}
			}

			Vector3r v[2];
			unsigned char index = 0;
			for (unsigned char l = 0; l < 3; l++)
			{
				if (l != pos)
				{
					v[index++] = Vector3r(U(0, l), U(1, l), U(2, l));
				}
			}
			Vector3r vec = v[0].cross(v[1]);
			vec.normalize();
			U(0, pos) = vec[0];
			U(1, pos) = vec[1];
			U(2, pos) = vec[2];
		}
	}
	else
	{
		Vector3r sigmaInv(static_cast<Real>(1.0) / sigma[0], static_cast<Real>(1.0) / sigma[1], static_cast<Real>(1.0) / sigma[2]);
		U = A * V;
		for (unsigned char l = 0; l < 3; l++)
		{
			for (unsigned char m = 0; m < 3; m++)
			{
				U(m, l) *= sigmaInv[l];
			}
		}
	}

	const Real detU = U.determinant();

	// U is a reflection => inversion
	if (detU < 0.0)
	{
		//std::cout << "Inversion!\n";
		Real minLambda = REAL_MAX;
		unsigned char pos = 0;
		for (unsigned char l = 0; l < 3; l++)
		{
			if (sigma[l] < minLambda)
			{
				pos = l;
				minLambda = sigma[l];
			}
		}

		// invert values of smallest singular value
		sigma[pos] = -sigma[pos];
		U(0, pos) = -U(0, pos);
		U(1, pos) = -U(1, pos);
		U(2, pos) = -U(2, pos);
	}
}

// ----------------------------------------------------------------------------------------------
Real MathFunctions::cotTheta(const Vector3r &v, const Vector3r &w)
{
	const Real cosTheta = v.dot(w);
	const Real sinTheta = (v.cross(w)).norm();
	return (cosTheta / sinTheta);
}

// ----------------------------------------------------------------------------------------------
void MathFunctions::crossProductMatrix(const Vector3r &v, Matrix3r &v_hat)
{
	v_hat << 0, -v(2), v(1),
		v(2), 0, -v(0),
		-v(1), v(0), 0;
}

// ----------------------------------------------------------------------------------------------
void MathFunctions::extractRotation(const Matrix3r &A, Quaternionr &q,	const unsigned int maxIter)
{
	for (unsigned int iter = 0; iter < maxIter; iter++)
	{
		Matrix3r R = q.matrix();
		Vector3r omega = (R.col(0).cross(A.col(0)) + R.col(1).cross(A.col(1)) + R.col(2).cross(A.col(2))) * 
			(1.0 / fabs(R.col(0).dot(A.col(0)) + R.col(1).dot(A.col(1)) + R.col(2).dot(A.col(2)) + 1.0e-9));
		Real w = omega.norm();
		if (w < 1.0e-9)
			break;
		q = Quaternionr(AngleAxisr(w, (1.0 / w) * omega)) *	q;
		q.normalize();
	}
}