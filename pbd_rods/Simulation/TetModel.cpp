#include "TetModel.h"
#include "PositionBasedDynamics/PositionBasedDynamics.h"
#include <iostream>
#include "Utils/Logger.h"

using namespace PBD;
using namespace Utilities;

TetModel::TetModel() :
	m_surfaceMesh(),
	m_visVertices(),
	m_visMesh(),
	m_particleMesh()	
{
	m_initialX.setZero();
	m_initialR.setIdentity();
	m_initialScale.setOnes();
	m_restitutionCoeff = static_cast<Real>(0.6);
	m_frictionCoeff = static_cast<Real>(0.2);
}

TetModel::~TetModel(void)
{
	cleanupModel();
}

void TetModel::cleanupModel()
{
	m_particleMesh.release();
}

TetModel::SurfaceMesh &TetModel::getSurfaceMesh()
{
	return m_surfaceMesh;
}

VertexData & TetModel::getVisVertices()
{
	return m_visVertices;
}

TetModel::SurfaceMesh &TetModel::getVisMesh()
{
	return m_visMesh;
}

void TetModel::initMesh(const unsigned int nPoints, const unsigned int nTets, const unsigned int indexOffset, unsigned int* indices)
{
	m_indexOffset = indexOffset;
	m_particleMesh.release();
	m_particleMesh.initMesh(nPoints, nTets * 6, nTets * 4, nTets);

	for (unsigned int i = 0; i < nTets; i++)
	{
		m_particleMesh.addTet(&indices[4 * i]);
	}
	m_particleMesh.buildNeighbors();

	createSurfaceMesh();
}

unsigned int TetModel::getIndexOffset() const
{
	return m_indexOffset;
}

void TetModel::createSurfaceMesh()
{
	const unsigned int nVerts = m_particleMesh.numVertices();

	m_surfaceMesh.initMesh(nVerts, m_particleMesh.numEdges(), m_particleMesh.numFaces());

	// Search for all border faces of the tet mesh
	const IndexedTetMesh::Face *faceData = m_particleMesh.getFaceData().data();
	const unsigned int *faces = m_particleMesh.getFaces().data();
	for (unsigned int i = 0; i < m_particleMesh.numFaces(); i++)
	{
		const IndexedTetMesh::Face &face = faceData[i];
		// Found border face
		if ((face.m_tets[1] == 0xffffffff) || (face.m_tets[0] == 0xffffffff))
		{
			m_surfaceMesh.addFace(&faces[3 * i]);
		}
	}
	m_surfaceMesh.buildNeighbors();
}

void TetModel::updateMeshNormals(const ParticleData &pd)
{
	m_surfaceMesh.updateNormals(pd, m_indexOffset);
	m_surfaceMesh.updateVertexNormals(pd);
}

void TetModel::attachVisMesh(const ParticleData &pd)
{
	const Real eps = static_cast<Real>(1.0e-6);

	// The created surface mesh defines the boundary of the tet mesh
	unsigned int *faces = m_surfaceMesh.getFaces().data();
	const unsigned int nFaces = m_surfaceMesh.numFaces();

 	const Vector3r *normals = m_surfaceMesh.getVertexNormals().data();

	// for each point find nearest triangle (TODO: optimize)
	const int nNearstT = 15;
	m_attachments.resize(m_visVertices.size());

	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)  
		for (int i = 0; i < (int)m_visVertices.size(); i++)
		{
			const Vector3r &p = m_visVertices.getPosition(i);
			Real curDist[nNearstT];
			int curT[nNearstT];
			for (int k = 0; k < nNearstT; k++)
			{
				curDist[k] = REAL_MAX;
				curT[k] = -1;
			}
			Vector3r curBary[nNearstT];
			Vector3r curInter[nNearstT];
			for (unsigned int j = 0; j < nFaces; j++)
			{
				const unsigned int indexA = faces[3 * j] + m_indexOffset;
				const unsigned int indexB = faces[3 * j + 1] + m_indexOffset;
				const unsigned int indexC = faces[3 * j + 2] + m_indexOffset;
				const Vector3r & a = pd.getPosition0(indexA);
				const Vector3r & b = pd.getPosition0(indexB);
				const Vector3r & c = pd.getPosition0(indexC);

				Vector3r inter, bary;
				// compute nearest point on triangle
				if (pointInTriangle(a, b, c, p, inter, bary))
				{
					Real len = (p - inter).norm();
					for (int k = nNearstT - 1; k >= 0; k--) // update the best triangles
					{
						if (len < curDist[k])
						{
							if (k < nNearstT - 1)
							{
								curDist[k + 1] = curDist[k];
								curBary[k + 1] = curBary[k];
								curT[k + 1] = curT[k];
								curInter[k + 1] = curInter[k];
							}
							curDist[k] = len;
							curBary[k] = bary;
							curT[k] = (int)j;
							curInter[k] = inter;
						}
					}
				}
			}
			if (curT[0] == -1)
			{
				LOG_ERR << "ERROR: vertex has no nearest triangle.";
				continue;
			}

			// take the best bary coords we find from the best 5 triangles
			Real error = REAL_MAX;
			int current_k = 0;
			Real current_dist = 0.0;
			Vector3r current_bary;
			for (int k = 0; k < nNearstT; k++)
			{
				if (curT[k] == -1)
					break;

				// see Kobbelt: Multiresolution Herarchies on unstructured triangle meshes
				const Vector3r& p = m_visVertices.getPosition(i);
				const Vector3r n1 = -normals[faces[3 * curT[k] + 0]];
				const Vector3r n2 = -normals[faces[3 * curT[k] + 1]];
				const Vector3r n3 = -normals[faces[3 * curT[k] + 2]];
				const Vector3r& p1 = pd.getPosition0(faces[3 * curT[k] + 0] + m_indexOffset);
				const Vector3r& p2 = pd.getPosition0(faces[3 * curT[k] + 1] + m_indexOffset);
				const Vector3r& p3 = pd.getPosition0(faces[3 * curT[k] + 2] + m_indexOffset);
				const Vector3r U = p.cross(n1);
				const Vector3r V = p.cross(n2);
				const Vector3r W = p.cross(n3);
				const Vector3r UU = n1.cross(p1);
				const Vector3r VV = n2.cross(p2);
				const Vector3r WW = n3.cross(p3);
				const Vector3r UV = (n2.cross(p1)) + (n1.cross(p2));
				const Vector3r UW = (n3.cross(p1)) + (n1.cross(p3));
				const Vector3r VW = (n3.cross(p2)) + (n2.cross(p3));
				// F(u,v) = F + Fu*u + Fv*v + Fuu*u*u + Fuv*u*v + Fvv*v*v == 0!
				const Vector3r F = W + WW;
				const Vector3r Fu = U + UW - W - WW * 2.0;
				const Vector3r Fv = V + VW - W - WW * 2.0;
				const Vector3r Fuu = UU - UW + WW;
				const Vector3r Fuv = UV - UW - VW + WW * 2.0;
				const Vector3r Fvv = VV - VW + WW;
				Real u = curBary[k][0];
				Real v = curBary[k][0];
				solveQuadraticForZero(F, Fu, Fv, Fuu, Fuv, Fvv, u, v);
				Real w = static_cast<Real>(1.0) - u - v;

				if (u < 0) u = 0.0;
				if (u > 1) u = 1.0;
				if (v < 0) v = 0.0;
				if (v > 1) v = 1.0;
				if (u + v > 1)
				{
					Real uv = u + v;
					Real u_ = u;
					Real v_ = v;
					u -= (uv - static_cast<Real>(1.0))*v_ / uv;
					v -= (uv - static_cast<Real>(1.0))*u_ / uv;
				}
				w = static_cast<Real>(1.0) - u - v;
				Vector3r curInter = p1*u + p2*v + p3*w;
				Real dist = (p - curInter).norm();

				Vector3r n = n1*u + n2*v + n3*w;
				Real err = dist;
				if ((p - curInter).dot(n) < 0.0)
					dist *= -1.0;
				Vector3r interP = curInter + n*dist;
				err += (interP - p).norm();

				if (err > error)
					continue;

				error = err;

				current_k = k;
				current_dist = dist;
				current_bary = Vector3r(u, v, w);

				if (error < eps)
					break;
			}


			Attachment &fp = m_attachments[i];
			fp.m_index = i;
			fp.m_triIndex = (unsigned int)curT[current_k];
			fp.m_bary[0] = current_bary.x();
			fp.m_bary[1] = current_bary.y();
			fp.m_bary[2] = current_bary.z();
			fp.m_dist = current_dist;
			fp.m_minError = error;
		}
	}
}
 
void TetModel::solveQuadraticForZero(const Vector3r& F, const Vector3r& Fu, const Vector3r& Fv, const Vector3r& Fuu,
	const Vector3r&Fuv, const Vector3r& Fvv, Real& u, Real& v)
{
	// newton iterations search F(u,v) = [0,0,0]
	Real eps = static_cast<Real>(1.0e-6);
	unsigned char k;
	for (k = 0; k < 50; k++)
	{
		// x(n+1) = x(n) - F'^(-1)(x(n))*F(x(n))
		// dx = -F'^(-1)*F => dF*dx = -F
		// => dF^T*dF*dx = dF^T*(-F)
		// solve for dx
		const Vector3r f = -(F + Fu*u + Fv*v + Fuu*u*u + Fuv*u*v + Fvv*v*v);
		if ((fabs(f[0]) < eps) && (fabs(f[1]) < eps) && (fabs(f[2]) < eps))
			break;
		Vector3r dF[2];
		dF[0] = Fu + Fuu*(u * 2) + Fuv*v;
		dF[1] = Fv + Fvv*(v * 2) + Fuv*u;
		Real dFdF[3];
		dFdF[0] = dF[0].dot(dF[0]);
		dFdF[1] = dF[0].dot(dF[1]);
		dFdF[2] = dF[1].dot(dF[1]);
		const Real det = dFdF[0] * dFdF[2] - dFdF[1] * dFdF[1];
		if (fabs(det) < eps)
			break;
		Real H[3];
		H[0] = dFdF[2] / det;
		H[1] = -dFdF[1] / det;
		H[2] = dFdF[0] / det;
		const Real h1 = dF[0].dot(f);
		const Real h2 = dF[1].dot(f);
		u += H[0] * h1 + H[1] * h2;
		v += H[1] * h1 + H[2] * h2;
	}
}
 
void TetModel::updateVisMesh(const ParticleData &pd)
{
	if (m_attachments.size() == 0)
		return;

	// The collision mesh is the boundary of the tet mesh
	unsigned int *faces = m_surfaceMesh.getFaces().data();
	const unsigned int nFaces = m_surfaceMesh.numFaces();

	const Vector3r *normals = m_surfaceMesh.getVertexNormals().data();

	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)  
		for (int i = 0; i < (int) m_attachments.size(); i++)
		{
			const unsigned int pindex = m_attachments[i].m_index;
			const unsigned int triindex = m_attachments[i].m_triIndex;
			const Real *bary = m_attachments[i].m_bary;

			const unsigned int indexA = faces[3 * triindex] + m_indexOffset;
			const unsigned int indexB = faces[3 * triindex + 1] + m_indexOffset;
			const unsigned int indexC = faces[3 * triindex + 2] + m_indexOffset;

			const Vector3r &a = pd.getPosition(indexA);
			const Vector3r &b = pd.getPosition(indexB);
			const Vector3r &c = pd.getPosition(indexC);
			Vector3r p2 = bary[0] * a + bary[1] * b + bary[2] * c;
			Vector3r n = bary[0] * normals[faces[3 * triindex]] + bary[1] * normals[faces[3 * triindex + 1]] + bary[2] * normals[faces[3 * triindex + 2]];
			n.normalize();

			Vector3r &p = m_visVertices.getPosition(pindex);
			p = p2 - n*m_attachments[i].m_dist;
		}
	}

	m_visMesh.updateNormals(m_visVertices, 0);
	m_visMesh.updateVertexNormals(m_visVertices);
}
 
 
 bool TetModel::pointInTriangle(const Vector3r& p0, const Vector3r& p1, const Vector3r& p2, const Vector3r& p,
 	Vector3r& inter, Vector3r &bary)
 {
 	// see Bridson: Robust treatment of collisions contact and friction for cloth animation
 	const Vector3r x43 = p - p2;
 	const Vector3r x13 = p0 - p2;
 	const Vector3r x23 = p1 - p2;
 
 	// compute inv matrix a,b,b,c
 	Real a = x13.dot(x13);
 	Real b = x13.dot(x23);
 	Real c = x23.dot(x23);
 	const Real det = a*c - b*b;
 	if (fabs(det) < 1.0e-9)
 		return false;
 
	Real d1 = x13.dot(x43);
 	Real d2 = x23.dot(x43);
 
 	Real w1 = (c*d1 - b*d2) / det;
 	Real w2 = (a*d2 - b*d1) / det;
 
 	// this clamping gives not an exact orthogonal point to the edge!!
 	if (w1 < 0) w1 = 0;
 	if (w1 > 1) w1 = 1;
 	if (w2 < 0) w2 = 0;
 	if (w2 > 1) w2 = 1;
 
 	bary[0] = w1;
 	bary[1] = w2;
 	bary[2] = (Real)1 - w1 - w2;
 
 	if (bary[2] < 0)
 	{
 		// this gives not an exact orthogonal point to the edge!!
 		const Real w12 = w1 + w2;
 		bary[0] -= w2 / (w12)*(w12 - 1);
 		bary[1] -= w1 / (w12)*(w12 - 1);
 		bary[2] = 0;
 	}
 
 	inter = p2 + bary[0] * x13 + bary[1] * x23;
 
 	return true;
 }