#include "RigidBodyGeometry.h"

using namespace PBD;

RigidBodyGeometry::RigidBodyGeometry() :
	m_mesh()
{	
}

RigidBodyGeometry::~RigidBodyGeometry(void)
{
	m_mesh.release();
}

RigidBodyGeometry::Mesh &RigidBodyGeometry::getMesh()
{
	return m_mesh;
}

void RigidBodyGeometry::initMesh(const unsigned int nVertices, const unsigned int nFaces, const Vector3r *vertices, const unsigned int* indices, const Mesh::UVIndices& uvIndices, const Mesh::UVs& uvs, const Vector3r &scale, const bool flatShading)
{
	m_mesh.release();
	m_mesh.initMesh(nVertices, nFaces * 2, nFaces);
	m_vertexData_local.resize(nVertices);
	m_vertexData.resize(nVertices);
	m_mesh.setFlatShading(flatShading);
	for (unsigned int i = 0; i < nVertices; i++)
	{
		m_vertexData_local.getPosition(i) = vertices[i].cwiseProduct(scale);
		m_vertexData.getPosition(i) = m_vertexData_local.getPosition(i);
	}

	for (unsigned int i = 0; i < nFaces; i++)
	{
		m_mesh.addFace(&indices[3 * i]);
	}
	m_mesh.copyUVs(uvIndices, uvs);
	m_mesh.buildNeighbors();
	updateMeshNormals(m_vertexData);
}

void RigidBodyGeometry::updateMeshNormals(const VertexData &vd)
{
	m_mesh.updateNormals(vd, 0);
	m_mesh.updateVertexNormals(vd);
}

void RigidBodyGeometry::updateMeshTransformation(const Vector3r &x, const Matrix3r &R)
{
	for (unsigned int i = 0; i < m_vertexData_local.size(); i++)
	{
		m_vertexData.getPosition(i) = R * m_vertexData_local.getPosition(i) + x;
	}
	updateMeshNormals(m_vertexData);
}

VertexData & RigidBodyGeometry::getVertexData()
{
	return m_vertexData;
}

const VertexData & RigidBodyGeometry::getVertexData() const
{
	return m_vertexData;
}

VertexData & RigidBodyGeometry::getVertexDataLocal()
{
	return m_vertexData_local;
}

const VertexData & RigidBodyGeometry::getVertexDataLocal() const
{
	return m_vertexData_local;
}
