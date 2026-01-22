#ifndef __RIGIDBODYGEOMETRY_H__
#define __RIGIDBODYGEOMETRY_H__

#include "Common/Common.h"
#include "Utils/IndexedFaceMesh.h"
#include "Simulation/ParticleData.h"
#include <vector>

namespace PBD 
{	
	class RigidBodyGeometry
	{
		public:
			RigidBodyGeometry();
			virtual ~RigidBodyGeometry();

			typedef Utilities::IndexedFaceMesh Mesh;

		protected:
			Mesh m_mesh;
			VertexData m_vertexData_local;
			VertexData m_vertexData;

		public:
			Mesh &getMesh();
			VertexData &getVertexData();
			const VertexData &getVertexData() const;
			VertexData &getVertexDataLocal();
			const VertexData &getVertexDataLocal() const;

			void initMesh(const unsigned int nVertices, const unsigned int nFaces, const Vector3r *vertices, const unsigned int* indices, const Mesh::UVIndices& uvIndices, const Mesh::UVs& uvs, const Vector3r &scale = Vector3r(1.0, 1.0, 1.0), const bool flatShading = false);
			void updateMeshTransformation(const Vector3r &x, const Matrix3r &R);
			void updateMeshNormals(const VertexData &vd);
			
	};
}

#endif