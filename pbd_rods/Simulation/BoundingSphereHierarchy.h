#ifndef __BOUNDINGSPHEREHIERARCHY_H__
#define __BOUNDINGSPHEREHIERARCHY_H__

#include "Common/Common.h"
#include "BoundingSphere.h"
#include "kdTree.h"

namespace PBD
{
	class PointCloudBSH : public KDTree<BoundingSphere>
	{

	public:

		using super = KDTree<BoundingSphere>;

		PointCloudBSH();

		void init(const Vector3r *vertices, const unsigned int numVertices);
		Vector3r const& entity_position(unsigned int i) const final;
		void compute_hull(unsigned int b, unsigned int n, BoundingSphere& hull)
			const final;
		void compute_hull_approx(unsigned int b, unsigned int n, BoundingSphere& hull)
			const final;

	private:
		const Vector3r *m_vertices;
		unsigned int m_numVertices;
	};


	class TetMeshBSH : public KDTree<BoundingSphere>
	{

	public:

		using super = KDTree<BoundingSphere>;

		TetMeshBSH();

		void init(const Vector3r *vertices, const unsigned int numVertices, const unsigned int *indices, const unsigned int numTets, const Real tolerance);
		Vector3r const& entity_position(unsigned int i) const final;
		void compute_hull(unsigned int b, unsigned int n, BoundingSphere& hull)
			const final;
		void compute_hull_approx(unsigned int b, unsigned int n, BoundingSphere& hull)
			const final;
		void updateVertices(const Vector3r* vertices);

	private:
		const Vector3r *m_vertices;
		unsigned int m_numVertices;
		const unsigned int *m_indices;
		unsigned int m_numTets;
		Real m_tolerance;
		std::vector<Vector3r> m_com;
	};

	class BVHTest
	{
	public:
		using TraversalCallback = std::function <void(unsigned int node_index1, unsigned int node_index2)>;

		static void traverse(PointCloudBSH const& b1, TetMeshBSH const& b2, TraversalCallback func);
		static void traverse(PointCloudBSH const& b1, const unsigned int node_index1, TetMeshBSH const& b2, const unsigned int node_index2, TraversalCallback func);
	};
}

#endif
