
#include "BoundingSphereHierarchy.h"

#include <iostream>
#include <unordered_set>
#include <set>

using namespace Eigen;
using pool_set = std::set<unsigned int>;
using namespace PBD;


PointCloudBSH::PointCloudBSH()
    : super(0, 10)
{
}

Vector3r const& PointCloudBSH::entity_position(unsigned int i) const
{
    return m_vertices[i];
}

void PointCloudBSH::compute_hull(unsigned int b, unsigned int n, BoundingSphere& hull) const
{
	auto vertices_subset = std::vector<Vector3r>(n);
	for (unsigned int i=b; i < b+n; ++i)
		vertices_subset[i-b] = m_vertices[m_lst[i]];

	const BoundingSphere s(vertices_subset);

	hull.x() = s.x();
	hull.r() = s.r();
}

void PointCloudBSH::compute_hull_approx(unsigned int b, unsigned int n, BoundingSphere& hull) const
{
	// compute center
	Vector3r x;
	x.setZero();
	for (unsigned int i = b; i < b+n; i++)
		x += m_vertices[m_lst[i]];
	x /= (Real)n;

	Real radius2 = 0.0;
	for (unsigned int i = b; i < b+n; i++)
	{
		radius2 = std::max(radius2, (x - m_vertices[m_lst[i]]).squaredNorm());
	}

	hull.x() = x;
	hull.r() = sqrt(radius2);
}

void PointCloudBSH::init(const Vector3r *vertices, const unsigned int numVertices)
{
	m_lst.resize(numVertices);
	m_vertices = vertices;
	m_numVertices = numVertices;
}

//////////////////////////////////////////////////////////////////////////


TetMeshBSH::TetMeshBSH()
	: super(0)
{
}

Vector3r const& TetMeshBSH::entity_position(unsigned int i) const
{
	return m_com[i];
}

void TetMeshBSH::compute_hull(unsigned int b, unsigned int n, BoundingSphere& hull) const
{
	compute_hull_approx(b, n, hull);
}

void TetMeshBSH::compute_hull_approx(unsigned int b, unsigned int n, BoundingSphere& hull) const
{
	// compute center
	Vector3r x;
	x.setZero();
	for (unsigned int i = b; i < b + n; i++)
	{
		const unsigned int tet = m_lst[i];	
		x += m_vertices[m_indices[4 * tet]];
		x += m_vertices[m_indices[4 * tet + 1]];
		x += m_vertices[m_indices[4 * tet + 2]];
		x += m_vertices[m_indices[4 * tet + 3]];
	}
	x /= ((Real)4.0* (Real)n);

	Real radius2 = 0.0;
	for (unsigned int i = b; i < b + n; i++)
	{
		const unsigned int tet = m_lst[i];
		radius2 = std::max(radius2, (x - m_vertices[m_indices[4 * tet]]).squaredNorm());
		radius2 = std::max(radius2, (x - m_vertices[m_indices[4 * tet+1]]).squaredNorm());
		radius2 = std::max(radius2, (x - m_vertices[m_indices[4 * tet+2]]).squaredNorm());
		radius2 = std::max(radius2, (x - m_vertices[m_indices[4 * tet+3]]).squaredNorm());
	}

	hull.x() = x;
	hull.r() = sqrt(radius2) + m_tolerance;
}

void TetMeshBSH::init(const Vector3r *vertices, const unsigned int numVertices, const unsigned int *indices, const unsigned int numTets, const Real tolerance)
{
	m_lst.resize(numTets);
	m_vertices = vertices;
	m_numVertices = numVertices;
	m_indices = indices;
	m_numTets = numTets;
	m_tolerance = tolerance;
	m_com.resize(numTets);
	for (unsigned int i = 0; i < numTets; i++)
	{
		m_com[i] = 0.25 * (m_vertices[m_indices[4*i]] + m_vertices[m_indices[4*i+1]] + m_vertices[m_indices[4*i+2]] + m_vertices[m_indices[4*i+3]]);
	}
}

void TetMeshBSH::updateVertices(const Vector3r* vertices)
{
	m_vertices = vertices;
}


void BVHTest::traverse(PointCloudBSH const& b1, TetMeshBSH const& b2, TraversalCallback func)
{
	traverse(b1, 0, b2, 0, func);
}

void BVHTest::traverse(PointCloudBSH const& b1, const unsigned int node_index1, TetMeshBSH const& b2, const unsigned int node_index2, TraversalCallback func)
{
	const BoundingSphere &bs1 = b1.hull(node_index1);
	const BoundingSphere &bs2 = b2.hull(node_index2);
	if (!bs1.overlaps(bs2))
		return;

	auto const& node1 = b1.node(node_index1);
	auto const& node2 = b2.node(node_index2);
	if (node1.is_leaf() && node2.is_leaf())
	{
		func(node_index1, node_index2);
		return;
	}

	if (bs1.r() < bs2.r())
	{
		if (!node1.is_leaf())
		{
			traverse(b1, node1.children[0], b2, node_index2, func);
			traverse(b1, node1.children[1], b2, node_index2, func);
		}
		else
		{
			traverse(b1, node_index1, b2, node2.children[0], func);
			traverse(b1, node_index1, b2, node2.children[1], func);
		}
	}
	else
	{
		if (!node2.is_leaf())
		{
			traverse(b1, node_index1, b2, node2.children[0], func);
			traverse(b1, node_index1, b2, node2.children[1], func);
		}
		else
		{
			traverse(b1, node1.children[0], b2, node_index2, func);
			traverse(b1, node1.children[1], b2, node_index2, func);
		}
	}

}