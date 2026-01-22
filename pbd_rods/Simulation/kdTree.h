#ifndef __KDTREE_H__
#define __KDTREE_H__

#include <vector>
#include <functional>
#include <algorithm>
#include <numeric>
#include <queue>
#include <iostream>

#include "Common/Common.h"

#include <array>
#include <list>

namespace PBD
{

	template <typename HullType>
	class KDTree
	{
	public:

		using TraversalPredicate = std::function<bool(unsigned int node_index, unsigned int depth)>;
		using TraversalCallback = std::function <void(unsigned int node_index, unsigned int depth)>;
		using TraversalPriorityLess = std::function<bool(std::array<int, 2> const& nodes)>;

		struct Node
		{
			Node(unsigned int b_, unsigned int n_)
				: children({ { -1, -1 } })
				, begin(b_), n(n_) {}

			Node() = default;

			bool is_leaf() const { return children[0] < 0 && children[1] < 0; }

			// Index of child nodes in nodes array.
			// -1 if child does not exist.
			std::array<int, 2> children;

			// Index according entries in entity list.
			unsigned int begin;

			// Number of owned entries.
			unsigned int n;
		};

		struct QueueItem { unsigned int n, d; };
		using TraversalQueue = std::queue<QueueItem>;

		KDTree(std::size_t n, unsigned int maxPrimitivesPerLeaf = 1)
			: m_lst(n), m_maxPrimitivesPerLeaf(maxPrimitivesPerLeaf) {}

		virtual ~KDTree() {}

		Node const& node(unsigned int i) const { return m_nodes[i]; }
		HullType const& hull(unsigned int i) const { return m_hulls[i]; }
		unsigned int entity(unsigned int i) const { return m_lst[i]; }

		void construct();
		void traverse_depth_first(TraversalPredicate pred, TraversalCallback cb,
			TraversalPriorityLess const& pless = nullptr) const;
		void traverse_breadth_first(TraversalPredicate const& pred, TraversalCallback const& cb, unsigned int start_node = 0, TraversalPriorityLess const& pless = nullptr, TraversalQueue& pending = TraversalQueue()) const;
		void traverse_breadth_first_parallel(TraversalPredicate pred, TraversalCallback cb) const;
		void update();

	protected:

		void construct(unsigned int node, AlignedBox3r const& box,
			unsigned int b, unsigned int n);
		void traverse_depth_first(unsigned int node, unsigned int depth,
			TraversalPredicate pred, TraversalCallback cb, TraversalPriorityLess const& pless) const;
		void traverse_breadth_first(TraversalQueue& pending,
			TraversalPredicate const& pred, TraversalCallback const& cb, TraversalPriorityLess const& pless = nullptr) const;

		unsigned int add_node(unsigned int b, unsigned int n);

		virtual Vector3r const& entity_position(unsigned int i) const = 0;
		virtual void compute_hull(unsigned int b, unsigned int n, HullType& hull) const = 0;
		virtual void compute_hull_approx(unsigned int b, unsigned int n, HullType& hull) const
		{
			compute_hull(b, n, hull);
		}

	protected:

		std::vector<unsigned int> m_lst;

		std::vector<Node> m_nodes;
		std::vector<HullType> m_hulls;
		unsigned int m_maxPrimitivesPerLeaf;
	};

#include "kdTree.inl"
}

#endif
