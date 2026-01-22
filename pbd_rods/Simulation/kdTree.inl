
#include "BoundingSphere.h"
#include <cstring>
#include "omp.h"

template<typename HullType> void
KDTree<HullType>::construct()
{
	m_nodes.clear();
	m_hulls.clear();
	if (m_lst.empty()) return;

	std::iota(m_lst.begin(), m_lst.end(), 0);

	// Determine bounding box of considered domain.
	auto box = AlignedBox3r{};
	for (auto i = 0u; i < m_lst.size(); ++i)
		box.extend(entity_position(i));

	auto ni = add_node(0, static_cast<unsigned int>(m_lst.size()));
	construct(ni, box, 0, static_cast<unsigned int>(m_lst.size()));
}

template<typename HullType> void
KDTree<HullType>::construct(unsigned int node, AlignedBox3r const& box, unsigned int b,
	unsigned int n)
{
	// If only one element is left end recursion.
	//if (n == 1) return;
	if (n <= m_maxPrimitivesPerLeaf) return;

	// Determine longest side of bounding box.
	auto max_dir = 0;
	auto d = box.diagonal().eval();
	if (d(1) >= d(0) && d(1) >= d(2))
		max_dir = 1;
	else if (d(2) >= d(0) && d(2) >= d(1))
		max_dir = 2;

#ifdef _DEBUG
	for (auto i = 0u; i < n; ++i)
	{
		if (!box.contains(entity_position(m_lst[b + i])))
			std::cerr << "ERROR: Bounding box wrong!" << std::endl;
	}
#endif

	// Sort range according to center of the longest side.
	std::sort(m_lst.begin() + b, m_lst.begin() + b + n, 
		[&](unsigned int a, unsigned int b)
		{
			return entity_position(a)(max_dir) < entity_position(b)(max_dir);
		}
	);

	auto hal = n / 2;
	auto n0 = add_node(b      , hal    );
	auto n1 = add_node(b + hal, n - hal);
	m_nodes[node].children[0] = n0;
	m_nodes[node].children[1] = n1;

	auto c = static_cast<Real>(0.5) * (
		entity_position(m_lst[b + hal -1])(max_dir) + 
		entity_position(m_lst[b + hal   ])(max_dir));
	auto l_box = box; l_box.max()(max_dir) = c;
	auto r_box = box; r_box.min()(max_dir) = c;

	construct(m_nodes[node].children[0], l_box, b, hal);
	construct(m_nodes[node].children[1], r_box, b + hal, n - hal);
}

template<typename HullType> void
KDTree<HullType>::traverse_depth_first(TraversalPredicate pred, TraversalCallback cb,
	TraversalPriorityLess const& pless) const
{
	if (m_nodes.empty())
		return;

	if (pred(0, 0)) 
		traverse_depth_first(0, 0, pred, cb, pless);
}

template<typename HullType> void
KDTree<HullType>::traverse_depth_first(unsigned int node_index, 
	unsigned int depth, TraversalPredicate pred, TraversalCallback cb,
	TraversalPriorityLess const& pless) const
{
	Node const& node = m_nodes[node_index];

	cb(node_index, depth);
	auto is_pred = pred(node_index, depth);
	if (!node.is_leaf() && is_pred)
	{
		if (pless && !pless(node.children))
		{
			traverse_depth_first(m_nodes[node_index].children[1], depth + 1, pred, cb, pless);
			traverse_depth_first(m_nodes[node_index].children[0], depth + 1, pred, cb, pless);
		}
		else
		{
			traverse_depth_first(m_nodes[node_index].children[0], depth + 1, pred, cb, pless);
			traverse_depth_first(m_nodes[node_index].children[1], depth + 1, pred, cb, pless);
		}
	}
}


template <typename HullType> void
KDTree<HullType>::traverse_breadth_first(TraversalPredicate const& pred, 
	TraversalCallback const& cb, unsigned int start_node, TraversalPriorityLess const& pless,
	TraversalQueue& pending) const
{
	cb(start_node, 0);
	if (pred(start_node, 0)) pending.push({ start_node, 0 });
	traverse_breadth_first(pending, pred, cb, pless);
}


template <typename HullType> void
KDTree<HullType>::traverse_breadth_first_parallel(TraversalPredicate pred,
	TraversalCallback cb) const
{
	auto start_nodes = std::vector<QueueItem>{};
#ifdef _DEBUG
	const unsigned int maxThreads = 1;
#else
	const unsigned int maxThreads = omp_get_max_threads();
#endif

	// compute ceiling of Log2
	// assuming double and long long have the same size.
	double d = maxThreads - 1;
	long long ll;  memcpy( &ll, &d, sizeof(d));
	const unsigned int targetDepth = (ll >> 52) - 1022ll;
	
	traverse_breadth_first(
		[&start_nodes, &maxThreads, &targetDepth](unsigned int node_index, unsigned int depth)
		{			
			return (depth < targetDepth) && (start_nodes.size() < maxThreads);
		},
		[&](unsigned int node_index, unsigned int depth)
		{			
			if ((depth == targetDepth) || (node(node_index).is_leaf()))
				start_nodes.push_back({ node_index, depth });
		}
		);

	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static) 
		for (int i = 0; i < start_nodes.size(); i++)
		{
			QueueItem const& qi = start_nodes[i];
			cb(qi.n, qi.d);
			traverse_depth_first(qi.n, qi.d, pred, cb, nullptr);
		}
	}
}

template <typename HullType> unsigned int
KDTree<HullType>::add_node(unsigned int b, unsigned int n)
{
	HullType hull;
	compute_hull(b, n, hull);
	m_hulls.push_back(hull);
	m_nodes.push_back({ b, n });
	return static_cast<unsigned int>(m_nodes.size() - 1);
}



template <typename HullType>
void 
KDTree<HullType>::traverse_breadth_first(TraversalQueue& pending, 
	TraversalPredicate const& pred, TraversalCallback const& cb, TraversalPriorityLess const& pless) const
{
	while (!pending.empty())
	{
		auto n = pending.front().n;
		auto d = pending.front().d;
		auto const& node = m_nodes[n];
		pending.pop();

		cb(n, d);
		auto is_pred = pred(n, d);
		if (!node.is_leaf() && is_pred)
		{
			if (pless && !pless(node.children))
			{
				pending.push({ static_cast<unsigned int>(node.children[1]), d + 1 });
				pending.push({ static_cast<unsigned int>(node.children[0]), d + 1 });
			}
			else
			{
				pending.push({ static_cast<unsigned int>(node.children[0]), d + 1 });
				pending.push({ static_cast<unsigned int>(node.children[1]), d + 1 });
			}
		}
	}
}

template <typename HullType> void
KDTree<HullType>::update()
{
	traverse_depth_first(
		[&](unsigned int, unsigned int) { return true; },
		[&](unsigned int node_index, unsigned int)
	{
		auto const& nd = node(node_index);
		compute_hull_approx(nd.begin, nd.n, m_hulls[node_index]);
	}
	);
}
