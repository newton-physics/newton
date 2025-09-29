# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warp as wp

from newton._src.solvers.style3d.collision.bvh.kernels import (
    aabb_vs_aabb_kernel,
    aabb_vs_line_kernel,
    compute_edge_aabbs_kernel,
    compute_tri_aabbs_kernel,
    edge_vs_edge_kernel,
    triangle_vs_point_kernel,
)

########################################################################################################################
######################################################    Bvh    #######################################################
########################################################################################################################


class BvhAabb:
    """A wrapper class for Warp's BVH (Bounding Volume Hierarchy) structure.

    This class manages a BVH for efficient spatial queries such as AABB vs AABB
    or AABB vs line segment intersections. It provides methods for building,
    rebuilding, and refitting the hierarchy, as well as performing queries.

    Query results are stored in a 2d-array for efficient read/write operations, where each
    column corresponds to a query thread and each row represents a result slot (see below):

    ---------------------------------------------------------------------------------
    |         |  thread_0 |  thread_1 |  thread_2 |  thread_3 |   ...   |  thread_m |
    ---------------------------------------------------------------------------------
    | slot_0 |     2     |     0     |     1     |   n - 1   |    ...   |     4     |
    ---------------------------------------------------------------------------------
    | slot_1 |    522    |     -     |    333    |    10     |    ...   |     0     |
    ---------------------------------------------------------------------------------
    | slot_2 |   1000    |     -     |     -     |    13     |    ...   |     1     |
    ---------------------------------------------------------------------------------
    |   ...  |     -     |     -     |     -     |    ...    |    ...   |    ...    |
    ---------------------------------------------------------------------------------
    | slot_n |     -     |     -     |     -     |    555    |    ...   |     -     |
    ---------------------------------------------------------------------------------

    Notes:
        - Row 0 stores the count of valid indices for each thread.
        - Columns should be at least the number of query objects, ideally aligned to 32 for performance.
        - Rows equal the maximum query count plus 1 (for the count row).
        - Use the following pattern to iterate over results:

    .. code-block:: python

        for i in range(query_results[0, tid])
            idx = query_results[i + 1, tid]
            ...
    """

    def __init__(self, num_leaves: int, device: wp.context.Device):
        self.bvh = None
        self.device = device
        self.lower_bounds = wp.zeros(num_leaves, dtype=wp.vec3, device=self.device)
        self.upper_bounds = wp.zeros(num_leaves, dtype=wp.vec3, device=self.device)

    def is_built(self) -> bool:
        """Returns True if the BVH has been built, otherwise False."""
        return self.bvh is not None

    def build(self):
        """Builds the BVH from the current lower and upper bounds.

        This method allocates and constructs the BVH hierarchy from scratch
        based on the provided leaf bounds. It must be called at least once
        before using the BVH for queries.

        Use this when:
            - Initializing the BVH for the first time.
            - The number of leaves has changed.
            - The hierarchy structure must be recomputed.

        Warning:
            This function **must not** be called inside a `wp.ScopedCapture()` context,
            since it performs out-of-place allocations.
        """
        self.bvh = wp.Bvh(self.lower_bounds, self.upper_bounds)

    def rebuild(self):
        """Rebuilds the BVH using the current lower and upper bounds.

        Unlike `build()`, this does not reallocate or create a new BVH, but
        instead updates the hierarchy using the existing BVH object. This is
        more efficient than a full `build()`, but still recomputes the tree
        topology.

        Use this when:
            - Leaf bounds have changed significantly.
            - The BVH structure needs to be updated without a full rebuild.

        Notes:
            - Can be safely called inside a `wp.ScopedCapture()` context,
              since all operations are performed in-place.
            - Raises:
                RuntimeError: If the BVH has not been built yet.
        """
        if self.bvh is None:
            raise RuntimeError("BVH hasn't been built yet!")
        else:
            self.bvh.rebuild()

    def refit(self):
        """Refits the existing BVH to updated leaf bounds.

        This is the most efficient update operation. It preserves the existing
        BVH structure and only updates bounding volumes to fit the new leaf
        positions.

        Use this when:
            - The number of leaves is unchanged.
            - Only the positions of primitives have moved (e.g., rigid or deforming objects).
            - You want the cheapest possible update.

        Raises:
            RuntimeError: If the BVH has not been built yet.
        """
        if self.bvh is None:
            raise RuntimeError("BVH hasn't been built yet!")
        else:
            self.bvh.refit()

    def aabb_vs_aabb(
        self,
        lower_bounds: wp.array(dtype=wp.vec3),
        upper_bounds: wp.array(dtype=wp.vec3),
        query_results: wp.array(dtype=int, ndim=2),
        query_radius: float = 0.0,
        ignore_self_hits: bool = False,
    ):
        """Queries the BVH for overlapping AABBs.

        For each query AABB defined by `lower_bounds[i]` and `upper_bounds[i]`, this method finds
        all leaf nodes in the BVH that overlap with the query box (optionally expanded by `query_radius`).

        Results are written to `query_results` in a row-major layout:
            - Each column corresponds to a query thread.
            - Row 0 stores the number of hits for that thread.
            - Rows 1..N store the indices of the intersecting leaf nodes.

        Args:
            lower_bounds (wp.array): Array of lower corners of query AABBs.
            upper_bounds (wp.array): Array of upper corners of query AABBs.
            query_results (wp.array): 2D integer array for storing results [max_results + 1, num_queries].
            query_radius (float, optional): Additional padding radius to apply to each query AABB.
            ignore_self_hits (bool, optional): If True, suppresses self-intersections (e.g., for symmetric queries).

        Note:
            - query_results.shape[1] must be ≥ number of aabbs (i.e., lower_bounds.shape[0]).
            - query_results.shape[0] must be ≥ max_results + 1 (for count row).
        """
        # ================================    Runtime checks    ================================
        assert query_results.ndim == 2, "query_results must be a 2D array."
        assert query_results.shape[0] >= 1, f"query_results must have at least 1 rows, got {query_results.shape[0]}."
        assert lower_bounds.shape == upper_bounds.shape, "lower_bounds and upper_bounds must have the same shape."
        assert self.bvh is not None, "BVH has not been built. Call build() or refit() first."
        # ================================    Runtime checks    ================================
        wp.launch(
            aabb_vs_aabb_kernel,
            dim=lower_bounds.shape[0],
            inputs=[self.bvh.id, query_results.shape[0], query_radius, ignore_self_hits, lower_bounds, upper_bounds],
            outputs=[query_results],
            device=self.device,
            block_dim=64,
        )

    def aabb_vs_line(
        self,
        vertices: wp.array(dtype=wp.vec3),
        edge_indices: wp.array(dtype=int, ndim=2),
        query_results: wp.array(dtype=int, ndim=2),
        ignore_self_hits: bool = False,
    ):
        """Queries the BVH for intersections between line segments and AABBs.

        This function casts each line segment defined by `vertices[edge_indices[i, 2]]` to
        `vertices[edge_indices[i, 3]]` against the BVH. For each segment, it collects all AABBs
        (from the BVH leaves) that intersect with the segment.

        Results are written to `query_results` in a row-major layout:
            - Each column corresponds to a query thread.
            - Row 0 stores the number of hits for that thread.
            - Rows 1..N store the indices of the intersecting leaf nodes.

        Args:
            vertices (wp.array): Array of 3D points representing geometry vertices.
            edge_indices (wp.array): (N, 2) array of vertex indices forming line segments.
            query_results (wp.array): 2D int array of shape (max_results + 1, num_segments) for output.
            ignore_self_hits (bool): Whether to ignore self-intersections (e.g., for symmetric geometry).

        Note:
            - query_results.shape[1] must be ≥ number of segments (i.e., edge_indices.shape[0]).
            - query_results.shape[0] must be ≥ max_results + 1 (for count row).
        """
        # ================================    Runtime checks    ================================
        assert query_results.ndim == 2, "query_results must be a 2D array."
        assert query_results.shape[0] >= 1, f"query_results must have at least 1 rows, got {query_results.shape[0]}."
        assert self.bvh is not None, "BVH has not been built. Call build() or refit() first."
        # ================================    Runtime checks    ================================
        wp.launch(
            aabb_vs_line_kernel,
            dim=edge_indices.shape[0],
            inputs=[
                self.bvh.id,
                query_results.shape[0],
                ignore_self_hits,
                vertices,
                edge_indices,
                self.lower_bounds,
                self.upper_bounds,
            ],
            outputs=[query_results],
            device=self.device,
            block_dim=64,
        )


########################################################################################################################
####################################################    Edge Bvh    ####################################################
########################################################################################################################


class BvhEdge(BvhAabb):
    """BVH structure specialized for edge primitives (line segments).

    This class extends :class:`BvhAabb` with functionality to compute AABBs
    from edges (pairs of vertices) and to perform edge-specific queries.
    It supports building, refitting, and querying against edge-based BVHs.
    """

    def __init__(self, edge_count: int, device: wp.context.Device):
        super().__init__(edge_count, device)

    def update_aabbs(self, pos: wp.array(dtype=wp.vec3), edge_indices: wp.array(dtype=int, ndim=2), enlarge: float):
        """Computes AABBs for all edges based on current vertex positions and edge indices.

        Args:
            pos (wp.array): Vertex position array (wp.vec3).
            edge_indices (wp.array): Integer array of shape (M, 4). Columns 2 and 3
                of each row contain indices into `pos` defining an edge.
            enlarge (float): Optional margin to expand each bounding box
                (useful for padding or motion blur).
        """
        # ================================    Runtime checks    ================================
        assert edge_indices.shape[1] == 4, f"edge_indices must be of shape (M, 4), got {edge_indices.shape}"
        assert edge_indices.shape[0] == self.lower_bounds.shape[0], "Mismatch between edge count and BVH leaf count."
        # ================================    Runtime checks    ================================
        wp.launch(
            compute_edge_aabbs_kernel,
            dim=self.lower_bounds.shape[0],
            inputs=[enlarge, pos, edge_indices],
            outputs=[self.lower_bounds, self.upper_bounds],
            device=self.device,
        )

    def build(self, pos: wp.array(dtype=wp.vec3), edge_indices: wp.array(dtype=int, ndim=2), enlarge: float = 0.0):
        """Builds the edge BVH from scratch using the given vertex positions and edge indices.

        This computes the AABBs for all edges and then constructs a new BVH hierarchy.

        Args:
            pos (wp.array): Vertex positions (wp.vec3).
            edge_indices (wp.array): Integer array of shape (M, 4). Columns 2 and 3
                of each row contain the vertex indices defining an edge.
            enlarge (float): Optional padding value to expand each edge's bounding box (default 0.0).

        Warning:
            This function **must not** be called inside a `wp.ScopedCapture()` context,
            since it triggers allocations and memory movement.
        """
        self.update_aabbs(pos, edge_indices, enlarge)
        super().build()

    def rebuild(self, pos: wp.array(dtype=wp.vec3), edge_indices: wp.array(dtype=int, ndim=2), enlarge: float = 0.0):
        """Rebuilds the edge BVH using the current vertex positions and edge indices.

        This recomputes the edge AABBs and reconstructs the BVH hierarchy
        from scratch (i.e., equivalent to `build()` but reuses the existing object).

        Args:
            pos (wp.array): Updated vertex positions (wp.vec3).
            edge_indices (wp.array): Integer array of shape (M, 4). Columns 2 and 3
                of each row contain the vertex indices defining an edge.
            enlarge (float): Optional padding value to expand each edge's bounding box (default 0.0).

        Notes:
            - Unlike :func:`refit`, this recomputes the BVH topology, not just the bounds.
            - May be significantly more expensive than `refit()` but more robust
              when edge connectivity has changed or large movements occurred.
        """
        self.update_aabbs(pos, edge_indices, enlarge)
        super().rebuild()

    def refit(self, pos: wp.array(dtype=wp.vec3), edge_indices: wp.array(dtype=int, ndim=2), enlarge: float = 0.0):
        """Refits the edge BVH after vertex positions have changed, without rebuilding the hierarchy.

        This updates the leaf AABBs for all edges and adjusts the internal BVH bounds,
        while preserving the existing hierarchy structure.

        Args:
            pos (wp.array): Updated vertex positions (wp.vec3).
            edge_indices (wp.array): Integer array of shape (M, 4). Columns 2 and 3
                of each row contain the vertex indices defining an edge.
            enlarge (float): Optional padding value to expand each edge's bounding box (default 0.0).

        Use this for dynamic geometry where connectivity stays the same but positions change.
        """
        self.update_aabbs(pos, edge_indices, enlarge)
        super().refit()

    def edge_vs_edge(
        self,
        test_pos: wp.array(dtype=wp.vec3),
        test_edge_indices: wp.array(dtype=int, ndim=2),
        edge_pos: wp.array(dtype=wp.vec3),
        edge_indices: wp.array(dtype=int, ndim=2),
        query_results: wp.array(dtype=int, ndim=2),
        ignore_self_hits: bool,
        max_dist: float,
        query_radius: float = 0.0,
    ):
        """Queries the BVH to find edges that are within a maximum distance from a set of points.

        For each input edge (defined by `test_pos` and `test_edge_indices`), this function identifies edge
        (defined by `edge_indices` and `edge_pos`) that fall within `max_dist` of the edge. It supports
        optional self-hit suppression and radius-based padding for edge bounds.

        Results are written to `query_results` in a row-major layout:
            - Each column corresponds to a query thread.
            - Row 0 stores the number of hits for that thread.
            - Rows 1..N store the indices of the intersecting leaf nodes.

        Args:
            test_pos (wp.array): Query edge vertex positions (wp.vec3).
            test_edge_indices (wp.array): Query edge indices (M x 4 int array).
            edge_pos (wp.array): Edge vertex positions (same as used when building BVH).
            edge_indices (wp.array): Edge indices (M x 4 int array).
            query_results (wp.array): 2D int array to store the result layout (max_results + 1, P).
            ignore_self_hits (bool): If True, skips hits between a point and its associated triangle (e.g. for self-collision).
            max_dist (float): Maximum allowed distance between point and triangle for a match to be considered.
            query_radius (float): Optional padding to enlarge triangle AABBs during the query (default: 0.0).
        """
        wp.launch(
            edge_vs_edge_kernel,
            dim=test_edge_indices.shape[0],
            inputs=[
                self.bvh.id,
                query_results.shape[0],
                query_radius,
                max_dist,
                ignore_self_hits,
                test_pos,
                test_edge_indices,
                edge_pos,
                edge_indices,
            ],
            outputs=[query_results],
            device=self.device,
            block_dim=64,
        )


########################################################################################################################
##################################################    Triangle Bvh    ##################################################
########################################################################################################################


class BvhTri(BvhAabb):
    """BVH structure specialized for triangular face primitives.

    This class extends :class:`BvhAabb` with functionality to compute AABBs
    from triangle primitives and perform triangle-specific queries.
    It supports building, rebuilding, refitting, and spatial queries
    involving triangles.
    """

    def __init__(self, tri_count: int, device: wp.context.Device):
        super().__init__(tri_count, device)

    def update_aabbs(self, pos: wp.array(dtype=wp.vec3), tri_indices: wp.array(dtype=int, ndim=2), enlarge: float):
        """Computes AABBs for all triangles based on current vertex positions and indices.

        Args:
            pos (wp.array): Vertex position array (wp.vec3).
            tri_indices (wp.array): Integer array of shape (M, 3),
                where each row contains vertex indices defining a triangle.
            enlarge (float): Optional margin to expand each bounding box
                (useful for padding or motion blur).
        """
        # ================================    Runtime checks    ================================
        assert tri_indices.shape[1] == 3, f"tri_indices must be of shape (M, 3), got {tri_indices.shape}"
        assert tri_indices.shape[0] == self.lower_bounds.shape[0], "Mismatch between triangle count and BVH leaf count."
        # ================================    Runtime checks    ================================
        wp.launch(
            compute_tri_aabbs_kernel,
            dim=self.lower_bounds.shape[0],
            inputs=[enlarge, pos, tri_indices],
            outputs=[self.lower_bounds, self.upper_bounds],
            device=self.device,
        )

    def build(self, pos: wp.array(dtype=wp.vec3), tri_indices: wp.array(dtype=int, ndim=2), enlarge: float = 0.0):
        """Builds the triangle BVH from scratch.

        This computes AABBs for all triangles and constructs a new BVH hierarchy.

        Args:
            pos (wp.array): Vertex positions (wp.vec3).
            tri_indices (wp.array): Integer array of shape (M, 3),
                where each row defines a triangle.
            enlarge (float): Optional padding value to expand each triangle's bounding box (default 0.0).

        Warning:
            This function **must not** be called inside a `wp.ScopedCapture()` context,
            since it triggers allocations and memory movement.
        """
        self.update_aabbs(pos, tri_indices, enlarge)
        super().build()

    def rebuild(self, pos: wp.array(dtype=wp.vec3), tri_indices: wp.array(dtype=int, ndim=2), enlarge: float = 0.0):
        """Rebuilds the triangle BVH using the current vertex positions and indices.

        This recomputes the triangle AABBs and rebuilds the BVH hierarchy
        in place (more expensive than `refit`, but more robust when triangles
        move significantly or topology has changed).

        Args:
            pos (wp.array): Updated vertex positions (wp.vec3).
            tri_indices (wp.array): Integer array of shape (M, 3),
                where each row defines a triangle.
            enlarge (float): Optional padding value to expand each triangle's bounding box (default 0.0).

        Notes:
            - Unlike :func:`refit`, this recomputes the BVH topology,
              not just the bounding volumes.
            - More efficient than a full :func:`build()` if the BVH already exists.
        """
        self.update_aabbs(pos, tri_indices, enlarge)
        super().rebuild()

    def refit(self, pos: wp.array(dtype=wp.vec3), tri_indices: wp.array(dtype=int, ndim=2), enlarge: float = 0.0):
        """Refits the triangle BVH after vertex positions have changed, without rebuilding the hierarchy.

        This updates AABBs for all triangles and propagates the changes up the hierarchy,
        while preserving the existing BVH structure.

        Args:
            pos (wp.array): Updated vertex positions (wp.vec3).
            tri_indices (wp.array): Integer array of shape (M, 3),
                where each row defines a triangle.
            enlarge (float): Optional bounding box padding for each triangle (default 0.0).

        Use this for dynamic geometry where connectivity stays the same but positions change.
        """
        self.update_aabbs(pos, tri_indices, enlarge)
        super().refit()

    def triangle_vs_point(
        self,
        pos: wp.array(dtype=wp.vec3),
        tri_pos: wp.array(dtype=wp.vec3),
        tri_indices: wp.array(dtype=int, ndim=2),
        query_results: wp.array(dtype=int, ndim=2),
        ignore_self_hits: bool,
        max_dist: float,
        query_radius: float = 0.0,
    ):
        """Queries the BVH to find triangles that are within a maximum distance from a set of points.

        For each input point in `pos`, this function identifies triangles (defined by `tri_indices` and `tri_pos`)
        that fall within `max_dist` of the point. It supports optional self-hit suppression and radius-based
        padding for triangle bounds.

        Results are written to `query_results` in a row-major layout:
            - Each column corresponds to a query thread.
            - Row 0 stores the number of hits for that thread.
            - Rows 1..N store the indices of the intersecting leaf nodes.

        Args:
            pos (wp.array): Query point positions (wp.vec3).
            tri_pos (wp.array): Triangle vertex positions (same as used when building BVH).
            tri_indices (wp.array): Triangle indices (M x 3 int array).
            query_results (wp.array): 2D int array to store the result layout (max_results + 1, P).
            ignore_self_hits (bool): If True, skips hits between a point and its associated triangle (e.g. for self-collision).
            max_dist (float): Maximum allowed distance between point and triangle for a match to be considered.
            query_radius (float): Optional padding to enlarge triangle AABBs during the query (default: 0.0).
        """
        # ================================    Runtime checks    ================================
        assert tri_indices.shape[1] == 3, f"tri_indices must be of shape (M, 3), got {tri_indices.shape}"
        assert tri_indices.shape[0] == self.lower_bounds.shape[0], "Mismatch between triangle count and BVH leaf count."
        # ================================    Runtime checks    ================================
        wp.launch(
            triangle_vs_point_kernel,
            dim=pos.shape[0],
            inputs=[
                self.bvh.id,
                query_results.shape[0],
                query_radius,
                max_dist,
                ignore_self_hits,
                pos,
                tri_pos,
                tri_indices,
            ],
            outputs=[query_results],
            device=self.device,
            block_dim=64,
        )


########################################################################################################################
#####################################################    Tests    ######################################################
########################################################################################################################


if __name__ == "__main__":
    wp.init()

    # unit cube
    verts = [
        wp.vec3(0, 0, 0),
        wp.vec3(1, 0, 0),
        wp.vec3(1, 1, 0),
        wp.vec3(0, 1, 0),
        wp.vec3(0, 0, 1),
        wp.vec3(1, 0, 1),
        wp.vec3(1, 1, 1),
        wp.vec3(0, 1, 1),
    ]

    pos = wp.array(verts, dtype=wp.vec3)
    edge_indices = wp.array([[0, 1, 2, 3]], dtype=int)
    tri_indices = wp.array([[0, 1, 6], [3, 4, 5], [6, 7, 0]], dtype=int)

    tri_bvh = BvhTri(3, wp.context.get_device())
    edge_bvh = BvhEdge(1, wp.context.get_device())

    tri_bvh.build(pos, tri_indices)
    edge_bvh.build(pos, edge_indices)

    tri_bvh.rebuild(pos, tri_indices)
    edge_bvh.rebuild(pos, edge_indices)

    print(f"tri_bvh.lower_bounds[0] = {tri_bvh.lower_bounds.numpy()[0]}")
    print(f"tri_bvh.upper_bounds[0] = {tri_bvh.upper_bounds.numpy()[0]}")
    print(f"edge_bvh.lower_bounds[0] = {edge_bvh.lower_bounds.numpy()[0]}")
    print(f"edge_bvh.upper_bounds[0] = {edge_bvh.upper_bounds.numpy()[0]}")

    test_vert = wp.array([wp.vec3(2, 0, 0.5), wp.vec3(0, 2, 0.5)], dtype=wp.vec3)
    test_edge = wp.array([[0, 1, 0, 1]], dtype=int)
    test_result = wp.array(shape=(2, 1), dtype=int)
    tri_bvh.aabb_vs_line(test_vert, test_edge, test_result)
    print(test_result)

    test_vert = wp.array([wp.vec3(0.5, 0.5, 1.5)], dtype=wp.vec3)
    tri_bvh.triangle_vs_point(test_vert, pos, tri_indices, test_result, False, 1, 1.0)
    print(test_result)
