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

import warnings
from enum import Enum
from typing import Literal

import numpy as np
import warp as wp


class ColoringAlgorithm(Enum):
    MCS = 0
    GREEDY = 1


@wp.kernel
def validate_graph_coloring(edge_indices: wp.array(dtype=int, ndim=2), colors: wp.array(dtype=int)):
    edge_idx = wp.tid()
    e_v_1 = edge_indices[edge_idx, 0]
    e_v_2 = edge_indices[edge_idx, 1]

    wp.expect_neq(colors[e_v_1], colors[e_v_2])


@wp.kernel
def count_color_group_size(
    colors: wp.array(dtype=int),
    group_sizes: wp.array(dtype=int),
):
    for particle_idx in range(colors.shape[0]):
        particle_color = colors[particle_idx]
        group_sizes[particle_color] = group_sizes[particle_color] + 1


@wp.kernel
def fill_color_groups(
    colors: wp.array(dtype=int),
    group_fill_count: wp.array(dtype=int),
    group_offsets: wp.array(dtype=int),
    # flattened color groups
    color_groups_flatten: wp.array(dtype=int),
):
    for particle_idx in range(colors.shape[0]):
        particle_color = colors[particle_idx]
        group_offset = group_offsets[particle_color]
        group_idx = group_fill_count[particle_color]
        color_groups_flatten[group_idx + group_offset] = wp.int32(particle_idx)

        group_fill_count[particle_color] = group_idx + 1


def convert_to_color_groups(num_colors, particle_colors, return_wp_array=False, device="cpu") -> list[int]:
    group_sizes = wp.zeros(shape=(num_colors,), dtype=int, device="cpu")
    wp.launch(kernel=count_color_group_size, inputs=[particle_colors, group_sizes], device="cpu", dim=1)

    group_sizes_np = group_sizes.numpy()
    group_offsets_np = np.concatenate([np.array([0]), np.cumsum(group_sizes_np)])
    group_offsets = wp.array(group_offsets_np, dtype=int, device="cpu")

    group_fill_count = wp.zeros(shape=(num_colors,), dtype=int, device="cpu")
    color_groups_flatten = wp.empty(shape=(group_sizes_np.sum(),), dtype=int, device="cpu")
    wp.launch(
        kernel=fill_color_groups,
        inputs=[particle_colors, group_fill_count, group_offsets, color_groups_flatten],
        device="cpu",
        dim=1,
    )

    color_groups_flatten_np = color_groups_flatten.numpy()

    color_groups = []
    if return_wp_array:
        for color_idx in range(num_colors):
            color_groups.append(
                wp.array(
                    color_groups_flatten_np[group_offsets_np[color_idx] : group_offsets_np[color_idx + 1]],
                    dtype=int,
                    device=device,
                )
            )
    else:
        for color_idx in range(num_colors):
            color_groups.append(color_groups_flatten_np[group_offsets_np[color_idx] : group_offsets_np[color_idx + 1]])

    return color_groups


def _canonicalize_edges_np(edges_np: np.ndarray) -> np.ndarray:
    """Sort edge endpoints and drop duplicate edges."""
    if edges_np.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    edges_sorted = np.sort(edges_np, axis=1)
    edges_unique = np.unique(edges_sorted, axis=0)
    return edges_unique.astype(np.int32)


def construct_tetmesh_graph_edges(tet_indices: np.array, tet_active_mask):
    """
    Convert tet connectivity (n_tets x 4) into unique graph edges (u, v).
    """
    if tet_indices is None:
        edges_np = np.empty((0, 2), dtype=np.int32)
    else:
        if isinstance(tet_indices, wp.array):
            tet_np = tet_indices.numpy()
        else:
            tet_np = np.asarray(tet_indices, dtype=np.int32)

        if tet_active_mask is not None:
            mask_arr = np.asarray(tet_active_mask > 0, dtype=bool)
            # Handle scalar mask (True means all active, False means none active)
            if mask_arr.ndim == 0:
                if not mask_arr:
                    tet_np = tet_np[:0]  # Empty array
                # else: all active, no filtering needed
            else:
                tet_np = tet_np[mask_arr]

        if tet_np.size == 0:
            edges_np = np.empty((0, 2), dtype=np.int32)
        else:
            v0 = tet_np[:, 0]
            v1 = tet_np[:, 1]
            v2 = tet_np[:, 2]
            v3 = tet_np[:, 3]
            edges_np = np.stack(
                [
                    np.stack([v0, v1], axis=1),
                    np.stack([v0, v2], axis=1),
                    np.stack([v0, v3], axis=1),
                    np.stack([v1, v2], axis=1),
                    np.stack([v1, v3], axis=1),
                    np.stack([v2, v3], axis=1),
                ],
                axis=0,
            ).reshape(-1, 2)
    edges_np = _canonicalize_edges_np(edges_np)

    return edges_np


def construct_trimesh_graph_edges(
    tri_indices,
    tri_active_mask=None,
    bending_edge_indices=None,
    bending_edge_active_mask=None,
    return_wp_array=True,
):
    """
    A function that generates vertex coloring for a trimesh, which is represented by the number of vertices and edges of the mesh.
    It will convert the trimesh to a graph and then apply coloring.
    It returns a list of `np.array` with `dtype`=`int`. The length of the list is the number of colors
    and each `np.array` contains the indices of vertices with this color.

    Args:
        num_nodes: The number of the nodes in the graph
        trimesh_edge_indices: A `wp.array` with of shape (number_edges, 4), each row is (o1, o2, v1, v2), see `sim.Model`'s definition of `edge_indices`.
        include_bending_energy: whether to consider bending energy in the coloring process. If set to `True`, the generated
            graph will contain all the edges connecting o1 and o2; otherwise, the graph will be equivalent to the trimesh.
    """
    edges_np_list = []

    # Primary triangle edges
    if tri_indices is not None:
        if isinstance(tri_indices, wp.array):
            tri_indices = tri_indices.numpy()

        if tri_indices.size > 0:
            if tri_active_mask is not None:
                mask_arr = np.asarray(tri_active_mask, dtype=bool)
                # Handle scalar mask (True means all active, False means none active)
                if mask_arr.ndim == 0:
                    if not mask_arr:
                        tri_indices = tri_indices[:0]  # Empty array
                    # else: all active, no filtering needed
                else:
                    tri_indices = tri_indices[mask_arr]
            if tri_indices.size > 0:
                v0 = tri_indices[:, 0]
                v1 = tri_indices[:, 1]
                v2 = tri_indices[:, 2]
                tri_edges = np.stack(
                    [
                        np.stack([v0, v1], axis=1),
                        np.stack([v0, v2], axis=1),
                        np.stack([v1, v2], axis=1),
                    ],
                    axis=0,
                ).reshape(-1, 2)
                edges_np_list.append(tri_edges)

    # Optional bending edges (hinges). Each row has four vertices; include all 2-combinations
    # of the active hinge vertices, skipping any vertex indices that are negative.
    if bending_edge_indices is not None:
        bend_np = np.asarray(bending_edge_indices, dtype=np.int32)
        if bend_np.size > 0:
            if bending_edge_active_mask is not None:
                mask = np.asarray(bending_edge_active_mask, dtype=np.int32) != 0
                bend_np = bend_np[mask]
            if bend_np.size > 0:
                v0 = bend_np[:, 0:1]
                v1 = bend_np[:, 1:2]
                v2 = bend_np[:, 2:3]
                v3 = bend_np[:, 3:4]

                pairs = np.concatenate(
                    [
                        np.concatenate([v0, v1], axis=1),
                        np.concatenate([v0, v2], axis=1),
                        np.concatenate([v0, v3], axis=1),
                        np.concatenate([v1, v2], axis=1),
                        np.concatenate([v1, v3], axis=1),
                        np.concatenate([v2, v3], axis=1),
                    ],
                    axis=0,
                )

                valid = np.all(pairs >= 0, axis=1)
                pairs = pairs[valid]

                if pairs.size > 0:
                    edges_np_list.append(pairs)

    if edges_np_list:
        edges_np = np.concatenate(edges_np_list, axis=0)
    else:
        edges_np = np.empty((0, 2), dtype=np.int32)

    edges = _canonicalize_edges_np(edges_np)

    if return_wp_array:
        edges = wp.array(edges, dtype=int, device="cpu")

    return edges


def construct_particle_graph(
    tri_graph_edges: np.array,
    tri_active_mask: np.array,
    bending_edge_indices: np.array,
    bending_edge_active_mask: np.array,
    tet_graph_edges_np: np.array,
    tet_active_mask: np.array,
):
    tri_graph_edges = construct_trimesh_graph_edges(
        tri_graph_edges,
        tri_active_mask,
        bending_edge_indices,
        bending_edge_active_mask,
        return_wp_array=False,
    )
    tet_graph_edges = construct_tetmesh_graph_edges(tet_graph_edges_np, tet_active_mask)

    merged_edges = _canonicalize_edges_np(np.vstack([tri_graph_edges, tet_graph_edges]).astype(np.int32))
    graph_edge_indices = wp.array(merged_edges, dtype=int, device="cpu")

    return graph_edge_indices


def color_graph(
    num_nodes,
    graph_edge_indices: wp.array(dtype=int, ndim=2),
    balance_colors: bool = True,
    target_max_min_color_ratio: float = 1.1,
    algorithm: ColoringAlgorithm = ColoringAlgorithm.MCS,
) -> list[int]:
    """
    A function that generates coloring for a graph, which is represented by the number of nodes and an array of edges.
    It returns a list of `np.array` with `dtype`=`int`. The length of the list is the number of colors
    and each `np.array` contains the indices of vertices with this color.

    Args:
        num_nodes: The number of the nodes in the graph
        graph_edge_indices: A `wp.array` of shape (number_edges, 2)
        balance_colors: Whether to apply the color balancing algorithm to balance the size of each color
        target_max_min_color_ratio: the color balancing algorithm will stop when the ratio between the largest color and
            the smallest color reaches this value
        algorithm: Value should an enum type of ColoringAlgorithm, otherwise it will raise an error. ColoringAlgorithm.mcs means using the MCS coloring algorithm,
            while ColoringAlgorithm.ordered_greedy means using the degree-ordered greedy algorithm. The MCS algorithm typically generates 30% to 50% fewer colors
            compared to the ordered greedy algorithm, while maintaining the same linear complexity. Although MCS has a constant overhead that makes it about twice
            as slow as the greedy algorithm, it produces significantly better coloring results. We recommend using MCS, especially if coloring is only part of the
            preprocessing stage.e.

    Note:

        References to the coloring algorithm:
        MCS: Pereira, F. M. Q., & Palsberg, J. (2005, November). Register allocation via coloring of chordal graphs. In Asian Symposium on Programming Languages and Systems (pp. 315-329). Berlin, Heidelberg: Springer Berlin Heidelberg.
        Ordered Greedy: Ton-That, Q. M., Kry, P. G., & Andrews, S. (2023). Parallel block Neo-Hookean XPBD using graph clustering. Computers & Graphics, 110, 1-10.
    """
    if num_nodes == 0:
        return []

    particle_colors = wp.empty(shape=(num_nodes), dtype=wp.int32, device="cpu")

    if graph_edge_indices.ndim != 2:
        raise ValueError(
            f"graph_edge_indices must be a 2 dimensional array! The provided one is {graph_edge_indices.ndim} dimensional."
        )
    if graph_edge_indices.device.is_cpu:
        indices = graph_edge_indices
    else:
        indices = wp.clone(graph_edge_indices, device="cpu")

    num_colors = wp._src.context.runtime.core.wp_graph_coloring(
        num_nodes,
        indices.__ctype__(),
        algorithm.value,
        particle_colors.__ctype__(),
    )

    if balance_colors:
        max_min_ratio = wp._src.context.runtime.core.wp_balance_coloring(
            num_nodes,
            indices.__ctype__(),
            num_colors,
            target_max_min_color_ratio,
            particle_colors.__ctype__(),
        )

        if max_min_ratio > target_max_min_color_ratio:
            warnings.warn(
                f"Color balancing terminated early: max/min ratio {max_min_ratio:.3f} "
                f"exceeds target {target_max_min_color_ratio:.3f}. "
                "The graph may not be further optimizable.",
                stacklevel=2,
            )

    color_groups = convert_to_color_groups(num_colors, particle_colors, return_wp_array=False)

    return color_groups


def plot_graph(
    vertices,
    edges,
    edge_labels=None,
    node_labels=None,
    node_colors=None,
    layout: Literal["spring", "kamada_kawai"] = "kamada_kawai",
):
    """
    Plots a graph using matplotlib and networkx.

    Args:
        vertices: A numpy array of shape (N,) containing the vertex indices.
        edges: A numpy array of shape (M, 2) containing the vertex indices of the edges.
        edge_labels: A list of edge labels.
        node_labels: A list of node labels.
        node_colors: A list of node colors.
        layout: The layout of the graph. Can be "spring" or "kamada_kawai".
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import networkx as nx  # noqa: PLC0415

    if edge_labels is None:
        edge_labels = []
    G = nx.DiGraph()
    name_to_index = {}
    for i, name in enumerate(vertices):
        G.add_node(i)
        name_to_index[name] = i
    g_edge_labels = {}
    for i, (a, b) in enumerate(edges):
        ai = a if isinstance(a, int) else name_to_index[a]
        bi = b if isinstance(b, int) else name_to_index[b]
        label = None
        if i < len(edge_labels):
            label = edge_labels[i]
            g_edge_labels[(ai, bi)] = label
        G.add_edge(ai, bi, label=label)

    if layout == "spring":
        pos = nx.spring_layout(G, k=3.5, iterations=200)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        raise ValueError(f"Invalid layout: {layout}")

    default_draw_args = {"alpha": 0.9, "edgecolors": "black", "linewidths": 0.5}
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, **default_draw_args)
    nx.draw_networkx_labels(
        G,
        pos,
        labels=dict(enumerate(node_labels if node_labels is not None else vertices)),
        font_size=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 0.5},
    )

    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrows=True, edge_color="black", node_size=1000)
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=g_edge_labels,
        font_color="darkslategray",
        font_size=8,
    )
    plt.axis("off")
    plt.show()


def combine_independent_particle_coloring(color_groups_1, color_groups_2) -> list[int]:
    """
    A function that combines 2 independent coloring groups. Note that color_groups_1 and color_groups_2 must be from 2 independent
    graphs so that there is no connection between them. This algorithm will sort color_groups_1 in ascending order and
    sort color_groups_2 in descending order, and combine each group with the same index, this way we are always combining
    the smaller group with the larger group.

    Args:
        color_groups_1: A list of `np.array` with `dtype`=`int`. The length of the list is the number of colors
            and each `np.array` contains the indices of vertices with this color.
        color_groups_2: A list of `np.array` with `dtype`=`int`. The length of the list is the number of colors
            and each `np.array` contains the indices of vertices with this color.

    """
    if len(color_groups_1) == 0:
        return color_groups_2
    if len(color_groups_2) == 0:
        return color_groups_1

    num_colors_after_combining = max(len(color_groups_1), len(color_groups_2))
    color_groups_combined = []

    # this made sure that the leftover groups are always the largest
    if len(color_groups_1) < len(color_groups_2):
        color_groups_1, color_groups_2 = color_groups_2, color_groups_1

    # sort group 1 in ascending order
    color_groups_1_sorted = sorted(color_groups_1, key=lambda group: len(group))
    # sort group 1 in descending order
    color_groups_2_sorted = sorted(color_groups_2, key=lambda group: -len(group))
    # so that we are combining the smaller group with the larger group
    # which will balance the load of each group

    for i in range(num_colors_after_combining):
        group_1 = color_groups_1_sorted[i] if i < len(color_groups_1) else None
        group_2 = color_groups_2_sorted[i] if i < len(color_groups_2) else None

        if group_1 is not None and group_2 is not None:
            color_groups_combined.append(np.concatenate([group_1, group_2]))
        elif group_1 is not None:
            color_groups_combined.append(group_1)
        else:
            color_groups_combined.append(group_2)

    return color_groups_combined


def color_rigid_bodies(
    num_bodies: int,
    joint_parent: list[int],
    joint_child: list[int],
    balance_colors: bool = True,
    target_max_min_color_ratio: float = 1.1,
    algorithm: ColoringAlgorithm = ColoringAlgorithm.MCS,
):
    """
    Generate a graph coloring for rigid bodies from joint connectivity.

    Bodies connected by a joint are treated as adjacent in the graph and cannot share
    the same color. The result can be used to schedule per-color parallel processing
    (e.g. in the VBD solver) without conflicts.

    Returns a list of ``np.ndarray`` with ``dtype=int``. The list length is the number
    of colors, and each array contains the body indices of that color. This mirrors the
    return format of ``color_trimesh``/``color_graph``.

    Args:
        num_bodies: Number of bodies (graph nodes).
        joint_parent: Parent body indices for each joint (use -1 for world).
        joint_child: Child body indices for each joint.
        balance_colors: Whether to balance color group sizes.
        target_max_min_color_ratio: Stop balancing when max/min group size ratio reaches this value.
        algorithm: Coloring algorithm to use.
    """
    if num_bodies == 0:
        return []

    # Build edge list from joint connections
    edge_list = []

    if len(joint_parent) != len(joint_child):
        raise ValueError(
            f"joint_parent and joint_child must have the same length (got {len(joint_parent)} and {len(joint_child)})"
        )

    for parent, child in zip(joint_parent, joint_child, strict=True):
        if parent != -1 and child != -1 and parent != child:
            edge_list.append([parent, child])

    if not edge_list:
        # No joints between bodies, all can have same color
        return [np.arange(num_bodies, dtype=int)]

    # Convert to numpy array for processing
    edge_indices = np.array(edge_list, dtype=int)

    # Convert to warp array for the existing color_graph function
    edge_indices_wp = wp.array(edge_indices, dtype=int, device="cpu")

    # Use existing color_graph function
    color_groups = color_graph(num_bodies, edge_indices_wp, balance_colors, target_max_min_color_ratio, algorithm)

    return color_groups
