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

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence
from typing import TypeVar, cast

NodeT = TypeVar("NodeT", int, str)
"""A generic type variable for nodes in a topology, which can be either integers or strings."""


def _joint_key(item: tuple[int, NodeT]) -> int:
    return item[0]


def topological_sort(
    joints: Sequence[tuple[NodeT, NodeT]],
    custom_indices: Sequence[int] | None = None,
    use_dfs: bool = True,
    ensure_single_root: bool = False,
) -> list[int]:
    """
    Topological sort of a list of joints connecting rigid bodies.

    Args:
        joints (Sequence[tuple[int, int]] | Sequence[tuple[str, str]]): A list of body link pairs (parent, child). Bodies can be identified by their name or index.
        custom_indices (Sequence[int] | None): A list of custom indices to return for the joints. If None, the joint indices will be used.
        use_dfs (bool): If True, use depth-first search for topological sorting.
            If False, use Kahn's algorithm. Default is True.
        ensure_single_root (bool): If True, raise a ValueError if there is more than one root body. Default is False.

    Returns:
        list[int]: A list of joint indices in topological order.
    """
    if custom_indices is not None and len(custom_indices) != len(joints):
        raise ValueError(
            f"Length of custom indices must match length of joints: {len(custom_indices)} != {len(joints)}"
        )

    incoming: dict[NodeT, set[tuple[int, NodeT]]] = defaultdict(set)
    outgoing: dict[NodeT, set[tuple[int, NodeT]]] = defaultdict(set)
    nodes: set[NodeT] = set()
    for joint_id, (parent, child) in enumerate(joints):
        if len(incoming[child]) == 1:
            raise ValueError(f"Multiple joints lead to body {child}")
        incoming[child].add((joint_id, parent))
        outgoing[parent].add((joint_id, child))
        nodes.add(parent)
        nodes.add(child)

    roots = nodes - set(incoming.keys())
    if len(roots) == 0:
        raise ValueError("No root found in the joint graph.")
    if ensure_single_root and len(roots) > 1:
        raise ValueError(f"Multiple roots found in the joint graph: {roots}")

    joint_order: list[int] = []
    visited = set()

    if use_dfs:

        def visit(node: NodeT) -> None:
            visited.add(node)
            # sort by joint ID to retain original order if topological order is not unique
            outs = sorted(outgoing[node], key=_joint_key)
            for joint_id, child in outs:
                if child in visited:
                    raise ValueError(f"Joint graph contains a cycle at body {child}")
                joint_order.append(joint_id)
                visit(child)

        roots = sorted(roots)
        for root in roots:
            visit(root)
    else:
        # Breadth-first search (Kahn's algorithm)
        queue = deque(sorted(roots))
        while queue:
            node = queue.popleft()
            visited.add(node)
            outs = sorted(outgoing[node], key=_joint_key)
            for joint_id, child in outs:
                if child in visited:
                    raise ValueError(f"Joint graph contains a cycle at body {child}")
                joint_order.append(joint_id)
                queue.append(child)

    if custom_indices is not None:
        joint_order = [custom_indices[i] for i in joint_order]
    return joint_order


def topological_sort_undirected(
    joints: Sequence[tuple[NodeT, NodeT]],
    custom_indices: Sequence[int] | None = None,
    use_dfs: bool = True,
    ensure_single_root: bool = False,
) -> tuple[list[int], list[int]]:
    """
    Topological sort of a list of joints treating the graph as undirected.

    This function orients each joint edge during traversal to produce a valid
    parent-before-child ordering, and returns the joints that were reversed
    relative to the input orientation.

    Args:
        joints (Sequence[tuple[int, int]] | Sequence[tuple[str, str]]): A list of body link pairs.
            Bodies can be identified by their name or index.
        custom_indices (Sequence[int] | None): A list of custom indices to return for the joints.
            If None, the joint indices will be used.
        use_dfs (bool): If True, use depth-first search for topological sorting.
            If False, use a breadth-first traversal. Default is True.
        ensure_single_root (bool): If True, raise a ValueError if there is more than one root
            component. Default is False.

    Returns:
        tuple[list[int], list[int]]: A tuple of (joint_order, reversed_joints).
            joint_order is a list of joint indices in topological order.
            reversed_joints contains joint indices that were reversed during traversal.
    """
    if custom_indices is not None and len(custom_indices) != len(joints):
        raise ValueError(
            f"Length of custom indices must match length of joints: {len(custom_indices)} != {len(joints)}"
        )

    adjacency: dict[NodeT, list[tuple[int, NodeT]]] = defaultdict(list)
    nodes: set[NodeT] = set()
    for joint_id, (parent, child) in enumerate(joints):
        adjacency[parent].append((joint_id, child))
        adjacency[child].append((joint_id, parent))
        nodes.add(parent)
        nodes.add(child)

    if not nodes:
        return [], []

    joint_order: list[int] = []
    reversed_joints: list[int] = []
    visited = set()

    def record_edge(node: NodeT, neighbor: NodeT, joint_id: int) -> None:
        original_parent, original_child = joints[joint_id]
        if original_parent == node and original_child == neighbor:
            reversed_edge = False
        elif original_parent == neighbor and original_child == node:
            reversed_edge = True
        else:
            raise ValueError(f"Joint {joint_id} does not connect {node} and {neighbor}")
        if reversed_edge:
            reversed_joints.append(joint_id)
        joint_order.append(joint_id)

    def sorted_roots() -> list[NodeT]:
        roots = sorted(nodes)
        if -1 in nodes:
            roots = cast(list[NodeT], [-1] + [node for node in roots if node != -1])
        return roots

    if use_dfs:

        def visit(node: NodeT, parent: NodeT | None = None) -> None:
            visited.add(node)
            outs = sorted(adjacency[node], key=_joint_key)
            for joint_id, neighbor in outs:
                if neighbor == parent:
                    continue
                if neighbor in visited:
                    raise ValueError(f"Joint graph contains a cycle at body {neighbor}")
                record_edge(node, neighbor, joint_id)
                visit(neighbor, node)

        for root in sorted_roots():
            if root in visited:
                continue
            if ensure_single_root and visited:
                raise ValueError("Multiple roots found in the joint graph.")
            visit(root)
    else:
        queue = deque()
        for root in sorted_roots():
            if root in visited:
                continue
            if ensure_single_root and visited:
                raise ValueError("Multiple roots found in the joint graph.")
            queue.append((root, None))
            visited.add(root)
            while queue:
                node, parent = queue.popleft()
                outs = sorted(adjacency[node], key=_joint_key)
                for joint_id, neighbor in outs:
                    if neighbor == parent:
                        continue
                    if neighbor in visited:
                        raise ValueError(f"Joint graph contains a cycle at body {neighbor}")
                    record_edge(node, neighbor, joint_id)
                    visited.add(neighbor)
                    queue.append((neighbor, node))

    if custom_indices is not None:
        joint_order = [custom_indices[i] for i in joint_order]
        reversed_joints = [custom_indices[i] for i in reversed_joints]
    return joint_order, reversed_joints
