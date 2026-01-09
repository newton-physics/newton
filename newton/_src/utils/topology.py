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


def topological_sort(
    joints: Sequence[tuple[int, int]] | Sequence[tuple[str, str]],
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

    incoming = defaultdict(set)
    outgoing = defaultdict(set)
    nodes = set()
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

        def visit(node):
            visited.add(node)
            # sort by joint ID to retain original order if topological order is not unique
            outs = sorted(outgoing[node], key=lambda x: x[0])
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
            outs = sorted(outgoing[node], key=lambda x: x[0])
            for joint_id, child in outs:
                if child in visited:
                    raise ValueError(f"Joint graph contains a cycle at body {child}")
                joint_order.append(joint_id)
                queue.append(child)

    if custom_indices is not None:
        joint_order = [custom_indices[i] for i in joint_order]
    return joint_order
