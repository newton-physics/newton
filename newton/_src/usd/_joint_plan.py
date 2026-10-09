# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Group and order articulation joints without reading USD or changing a builder."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from ..utils import topology


@dataclass
class _ArticulationJointPlan:
    """Group and order joints using already-resolved body IDs.

    IDs are local to one articulation, with -1 for the world. The caller reads
    and filters the joints after preparing the bodies; this plan stores no USD objects.
    """

    joint_names: list[str] = field(default_factory=list, init=False)
    joint_edges: list[tuple[int, int]] = field(default_factory=list, init=False)
    merged_joint_groups: dict[str, list[str]] = field(default_factory=dict, init=False)
    joint_excluded: set[str] = field(default_factory=set, init=False)
    _body_pair_to_representative: dict[tuple[int, int], str] = field(default_factory=dict, init=False, repr=False)

    def add_joint(self, joint_path: str, parent_id: int, child_id: int, *, excluded: bool) -> None:
        """Group a joint by its ordered body pair, or keep it outside the tree."""
        if excluded:
            self.joint_excluded.add(joint_path)
            return
        body_pair = (parent_id, child_id)
        if body_pair in self._body_pair_to_representative:
            rep = self._body_pair_to_representative[body_pair]
            self.merged_joint_groups[rep].append(joint_path)
        else:
            self._body_pair_to_representative[body_pair] = joint_path
            self.merged_joint_groups[joint_path] = [joint_path]
            self.joint_edges.append(body_pair)
            self.joint_names.append(joint_path)

    def get_joint_order(
        self, joint_ordering: Literal["bfs", "dfs"] | None, *, verbose: bool = False
    ) -> list[int] | npt.NDArray[np.intp]:
        """Order the joints and report invalid trees.

        ``None`` keeps source order without checking the graph. Excluded joints
        are not included in the ordering.
        """
        if not self.joint_edges:
            return []
        if joint_ordering is None:
            return np.arange(len(self.joint_names))
        if verbose:
            print(f"Sorting joints using {joint_ordering} ordering...")
        sorted_joints, reversed_joint_list = topology.topological_sort_undirected(
            self.joint_edges, use_dfs=joint_ordering == "dfs", ensure_single_root=True
        )
        if reversed_joint_list:
            reversed_joint_paths = [self.joint_names[joint_id] for joint_id in reversed_joint_list]
            reversed_joint_names = ", ".join(reversed_joint_paths)
            raise ValueError(
                f"Reversed joints are not supported: {reversed_joint_names}. Ensure that the joint parent body is defined as physics:body0 and the child is defined as physics:body1 in the joint prim."
            )
        if verbose:
            print("Joint ordering:", sorted_joints)
        return sorted_joints
