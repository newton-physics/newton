# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rod topology for :class:`~newton.solvers.SolverVBD`'s direct rod solve (``rigid_rod_direct_solve``)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from ...sim import JointType


@dataclass
class RodDirectLayout:
    """Device arrays describing the rods solved directly (see :func:`allocate_rod_direct_layout`).

    The arrays are allocated once, at a capacity set by the model's ROD joints, and
    :func:`update_rod_direct_layout` rewrites their contents in place, so a captured graph keeps reading the current
    layout. Unused rods are empty (``rod_start[r] == rod_start[r + 1]``).
    """

    rod_capacity: int
    node_capacity: int
    candidate_bodies: np.ndarray  # host: every body on a ROD joint, sorted
    candidate_joints: np.ndarray  # host: every ROD joint, sorted
    rod_bodies: wp.array  # (node_capacity,) the candidate bodies; those not in a rod are skipped by body_node
    rod_start: wp.array  # (rod_capacity + 1,) node offsets; nodes of a rod are in elimination order
    node_body: wp.array  # (node_capacity,) body index
    node_parent: wp.array  # (node_capacity,) node index of the elimination parent, -1 at the root
    node_joint: wp.array  # (node_capacity,) ROD joint to the elimination parent, -1 at the root
    node_side_parent: wp.array  # (node_capacity,) 1 if the node's body is node_joint's parent body
    bnd_start: wp.array  # (node_capacity + 1,) CSR offsets of one-sided ROD joints per node
    bnd_joint: wp.array  # one-sided ROD joints (other end kinematic, or closing a loop)
    body_node: wp.array  # (body_count,) node index of a directly solved body, -1 otherwise
    rod_count: int = 0  # rods in the current layout
    node_count: int = 0  # directly solved bodies in the current layout
    is_rod_body: np.ndarray | None = None  # host mask (body_count,) of the current layout


def rod_candidates(model) -> tuple[np.ndarray, np.ndarray]:
    """Every body on a ROD joint and every ROD joint, whatever their enablement and body flags (sorted)."""
    jt = model.joint_type.numpy()
    joints = np.flatnonzero(jt == int(JointType.ROD))
    ends = np.concatenate((model.joint_parent.numpy()[joints], model.joint_child.numpy()[joints]))
    return np.unique(ends[ends >= 0]).astype(np.int32), joints.astype(np.int32)


def allocate_rod_direct_layout(model, device) -> RodDirectLayout | None:
    """Allocate the layout at the capacity of the model's ROD joints and fill it; ``None`` without ROD joints.

    The capacity counts every ROD joint and every body on one, whatever their enablement and body flags, so later
    enablement, body-flag and mass changes fit without reallocation.
    """
    bodies, joints = rod_candidates(model)
    if bodies.size == 0:
        return None
    n = int(bodies.size)

    def arr(size: int) -> wp.array:
        return wp.full(max(size, 1), -1, dtype=wp.int32, device=device)

    layout = RodDirectLayout(
        rod_capacity=n,
        node_capacity=n,
        candidate_bodies=bodies,
        candidate_joints=joints,
        rod_bodies=wp.array(bodies, dtype=wp.int32, device=device),
        rod_start=arr(n + 1),
        node_body=arr(n),
        node_parent=arr(n),
        node_joint=arr(n),
        node_side_parent=arr(n),
        bnd_start=arr(n + 1),
        bnd_joint=arr(2 * int(joints.size)),
        body_node=arr(model.body_count),
    )
    update_rod_direct_layout(layout, model)
    return layout


def update_rod_direct_layout(layout: RodDirectLayout, model) -> None:
    """Group dynamic bodies joined by enabled ROD joints into rods (a forest) and write the layout in place.

    A dynamic body whose only enabled ROD joints lead to kinematic or static bodies is a rod of one segment. The
    layout reflects the model's joint enablement, body flags and inverse masses when it is written; the solver
    rewrites it from :meth:`~newton.solvers.SolverVBD.notify_model_changed`.

    Raises:
        RuntimeError: If a joint became a ROD joint, or a ROD joint moved to a body without one, since the layout was
            allocated (the layout would exceed its capacity).
    """
    bodies, joints = rod_candidates(model)
    if not (np.isin(joints, layout.candidate_joints).all() and np.isin(bodies, layout.candidate_bodies).all()):
        raise RuntimeError(
            "SolverVBD rigid_rod_direct_solve: a joint became a ROD joint, or a ROD joint moved to a body that had "
            "none, after the solver was created; the rod layout's capacity is fixed, so create a new solver (and "
            "re-capture any graph that steps it)."
        )
    rods = _rod_forest(model)
    n_cap, r_cap = layout.node_capacity, layout.rod_capacity
    node_count = len(rods["node_body"])

    def pad(x, size: int, fill: int) -> np.ndarray:
        out = np.full(max(size, 1), fill, dtype=np.int32)
        out[: len(x)] = x
        return out

    layout.rod_start.assign(pad(rods["rod_start"], r_cap + 1, node_count))
    layout.node_body.assign(pad(rods["node_body"], n_cap, -1))
    layout.node_parent.assign(pad(rods["node_parent"], n_cap, -1))
    layout.node_joint.assign(pad(rods["node_joint"], n_cap, -1))
    layout.node_side_parent.assign(pad(rods["node_side"], n_cap, 0))
    layout.bnd_start.assign(pad(rods["bnd_start"], n_cap + 1, len(rods["bnd_joint"])))
    layout.bnd_joint.assign(pad(rods["bnd_joint"], layout.bnd_joint.shape[0], -1))
    body_node = np.full(model.body_count, -1, dtype=np.int32)
    body_node[rods["node_body"]] = np.arange(node_count, dtype=np.int32)
    layout.body_node.assign(body_node)
    layout.rod_count = len(rods["rod_start"]) - 1
    layout.node_count = node_count
    layout.is_rod_body = body_node >= 0


def _rod_forest(model) -> dict[str, list[int]]:
    jt = model.joint_type.numpy()
    jp = model.joint_parent.numpy()
    jc = model.joint_child.numpy()
    en = model.joint_enabled.numpy().astype(bool)
    inv_m = model.body_inv_mass.numpy()
    flags = model.body_flags.numpy() if getattr(model, "body_flags", None) is not None else np.zeros_like(inv_m)
    kinematic_bit = int(_body_flag_kinematic())
    dynamic = (inv_m > 0.0) & ((flags.astype(np.int64) & kinematic_bit) == 0)

    rod_joints = [j for j in range(model.joint_count) if jt[j] == int(JointType.ROD) and en[j]]
    parent = np.arange(model.body_count)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    adj: dict[int, list[tuple[int, int]]] = {}
    one_sided: dict[int, list[int]] = {}
    for j in rod_joints:
        p, c = int(jp[j]), int(jc[j])
        dp, dc = p >= 0 and dynamic[p], c >= 0 and dynamic[c]
        if dp and dc and find(p) != find(c):
            parent[find(p)] = find(c)
            adj.setdefault(p, []).append((c, j))
            adj.setdefault(c, []).append((p, j))
            continue
        for b, ok in ((p, dp), (c, dc)):  # loop closure or a kinematic/static partner
            if ok:
                one_sided.setdefault(b, []).append(j)

    out: dict[str, list[int]] = {k: [] for k in ("node_body", "node_parent", "node_joint", "node_side", "bnd_joint")}
    out["rod_start"], out["bnd_start"] = [0], [0]
    seen: set[int] = set()
    for b0 in sorted(set(adj) | set(one_sided)):
        if b0 in seen:
            continue
        comp = _component(b0, adj)
        root = min((b for b in comp if len(adj.get(b, [])) <= 1), default=b0)
        order, up = _bfs_tree(root, adj)  # up[b] = (tree parent body, joint)
        base = len(out["node_body"])
        index = {b: base + k for k, b in enumerate(reversed(order))}  # leaves first, root last
        for b in reversed(order):
            pb, j = up.get(b, (-1, -1))
            out["node_body"].append(b)
            out["node_parent"].append(index[pb] if pb >= 0 else -1)
            out["node_joint"].append(j)
            out["node_side"].append(1 if j >= 0 and int(jp[j]) == b else 0)
        seen.update(comp)
        out["rod_start"].append(len(out["node_body"]))
    for b in out["node_body"]:
        out["bnd_joint"].extend(one_sided.get(b, []))
        out["bnd_start"].append(len(out["bnd_joint"]))
    return out


def _body_flag_kinematic():
    from ...sim import BodyFlags  # noqa: PLC0415

    return BodyFlags.KINEMATIC


def _component(b0: int, adj: dict[int, list[tuple[int, int]]]) -> list[int]:
    stack, comp = [b0], {b0}
    while stack:
        b = stack.pop()
        for nb, _ in adj.get(b, []):
            if nb not in comp:
                comp.add(nb)
                stack.append(nb)
    return sorted(comp)


def _bfs_tree(root: int, adj: dict[int, list[tuple[int, int]]]) -> tuple[list[int], dict[int, tuple[int, int]]]:
    order, up, frontier = [root], {}, [root]
    seen = {root}
    while frontier:
        nxt = []
        for b in frontier:
            for nb, j in adj.get(b, []):
                if nb not in seen:
                    seen.add(nb)
                    up[nb] = (b, j)
                    order.append(nb)
                    nxt.append(nb)
        frontier = nxt
    return order, up
