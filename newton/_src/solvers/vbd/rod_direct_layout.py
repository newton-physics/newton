# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rod topology for :class:`~newton.solvers.SolverVBD`'s direct rod solve (``rigid_rod_direct_solve``): the layout,
its membership masks and status bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from ...sim import BodyFlags, JointType
from .rigid_vbd_kernels import _reset_world_selected


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
    body_dynamic: wp.array  # (body_count,) 1 if the body was dynamic when the layout was written
    # (node_capacity,) stale slot of the node's rod (see check_rod_membership): its world, or world_count + 1 if the
    # rod holds global bodies or spans worlds
    node_slot: wp.array
    is_rod_body: np.ndarray | None = None  # host mask (body_count,) of the current layout


@wp.kernel(enable_backward=False)
def split_rod_inverse_mass(
    body_node: wp.array[wp.int32],
    node_slot: wp.array[wp.int32],
    body_inv_mass: wp.array[float],
    stale: wp.array[wp.int32],
    global_slot: int,
    # outputs
    inv_mass_color: wp.array[float],
    inv_mass_rod: wp.array[float],
):
    """Split inverse masses between the color loop (rod bodies zeroed, so its kernels skip them) and the rod step.

    Bodies of a rod whose layout is stale (see :func:`check_rod_membership`) stay in the color loop."""
    b = wp.tid()
    n = body_node[b]
    direct = False
    if n >= 0:
        direct = stale[node_slot[n]] == 0 and stale[global_slot] == 0
    if direct:
        inv_mass_color[b] = 0.0
        inv_mass_rod[b] = body_inv_mass[b]
    else:
        inv_mass_color[b] = body_inv_mass[b]
        inv_mass_rod[b] = 0.0


@wp.kernel(enable_backward=False)
def check_rod_membership(
    rod_bodies: wp.array[wp.int32],
    body_dynamic: wp.array[wp.int32],
    body_inv_mass: wp.array[float],
    body_flags: wp.array[wp.int32],
    kinematic_bit: int,
    body_world: wp.array[wp.int32],
    world_count: int,
    # output
    stale: wp.array[wp.int32],
):
    """Flag the layout as stale where a candidate body became dynamic or stopped being dynamic since it was written.

    ``stale`` has ``world_count + 2`` slots: one per world, one for global bodies (``world_count``) and one for any
    change (``world_count + 1``). A rod is stale if its own slot (``node_slot``) or the global slot is set, so a change
    in one world leaves the rods of other worlds on the direct path, while rods that hold global bodies or span worlds
    react to any change. A set slot is 1, or 2 once the rod step has reported it.
    """
    b = rod_bodies[wp.tid()]
    dynamic = int(0)
    if body_inv_mass[b] > 0.0 and (body_flags[b] & kinematic_bit) == 0:
        dynamic = 1
    if dynamic != body_dynamic[b]:
        w = body_world[b]
        if w < 0:
            w = world_count
        wp.atomic_max(stale, w, 1)
        wp.atomic_max(stale, world_count + 1, 1)


@wp.kernel(enable_backward=False)
def clear_rod_direct_status(
    world_mask: wp.array[wp.bool],
    reset_all: bool,
    world_count: int,
    body_world: wp.array[wp.int32],
    rod_start: wp.array[wp.int32],
    node_body: wp.array[wp.int32],
    # in/out
    status: wp.array[wp.int32],
    prior_status: wp.array[wp.int32],
):
    """Clear the status of rods in the reset worlds; the bits of earlier layouts clear only when every world resets."""
    r = wp.tid()
    if r == status.shape[0]:
        all_selected = bool(reset_all)
        if not reset_all:
            unselected = int(0)
            for w in range(world_mask.shape[0]):
                if not world_mask[w]:
                    unselected = unselected + 1
            all_selected = unselected == 0
        if all_selected:
            prior_status[0] = 0
        return
    s = rod_start[r]
    if rod_start[r + 1] > s:
        if _reset_world_selected(body_world[node_body[s]], world_mask, reset_all, world_count):
            status[r] = 0


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
        body_dynamic=arr(model.body_count),
        node_slot=arr(n),
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
    rods, dynamic = _rod_forest(model)
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
    layout.body_dynamic.assign(dynamic.astype(np.int32))
    layout.node_slot.assign(pad(_rod_slots(rods, model), n_cap, model.world_count + 1))
    layout.is_rod_body = body_node >= 0


def _rod_slots(rods: dict[str, list[int]], model) -> np.ndarray:
    """Stale slot per node: the world of the node's rod, or ``world_count + 1`` for rods with global bodies or bodies
    of several worlds."""
    body_world = model.body_world.numpy()
    node_world = body_world[np.asarray(rods["node_body"], dtype=np.int64)]
    slots = np.full(len(node_world), model.world_count + 1, dtype=np.int32)
    start = rods["rod_start"]
    for r in range(len(start) - 1):
        worlds = np.unique(node_world[start[r] : start[r + 1]])
        if worlds.size == 1 and worlds[0] >= 0:
            slots[start[r] : start[r + 1]] = worlds[0]
    return slots


def _rod_forest(model) -> tuple[dict[str, list[int]], np.ndarray]:
    jt = model.joint_type.numpy()
    jp = model.joint_parent.numpy()
    jc = model.joint_child.numpy()
    en = model.joint_enabled.numpy().astype(bool)
    inv_m = model.body_inv_mass.numpy()
    flags = model.body_flags.numpy() if getattr(model, "body_flags", None) is not None else np.zeros_like(inv_m)
    dynamic = body_dynamic_mask(inv_m, flags)

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
    return out, dynamic


def body_dynamic_mask(inv_mass: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """Bodies the direct solve may move: positive inverse mass and not kinematic."""
    return (inv_mass > 0.0) & ((flags.astype(np.int64) & int(BodyFlags.KINEMATIC)) == 0)


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
