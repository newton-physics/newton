# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cached propagation response of SolverFeatherPGS against the tree walk.

Within one pass the tree factorization is fixed, so the cached response matrices apply the same linear map as the
tree walk. Each lockstep test steps both from the same input state at every step of a contact-rich trajectory,
since the chaotic scenes diverge in free run from float reassociation alone.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

N_WORLDS = 4
DT = 1.0 / 240.0


def _add_floating_chain(
    builder: newton.ModelBuilder, *, n_links: int, base_z: float, base_y: float, spheres: tuple[str, ...]
) -> None:
    """A floating box base trailing ``n_links`` revolute links, with collision spheres on ``spheres``."""
    no_collision = builder.default_shape_cfg.copy()
    no_collision.has_shape_collision = False
    no_collision.collision_group = 0

    base = builder.add_link(xform=wp.transform(wp.vec3(0.0, base_y, base_z), wp.quat_identity()))
    builder.add_shape_box(base, hx=0.1, hy=0.1, hz=0.1, cfg=no_collision)
    if "base" in spheres:
        builder.add_shape_sphere(base, radius=0.1)
    joints = [builder.add_joint_free(base)]
    prev = base
    for i in range(n_links):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.12, hy=0.03, hz=0.03, cfg=no_collision)
        if "deep" in spheres and i == n_links - 1:
            builder.add_shape_sphere(link, radius=0.1)
        if "mid" in spheres and i == n_links // 2:
            builder.add_shape_sphere(link, radius=0.1)
        offset = 0.22 if i == 0 else 0.12
        joints.append(
            builder.add_joint_revolute(
                parent=prev,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(wp.vec3(offset, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.12, 0.0, 0.0), wp.quat_identity()),
            )
        )
        prev = link
    builder.add_articulation(joints)


def _add_fixed_chain(
    builder: newton.ModelBuilder, *, n_links: int, base_z: float, base_y: float, spheres: tuple[str, ...]
) -> None:
    """A world-anchored all-revolute chain drooping onto the ground, with collision spheres on ``spheres``."""
    no_collision = builder.default_shape_cfg.copy()
    no_collision.has_shape_collision = False
    no_collision.collision_group = 0

    joints = []
    prev = -1
    for i in range(n_links):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.12, hy=0.03, hz=0.03, cfg=no_collision)
        if "base" in spheres and i == 0:
            builder.add_shape_sphere(link, radius=0.1)
        if "deep" in spheres and i == n_links - 1:
            builder.add_shape_sphere(link, radius=0.1)
        if "mid" in spheres and i == n_links // 2:
            builder.add_shape_sphere(link, radius=0.1)
        parent_xform = (
            wp.transform(wp.vec3(0.0, base_y, base_z), wp.quat_identity())
            if prev == -1
            else wp.transform(wp.vec3(0.12, 0.0, 0.0), wp.quat_identity())
        )
        joints.append(
            builder.add_joint_revolute(
                parent=prev,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=parent_xform,
                child_xform=wp.transform(wp.vec3(-0.12, 0.0, 0.0), wp.quat_identity()),
            )
        )
        prev = link
    builder.add_articulation(joints)


def _build_model(
    device: str, *, chains: tuple[int, ...], spheres: tuple[str, ...], fixed_base: bool = False
) -> newton.Model:
    """N_WORLDS identical worlds, each holding one chain per entry of ``chains``."""
    scene = newton.ModelBuilder()
    add_chain = _add_fixed_chain if fixed_base else _add_floating_chain
    for _ in range(N_WORLDS):
        world = newton.ModelBuilder()
        world.default_shape_cfg.density = 1000.0
        world.default_shape_cfg.mu = 0.7
        for k, n_links in enumerate(chains):
            add_chain(world, n_links=n_links, base_z=0.105, base_y=0.8 * k, spheres=spheres)
        scene.add_world(world)
    scene.add_ground_plane()
    return scene.finalize(device=device)


def _build_clutter_model(device: str, *, n_boxes: int) -> newton.Model:
    """A fixed-base arm plus free spheres resting on the ground.

    The spheres are frictionless so each has one disjoint row, which keeps the tree walk reproducible.
    """
    scene = newton.ModelBuilder()
    for _ in range(N_WORLDS):
        world = newton.ModelBuilder()
        world.default_shape_cfg.density = 1000.0
        world.default_shape_cfg.mu = 0.7
        # Every arm sphere starts pressed into the ground, so the active-body set is stable.
        _add_fixed_chain(world, n_links=6, base_z=0.095, base_y=0.0, spheres=("base", "mid", "deep"))
        clutter_cfg = world.default_shape_cfg.copy()
        clutter_cfg.mu = 0.0
        for b in range(n_boxes):
            ball = world.add_link(
                xform=wp.transform(wp.vec3(-0.4 - 0.25 * (b % 3), 0.6 + 0.25 * (b // 3), 0.0995), wp.quat_identity())
            )
            world.add_shape_sphere(ball, radius=0.1, cfg=clutter_cfg)
            world.add_articulation([world.add_joint_free(parent=-1, child=ball)])
        scene.add_world(world)
    scene.add_ground_plane()
    return scene.finalize(device=device)


def _tree_inputs(model, solver, art):
    return {
        "S": solver.propagation_joint_S_flat.numpy().astype(np.float64),
        "U": solver.propagation_tree_U.numpy().astype(np.float64),
        "Dinv": solver.propagation_tree_D_inv.numpy().astype(np.float64),
        "com": solver.propagation_body_com_rel.numpy().astype(np.float64),
        "jp": model.joint_parent.numpy(),
        "jc": model.joint_child.numpy(),
        "qd_start": model.joint_qd_start.numpy(),
        "j0": int(model.articulation_start.numpy()[art]),
        "j1": int(model.articulation_start.numpy()[art + 1]),
    }


def _xlt_wrench(w, e):
    return np.concatenate([w[:3], w[3:] + np.cross(e, w[:3])])


def _xlt_twist(v, e):
    return np.concatenate([v[:3] + np.cross(v[3:], e), v[3:]])


def _ref_propagate(ti, imp_by_body, v_out):
    """Mirror of kernels.propagate_tree_impulses_for_size (one articulation)."""
    S, U, Dinv, com = ti["S"], ti["U"], ti["Dinv"], ti["com"]
    jp, jc, qd_start = ti["jp"], ti["jc"], ti["qd_start"]
    j0, j1 = ti["j0"], ti["j1"]
    pA = {}
    u = np.zeros(S.shape[0])
    body_delta = {}
    for j in range(j0, j1):
        pA[int(jc[j])] = -np.asarray(imp_by_body.get(int(jc[j]), np.zeros(6)), dtype=np.float64)
        body_delta[int(jc[j])] = np.zeros(6)
    for j in range(j1 - 1, j0 - 1, -1):
        child, parent = int(jc[j]), int(jp[j])
        d0, d1 = int(qd_start[j]), int(qd_start[j + 1])
        for g in range(d0, d1):
            u[g] = -np.dot(S[g], pA[child])
        if parent >= 0:
            p = pA[child].copy()
            for a in range(d0, d1):
                coeff = sum(Dinv[j, a - d0, b - d0] * u[b] for b in range(d0, d1))
                p += U[a] * coeff
            pA[parent] += _xlt_wrench(p, com[child] - com[parent])
    for j in range(j0, j1):
        child, parent = int(jc[j]), int(jp[j])
        d0, d1 = int(qd_start[j]), int(qd_start[j + 1])
        pd = np.zeros(6)
        if parent >= 0:
            pd = _xlt_twist(body_delta[parent], com[child] - com[parent])
        qdd = np.zeros(d1 - d0)
        for a in range(d0, d1):
            acc = 0.0
            for b in range(d0, d1):
                pt = np.dot(U[b], pd) if parent >= 0 else 0.0
                acc += Dinv[j, a - d0, b - d0] * (u[b] - pt)
            qdd[a - d0] = acc
            v_out[a] += acc
        body_delta[child] = pd + sum(S[a] * qdd[a - d0] for a in range(d0, d1))
    # Recompute body twists from the updated v_out.
    body_qd = {}
    twist = {}
    for j in range(j0, j1):
        child, parent = int(jc[j]), int(jp[j])
        d0, d1 = int(qd_start[j]), int(qd_start[j + 1])
        val = np.zeros(6)
        if parent >= 0:
            val = _xlt_twist(twist[parent], com[child] - com[parent])
        for a in range(d0, d1):
            val = val + S[a] * v_out[a]
        twist[child] = val
        body_qd[child] = val
    return v_out, body_qd


def _make_solver(
    model: newton.Model, response: str, *, cached: bool, cache_max_bodies: int = 8, **solver_kwargs
) -> SolverFeatherPGS:
    return SolverFeatherPGS(
        model,
        articulated_contact_response=response,
        pgs_iterations=8,
        friction_anchor_beta=0.0,
        propagation_cached_response=cached,
        propagation_cached_response_max_bodies=cache_max_bodies,
        **solver_kwargs,
    )


def _seed_joint_qd(model: newton.Model) -> np.ndarray:
    n = int(model.joint_dof_count)
    return (0.4 * np.sin(0.7 * np.arange(n, dtype=np.float64) + 0.3)).astype(np.float32)


def _active_body_qd(solver, body_map_solver) -> dict[int, np.ndarray]:
    """COM twists of the contact-active bodies in ``body_map_solver``'s body list."""
    qd = solver.propagation_body_qd.numpy()
    counts = body_map_solver.propagation_body_count.numpy()
    body_list = body_map_solver.propagation_body_list.numpy()
    out: dict[int, np.ndarray] = {}
    for world in range(body_list.shape[0]):
        n = min(int(counts[world]), body_list.shape[1])
        for slot in range(n):
            body = int(body_list[world, slot])
            if body >= 0:
                out[body] = qd[body].copy()
    return out


class _LockstepStats:
    def __init__(self):
        self.max_diffs = {"joint_q": 0.0, "joint_qd": 0.0, "v_out": 0.0, "active_body_qd": 0.0}
        self.max_active_bodies = 0
        self.total_rows = 0


def _run_lockstep(model, response, n_steps, *, cache_max_bodies: int = 8, **solver_kwargs):
    """Step the cached solver from each state of a tree-walk trajectory and record the output diffs.

    A second tree-walk solver must match the first bitwise, so the diffs measure the cache alone.
    """
    ref = _make_solver(model, response, cached=False, cache_max_bodies=cache_max_bodies, **solver_kwargs)
    ref2 = _make_solver(model, response, cached=False, cache_max_bodies=cache_max_bodies, **solver_kwargs)
    test = _make_solver(model, response, cached=True, cache_max_bodies=cache_max_bodies, **solver_kwargs)

    state_in = model.state()
    ref_out = model.state()
    ref2_out = model.state()
    test_out = model.state()
    control = model.control()
    state_in.joint_qd.assign(_seed_joint_qd(model))
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    collision_pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = collision_pipeline.contacts()

    stats = _LockstepStats()
    for _ in range(n_steps):
        state_in.clear_forces()
        # A deterministic collide before each solver gives all three the same contacts.
        collision_pipeline.collide(state_in, contacts)
        test.step(state_in, test_out, control, contacts, DT)
        collision_pipeline.collide(state_in, contacts)
        ref2.step(state_in, ref2_out, control, contacts, DT)
        collision_pipeline.collide(state_in, contacts)
        ref.step(state_in, ref_out, control, contacts, DT)
        wp.synchronize()

        for key in ("joint_q", "joint_qd"):
            np.testing.assert_array_equal(
                getattr(ref_out, key).numpy(),
                getattr(ref2_out, key).numpy(),
                err_msg=f"tree-walk solve not reproducible for {key}; scene invalidates the gate",
            )

        for key in ("joint_q", "joint_qd"):
            diff = float(np.max(np.abs(getattr(test_out, key).numpy() - getattr(ref_out, key).numpy())))
            stats.max_diffs[key] = max(stats.max_diffs[key], diff)
        stats.max_diffs["v_out"] = max(
            stats.max_diffs["v_out"], float(np.max(np.abs(test.v_out.numpy() - ref.v_out.numpy())))
        )
        # The tree walk builds no body list, so both read the cached solver's.
        ref_body_qd = _active_body_qd(ref, test)
        test_body_qd = _active_body_qd(test, test)
        for body, ref_qd in ref_body_qd.items():
            diff = float(np.max(np.abs(test_body_qd[body] - ref_qd)))
            stats.max_diffs["active_body_qd"] = max(stats.max_diffs["active_body_qd"], diff)

        counts = test.propagation_body_count.numpy()
        stats.max_active_bodies = max(stats.max_active_bodies, int(counts.max()))
        stats.total_rows += int(ref.propagation_constraint_count.numpy().sum())

        state_in, ref_out = ref_out, state_in
    return stats, ref, test


TOL = 1.0e-4


def _assert_stats(test, stats, *, expect_min_bodies):
    test.assertGreater(stats.total_rows, 0, "no propagation contact rows produced")
    test.assertGreaterEqual(stats.max_active_bodies, expect_min_bodies)
    for key, diff in stats.max_diffs.items():
        test.assertLessEqual(diff, TOL, f"{key} per-step max abs diff {diff:.3e} exceeds {TOL:g}")


def _step_cached(model, steps):
    solver = _make_solver(model, "propagation", cached=True)
    state_in, state_out = model.state(), model.state()
    control = model.control()
    state_in.joint_qd.assign(_seed_joint_qd(model))
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    collision_pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = collision_pipeline.contacts()
    for _ in range(steps):
        state_in.clear_forces()
        collision_pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, DT)
        state_in, state_out = state_out, state_in
    wp.synchronize()
    return solver


def test_cached_path_detected(test, device):
    """Build a cached kernel per size group and only with the option on."""
    model = _build_model(device, chains=(6,), spheres=("base", "mid", "deep"))
    solver = _make_solver(model, "propagation", cached=True)
    test.assertIsNotNone(solver._propagation_cached_gemv_kernel)
    test.assertEqual(
        solver.propagation_cache_max_bodies,
        min(solver.max_propagation_bodies, solver.propagation_cached_response_max_bodies),
    )
    for size in solver._propagation_tree_sizes:
        test.assertIn(size, solver._propagation_cached_kernels)
    test.assertTrue(np.all(solver.propagation_cache_art_eligible.numpy() == 1))

    off = _make_solver(model, "propagation", cached=False)
    test.assertFalse(off._propagation_cached_kernels)
    test.assertIsNone(off._propagation_cached_gemv_kernel)


def test_lockstep_multi_contact_bodies(test, device):
    """Match the tree walk with three contact spheres per articulation, so cross blocks are live."""
    model = _build_model(device, chains=(6,), spheres=("base", "mid", "deep"))
    stats, _, cached = _run_lockstep(model, "propagation", 40)
    test.assertTrue(np.all(cached.propagation_cache_world_flag.numpy() == 1))
    _assert_stats(test, stats, expect_min_bodies=2)


def test_lockstep_second_size_group(test, device):
    """Match the tree walk with two articulation sizes per world, both cached."""
    model = _build_model(device, chains=(6, 3), spheres=("base", "deep"))
    stats, _, cached = _run_lockstep(model, "propagation", 40)
    sizes = cached._propagation_tree_sizes
    test.assertEqual(len(sizes), 2, f"expected two size groups, got {sizes}")
    for size in sizes:
        test.assertIn(size, cached._propagation_cached_kernels)
    _assert_stats(test, stats, expect_min_bodies=2)


def test_lockstep_single_contact(test, device):
    """Match the tree walk with one contact sphere per world."""
    model = _build_model(device, chains=(6,), spheres=("base",))
    stats, _, _ = _run_lockstep(model, "propagation", 40)
    _assert_stats(test, stats, expect_min_bodies=1)


def test_lockstep_fixed_base_single_dof(test, device):
    """Match the tree walk with fixed-base all-revolute chains, the single-DOF extraction."""
    model = _build_model(device, chains=(6,), spheres=("base", "mid", "deep"), fixed_base=True)
    stats, _, cached = _run_lockstep(model, "propagation", 40)
    for size in cached._propagation_tree_sizes:
        test.assertFalse(cached._propagation_free_root_tree[size])
        test.assertIn(size, cached._propagation_cached_kernels)
    test.assertTrue(np.all(cached.propagation_cache_world_flag.numpy() == 1))
    _assert_stats(test, stats, expect_min_bodies=2)


def test_overflow_falls_back_to_tree_walk(test, device):
    """Take the per-world tree walk when the active bodies exceed the cache capacity."""
    model = _build_model(device, chains=(6,), spheres=("base", "mid", "deep"))
    stats, _, cached = _run_lockstep(model, "propagation", 40, cache_max_bodies=2)
    test.assertEqual(cached.propagation_cache_max_bodies, 2)
    counts = cached.propagation_body_count.numpy()
    flags = cached.propagation_cache_world_flag.numpy()
    test.assertTrue(np.all(counts > 2), f"scene must overflow the cap, got counts {counts}")
    test.assertTrue(np.all(flags == 0), f"overflowing worlds must clear the cache flag, got {flags}")
    _assert_stats(test, stats, expect_min_bodies=3)


def test_clutter_does_not_evict_cache(test, device):
    """Count only the articulation's bodies against the cache capacity, not free clutter."""
    model = _build_clutter_model(device, n_boxes=2)
    stats, _, cached = _run_lockstep(model, "propagation", 40, cache_max_bodies=4, dense_max_constraints=128)
    counts = cached.propagation_body_count.numpy()
    test.assertTrue(np.all(counts > 4), f"scene must overflow the total active-body count, got {counts}")
    eligible_counts = cached.propagation_cache_body_count.numpy()
    test.assertTrue(np.all((eligible_counts >= 2) & (eligible_counts <= 4)), f"eligible counts {eligible_counts}")
    test.assertTrue(np.all(cached.propagation_cache_world_flag.numpy() == 1))
    # Eligible bodies occupy the prefix of each world's body list.
    art_eligible = cached.propagation_cache_art_eligible.numpy()
    body_to_art = cached.body_to_articulation.numpy()
    body_list = cached.propagation_body_list.numpy()
    for world in range(body_list.shape[0]):
        for slot in range(min(int(counts[world]), body_list.shape[1])):
            body = int(body_list[world, slot])
            test.assertGreaterEqual(body, 0)
            is_eligible = int(art_eligible[int(body_to_art[body])]) == 1
            test.assertEqual(is_eligible, slot < int(eligible_counts[world]), f"world {world} slot {slot}")
    _assert_stats(test, stats, expect_min_bodies=5)


def test_cached_matrices_match_numpy_reference(test, device, fixed_base=False):
    """Match R and B to a float64 mirror of the tree walk.

    R_b column j is the joint response to a unit wrench j at body b; B_ab column j is the twist of body a.
    """
    model = _build_model(device, chains=(6,), spheres=("base", "mid", "deep"), fixed_base=fixed_base)
    solver = _step_cached(model, 20)
    counts = solver.propagation_body_count.numpy()
    body_list = solver.propagation_body_list.numpy()
    body_to_art = solver.body_to_articulation.numpy()
    flags = solver.propagation_cache_world_flag.numpy()
    art_dof_start = model.joint_qd_start.numpy()[model.articulation_start.numpy()[:-1]]
    R = solver.propagation_cache_R.numpy()
    B = solver.propagation_cache_B.numpy()

    checked_pairs = 0
    for world in range(min(2, body_list.shape[0])):
        test.assertEqual(int(flags[world]), 1)
        n = int(counts[world])
        test.assertGreaterEqual(n, 2, "need at least two active bodies for cross blocks")
        for slot_b in range(n):
            body_b = int(body_list[world, slot_b])
            art = int(body_to_art[body_b])
            ti = _tree_inputs(model, solver, art)
            dof0 = int(art_dof_start[art])
            n_dofs = int(ti["qd_start"][ti["j1"]] - ti["qd_start"][ti["j0"]])
            for basis in range(6):
                imp = np.zeros(6)
                imp[basis] = 1.0
                v_ref, body_qd_ref = _ref_propagate(
                    ti, {body_b: imp}, np.zeros(solver.v_out.shape[0], dtype=np.float64)
                )
                got_R = R[world, slot_b, :n_dofs, basis]
                ref_R = v_ref[dof0 : dof0 + n_dofs]
                scale = max(1.0, float(np.max(np.abs(ref_R))))
                test.assertLess(float(np.max(np.abs(got_R - ref_R))), 1e-4 * scale, f"R {world} {slot_b} {basis}")
                for slot_a in range(n):
                    body_a = int(body_list[world, slot_a])
                    ref_col = body_qd_ref[body_a] if int(body_to_art[body_a]) == art else np.zeros(6)
                    got_col = B[world, slot_a, slot_b].reshape(6, 6)[:, basis]
                    scale = max(1.0, float(np.max(np.abs(ref_col))))
                    test.assertLess(
                        float(np.max(np.abs(got_col - ref_col))), 1e-4 * scale, f"B {world} {slot_a} {slot_b} {basis}"
                    )
                    checked_pairs += 1
    test.assertGreater(checked_pairs, 0)


def test_impulses_consumed(test, device):
    """Clear the deferred body impulses in the GEMV of every iteration."""
    model = _build_model(device, chains=(6,), spheres=("base", "mid", "deep"))
    solver = _step_cached(model, 10)
    test.assertEqual(float(np.max(np.abs(solver.propagation_body_impulses.numpy()))), 0.0)


def test_captured_cached_response_matches_eager(test, device):
    """Replay a captured step of the cached response, with a capacity overflow, with the eager result."""
    for cache_max_bodies in (8, 2):
        finals = []
        for capture in (False, True):
            model = _build_model(device, chains=(6,), spheres=("base", "mid", "deep"))
            solver = _make_solver(model, "propagation", cached=True, cache_max_bodies=cache_max_bodies)
            state_0, state_1 = model.state(), model.state()
            state_0.joint_qd.assign(_seed_joint_qd(model))
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            pipeline = newton.CollisionPipeline(model, deterministic=True)
            contacts = pipeline.contacts()
            control = model.control()

            def substep(
                state_0=state_0, state_1=state_1, solver=solver, pipeline=pipeline, contacts=contacts, control=control
            ):
                pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, DT)
                for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                    wp.copy(getattr(state_0, name), getattr(state_1, name))

            substep()
            if capture:
                with wp.ScopedCapture(device) as scope:
                    substep()
                for _ in range(5):
                    wp.capture_launch(scope.graph)
            else:
                for _ in range(5):
                    substep()
            finals.append(state_0.joint_qd.numpy())
        np.testing.assert_allclose(finals[1], finals[0], rtol=0.0, atol=1.0e-5)


class TestFeatherPGSPropagationCachedResponse(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _fn in (
    test_cached_path_detected,
    test_lockstep_multi_contact_bodies,
    test_lockstep_second_size_group,
    test_lockstep_single_contact,
    test_lockstep_fixed_base_single_dof,
    test_overflow_falls_back_to_tree_walk,
    test_clutter_does_not_evict_cache,
    test_cached_matrices_match_numpy_reference,
    test_impulses_consumed,
    test_captured_cached_response_matches_eager,
):
    add_function_test(TestFeatherPGSPropagationCachedResponse, _fn.__name__, _fn, devices=devices)
add_function_test(
    TestFeatherPGSPropagationCachedResponse,
    "test_fixed_base_cached_matrices_match_numpy_reference",
    test_cached_matrices_match_numpy_reference,
    devices=devices,
    fixed_base=True,
)


if __name__ == "__main__":
    unittest.main()
