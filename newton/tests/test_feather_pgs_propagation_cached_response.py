# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Numerical equivalence of the cached-response propagation path vs the tree walk.

``propagation_cached_response=True`` replaces the per-GS-iteration propagation
tree walk (backward+forward Featherstone sweep) with cached per-body response
matrices applied as a small GEMV. Within one solver pass the tree
factorization is fixed, so deferred body impulse -> joint velocity is a fixed
linear map and superposition over precomputed response columns is the SAME
math — results may differ from the tree walk only by float reassociation.
These tests gate that claim:

1. Kernel-level: the extracted response matrices R (D x 6 joint-space
   response per active body) and B (6 x 6 active-pair body response blocks)
   against a float64 numpy mirror of the serial tree-walk algorithm.
2. Lockstep stepping: cached and tree-walk solvers advance one step from the
   IDENTICAL input state at every point of a contact-rich trajectory and
   their outputs (joint_q/joint_qd/v_out/active-body twists) are compared
   per step. Free-running trajectory comparison is useless here: bouncing
   multi-contact scenes are chaotic enough that even the existing
   warp-vs-serial tree-walk pair diverges to O(0.1) within 20 steps from
   float reassociation alone (measured on RTX 5080), so a per-step gate from
   shared inputs is the strictly stronger test. A tree-walk determinism
   precondition (two identical fallback solvers must agree bitwise) keeps
   the gate honest.
3. Scenes: a multi-contact-body chain (three collision spheres per
   articulation, so cross blocks B_ab with a != b carry real impulses), a
   two-articulations-per-world scene producing a second articulation
   DOF-size group (per-world DOF totals stay homogeneous, which the
   propagation modes require), the serial "propagation" mode, and a
   cache-capacity overflow that must take the per-world device-side
   tree-walk fallback.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_propagation_free_root_warp import (
    _ref_propagate,
    _tree_inputs,
)

N_WORLDS = 4
DT = 1.0 / 240.0


def _add_floating_chain(
    builder: newton.ModelBuilder, *, n_links: int, base_z: float, base_y: float, spheres: tuple[str, ...]
) -> None:
    """A floating box base (free joint) trailing ``n_links`` revolute links.

    ``spheres`` selects which bodies carry a collision sphere ("base", "mid",
    "deep"); all other shapes are density-only. Multiple spheres per world
    give several simultaneously contact-active bodies on one articulation,
    which is what exercises the cached cross response blocks B_ab.
    """
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
    """A world-anchored all-revolute chain: the fixed-base extraction variant.

    No free root — the first joint is a world-rooted revolute, so every joint
    is 1-DOF and the articulation takes the cached-response extraction's
    fixed-base (single-DOF) variant, the default shape of a fixed-base arm.
    ``spheres`` selects collision spheres on the first ("base"), middle
    ("mid"), and last ("deep") link; the chain droops under gravity onto the
    ground plane, so several bodies are simultaneously contact-active and the
    cross response blocks B_ab carry real impulses.
    """
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
    """N_WORLDS identical worlds, each holding one chain per entry of ``chains``.

    Multiple chain lengths inside one world create several articulation
    DOF-size groups while keeping per-world DOF totals homogeneous (the
    propagation modes reject heterogeneous per-world DOF counts).
    ``fixed_base`` swaps the floating chains for world-anchored ones.
    """
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
    """A fixed-base arm plus free-rigid clutter boxes resting on the ground.

    The clutter bodies are contact-active every step but never take the
    cached GEMV path (free bodies keep the sweep-estimate + flush path), so
    they must not count against the cache capacity: only the arm's bodies
    are cache-eligible. Clutter is frictionless spheres: one normal-only
    ground row per clutter body keeps every clutter row's state fully
    disjoint, so the lockstep gate's tree-walk determinism precondition
    survives the atomic (run-varying) row ordering — box corner contacts or
    friction pairs share a body and reorder its float accumulation between
    runs.
    """
    scene = newton.ModelBuilder()
    for _ in range(N_WORLDS):
        world = newton.ModelBuilder()
        world.default_shape_cfg.density = 1000.0
        world.default_shape_cfg.mu = 0.7
        # Anchor low enough that every arm sphere starts pressed into the
        # ground: a marginal (flickering) tip contact makes the active-body
        # set differ between the lockstepped solvers.
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


def _make_solver(
    model: newton.Model, response: str, *, cached: bool, cache_max_bodies: int = 8, **solver_kwargs
) -> SolverFeatherPGS:
    return SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        articulated_contact_response=response,
        pgs_iterations=8,
        pgs_warmstart=False,
        mf_warmstart=False,
        propagation_cached_response=cached,
        propagation_cached_response_max_bodies=cache_max_bodies,
        **solver_kwargs,
    )


def _seed_joint_qd(model: newton.Model) -> np.ndarray:
    n = int(model.joint_dof_count)
    return (0.4 * np.sin(0.7 * np.arange(n, dtype=np.float64) + 0.3)).astype(np.float32)


def _active_body_qd(solver) -> dict[int, np.ndarray]:
    """Live COM twists keyed by body, restricted to contact-active bodies.

    The cached path deliberately leaves NON-active bodies' propagation_body_qd
    stale between forced refreshes (the GS sweep only ever reads active
    bodies, and every pass setup force-refreshes all bodies from v_out), so
    only active entries are comparable across paths.
    """
    qd = solver.propagation_body_qd.numpy()
    counts = solver.propagation_body_count.numpy()
    body_list = solver.propagation_body_list.numpy()
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


def _run_lockstep(test_case, model, response, n_steps, *, cache_max_bodies: int = 8, **solver_kwargs):
    """Advance a tree-walk reference trajectory; at every step, run the cached
    solver one step from the identical input state and record output diffs.

    A second tree-walk solver steps alongside as a bitwise determinism
    precondition: if the fallback itself were not reproducible from shared
    inputs, the cached-vs-fallback diff would be meaningless.
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
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()

    stats = _LockstepStats()
    for _ in range(n_steps):
        state_in.clear_forces()
        # A fresh collide before each solver isolates them from any contact
        # buffer bookkeeping a step might do; collide is deterministic in
        # state, so all three consume identical contact sets.
        collision_pipeline.collide(state_in, contacts)
        test.step(state_in, test_out, control, contacts, DT)
        collision_pipeline.collide(state_in, contacts)
        ref2.step(state_in, ref2_out, control, contacts, DT)
        collision_pipeline.collide(state_in, contacts)
        ref.step(state_in, ref_out, control, contacts, DT)
        wp.synchronize()

        # determinism precondition
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
        ref_body_qd = _active_body_qd(ref)
        test_body_qd = _active_body_qd(test)
        test_case.assertEqual(sorted(ref_body_qd), sorted(test_body_qd), "active body sets differ")
        for body, ref_qd in ref_body_qd.items():
            diff = float(np.max(np.abs(test_body_qd[body] - ref_qd)))
            stats.max_diffs["active_body_qd"] = max(stats.max_diffs["active_body_qd"], diff)

        counts = ref.propagation_body_count.numpy()
        stats.max_active_bodies = max(stats.max_active_bodies, int(counts.max()))
        stats.total_rows += int(ref.propagation_constraint_count.numpy().sum())

        # advance the shared trajectory along the reference solve
        state_in, ref_out = ref_out, state_in
    return stats, ref, test


@unittest.skipUnless(wp.get_cuda_device_count() > 0, "requires CUDA")
class TestPropagationCachedResponse(unittest.TestCase):
    TOL = 1.0e-4

    @classmethod
    def setUpClass(cls):
        cls.device = "cuda:0"
        cls.model_multi = _build_model(cls.device, chains=(6,), spheres=("base", "mid", "deep"))
        cls.model_two_groups = _build_model(cls.device, chains=(6, 3), spheres=("base", "deep"))
        cls.model_single = _build_model(cls.device, chains=(6,), spheres=("base",))
        cls.model_fixed = _build_model(cls.device, chains=(6,), spheres=("base", "mid", "deep"), fixed_base=True)

    def test_cached_path_detected(self):
        solver = _make_solver(self.model_multi, "propagation-colored", cached=True)
        self.assertTrue(solver._propagation_cached_response_active)
        self.assertIsNotNone(solver._propagation_cached_gemv_kernel)
        self.assertIsNotNone(solver.propagation_cache_R)
        self.assertIsNotNone(solver.propagation_cache_B)
        self.assertIsNotNone(solver.propagation_cache_qd_base)
        self.assertIsNotNone(solver.propagation_cache_world_flag)
        self.assertEqual(
            solver.propagation_cache_max_bodies,
            min(solver.max_propagation_bodies, solver.propagation_cached_response_max_bodies),
        )
        sizes = [int(s) for s in solver.size_groups if solver.n_arts_by_size[s] > 0]
        for size in sizes:
            self.assertIsNotNone(
                solver._cached_response_tree_warp_kernels_by_size.get(size),
                f"no cached-response kernel for size group {size}",
            )
        # eligibility must exclude nothing here: all arts are free-root trees
        self.assertTrue(np.all(solver.propagation_cache_art_eligible.numpy() == 1))

        off = _make_solver(self.model_multi, "propagation-colored", cached=False)
        self.assertFalse(off._propagation_cached_response_active)
        self.assertIsNone(off.propagation_cache_R)

    def _assert_stats(self, stats, *, expect_min_bodies):
        self.assertGreater(stats.total_rows, 0, "no propagation contact rows produced")
        self.assertGreaterEqual(
            stats.max_active_bodies,
            expect_min_bodies,
            "scene did not produce the expected number of simultaneously active bodies",
        )
        for key, diff in stats.max_diffs.items():
            self.assertLessEqual(diff, self.TOL, f"{key} per-step max abs diff {diff:.3e} exceeds {self.TOL:g}")

    def test_lockstep_multi_contact_bodies(self):
        """Three contact spheres per articulation: cross blocks B_ab are live."""
        stats, _, test = _run_lockstep(self, self.model_multi, "propagation-colored", 40)
        self.assertTrue(np.all(test.propagation_cache_world_flag.numpy() == 1))
        self._assert_stats(stats, expect_min_bodies=2)

    def test_lockstep_second_size_group(self):
        """Two articulations per world -> two DOF-size groups, both cached."""
        stats, _, test = _run_lockstep(self, self.model_two_groups, "propagation-colored", 40)
        sizes = [int(s) for s in test.size_groups if test.n_arts_by_size[s] > 0]
        self.assertEqual(len(sizes), 2, f"expected two size groups, got {sizes}")
        for size in sizes:
            self.assertIsNotNone(test._cached_response_tree_warp_kernels_by_size.get(size))
        self._assert_stats(stats, expect_min_bodies=2)

    def test_lockstep_serial_propagation_mode(self):
        """Non-colored 'propagation' mode; one contact per world keeps the
        serial row sweep deterministic (same constraint as the free-root
        warp tests)."""
        stats, _, _ = _run_lockstep(self, self.model_single, "propagation", 40)
        self._assert_stats(stats, expect_min_bodies=1)

    def test_lockstep_fixed_base_single_dof(self):
        """Fixed-base all-revolute chains: the single-DOF extraction variant.

        Every prior lockstep scene is floating-base, which takes the
        free-root extraction variant; a fixed-base arm — the default shape of
        every table-mounted robot — takes the other one, so gate it
        explicitly against the tree walk on the same contact-rich scene.
        """
        stats, _, test = _run_lockstep(self, self.model_fixed, "propagation-colored", 40)
        sizes = [int(s) for s in test.size_groups if test.n_arts_by_size[s] > 0]
        for size in sizes:
            # A fully single-DOF group takes the plain fixed-base extraction
            # variant (the kernel factories receive has_free_root=not
            # single_dof), even though a 1-DOF world-rooted chain also
            # satisfies the free-root SHAPE check.
            self.assertTrue(
                test._propagation_tree_single_dof_by_size.get(size, False),
                f"size group {size} did not take the fixed-base single-DOF variant",
            )
            self.assertIsNotNone(test._cached_response_tree_warp_kernels_by_size.get(size))
        self.assertTrue(np.all(test.propagation_cache_world_flag.numpy() == 1))
        self._assert_stats(stats, expect_min_bodies=2)

    def test_overflow_falls_back_to_tree_walk(self):
        """Cache capacity below the active-body count: every world must take
        the device-side tree-walk fallback and match the cached=False run."""
        stats, ref, test = _run_lockstep(self, self.model_multi, "propagation-colored", 40, cache_max_bodies=2)
        self.assertTrue(test._propagation_cached_response_active)
        self.assertEqual(test.propagation_cache_max_bodies, 2)
        flags = test.propagation_cache_world_flag.numpy()
        counts = ref.propagation_body_count.numpy()
        self.assertTrue(np.all(counts > 2), f"scene must overflow the cap, got counts {counts}")
        self.assertTrue(np.all(flags == 0), f"overflowing worlds must clear the cache flag, got {flags}")
        self._assert_stats(stats, expect_min_bodies=3)

    def test_clutter_does_not_evict_cache(self):
        """Free-rigid clutter must not count against the cache capacity.

        A fixed-base arm (three contact-active tree bodies) plus two free
        spheres on the ground against a cache capacity of 4: the TOTAL active
        count (5) exceeds the cap, but only the arm's bodies are
        cache-eligible, so every world must stay on the cached GEMV path —
        and still match the tree walk in lockstep. Before the partitioned
        capacity gate this scene silently self-disabled the cache. Clutter is
        kept small because larger free-body counts make the tree-walk
        REFERENCE itself non-reproducible run-to-run on this branch
        (reassociation-level, and near the default 32-row cap much larger),
        which invalidates the lockstep gate's determinism precondition.
        """
        model = _build_clutter_model(self.device, n_boxes=2)
        stats, ref, test = _run_lockstep(
            self, model, "propagation-colored", 40, cache_max_bodies=4, dense_max_constraints=128
        )
        self.assertTrue(test._propagation_cached_response_active)
        counts = ref.propagation_body_count.numpy()
        self.assertTrue(np.all(counts > 4), f"scene must overflow the TOTAL active-body count, got {counts}")
        eligible_counts = test.propagation_cache_body_count.numpy()
        self.assertTrue(
            np.all((eligible_counts >= 2) & (eligible_counts <= 4)),
            f"expected 2..4 eligible arm bodies per world, got {eligible_counts}",
        )
        self.assertTrue(
            np.all(test.propagation_cache_world_flag.numpy() == 1),
            "clutter must not evict the arm from the cached path",
        )
        # Partition invariant: cache-eligible bodies occupy the slot prefix.
        art_eligible = test.propagation_cache_art_eligible.numpy()
        body_to_art = test.body_to_articulation.numpy()
        body_list = test.propagation_body_list.numpy()
        test_counts = test.propagation_body_count.numpy()
        for world in range(body_list.shape[0]):
            n_elig = int(eligible_counts[world])
            for slot in range(min(int(test_counts[world]), body_list.shape[1])):
                body = int(body_list[world, slot])
                self.assertGreaterEqual(body, 0)
                is_eligible = int(art_eligible[int(body_to_art[body])]) == 1
                self.assertEqual(is_eligible, slot < n_elig, f"world {world} slot {slot}: eligible prefix violated")
        self._assert_stats(stats, expect_min_bodies=5)

    def test_cached_matrices_match_numpy_reference(self):
        """Floating-base R/B against the float64 tree-walk mirror."""
        self._check_cached_matrices_vs_numpy(self.model_multi)

    def test_fixed_base_cached_matrices_match_numpy_reference(self):
        """Fixed-base (single-DOF variant) R/B against the same mirror."""
        self._check_cached_matrices_vs_numpy(self.model_fixed)

    def _check_cached_matrices_vs_numpy(self, model):
        """R and B against a float64 mirror of the serial tree-walk algorithm.

        R_b column j is the qdd over all articulation DOFs from a unit basis
        wrench j at body b; B_ab column j is the resulting COM twist delta at
        body a. Both fall out of _ref_propagate run at zero initial velocity.
        """
        solver = _make_solver(model, "propagation-colored", cached=True)
        state_in, state_out = model.state(), model.state()
        control = model.control()
        state_in.joint_qd.assign(_seed_joint_qd(model))
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        for _ in range(20):
            state_in.clear_forces()
            collision_pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, DT)
            state_in, state_out = state_out, state_in
        wp.synchronize()

        counts = solver.propagation_body_count.numpy()
        body_list = solver.propagation_body_list.numpy()
        body_to_art = solver.body_to_articulation.numpy()
        flags = solver.propagation_cache_world_flag.numpy()
        art_dof_start = solver.model.joint_qd_start.numpy()[solver.model.articulation_start.numpy()[:-1]]
        R = solver.propagation_cache_R.numpy()
        B = solver.propagation_cache_B.numpy()

        checked_pairs = 0
        for world in range(min(2, body_list.shape[0])):
            self.assertEqual(int(flags[world]), 1)
            n = int(counts[world])
            self.assertGreaterEqual(n, 2, "need at least two active bodies for cross blocks")
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
                    self.assertLess(
                        float(np.max(np.abs(got_R - ref_R))),
                        1e-4 * scale,
                        f"R mismatch world {world} slot {slot_b} basis {basis}",
                    )
                    for slot_a in range(n):
                        body_a = int(body_list[world, slot_a])
                        if int(body_to_art[body_a]) == art:
                            ref_col = body_qd_ref[body_a]
                        else:
                            ref_col = np.zeros(6)
                        got_col = B[world, slot_a, slot_b].reshape(6, 6)[:, basis]
                        scale = max(1.0, float(np.max(np.abs(ref_col))))
                        self.assertLess(
                            float(np.max(np.abs(got_col - ref_col))),
                            1e-4 * scale,
                            f"B mismatch world {world} a {slot_a} b {slot_b} basis {basis}",
                        )
                        checked_pairs += 1
        self.assertGreater(checked_pairs, 0)

    def test_impulses_consumed(self):
        """Deferred body impulses must be cleared by the GEMV each iteration."""
        model = self.model_multi
        solver = _make_solver(model, "propagation-colored", cached=True)
        state_in, state_out = model.state(), model.state()
        control = model.control()
        state_in.joint_qd.assign(_seed_joint_qd(model))
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        for _ in range(10):
            state_in.clear_forces()
            collision_pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, DT)
            state_in, state_out = state_out, state_in
        wp.synchronize()
        self.assertEqual(float(np.max(np.abs(solver.propagation_body_impulses.numpy()))), 0.0)


if __name__ == "__main__":
    unittest.main()
