# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
import warnings
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import kernels as feather_pgs_kernels
from newton._src.solvers.feather_pgs.solver_feather_pgs import (
    _DENSE_META_MAX_PARENT,
    _DENSE_META_ROW_TYPE_MASK,
    _estimate_cholesky_shared_memory,
    _FeatherPGSExecutionPlan,
    _select_hinv_jt_chunk_size,
    _use_resident_mfgs_metadata,
    _validate_dense_metadata_encoding,
)
from newton.solvers import SolverFeatherPGS


def _build_chain_model(num_links=3, num_worlds=2, *, with_free_body=False):
    chain = newton.ModelBuilder()
    hx = 0.3
    joints = []
    parent = -1
    root_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.45 * wp.pi)
    for _ in range(num_links):
        link = chain.add_link()
        chain.add_shape_box(link, hx=hx - 0.08, hy=0.05, hz=0.05)
        if parent == -1:
            parent_xform = wp.transform(p=wp.vec3(0.0, 0.0, 2.5), q=root_rot)
        else:
            parent_xform = wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity())
        joints.append(
            chain.add_joint_revolute(
                parent=parent,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=parent_xform,
                child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
            )
        )
        parent = link
    chain.add_articulation(joints)
    if with_free_body:
        body = chain.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        chain.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        chain.add_articulation([chain.add_joint_free(parent=-1, child=body)])
    main = newton.ModelBuilder()
    main.replicate(chain, num_worlds, spacing=(3.0, 3.0, 0.0))
    return main.finalize()


def _build_fixed_base_star_model(num_branches=16, num_worlds=2):
    star = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    base = star.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joints = [star.add_joint_fixed(parent=-1, child=base)]
    for branch in range(num_branches):
        child = star.add_link(mass=1.0 + 0.01 * branch, inertia=wp.mat33(np.eye(3)))
        joints.append(
            star.add_joint_prismatic(
                parent=base,
                child=child,
                axis=newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(0.02 * branch, 0.0, 0.0), wp.quat_identity()),
            )
        )
    star.add_articulation(joints)
    main = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    main.replicate(star, num_worlds, spacing=(2.0, 2.0, 0.0))
    return main.finalize()


def _build_sparse_diagonal_pair_model(num_branches=16, num_worlds=2, *, device=None, revolute_branches=False):
    """Build one independent fixed-base star and one compact serial chain per world.

    ``revolute_branches`` hinges the star branches about Y instead of sliding them along Z, so the branch
    response depends on the link inertia and center of mass rather than on the mass alone.
    """
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    base = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    star_joints = [scene.add_joint_fixed(parent=-1, child=base)]
    add_branch_joint = scene.add_joint_revolute if revolute_branches else scene.add_joint_prismatic
    for branch in range(num_branches):
        child = scene.add_link(mass=1.0 + 0.01 * branch, inertia=wp.mat33(np.eye(3)))
        star_joints.append(
            add_branch_joint(
                parent=base,
                child=child,
                axis=newton.Axis.Y if revolute_branches else newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(0.03 * branch, 0.0, 0.0), wp.quat_identity()),
                limit_lower=-0.1,
                limit_upper=0.1,
            )
        )
    scene.add_articulation(star_joints)

    chain_joints = []
    parent = -1
    for _link_index in range(3):
        child = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        chain_joints.append(
            scene.add_joint_revolute(
                parent=parent,
                child=child,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
                limit_lower=-0.2,
                limit_upper=0.2,
            )
        )
        parent = child
    scene.add_articulation(chain_joints)

    replicated = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    replicated.replicate(scene, num_worlds, spacing=(3.0, 3.0, 0.0))
    return replicated.finalize(device=device)


def _build_sparse_contact_friction_model(num_branches=16, num_worlds=2, *, device=None):
    """Build the sparse pair model with tilted sliding branches resting on a static ground plane.

    Each branch slides along ``(1, 0, 1) / sqrt(2)`` and carries a small box that touches the ground, so every
    contact normal and tangent row acts on that branch's single coordinate and friction is active.
    """
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    scene.default_shape_cfg.ke = 1.0e5
    scene.default_shape_cfg.kd = 1.0e3
    scene.default_shape_cfg.mu = 0.8
    scene.default_shape_cfg.margin = 0.0
    scene.default_shape_cfg.gap = 0.0
    scene.add_ground_plane()
    base = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    star_joints = [
        scene.add_joint_fixed(
            parent=-1, child=base, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity())
        )
    ]
    for branch in range(num_branches):
        child = scene.add_link(mass=1.0 + 0.01 * branch, inertia=wp.mat33(np.eye(3)))
        # Four boxes keep the four-point patches within the dense row capacity of the test solvers.
        if branch % 4 == 0:
            scene.add_shape_box(child, hx=0.01, hy=0.01, hz=0.01)
        star_joints.append(
            scene.add_joint_prismatic(
                parent=base,
                child=child,
                axis=wp.normalize(wp.vec3(1.0, 0.0, 1.0)),
                parent_xform=wp.transform(wp.vec3(0.05 * branch, 0.0, -0.4905), wp.quat_identity()),
                limit_lower=-0.1,
                limit_upper=0.1,
            )
        )
    scene.add_articulation(star_joints)

    chain_joints = []
    parent = -1
    for _link_index in range(3):
        child = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        chain_joints.append(
            scene.add_joint_revolute(
                parent=parent,
                child=child,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(wp.vec3(0.0, 1.0, 1.0), wp.quat_identity()),
                limit_lower=-0.2,
                limit_upper=0.2,
            )
        )
        parent = child
    scene.add_articulation(chain_joints)

    replicated = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    replicated.replicate(scene, num_worlds, spacing=(3.0, 3.0, 0.0))
    return replicated.finalize(device=device)


def _build_heterogeneous_world_model():
    free_template = newton.ModelBuilder()
    free_body = free_template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    free_joint = free_template.add_joint_free(parent=-1, child=free_body)
    free_template.add_articulation([free_joint])

    slider_template = newton.ModelBuilder()
    slider_body = slider_template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    slider_joint = slider_template.add_joint_prismatic(parent=-1, child=slider_body, axis=newton.Axis.X)
    slider_template.add_articulation([slider_joint])

    builder = newton.ModelBuilder()
    builder.add_world(free_template)
    builder.add_world(slider_template)
    return builder.finalize()


class TestFeatherPGSLaunchConfig(unittest.TestCase):
    def setUp(self):
        # Launch validation selects point-contact solvers without choosing a friction law.
        filters = warnings.catch_warnings()
        filters.__enter__()
        self.addCleanup(filters.__exit__, None, None, None)
        warnings.filterwarnings(
            "ignore", message=r"The selected point-contact solver uses velocity-only friction", category=UserWarning
        )

    def test_launch_geometry_kernels_use_dedicated_modules(self):
        """Keep custom-block-dimension kernels out of the general module."""
        expected_modules = {
            "update_articulation_origins": "kinematics",
            "eval_rigid_fk": "kinematics",
            "eval_rigid_id": "kinematics",
            "eval_rigid_tau": "inverse_dynamics",
            "compute_composite_inertia": "mass_dynamics",
            "crba_fill_par_dof": "mass_dynamics",
        }
        general_module = feather_pgs_kernels.__name__
        for kernel_name, module_suffix in expected_modules.items():
            with self.subTest(kernel=kernel_name):
                kernel_module = getattr(feather_pgs_kernels, kernel_name).module.name
                self.assertNotEqual(kernel_module, general_module)
                self.assertEqual(kernel_module, f"{general_module}.{module_suffix}")

    def test_defaults_preserved(self):
        model = newton.ModelBuilder().finalize()
        solver = SolverFeatherPGS(model)
        self.assertEqual(solver.serial_kernel_block_dim, 256)
        self.assertEqual(solver.tile_threads, 64)
        self.assertEqual(solver.articulated_contact_response, "immediate")

    @unittest.skipUnless(wp.is_cuda_available(), "matrix-free contact responses require CUDA")
    def test_articulated_contact_response_validation(self):
        model = _build_chain_model(num_links=2, num_worlds=1)
        solver = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            articulated_contact_response="propagation",
        )
        self.assertEqual(solver.articulated_contact_response, "propagation")
        fused = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            articulated_contact_response="propagation-fused",
        )
        self.assertEqual(fused.articulated_contact_response, "propagation-fused")

        with self.assertRaisesRegex(ValueError, "articulated_contact_response"):
            SolverFeatherPGS(model, articulated_contact_response="bad")

        with self.assertRaisesRegex(NotImplementedError, "pgs_mode='matrix_free'"):
            SolverFeatherPGS(model, pgs_mode="split", articulated_contact_response="propagation")
        with self.assertRaisesRegex(NotImplementedError, "pgs_mode='matrix_free'"):
            SolverFeatherPGS(model, pgs_mode="split", articulated_contact_response="propagation-fused")

        with self.assertRaisesRegex(NotImplementedError, "friction_mode='current'"):
            SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                articulated_contact_response="propagation",
                friction_mode="bisection",
            )
        propagation_debug = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            articulated_contact_response="propagation",
            pgs_debug=True,
        )
        self.assertEqual(propagation_debug.articulated_contact_response, "propagation")

    def test_joint_limit_solver_compatibility_validation(self):
        model = _build_chain_model(num_links=2, num_worlds=1)

        with self.assertRaisesRegex(NotImplementedError, "requires pgs_mode='matrix_free'"):
            SolverFeatherPGS(
                model,
                pgs_mode="split",
                enable_joint_velocity_limits=True,
            )

        for pgs_kernel in ("tiled_contact", "streaming"):
            with self.subTest(pgs_kernel=pgs_kernel):
                with self.assertRaisesRegex(ValueError, "contact-only"):
                    SolverFeatherPGS(
                        model,
                        pgs_mode="split",
                        pgs_kernel=pgs_kernel,
                        enable_joint_limits=True,
                    )

    def test_default_kwargs_produce_identical_kernel_selection(self):
        model = _build_chain_model()
        implicit = SolverFeatherPGS(model)
        explicit = SolverFeatherPGS(model, serial_kernel_block_dim=256, tile_threads=64)
        for attr in (
            "_cholesky_kernels_by_size",
            "_triangular_solve_kernels_by_size",
            "_hinv_jt_kernels_by_size",
            "_hinv_jt_fused_kernels_by_size",
        ):
            implicit_kernels = getattr(implicit, attr)
            explicit_kernels = getattr(explicit, attr)
            self.assertEqual(set(implicit_kernels), set(explicit_kernels))
            for size, kernel in implicit_kernels.items():
                # The factories are functools.cache'd, so identical knob values
                # must resolve to the *same* kernel objects as the defaults.
                self.assertIs(kernel, explicit_kernels[size], f"{attr}[{size}]")

    def test_non_default_tile_threads_selects_distinct_kernels(self):
        model = _build_chain_model()
        default = SolverFeatherPGS(model)
        wide = SolverFeatherPGS(model, tile_threads=128)
        for size, kernel in default._cholesky_kernels_by_size.items():
            if kernel is None:
                continue
            other = wide._cholesky_kernels_by_size[size]
            self.assertIsNot(kernel, other)
            self.assertIn("_bd64", kernel.key)
            self.assertIn("_bd128", other.key)

    @unittest.skipUnless(wp.is_cuda_available(), "compact response mapping requires CUDA matrix-free mode")
    def test_compact_world_dof_mapping_pads_heterogeneous_worlds(self):
        solver = SolverFeatherPGS(_build_heterogeneous_world_model(), pgs_mode="matrix_free")
        self.assertEqual(solver.max_world_dofs, 6)
        np.testing.assert_array_equal(solver.world_dof_count.numpy(), np.array((6, 1), dtype=np.int32))
        indices = solver.world_dof_indices.numpy()
        np.testing.assert_array_equal(indices[0], np.arange(6, dtype=np.int32))
        self.assertGreaterEqual(int(indices[1, 0]), 0)
        np.testing.assert_array_equal(indices[1, 1:], np.full(5, -1, dtype=np.int32))

    def test_hinv_chunk_selection_respects_shared_memory(self):
        """Select the largest response chunk that fits shared memory."""
        cases = (
            (23, 384, 101376, 32),
            (128, 384, 101376, 16),
            (160, 384, 101376, None),
            (23, 0, 101376, None),
            (23, 384, 8192, 16),
            (23, 384, 2500, None),
            (100, 1, 49152, 1),
        )
        for n_dofs, max_constraints, shared_memory, expected in cases:
            with self.subTest(n_dofs=n_dofs, max_constraints=max_constraints, shared_memory=shared_memory):
                self.assertEqual(_select_hinv_jt_chunk_size(n_dofs, max_constraints, shared_memory, 64), expected)

    def test_hinv_chunk_selection_accounts_for_tile_threads(self):
        """Include tile scratch space when selecting response chunks."""
        self.assertEqual(_select_hinv_jt_chunk_size(50, 384, 30000, 64), 32)
        self.assertEqual(_select_hinv_jt_chunk_size(50, 384, 30000, 256), 16)

    def test_hinv_chunk_selection_caps_response_tiles(self):
        """Cap response chunks at 32 rows across articulation sizes."""
        self.assertEqual(_select_hinv_jt_chunk_size(20, 384, 101376, 64), 32)
        self.assertEqual(_select_hinv_jt_chunk_size(21, 384, 101376, 64), 32)

    def test_dense_metadata_encoding_bounds(self):
        _validate_dense_metadata_encoding(32)
        self.assertGreaterEqual(_DENSE_META_ROW_TYPE_MASK + 1, 5)
        with self.assertRaisesRegex(ValueError, "packed parent capacity"):
            _validate_dense_metadata_encoding(_DENSE_META_MAX_PARENT + 2)

    def test_hinv_execution_plan_falls_back_or_rejects_safely(self):
        fallback = _FeatherPGSExecutionPlan.build(
            [160],
            max_constraints=384,
            max_shared_memory=101376,
            cholesky_kernel="auto",
            hinv_jt_kernel="auto",
            small_dof_threshold=12,
            tile_threads=64,
        )
        self.assertFalse(fallback.use_tiled_hinv_jt(160))

        zero_capacity = _FeatherPGSExecutionPlan.build(
            [23],
            max_constraints=0,
            max_shared_memory=101376,
            cholesky_kernel="auto",
            hinv_jt_kernel="tiled",
            small_dof_threshold=12,
            tile_threads=64,
        )
        self.assertFalse(zero_capacity.use_tiled_hinv_jt(23))

        with self.assertRaisesRegex(ValueError, "hinv_jt_kernel='tiled'"):
            _FeatherPGSExecutionPlan.build(
                [160],
                max_constraints=384,
                max_shared_memory=101376,
                cholesky_kernel="auto",
                hinv_jt_kernel="tiled",
                small_dof_threshold=12,
                tile_threads=64,
            )

    def test_hinv_fusion_requires_full_working_set_to_fit(self):
        fitting = _FeatherPGSExecutionPlan.build(
            [23],
            max_constraints=64,
            max_shared_memory=101376,
            cholesky_kernel="auto",
            hinv_jt_kernel="auto",
            small_dof_threshold=12,
            tile_threads=64,
        )
        oversized = _FeatherPGSExecutionPlan.build(
            [23],
            max_constraints=384,
            max_shared_memory=101376,
            cholesky_kernel="auto",
            hinv_jt_kernel="auto",
            small_dof_threshold=12,
            tile_threads=64,
        )
        self.assertTrue(fitting.use_fused_hinv_jt(23))
        self.assertFalse(oversized.use_fused_hinv_jt(23))

    def test_cholesky_execution_plan_respects_shared_memory(self):
        """Fall back for oversized auto tiles and reject forced invalid launches."""
        self.assertEqual(_estimate_cholesky_shared_memory(108), 140400)
        automatic = _FeatherPGSExecutionPlan.build(
            [23, 108],
            max_constraints=0,
            max_shared_memory=101376,
            cholesky_kernel="auto",
            hinv_jt_kernel="auto",
            small_dof_threshold=12,
            tile_threads=64,
        )
        self.assertTrue(automatic.use_tiled_cholesky(23))
        self.assertFalse(automatic.use_tiled_cholesky(108))

        with self.assertRaisesRegex(ValueError, "cholesky_kernel='tiled'.*140400.*101376"):
            _FeatherPGSExecutionPlan.build(
                [108],
                max_constraints=0,
                max_shared_memory=101376,
                cholesky_kernel="tiled",
                hinv_jt_kernel="auto",
                small_dof_threshold=12,
                tile_threads=64,
            )

    def test_fixed_base_sibling_branches_detect_diagonal_mass_topology(self):
        """Detect independent fixed-base branches on every supported device."""
        solver = SolverFeatherPGS(_build_fixed_base_star_model(), enable_joint_limits=False, dense_max_constraints=16)
        self.assertEqual(solver._diagonal_mass_sizes, frozenset((16,)))

    @unittest.skipUnless(wp.is_cuda_available(), "diagonal mass execution-path parity requires CUDA")
    def test_fixed_base_sibling_branches_use_diagonal_mass_path(self):
        """Match the dense reference while selecting independent branch solves."""
        model = _build_fixed_base_star_model()
        optimized = SolverFeatherPGS(model, enable_joint_limits=False, dense_max_constraints=16)
        self.assertEqual(optimized._diagonal_mass_sizes, frozenset((16,)))
        self.assertTrue(optimized._execution_plan.use_diagonal_mass(16))
        self.assertIsNone(optimized._cholesky_kernels_by_size[16])
        self.assertIsNone(optimized._triangular_solve_kernels_by_size[16])
        self.assertIsNone(optimized._hinv_jt_kernels_by_size[16])

        forced = {"cholesky_kernel": "loop", "trisolve_kernel": "loop", "hinv_jt_kernel": "par_row"}
        with mock.patch.object(SolverFeatherPGS, "_kernel_overrides", forced):
            reference = SolverFeatherPGS(model, enable_joint_limits=False, dense_max_constraints=16)
        self.assertFalse(reference._execution_plan.use_diagonal_mass(16))

        q = np.tile(np.linspace(-0.04, 0.04, 16, dtype=np.float32), 2)
        qd = np.tile(np.linspace(0.1, -0.1, 16, dtype=np.float32), 2)
        trajectories = []
        for solver in (optimized, reference):
            state_0, state_1 = model.state(), model.state()
            state_0.joint_q.assign(q)
            state_0.joint_qd.assign(qd)
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            control = model.control()
            history = []
            for _ in range(5):
                state_0.clear_forces()
                solver.step(state_0, state_1, control, None, 1.0 / 120.0)
                state_0, state_1 = state_1, state_0
                history.append((state_0.joint_q.numpy().copy(), state_0.joint_qd.numpy().copy()))
            trajectories.append(history)

        for diagonal, dense in zip(*trajectories, strict=True):
            np.testing.assert_allclose(diagonal[0], dense[0], rtol=0.0, atol=2.0e-6)
            np.testing.assert_allclose(diagonal[1], dense[1], rtol=0.0, atol=2.0e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "sparse diagonal response requires CUDA")
    def test_sparse_diagonal_response_matches_dense_joint_limits(self):
        """Match dense position-limit rows while removing the wide response storage."""
        model = _build_sparse_diagonal_pair_model(device="cuda:0")
        optimized = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            dense_max_constraints=64,
            mf_max_constraints=32,
            pgs_iterations=8,
            enable_joint_limits=True,
            # Point friction keeps the contact-triple schedule under test; patch rows are not uniform triples.
            friction_anchor_beta=0.0,
        )
        reference = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            dense_max_constraints=64,
            mf_max_constraints=32,
            pgs_iterations=8,
            enable_joint_limits=True,
            # Point friction keeps the contact-triple schedule under test; patch rows are not uniform triples.
            friction_anchor_beta=0.0,
            use_parallel_streams=False,
        )

        self.assertTrue(optimized._sparse_diagonal_contact_solve)
        self.assertTrue(optimized._sparse_diagonal_contact_triples)
        self.assertTrue(optimized._sparse_diagonal_speculative_contact_batches)
        self.assertEqual(optimized._direct_diagonal_inertia_sizes, frozenset((16,)))
        self.assertTrue(optimized._direct_compact_diagonal_inertia)
        self.assertEqual(optimized._composite_articulation_count, 2)
        self.assertEqual(optimized._selected_tau_articulation_count, 2)
        self.assertEqual(optimized._sparse_diagonal_response_size, 16)
        self.assertEqual(optimized._sparse_diagonal_dense_size, 3)
        self.assertEqual(optimized.H_by_size[16].shape, (1, 1, 1))
        self.assertEqual(optimized.J_by_size[16].shape, (1, 1, 1))
        self.assertEqual(optimized.J_world.shape, (1, 1, 1))
        self.assertFalse(reference._sparse_diagonal_contact_solve)
        self.assertFalse(reference._sparse_diagonal_contact_triples)

        q = np.tile(np.concatenate((np.full(16, -0.105), np.full(3, 0.205))).astype(np.float32), 2)
        qd = np.tile(np.concatenate((np.full(16, -0.2), np.full(3, 0.1))).astype(np.float32), 2)
        trajectories = []
        for solver in (optimized, reference):
            state_in, state_out = model.state(), model.state()
            state_in.joint_q.assign(q)
            state_in.joint_qd.assign(qd)
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
            control = model.control()
            history = []
            for _ in range(3):
                state_in.clear_forces()
                solver.step(state_in, state_out, control, None, 1.0 / 120.0)
                state_in, state_out = state_out, state_in
                history.append((state_in.joint_q.numpy().copy(), state_in.joint_qd.numpy().copy()))
            trajectories.append(history)

        first_q, first_qd = trajectories[0][0]
        np.testing.assert_allclose(first_q.reshape(2, 19)[:, :16], -0.104, rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(first_qd.reshape(2, 19)[:, :16], 0.12, rtol=0.0, atol=2.0e-6)
        for sparse, dense in zip(*trajectories, strict=True):
            np.testing.assert_allclose(sparse[0], dense[0], rtol=2.0e-5, atol=2.0e-6)
            np.testing.assert_allclose(sparse[1], dense[1], rtol=2.0e-5, atol=2.0e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "sparse diagonal contact response requires CUDA")
    def test_sparse_diagonal_contact_friction_matches_general_owner(self):
        """Match the general owner's paired friction on sparse contacts with patch and point friction."""
        model = _build_sparse_contact_friction_model(device="cuda:0")

        def make_solver(**kwargs):
            return SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                dense_max_constraints=96,
                mf_max_constraints=32,
                pgs_iterations=8,
                enable_joint_limits=True,
                **kwargs,
            )

        def run(solver):
            state_in, state_out = model.state(), model.state()
            joint_qd = state_in.joint_qd.numpy()
            for art in np.flatnonzero(solver._model_plan.response_dof_count == 16):
                start = int(solver._model_plan.articulation_dof_start[art])
                joint_qd[start : start + 16] = -0.3
            state_in.joint_qd.assign(joint_qd)
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
            pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False)
            contacts = pipeline.contacts()
            control = model.control()
            history = []
            for _ in range(4):
                state_in.clear_forces()
                pipeline.collide(state_in, contacts)
                solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
                state_in, state_out = state_out, state_in
                history.append((state_in.joint_q.numpy().copy(), state_in.joint_qd.numpy().copy()))
            return history

        slick = run(make_solver(enable_contact_friction=False))
        for label, friction_kwargs, triples in (("patch", {}, False), ("point", {"friction_anchor_beta": 0.0}, True)):
            with self.subTest(friction=label):
                optimized = make_solver(**friction_kwargs)
                reference = make_solver(use_parallel_streams=False, **friction_kwargs)
                self.assertTrue(optimized._sparse_diagonal_contact_solve)
                self.assertEqual(optimized._friction_anchors_enabled, label == "patch")
                self.assertEqual(optimized._sparse_diagonal_contact_triples, triples)
                self.assertEqual(optimized._sparse_diagonal_speculative_contact_batches, triples)
                self.assertFalse(reference._sparse_diagonal_contact_solve)

                sparse = run(optimized)
                general = run(reference)
                self.assertGreater(int(optimized.constraint_count.numpy().max()), 0, "no contact rows were generated")
                self.assertGreater(
                    float(np.abs(sparse[-1][1] - slick[-1][1]).max()),
                    1.0e-3,
                    "friction did not change the sparse response",
                )
                for step, ((sparse_q, sparse_qd), (general_q, general_qd)) in enumerate(
                    zip(sparse, general, strict=True)
                ):
                    np.testing.assert_allclose(
                        sparse_q, general_q, rtol=2.0e-5, atol=2.0e-6, err_msg=f"joint_q step {step}"
                    )
                    np.testing.assert_allclose(
                        sparse_qd, general_qd, rtol=2.0e-5, atol=2.0e-6, err_msg=f"joint_qd step {step}"
                    )

    @unittest.skipUnless(wp.is_cuda_available(), "sparse diagonal contact response requires CUDA")
    def test_sparse_diagonal_owner_yields_to_contact_torsion(self):
        """Keep torsion-enabled worlds on the general owner, which solves the appended torsion row."""
        model = _build_sparse_diagonal_pair_model(device="cuda:0")
        solvers = {
            radius: SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                dense_max_constraints=64,
                mf_max_constraints=32,
                pgs_iterations=8,
                enable_joint_limits=True,
                contact_torsion_radius=radius,
            )
            for radius in (0.0, 0.01)
        }
        self.assertTrue(solvers[0.0]._sparse_diagonal_contact_solve)
        self.assertTrue(solvers[0.01]._contact_torsion_enabled)
        self.assertFalse(solvers[0.01]._sparse_diagonal_contact_solve)
        self.assertFalse(solvers[0.01]._sparse_diagonal_contact_triples)

    @unittest.skipUnless(wp.is_cuda_available(), "direct compact diagonal inertia requires CUDA")
    def test_direct_diagonal_inertia_refreshes_masked_inertial_changes(self):
        """Refresh the compact inertia terms when an inertial model change lands on a mass-reuse step."""
        model = _build_sparse_diagonal_pair_model(device="cuda:0", revolute_branches=True)
        solver = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            dense_max_constraints=64,
            mf_max_constraints=32,
            pgs_iterations=8,
            enable_joint_limits=True,
            update_mass_matrix_interval=4,
        )
        self.assertTrue(solver._direct_compact_diagonal_inertia)
        self.assertEqual(solver._compact_diagonal_mass_size, 16)

        # Star branch joints and their child links (skip the fixed root joint of each star).
        articulation_start = model.articulation_start.numpy()
        star_articulations = np.flatnonzero(solver._model_plan.response_dof_count == 16)
        branch_joints = np.concatenate(
            [np.arange(articulation_start[art] + 1, articulation_start[art + 1]) for art in star_articulations]
        )
        branch_bodies = model.joint_child.numpy()[branch_joints]
        branch_dofs = model.joint_qd_start.numpy()[branch_joints]

        state_in, state_out = model.state(), model.state()
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        control = model.control()

        def step():
            nonlocal state_in, state_out
            state_in.clear_forces()
            solver.step(state_in, state_out, control, None, 1.0 / 120.0)
            state_in, state_out = state_out, state_in
            return solver._diagonal_inverse_mass.numpy()[branch_dofs].copy()

        body_mass = model.body_mass.numpy()
        armature = model.joint_armature.numpy()[branch_dofs]
        # Unit inertia hinged about an axis through the COM: the response is 1 / (1 + armature).
        np.testing.assert_allclose(step(), 1.0 / (1.0 + armature), rtol=1.0e-6, atol=0.0)

        # Double the branch inertia and move the COM 0.1 m off the hinge axis on a mass-reuse step.
        body_inertia = model.body_inertia.numpy()
        body_com = model.body_com.numpy()
        body_inertia[branch_bodies] *= 2.0
        body_com[branch_bodies, 0] += 0.1
        model.body_inertia.assign(body_inertia)
        model.body_com.assign(body_com)
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        self.assertFalse(solver._step % solver.update_mass_matrix_interval == 0)
        masked = step()

        # Parallel-axis inertia about the hinge: 2 + m * 0.1^2 (+ armature).
        expected = 1.0 / (2.0 + body_mass[branch_bodies] * 0.1**2 + armature)
        np.testing.assert_allclose(masked, expected, rtol=1.0e-5, atol=0.0)
        # The next global refresh must agree with the masked refresh.
        for _ in range(2):
            step()
        self.assertTrue(solver._step % solver.update_mass_matrix_interval == 0)
        np.testing.assert_allclose(step(), masked, rtol=1.0e-6, atol=0.0)

    @unittest.skipUnless(wp.is_cuda_available(), "topology-owned CRBA requires CUDA")
    def test_topology_owned_crba_matches_generic_trajectory(self):
        """Match the generic mass pipeline while fusing topology-known factorization."""
        model = _build_chain_model(num_links=3, num_worlds=2)
        optimized = SolverFeatherPGS(model, enable_joint_limits=False, dense_max_constraints=16)
        reference = SolverFeatherPGS(
            model,
            enable_joint_limits=False,
            dense_max_constraints=16,
            use_parallel_streams=False,
        )
        self.assertTrue(optimized._parallel_augmented_drive_topology)
        self.assertIsNotNone(optimized._crba_cholesky_warp_kernels_by_size[3])
        self.assertFalse(reference._parallel_augmented_drive_topology)

        q = np.tile(np.array([0.15, -0.2, 0.3], dtype=np.float32), 2)
        qd = np.tile(np.array([0.1, -0.05, 0.2], dtype=np.float32), 2)
        trajectories = []
        for solver in (optimized, reference):
            state_0, state_1 = model.state(), model.state()
            state_0.joint_q.assign(q)
            state_0.joint_qd.assign(qd)
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            control = model.control()
            control.joint_target_q.assign(np.tile(np.array([0.1, -0.1, 0.2], dtype=np.float32), 2))
            history = []
            for _ in range(5):
                state_0.clear_forces()
                solver.step(state_0, state_1, control, None, 1.0 / 120.0)
                state_0, state_1 = state_1, state_0
                history.append((state_0.joint_q.numpy().copy(), state_0.joint_qd.numpy().copy()))
            trajectories.append(history)

        for fused, generic in zip(*trajectories, strict=True):
            np.testing.assert_allclose(fused[0], generic[0], rtol=0.0, atol=2.0e-6)
            np.testing.assert_allclose(fused[1], generic[1], rtol=0.0, atol=2.0e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "matrix-free diagonal fusion requires CUDA")
    def test_matrix_free_diagonal_fusion_requires_nonaliased_world_response(self):
        """Keep the extra tiled reduction off response buffers that already alias world storage."""
        aliased = SolverFeatherPGS(
            _build_chain_model(num_links=23, num_worlds=1),
            pgs_mode="matrix_free",
            dense_max_constraints=192,
        )
        direct = SolverFeatherPGS(
            _build_chain_model(num_links=23, num_worlds=1, with_free_body=True),
            pgs_mode="matrix_free",
            dense_max_constraints=192,
        )

        self.assertTrue(aliased._jy_world_aliased)
        self.assertFalse(aliased._hinv_jt_writes_world)
        self.assertEqual(aliased._hinv_jt_diag_sizes, frozenset())
        self.assertFalse(direct._jy_world_aliased)
        self.assertTrue(direct._hinv_jt_writes_world)
        self.assertEqual(direct._hinv_jt_diag_sizes, frozenset((23,)))

    def test_mfgs_metadata_storage_respects_resource_budget(self):
        """Keep resident metadata only when its complete working set fits."""
        compact = _use_resident_mfgs_metadata(192, 64, 29, 101376, has_drive_rows=False, fuse_vel_limits=False)
        large = _use_resident_mfgs_metadata(1024, 4096, 604, 101376, has_drive_rows=False, fuse_vel_limits=False)
        overcommitted = _use_resident_mfgs_metadata(192, 64, 29, 4096, has_drive_rows=False, fuse_vel_limits=False)

        self.assertTrue(compact)
        self.assertFalse(large)
        self.assertFalse(overcommitted)

    def test_serial_kernel_block_dim_validation(self):
        model = newton.ModelBuilder().finalize()
        for bad in (0, -32, 1, 33, 100):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(ValueError, "serial_kernel_block_dim"):
                    SolverFeatherPGS(model, serial_kernel_block_dim=bad)
        for good in (32, 64, 128, 256, 512):
            with self.subTest(value=good):
                solver = SolverFeatherPGS(model, serial_kernel_block_dim=good)
                self.assertEqual(solver.serial_kernel_block_dim, good)

    def test_tile_threads_validation(self):
        model = newton.ModelBuilder().finalize()
        for bad in (0, -64, 16, 48, 96, 512):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(ValueError, "tile_threads"):
                    SolverFeatherPGS(model, tile_threads=bad)
        for good in (32, 64, 128, 256):
            with self.subTest(value=good):
                solver = SolverFeatherPGS(model, tile_threads=good)
                self.assertEqual(solver.tile_threads, good)

    @unittest.skipUnless(wp.is_cuda_available(), "tiled launch-config step test requires CUDA")
    def test_non_default_tile_threads_compiles_and_steps(self):
        model = _build_chain_model()
        forced = {"cholesky_kernel": "tiled", "trisolve_kernel": "tiled", "hinv_jt_kernel": "tiled"}
        with mock.patch.object(SolverFeatherPGS, "_kernel_overrides", forced):
            solver = SolverFeatherPGS(
                model,
                serial_kernel_block_dim=64,
                tile_threads=128,
                pgs_mode="split",
                pgs_kernel="loop",
                dense_max_constraints=384,
            )
        self.assertTrue(all(solver._execution_plan.use_tiled_hinv_jt(size) for size in solver.size_groups))
        self.assertFalse(any(solver._execution_plan.use_fused_hinv_jt(size) for size in solver.size_groups))
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        for _ in range(5):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, 1.0 / 600.0)
            state_0, state_1 = state_1, state_0
        wp.synchronize()
        self.assertTrue(np.isfinite(state_0.joint_q.numpy()).all())
        self.assertTrue(np.isfinite(state_0.joint_qd.numpy()).all())

    @unittest.skipUnless(wp.is_cuda_available(), "propagation matrix-free step test requires CUDA")
    def test_propagation_matrix_free_compiles_and_steps_with_velocity_limit_rows(self):
        model = _build_chain_model(num_links=3, num_worlds=1)
        model.joint_velocity_limit.assign(np.full(model.joint_dof_count, 0.1, dtype=np.float32))

        solver = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            articulated_contact_response="propagation",
            enable_joint_velocity_limits=True,
            pgs_iterations=2,
            pgs_velocity_iterations=1,
            dense_max_constraints=16,
            mf_max_constraints=16,
        )
        state_0, state_1 = model.state(), model.state()
        state_0.joint_qd.assign(np.full(model.joint_dof_count, 1.0, dtype=np.float32))
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

        solver.step(state_0, state_1, model.control(), None, 1.0 / 600.0)
        wp.synchronize()

        self.assertGreater(int(solver.constraint_count.numpy()[0]), 0)
        self.assertTrue(np.isfinite(state_1.joint_q.numpy()).all())
        self.assertTrue(np.isfinite(state_1.joint_qd.numpy()).all())

    def test_fuse_joint_velocity_limits_validation(self):
        # fuse_joint_velocity_limits defaults to True but only engages in the
        # matrix_free + physx_pgs + velocity-limits formulation; anywhere else
        # it must be silently inert (no error), so the default doesn't
        # constrain unrelated solver modes.
        model = _build_chain_model(num_links=2, num_worlds=1)
        for kwargs in (
            {},  # neither matrix_free nor velocity limits nor physx drive
            {"pgs_mode": "matrix_free", "drive_mode": "physx_pgs"},  # no velocity limits
        ):
            with self.subTest(kwargs=kwargs):
                if kwargs.get("pgs_mode") == "matrix_free" and model.device.is_cpu:
                    with self.assertRaisesRegex(NotImplementedError, "requires CUDA"):
                        SolverFeatherPGS(model, fuse_joint_velocity_limits=True, **kwargs)
                    continue
                solver = SolverFeatherPGS(model, fuse_joint_velocity_limits=True, **kwargs)
                self.assertFalse(solver.fuse_joint_velocity_limits)
        if wp.is_cuda_available():
            # drive_mode='augmented' with velocity limits enabled is inert too
            # (matrix_free construction requires CUDA).
            solver = SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                enable_joint_velocity_limits=True,
                drive_mode="augmented",
                fuse_joint_velocity_limits=True,
            )
            self.assertFalse(solver.fuse_joint_velocity_limits)
            # The applicable combination engages by default (no kwarg passed),
            # and an explicit False opts out.
            solver = SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                enable_joint_velocity_limits=True,
                drive_mode="physx_pgs",
            )
            self.assertTrue(solver.fuse_joint_velocity_limits)
            solver = SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                enable_joint_velocity_limits=True,
                drive_mode="physx_pgs",
                fuse_joint_velocity_limits=False,
            )
            self.assertFalse(solver.fuse_joint_velocity_limits)
            # velocity_limit_activation_fraction=inf is the explicit
            # never-activate kill-switch for velocity limits; the fused clamp
            # rides the drive-row visit (which the inf gate does not cover),
            # so an explicit opt-in must stay inert rather than re-enforce
            # limits the fraction disabled.
            solver = SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                enable_joint_velocity_limits=True,
                drive_mode="physx_pgs",
                fuse_joint_velocity_limits=True,
                velocity_limit_activation_fraction=float("inf"),
            )
            self.assertFalse(solver.fuse_joint_velocity_limits)
            # The clamp pass runs at the end of each iteration, independent of
            # the drive-row visit, so frozen drive rows during velocity
            # iterations no longer disable fusion: the clamp keeps enforcing
            # the limit there, exactly like the dedicated rows it replaces.
            for velocity_drive_mode in ("freeze", "active"):
                solver = SolverFeatherPGS(
                    model,
                    pgs_mode="matrix_free",
                    enable_joint_velocity_limits=True,
                    drive_mode="physx_pgs",
                    fuse_joint_velocity_limits=True,
                    pgs_velocity_iterations=2,
                    pgs_velocity_drive_mode=velocity_drive_mode,
                )
                self.assertTrue(solver.fuse_joint_velocity_limits)

    @unittest.skipUnless(wp.is_cuda_available(), "fused velocity-limit clamp requires CUDA")
    def test_fuse_joint_velocity_limits_clamps_driven_dofs_without_rows(self):
        # A strongly driven chain with a low joint velocity limit: with
        # fuse_joint_velocity_limits=True the driven DOFs must (a) stay within
        # the limit plus a small GS-convergence tolerance and (b) use fewer
        # dense rows than the dedicated-row formulation (the two vel-limit
        # rows per driven DOF are replaced by a stateless end-of-iteration
        # clamp).
        num_links = 3
        qdot_max = 1.0

        def run(fuse):
            model = _build_chain_model(num_links=num_links, num_worlds=1)
            n = model.joint_dof_count
            model.joint_target_ke.assign(np.full(n, 200.0, dtype=np.float32))
            model.joint_target_kd.assign(np.full(n, 5.0, dtype=np.float32))
            model.joint_velocity_limit.assign(np.full(n, qdot_max, dtype=np.float32))
            solver = SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                drive_mode="physx_pgs",
                enable_joint_velocity_limits=True,
                fuse_joint_velocity_limits=fuse,
                pgs_iterations=64,
                dense_max_constraints=16,
                mf_max_constraints=16,
            )
            state_0, state_1 = model.state(), model.state()
            control = model.control()
            control.joint_target_q.assign(np.full(n, 3.0, dtype=np.float32))
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            max_speed = 0.0
            for _ in range(60):
                state_0.clear_forces()
                solver.step(state_0, state_1, control, None, 1.0 / 60.0)
                state_0, state_1 = state_1, state_0
                max_speed = max(max_speed, float(np.max(np.abs(state_0.joint_qd.numpy()))))
            wp.synchronize()
            return max_speed, int(solver.constraint_count.numpy()[0])

        fused_speed, fused_rows = run(True)
        dedicated_speed, dedicated_rows = run(False)

        # (a) the fused clamp reproduces the dedicated-row behavior it
        # replaces. Both are stateless passes at the end of each iteration,
        # and on a coupled chain both leave the same small GS residual above
        # the limit (clamping DOF i perturbs DOF j through the tree response,
        # and the sweep visits each DOF once per iteration) — dedicated rows
        # measure ~1.12x the limit here. Assert against that reference, not
        # an exact limit no last-word formulation achieves on this scene.
        self.assertLessEqual(fused_speed, dedicated_speed * 1.05)
        self.assertLessEqual(fused_speed, qdot_max * 1.25)
        # (b) the fused solve drops the two dedicated vel-limit rows per
        # driven DOF: 3 drive rows vs 3 drive + 6 vel-limit rows.
        self.assertGreater(fused_rows, 0)
        self.assertLess(fused_rows, dedicated_rows)
        self.assertEqual(dedicated_rows - fused_rows, 2 * num_links)


if __name__ == "__main__":
    unittest.main(verbosity=2)
