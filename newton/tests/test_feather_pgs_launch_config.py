# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Construction, option validation and kernel selection of SolverFeatherPGS."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import kernels as feather_pgs_kernels
from newton._src.solvers.feather_pgs.solver_feather_pgs import (
    _DENSE_META_MAX_PARENT,
    _estimate_cholesky_shared_memory,
    _FeatherPGSExecutionPlan,
    _select_hinv_jt_chunk_size,
    _use_resident_mfgs_metadata,
)
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def _build_chain_model(device, num_links=3, num_worlds=2, *, with_free_body=False):
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
    return main.finalize(device=device)


def _build_heterogeneous_world_model(device):
    free_template = newton.ModelBuilder()
    free_body = free_template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    free_template.add_articulation([free_template.add_joint_free(parent=-1, child=free_body)])

    slider_template = newton.ModelBuilder()
    slider_body = slider_template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    slider_template.add_articulation(
        [slider_template.add_joint_prismatic(parent=-1, child=slider_body, axis=newton.Axis.X)]
    )

    builder = newton.ModelBuilder()
    builder.add_world(free_template)
    builder.add_world(slider_template)
    return builder.finalize(device=device)


def test_defaults(test, device):
    """Keep the documented constructor defaults."""
    solver = SolverFeatherPGS(_build_chain_model(device, num_links=2, num_worlds=1))
    test.assertEqual(solver.pgs_iterations, 12)
    test.assertAlmostEqual(solver.pgs_beta, 0.2)
    test.assertAlmostEqual(solver.pgs_cfm, 1.0e-6)
    test.assertEqual(solver.pgs_omega, 1.0)
    test.assertEqual(solver.update_mass_matrix_interval, 1)
    test.assertFalse(solver.enable_joint_limits)
    test.assertEqual(solver.joint_limit_activation_gap, float("inf"))
    test.assertFalse(solver.enable_joint_velocity_limits)
    test.assertEqual(solver.velocity_limit_activation_fraction, 0.0)
    test.assertEqual(solver.dense_max_constraints, 32)
    test.assertEqual(solver.mf_max_constraints, 512)
    test.assertTrue(solver.warn_constraint_overflow)
    np.testing.assert_allclose(solver.rigid_body_angular_damping.numpy(), 0.05)


def test_constructor_validates_options(test, device):
    """Reject option values that cannot be honored."""
    model = _build_chain_model(device, num_links=2, num_worlds=1)
    invalid = (
        ({"update_mass_matrix_interval": 0}, "update_mass_matrix_interval"),
        ({"pgs_iterations": -1}, "pgs_iterations"),
        ({"joint_limit_activation_gap": -0.1}, "joint_limit_activation_gap"),
        ({"joint_limit_activation_gap": float("nan")}, "joint_limit_activation_gap"),
        ({"velocity_limit_activation_fraction": -0.1}, "velocity_limit_activation_fraction"),
        ({"velocity_limit_activation_fraction": 1.5}, "velocity_limit_activation_fraction"),
        ({"velocity_limit_activation_fraction": float("nan")}, "velocity_limit_activation_fraction"),
        ({"dense_max_constraints": 0}, "dense_max_constraints"),
        ({"dense_max_constraints": _DENSE_META_MAX_PARENT + 2}, "dense_max_constraints"),
        ({"mf_max_constraints": 0}, "mf_max_constraints"),
    )
    for kwargs, message in invalid:
        with test.subTest(**kwargs):
            with test.assertRaisesRegex(ValueError, message):
                SolverFeatherPGS(model, **kwargs)
    test.assertEqual(
        SolverFeatherPGS(model, velocity_limit_activation_fraction=float("inf")).velocity_limit_activation_fraction,
        float("inf"),
    )


def test_unsupported_model_features_raise(test, device):
    """Reject model features the solver would otherwise silently ignore."""
    mimic = newton.ModelBuilder()
    links = [mimic.add_link(mass=1.0, inertia=wp.mat33(np.eye(3))) for _ in range(2)]
    leader = mimic.add_joint_revolute(-1, links[0], axis=newton.Axis.Y)
    follower = mimic.add_joint_revolute(links[0], links[1], axis=newton.Axis.Y)
    mimic.add_articulation([leader, follower])
    mimic.add_constraint_mimic(follower, leader, coef1=1.0)
    with test.assertRaisesRegex(NotImplementedError, "mimic"):
        SolverFeatherPGS(mimic.finalize(device=device))

    particles = newton.ModelBuilder()
    body = particles.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    particles.add_articulation([particles.add_joint_revolute(-1, body, axis=newton.Axis.Y)])
    particles.add_particle(wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0), mass=1.0)
    with test.assertRaisesRegex(NotImplementedError, "particles"):
        SolverFeatherPGS(particles.finalize(device=device))

    disabled = newton.ModelBuilder()
    body = disabled.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    disabled.add_articulation([disabled.add_joint_revolute(-1, body, axis=newton.Axis.Y, enabled=False)])
    with test.assertRaisesRegex(NotImplementedError, "disabled joints"):
        SolverFeatherPGS(disabled.finalize(device=device))


def test_default_kernel_selection_is_cached(test, device):
    """Resolve identical solver shapes to the same cached kernel objects."""
    model = _build_chain_model(device)
    first = SolverFeatherPGS(model)
    second = SolverFeatherPGS(model)
    for attr in ("_cholesky_kernels_by_size", "_triangular_solve_kernels_by_size", "_hinv_jt_kernels_by_size"):
        first_kernels = getattr(first, attr)
        second_kernels = getattr(second, attr)
        test.assertEqual(set(first_kernels), set(second_kernels))
        for size, kernel in first_kernels.items():
            test.assertIs(kernel, second_kernels[size], f"{attr}[{size}]")
    test.assertIs(first._pgs_solve_mf_gs_kernel, second._pgs_solve_mf_gs_kernel)


def test_compact_world_dof_mapping_pads_heterogeneous_worlds(test, device):
    """Pad the per-world response DOF map of worlds with fewer DOFs."""
    solver = SolverFeatherPGS(_build_heterogeneous_world_model(device))
    test.assertEqual(solver.max_world_dofs, 6)
    np.testing.assert_array_equal(solver.world_dof_count.numpy(), np.array((6, 1), dtype=np.int32))
    indices = solver.world_dof_indices.numpy()
    np.testing.assert_array_equal(indices[0], np.arange(6, dtype=np.int32))
    test.assertGreaterEqual(int(indices[1, 0]), 0)
    np.testing.assert_array_equal(indices[1, 1:], np.full(5, -1, dtype=np.int32))


def test_diagonal_fusion_requires_nonaliased_world_response(test, device):
    """Compute the row diagonal in H^-1 J^T only when it writes separate world storage."""
    aliased = SolverFeatherPGS(_build_chain_model(device, num_links=23, num_worlds=1), dense_max_constraints=192)
    direct = SolverFeatherPGS(
        _build_chain_model(device, num_links=23, num_worlds=1, with_free_body=True), dense_max_constraints=192
    )
    test.assertTrue(aliased._jy_world_aliased)
    test.assertFalse(aliased._hinv_jt_writes_world)
    test.assertEqual(aliased._hinv_jt_diag_sizes, frozenset())
    test.assertFalse(direct._jy_world_aliased)
    test.assertTrue(direct._hinv_jt_writes_world)
    test.assertEqual(direct._hinv_jt_diag_sizes, frozenset((23,)))


def test_tiled_and_loop_kernels_step_identically(test, device):
    """Match the tiled and loop factorizations on a chain that selects tiled kernels by default."""
    trajectories = []
    try:
        for overrides in ({}, {"cholesky_kernel": "loop", "trisolve_kernel": "loop", "hinv_jt_kernel": "par_row"}):
            SolverFeatherPGS._kernel_overrides = overrides
            model = _build_chain_model(device, num_links=14, num_worlds=2)
            solver = SolverFeatherPGS(model, dense_max_constraints=64)
            if not overrides:
                test.assertTrue(solver._execution_plan.use_tiled_cholesky(14))
                test.assertTrue(solver._execution_plan.use_tiled_hinv_jt(14))
            state_0, state_1 = model.state(), model.state()
            control = model.control()
            for _ in range(10):
                solver.step(state_0, state_1, control, None, 1.0 / 120.0)
                state_0, state_1 = state_1, state_0
            trajectories.append(state_0.joint_q.numpy().copy())
    finally:
        SolverFeatherPGS._kernel_overrides = {}
    np.testing.assert_allclose(trajectories[0], trajectories[1], rtol=0.0, atol=1.0e-4)


def _build_box_with_equality(device, *, enabled: bool, target_kind: int = 0, target: int = -1):
    """Build a free box held to the world by a MuJoCo CONNECT equality row."""
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_custom_values(
        **{
            "mujoco:equality_constraint_type": 0,
            "mujoco:equality_constraint_objtype": 1,
            "mujoco:equality_constraint_body1": body,
            "mujoco:equality_constraint_body2": -1,
            "mujoco:equality_constraint_anchor": wp.vec3(0.0, 0.0, 1.0),
            "mujoco:equality_constraint_enabled": enabled,
            "mujoco:equality_constraint_target_kind": target_kind,
            "mujoco:equality_constraint_target": target,
        }
    )
    return builder.finalize(device=device)


def test_unconverted_equality_constraints_raise(test, device):
    """Reject an enabled MuJoCo equality row instead of letting the body fall through it."""
    with test.assertRaisesRegex(NotImplementedError, r"equality_constraint rows \[0\]"):
        SolverFeatherPGS(_build_box_with_equality(device, enabled=True))
    # A link to a projected entity that does not exist does not make the row enforced.
    for target_kind in (1, 2):
        with test.assertRaisesRegex(NotImplementedError, "equality"):
            SolverFeatherPGS(_build_box_with_equality(device, enabled=True, target_kind=target_kind, target=7))

    # Imported equalities are converted to Newton loop joints by default; those rows are
    # judged through the loop joint, which this solver rejects separately.
    mjcf = """
    <mujoco>
      <worldbody>
        <body name="link" pos="0 0 1">
          <joint name="hinge" type="hinge" axis="0 1 0"/>
          <geom type="box" size="0.1 0.1 0.1"/>
        </body>
      </worldbody>
      <equality><connect body1="link" anchor="0.1 0 0"/></equality>
    </mujoco>
    """
    for convert, message in ((True, "loop-closing"), (False, "equality")):
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf, convert_mjc_equality_constraints=convert)
        imported = builder.finalize(device=device)
        test.assertEqual(imported.mujoco.equality_constraint_count, 1)
        with test.assertRaisesRegex(NotImplementedError, message):
            SolverFeatherPGS(imported)

    # A disabled row constructs and has no effect.
    model = _build_box_with_equality(device, enabled=False)
    solver = SolverFeatherPGS(model)
    state_0, state_1 = model.state(), model.state()
    for _ in range(10):
        solver.step(state_0, state_1, model.control(), None, 0.01)
        state_0, state_1 = state_1, state_0
    test.assertLess(float(state_0.body_q.numpy()[0, 2]), 0.95)


def test_equality_link_must_name_the_projected_constraint(test, device):
    """Reject an equality row whose projection link names an entity that does not enforce it."""
    # A CONNECT row holding a free box to the world that claims the box's own free joint: the
    # joint is in range but is not a loop joint, so the box would fall through the constraint.
    with test.assertRaisesRegex(NotImplementedError, r"equality_constraint rows \[0\]"):
        SolverFeatherPGS(_build_box_with_equality(device, enabled=True, target_kind=1, target=0))

    def build_pendulum_with_connect(eq_type: int, joint_kind: str, row_names_world: bool):
        builder = newton.ModelBuilder()
        link = builder.add_link(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
        hinge = builder.add_joint_revolute(-1, link, axis=newton.Axis.Y)
        builder.add_articulation([hinge])
        other = builder.add_body(xform=wp.transform((0.5, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_box(other, hx=0.1, hy=0.1, hz=0.1)
        add_joint = builder.add_joint_ball if joint_kind == "ball" else builder.add_joint_fixed
        loop = add_joint(link, other)
        builder.add_custom_values(
            **{
                "mujoco:equality_constraint_type": eq_type,
                "mujoco:equality_constraint_objtype": 1,
                "mujoco:equality_constraint_body1": link,
                "mujoco:equality_constraint_body2": -1 if row_names_world else other,
                "mujoco:equality_constraint_enabled": True,
                "mujoco:equality_constraint_target_kind": 1,
                "mujoco:equality_constraint_target": loop,
            }
        )
        return builder.finalize(device=device)

    # Mismatched projections: a WELD row linking a ball joint, and a row whose bodies are not
    # the linked joint's endpoints. Each is an unenforced equality, not a loop joint.
    for eq_type, joint_kind, row_names_world in ((1, "ball", False), (0, "ball", True)):
        with test.subTest(eq_type=eq_type, joint_kind=joint_kind, row_names_world=row_names_world):
            with test.assertRaisesRegex(NotImplementedError, "equality"):
                SolverFeatherPGS(build_pendulum_with_connect(eq_type, joint_kind, row_names_world))
    # A tree joint between the row's bodies is not a projection either: the row's anchor is
    # not enforced by it.
    builder = newton.ModelBuilder()
    root = builder.add_link(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
    tip = builder.add_link(xform=wp.transform((0.5, 0.0, 1.0), wp.quat_identity()))
    for body in (root, tip):
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    hinge = builder.add_joint_revolute(-1, root, axis=newton.Axis.Y)
    ball = builder.add_joint_ball(root, tip, parent_xform=wp.transform((0.5, 0.0, 0.0), wp.quat_identity()))
    builder.add_articulation([hinge, ball])
    builder.add_custom_values(
        **{
            "mujoco:equality_constraint_type": 0,
            "mujoco:equality_constraint_objtype": 1,
            "mujoco:equality_constraint_body1": root,
            "mujoco:equality_constraint_body2": tip,
            "mujoco:equality_constraint_enabled": True,
            "mujoco:equality_constraint_target_kind": 1,
            "mujoco:equality_constraint_target": ball,
        }
    )
    with test.subTest(target="tree ball joint"):
        with test.assertRaisesRegex(NotImplementedError, "equality"):
            SolverFeatherPGS(builder.finalize(device=device))
    # The matching projection is judged through its loop joint instead.
    for eq_type, joint_kind in ((0, "ball"), (1, "fixed")):
        with test.subTest(eq_type=eq_type, joint_kind=joint_kind, matching=True):
            with test.assertRaisesRegex(NotImplementedError, "loop-closing"):
                SolverFeatherPGS(build_pendulum_with_connect(eq_type, joint_kind, False))

    # A JOINT row linking a mimic between other joints is likewise unenforced; the importer's
    # own projection is judged through its mimic constraint.
    mjcf = """
    <mujoco>
      <worldbody>
        <body name="a" pos="0 0 1">
          <joint name="ja" type="hinge" axis="0 1 0"/>
          <geom type="box" size="0.1 0.1 0.1"/>
          <body name="b" pos="0.3 0 0">
            <joint name="jb" type="hinge" axis="0 1 0"/>
            <geom type="box" size="0.1 0.1 0.1"/>
            <body name="c" pos="0.3 0 0">
              <joint name="jc" type="hinge" axis="0 1 0"/>
              <geom type="box" size="0.1 0.1 0.1"/>
            </body>
          </body>
        </body>
      </worldbody>
      <equality><joint joint1="jb" joint2="ja"/></equality>
    </mujoco>
    """
    for swap, message in ((False, "mimic constraints"), (True, "equality")):
        with test.subTest(mimic_joints_swapped=swap):
            builder = newton.ModelBuilder()
            builder.add_mjcf(mjcf)
            model = builder.finalize(device=device)
            test.assertEqual(int(model.mujoco.equality_constraint_target_kind.numpy()[0]), 2)
            if swap:
                joint0 = model.constraint_mimic_joint0.numpy()
                joint1 = model.constraint_mimic_joint1.numpy()
                model.constraint_mimic_joint0.assign(joint1)
                model.constraint_mimic_joint1.assign(joint0)
            with test.assertRaisesRegex(NotImplementedError, message):
                SolverFeatherPGS(model)


def test_enabling_equality_constraint_at_runtime_raises(test, device):
    """Re-check the MuJoCo equality rows when constraint properties change."""
    model = _build_box_with_equality(device, enabled=False)
    solver = SolverFeatherPGS(model)
    # Other notifications do not re-read the constraint rows.
    solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)
    model.mujoco.equality_constraint_enabled.assign(np.array([True]))
    with test.assertRaisesRegex(NotImplementedError, "equality"):
        solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)
    with test.assertRaisesRegex(NotImplementedError, "equality"):
        solver.notify_model_changed(newton.ModelFlags.ALL)
    model.mujoco.equality_constraint_enabled.assign(np.array([False]))
    solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)


class TestFeatherPGSLaunchConfig(unittest.TestCase):
    def test_cpu_construction_raises(self):
        """Reject construction on a CPU device with a clear error."""
        model = _build_chain_model("cpu", num_links=2, num_worlds=1)
        with self.assertRaisesRegex(NotImplementedError, "requires a CUDA device"):
            SolverFeatherPGS(model)

    def test_launch_geometry_kernels_use_dedicated_modules(self):
        """Keep custom-block-dimension kernels out of the general module."""
        expected_modules = {
            "eval_rigid_fk_id": "kinematics",
            "eval_rigid_tau": "inverse_dynamics",
            "eval_rigid_tau_and_augmented_drives": "inverse_dynamics",
            "compute_composite_inertia": "mass_dynamics",
            "crba_fill_par_dof": "mass_dynamics",
        }
        general_module = feather_pgs_kernels.__name__
        for kernel_name, module_suffix in expected_modules.items():
            with self.subTest(kernel=kernel_name):
                kernel_module = getattr(feather_pgs_kernels, kernel_name).module.name
                self.assertEqual(kernel_module, f"{general_module}.{module_suffix}")

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

    def test_hinv_execution_plan_falls_back_or_rejects_safely(self):
        """Fall back to the row-parallel H^-1 J^T for oversized groups and reject forced invalid tiles."""
        plan_args = {"max_shared_memory": 101376, "cholesky_kernel": "auto", "small_dof_threshold": 12}
        fallback = _FeatherPGSExecutionPlan.build(
            [160], max_constraints=384, hinv_jt_kernel="auto", tile_threads=64, **plan_args
        )
        self.assertFalse(fallback.use_tiled_hinv_jt(160))
        zero_capacity = _FeatherPGSExecutionPlan.build(
            [23], max_constraints=0, hinv_jt_kernel="tiled", tile_threads=64, **plan_args
        )
        self.assertFalse(zero_capacity.use_tiled_hinv_jt(23))
        with self.assertRaisesRegex(ValueError, "hinv_jt_kernel='tiled'"):
            _FeatherPGSExecutionPlan.build(
                [160], max_constraints=384, hinv_jt_kernel="tiled", tile_threads=64, **plan_args
            )

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

    def test_mfgs_metadata_storage_respects_resource_budget(self):
        """Keep the dense row metadata resident only when its working set fits."""
        self.assertTrue(_use_resident_mfgs_metadata(192, 64, 29, 101376))
        self.assertFalse(_use_resident_mfgs_metadata(1024, 4096, 604, 101376))
        self.assertFalse(_use_resident_mfgs_metadata(192, 64, 29, 4096))


devices = get_cuda_test_devices()
for _name, _func in (
    ("test_defaults", test_defaults),
    ("test_constructor_validates_options", test_constructor_validates_options),
    ("test_unsupported_model_features_raise", test_unsupported_model_features_raise),
    ("test_default_kernel_selection_is_cached", test_default_kernel_selection_is_cached),
    (
        "test_compact_world_dof_mapping_pads_heterogeneous_worlds",
        test_compact_world_dof_mapping_pads_heterogeneous_worlds,
    ),
    (
        "test_diagonal_fusion_requires_nonaliased_world_response",
        test_diagonal_fusion_requires_nonaliased_world_response,
    ),
    ("test_tiled_and_loop_kernels_step_identically", test_tiled_and_loop_kernels_step_identically),
    ("test_unconverted_equality_constraints_raise", test_unconverted_equality_constraints_raise),
    (
        "test_equality_link_must_name_the_projected_constraint",
        test_equality_link_must_name_the_projected_constraint,
    ),
    ("test_enabling_equality_constraint_at_runtime_raises", test_enabling_equality_constraint_at_runtime_raises),
):
    add_function_test(TestFeatherPGSLaunchConfig, _name, _func, devices=devices)


if __name__ == "__main__":
    unittest.main()
