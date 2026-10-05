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
    _STATIC_SHARED_MEMORY_BYTES,
    _estimate_cholesky_shared_memory,
    _estimate_mf_solve_shared_memory,
    _estimate_tiled_row_shared_memory,
    _FeatherPGSExecutionPlan,
    _select_delassus_chunk_size,
    _select_hinv_jt_chunk_size,
    _use_resident_mfgs_metadata,
)
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


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


def _build_limited_chain_on_ground(device, num_links, num_worlds=2, *, with_free_body=False):
    """A hanging chain of driven, limited capsule links above the ground, optionally with a falling box."""
    world = newton.ModelBuilder()
    world.add_ground_plane()
    joints = []
    parent = -1
    top = 0.3 * num_links + 0.2
    for i in range(num_links):
        link = world.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, top - 0.3 * i), wp.quat_identity()))
        world.add_shape_capsule(link, radius=0.04, half_height=0.12)
        joints.append(
            world.add_joint_revolute(
                parent,
                link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(
                    wp.vec3(0.0, 0.0, top + 0.15) if parent < 0 else wp.vec3(0.0, 0.0, -0.15), wp.quat_identity()
                ),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.15), wp.quat_identity()),
                limit_lower=-0.6,
                limit_upper=0.6,
                target_ke=200.0,
                target_kd=5.0,
                target_pos=0.3,
            )
        )
        parent = link
    world.add_articulation(joints)
    if with_free_body:
        body = world.add_body(xform=wp.transform(wp.vec3(0.3, 0.0, 0.3), wp.quat_rpy(0.2, 0.1, 0.3)))
        world.add_shape_box(body, hx=0.1, hy=0.08, hz=0.06)
    builder = newton.ModelBuilder()
    builder.replicate(world, num_worlds)
    return builder.finalize(device=device)


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
    test.assertEqual(solver.pgs_mode, "matrix_free")
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
                SolverFeatherPGS(model, pgs_mode="matrix_free", **kwargs)
    test.assertEqual(
        SolverFeatherPGS(
            model, pgs_mode="matrix_free", velocity_limit_activation_fraction=float("inf")
        ).velocity_limit_activation_fraction,
        float("inf"),
    )


def test_unsupported_model_features_raise(test, device):
    """Reject model features the solver would otherwise silently ignore."""
    mimic = newton.ModelBuilder()
    links = [mimic.add_link(mass=1.0, inertia=wp.mat33(np.eye(3))) for _ in range(3)]
    root = mimic.add_joint_revolute(-1, links[0], axis=newton.Axis.Y)
    leader = mimic.add_joint_ball(links[0], links[1])
    follower = mimic.add_joint_ball(links[1], links[2])
    mimic.add_articulation([root, leader, follower])
    # Quaternion-coordinate mimics have no componentwise row.
    mimic.set_joint_mimic(follower, leader)
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
    first = SolverFeatherPGS(model, pgs_mode="matrix_free")
    second = SolverFeatherPGS(model, pgs_mode="matrix_free")
    for attr in ("_cholesky_kernels_by_size", "_triangular_solve_kernels_by_size", "_hinv_jt_kernels_by_size"):
        first_kernels = getattr(first, attr)
        second_kernels = getattr(second, attr)
        test.assertEqual(set(first_kernels), set(second_kernels))
        for size, kernel in first_kernels.items():
            test.assertIs(kernel, second_kernels[size], f"{attr}[{size}]")
    test.assertIs(first._pgs_solve_mf_gs_kernel, second._pgs_solve_mf_gs_kernel)


def test_compact_world_dof_mapping_pads_heterogeneous_worlds(test, device):
    """Pad the per-world response DOF map of worlds with fewer DOFs."""
    solver = SolverFeatherPGS(_build_heterogeneous_world_model(device), pgs_mode="matrix_free")
    test.assertEqual(solver.max_world_dofs, 6)
    np.testing.assert_array_equal(solver.world_dof_count.numpy(), np.array((6, 1), dtype=np.int32))
    indices = solver.world_dof_indices.numpy()
    np.testing.assert_array_equal(indices[0], np.arange(6, dtype=np.int32))
    test.assertGreaterEqual(int(indices[1, 0]), 0)
    np.testing.assert_array_equal(indices[1, 1:], np.full(5, -1, dtype=np.int32))


def test_diagonal_fusion_requires_nonaliased_world_response(test, device):
    """Compute the row diagonal in H^-1 J^T only when it writes separate world storage."""
    aliased = SolverFeatherPGS(
        _build_chain_model(device, num_links=23, num_worlds=1), pgs_mode="matrix_free", dense_max_constraints=192
    )
    direct = SolverFeatherPGS(
        _build_chain_model(device, num_links=23, num_worlds=1, with_free_body=True),
        pgs_mode="matrix_free",
        dense_max_constraints=192,
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
            solver = SolverFeatherPGS(model, pgs_mode="matrix_free", dense_max_constraints=64)
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
    # enforced through the loop joint (see test_feather_pgs_connect). Unconverted, they raise.
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
    builder = newton.ModelBuilder()
    builder.add_mjcf(mjcf, convert_mjc_equality_constraints=False)
    imported = builder.finalize(device=device)
    test.assertEqual(imported.mujoco.equality_constraint_count, 1)
    with test.assertRaisesRegex(NotImplementedError, "equality"):
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
    # own projection is enforced through its mimic constraint.
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
    for swap in (False, True):
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
            if swap:
                with test.assertRaisesRegex(NotImplementedError, "equality"):
                    SolverFeatherPGS(model, pgs_mode="matrix_free")
            else:
                SolverFeatherPGS(model, pgs_mode="matrix_free")


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


def test_joint_limit_solver_compatibility_validation(test, device):
    """Reject the options the split solve cannot honor and unknown solve modes."""
    model = _build_chain_model(device, num_links=2, num_worlds=1)
    with test.assertRaisesRegex(NotImplementedError, "requires pgs_mode='matrix_free'"):
        SolverFeatherPGS(model, pgs_mode="split", enable_joint_velocity_limits=True)
    with test.assertRaisesRegex(ValueError, "pgs_mode"):
        SolverFeatherPGS(model, pgs_mode="dense")


def test_split_rejects_matrix_free_only_options(test, device):
    """Reject in the split solve every option that only the matrix-free solve implements."""
    model = _build_chain_model(device, num_links=2, num_worlds=1)
    options = (
        ({"enable_joint_velocity_limits": True}, "enable_joint_velocity_limits"),
        ({"drive_mode": "physx_pgs"}, "physx_pgs"),
        ({"friction_anchor_beta": 0.2}, "friction_anchor_beta"),
        ({"pgs_contact_regularization": 0.02}, "pgs_contact_regularization"),
        ({"pgs_velocity_iterations": 2}, "pgs_velocity_iterations"),
        ({"pgs_warmstart": True}, "pgs_warmstart"),
        ({"contact_compliance": True, "friction_anchor_beta": 0.0}, "contact_compliance"),
        ({"enable_sleeping": True}, "enable_sleeping"),
    )
    for kwargs, name in options:
        with test.subTest(option=name):
            with test.assertRaisesRegex(NotImplementedError, f"{name}.*requires pgs_mode='matrix_free'"):
                SolverFeatherPGS(model, pgs_mode="split", **kwargs)
    # Contact torsion is CUDA-only; on CUDA the split solve rejects it.
    error = NotImplementedError if wp.get_device(device).is_cuda else ValueError
    with test.assertRaises(error):
        SolverFeatherPGS(model, pgs_mode="split", contact_torsion_radius=0.01)
    # The default friction resolves to point friction in the split solve.
    test.assertEqual(SolverFeatherPGS(model, pgs_mode="split").friction_anchor_beta, 0.0)
    # Positive shape restitution is rejected at construction and when notified.
    restitution = model.shape_material_restitution.numpy().copy()
    model.shape_material_restitution.fill_(0.5)
    with test.assertRaisesRegex(NotImplementedError, "restitution requires pgs_mode='matrix_free'"):
        SolverFeatherPGS(model, pgs_mode="split")
    model.shape_material_restitution.assign(restitution)
    solver = SolverFeatherPGS(model, pgs_mode="split")
    model.shape_material_restitution.fill_(0.5)
    with test.assertRaisesRegex(NotImplementedError, "restitution requires pgs_mode='matrix_free'"):
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)


def test_split_defaults(test, device):
    """Keep the documented defaults in split mode and allocate its Delassus storage."""
    solver = SolverFeatherPGS(_build_chain_model(device, num_links=2, num_worlds=2), pgs_mode="split")
    test.assertEqual(solver.pgs_mode, "split")
    test.assertEqual(solver.pgs_iterations, 12)
    test.assertEqual(solver.dense_max_constraints, 32)
    test.assertEqual(solver.C.shape, (2, 32, 32))
    test.assertFalse(solver._jy_world_aliased)
    test.assertFalse(solver._hinv_jt_writes_world)


def test_split_kernel_selection(test, device):
    """Select the native split-mode kernels on CUDA and the scalar Warp kernels on CPU."""
    is_cuda = wp.get_device(device).is_cuda
    # One 14-DOF chain per world: tiled H^-1 J^T, fused with the Delassus assembly on CUDA.
    single = SolverFeatherPGS(_build_chain_model(device, num_links=14, num_worlds=2), pgs_mode="split")
    test.assertEqual(single._split_fused_response, is_cuda)
    test.assertEqual(single._pgs_solve_tiled_row_kernel is not None, is_cuda)
    test.assertIsNone(single._pgs_solve_mf_kernel)
    # A free body next to the chain: two solved articulations per world, so no fusion.
    mixed = SolverFeatherPGS(
        _build_chain_model(device, num_links=14, num_worlds=2, with_free_body=True), pgs_mode="split"
    )
    test.assertTrue(mixed._has_mixed_contacts)
    test.assertFalse(mixed._split_fused_response)
    test.assertEqual(mixed._max_free_bodies_per_world, 1)
    for size, kernel in mixed._delassus_kernels_by_size.items():
        test.assertEqual(kernel is not None, is_cuda, f"Delassus kernel of size {size}")
    test.assertEqual(mixed._pgs_solve_mf_kernel is not None, is_cuda)
    if not is_cuda:
        test.assertEqual(
            (mixed.cholesky_kernel, mixed.trisolve_kernel, mixed.hinv_jt_kernel, mixed.pgs_kernel),
            ("loop", "loop", "par_row", "loop"),
        )


def test_split_kernel_selection_is_cached(test, device):
    """Resolve identical split-mode solver shapes to the same cached kernel objects."""
    model = _build_chain_model(device, num_links=14, num_worlds=2, with_free_body=True)
    first = SolverFeatherPGS(model, pgs_mode="split")
    second = SolverFeatherPGS(model, pgs_mode="split")
    test.assertEqual(first._delassus_kernels_by_size.keys(), second._delassus_kernels_by_size.keys())
    for size, kernel in first._delassus_kernels_by_size.items():
        test.assertIs(kernel, second._delassus_kernels_by_size[size])
    test.assertIs(first._pgs_solve_tiled_row_kernel, second._pgs_solve_tiled_row_kernel)
    test.assertIs(first._pgs_solve_mf_kernel, second._pgs_solve_mf_kernel)


def test_split_native_and_scalar_kernels_step_identically(test, device):
    """Match the native split-mode kernels and the scalar Warp kernels of the CPU path on CUDA."""
    scalar = {
        "cholesky_kernel": "loop",
        "trisolve_kernel": "loop",
        "hinv_jt_kernel": "par_row",
        "delassus_kernel": "par_row_col",
        "pgs_kernel": "loop",
    }
    for with_free_body in (False, True):
        trajectories = []
        for overrides in ({}, scalar):
            SolverFeatherPGS._kernel_overrides = overrides
            try:
                model = _build_limited_chain_on_ground(device, 14, with_free_body=with_free_body)
                solver = SolverFeatherPGS(model, pgs_mode="split", enable_joint_limits=True, dense_max_constraints=64)
            finally:
                SolverFeatherPGS._kernel_overrides = {}
            if not overrides:
                test.assertIsNotNone(solver._pgs_solve_tiled_row_kernel)
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()
            state_0, state_1 = model.state(), model.state()
            control = model.control()
            for _ in range(120):
                pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
                state_0, state_1 = state_1, state_0
            solver.check_constraint_capacity()
            test.assertGreater(int(solver.constraint_count.numpy().max()), 0)
            trajectories.append(state_0.joint_q.numpy().copy())
        with test.subTest(with_free_body=with_free_body):
            np.testing.assert_allclose(trajectories[0], trajectories[1], rtol=0.0, atol=1.0e-4)


def test_non_default_tile_threads_compiles_and_steps(test, device):
    """Step a forced tiled split solve whose row capacity is too large to fuse the Delassus assembly."""
    model = _build_chain_model(device)
    forced = {"cholesky_kernel": "tiled", "trisolve_kernel": "tiled", "hinv_jt_kernel": "tiled", "pgs_kernel": "loop"}
    SolverFeatherPGS._kernel_overrides = forced
    try:
        solver = SolverFeatherPGS(model, pgs_mode="split", dense_max_constraints=384)
    finally:
        SolverFeatherPGS._kernel_overrides = {}
    test.assertTrue(all(solver._execution_plan.use_tiled_hinv_jt(size) for size in solver.size_groups))
    test.assertFalse(any(solver._execution_plan.use_fused_hinv_jt(size) for size in solver.size_groups))
    test.assertFalse(solver._split_fused_response)
    test.assertIsNone(solver._pgs_solve_tiled_row_kernel)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    for _ in range(5):
        solver.step(state_0, state_1, control, None, 1.0 / 600.0)
        state_0, state_1 = state_1, state_0
    test.assertTrue(np.isfinite(state_0.joint_q.numpy()).all())
    test.assertTrue(np.isfinite(state_0.joint_qd.numpy()).all())


def test_articulated_contact_response_validation(test, device):
    """Accept the propagation responses, size their rows, and reject unsupported combinations."""
    model = _build_chain_model(device, num_links=2, num_worlds=1)
    solver = SolverFeatherPGS(model)
    test.assertEqual(solver.articulated_contact_response, "immediate")
    test.assertFalse(solver.propagation_same_articulation_rows)
    solver = SolverFeatherPGS(model, articulated_contact_response="propagation", dense_max_constraints=16)
    test.assertEqual(solver.articulated_contact_response, "propagation")
    # Every contact row, free-body rows included, uses the propagation family.
    test.assertEqual(solver.propagation_max_constraints, 512 + 16)
    solver = SolverFeatherPGS(model, articulated_contact_response="propagation-fused", dense_max_constraints=16)
    test.assertEqual(solver.articulated_contact_response, "propagation-fused")
    test.assertEqual(solver.propagation_max_constraints, 16)

    with test.assertRaisesRegex(ValueError, "articulated_contact_response"):
        SolverFeatherPGS(model, articulated_contact_response="bad")
    for response in ("immediate", "propagation-fused"):
        with test.subTest(response=response):
            with test.assertRaisesRegex(ValueError, "propagation_same_articulation_rows"):
                SolverFeatherPGS(model, articulated_contact_response=response, propagation_same_articulation_rows=True)

    # The fused kernel runs the tree passes of a single articulation size.
    mixed = newton.ModelBuilder()
    for num_links in (2, 3):
        chain = newton.ModelBuilder()
        joints = []
        parent = -1
        for _ in range(num_links):
            link = chain.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            joints.append(chain.add_joint_revolute(parent, link, axis=newton.Axis.Y))
            parent = link
        chain.add_articulation(joints)
        mixed.add_world(chain)
    mixed_model = mixed.finalize(device=device)
    with test.assertRaisesRegex(NotImplementedError, "propagation-fused"):
        SolverFeatherPGS(mixed_model, articulated_contact_response="propagation-fused")
    test.assertEqual(
        SolverFeatherPGS(mixed_model, articulated_contact_response="propagation").articulated_contact_response,
        "propagation",
    )


def test_propagation_rejects_unsupported_options(test, device):
    """Reject the split solve and the options the propagation rows do not implement; accept the rest."""
    model = _build_chain_model(device, num_links=2, num_worlds=1)
    for response in ("propagation", "propagation-fused"):
        with test.subTest(response=response, option="pgs_mode"):
            with test.assertRaisesRegex(NotImplementedError, "requires pgs_mode='matrix_free'"):
                SolverFeatherPGS(model, pgs_mode="split", articulated_contact_response=response)
        # The default friction selects friction patches, as with the immediate response.
        test.assertEqual(SolverFeatherPGS(model, articulated_contact_response=response).friction_anchor_beta, 0.2)
        options = (
            ({"contact_torsion_radius": 0.01}, "contact_torsion_radius"),
            ({"contact_compliance": True, "friction_anchor_beta": 0.0}, "contact_compliance"),
            ({"enable_sleeping": True}, "enable_sleeping"),
        )
        for kwargs, name in options:
            with test.subTest(response=response, option=name):
                with test.assertRaisesRegex(NotImplementedError, f"{name}.*articulated_contact_response"):
                    SolverFeatherPGS(model, articulated_contact_response=response, **kwargs)
        for kwargs in (
            {"drive_mode": "physx_pgs"},
            {"pgs_contact_regularization": 0.02},
            {"pgs_velocity_iterations": 2},
            {"pgs_warmstart": True},
        ):
            with test.subTest(response=response, **kwargs):
                solver = SolverFeatherPGS(model, articulated_contact_response=response, **kwargs)
                test.assertTrue(solver._propagation_active)
        restitution = model.shape_material_restitution.numpy().copy()
        model.shape_material_restitution.fill_(0.5)
        with test.subTest(response=response, option="restitution"):
            solver = SolverFeatherPGS(model, articulated_contact_response=response)
            solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        model.shape_material_restitution.assign(restitution)


def test_propagation_matrix_free_compiles_and_steps_with_velocity_limit_rows(test, device):
    """Step both propagation responses with dense joint velocity-limit rows."""
    model = _build_chain_model(device, num_links=3, num_worlds=1)
    model.joint_velocity_limit.assign(np.full(model.joint_dof_count, 0.1, dtype=np.float32))
    for response in ("propagation", "propagation-fused"):
        with test.subTest(response=response):
            solver = SolverFeatherPGS(
                model,
                articulated_contact_response=response,
                enable_joint_velocity_limits=True,
                pgs_iterations=2,
                dense_max_constraints=16,
                mf_max_constraints=16,
            )
            state_0, state_1 = model.state(), model.state()
            state_0.joint_qd.assign(np.full(model.joint_dof_count, 1.0, dtype=np.float32))
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            solver.step(state_0, state_1, model.control(), None, 1.0 / 600.0)
            test.assertGreater(int(solver.constraint_count.numpy()[0]), 0)
            test.assertTrue(np.isfinite(state_1.joint_q.numpy()).all())
            test.assertTrue(np.isfinite(state_1.joint_qd.numpy()).all())


class TestFeatherPGSLaunchConfig(unittest.TestCase):
    def test_cpu_construction_raises(self):
        """Reject the default CUDA-only matrix-free solve on CPU, naming the split solve, which constructs."""
        model = _build_chain_model("cpu", num_links=2, num_worlds=1)
        for options in ({}, {"pgs_mode": "matrix_free"}):
            with self.subTest(**options):
                with self.assertRaisesRegex(NotImplementedError, "requires a CUDA device; use pgs_mode='split' on CPU"):
                    SolverFeatherPGS(model, **options)
        self.assertEqual(SolverFeatherPGS(model, pgs_mode="split").pgs_mode, "split")

    def test_hinv_fusion_requires_full_working_set_to_fit(self):
        """Fuse H^-1 J^T with the Delassus assembly only when the whole row set and its Delassus tile fit."""
        plan_args = {
            "max_shared_memory": 101376,
            "cholesky_kernel": "auto",
            "hinv_jt_kernel": "auto",
            "small_dof_threshold": 12,
            "tile_threads": 64,
        }
        fitting = _FeatherPGSExecutionPlan.build([23], max_constraints=64, **plan_args)
        oversized = _FeatherPGSExecutionPlan.build([23], max_constraints=384, **plan_args)
        self.assertTrue(fitting.use_fused_hinv_jt(23))
        self.assertTrue(oversized.use_tiled_hinv_jt(23))
        self.assertFalse(oversized.use_fused_hinv_jt(23))

    def test_split_kernels_respect_shared_memory(self):
        """Stage Delassus rows in chunks that fit, and fall back when a native solve does not fit."""
        self.assertEqual(_select_delassus_chunk_size(14, 32), 32)
        self.assertEqual(_select_delassus_chunk_size(100, 384), 56)
        self.assertEqual(_select_delassus_chunk_size(30, 384), 64)
        self.assertIsNone(_select_delassus_chunk_size(6000, 32))
        self.assertLessEqual(_estimate_tiled_row_shared_memory(128), _STATIC_SHARED_MEMORY_BYTES)
        self.assertGreater(_estimate_tiled_row_shared_memory(160), _STATIC_SHARED_MEMORY_BYTES)
        self.assertLessEqual(_estimate_mf_solve_shared_memory(512, 64), _STATIC_SHARED_MEMORY_BYTES)
        self.assertGreater(_estimate_mf_solve_shared_memory(512, 2000), _STATIC_SHARED_MEMORY_BYTES)

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

    def test_mfgs_metadata_budget_counts_drive_arrays(self):
        """Count the drive-row and fused-clamp arrays in the resident metadata budget."""
        # The resident budget is 4 KiB: 128 rows hold 8 arrays (4 KiB) but not 9.
        self.assertTrue(_use_resident_mfgs_metadata(128, 64, 29, 101376))
        self.assertTrue(_use_resident_mfgs_metadata(128, 64, 29, 101376, has_drive_rows=True))
        self.assertFalse(_use_resident_mfgs_metadata(128, 64, 29, 101376, has_drive_rows=True, fuse_vel_limits=True))
        self.assertFalse(_use_resident_mfgs_metadata(256, 64, 29, 101376, has_drive_rows=True))


def test_drive_mode_validation(test, device):
    """Default to the implicit drive and reject unknown drive formulations."""
    model = _build_chain_model(device, num_links=2, num_worlds=1)
    solver = SolverFeatherPGS(model)
    test.assertEqual(solver.drive_mode, "augmented")
    test.assertFalse(solver.fuse_joint_velocity_limits)
    test.assertEqual(SolverFeatherPGS(model, pgs_mode="matrix_free", drive_mode="physx_pgs").drive_mode, "physx_pgs")
    with test.assertRaisesRegex(NotImplementedError, "requires pgs_mode='matrix_free'"):
        SolverFeatherPGS(model, pgs_mode="split", drive_mode="physx_pgs")
    with test.assertRaisesRegex(ValueError, "drive_mode"):
        SolverFeatherPGS(model, drive_mode="implicit")


def test_fuse_joint_velocity_limits_validation(test, device):
    """Engage the fused clamp only with PGS drive rows and active velocity limits, otherwise leave it inert."""
    model = _build_chain_model(device, num_links=2, num_worlds=1)
    inert = (
        {"drive_mode": "augmented"},
        {"drive_mode": "augmented", "enable_joint_velocity_limits": True},
        {"drive_mode": "physx_pgs"},
        {
            "drive_mode": "physx_pgs",
            "enable_joint_velocity_limits": True,
            "velocity_limit_activation_fraction": float("inf"),
        },
        {"drive_mode": "physx_pgs", "enable_joint_velocity_limits": True, "fuse_joint_velocity_limits": False},
    )
    for kwargs in inert:
        with test.subTest(**kwargs):
            test.assertFalse(SolverFeatherPGS(model, pgs_mode="matrix_free", **kwargs).fuse_joint_velocity_limits)
    # The applicable combination engages by default.
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", drive_mode="physx_pgs", enable_joint_velocity_limits=True)
    test.assertTrue(solver.fuse_joint_velocity_limits)


def test_fuse_joint_velocity_limits_clamps_driven_dofs_without_rows(test, device):
    """Hold the velocity limit of driven DOFs with the fused clamp and two fewer rows per driven DOF."""
    num_links = 3
    qdot_max = 1.0

    def run(fuse):
        model = _build_chain_model(device, num_links=num_links, num_worlds=1)
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
            solver.step(state_0, state_1, control, None, 1.0 / 60.0)
            state_0, state_1 = state_1, state_0
            max_speed = max(max_speed, float(np.max(np.abs(state_0.joint_qd.numpy()))))
        return max_speed, int(solver.constraint_count.numpy()[0])

    fused_speed, fused_rows = run(True)
    dedicated_speed, dedicated_rows = run(False)
    # Both formulations are stateless passes at the end of each iteration and leave the
    # same small Gauss-Seidel residual above the limit on a coupled chain.
    test.assertLessEqual(fused_speed, dedicated_speed * 1.05)
    test.assertLessEqual(fused_speed, qdot_max * 1.25)
    # The fused clamp drops the two velocity-limit rows of every driven DOF; the drive and
    # position-limit rows remain.
    test.assertGreaterEqual(fused_rows, num_links)
    test.assertEqual(dedicated_rows - fused_rows, 2 * num_links)


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
    ("test_drive_mode_validation", test_drive_mode_validation),
    ("test_fuse_joint_velocity_limits_validation", test_fuse_joint_velocity_limits_validation),
    (
        "test_fuse_joint_velocity_limits_clamps_driven_dofs_without_rows",
        test_fuse_joint_velocity_limits_clamps_driven_dofs_without_rows,
    ),
    ("test_articulated_contact_response_validation", test_articulated_contact_response_validation),
    ("test_propagation_rejects_unsupported_options", test_propagation_rejects_unsupported_options),
    (
        "test_propagation_matrix_free_compiles_and_steps_with_velocity_limit_rows",
        test_propagation_matrix_free_compiles_and_steps_with_velocity_limit_rows,
    ),
):
    add_function_test(TestFeatherPGSLaunchConfig, _name, _func, devices=devices)
split_devices = get_test_devices()
for _name, _func, _devices in (
    (
        "test_joint_limit_solver_compatibility_validation",
        test_joint_limit_solver_compatibility_validation,
        split_devices,
    ),
    (
        "test_split_rejects_matrix_free_only_options",
        test_split_rejects_matrix_free_only_options,
        split_devices,
    ),
    ("test_split_defaults", test_split_defaults, split_devices),
    ("test_split_kernel_selection", test_split_kernel_selection, split_devices),
    ("test_split_kernel_selection_is_cached", test_split_kernel_selection_is_cached, split_devices),
    (
        "test_split_native_and_scalar_kernels_step_identically",
        test_split_native_and_scalar_kernels_step_identically,
        devices,
    ),
    ("test_non_default_tile_threads_compiles_and_steps", test_non_default_tile_threads_compiles_and_steps, devices),
):
    add_function_test(TestFeatherPGSLaunchConfig, _name, _func, devices=_devices)


if __name__ == "__main__":
    unittest.main()
