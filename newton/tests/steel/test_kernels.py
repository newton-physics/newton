# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kernel selection, launch configuration, streams and double buffering of SolverSteel."""

import itertools
import types
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel import kernels as steel_kernels
from newton._src.solvers.steel import solver_steel
from newton._src.solvers.steel._testing import kernel_overrides
from newton._src.solvers.steel.solver_steel import (
    _DENSE_META_MAX_PARENT,
    _estimate_cholesky_shared_memory,
    _get_pgs_solve_mf_gs_kernel,
    _select_hinv_jt_chunk_size,
    _SteelExecutionPlan,
    _use_mfgs_pair_cache,
    _use_resident_mfgs_metadata,
)
from newton.solvers import SolverSteel
from newton.tests.steel._fixtures import (
    _build_contact_scene,
    _build_four_bar,
    _run,
    _set_box_velocities,
    _trajectory,
    _tree,
    set_torsion_radius,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices, patch_sys_module


def _assert_bitwise(test, a, b, label):
    np.testing.assert_array_equal(a[1], b[1], err_msg=f"{label}: joint_q")
    np.testing.assert_array_equal(a[2], b[2], err_msg=f"{label}: joint_qd")


# Dense row capacity of the stepping tests, shared so that their kernel specializations compile once.
_ROWS = 64


def test_default_launch_settings(test, device):
    """Select the kernels and launch widths automatically, matching explicitly pinned defaults bitwise."""
    solver = SolverSteel(_build_contact_scene(device))
    test.assertEqual((solver._tile_threads, solver._serial_kernel_block_dim), (64, 256))
    run = {"init": _set_box_velocities, "steps": 8}
    explicit = {"tile_threads": 64, "serial_kernel_block_dim": 256}
    _assert_bitwise(
        test,
        _run(_build_contact_scene(device), **run),
        _run(_build_contact_scene(device), overrides=explicit, **run),
        "defaults",
    )


def test_kernel_overrides_are_scoped_to_construction(test, device):
    """Apply overrides to solvers built inside the block only, and keep them on that instance."""
    model = _build_contact_scene(device)
    with kernel_overrides({"tile_threads": 32, "serial_kernel_block_dim": 64}):
        pinned = SolverSteel(model)
        with kernel_overrides({"tile_threads": 128}):
            nested = SolverSteel(model)
    default = SolverSteel(model)
    test.assertEqual((pinned._tile_threads, pinned._serial_kernel_block_dim), (32, 64))
    test.assertEqual((nested._tile_threads, nested._serial_kernel_block_dim), (128, 256))
    test.assertEqual((default._tile_threads, default._serial_kernel_block_dim), (64, 256))
    test.assertEqual(pinned._kernel_overrides, {"tile_threads": 32, "serial_kernel_block_dim": 64})
    test.assertEqual(default._kernel_overrides, {})


def test_serial_block_dim_is_bitwise(test, device):
    """Leave matrix-free results unchanged by the serial block size."""
    run = {"init": _set_box_velocities, "steps": 8}
    default = _run(_build_contact_scene(device), **run)
    for overrides in ({"serial_kernel_block_dim": 64}, {"serial_kernel_block_dim": 512}):
        with test.subTest(overrides=overrides):
            result = _run(_build_contact_scene(device), overrides=overrides, **run)
            _assert_bitwise(test, default, result, str(overrides))


def test_kernel_overrides_validate_their_values(test, device):
    """Reject unsupported launch widths."""
    model = _build_contact_scene(device)
    for overrides, pattern in (
        ({"tile_threads": 48}, "tile_threads"),
        ({"serial_kernel_block_dim": 0}, "serial_kernel_block_dim"),
        ({"serial_kernel_block_dim": 48}, "serial_kernel_block_dim"),
    ):
        with test.subTest(overrides=overrides), test.assertRaisesRegex(ValueError, pattern):
            with kernel_overrides(overrides):
                SolverSteel(model)


class _FakeNvtx(types.ModuleType):
    """Record NVTX ranges in place of the nvtx package."""

    def __init__(self):
        super().__init__("nvtx")
        self.events = []

    def start_range(self, message, color=None):
        self.events.append(("start", message))
        return len(self.events)

    def end_range(self, range_id):
        self.events.append(("end", range_id))


def test_profile_annotates_every_stage_without_changing_results(test, device):
    """Open and close one NVTX range per step stage under NEWTON_PROFILE, leaving the results bitwise unchanged."""
    fake = _FakeNvtx()
    run = {"init": _set_box_velocities, "steps": 3}
    with patch_sys_module("nvtx", fake), mock.patch.object(solver_steel, "PROFILE_ENABLED", True):
        annotated = _run(_build_contact_scene(device), **run)
    _assert_bitwise(test, annotated, _run(_build_contact_scene(device), **run), "profile")
    stages = [message for kind, message in fake.events if kind == "start"]
    expected = [
        "Steel::fk_id_crba",
        "Steel::factor",
        "Steel::unconstrained_velocity",
        "Steel::rows",
        "Steel::solve",
        "Steel::integrate",
    ]
    test.assertEqual(stages, expected * 3)
    # Each range closes before the next one opens, and the last one closes at the end of the step.
    test.assertEqual([kind for kind, _ in fake.events], ["start", "end"] * len(stages))
    for index in range(0, len(fake.events), 2):
        test.assertEqual(fake.events[index + 1][1], index + 1)


def test_profile_closes_the_open_range_when_a_stage_raises(test, device):
    """Close the open stage range when a step raises."""
    fake = _FakeNvtx()
    model = _build_contact_scene(device)
    solver = SolverSteel(model)
    state_in, state_out = model.state(), model.state()
    with (
        patch_sys_module("nvtx", fake),
        mock.patch.object(solver_steel, "PROFILE_ENABLED", True),
        mock.patch.object(solver, "_stage1_crba", side_effect=RuntimeError("injected")),
        test.assertRaisesRegex(RuntimeError, "injected"),
    ):
        solver.step(state_in, state_out, model.control(), None, 1.0 / 240.0)
    test.assertEqual([kind for kind, _ in fake.events], ["start", "end"])
    test.assertIsNone(solver._profile_timer)


class TestSteelAdvancedOptions(unittest.TestCase):
    pass


for _fn in (
    test_default_launch_settings,
    test_kernel_overrides_are_scoped_to_construction,
    test_kernel_overrides_validate_their_values,
    test_profile_annotates_every_stage_without_changing_results,
    test_profile_closes_the_open_range_when_a_stage_raises,
):
    add_function_test(TestSteelAdvancedOptions, _fn.__name__, _fn, devices=get_test_devices())


add_function_test(
    TestSteelAdvancedOptions,
    test_serial_block_dim_is_bitwise.__name__,
    test_serial_block_dim_is_bitwise,
    devices=get_cuda_test_devices(),
)


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
                target_q=0.3,
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
    solver = SolverSteel(_build_chain_model(device, num_links=2, num_worlds=1))
    test.assertEqual(solver.solve_mode, "serial")
    test.assertEqual(solver.articulated_contact_response, "immediate")
    test.assertEqual(solver.iterations, 12)
    test.assertAlmostEqual(solver.baumgarte, 0.2)
    test.assertAlmostEqual(solver.cfm, 1.0e-6)
    test.assertEqual(solver.relaxation, 1.0)
    test.assertEqual(solver.update_mass_matrix_interval, 1)
    test.assertFalse(solver.enable_joint_limits)
    test.assertEqual(solver.joint_limit_activation_gap, float("inf"))
    test.assertFalse(solver.enable_joint_velocity_limits)
    test.assertEqual(solver.velocity_limit_activation_fraction, 0.0)
    test.assertEqual(solver.articulation_constraint_max, 32)
    test.assertEqual(solver.free_body_constraint_max, 512)
    test.assertTrue(solver.warn_constraint_overflow)
    np.testing.assert_allclose(solver._rigid_body_angular_damping.numpy(), 0.05)


def test_constructor_validates_options(test, device):
    """Reject option values that cannot be honored."""
    model = _build_chain_model(device, num_links=2, num_worlds=1)
    invalid = (
        ({"update_mass_matrix_interval": 0}, "update_mass_matrix_interval"),
        ({"iterations": -1}, "iterations"),
        ({"joint_limit_activation_gap": -0.1}, "joint_limit_activation_gap"),
        ({"joint_limit_activation_gap": float("nan")}, "joint_limit_activation_gap"),
        ({"velocity_limit_activation_fraction": -0.1}, "velocity_limit_activation_fraction"),
        ({"velocity_limit_activation_fraction": 1.5}, "velocity_limit_activation_fraction"),
        ({"velocity_limit_activation_fraction": float("nan")}, "velocity_limit_activation_fraction"),
        ({"articulation_constraint_max": 0}, "articulation_constraint_max"),
        ({"articulation_constraint_max": _DENSE_META_MAX_PARENT + 2}, "articulation_constraint_max"),
        ({"free_body_constraint_max": 0}, "free_body_constraint_max"),
    )
    for kwargs, message in invalid:
        with test.subTest(**kwargs):
            with test.assertRaisesRegex(ValueError, message):
                SolverSteel(model, solve_mode="serial", **kwargs)
    test.assertEqual(
        SolverSteel(
            model, solve_mode="serial", velocity_limit_activation_fraction=float("inf")
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
    # Quaternion-coordinate mimics have no componentwise row. Bypass the authoring guard to
    # exercise defensive solver handling of unsupported metadata supplied directly by callers.
    mimic.joint_mimic_joint[follower] = leader
    with test.assertRaisesRegex(NotImplementedError, "mimic"):
        SolverSteel(mimic.finalize(device=device))

    particles = newton.ModelBuilder()
    body = particles.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    particles.add_articulation([particles.add_joint_revolute(-1, body, axis=newton.Axis.Y)])
    particles.add_particle(wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0), mass=1.0)
    with test.assertRaisesRegex(NotImplementedError, "particles"):
        SolverSteel(particles.finalize(device=device))

    disabled = newton.ModelBuilder()
    body = disabled.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    disabled.add_articulation([disabled.add_joint_revolute(-1, body, axis=newton.Axis.Y, enabled=False)])
    with test.assertRaisesRegex(NotImplementedError, "disabled joints"):
        SolverSteel(disabled.finalize(device=device))


def test_default_kernel_selection_is_cached(test, device):
    """Resolve identical solver shapes to the same cached kernel objects."""
    model = _build_chain_model(device)
    first = SolverSteel(model, solve_mode="serial")
    second = SolverSteel(model, solve_mode="serial")
    for attr in ("_cholesky_kernels_by_size", "_triangular_solve_kernels_by_size", "_hinv_jt_kernels_by_size"):
        first_kernels = getattr(first, attr)
        second_kernels = getattr(second, attr)
        test.assertEqual(set(first_kernels), set(second_kernels))
        for size, kernel in first_kernels.items():
            test.assertIs(kernel, second_kernels[size], f"{attr}[{size}]")
    test.assertIs(first._pgs_solve_mf_gs_kernel, second._pgs_solve_mf_gs_kernel)


def test_compact_world_dof_mapping_pads_heterogeneous_worlds(test, device):
    """Pad the per-world response DOF map of worlds with fewer DOFs."""
    solver = SolverSteel(_build_heterogeneous_world_model(device), solve_mode="serial")
    test.assertEqual(solver._max_world_dofs, 6)
    np.testing.assert_array_equal(solver._world_dof_count.numpy(), np.array((6, 1), dtype=np.int32))
    indices = solver._world_dof_indices.numpy()
    np.testing.assert_array_equal(indices[0], np.arange(6, dtype=np.int32))
    test.assertGreaterEqual(int(indices[1, 0]), 0)
    np.testing.assert_array_equal(indices[1, 1:], np.full(5, -1, dtype=np.int32))


def test_diagonal_fusion_requires_nonaliased_world_response(test, device):
    """Compute the row diagonal in H^-1 J^T only when it writes separate world storage."""
    aliased = SolverSteel(
        _build_chain_model(device, num_links=23, num_worlds=1),
        solve_mode="serial",
        articulation_constraint_max=192,
    )
    direct = SolverSteel(
        _build_chain_model(device, num_links=23, num_worlds=1, with_free_body=True),
        solve_mode="serial",
        articulation_constraint_max=192,
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
    for overrides in ({}, {"cholesky_kernel": "loop", "trisolve_kernel": "loop", "hinv_jt_kernel": "par_row"}):
        model = _build_chain_model(device, num_links=14, num_worlds=2)
        with kernel_overrides(overrides):
            solver = SolverSteel(model, solve_mode="serial", articulation_constraint_max=64)
        if not overrides:
            test.assertTrue(solver._execution_plan.use_tiled_cholesky(14))
            test.assertTrue(solver._execution_plan.use_tiled_hinv_jt(14))
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        for _ in range(10):
            solver.step(state_0, state_1, control, None, 1.0 / 120.0)
            state_0, state_1 = state_1, state_0
        trajectories.append(state_0.joint_q.numpy().copy())
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
        SolverSteel(_build_box_with_equality(device, enabled=True))
    # A link to a projected entity that does not exist does not make the row enforced.
    for target_kind in (1, 2):
        with test.assertRaisesRegex(NotImplementedError, "equality"):
            SolverSteel(_build_box_with_equality(device, enabled=True, target_kind=target_kind, target=7))

    # Imported equalities are converted to Newton loop joints by default; those rows are
    # enforced through the loop joint (see the loop-closure tests). Unconverted, they raise.
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
        SolverSteel(imported)

    # A disabled row constructs and has no effect.
    model = _build_box_with_equality(device, enabled=False)
    solver = SolverSteel(model)
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
        SolverSteel(_build_box_with_equality(device, enabled=True, target_kind=1, target=0))

    def build_pendulum_with_connect(eq_type: int, joint_kind: str, row_names_world: bool, torquescale: float = 1.0):
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
                "mujoco:equality_constraint_torquescale": torquescale,
            }
        )
        return builder.finalize(device=device)

    # Mismatched projections: a WELD row linking a ball joint, a rotation-free WELD row linking a
    # fixed joint, and a row whose bodies are not the linked joint's endpoints. Each is an
    # unenforced equality, not a loop joint.
    for eq_type, joint_kind, row_names_world, torquescale in (
        (1, "ball", False, 1.0),
        (1, "fixed", False, 0.0),
        (0, "ball", True, 1.0),
    ):
        with test.subTest(eq_type=eq_type, joint_kind=joint_kind, row_names_world=row_names_world):
            with test.assertRaisesRegex(NotImplementedError, "equality"):
                SolverSteel(build_pendulum_with_connect(eq_type, joint_kind, row_names_world, torquescale))
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
            SolverSteel(builder.finalize(device=device))
    # The matching projection is judged through its loop joint instead. A WELD with torquescale 0
    # is rotation free, so its projection is a ball joint.
    for eq_type, joint_kind, torquescale in ((0, "ball", 1.0), (1, "fixed", 1.0), (1, "ball", 0.0)):
        with test.subTest(eq_type=eq_type, joint_kind=joint_kind, torquescale=torquescale, matching=True):
            with test.assertRaisesRegex(NotImplementedError, "loop-closing"):
                SolverSteel(build_pendulum_with_connect(eq_type, joint_kind, False, torquescale))

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
                    SolverSteel(model, solve_mode="serial")
            else:
                SolverSteel(model, solve_mode="serial")


def test_enabling_equality_constraint_at_runtime_raises(test, device):
    """Re-check the MuJoCo equality rows when constraint properties change."""
    model = _build_box_with_equality(device, enabled=False)
    solver = SolverSteel(model)
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
    """Accept joint velocity limits in every solve mode and reject unknown solve modes."""
    model = _build_chain_model(device, num_links=2, num_worlds=1)
    test.assertTrue(SolverSteel(model, enable_joint_velocity_limits=True).enable_joint_velocity_limits)
    with test.assertRaisesRegex(ValueError, "solve_mode"):
        SolverSteel(model, solve_mode="dense")


def test_articulated_contact_response_validation(test, device):
    """Accept the propagation responses, size their rows, and reject unsupported combinations."""
    model = _build_chain_model(device, num_links=2, num_worlds=1)
    solver = SolverSteel(model)
    test.assertEqual(solver.articulated_contact_response, "immediate")
    test.assertFalse(solver.propagation_same_articulation_rows)
    solver = SolverSteel(model, articulated_contact_response="propagation", articulation_constraint_max=16)
    test.assertEqual(solver.articulated_contact_response, "propagation")
    # Every contact row, free-body rows included, uses the propagation family.
    test.assertEqual(solver._propagation_max_constraints, 512 + 16)

    for response in ("bad", "propagation_fused"):
        with test.subTest(response=response):
            with test.assertRaisesRegex(ValueError, f"articulated_contact_response must be .*got '{response}'"):
                SolverSteel(model, articulated_contact_response=response)
    with test.assertRaisesRegex(ValueError, "propagation_same_articulation_rows"):
        SolverSteel(model, articulated_contact_response="immediate", propagation_same_articulation_rows=True)

    # The propagation responses solve trees of different articulation sizes in one model.
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
    for solve_mode in ("serial", "colored"):
        with test.subTest(solve_mode=solve_mode):
            solver = SolverSteel(mixed_model, solve_mode=solve_mode, articulated_contact_response="propagation")
            test.assertEqual((solver.solve_mode, solver.articulated_contact_response), (solve_mode, "propagation"))


def test_propagation_rejects_unsupported_options(test, device):
    """Reject the options the propagation rows do not implement; accept the rest."""
    model = _build_chain_model(device, num_links=2, num_worlds=1)
    for response in ("serial", "colored"):
        propagation = {"solve_mode": response, "articulated_contact_response": "propagation"}
        # The default friction selects friction patches, as with the immediate response.
        test.assertEqual(SolverSteel(model, **propagation).friction_anchor_beta, 0.2)
        set_torsion_radius(model, 0.01)
        with test.subTest(response=response, option="contact torsion"):
            with test.assertRaisesRegex(NotImplementedError, "contact torsion.*articulated_contact_response"):
                SolverSteel(model, **propagation)
        set_torsion_radius(model, 0.0)
        compliance = {"contact_compliance": True, "friction_anchor_beta": 0.0}
        options = []
        if response == "serial":
            options.append(({"enable_sleeping": True}, "enable_sleeping"))
        else:
            with test.subTest(response=response, option="enable_sleeping"):
                test.assertIsNotNone(SolverSteel(model, enable_sleeping=True, **propagation)._sleeping)
        accepted = [
            {"drive_mode": "rows"},
            {"contact_regularization": 0.02},
            {"velocity_iterations": 2},
            {"contact_warmstart": True},
        ]
        if response == "serial":
            options.append((compliance, "contact_compliance"))
        else:
            accepted.append(compliance)
        for kwargs, name in options:
            with test.subTest(response=response, option=name):
                with test.assertRaisesRegex(NotImplementedError, f"{name}.*articulated_contact_response"):
                    SolverSteel(model, **propagation, **kwargs)
        for kwargs in accepted:
            with test.subTest(response=response, **kwargs):
                solver = SolverSteel(model, **propagation, **kwargs)
                test.assertTrue(solver._propagation_active)
        restitution = model.shape_material_restitution.numpy().copy()
        model.shape_material_restitution.fill_(0.5)
        with test.subTest(response=response, option="restitution"):
            solver = SolverSteel(model, **propagation)
            solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        model.shape_material_restitution.assign(restitution)


def test_propagation_compiles_and_steps_with_velocity_limit_rows(test, device):
    """Step the propagation response in both solve modes with dense joint velocity-limit rows."""
    model = _build_chain_model(device, num_links=3, num_worlds=1)
    model.joint_velocity_limit.assign(np.full(model.joint_dof_count, 0.1, dtype=np.float32))
    for solve_mode in ("serial", "colored"):
        with test.subTest(solve_mode=solve_mode):
            solver = SolverSteel(
                model,
                solve_mode=solve_mode,
                articulated_contact_response="propagation",
                enable_joint_velocity_limits=True,
                iterations=2,
                articulation_constraint_max=16,
                free_body_constraint_max=16,
            )
            state_0, state_1 = model.state(), model.state()
            state_0.joint_qd.assign(np.full(model.joint_dof_count, 1.0, dtype=np.float32))
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            solver.step(state_0, state_1, model.control(), None, 1.0 / 600.0)
            test.assertGreater(int(solver._constraint_count.numpy()[0]), 0)
            test.assertTrue(np.isfinite(state_1.joint_q.numpy()).all())
            test.assertTrue(np.isfinite(state_1.joint_qd.numpy()).all())


class TestSteelLaunchConfig(unittest.TestCase):
    def test_cpu_runs_the_matrix_free_solve(self):
        """Construct the default solve on CPU with its portable sweep and the scalar factor kernels."""
        model = _build_chain_model("cpu", num_links=2, num_worlds=1, with_free_body=True)
        solver = SolverSteel(model)
        self.assertEqual(solver.solve_mode, "serial")
        self.assertIs(solver._pgs_solve_mf_gs_kernel, solver_steel.pgs_solve_mf_gs_cpu)
        self.assertIsNone(solver._pack_mf_meta_kernel)
        self.assertEqual(
            (solver._cholesky_kernel, solver._trisolve_kernel, solver._hinv_jt_kernel), ("loop", "loop", "par_row")
        )
        with self.assertWarnsRegex(newton.exceptions.NewtonWarning, "serial propagation sweep on CPU"):
            colored = SolverSteel(model, solve_mode="colored", articulated_contact_response="propagation")
        self.assertEqual(colored.solve_mode, "colored")
        self.assertFalse(colored._propagation_colored)

    def test_launch_geometry_kernels_use_dedicated_modules(self):
        """Keep custom-block-dimension kernels out of the general module."""
        expected_modules = {
            "eval_rigid_fk_id": "kinematics",
            "eval_rigid_tau": "inverse_dynamics",
            "eval_rigid_tau_and_augmented_drives": "inverse_dynamics",
            "compute_composite_inertia": "mass_dynamics",
            "crba_fill_par_dof": "mass_dynamics",
        }
        general_module = steel_kernels.__name__
        for kernel_name, module_suffix in expected_modules.items():
            with self.subTest(kernel=kernel_name):
                kernel_module = getattr(steel_kernels, kernel_name).module.name
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
        fallback = _SteelExecutionPlan.build(
            [160], max_constraints=384, hinv_jt_kernel="auto", tile_threads=64, **plan_args
        )
        self.assertFalse(fallback.use_tiled_hinv_jt(160))
        zero_capacity = _SteelExecutionPlan.build(
            [23], max_constraints=0, hinv_jt_kernel="tiled", tile_threads=64, **plan_args
        )
        self.assertFalse(zero_capacity.use_tiled_hinv_jt(23))
        with self.assertRaisesRegex(ValueError, "hinv_jt_kernel='tiled'"):
            _SteelExecutionPlan.build([160], max_constraints=384, hinv_jt_kernel="tiled", tile_threads=64, **plan_args)

    def test_cholesky_execution_plan_respects_shared_memory(self):
        """Fall back for oversized auto tiles and reject forced invalid launches."""
        self.assertEqual(_estimate_cholesky_shared_memory(108), 140400)
        automatic = _SteelExecutionPlan.build(
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
            _SteelExecutionPlan.build(
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
    solver = SolverSteel(model)
    test.assertEqual(solver.drive_mode, "augmented")
    test.assertFalse(solver.velocity_limits_fused)
    test.assertEqual(SolverSteel(model, drive_mode="rows").drive_mode, "rows")
    with test.assertRaisesRegex(ValueError, "drive_mode"):
        SolverSteel(model, drive_mode="implicit")


def test_fuse_joint_velocity_limits_validation(test, device):
    """Engage the fused clamp only with PGS drive rows and active velocity limits, otherwise leave it inert."""
    model = _build_chain_model(device, num_links=2, num_worlds=1)
    inert = (
        {"drive_mode": "augmented"},
        {"drive_mode": "augmented", "enable_joint_velocity_limits": True},
        {"drive_mode": "rows"},
        {
            "drive_mode": "rows",
            "enable_joint_velocity_limits": True,
            "velocity_limit_activation_fraction": float("inf"),
        },
        {"drive_mode": "rows", "enable_joint_velocity_limits": True, "fuse_joint_velocity_limits": False},
    )
    for kwargs in inert:
        with test.subTest(**kwargs):
            test.assertFalse(SolverSteel(model, solve_mode="serial", **kwargs).velocity_limits_fused)
    # The applicable combination engages by default.
    solver = SolverSteel(model, solve_mode="serial", drive_mode="rows", enable_joint_velocity_limits=True)
    test.assertTrue(solver.velocity_limits_fused)


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
        solver = SolverSteel(
            model,
            solve_mode="serial",
            drive_mode="rows",
            enable_joint_velocity_limits=True,
            fuse_joint_velocity_limits=fuse,
            iterations=64,
            articulation_constraint_max=16,
            free_body_constraint_max=16,
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
        return max_speed, int(solver._constraint_count.numpy()[0])

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
        "test_propagation_compiles_and_steps_with_velocity_limit_rows",
        test_propagation_compiles_and_steps_with_velocity_limit_rows,
    ),
):
    add_function_test(TestSteelLaunchConfig, _name, _func, devices=devices)


add_function_test(
    TestSteelLaunchConfig,
    "test_joint_limit_solver_compatibility_validation",
    test_joint_limit_solver_compatibility_validation,
    devices=get_test_devices(),
)


def _overrides(**overrides):
    return kernel_overrides(overrides)


def _legged_base(device, worlds=2):
    """A floating base with four hinged legs dropped on the ground: articulated (dense) contact rows only."""
    template = newton.ModelBuilder()
    template.default_shape_cfg.mu = 0.6
    root = template.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.25), wp.quat_rpy(0.05, 0.02, 0.0)))
    template.add_shape_box(root, hx=0.2, hy=0.12, hz=0.04)
    joints = [template.add_joint_free(root)]
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            leg = template.add_link()
            template.add_shape_capsule(leg, radius=0.03, half_height=0.1)
            joints.append(
                template.add_joint_revolute(
                    root,
                    leg,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    parent_xform=wp.transform(wp.vec3(0.18 * sx, 0.1 * sy, -0.03), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.13), wp.quat_identity()),
                )
            )
    template.add_articulation(joints)
    builder = newton.ModelBuilder()
    builder.replicate(template, worlds)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _legged_trajectory(model, steps=90, **solver_kwargs):
    solver = SolverSteel(model, **solver_kwargs)
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    qd = np.zeros(model.joint_dof_count, dtype=np.float32)
    qd[0:6:2] = 0.4
    state_0.joint_qd.assign(qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control = model.control()
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    solver.check_constraint_capacity()
    return solver, state_0.joint_q.numpy().copy()


def test_launch_widths_step_identically(test, device, solve_mode="serial"):
    """Step identically with non-default tile and serial-kernel launch widths."""
    model = _legged_base(device)
    options = {"solve_mode": solve_mode, "articulation_constraint_max": _ROWS, "friction_anchor_beta": 0.0}
    with _overrides(sparse_mass_matrix=False, cholesky_kernel="tiled", trisolve_kernel="tiled", hinv_jt_kernel="tiled"):
        default, reference = _legged_trajectory(model, **options)
    with _overrides(
        cholesky_kernel="tiled",
        trisolve_kernel="tiled",
        hinv_jt_kernel="tiled",
        tile_threads=128,
        serial_kernel_block_dim=64,
    ):
        solver, joint_q = _legged_trajectory(model, **options)
    test.assertEqual((default._tile_threads, default._serial_kernel_block_dim), (64, 256))
    test.assertEqual((solver._tile_threads, solver._serial_kernel_block_dim), (128, 64))
    size = solver._size_groups[0]
    test.assertIsNot(solver._cholesky_kernels_by_size[size], default._cholesky_kernels_by_size[size])
    np.testing.assert_allclose(joint_q, reference, rtol=0.0, atol=1.0e-4)


class TestSteelKernelOverrides(unittest.TestCase):
    pass


cuda_devices = get_cuda_test_devices()


add_function_test(
    TestSteelKernelOverrides,
    test_launch_widths_step_identically.__name__,
    test_launch_widths_step_identically,
    devices=cuda_devices,
)


def _scene(device, worlds=2):
    """Two articulation sizes (a 2-link and a 3-link chain) and a box, falling onto the ground."""
    template = newton.ModelBuilder()
    for x, links in ((0.0, 2), (0.8, 3)):
        parent = -1
        joints = []
        for i in range(links):
            link = template.add_link(xform=wp.transform(wp.vec3(x + 0.25 * i, 0.0, 0.3), wp.quat_identity()))
            template.add_shape_capsule(
                link,
                radius=0.04,
                half_height=0.1,
                xform=wp.transform(wp.vec3(), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * np.pi)),
            )
            joints.append(
                template.add_joint_revolute(
                    parent,
                    link,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    parent_xform=wp.transform(
                        wp.vec3(x, 0.0, 0.3) if parent < 0 else wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()
                    ),
                    child_xform=wp.transform_identity(),
                    limit_lower=-1.0,
                    limit_upper=1.0,
                )
            )
            parent = link
        template.add_articulation(joints)
    box = template.add_body(xform=wp.transform(wp.vec3(0.4, 0.4, 0.3), wp.quat_rpy(0.2, 0.1, 0.0)))
    template.add_shape_box(box, hx=0.06, hy=0.06, hz=0.06)
    builder = newton.ModelBuilder()
    builder.replicate(template, worlds)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _stream_trajectory(model, steps=60, capture=False, **solver_kwargs):
    solver = SolverSteel(
        model, enable_joint_limits=True, friction_anchor_beta=0.0, articulation_constraint_max=96, **solver_kwargs
    )
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control = model.control()

    def substep():
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            wp.copy(getattr(state_0, name), getattr(state_1, name))

    if capture:
        # A graph of two steps cycles both buffer sets.
        with wp.ScopedCapture(model.device) as scope:
            substep()
            substep()
        for _ in range(steps // 2):
            wp.capture_launch(scope.graph)
    else:
        for _ in range(steps):
            substep()
    solver.check_constraint_capacity()
    return solver, state_0.joint_q.numpy().copy()


def test_streams_and_double_buffer_step_identically(test, device, solve_mode="serial"):
    """Match the serial single-buffer trajectory bitwise with streams and double buffering."""
    model = _scene(device)
    reference_solver, reference = _stream_trajectory(model, solve_mode=solve_mode)
    test.assertEqual(reference_solver._size_streams, {})
    test.assertFalse(reference_solver._double_buffered)
    test.assertGreater(len(reference_solver._size_groups), 1)
    for options in (
        {"use_parallel_streams": True},
        {"double_buffer": True},
        {"use_parallel_streams": True, "double_buffer": True},
        {"use_parallel_streams": True, "double_buffer": True, "update_mass_matrix_interval": 3},
    ):
        with test.subTest(**options):
            solver, joint_q = _stream_trajectory(model, solve_mode=solve_mode, **options)
            test.assertEqual(bool(solver._size_streams), options.get("use_parallel_streams", False))
            test.assertEqual(solver._double_buffered, options.get("double_buffer", False))
            if solver._double_buffered:
                # Both buffer sets were cleared on the memset stream after their last use.
                wp.synchronize_device(device)
                used = 1 - solver._buffer_index
                test.assertIs(solver._J_bufs[used], solver._J_by_size)
                for index in (0, 1):
                    for size in solver._size_groups:
                        test.assertFalse(solver._J_bufs[index][size].numpy().any())
                        if "update_mass_matrix_interval" not in options:
                            test.assertFalse(solver._H_bufs[index][size].numpy().any())
            expected = reference
            if "update_mass_matrix_interval" in options:
                _, expected = _stream_trajectory(model, solve_mode=solve_mode, update_mass_matrix_interval=3)
            np.testing.assert_array_equal(joint_q, expected)


def test_captured_double_buffer_matches_eager(test, device, solve_mode="serial"):
    """Replay a two-step graph of the double-buffered solve with the eager result."""
    model = _scene(device)
    _, eager = _stream_trajectory(model, solve_mode=solve_mode, use_parallel_streams=True, double_buffer=True)
    _, captured = _stream_trajectory(
        model, solve_mode=solve_mode, use_parallel_streams=True, double_buffer=True, capture=True
    )
    np.testing.assert_array_equal(captured, eager)


@wp.kernel
def _spin(iterations: int, sink: wp.array[float]):
    value = float(0.0)
    for i in range(iterations):
        value = wp.sin(value + float(i))
    sink[0] = value


def _run_sequence(model, plan, *, delay_clear=False, **solver_kwargs):
    """Run ``plan`` and return ``joint_q`` after each operation.

    Operations: ``("eager", n)``, ``("capture", name, n)`` (records ``n`` steps without
    running them), ``("capture_paused", name, n)`` (the same with a conditional node
    between steps, which pauses the capture), ``("replay", name, n)`` and ``("reset",)``.
    With ``delay_clear`` the double-buffer clearing of the last eager step before each
    capture first waits for a long kernel on its stream, so it is still pending when the
    graph is first launched.
    """
    solver = SolverSteel(model, friction_anchor_beta=0.0, articulation_constraint_max=96, **solver_kwargs)
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control = model.control()
    sink = wp.zeros(1, dtype=float, device=model.device)
    condition = wp.ones(1, dtype=wp.int32, device=model.device)

    def substep():
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            wp.copy(getattr(state_0, name), getattr(state_1, name))

    graphs = {}
    results = []
    for index, (operation, *args) in enumerate(plan):
        if operation == "eager":
            for step in range(args[0]):
                last_before_capture = step == args[0] - 1 and index + 1 < len(plan) and plan[index + 1][0] == "capture"
                if delay_clear and solver._double_buffered and last_before_capture:
                    with wp.ScopedStream(solver._memset_stream, sync_enter=False):
                        wp.launch(_spin, dim=1, inputs=[20_000_000], outputs=[sink])
                substep()
        elif operation in ("capture", "capture_paused"):
            with wp.ScopedCapture(model.device) as scope:
                for step in range(args[1]):
                    if operation == "capture_paused" and step > 0:
                        wp.capture_if(condition, on_true=lambda: condition.fill_(1))
                    substep()
            graphs[args[0]] = scope.graph
        elif operation == "replay":
            for _ in range(args[1]):
                wp.capture_launch(graphs[args[0]])
        else:
            solver.reset(state_0)
        results.append(state_0.joint_q.numpy().copy())
    solver.check_constraint_capacity()
    return solver, results


_SEQUENCES = {
    "cold capture": [("capture", "a", 2), ("replay", "a", 6)],
    "eager then capture": [("eager", 3), ("capture", "a", 2), ("replay", "a", 6), ("eager", 2)],
    "odd steps per graph": [("eager", 2), ("capture", "a", 3), ("replay", "a", 5), ("eager", 1)],
    "one step per graph": [("eager", 1), ("capture", "a", 1), ("replay", "a", 5)],
    "two graphs alternated": [
        ("eager", 2),
        ("capture", "a", 1),
        ("capture", "b", 2),
        ("replay", "a", 2),
        ("replay", "b", 2),
        ("replay", "a", 3),
        ("eager", 1),
        ("replay", "b", 1),
    ],
    "capture after reset": [("eager", 2), ("reset",), ("capture", "a", 2), ("replay", "a", 4), ("reset",)],
    "capture paused between steps": [("capture_paused", "a", 3), ("replay", "a", 4), ("eager", 1)],
}


def test_double_buffer_capture_sequences_match_single_buffer(test, device):
    """Match the single-buffer trajectory bitwise across capture, replay, eager and reset sequences."""
    model = _scene(device)
    for name, plan in _SEQUENCES.items():
        if any(operation == "capture_paused" for operation, *_ in plan) and not wp.is_conditional_graph_supported():
            with test.subTest(sequence=name):
                test.skipTest("requires conditional graph nodes")
            continue
        _, expected = _run_sequence(model, plan)
        for streams in (False, True):
            with test.subTest(sequence=name, use_parallel_streams=streams):
                solver, actual = _run_sequence(model, plan, double_buffer=True, use_parallel_streams=streams)
                test.assertTrue(solver._double_buffered)
                for got, want in zip(actual, expected, strict=True):
                    np.testing.assert_array_equal(got, want)


def test_capture_after_eager_orders_the_pending_clear(test, device):
    """Order an eager step's still-running buffer clearing before the first replay of a later graph.

    The four-bar's loop-closure rows accumulate into the Jacobian, so a set that is not
    cleared yet changes the replayed trajectory.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.replicate(_build_four_bar(), 8)
    model = builder.finalize(device=device)
    control = model.control()
    targets = model.joint_target_q.numpy().reshape(8, -1)
    targets[:, 0] = 0.6
    control.joint_target_q.assign(targets.reshape(-1))
    sink = wp.zeros(1, dtype=float, device=device)
    results = []
    for double_buffer in (False, True):
        solver = SolverSteel(model, solve_mode="serial", iterations=16, baumgarte=0.1, double_buffer=double_buffer)
        test.assertEqual(solver._double_buffered, double_buffer)
        state_0, state_1 = model.state(), model.state()

        def step(solver=solver, state_0=state_0, state_1=state_1):
            solver.step(state_0, state_1, control, None, 1.0 / 240.0)
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                wp.copy(getattr(state_0, name), getattr(state_1, name))

        step()
        step()
        if double_buffer:
            # Hold the last eager step's clearing back until long after the graph launches.
            with wp.ScopedStream(solver._memset_stream, sync_enter=False):
                wp.launch(_spin, dim=1, inputs=[100_000_000], outputs=[sink])
        step()
        with wp.ScopedCapture(device) as scope:
            step()
            step()
        for _ in range(6):
            wp.capture_launch(scope.graph)
        results.append(state_0.joint_q.numpy())
    np.testing.assert_array_equal(results[1], results[0])


def test_reset_after_an_eager_error_steps_like_a_fresh_solver(test, device):
    """Recover from an eager step that raised mid-step: after reset, step bitwise like a fresh solver.

    The torsion error raises after the closure rows were written into the active buffer set (the
    four-bar starts with a closure error so those rows are nonzero); the early error raises on the
    first step, before the mass-matrix stage.
    """
    results = []
    for fail_first in (None, "torsion", "early"):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.replicate(_build_four_bar(), 2)
        model = builder.finalize(device=device)
        set_torsion_radius(model, 0.01)
        solver = SolverSteel(
            model,
            solve_mode="serial",
            iterations=16,
            baumgarte=0.1,
            double_buffer=True,
            contact_torsion_device=True,
        )
        test.assertTrue(solver._double_buffered)
        state, output = model.state(), model.state()
        joint_q = state.joint_q.numpy()
        joint_q[0] = 0.2
        state.joint_q.assign(joint_q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        if fail_first == "torsion":
            # A latched torsion error makes the eager step raise during torsion preparation.
            solver._device_torsion.work.status.assign(np.array([16], dtype=np.int32))
            with test.assertRaisesRegex(RuntimeError, "load ring"):
                solver.step(state, output, model.control(), None, 1.0 / 240.0)
        elif fail_first == "early":
            with (
                mock.patch.object(solver, "_stage1_fk_id", side_effect=KeyboardInterrupt("early")),
                test.assertRaises(KeyboardInterrupt),
            ):
                solver.step(state, output, model.control(), None, 1.0 / 240.0)
        if fail_first:
            solver.reset(state)
            solver.check_constraint_capacity()
        for _ in range(3):
            solver.step(state, output, model.control(), None, 1.0 / 240.0)
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                wp.copy(getattr(state, name), getattr(output, name))
        results.append((state.joint_q.numpy(), state.joint_qd.numpy()))
    for recovered in results[1:]:
        for got, want in zip(recovered, results[0], strict=True):
            np.testing.assert_array_equal(got, want)


def test_flags_are_ignored_where_they_do_not_apply(test, device):
    """Keep one stream and one buffer set on CPU, with sparse factors and with one size group."""
    if wp.get_device(device).is_cpu:
        solver = SolverSteel(_scene(device), use_parallel_streams=True, double_buffer=True)
        test.assertEqual(solver._size_streams, {})
        test.assertFalse(solver._double_buffered)
        return
    builder = newton.ModelBuilder()
    root = builder.add_link()
    builder.add_shape_box(root, hx=0.1, hy=0.1, hz=0.1)
    joints = [builder.add_joint_revolute(-1, root, axis=newton.Axis.Z)]
    for side in (-1.0, 1.0):
        leg = builder.add_link()
        builder.add_shape_box(leg, hx=0.05, hy=0.05, hz=0.05)
        joints.append(
            builder.add_joint_revolute(
                root, leg, axis=newton.Axis.Y, parent_xform=wp.transform(wp.vec3(side * 0.2, 0, 0), wp.quat_identity())
            )
        )
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    solver = SolverSteel(model, friction_anchor_beta=0.0, use_parallel_streams=True, double_buffer=True)
    test.assertIsNotNone(solver._sparse_mass_matrix_size)
    test.assertFalse(solver._double_buffered)
    # One size group needs no extra streams.
    test.assertEqual(solver._size_streams, {})


class TestSteelStreams(unittest.TestCase):
    pass


for _fn in (
    test_double_buffer_capture_sequences_match_single_buffer,
    test_capture_after_eager_orders_the_pending_clear,
    test_reset_after_an_eager_error_steps_like_a_fresh_solver,
):
    add_function_test(TestSteelStreams, _fn.__name__, _fn, devices=cuda_devices)


for _fn in (test_streams_and_double_buffer_step_identically, test_captured_double_buffer_matches_eager):
    add_function_test(TestSteelStreams, _fn.__name__, _fn, devices=cuda_devices)


add_function_test(
    TestSteelStreams,
    "test_flags_are_ignored_where_they_do_not_apply",
    test_flags_are_ignored_where_they_do_not_apply,
    devices=get_test_devices(),
)


def test_fused_assembly_matches_the_separate_pipeline(test, device, solve_mode="serial"):
    """Match separate assembly and factorization with the warp-per-articulation and tiled kernels."""
    for links, branched, kernel in ((3, False, "warp"), (6, True, "warp"), (14, False, "tiled"), (14, True, "tiled")):
        with test.subTest(links=links, branched=branched):
            model = _tree(device, links, branched=branched)
            options = {"solve_mode": solve_mode, "articulation_constraint_max": _ROWS}
            # The branched trees would take the sparse factors; this test is about the dense fused kernels.
            with kernel_overrides({"sparse_mass_matrix": False}):
                fused, q, qd = _trajectory(model, use_parallel_streams=True, **options)
                reference, q_ref, qd_ref = _trajectory(model, **options)
            size = fused._size_groups[0]
            selected = {
                "warp": fused._crba_cholesky_warp_kernels_by_size[size],
                "tiled": fused._crba_cholesky_kernels_by_size[size],
            }
            test.assertIsNotNone(selected[kernel])
            test.assertIsNone(reference._crba_cholesky_warp_kernels_by_size[size])
            test.assertIsNone(reference._crba_cholesky_kernels_by_size[size])
            np.testing.assert_allclose(fused._L_by_size[size].numpy(), reference._L_by_size[size].numpy(), atol=2.0e-5)
            np.testing.assert_allclose(q, q_ref, rtol=0.0, atol=2.0e-5)
            np.testing.assert_allclose(qd, qd_ref, rtol=0.0, atol=2.0e-4)


def test_fused_assembly_covers_articulations_wider_than_the_block(test, device):
    """Match separate assembly when the tiled block has fewer threads than the articulation has DOFs."""
    # With 32 threads, 33 DOFs leave the second force-staging round one lane wide; both widths share its tile kernels.
    model = _tree(device, 33, worlds=1)
    with kernel_overrides({"sparse_mass_matrix": False}):
        reference, q_ref, qd_ref = _trajectory(model, steps=2)
    size = reference._size_groups[0]
    test.assertGreater(size, 32)
    test.assertIsNone(reference._crba_cholesky_kernels_by_size[size])
    for tile_threads in (32, 64):
        with test.subTest(tile_threads=tile_threads):
            overrides = {"tile_threads": tile_threads, "sparse_mass_matrix": False}
            with kernel_overrides(overrides):
                fused, q, qd = _trajectory(model, steps=2, use_parallel_streams=True)
            test.assertIsNotNone(fused._crba_cholesky_kernels_by_size[size])
            test.assertEqual(fused._tile_threads, tile_threads)
            np.testing.assert_allclose(fused._L_by_size[size].numpy(), reference._L_by_size[size].numpy(), atol=2.0e-5)
            np.testing.assert_allclose(q, q_ref, rtol=1.0e-5, atol=2.0e-5)
            np.testing.assert_allclose(qd, qd_ref, rtol=1.0e-5, atol=2.0e-4)


def test_fused_assembly_follows_the_mass_update_interval(test, device):
    """Refresh the fused factors on the interval and after an inertia notification only."""
    model = _tree(device, 3)
    results = []
    for streams in (True, False):
        solver, q, _ = _trajectory(model, steps=7, use_parallel_streams=streams, update_mass_matrix_interval=3)
        body_mass = model.body_mass.numpy()
        model.body_mass.assign(body_mass * 1.5)
        model.body_inv_mass.assign(1.0 / (body_mass * 1.5))
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        solver.step(state_0, state_1, model.control(), None, 1.0 / 240.0)
        model.body_mass.assign(body_mass)
        model.body_inv_mass.assign(1.0 / body_mass)
        results.append((q, solver._L_by_size[solver._size_groups[0]].numpy()))
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=0.0, atol=2.0e-5)
    np.testing.assert_allclose(results[0][1], results[1][1], rtol=0.0, atol=2.0e-5)


def test_fused_assembly_is_selected_only_where_supported(test, device):
    """Keep the separate pipeline without streams, with loop joints, PGS drive rows or on CPU."""
    is_cuda = wp.get_device(device).is_cuda
    solver = SolverSteel(_tree(device, 3), use_parallel_streams=True)
    test.assertEqual(bool(solver._crba_source_dof_by_size), is_cuda)
    if not is_cuda:
        return
    for model, options in (
        (_tree(device, 3), {}),
        (_tree(device, 3), {"use_parallel_streams": True, "drive_mode": "rows"}),
        (_tree(device, 3, loop=True), {"use_parallel_streams": True}),
    ):
        with test.subTest(**options):
            solver = SolverSteel(model, **options)
            test.assertFalse(solver._crba_source_dof_by_size)
            test.assertIsNone(solver._crba_cholesky_warp_kernels_by_size[solver._size_groups[0]])


class TestSteelFusedCrba(unittest.TestCase):
    pass


add_function_test(
    TestSteelFusedCrba,
    "test_fused_assembly_matches_the_separate_pipeline",
    test_fused_assembly_matches_the_separate_pipeline,
    devices=cuda_devices,
)


add_function_test(
    TestSteelFusedCrba,
    "test_fused_assembly_covers_articulations_wider_than_the_block",
    test_fused_assembly_covers_articulations_wider_than_the_block,
    devices=cuda_devices,
)


add_function_test(
    TestSteelFusedCrba,
    "test_fused_assembly_follows_the_mass_update_interval",
    test_fused_assembly_follows_the_mass_update_interval,
    devices=cuda_devices,
)


add_function_test(
    TestSteelFusedCrba,
    "test_fused_assembly_is_selected_only_where_supported",
    test_fused_assembly_is_selected_only_where_supported,
    devices=get_test_devices(),
)


def test_register_lower_inverse_matches_the_serial_inversion(test, device):
    """Match the per-articulation forward substitution column for column on refreshed lower factors."""
    rng = np.random.default_rng(7)
    for n_dofs in (6, 23):
        n_arts = 5
        factors = np.zeros((n_arts, n_dofs, n_dofs), dtype=np.float32)
        for art in range(n_arts):
            a = rng.standard_normal((n_dofs, n_dofs)).astype(np.float64)
            factors[art] = np.linalg.cholesky(a @ a.T + n_dofs * np.eye(n_dofs)).astype(np.float32)
        L = wp.array(factors, dtype=wp.float32, device=device)
        group_to_art = wp.array(np.arange(n_arts, dtype=np.int32), dtype=wp.int32, device=device)
        # The third articulation is not refreshed; both kernels must leave its stale inverse alone.
        mask = wp.array(np.array([1, 1, 0, 1, 1], dtype=np.int32), dtype=wp.int32, device=device)
        expected = wp.full((n_arts, n_dofs, n_dofs), 42.0, dtype=wp.float32, device=device)
        actual = wp.full((n_arts, n_dofs, n_dofs), 42.0, dtype=wp.float32, device=device)
        wp.launch(
            steel_kernels.invert_lower_factor_grouped,
            dim=n_arts,
            inputs=[group_to_art, mask, n_dofs, L],
            outputs=[expected],
            device=device,
        )
        kernel = solver_steel._get_inverse_cholesky_register_kernel(n_dofs, device.arch, lower_only=True)
        wp.launch(
            kernel, dim=n_arts * 32, inputs=[L, group_to_art, mask], outputs=[actual], block_dim=256, device=device
        )
        expected_np, actual_np = expected.numpy(), actual.numpy()
        np.testing.assert_array_equal(actual_np[2], 42.0, err_msg=f"n_dofs={n_dofs}: masked articulation was written")
        refreshed = [0, 1, 3, 4]
        np.testing.assert_allclose(
            actual_np[refreshed], expected_np[refreshed], rtol=1.0e-5, atol=1.0e-6, err_msg=f"n_dofs={n_dofs}"
        )
        identity = np.einsum("aij,ajk->aik", factors[refreshed], actual_np[refreshed])
        np.testing.assert_allclose(
            identity, np.broadcast_to(np.eye(n_dofs), identity.shape), atol=2.0e-4, err_msg=f"n_dofs={n_dofs}: L L^-1"
        )


class TestSteelRegisterLowerInverse(unittest.TestCase):
    pass


add_function_test(
    TestSteelRegisterLowerInverse,
    "test_register_lower_inverse_matches_the_serial_inversion",
    test_register_lower_inverse_matches_the_serial_inversion,
    devices=cuda_devices,
)


def test_split_kinematics_match_the_fused_fk_id_kernel(test, device):
    """Reproduce the fused FK/ID kernel bitwise with the serial kinematics pass plus the per-body finaliser."""
    model = _build_contact_scene(device)
    # Only the solver's index maps and augmented buffers are used.
    solver = SolverSteel(model)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    rng = np.random.default_rng(3)
    state.joint_qd.assign(rng.normal(scale=0.7, size=model.joint_dof_count).astype(np.float32))
    aug = solver._prepare_augmented_state(state)

    def fresh():
        return {
            "body_q": wp.clone(state.body_q),
            "body_q_com": wp.zeros_like(aug._body_q_com),
            "origin": wp.zeros_like(solver._articulation_origin),
            "joint_S_s": wp.zeros_like(aug._joint_S_s),
            "body_I_s": wp.zeros_like(aug._body_I_s),
            # The solver keeps a one-row stand-in for the inertia terms (it never asks for them); size them per body here.
            "terms": wp.zeros((model.body_count, 12), dtype=wp.float32, device=device),
            "body_v_s": wp.zeros_like(aug._body_v_s),
            "body_f_s": wp.zeros_like(aug._body_f_s),
            "body_a_s": wp.zeros_like(aug._body_a_s),
            "body_qd": wp.zeros_like(state.body_qd),
        }

    for materialize_inertia, materialize_terms in ((1, 1), (0, 0), (1, 0)):
        with test.subTest(inertia=materialize_inertia, terms=materialize_terms):
            fused, split = fresh(), fresh()
            common = [
                model.articulation_start,
                solver._articulation_joint_end,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_q_start,
                model.joint_qd_start,
                state.joint_q,
                state.joint_qd,
                model.joint_X_p,
                model.joint_X_c,
                solver._body_X_com,
                model.joint_axis,
                model.joint_dof_dim,
                model.body_com,
            ]
            wp.launch(
                steel_kernels.eval_rigid_fk_id,
                dim=model.articulation_count,
                inputs=[
                    *common,
                    model.body_mass,
                    model.body_inertia,
                    solver._is_free_rigid,
                    materialize_inertia,
                    materialize_terms,
                    model.body_world,
                    model.gravity,
                    solver._dynamics_art_active,
                ],
                outputs=[
                    fused["body_q"],
                    fused["body_q_com"],
                    fused["origin"],
                    fused["joint_S_s"],
                    fused["body_I_s"],
                    fused["terms"],
                    fused["body_v_s"],
                    fused["body_f_s"],
                    fused["body_a_s"],
                ],
                block_dim=16,
                device=device,
            )
            wp.launch(
                steel_kernels.update_body_qd_from_featherstone,
                dim=model.body_count,
                inputs=[
                    fused["body_v_s"],
                    fused["body_q"],
                    model.body_com,
                    solver._body_to_articulation,
                    fused["origin"],
                    solver._dynamics_body_active,
                ],
                outputs=[fused["body_qd"]],
                device=device,
            )
            none = wp.zeros(model.articulation_count, dtype=wp.int32, device=device)
            valid = wp.zeros(model.articulation_count, dtype=wp.int32, device=device)
            wp.launch(
                steel_kernels.eval_rigid_fk_kinematics,
                dim=model.articulation_count,
                inputs=[*common, solver._dynamics_art_active, none],
                outputs=[
                    split["body_q"],
                    split["body_q_com"],
                    split["origin"],
                    split["joint_S_s"],
                    split["body_v_s"],
                    split["body_a_s"],
                    valid,
                ],
                block_dim=16,
                device=device,
            )
            np.testing.assert_array_equal(valid.numpy(), 1)
            wp.launch(
                steel_kernels.finalize_body_dynamics,
                dim=model.body_count,
                inputs=[
                    solver._body_to_articulation,
                    split["body_q"],
                    split["body_q_com"],
                    model.body_com,
                    model.body_mass,
                    model.body_inertia,
                    solver._is_free_rigid,
                    split["origin"],
                    materialize_inertia,
                    materialize_terms,
                    model.body_world,
                    model.gravity,
                    split["body_v_s"],
                    split["body_a_s"],
                    solver._dynamics_body_active,
                    none,
                ],
                outputs=[split["body_I_s"], split["terms"], split["body_f_s"], split["body_qd"]],
                device=device,
            )
            for name in fused:
                np.testing.assert_array_equal(split[name].numpy(), fused[name].numpy(), err_msg=name)


def _carry_trajectory(model, steps, reuse, edit=None, dts=None, **options):
    """Step with alternating state objects, applying ``edit(solver, state_in, step)`` between steps; record the states.

    ``edit`` may return a replacement input state; ``dts`` gives each step's timestep.
    """
    solver = SolverSteel(model, reuse_kinematics=reuse, **options)
    states = [model.state(), model.state()]
    newton.eval_fk(model, model.joint_q, model.joint_qd, states[0])
    control = model.control()
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=256)
    contacts = pipeline.contacts()
    history = []
    for step in range(steps):
        if edit is not None:
            replacement = edit(solver, states[0], step)
            if replacement is not None:
                states[0] = replacement
        states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, 1.0 / 240.0 if dts is None else dts[step])
        history.append(
            tuple(getattr(states[1], name).numpy().copy() for name in ("joint_q", "joint_qd", "body_q", "body_qd"))
        )
        states.reverse()
    return solver, history


def _assert_histories_close(test, expected, actual):
    """Allow the roundoff of publishing body velocities through the solver's own kernel instead of eval_fk."""
    for step, (want, got) in enumerate(zip(expected, actual, strict=True)):
        for name, a, b in zip(("joint_q", "joint_qd", "body_q", "body_qd"), want, got, strict=True):
            tolerance = 2.0e-4 if name.endswith("qd") else 2.0e-5
            np.testing.assert_allclose(b, a, rtol=2.0e-5, atol=tolerance, err_msg=f"{name} at step {step}")


def test_reused_kinematics_match_the_recomputed_trajectory(test, device):
    """Carry the published kinematics into the next step and reproduce the recomputing solver.

    The joint coordinates agree bitwise on the first step; the published body poses and velocities can
    differ in their last bit from eval_fk's, which seeds roundoff-level differences afterwards.
    """
    model = _build_contact_scene(device)
    reference_solver, reference = _carry_trajectory(model, 24, reuse=False)
    carried_solver, carried = _carry_trajectory(model, 24, reuse=True)
    test.assertFalse(reference_solver._reuse_kinematics_enabled)
    test.assertTrue(carried_solver._reuse_kinematics_enabled)
    test.assertTrue(np.all(carried_solver._fk_id_valid.numpy() == 1))
    for name, a, b in zip(("joint_q", "joint_qd"), reference[0][:2], carried[0][:2], strict=True):
        np.testing.assert_array_equal(b, a, err_msg=f"{name} at the first step")
    _assert_histories_close(test, reference, carried)


def test_reused_kinematics_follow_resets_and_notices(test, device):
    """Re-derive the carried terms after a masked reset, a state notice or a new input state; stay stale without them."""
    model = _build_contact_scene(device)
    world_mask = np.zeros(model.world_count + 1, dtype=bool)
    world_mask[1] = True

    def teleport(state):
        joint_q = state.joint_q.numpy()
        start = int(model.articulation_start.numpy()[int(np.flatnonzero(model.articulation_world.numpy() == 1)[0])])
        q_start = int(model.joint_q_start.numpy()[start])
        joint_q[q_start : q_start + 3] += 0.3
        state.joint_q.assign(joint_q)

    def reset_edit(solver, state, step):
        if step == 8:
            teleport(state)
            solver.reset(state, world_mask=wp.array(world_mask, dtype=wp.bool, device=device))

    def notice_edit(solver, state, step):
        if step == 8:
            teleport(state)
            solver.notify_state_changed(state)

    def stale_edit(solver, state, step):
        if step == 8:
            teleport(state)

    articulations = np.flatnonzero(model.articulation_world.numpy() == 1)
    others = np.setdiff1d(np.arange(model.articulation_count), articulations)

    def selected_notice_edit(solver, state, step):
        if step == 8:
            teleport(state)
            solver.notify_state_changed(state, world_mask=wp.array(world_mask, dtype=wp.bool, device=device))
            valid = solver._fk_id_valid.numpy()
            np.testing.assert_array_equal(valid[articulations], 0)
            np.testing.assert_array_equal(valid[others], 1)

    def new_state_edit(solver, state, step):
        if step == 8:
            teleport(state)
            fresh = model.state()
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                wp.copy(getattr(fresh, name), getattr(state, name))
            return fresh
        return None

    _, reference = _carry_trajectory(model, 12, reuse=False, edit=stale_edit)
    for label, edit in (
        ("reset", reset_edit),
        ("notice", notice_edit),
        ("selected notice", selected_notice_edit),
        ("new state", new_state_edit),
    ):
        with test.subTest(label):
            _, carried = _carry_trajectory(model, 12, reuse=True, edit=edit)
            _assert_histories_close(test, reference, carried)
    with test.subTest("stale"):
        _, stale = _carry_trajectory(model, 12, reuse=True, edit=stale_edit)
        # Without a notice the step after the in-place write keeps the carried poses of the edited world.
        test.assertGreater(np.abs(stale[8][2] - reference[8][2]).max(), 1.0e-2)
        for step in range(8):
            np.testing.assert_allclose(stale[step][2], reference[step][2], rtol=2.0e-5, atol=2.0e-5)


def test_reused_kinematics_follow_a_timestep_change(test, device):
    """Re-derive the carried terms when a timestep change forces a mass rebuild between cadence refreshes."""
    model = _build_contact_scene(device)
    dts = [1.0 / 120.0] * 6 + [1.0 / 240.0] * 6
    _, reference = _carry_trajectory(model, 12, reuse=False, dts=dts, update_mass_matrix_interval=4)
    carried_solver, carried = _carry_trajectory(model, 12, reuse=True, dts=dts, update_mass_matrix_interval=4)
    test.assertTrue(carried_solver._reuse_kinematics_enabled)
    _assert_histories_close(test, reference, carried)


def test_captured_step_replays_the_same_result_from_an_unchanged_input(test, device):
    """A step captured on a fresh solver and replayed from the same input gives the same result every replay."""
    model = _build_contact_scene(device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=256)
    contacts = pipeline.contacts()
    control = model.control()
    names = ("joint_q", "joint_qd", "body_q", "body_qd")

    def replays(reuse):
        solver = SolverSteel(model, reuse_kinematics=reuse)
        state_in, state_out = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        with wp.ScopedCapture(device) as capture:
            solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
        outputs = []
        for _ in range(3):
            wp.capture_launch(capture.graph)
            outputs.append(tuple(getattr(state_out, name).numpy().copy() for name in names))
        return solver, outputs

    _, reference = replays(False)
    carried_solver, carried = replays(True)
    test.assertTrue(carried_solver._reuse_kinematics_enabled)
    for replay in (1, 2):
        for name, first, later in zip(names, carried[0], carried[replay], strict=True):
            np.testing.assert_array_equal(later, first, err_msg=f"{name} changed on replay {replay}")
    _assert_histories_close(test, reference, carried)


def test_extra_free_joints_keep_the_generic_publication(test, device):
    """A FREE joint below another body, or a second free root, is published through eval_fk.

    Their twists are referenced at the articulation origin, a convention eval_fk does not share.
    """
    for label, second_parent in (("free child", "root"), ("second free root", "world")):
        with test.subTest(label):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            inertia = wp.mat33(np.eye(3) * 0.1)
            root = builder.add_link(
                xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), mass=1.0, inertia=inertia
            )
            other = builder.add_link(
                xform=wp.transform(wp.vec3(0.5, 0.0, 1.0), wp.quat_identity()), mass=1.0, inertia=inertia
            )
            first = builder.add_joint_free(root)
            second = builder.add_joint_free(other, parent=root if second_parent == "root" else -1)
            builder.add_articulation([first, second])
            model = builder.finalize(device=device)
            solver = SolverSteel(model, reuse_kinematics=True)
            test.assertFalse(solver.reuse_kinematics)
            state_in, state_out = model.state(), model.state()
            joint_qd = np.zeros(model.joint_dof_count, dtype=np.float32)
            joint_qd[5] = 3.0
            joint_qd[6] = 0.2
            joint_qd[11] = 3.0
            state_in.joint_qd.assign(joint_qd)
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
            solver.step(state_in, state_out, model.control(), None, 1.0 / 240.0)
            expected = model.state()
            newton.eval_fk(model, state_out.joint_q, state_out.joint_qd, expected)
            np.testing.assert_array_equal(state_out.body_qd.numpy(), expected.body_qd.numpy())


def test_state_notice_validates_its_world_mask(test, device):
    """Reject a short, host or non-boolean world mask as reset does, whether or not the carry runs."""
    model = _build_contact_scene(device)
    state = model.state()
    bad_masks = (
        (wp.ones(model.world_count, dtype=wp.bool, device=device), ValueError),
        (wp.ones(model.world_count + 1, dtype=wp.bool, device="cpu"), ValueError),
        (wp.ones(model.world_count + 1, dtype=wp.int32, device=device), TypeError),
    )
    for reuse in (True, False):
        solver = SolverSteel(model, reuse_kinematics=reuse)
        test.assertEqual(solver.reuse_kinematics, reuse)
        for mask, error in bad_masks:
            with test.subTest(reuse=reuse, shape=mask.shape, dtype=str(mask.dtype), device=str(mask.device)):
                with test.assertRaisesRegex(error, "world_mask"):
                    solver.notify_state_changed(state, world_mask=mask)
        solver.notify_state_changed(state, world_mask=wp.ones(model.world_count + 1, dtype=wp.bool, device=device))


def test_torsion_rollback_drops_the_carried_kinematics(test, device):
    """A captured step whose torsion status latches rolls its outputs back; the carry must not outlive the rollback."""
    model = _build_contact_scene(device)
    set_torsion_radius(model, 0.01)
    names = ("joint_q", "joint_qd", "body_q", "body_qd")

    def replays(reuse):
        solver = SolverSteel(model, reuse_kinematics=reuse, contact_torsion_device=True)
        test.assertEqual(solver.reuse_kinematics, reuse)
        states = [model.state(), model.state()]
        newton.eval_fk(model, model.joint_q, model.joint_qd, states[0])
        control = model.control()
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=256)
        contacts = pipeline.contacts()
        states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        with wp.ScopedCapture(device) as capture:
            solver.step(states[0], states[1], control, contacts, 1.0 / 240.0)
            solver.step(states[1], states[0], control, contacts, 1.0 / 240.0)
        for _ in range(3):
            wp.capture_launch(capture.graph)
        # A latched error makes both captured steps roll their outputs back to their inputs.
        solver._device_torsion.work.status.assign(np.array([16], dtype=np.int32))
        wp.capture_launch(capture.graph)
        # A masked reset clears the solver-wide status but drops no world's carried terms.
        solver.reset(states[0], world_mask=wp.zeros(model.world_count + 1, dtype=wp.bool, device=device))
        outputs = []
        for _ in range(3):
            wp.capture_launch(capture.graph)
            outputs.append(tuple(getattr(states[0], name).numpy().copy() for name in names))
        return outputs

    _assert_histories_close(test, replays(False), replays(True))


def test_captured_chain_with_a_refresh_interval_matches_the_recomputing_solver(test, device):
    """Replay a captured two-step chain whose mass-refresh cadence is baked at capture, carrying against recomputing."""
    model = _build_contact_scene(device)
    names = ("joint_q", "joint_qd", "body_q", "body_qd")

    def replays(reuse):
        solver = SolverSteel(model, reuse_kinematics=reuse, update_mass_matrix_interval=4)
        states = [model.state(), model.state()]
        newton.eval_fk(model, model.joint_q, model.joint_qd, states[0])
        control = model.control()
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=256)
        contacts = pipeline.contacts()
        states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        with wp.ScopedCapture(device) as capture:
            solver.step(states[0], states[1], control, contacts, 1.0 / 240.0)
            solver.step(states[1], states[0], control, contacts, 1.0 / 240.0)
        outputs = []
        for _ in range(6):
            wp.capture_launch(capture.graph)
            outputs.append(tuple(getattr(states[0], name).numpy().copy() for name in names))
        return solver, outputs

    _, reference = replays(False)
    carried_solver, carried = replays(True)
    test.assertTrue(carried_solver._reuse_kinematics_enabled)
    _assert_histories_close(test, reference, carried)


class TestSteelReusedKinematics(unittest.TestCase):
    pass


for _fn in (
    test_reused_kinematics_match_the_recomputed_trajectory,
    test_reused_kinematics_follow_resets_and_notices,
    test_reused_kinematics_follow_a_timestep_change,
    test_captured_step_replays_the_same_result_from_an_unchanged_input,
    test_extra_free_joints_keep_the_generic_publication,
    test_state_notice_validates_its_world_mask,
    test_torsion_rollback_drops_the_carried_kinematics,
    test_captured_chain_with_a_refresh_interval_matches_the_recomputing_solver,
):
    add_function_test(TestSteelReusedKinematics, _fn.__name__, _fn, devices=cuda_devices)


class TestSteelSplitKinematics(unittest.TestCase):
    pass


add_function_test(
    TestSteelSplitKinematics,
    "test_split_kinematics_match_the_fused_fk_id_kernel",
    test_split_kinematics_match_the_fused_fk_id_kernel,
    devices=get_test_devices(),
)


def _build_chains_on_boxes(device):
    """Two driven, limited chains, one resting on a free box, and a lone box, in two worlds."""
    world = newton.ModelBuilder()
    for x, top in ((0.0, 0.92), (1.5, 1.05)):
        joints, parent = [], -1
        for i in range(3):
            link = world.add_link(xform=wp.transform(wp.vec3(x, 0.0, top - 0.3 * i), wp.quat_identity()))
            world.add_shape_capsule(link, radius=0.04, half_height=0.12)
            joints.append(
                world.add_joint_revolute(
                    parent,
                    link,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    parent_xform=wp.transform(
                        wp.vec3(x, 0.0, top + 0.15) if parent < 0 else wp.vec3(0.0, 0.0, -0.15), wp.quat_identity()
                    ),
                    child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.15), wp.quat_identity()),
                    limit_lower=-0.6,
                    limit_upper=0.6,
                    target_ke=200.0,
                    target_kd=5.0,
                    target_q=0.3,
                )
            )
            parent = link
        world.add_articulation(joints)
    for x, z in ((0.0, 0.09), (3.0, 0.2)):
        body = world.add_link(xform=wp.transform(wp.vec3(x, 0.0, z), wp.quat_identity()))
        world.add_shape_box(body, hx=0.08, hy=0.08, hz=0.08)
        world.add_articulation([world.add_joint_free(parent=-1, child=body)])
    scene = newton.ModelBuilder()
    scene.add_ground_plane()
    scene.replicate(world, 2, spacing=(0.0, 6.0, 0.0))
    return scene.finalize(device=device)


def _row_phase_rollout(model, *, reference=None, capture=False, **options):
    options = {"enable_joint_limits": True, "drive_mode": "rows", "articulation_constraint_max": _ROWS, **options}
    solver = SolverSteel(
        model,
        solve_mode="serial",
        iterations=12,
        free_body_constraint_max=_ROWS,
        deterministic=wp.DeterministicMode.RUN_TO_RUN,
        **options,
    )
    prebuilt = set(solver._pgs_solve_mf_gs_kernels)
    if reference is not None:
        # Every launch takes one kernel built with these options, which reads the phase argument.
        solver._pgs_solve_mf_gs_kernels = {0: solver._pgs_solve_mf_gs_kernel_for_phase(**reference)}
    solver.launched_phases = set()
    launch_phase = solver._launch_mf_gs_phase

    def record_phase(row_phase, *args, **kwargs):
        solver.launched_phases.add(row_phase)
        launch_phase(row_phase, *args, **kwargs)

    solver._launch_mf_gs_phase = record_phase
    # Ordered contacts and rows, so separate rollouts can be compared bit for bit.
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    state, next_state, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    def step():
        pipeline.collide(state, contacts)
        solver.step(state, next_state, control, contacts, 1.0 / 240.0)
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            wp.copy(getattr(state, name), getattr(next_state, name))

    if capture:
        # The solver's first step is the captured one.
        with wp.ScopedCapture(model.device) as graph:
            step()
    trace = []
    for _ in range(10):
        if capture:
            wp.capture_launch(graph.graph)
        else:
            step()
        trace.append(np.concatenate([state.body_q.numpy().ravel(), state.body_qd.numpy().ravel()]))
    return solver, prebuilt, np.asarray(trace)


def test_compiled_row_phase_matches_runtime_phase(test, device, **options):
    """Kernels with the row phase compiled in reproduce the runtime-phase kernel bit for bit."""
    model = _build_chains_on_boxes(device)
    solver, prebuilt, compiled = _row_phase_rollout(model, **options)
    _, _, runtime = _row_phase_rollout(model, reference={}, **options)
    # Construction built exactly the phases the schedule launched.
    test.assertEqual(set(solver._pgs_solve_mf_gs_kernels), prebuilt)
    test.assertEqual(solver.launched_phases, prebuilt)
    for phase, kernel in solver._pgs_solve_mf_gs_kernels.items():
        test.assertIn(f"_phase{phase}", kernel.key)
    test.assertTrue(np.isfinite(runtime).all())
    np.testing.assert_array_equal(compiled.view(np.uint32), runtime.view(np.uint32))


def test_compiled_row_phases_replay_a_captured_first_step(test, device):
    """A propagation solver whose first step is captured replays its phase kernels like eager steps."""
    model = _build_chains_on_boxes(device)
    options = {"articulated_contact_response": "propagation", **_VELOCITY_LIMITS}
    solver, prebuilt, captured = _row_phase_rollout(model, capture=True, **options)
    _, _, eager = _row_phase_rollout(model, **options)
    test.assertEqual(set(solver._pgs_solve_mf_gs_kernels), prebuilt)
    np.testing.assert_array_equal(captured.view(np.uint32), eager.view(np.uint32))


class TestSteelMatrixFreeRowPhase(unittest.TestCase):
    pass


# Drive rows and velocity limits give every phase rows to solve, so a wrong phase changes the result.
_VELOCITY_LIMITS = {"enable_joint_velocity_limits": True, "velocity_limit_activation_fraction": 0.0}
for _name, _options in (
    ("phased", {"row_schedule": "phased", **_VELOCITY_LIMITS}),
    ("contact_then_internal", {"row_schedule": "contact_then_internal"}),
    ("propagation", {"articulated_contact_response": "propagation", **_VELOCITY_LIMITS}),
    (
        "propagation_contact_then_internal",
        {"articulated_contact_response": "propagation", "row_schedule": "contact_then_internal"},
    ),
    (
        "propagation_contact_rows_only",
        {"articulated_contact_response": "propagation", "enable_joint_limits": False, "drive_mode": "augmented"},
    ),
):
    add_function_test(
        TestSteelMatrixFreeRowPhase,
        f"test_compiled_row_phase_matches_runtime_phase_{_name}",
        test_compiled_row_phase_matches_runtime_phase,
        devices=get_cuda_test_devices(),
        **_options,
    )
add_function_test(
    TestSteelMatrixFreeRowPhase,
    "test_compiled_row_phases_replay_a_captured_first_step",
    test_compiled_row_phases_replay_a_captured_first_step,
    devices=get_cuda_test_devices(),
)


def test_pair_cache_matches_the_uncached_solve(test, device, torsion_radius=0.0, **options):
    """Reproduce the uncached resident-metadata MF-GS solve bit for bit with the pair and row caches."""
    model = _build_chains_on_boxes(device)
    if torsion_radius > 0.0:
        set_torsion_radius(model, torsion_radius)
    solver, _, cached = _row_phase_rollout(model, **options)
    test.assertTrue(solver._mfgs_pair_cache)
    test.assertIn("_paircache", solver._pgs_solve_mf_gs_kernel.key)
    # Streamed metadata also selects the row type and parent cache.
    test.assertEqual(
        options.get("articulation_constraint_max", _ROWS) > _ROWS, "_gmeta" in solver._pgs_solve_mf_gs_kernel.key
    )
    test.assertEqual(torsion_radius > 0.0, solver._contact_torsion_enabled)
    # Resident metadata, so the streamed cases also check the streamed-metadata prefetch and row cache.
    _, _, uncached = _row_phase_rollout(model, reference={"pair_cache": False, "shared_metadata": True}, **options)
    test.assertTrue(np.isfinite(uncached).all())
    np.testing.assert_array_equal(cached.view(np.uint32), uncached.view(np.uint32))


def test_pair_cache_selection(test, device):
    """Select the pair cache only for the fixed-normal pair solve outside phased and propagation, all worlds resident."""
    shape = {"shared_metadata": False, "has_drive_rows": True, "fuse_vel_limits": False}
    test.assertTrue(_use_mfgs_pair_cache(128, 64, 39, 48 * 1024, world_count=64, sm_count=100, **shape))
    test.assertFalse(_use_mfgs_pair_cache(128, 64, 39, 48 * 1024, world_count=100_000, sm_count=100, **shape))
    # The cached arrays fit the opt-in limit but exceed 48 KiB of static shared memory.
    test.assertTrue(_use_mfgs_pair_cache(1024, 1024, 200, 101376, world_count=64, sm_count=100, **shape))
    test.assertFalse(_use_mfgs_pair_cache(3072, 1024, 200, 101376, world_count=64, sm_count=100, **shape))
    model = _build_chains_on_boxes(device)
    options = {"articulation_constraint_max": _ROWS, "free_body_constraint_max": _ROWS}
    test.assertTrue(SolverSteel(model, **options)._mfgs_pair_cache)
    for excluded in (
        {"row_schedule": "phased"},
        {"articulated_contact_response": "propagation"},
        {"friction_mode": "bisection", "friction_anchor_beta": 0.0},
    ):
        with test.subTest(**excluded):
            test.assertFalse(SolverSteel(model, **options, **excluded)._mfgs_pair_cache)


class TestSteelMatrixFreePairCache(unittest.TestCase):
    pass


for _name, _options in (
    ("resident_metadata", {}),
    ("streamed_metadata", {"articulation_constraint_max": 192}),
    ("contact_torsion", {"articulation_constraint_max": 192, "torsion_radius": 0.02}),
):
    add_function_test(
        TestSteelMatrixFreePairCache,
        f"test_pair_cache_matches_the_uncached_solve_{_name}",
        test_pair_cache_matches_the_uncached_solve,
        devices=get_cuda_test_devices(),
        **_options,
    )
add_function_test(
    TestSteelMatrixFreePairCache,
    "test_pair_cache_selection",
    test_pair_cache_selection,
    devices=get_cuda_test_devices(),
)


def test_velocity_limit_pass_visits_every_active_row(test, device):
    """Project velocity-limit rows at the first and last dense rows and between contacts, and skip inactive rows.

    The pass walks only the span of each world's velocity-limit rows, so this places them at the span's ends.
    """
    # More than 32 rows, so each lane's span scan takes a second stride.
    rows, dofs = 40, 12
    contact, friction, vlim = (
        steel_kernels.PGS_CONSTRAINT_TYPE_CONTACT,
        steel_kernels.PGS_CONSTRAINT_TYPE_FRICTION,
        steel_kernels.PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    )
    layouts = {
        "first": [0, 1],
        "last": [38, 39],
        "between_contacts": [3, 10, 17, 33],
        # Rows 5 and 37 share a lane, which must keep the later row as its upper bound.
        "same_lane": [5, 37],
        "none": [],
    }
    worlds = len(layouts)
    types = np.full((worlds, rows), steel_kernels.PGS_CONSTRAINT_TYPE_JOINT_LIMIT, np.int32)
    parents = np.full((worlds, rows), -1, np.int32)
    counts = np.full(worlds, rows - 2, np.int32)
    for w, vlim_rows in enumerate(layouts.values()):
        # Whole contact triplets clear of the velocity-limit rows, since friction also writes its sibling's impulse.
        start = 0
        for _ in range(5):
            while set(range(start, start + 3)) & set(vlim_rows):
                start += 1
            types[w, start] = contact
            types[w, start + 1 : start + 3] = friction
            parents[w, start + 1 : start + 3] = start
            start += 3
        types[w, vlim_rows] = vlim
        if vlim_rows and vlim_rows[-1] >= rows - 2:
            counts[w] = rows
        else:
            # An inactive row past the world's count is never visited.
            types[w, rows - 1] = vlim
    rng = np.random.default_rng(5)
    J = rng.normal(size=(worlds, rows, dofs)).astype(np.float32)
    Y = (0.5 * J).astype(np.float32)
    diag = (np.einsum("wrd,wrd->wr", J, Y) + 1.0e-3).astype(np.float32)
    rhs = np.full((worlds, rows), -0.5, np.float32)
    # The inactive row is strongly violated, so visiting it would change the output velocity.
    rhs[:, rows - 1] = -100.0
    # The same worlds with the inactive row retyped give the velocity of a solve that skips it.
    types_without_inactive = types.copy()
    types_without_inactive[counts < rows, rows - 1] = steel_kernels.PGS_CONSTRAINT_TYPE_JOINT_LIMIT
    # A visit overwrites the impulse with a non-negative one, so a row left at -1 was skipped.
    initial_impulses = np.where(types == vlim, -1.0, 0.0).astype(np.float32)
    initial_impulses[np.arange(rows) >= counts[:, None]] = 0.0
    mu = np.where(types == friction, 0.5, 0.0).astype(np.float32)
    v_in = rng.normal(size=worlds * dofs).astype(np.float32)
    for shared_metadata, pair_cache in ((True, False), (False, False), (False, True)):
        kernel = _get_pgs_solve_mf_gs_kernel(
            rows,
            1,
            dofs,
            wp.get_device(device).arch,
            has_dense_velocity_limit_rows=True,
            shared_metadata=shared_metadata,
            pair_cache=pair_cache,
        )

        def arr(value, dtype=float):
            return wp.array(value, dtype=dtype, device=device)

        def zeros(shape, dtype=float):
            return wp.zeros(shape, dtype=dtype, device=device)

        lam, v = [], []
        for row_types in (types, types_without_inactive):
            impulses = arr(initial_impulses)
            v_out = arr(v_in)
            wp.launch_tiled(
                kernel,
                dim=[worlds],
                inputs=[
                    arr(counts, int),
                    arr(np.arange(worlds * dofs, dtype=np.int32).reshape(worlds, dofs), int),
                    arr(rhs),
                    arr(diag),
                    impulses,
                    arr(J),
                    arr(Y),
                    arr(row_types, int),
                    arr(parents, int),
                    arr(mu),
                    *(zeros((worlds, rows)) for _ in range(5)),
                    zeros(worlds, int),
                    zeros(worlds, int),
                    zeros((worlds, 4), int),
                    zeros((worlds, 1)),
                    *(zeros((worlds, 1, 6)) for _ in range(4)),
                    zeros((worlds, 1)),
                    arr(np.ones((worlds, rows), np.float32)),
                    zeros((worlds, 1)),
                    arr(np.full((worlds, rows), -1, np.int32), int),
                    zeros((worlds, rows)),
                    zeros((worlds, rows)),
                    zeros((worlds, 1)),
                    1,
                    1.0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                outputs=[v_out],
                block_dim=32,
                device=device,
            )
            lam.append(impulses.numpy())
            v.append(v_out.numpy().reshape(worlds, dofs))
        for w, (name, vlim_rows) in enumerate(layouts.items()):
            with test.subTest(shared_metadata=shared_metadata, pair_cache=pair_cache, layout=name):
                test.assertTrue(np.all(lam[0][w, vlim_rows] >= 0.0))
                test.assertFalse(np.any(lam[0][w, counts[w] :]))
                np.testing.assert_array_equal(v[0][w], v[1][w])


class TestSteelMatrixFreeVelocityLimitPass(unittest.TestCase):
    pass


add_function_test(
    TestSteelMatrixFreeVelocityLimitPass,
    "test_velocity_limit_pass_visits_every_active_row",
    test_velocity_limit_pass_visits_every_active_row,
    devices=get_cuda_test_devices(),
)


def test_world_row_slot_reservation_matches_prefix_sums(test, device):
    """Give each element its world's counter plus the earlier row counts of that world, in world-element order.

    Worlds hold no elements, a few, exactly one warp's worth, and more than the largest block.
    """
    rng = np.random.default_rng(11)
    sizes = [0, 3, 32, 33, 0, 1500, 7]
    starts = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int32)
    element_count = int(starts[-1])
    # Elements are listed per world in a shuffled element-id order, as the solver's world indices are.
    elements = rng.permutation(element_count).astype(np.int32)
    counts = rng.integers(0, 7, element_count).astype(np.int32)
    counters = rng.integers(0, 50, len(sizes)).astype(np.int32)
    expected_cursor = np.empty(element_count, np.int32)
    expected_counters = counters.copy()
    for world, (lo, hi) in enumerate(itertools.pairwise(starts)):
        for i in range(lo, hi):
            expected_cursor[elements[i]] = expected_counters[world]
            expected_counters[world] += counts[elements[i]]
    inputs = [wp.array(starts, dtype=int, device=device), wp.array(elements, dtype=int, device=device)]
    # The solver launches only the worlds that have elements.
    listed_worlds = wp.array(np.flatnonzero(sizes), dtype=int, device=device)
    for block_dim in (32, 256, 1024) if wp.get_device(device).is_cuda else (None,):
        cursor = wp.array(counts, dtype=int, device=device)
        world_counters = wp.array(counters, dtype=int, device=device)
        if block_dim is None:
            wp.launch(
                steel_kernels.reserve_world_row_slots_serial,
                dim=len(sizes),
                inputs=inputs,
                outputs=[world_counters, cursor],
                device=device,
            )
        else:
            wp.launch_tiled(
                steel_kernels.reserve_world_row_slots,
                dim=[listed_worlds.shape[0]],
                inputs=[listed_worlds, *inputs],
                outputs=[world_counters, cursor],
                block_dim=block_dim,
                device=device,
            )
        np.testing.assert_array_equal(cursor.numpy(), expected_cursor, err_msg=f"block_dim={block_dim}")
        np.testing.assert_array_equal(world_counters.numpy(), expected_counters, err_msg=f"block_dim={block_dim}")


class TestSteelWorldRowSlotReservation(unittest.TestCase):
    pass


add_function_test(
    TestSteelWorldRowSlotReservation,
    "test_world_row_slot_reservation_matches_prefix_sums",
    test_world_row_slot_reservation_matches_prefix_sums,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
