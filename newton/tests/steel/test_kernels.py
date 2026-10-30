# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kernel selection, launch configuration, streams and double buffering of SolverSteel."""

import types
import unittest
import warnings
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel import kernels as steel_kernels
from newton._src.solvers.steel import solver_steel
from newton._src.solvers.steel._testing import kernel_overrides
from newton._src.solvers.steel.solver_steel import (
    _DENSE_META_MAX_PARENT,
    _STATIC_SHARED_MEMORY_BYTES,
    _estimate_cholesky_shared_memory,
    _estimate_mf_solve_shared_memory,
    _estimate_tiled_row_shared_memory,
    _select_delassus_chunk_size,
    _select_hinv_jt_chunk_size,
    _SteelExecutionPlan,
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

_POINT = {
    "init": _set_box_velocities,
    "steps": 8,
    "friction_anchor_beta": 0.0,
    "solve_mode": "split",
    "articulation_constraint_max": _ROWS,
}


def test_default_launch_settings(test, device):
    """Select the kernels and launch widths automatically, matching explicitly pinned defaults bitwise."""
    solver = SolverSteel(_build_contact_scene(device), solve_mode="split")
    test.assertEqual(solver._pgs_kernel, "auto" if device.is_cuda else "loop")
    test.assertEqual((solver._pgs_chunk_size, solver._tile_threads, solver._serial_kernel_block_dim), (1, 64, 256))
    modes = ({"solve_mode": "split"}, {}) if device.is_cuda else ({"solve_mode": "split"},)
    for mode in modes:
        run = {"init": _set_box_velocities, "steps": 8, **mode}
        explicit = {"pgs_kernel": "auto", "pgs_chunk_size": 1, "tile_threads": 64, "serial_kernel_block_dim": 256}
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
        pinned = SolverSteel(model, solve_mode="split")
        with kernel_overrides({"tile_threads": 128}):
            nested = SolverSteel(model, solve_mode="split")
    default = SolverSteel(model, solve_mode="split")
    test.assertEqual((pinned._tile_threads, pinned._serial_kernel_block_dim), (32, 64))
    test.assertEqual((nested._tile_threads, nested._serial_kernel_block_dim), (128, 256))
    test.assertEqual((default._tile_threads, default._serial_kernel_block_dim), (64, 256))
    test.assertEqual(pinned._kernel_overrides, {"tile_threads": 32, "serial_kernel_block_dim": 64})
    test.assertEqual(default._kernel_overrides, {})


def test_split_kernels_agree_with_the_loop_kernel(test, device):
    """Reach the loop kernel's result with the row kernels, and the block kernels' result with each other."""
    reference = _run(_build_contact_scene(device), overrides={"pgs_kernel": "loop"}, **_POINT)
    for overrides in ({"pgs_kernel": "tiled"}, {"tile_threads": 32}, {"tile_threads": 256}):
        with test.subTest(overrides=overrides):
            result = _run(_build_contact_scene(device), overrides=overrides, **_POINT)
            np.testing.assert_allclose(result[1], reference[1], rtol=0.0, atol=1.0e-5)
            np.testing.assert_allclose(result[2], reference[2], rtol=0.0, atol=1.0e-3)
    block = _run(_build_contact_scene(device), overrides={"pgs_kernel": "tiled_contact"}, **_POINT)
    np.testing.assert_allclose(block[1], reference[1], rtol=0.0, atol=2.0e-2)
    for overrides in ({"pgs_kernel": "streaming"}, {"pgs_kernel": "streaming", "pgs_chunk_size": 4}):
        with test.subTest(overrides=overrides):
            result = _run(_build_contact_scene(device), overrides=overrides, **_POINT)
            np.testing.assert_allclose(result[1], block[1], rtol=0.0, atol=1.0e-5)
            np.testing.assert_allclose(result[2], block[2], rtol=0.0, atol=1.0e-3)


def test_serial_block_dim_and_matrix_free_kernel_choice_are_bitwise(test, device):
    """Leave matrix-free results unchanged by the serial block size and the split-only kernel choice."""
    run = {"init": _set_box_velocities, "steps": 8}
    default = _run(_build_contact_scene(device), **run)
    for overrides in ({"serial_kernel_block_dim": 64}, {"serial_kernel_block_dim": 512}, {"pgs_kernel": "loop"}):
        with test.subTest(overrides=overrides):
            result = _run(_build_contact_scene(device), overrides=overrides, **run)
            _assert_bitwise(test, default, result, str(overrides))


def test_kernel_overrides_validate_their_values(test, device):
    """Reject unknown kernels, unsupported widths and contact-only kernels on rows they cannot solve."""
    model = _build_contact_scene(device)
    for overrides, pattern in (
        ({"pgs_kernel": "fast"}, "pgs_kernel"),
        ({"pgs_chunk_size": 0}, "pgs_chunk_size"),
        ({"tile_threads": 48}, "tile_threads"),
        ({"serial_kernel_block_dim": 0}, "serial_kernel_block_dim"),
        ({"serial_kernel_block_dim": 48}, "serial_kernel_block_dim"),
    ):
        with test.subTest(overrides=overrides), test.assertRaisesRegex(ValueError, pattern):
            with kernel_overrides(overrides):
                SolverSteel(model, solve_mode="split")
    if not device.is_cuda:
        return
    for kernel in ("tiled_contact", "streaming"):
        for options in (
            {"enable_joint_limits": True},
            {"contact_friction_gap_max": 0.01},
            {"contact_friction_position_iterations": 2},
        ):
            with test.subTest(kernel=kernel, options=options), test.assertRaisesRegex(ValueError, "contact rows only"):
                with kernel_overrides({"pgs_kernel": kernel}):
                    SolverSteel(model, solve_mode="split", **options)
    with kernel_overrides({"pgs_kernel": "tiled_contact"}), test.assertRaisesRegex(ValueError, "shared memory"):
        SolverSteel(model, solve_mode="split", articulation_constraint_max=768)


def test_cpu_ignores_contact_only_kernels(test, device):
    """Run the scalar loop on CPU for every split kernel override, including rows the contact-only kernels reject."""
    model = _build_contact_scene(device)
    for kernel in ("tiled_contact", "streaming"):
        for options in ({}, {"contact_friction_gap_max": -float("inf")}, {"enable_joint_limits": True}):
            with test.subTest(kernel=kernel, options=options):
                with kernel_overrides({"pgs_kernel": kernel}):
                    solver = SolverSteel(model, solve_mode="split", **options)
                test.assertEqual(solver._pgs_kernel, "loop")


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
    mode = {} if device.is_cuda else {"solve_mode": "split"}
    run = {"init": _set_box_velocities, "steps": 3, **mode}
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
    mode = {} if device.is_cuda else {"solve_mode": "split"}
    model = _build_contact_scene(device)
    solver = SolverSteel(model, **mode)
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


for _fn in (
    test_split_kernels_agree_with_the_loop_kernel,
    test_serial_block_dim_and_matrix_free_kernel_choice_are_bitwise,
):
    add_function_test(TestSteelAdvancedOptions, _fn.__name__, _fn, devices=get_cuda_test_devices())


add_function_test(
    TestSteelAdvancedOptions,
    test_cpu_ignores_contact_only_kernels.__name__,
    test_cpu_ignores_contact_only_kernels,
    devices=[device for device in get_test_devices() if device.is_cpu],
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
    solver = SolverSteel(_build_chain_model(device, num_links=2, num_worlds=1))
    test.assertEqual(solver.solve_mode, "matrix_free")
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
                SolverSteel(model, solve_mode="matrix_free", **kwargs)
    test.assertEqual(
        SolverSteel(
            model, solve_mode="matrix_free", velocity_limit_activation_fraction=float("inf")
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
    first = SolverSteel(model, solve_mode="matrix_free")
    second = SolverSteel(model, solve_mode="matrix_free")
    for attr in ("_cholesky_kernels_by_size", "_triangular_solve_kernels_by_size", "_hinv_jt_kernels_by_size"):
        first_kernels = getattr(first, attr)
        second_kernels = getattr(second, attr)
        test.assertEqual(set(first_kernels), set(second_kernels))
        for size, kernel in first_kernels.items():
            test.assertIs(kernel, second_kernels[size], f"{attr}[{size}]")
    test.assertIs(first._pgs_solve_mf_gs_kernel, second._pgs_solve_mf_gs_kernel)


def test_compact_world_dof_mapping_pads_heterogeneous_worlds(test, device):
    """Pad the per-world response DOF map of worlds with fewer DOFs."""
    solver = SolverSteel(_build_heterogeneous_world_model(device), solve_mode="matrix_free")
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
        solve_mode="matrix_free",
        articulation_constraint_max=192,
    )
    direct = SolverSteel(
        _build_chain_model(device, num_links=23, num_worlds=1, with_free_body=True),
        solve_mode="matrix_free",
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
            solver = SolverSteel(model, solve_mode="matrix_free", articulation_constraint_max=64)
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
                SolverSteel(build_pendulum_with_connect(eq_type, joint_kind, row_names_world))
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
    # The matching projection is judged through its loop joint instead.
    for eq_type, joint_kind in ((0, "ball"), (1, "fixed")):
        with test.subTest(eq_type=eq_type, joint_kind=joint_kind, matching=True):
            with test.assertRaisesRegex(NotImplementedError, "loop-closing"):
                SolverSteel(build_pendulum_with_connect(eq_type, joint_kind, False))

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
                    SolverSteel(model, solve_mode="matrix_free")
            else:
                SolverSteel(model, solve_mode="matrix_free")


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
    """Reject the options the split solve cannot honor and unknown solve modes."""
    model = _build_chain_model(device, num_links=2, num_worlds=1)
    with test.assertRaisesRegex(NotImplementedError, "requires solve_mode='matrix_free'"):
        SolverSteel(model, solve_mode="split", enable_joint_velocity_limits=True)
    with test.assertRaisesRegex(ValueError, "solve_mode"):
        SolverSteel(model, solve_mode="dense")


def test_split_rejects_matrix_free_only_options(test, device):
    """Reject in the split solve every option that only the matrix-free solve implements."""
    model = _build_chain_model(device, num_links=2, num_worlds=1)
    options = (
        ({"enable_joint_velocity_limits": True}, "enable_joint_velocity_limits"),
        ({"drive_mode": "rows"}, "rows"),
        ({"friction_anchor_beta": 0.2}, "friction_anchor_beta"),
        ({"contact_regularization": 0.02}, "contact_regularization"),
        ({"velocity_iterations": 2}, "velocity_iterations"),
        ({"contact_warmstart": True}, "contact_warmstart"),
        ({"contact_compliance": True, "friction_anchor_beta": 0.0}, "contact_compliance"),
        ({"enable_sleeping": True}, "enable_sleeping"),
    )
    for kwargs, name in options:
        with test.subTest(option=name):
            with test.assertRaisesRegex(NotImplementedError, f"{name}.*requires solve_mode='matrix_free'"):
                SolverSteel(model, solve_mode="split", **kwargs)
    # Contact torsion is CUDA-only; on CUDA the split solve rejects it.
    error = NotImplementedError if wp.get_device(device).is_cuda else ValueError
    set_torsion_radius(model, 0.01)
    with test.assertRaises(error):
        SolverSteel(model, solve_mode="split")
    set_torsion_radius(model, 0.0)
    # The default friction resolves to point friction in the split solve.
    test.assertEqual(SolverSteel(model, solve_mode="split").friction_anchor_beta, 0.0)
    # Positive shape restitution is rejected at construction and when notified.
    restitution = model.shape_material_restitution.numpy().copy()
    model.shape_material_restitution.fill_(0.5)
    with test.assertRaisesRegex(NotImplementedError, "restitution requires solve_mode='matrix_free'"):
        SolverSteel(model, solve_mode="split")
    model.shape_material_restitution.assign(restitution)
    solver = SolverSteel(model, solve_mode="split")
    model.shape_material_restitution.fill_(0.5)
    with test.assertRaisesRegex(NotImplementedError, "restitution requires solve_mode='matrix_free'"):
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)


def test_split_defaults(test, device):
    """Keep the documented defaults in split mode and allocate its Delassus storage."""
    solver = SolverSteel(_build_chain_model(device, num_links=2, num_worlds=2), solve_mode="split")
    test.assertEqual(solver.solve_mode, "split")
    test.assertEqual(solver.iterations, 12)
    test.assertEqual(solver.articulation_constraint_max, 32)
    test.assertEqual(solver._C.shape, (2, 32, 32))
    test.assertFalse(solver._jy_world_aliased)
    test.assertFalse(solver._hinv_jt_writes_world)


def test_split_kernel_selection(test, device):
    """Select the native split-mode kernels on CUDA and the scalar Warp kernels on CPU."""
    is_cuda = wp.get_device(device).is_cuda
    # One 14-DOF chain per world: tiled H^-1 J^T, fused with the Delassus assembly on CUDA.
    single = SolverSteel(_build_chain_model(device, num_links=14, num_worlds=2), solve_mode="split")
    test.assertEqual(single._split_fused_response, is_cuda)
    test.assertEqual(single._pgs_solve_tiled_row_kernel is not None, is_cuda)
    test.assertIsNone(single._pgs_solve_mf_kernel)
    # A free body next to the chain: two solved articulations per world, so no fusion.
    mixed = SolverSteel(_build_chain_model(device, num_links=14, num_worlds=2, with_free_body=True), solve_mode="split")
    test.assertTrue(mixed._has_mixed_contacts)
    test.assertFalse(mixed._split_fused_response)
    test.assertEqual(mixed._max_free_bodies_per_world, 1)
    for size, kernel in mixed._delassus_kernels_by_size.items():
        test.assertEqual(kernel is not None, is_cuda, f"Delassus kernel of size {size}")
    test.assertEqual(mixed._pgs_solve_mf_kernel is not None, is_cuda)
    if not is_cuda:
        test.assertEqual(
            (mixed._cholesky_kernel, mixed._trisolve_kernel, mixed._hinv_jt_kernel, mixed._pgs_kernel),
            ("loop", "loop", "par_row", "loop"),
        )


def test_split_kernel_selection_is_cached(test, device):
    """Resolve identical split-mode solver shapes to the same cached kernel objects."""
    model = _build_chain_model(device, num_links=14, num_worlds=2, with_free_body=True)
    first = SolverSteel(model, solve_mode="split")
    second = SolverSteel(model, solve_mode="split")
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
            model = _build_limited_chain_on_ground(device, 14, with_free_body=with_free_body)
            with kernel_overrides(overrides):
                solver = SolverSteel(
                    model, solve_mode="split", enable_joint_limits=True, articulation_constraint_max=64
                )
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
            test.assertGreater(int(solver._constraint_count.numpy().max()), 0)
            trajectories.append(state_0.joint_q.numpy().copy())
        with test.subTest(with_free_body=with_free_body):
            np.testing.assert_allclose(trajectories[0], trajectories[1], rtol=0.0, atol=1.0e-4)


def test_non_default_tile_threads_compiles_and_steps(test, device):
    """Step a forced tiled split solve whose row capacity is too large to fuse the Delassus assembly."""
    model = _build_chain_model(device)
    forced = {"cholesky_kernel": "tiled", "trisolve_kernel": "tiled", "hinv_jt_kernel": "tiled", "pgs_kernel": "loop"}
    with kernel_overrides(forced):
        solver = SolverSteel(model, solve_mode="split", articulation_constraint_max=384)
    test.assertTrue(all(solver._execution_plan.use_tiled_hinv_jt(size) for size in solver._size_groups))
    test.assertFalse(any(solver._execution_plan.use_fused_hinv_jt(size) for size in solver._size_groups))
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
    for response in ("immediate", "propagation_colored"):
        with test.subTest(response=response):
            with test.assertRaisesRegex(ValueError, "propagation_same_articulation_rows"):
                SolverSteel(model, articulated_contact_response=response, propagation_same_articulation_rows=True)

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
    for response in ("propagation", "propagation_colored"):
        with test.subTest(response=response):
            test.assertEqual(
                SolverSteel(mixed_model, articulated_contact_response=response).articulated_contact_response, response
            )


def test_propagation_rejects_unsupported_options(test, device):
    """Reject the split solve and the options the propagation rows do not implement; accept the rest."""
    model = _build_chain_model(device, num_links=2, num_worlds=1)
    for response in ("propagation", "propagation_colored"):
        with test.subTest(response=response, option="solve_mode"):
            with test.assertRaisesRegex(NotImplementedError, "requires solve_mode='matrix_free'"):
                SolverSteel(model, solve_mode="split", articulated_contact_response=response)
        # The default friction selects friction patches, as with the immediate response.
        test.assertEqual(SolverSteel(model, articulated_contact_response=response).friction_anchor_beta, 0.2)
        set_torsion_radius(model, 0.01)
        with test.subTest(response=response, option="contact torsion"):
            with test.assertRaisesRegex(NotImplementedError, "contact torsion.*articulated_contact_response"):
                SolverSteel(model, articulated_contact_response=response)
        set_torsion_radius(model, 0.0)
        options = (
            ({"contact_compliance": True, "friction_anchor_beta": 0.0}, "contact_compliance"),
            ({"enable_sleeping": True}, "enable_sleeping"),
        )
        for kwargs, name in options:
            with test.subTest(response=response, option=name):
                with test.assertRaisesRegex(NotImplementedError, f"{name}.*articulated_contact_response"):
                    SolverSteel(model, articulated_contact_response=response, **kwargs)
        for kwargs in (
            {"drive_mode": "rows"},
            {"contact_regularization": 0.02},
            {"velocity_iterations": 2},
            {"contact_warmstart": True},
        ):
            with test.subTest(response=response, **kwargs):
                solver = SolverSteel(model, articulated_contact_response=response, **kwargs)
                test.assertTrue(solver._propagation_active)
        restitution = model.shape_material_restitution.numpy().copy()
        model.shape_material_restitution.fill_(0.5)
        with test.subTest(response=response, option="restitution"):
            solver = SolverSteel(model, articulated_contact_response=response)
            solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        model.shape_material_restitution.assign(restitution)


def test_propagation_matrix_free_compiles_and_steps_with_velocity_limit_rows(test, device):
    """Step both propagation responses with dense joint velocity-limit rows."""
    model = _build_chain_model(device, num_links=3, num_worlds=1)
    model.joint_velocity_limit.assign(np.full(model.joint_dof_count, 0.1, dtype=np.float32))
    for response in ("propagation", "propagation_colored"):
        with test.subTest(response=response):
            solver = SolverSteel(
                model,
                articulated_contact_response=response,
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
    def test_cpu_construction_raises(self):
        """Reject the default CUDA-only matrix-free solve on CPU, naming the split solve, which constructs."""
        model = _build_chain_model("cpu", num_links=2, num_worlds=1)
        for options in ({}, {"solve_mode": "matrix_free"}):
            with self.subTest(**options):
                with self.assertRaisesRegex(
                    NotImplementedError, "requires a CUDA device; use solve_mode='split' on CPU"
                ):
                    SolverSteel(model, **options)
        self.assertEqual(SolverSteel(model, solve_mode="split").solve_mode, "split")

    def test_hinv_fusion_requires_full_working_set_to_fit(self):
        """Fuse H^-1 J^T with the Delassus assembly only when the whole row set and its Delassus tile fit."""
        plan_args = {
            "max_shared_memory": 101376,
            "cholesky_kernel": "auto",
            "hinv_jt_kernel": "auto",
            "small_dof_threshold": 12,
            "tile_threads": 64,
        }
        fitting = _SteelExecutionPlan.build([23], max_constraints=64, **plan_args)
        oversized = _SteelExecutionPlan.build([23], max_constraints=384, **plan_args)
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
    test.assertEqual(SolverSteel(model, solve_mode="matrix_free", drive_mode="rows").drive_mode, "rows")
    with test.assertRaisesRegex(NotImplementedError, "requires solve_mode='matrix_free'"):
        SolverSteel(model, solve_mode="split", drive_mode="rows")
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
            test.assertFalse(SolverSteel(model, solve_mode="matrix_free", **kwargs).velocity_limits_fused)
    # The applicable combination engages by default.
    solver = SolverSteel(model, solve_mode="matrix_free", drive_mode="rows", enable_joint_velocity_limits=True)
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
            solve_mode="matrix_free",
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
        "test_propagation_matrix_free_compiles_and_steps_with_velocity_limit_rows",
        test_propagation_matrix_free_compiles_and_steps_with_velocity_limit_rows,
    ),
):
    add_function_test(TestSteelLaunchConfig, _name, _func, devices=devices)


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
    add_function_test(TestSteelLaunchConfig, _name, _func, devices=_devices)


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


def test_contact_kernels_match_the_scalar_split_solve(test, device):
    """Converge the split solve's contact-only Gauss-Seidel kernels to the scalar loop's solution.

    They sweep whole contacts as 3x3 blocks, so they agree with the row-by-row loop at convergence.
    """
    model = _legged_base(device)
    options = {"solve_mode": "split", "articulation_constraint_max": _ROWS, "iterations": 64}
    with _overrides(pgs_kernel="loop"):
        _, reference = _legged_trajectory(model, steps=30, **options)
    for overrides in (
        {"pgs_kernel": "tiled_contact"},
        {"pgs_kernel": "streaming"},
        {"pgs_kernel": "streaming", "pgs_chunk_size": 4},
    ):
        with test.subTest(**overrides):
            with _overrides(**overrides):
                solver, joint_q = _legged_trajectory(model, steps=30, **options)
            test.assertIsNotNone(solver._pgs_solve_contact_kernel)
            test.assertGreater(int(solver._constraint_count.numpy().max()), 0)
            np.testing.assert_allclose(joint_q, reference, rtol=0.0, atol=1.0e-5)


def test_matrix_free_ignores_the_dense_pgs_kernel(test, device):
    """Resolve the dense Gauss-Seidel kernel to the scalar loop in the matrix-free solve; keep friction patches."""
    model = _legged_base(device)
    for kernel in ("tiled_contact", "streaming", "tiled"):
        with test.subTest(pgs_kernel=kernel), _overrides(pgs_kernel=kernel), warnings.catch_warnings():
            warnings.simplefilter("error")
            solver = SolverSteel(model, enable_joint_limits=True)
            test.assertEqual(solver._pgs_kernel, "loop")
            test.assertTrue(solver._friction_anchors_enabled)
            test.assertAlmostEqual(solver.friction_anchor_beta, 0.2)


def test_launch_widths_step_identically(test, device, solve_mode="matrix_free"):
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


for _fn in (
    test_contact_kernels_match_the_scalar_split_solve,
    test_matrix_free_ignores_the_dense_pgs_kernel,
    test_launch_widths_step_identically,
):
    add_function_test(TestSteelKernelOverrides, _fn.__name__, _fn, devices=cuda_devices)


add_function_test(
    TestSteelKernelOverrides,
    "test_launch_widths_step_identically_split",
    test_launch_widths_step_identically,
    devices=cuda_devices,
    solve_mode="split",
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


def test_streams_and_double_buffer_step_identically(test, device, solve_mode="matrix_free"):
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


def test_captured_double_buffer_matches_eager(test, device, solve_mode="matrix_free"):
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
        solver = SolverSteel(model, solve_mode="matrix_free", iterations=16, baumgarte=0.1, double_buffer=double_buffer)
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
            solve_mode="matrix_free",
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
        solver = SolverSteel(_scene(device), solve_mode="split", use_parallel_streams=True, double_buffer=True)
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
    add_function_test(TestSteelStreams, f"{_fn.__name__}_split", _fn, devices=cuda_devices, solve_mode="split")


add_function_test(
    TestSteelStreams,
    "test_flags_are_ignored_where_they_do_not_apply",
    test_flags_are_ignored_where_they_do_not_apply,
    devices=get_test_devices(),
)


def test_fused_assembly_matches_the_separate_pipeline(test, device, solve_mode="matrix_free"):
    """Match separate assembly and factorization with the warp-per-articulation and tiled kernels."""
    for links, branched, kernel in ((3, False, "warp"), (6, True, "warp"), (14, False, "tiled"), (14, True, "tiled")):
        with test.subTest(links=links, branched=branched):
            model = _tree(device, links, branched=branched)
            options = {"solve_mode": solve_mode, "articulation_constraint_max": _ROWS}
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
    solve_mode = "matrix_free" if is_cuda else "split"
    solver = SolverSteel(_tree(device, 3), solve_mode=solve_mode, use_parallel_streams=True)
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
    "test_fused_assembly_matches_the_separate_pipeline_split",
    test_fused_assembly_matches_the_separate_pipeline,
    devices=cuda_devices,
    solve_mode="split",
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
                    target_pos=0.3,
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


def _row_phase_rollout(model, *, runtime_phase=False, capture=False, **options):
    options = {"enable_joint_limits": True, "drive_mode": "rows", **options}
    solver = SolverSteel(
        model,
        solve_mode="matrix_free",
        iterations=12,
        articulation_constraint_max=_ROWS,
        free_body_constraint_max=_ROWS,
        deterministic=wp.DeterministicMode.RUN_TO_RUN,
        **options,
    )
    prebuilt = set(solver._pgs_solve_mf_gs_kernels)
    if runtime_phase:
        # Every launch takes the kernel that reads the phase argument.
        solver._pgs_solve_mf_gs_kernels = {0: solver._pgs_solve_mf_gs_kernel_for_phase()}
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
    _, _, runtime = _row_phase_rollout(model, runtime_phase=True, **options)
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


if __name__ == "__main__":
    unittest.main()
