# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver-wide physics overrides of SolverFeatherPGS: damping, friction and restitution switches."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_FRICTION
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_contact_compliance import run_fixture as run_compliance_fixture
from newton.tests.test_feather_pgs_local_owned import _build_mixed_response_model
from newton.tests.test_feather_pgs_restitution import _build_plane_model, _reset_state
from newton.tests.test_feather_pgs_sparse_diagonal import _build_sparse_contact_friction_model
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

DT = 1.0 / 240.0
FLOAT32_MAX = float(np.finfo(np.float32).max)


def _build_contact_scene(device, *, mu=0.6, restitution=0.0, worlds=2):
    """Build a sliding articulated box with a driven arm and two sliding free boxes per world."""
    scene = newton.ModelBuilder()
    scene.default_shape_cfg.mu = mu
    scene.default_shape_cfg.restitution = restitution
    scene.add_ground_plane()
    for i in range(2):
        body = scene.add_body(xform=wp.transform(wp.vec3(0.4 * i, 0.6, 0.099), wp.quat_rpy(0.0, 0.0, 0.2 * i)))
        scene.add_shape_box(body, hx=0.1, hy=0.08, hz=0.1)
    xform = wp.transform(wp.vec3(0.0, 0.0, 0.049), wp.quat_identity())
    base = scene.add_link(xform=xform)
    scene.add_shape_box(base, hx=0.15, hy=0.1, hz=0.05)
    arm = scene.add_link()
    scene.add_shape_capsule(arm, radius=0.03, half_height=0.08)
    axis = newton.ModelBuilder.JointDofConfig
    scene.add_articulation(
        [
            scene.add_joint_d6(
                parent=-1,
                child=base,
                parent_xform=xform,
                linear_axes=[axis(axis=newton.Axis.X), axis(axis=newton.Axis.Z)],
            ),
            scene.add_joint_revolute(
                base,
                arm,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.12), wp.quat_identity()),
                target_ke=20.0,
                target_kd=1.0,
                target_pos=0.4,
            ),
        ]
    )
    builder = newton.ModelBuilder()
    builder.replicate(scene, worlds)
    return builder.finalize(device=device)


def _set_box_velocities(model, state, speed=1.5):
    """Give every free box and articulated box a horizontal velocity so friction does work."""
    joint_qd = state.joint_qd.numpy()
    starts = model.joint_qd_start.numpy()
    for joint, joint_type in enumerate(model.joint_type.numpy()):
        if joint_type in (int(newton.JointType.FREE), int(newton.JointType.D6)):
            joint_qd[starts[joint]] = speed
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def _friction_impulses(solver):
    """Return every friction row impulse of the dense, free-body and propagation row families."""
    values = []
    families = [
        (solver.constraint_count, solver.row_type, solver.impulses),
        (solver.mf_constraint_count, solver.mf_row_type, solver.mf_impulses),
    ]
    if solver._propagation_active:
        families.append((solver.propagation_constraint_count, solver.propagation_row_type, solver.propagation_impulses))
    for count_array, type_array, impulse_array in families:
        row_type = type_array.numpy()
        impulses = impulse_array.numpy()
        for world, rows in enumerate(count_array.numpy()):
            friction = row_type[world, :rows] == PGS_CONSTRAINT_TYPE_FRICTION
            values.append(impulses[world, :rows][friction])
    return np.concatenate(values) if values else np.zeros(0)


def _run(model, *, steps=12, init=None, overrides=None, capture=False, **options):
    """Step ``model`` and return the solver, the final joint state and each step's friction impulses."""
    with mock.patch.object(SolverFeatherPGS, "_kernel_overrides", overrides or {}):
        solver = SolverFeatherPGS(model, **options)
    state_in, state_out = model.state(), model.state()
    if init is not None:
        init(model, state_in)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False)
    contacts = pipeline.contacts()
    control = model.control()
    friction = []

    def step():
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, DT)
        for src, dst in (
            (state_out.joint_q, state_in.joint_q),
            (state_out.joint_qd, state_in.joint_qd),
            (state_out.body_q, state_in.body_q),
            (state_out.body_qd, state_in.body_qd),
        ):
            wp.copy(dst, src)

    if capture:
        step()
        friction.append(_friction_impulses(solver))
        with wp.ScopedCapture(model.device) as graph:
            step()
        for _ in range(steps - 1):
            wp.capture_launch(graph.graph)
            friction.append(_friction_impulses(solver))
    else:
        for _ in range(steps):
            step()
            friction.append(_friction_impulses(solver))
    return solver, state_in.joint_q.numpy(), state_in.joint_qd.numpy(), friction


def _assert_bitwise(test, a, b, label):
    """Assert two runs produced identical joint states."""
    np.testing.assert_array_equal(a[1], b[1], err_msg=f"{label}: joint_q")
    np.testing.assert_array_equal(a[2], b[2], err_msg=f"{label}: joint_qd")


# ---------------------------------------------------------------------------------------------------------------
# angular_damping


def _build_spinning_model(device, *, register=False, damping=None):
    """Build a spinning sphere and a floating base with a hinged child, without gravity or contacts."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    if register:
        SolverFeatherPGS.register_custom_attributes(builder)
    attributes = None if damping is None else {"rigid_body_angular_damping": damping}
    sphere = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3) * 0.01),
        lock_inertia=True,
        custom_attributes=attributes,
    )
    builder.add_shape_sphere(sphere, radius=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    base = builder.add_link(
        xform=wp.transform(wp.vec3(2.0, 0.0, 1.0), wp.quat_identity()), custom_attributes=attributes
    )
    builder.add_shape_box(base, hx=0.2, hy=0.1, hz=0.1)
    child = builder.add_link(custom_attributes=attributes)
    builder.add_shape_box(child, hx=0.1, hy=0.05, hz=0.05)
    builder.add_articulation(
        [
            builder.add_joint_free(parent=-1, child=base),
            builder.add_joint_revolute(
                parent=base,
                child=child,
                axis=wp.vec3(0.0, 0.0, 1.0),
                parent_xform=wp.transform(wp.vec3(0.3, 0.0, 0.0), wp.quat_identity()),
            ),
        ]
    )
    return builder.finalize(device=device)


def _spin(model, state):
    """Spin both free roots about z and the hinge."""
    joint_qd = np.zeros(model.joint_dof_count, dtype=np.float32)
    joint_qd[5] = 4.0
    joint_qd[11] = 3.0
    joint_qd[12] = 1.0
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def _split_or_default(device):
    return {} if device.is_cuda else {"pgs_mode": "split"}


def test_angular_damping_matches_the_per_body_attribute(test, device):
    """Apply one global value exactly as a per-body attribute of that value, and override the attribute."""
    mode = _split_or_default(device)
    run = {"init": _spin, "steps": 20, **mode}
    default = _run(_build_spinning_model(device), **run)
    explicit = _run(_build_spinning_model(device), angular_damping=0.05, **run)
    _assert_bitwise(test, default, explicit, "angular_damping=0.05 vs default")
    attribute = _run(_build_spinning_model(device, register=True, damping=0.3), **run)
    for damping in (None, 0.0):
        with test.subTest(model_damping=damping):
            model = _build_spinning_model(device, register=damping is not None, damping=damping)
            _assert_bitwise(test, _run(model, angular_damping=0.3, **run), attribute, "override vs attribute")
    undamped = _run(_build_spinning_model(device), angular_damping=0.0, **run)
    test.assertGreater(float(np.abs(undamped[2] - default[2]).max()), 1.0e-3)


def test_angular_damping_decays_free_root_spin_analytically(test, device):
    """Scale a free sphere's spin by ``1 - d dt`` per step and leave hinge rates alone."""
    damping = 0.4
    steps = 25
    solver, _, joint_qd, _ = _run(
        _build_spinning_model(device), init=_spin, steps=steps, angular_damping=damping, **_split_or_default(device)
    )
    test.assertAlmostEqual(float(joint_qd[5]), 4.0 * (1.0 - damping * DT) ** steps, delta=1.0e-5)
    test.assertEqual(solver.angular_damping, damping)
    np.testing.assert_allclose(solver.rigid_body_angular_damping.numpy(), damping)


def test_angular_damping_validates_its_value(test, device):
    """Reject negative, nonfinite and non-numeric damping."""
    model = _build_spinning_model(device)
    for value in (-0.1, float("nan"), float("inf"), "fast"):
        with test.subTest(value=value), test.assertRaisesRegex(ValueError, "angular_damping"):
            SolverFeatherPGS(model, angular_damping=value, **_split_or_default(device))


def test_angular_damping_survives_graph_capture(test, device):
    """Replay a captured step with the global damping exactly like eager steps."""
    run = {"init": _spin, "steps": 6, "angular_damping": 0.3}
    _assert_bitwise(
        test,
        _run(_build_spinning_model(device), capture=True, **run),
        _run(_build_spinning_model(device), **run),
        "captured",
    )


# ---------------------------------------------------------------------------------------------------------------
# enable_contact_friction and contact_friction_scale

_SOLVES = {
    "matrix_free": {},
    "physx_grasp": {"pgs_schedule": "physx_grasp"},
    "propagation": {"articulated_contact_response": "propagation"},
    "propagation-fused": {"articulated_contact_response": "propagation-fused"},
    "propagation-colored": {"articulated_contact_response": "propagation-colored"},
    "split": {"pgs_mode": "split"},
}


def _solves(device):
    return _SOLVES.items() if device.is_cuda else [("split", _SOLVES["split"])]


def test_disabled_contact_friction_builds_normal_rows_only(test, device):
    """Give every contact a normal row only, exactly as an unreachable friction gap threshold does."""
    for label, options in _solves(device):
        with test.subTest(solve=label):
            run = {"init": _set_box_velocities, "steps": 8, **options}
            disabled = _run(_build_contact_scene(device), enable_contact_friction=False, **run)
            gated = _run(_build_contact_scene(device), contact_friction_gap_threshold=-float("inf"), **run)
            _assert_bitwise(test, disabled, gated, f"{label}: disabled vs -inf gap threshold")
            for pairs_only in (False, True):
                pairs = _run(
                    _build_contact_scene(device),
                    enable_contact_friction=False,
                    contact_friction_articulation_pairs_only=pairs_only,
                    **run,
                )
                _assert_bitwise(test, disabled, pairs, f"{label}: pairs_only={pairs_only}")
            test.assertTrue(all(f.size == 0 for f in disabled[3]), f"{label}: friction rows were created")
            default = _run(_build_contact_scene(device), **run)
            test.assertTrue(any(f.size for f in default[3]), f"{label}: the scene creates no friction rows")
            test.assertGreater(float(np.abs(default[2] - disabled[2]).max()), 1.0e-3)


def test_contact_friction_scale_matches_scaled_materials(test, device):
    """Scale friction exactly as scaling every shape's coefficient does, and keep 1.0 bitwise."""
    for label, options in _solves(device):
        with test.subTest(solve=label):
            run = {"init": _set_box_velocities, "steps": 8, **options}
            default = _run(_build_contact_scene(device), **run)
            _assert_bitwise(test, default, _run(_build_contact_scene(device), contact_friction_scale=1.0, **run), label)
            scaled = _run(_build_contact_scene(device), contact_friction_scale=0.5, **run)
            _assert_bitwise(test, scaled, _run(_build_contact_scene(device, mu=0.3), **run), f"{label}: 0.5x")
            test.assertGreater(float(np.abs(default[2] - scaled[2]).max()), 1.0e-3)


def test_contact_friction_scale_applies_to_notified_materials(test, device):
    """Rescale the solver's friction copy whenever SHAPE_PROPERTIES publishes new coefficients."""
    model = _build_contact_scene(device)
    solver = SolverFeatherPGS(model, contact_friction_scale=0.25, **_split_or_default(device))
    np.testing.assert_allclose(solver.shape_material_mu.numpy()[: model.shape_count], 0.25 * 0.6, rtol=1.0e-7)
    model.shape_material_mu = wp.full(model.shape_count, 0.8, dtype=float, device=device)
    solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    np.testing.assert_allclose(solver.shape_material_mu.numpy()[: model.shape_count], 0.2, rtol=1.0e-7)
    np.testing.assert_allclose(model.shape_material_mu.numpy(), 0.8)


def test_friction_switches_validate_their_values(test, device):
    """Reject invalid friction scales and torsion without friction rows."""
    model = _build_contact_scene(device)
    mode = _split_or_default(device)
    for value in (-0.5, float("nan"), float("inf")):
        with test.subTest(scale=value), test.assertRaisesRegex(ValueError, "contact_friction_scale"):
            SolverFeatherPGS(model, contact_friction_scale=value, **mode)
    if device.is_cuda:
        with test.assertRaisesRegex(ValueError, "enable_contact_friction"):
            SolverFeatherPGS(model, enable_contact_friction=False, contact_torsion_radius=0.01)


# ---------------------------------------------------------------------------------------------------------------
# contact_friction_position_iterations


def _friction_start_runs(device):
    """Yield (label, model factory, options, required path flag) covering every position-solve path."""
    for label, options in _solves(device):
        yield label, lambda: _build_contact_scene(device), {"init": _set_box_velocities, **options}, None
    if not device.is_cuda:
        return
    yield (
        "split-loop",
        lambda: _build_contact_scene(device),
        {"init": _set_box_velocities, "pgs_mode": "split", "overrides": {"pgs_kernel": "loop"}},
        None,
    )
    mixed = {
        "dof_count": 13,
        "friction": 0.7,
        "static_plane": True,
    }

    def mixed_init(model, state):
        joint_qd = state.joint_qd.numpy()
        free_joint = int(np.flatnonzero(model.joint_type.numpy() == int(newton.JointType.FREE))[0])
        start = int(model.joint_qd_start.numpy()[free_joint])
        joint_qd[start] = 2.0
        joint_qd[start + 2] = -3.0
        state.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    common = {
        "init": mixed_init,
        "enable_joint_limits": True,
        "joint_limit_activation_gap": 0.0,
        "friction_anchor_beta": 0.0,
        "dense_max_constraints": 64,
        "mf_max_constraints": 32,
    }
    yield (
        "local",
        lambda: _build_mixed_response_model(device, **mixed),
        {**common, "overrides": {"hinv_jt_kernel": "par_row"}},
        "_local_internal_fast_path",
    )
    yield (
        "paired",
        lambda: _build_mixed_response_model(device, **{**mixed, "dof_count": 23}),
        {**common, "dense_max_constraints": 96},
        "_paired_factor_coordinates",
    )

    def slide_branches(model, state):
        state.joint_qd.fill_(-0.5)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    yield (
        "sparse-diagonal",
        lambda: _build_sparse_contact_friction_model(device=device),
        {
            "init": slide_branches,
            "enable_joint_limits": True,
            "friction_anchor_beta": 0.0,
            "dense_max_constraints": 96,
            "use_parallel_streams": True,
        },
        "_sparse_diagonal_contact_solve",
    )


def test_friction_position_iterations_gate_every_solve_path(test, device):
    """Hold friction at zero before the last k position iterations on every solve path.

    ``k = pgs_iterations`` (and larger) is bitwise the default, ``k = 0`` leaves every friction row at zero,
    and an intermediate ``k`` solves friction, ending with a state different from both.
    """
    iterations = 8
    for label, build, options, path in _friction_start_runs(device):
        with test.subTest(solve=label):
            run = {"steps": 6, "pgs_iterations": iterations, **options}
            default = _run(build(), **run)
            if path is not None:
                test.assertTrue(getattr(default[0], path), f"{label}: {path} was not selected")
            for k in (iterations, iterations + 5):
                gated = _run(build(), contact_friction_position_iterations=k, **run)
                _assert_bitwise(test, default, gated, f"{label}: k={k}")
                if path is not None:
                    test.assertTrue(getattr(gated[0], path), f"{label}: {path} was not selected with k={k}")
            none = _run(build(), contact_friction_position_iterations=0, **run)
            test.assertTrue(any(f.size for f in none[3]), f"{label}: the scene creates no friction rows")
            for step, friction in enumerate(none[3]):
                np.testing.assert_array_equal(friction, 0.0, err_msg=f"{label}: k=0 step {step}")
            late = _run(build(), contact_friction_position_iterations=3, **run)
            test.assertTrue(any(np.any(f != 0.0) for f in late[3]), f"{label}: k=3 solved no friction")
            test.assertGreater(float(np.abs(late[2] - none[2]).max()), 0.0, f"{label}: k=3 equals k=0")
            test.assertGreater(float(np.abs(late[2] - default[2]).max()), 0.0, f"{label}: k=3 equals default")


def test_velocity_iterations_always_solve_friction(test, device):
    """Solve friction in the velocity-only iterations even when no position iteration does."""
    run = {"init": _set_box_velocities, "steps": 4, "pgs_velocity_iterations": 4}
    gated = _run(_build_contact_scene(device), contact_friction_position_iterations=0, **run)
    test.assertTrue(any(np.any(f != 0.0) for f in gated[3]))


def test_friction_position_iterations_survive_graph_capture(test, device):
    """Replay a captured step with gated friction exactly like eager steps."""
    run = {"init": _set_box_velocities, "steps": 6, "contact_friction_position_iterations": 3}
    for label, options in _SOLVES.items():
        with test.subTest(solve=label):
            _assert_bitwise(
                test,
                _run(_build_contact_scene(device), capture=True, **run, **options),
                _run(_build_contact_scene(device), **run, **options),
                label,
            )


def test_friction_position_iterations_validate_their_value(test, device):
    """Reject values below -1 and the options whose friction state the gate cannot hold."""
    model = _build_contact_scene(device)
    mode = _split_or_default(device)
    with test.assertRaisesRegex(ValueError, "contact_friction_position_iterations"):
        SolverFeatherPGS(model, contact_friction_position_iterations=-2, **mode)
    if device.is_cuda:
        with test.assertRaisesRegex(NotImplementedError, "pgs_warmstart"):
            SolverFeatherPGS(model, contact_friction_position_iterations=2, pgs_warmstart=True)
        with test.assertRaisesRegex(ValueError, "contact_friction_position_iterations"):
            run_compliance_fixture(
                device=device,
                articulated=True,
                enabled=True,
                steps=1,
                solver_options={"contact_friction_position_iterations": 2},
            )


# ---------------------------------------------------------------------------------------------------------------
# enable_restitution


def _bounce(device, scene, **options):
    """Return the post-impact vertical velocity of a sphere with restitution 0.8 hitting a plane."""
    model, body = _build_plane_model(device, separation=0.0008, restitution=0.8, scene=scene)
    solver = SolverFeatherPGS(model, pgs_iterations=16, pgs_cfm=0.0, **options)
    solver.rigid_body_angular_damping.zero_()
    state_in, state_out = model.state(), model.state()
    _reset_state(model, state_in, -2.0)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, 1.0e-3)
    return state_out.body_qd.numpy()[body].copy(), state_out.body_q.numpy()[body].copy()


def test_disabled_restitution_matches_the_largest_threshold(test, device):
    """Turn rebounds off exactly as the float32-maximum threshold does, on free and articulated contacts."""
    for scene in ("free", "articulated"):
        with test.subTest(scene=scene):
            bouncing = _bounce(device, scene, restitution_velocity_threshold=0.0)
            disabled = _bounce(device, scene, enable_restitution=False, restitution_velocity_threshold=0.0)
            threshold = _bounce(device, scene, restitution_velocity_threshold=FLOAT32_MAX)
            test.assertGreater(float(bouncing[0][2]), 1.0)
            for got, expected in zip(disabled, threshold, strict=True):
                np.testing.assert_array_equal(got, expected)
    _assert_bitwise(
        test,
        _run(_build_contact_scene(device, restitution=0.5), init=_set_box_velocities),
        _run(_build_contact_scene(device, restitution=0.5), init=_set_box_velocities, enable_restitution=True),
        "enable_restitution=True vs default",
    )


def test_disabled_restitution_lets_split_accept_restitution(test, device):
    """Construct and notify a split solve with positive shape restitution once restitution is disabled."""
    model = _build_contact_scene(device, restitution=0.5)
    with test.assertRaisesRegex(NotImplementedError, "restitution"):
        SolverFeatherPGS(model, pgs_mode="split")
    disabled = _run(model, init=_set_box_velocities, pgs_mode="split", enable_restitution=False)
    inelastic = _run(_build_contact_scene(device), init=_set_box_velocities, pgs_mode="split")
    _assert_bitwise(test, disabled, inelastic, "split")
    model.shape_material_restitution.fill_(0.9)
    disabled[0].notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)


def test_disabled_restitution_lets_compliance_accept_restitution(test, device):
    """Run compliant contacts on shapes with positive restitution once restitution is disabled."""
    options = {"solver_options": {"enable_restitution": False}, "articulated": True, "enabled": True, "steps": 20}
    trace, _, _, _ = run_compliance_fixture(device=device, restitution=0.5, **options)
    reference, _, _, _ = run_compliance_fixture(device=device, restitution=0.0, **options)
    np.testing.assert_array_equal(trace, reference)


class TestFeatherPGSGlobalOptions(unittest.TestCase):
    pass


for _fn in (
    test_angular_damping_matches_the_per_body_attribute,
    test_angular_damping_decays_free_root_spin_analytically,
    test_angular_damping_validates_its_value,
    test_disabled_contact_friction_builds_normal_rows_only,
    test_contact_friction_scale_matches_scaled_materials,
    test_contact_friction_scale_applies_to_notified_materials,
    test_friction_switches_validate_their_values,
    test_friction_position_iterations_gate_every_solve_path,
    test_friction_position_iterations_validate_their_value,
    test_disabled_restitution_lets_split_accept_restitution,
):
    add_function_test(TestFeatherPGSGlobalOptions, _fn.__name__, _fn, devices=get_test_devices())

for _fn in (
    test_angular_damping_survives_graph_capture,
    test_velocity_iterations_always_solve_friction,
    test_friction_position_iterations_survive_graph_capture,
    test_disabled_restitution_matches_the_largest_threshold,
    test_disabled_restitution_lets_compliance_accept_restitution,
):
    add_function_test(TestFeatherPGSGlobalOptions, _fn.__name__, _fn, devices=get_cuda_test_devices())


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
