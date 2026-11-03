# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver-wide options, global worlds, target layouts and the private API surface of SolverSteel."""

import ast
import inspect
import re
import typing
import unittest
import warnings
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel.solver_steel import _ConstructionOption, _StepOption
from newton.solvers import SolverSteel
from newton.tests.steel._fixtures import (
    DT,
    _build_contact_scene,
    _build_mixed_response_model,
    _build_plane_model,
    _build_sparse_contact_friction_model,
    _reset_state,
    _run,
    _set_box_velocities,
)
from newton.tests.steel._fixtures import run_fixture as run_compliance_fixture
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _assert_bitwise(test, a, b, label):
    """Assert two runs produced identical joint states."""
    np.testing.assert_array_equal(a[1], b[1], err_msg=f"{label}: joint_q")
    np.testing.assert_array_equal(a[2], b[2], err_msg=f"{label}: joint_qd")


def _build_spinning_model(device, *, register=False, damping=None):
    """Build a spinning sphere and a floating base with a hinged child, without gravity or contacts."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    if register:
        SolverSteel.register_custom_attributes(builder)
    attributes = None if damping is None else {"steel:rigid_body_angular_damping": damping}
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
    return {} if device.is_cuda else {"solve_mode": "split"}


def test_angular_damping_defaults_to_the_featherstone_value(test, device):
    """Damp free roots by 0.05 without the attribute, exactly as a registered attribute of that value."""
    mode = _split_or_default(device)
    run = {"init": _spin, "steps": 20, **mode}
    default = _run(_build_spinning_model(device), **run)
    registered = _run(_build_spinning_model(device, register=True, damping=0.05), **run)
    _assert_bitwise(test, default, registered, "unregistered vs 0.05 attribute")
    undamped = _run(_build_spinning_model(device, register=True, damping=0.0), **run)
    test.assertGreater(float(np.abs(undamped[2] - default[2]).max()), 1.0e-3)


def test_angular_damping_decays_free_root_spin_analytically(test, device):
    """Scale a free sphere's spin by ``1 - d dt`` per step and leave hinge rates alone."""
    damping = 0.4
    steps = 25
    _, _, joint_qd, _ = _run(
        _build_spinning_model(device, register=True, damping=damping),
        init=_spin,
        steps=steps,
        **_split_or_default(device),
    )
    test.assertAlmostEqual(float(joint_qd[5]), 4.0 * (1.0 - damping * DT) ** steps, delta=1.0e-5)


def test_angular_damping_follows_in_place_edits(test, device):
    """Apply an in-place edit of the attribute on the next step, as on the per-body path."""
    model = _build_spinning_model(device, register=True, damping=0.3)
    solver = SolverSteel(model, **_split_or_default(device))
    state_in, state_out = model.state(), model.state()
    _spin(model, state_in)
    model.steel.rigid_body_angular_damping.fill_(0.0)
    solver.step(state_in, state_out, model.control(), None, DT)
    test.assertAlmostEqual(float(state_out.joint_qd.numpy()[5]), 4.0, delta=1.0e-5)


def test_physx_attributes_import_from_usd_and_are_read_every_step(test, device):
    """Import the PhysX body and collision USD attributes into the steel namespace, and step with replaced arrays."""
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    cube = UsdGeom.Cube.Define(stage, "/Cube")
    cube.CreateSizeAttr(0.2)
    cube.AddTranslateOp().Set((0.0, 0.0, 1.0))
    prim = cube.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(1.0)
    for name, value in (
        ("physxRigidBody:angularDamping", 0.3),
        ("physxRigidBody:maxLinearVelocity", 2.0),
        ("physxRigidBody:maxAngularVelocity", 90.0),
        ("physxRigidBody:maxDepenetrationVelocity", 1.5),
        ("physxCollision:minTorsionalPatchRadius", 0.004),
    ):
        prim.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(value)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverSteel.register_custom_attributes(builder)
    builder.add_usd(stage)
    model = builder.finalize(device=device)
    np.testing.assert_allclose(model.steel.rigid_body_angular_damping.numpy(), [0.3])
    np.testing.assert_allclose(model.steel.rigid_body_max_linear_velocity.numpy(), [2.0])
    np.testing.assert_allclose(model.steel.rigid_body_max_angular_velocity.numpy(), [0.5 * np.pi], rtol=1.0e-6)
    np.testing.assert_allclose(model.steel.rigid_body_max_depenetration_velocity.numpy(), [1.5])
    np.testing.assert_allclose(model.steel.min_torsional_patch_radius.numpy(), [0.004])
    # Contact torsion is CUDA-only; this check covers the body attributes.
    model.steel.min_torsional_patch_radius.zero_()

    solver = SolverSteel(model, **_split_or_default(device))
    model.steel.rigid_body_angular_damping = wp.zeros(1, dtype=wp.float32, device=device)
    model.steel.rigid_body_max_angular_velocity = wp.full(1, 100.0, dtype=wp.float32, device=device)
    state_in, state_out = model.state(), model.state()
    joint_qd = state_in.joint_qd.numpy()
    joint_qd[5] = 4.0
    state_in.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    solver.step(state_in, state_out, model.control(), None, DT)
    # The replaced arrays apply: no damping and no clamp at the imported 90 deg/s.
    test.assertAlmostEqual(float(state_out.joint_qd.numpy()[5]), 4.0, delta=1.0e-5)


def test_angular_damping_survives_graph_capture(test, device):
    """Replay a captured step with per-body damping exactly like eager steps."""
    run = {"init": _spin, "steps": 6}
    _assert_bitwise(
        test,
        _run(_build_spinning_model(device, register=True, damping=0.3), capture=True, **run),
        _run(_build_spinning_model(device, register=True, damping=0.3), **run),
        "captured",
    )


_SOLVES = {
    "matrix_free": {},
    "phased": {"row_schedule": "phased"},
    "propagation": {"articulated_contact_response": "propagation"},
    "propagation_colored": {"articulated_contact_response": "propagation_colored"},
    "split": {"solve_mode": "split"},
}


def _solves(device):
    return _SOLVES.items() if device.is_cuda else [("split", _SOLVES["split"])]


def test_zero_friction_matches_normal_rows_only(test, device):
    """Move contacts with zero friction coefficients as with normal rows only, whose friction rows carry no impulse."""
    for label, options in _solves(device):
        with test.subTest(solve=label):
            run = {"init": _set_box_velocities, "steps": 8, **options}
            normal_only = _run(_build_contact_scene(device), contact_friction_gap_max=-float("inf"), **run)
            test.assertTrue(all(f.size == 0 for f in normal_only[3]), f"{label}: friction rows were created")
            frictionless = _run(_build_contact_scene(device), contact_friction_scale=0.0, **run)
            test.assertTrue(any(f.size for f in frictionless[3]), f"{label}: zero friction dropped the rows")
            test.assertTrue(all(np.all(f == 0.0) for f in frictionless[3]), f"{label}: zero friction carried impulse")
            np.testing.assert_allclose(frictionless[1], normal_only[1], rtol=0.0, atol=1.0e-5, err_msg=label)
            np.testing.assert_allclose(frictionless[2], normal_only[2], rtol=0.0, atol=1.0e-4, err_msg=label)
            default = _run(_build_contact_scene(device), **run)
            test.assertGreater(float(np.abs(default[2] - normal_only[2]).max()), 1.0e-3)


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
    solver = SolverSteel(model, contact_friction_scale=0.25, **_split_or_default(device))
    np.testing.assert_allclose(solver._shape_mu.numpy()[: model.shape_count], 0.25 * 0.6, rtol=1.0e-7)
    model.shape_material_mu = wp.full(model.shape_count, 0.8, dtype=float, device=device)
    solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    np.testing.assert_allclose(solver._shape_mu.numpy()[: model.shape_count], 0.2, rtol=1.0e-7)
    np.testing.assert_allclose(model.shape_material_mu.numpy(), 0.8)


def test_friction_switches_validate_their_values(test, device):
    """Reject invalid friction scales."""
    model = _build_contact_scene(device)
    mode = _split_or_default(device)
    for value in (-0.5, float("nan"), float("inf")):
        with test.subTest(scale=value), test.assertRaisesRegex(ValueError, "contact_friction_scale"):
            SolverSteel(model, contact_friction_scale=value, **mode)


def _friction_start_runs(device):
    """Yield (label, model factory, options, required path flag) covering every position-solve path."""
    for label, options in _solves(device):
        yield label, lambda: _build_contact_scene(device), {"init": _set_box_velocities, **options}, None
    if not device.is_cuda:
        return
    yield (
        "split-loop",
        lambda: _build_contact_scene(device),
        {"init": _set_box_velocities, "solve_mode": "split", "overrides": {"pgs_kernel": "loop"}},
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
        "articulation_constraint_max": 64,
        "free_body_constraint_max": 32,
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
        {**common, "articulation_constraint_max": 96},
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
            "articulation_constraint_max": 96,
            "use_parallel_streams": True,
        },
        "_sparse_diagonal_contact_solve",
    )


def test_friction_position_iterations_gate_every_solve_path(test, device):
    """Hold friction at zero before the last k position iterations on every solve path.

    ``k = iterations`` (and larger) is bitwise the default, ``k = 0`` leaves every friction row at zero,
    and an intermediate ``k`` solves friction, ending with a state different from both.
    """
    iterations = 8
    for label, build, options, path in _friction_start_runs(device):
        with test.subTest(solve=label):
            run = {"steps": 6, "iterations": iterations, **options}
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
    run = {"init": _set_box_velocities, "steps": 4, "velocity_iterations": 4}
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
        SolverSteel(model, contact_friction_position_iterations=-2, **mode)
    if device.is_cuda:
        with test.assertRaisesRegex(NotImplementedError, "contact_warmstart"):
            SolverSteel(model, contact_friction_position_iterations=2, contact_warmstart=True)
        with test.assertRaisesRegex(ValueError, "contact_friction_position_iterations"):
            run_compliance_fixture(
                device=device,
                articulated=True,
                enabled=True,
                steps=1,
                solver_options={"contact_friction_position_iterations": 2},
            )


def _bounce(device, scene, restitution=0.8, **options):
    """Return the post-impact velocity and pose of a sphere with ``restitution`` hitting a plane."""
    model, body = _build_plane_model(device, separation=0.0008, restitution=restitution, scene=scene)
    solver = SolverSteel(model, iterations=16, cfm=0.0, **options)
    solver._rigid_body_angular_damping.zero_()
    state_in, state_out = model.state(), model.state()
    _reset_state(model, state_in, -2.0)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, 1.0e-3)
    return state_out.body_qd.numpy()[body].copy(), state_out.body_q.numpy()[body].copy()


def test_disabled_restitution_matches_inelastic_materials(test, device):
    """Turn rebounds off exactly as zero restitution coefficients do, on free and articulated contacts."""
    for scene in ("free", "articulated"):
        with test.subTest(scene=scene):
            bouncing = _bounce(device, scene, restitution_velocity_threshold=0.0)
            disabled = _bounce(device, scene, enable_restitution=False, restitution_velocity_threshold=0.0)
            inelastic = _bounce(device, scene, restitution=0.0, restitution_velocity_threshold=0.0)
            test.assertGreater(float(bouncing[0][2]), 1.0)
            for got, expected in zip(disabled, inelastic, strict=True):
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
        SolverSteel(model, solve_mode="split")
    # A huge threshold is a physical parameter, not a second off switch.
    with test.assertRaisesRegex(NotImplementedError, "restitution"):
        SolverSteel(model, solve_mode="split", restitution_velocity_threshold=float(np.finfo(np.float32).max))
    disabled = _run(model, init=_set_box_velocities, solve_mode="split", enable_restitution=False)
    inelastic = _run(_build_contact_scene(device), init=_set_box_velocities, solve_mode="split")
    _assert_bitwise(test, disabled, inelastic, "split")
    model.shape_material_restitution.fill_(0.9)
    disabled[0].notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)


def test_disabled_restitution_lets_compliance_accept_restitution(test, device):
    """Run compliant contacts on shapes with positive restitution once restitution is disabled."""
    options = {"solver_options": {"enable_restitution": False}, "articulated": True, "enabled": True, "steps": 20}
    trace, _, _, _ = run_compliance_fixture(device=device, restitution=0.5, **options)
    reference, _, _, _ = run_compliance_fixture(device=device, restitution=0.0, **options)
    np.testing.assert_array_equal(trace, reference)


class TestSteelAngularDamping(unittest.TestCase):
    pass


class TestSteelFrictionOptions(unittest.TestCase):
    pass


class TestSteelRestitutionOptions(unittest.TestCase):
    pass


for _cls, _fn, _devices in (
    (TestSteelAngularDamping, test_angular_damping_defaults_to_the_featherstone_value, get_test_devices()),
    (TestSteelAngularDamping, test_angular_damping_decays_free_root_spin_analytically, get_test_devices()),
    (TestSteelAngularDamping, test_angular_damping_follows_in_place_edits, get_test_devices()),
    (TestSteelAngularDamping, test_physx_attributes_import_from_usd_and_are_read_every_step, get_test_devices()),
    (TestSteelAngularDamping, test_angular_damping_survives_graph_capture, get_cuda_test_devices()),
    (TestSteelFrictionOptions, test_zero_friction_matches_normal_rows_only, get_test_devices()),
    (TestSteelFrictionOptions, test_contact_friction_scale_matches_scaled_materials, get_test_devices()),
    (TestSteelFrictionOptions, test_contact_friction_scale_applies_to_notified_materials, get_test_devices()),
    (TestSteelFrictionOptions, test_friction_switches_validate_their_values, get_test_devices()),
    (TestSteelFrictionOptions, test_friction_position_iterations_gate_every_solve_path, get_test_devices()),
    (TestSteelFrictionOptions, test_friction_position_iterations_validate_their_value, get_test_devices()),
    (TestSteelFrictionOptions, test_velocity_iterations_always_solve_friction, get_cuda_test_devices()),
    (TestSteelFrictionOptions, test_friction_position_iterations_survive_graph_capture, get_cuda_test_devices()),
    (TestSteelRestitutionOptions, test_disabled_restitution_lets_split_accept_restitution, get_test_devices()),
    (TestSteelRestitutionOptions, test_disabled_restitution_matches_inelastic_materials, get_cuda_test_devices()),
    (
        TestSteelRestitutionOptions,
        test_disabled_restitution_lets_compliance_accept_restitution,
        get_cuda_test_devices(),
    ),
):
    add_function_test(_cls, _fn.__name__, _fn, devices=_devices)
# Unbind the loop variables so unittest does not load the last test case class a second time.
del _cls, _fn, _devices


_DT = 0.01


_UNSOLVABLE_GLOBAL_WARNING = r"global \(world -1\) articulations \[\d+\] .* cannot be solved"


def _floor_model(device, floor: str, box_x=(-2.0, 2.0), box_z=0.999, kinematic_world0=False):
    """Two worlds with one unit box each on a floor box whose top is at z = 0.5.

    ``floor`` is ``"static"`` (world geometry), ``"kinematic"`` (a global kinematic body)
    or ``"dynamic"`` (a global dynamic body). ``kinematic_world0`` makes world 0's box kinematic.
    """
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    for x in box_x:
        world = newton.ModelBuilder(gravity=wp.vec3(0.0))
        kinematic = kinematic_world0 and builder.world_count == 0
        body = world.add_body(
            xform=wp.transform(wp.vec3(x, 0.0, box_z), wp.quat_identity()), mass=1.0, is_kinematic=kinematic
        )
        world.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5, cfg=newton.ModelBuilder.ShapeConfig(mu=0.0))
        builder.add_world(world)
    if floor == "static":
        floor_body = -1
    else:
        floor_body = builder.add_body(is_kinematic=floor == "kinematic", mass=1.0)
    builder.add_shape_box(floor_body, hx=5.0, hy=5.0, hz=0.5, cfg=newton.ModelBuilder.ShapeConfig(mu=0.0))
    return builder.finalize(device=device)


def _step_once(model, solver, floor_velocity=0.0, box_velocity=-1.0):
    """Step once with the boxes moving down and the floor moving up; return every body's z velocity."""
    state, output = model.state(), model.state()
    joint_qd = state.joint_qd.numpy()
    for world in range(2):
        joint_qd[6 * world + 2] = box_velocity
    if joint_qd.size > 12:
        joint_qd[14] = floor_velocity
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    solver.step(state, output, model.control(), contacts, _DT)
    return output.body_qd.numpy()[:, 2], contacts


def test_global_kinematic_floor_supports_every_world(test, device, solve_mode="matrix_free", response="immediate"):
    """Stop the boxes of every world on a global kinematic floor exactly as on world geometry."""
    static_model = _floor_model(device, "static")
    static_v = _step_once(
        static_model, SolverSteel(static_model, solve_mode=solve_mode, articulated_contact_response=response)
    )[0][:2]
    test.assertLess(np.max(np.abs(static_v)), 0.05)

    model = _floor_model(device, "kinematic")
    solver = SolverSteel(
        model, solve_mode=solve_mode, warn_constraint_overflow=False, articulated_contact_response=response
    )
    v, contacts = _step_once(model, solver)
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    np.testing.assert_array_equal(solver._contact_path.numpy()[:count] >= 0, True)
    np.testing.assert_allclose(v[:2], static_v, atol=1.0e-5)
    np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [False, False, False])

    # The prescribed floor velocity enters every world's contact target.
    moving_v = _step_once(
        model, SolverSteel(model, solve_mode=solve_mode, articulated_contact_response=response), floor_velocity=0.5
    )[0][:2]
    np.testing.assert_allclose(moving_v, static_v + 0.5, atol=1.0e-4)

    # The floor is elided from the response even when world 0, where it is stored, has no dynamic body.
    model = _floor_model(device, "kinematic", kinematic_world0=True)
    solver = SolverSteel(
        model, solve_mode=solve_mode, warn_constraint_overflow=False, articulated_contact_response=response
    )
    state, output = model.state(), model.state()
    joint_qd = state.joint_qd.numpy()
    joint_qd[8] = -1.0
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    solver.step(state, output, model.control(), contacts, _DT)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
    np.testing.assert_allclose(output.body_qd.numpy()[1, 2], static_v[1], atol=1.0e-5)
    np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [False, False, False])


def test_global_kinematic_floor_capture_replay(test, device, solve_mode="matrix_free", response="immediate"):
    """Match eager stepping for both worlds on a global kinematic floor with captured collide and step."""
    model = _floor_model(device, "kinematic")
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    control = model.control()

    def run(solver, steps, capture):
        state_0, state_1 = model.state(), model.state()
        joint_qd = state_0.joint_qd.numpy()
        joint_qd[[2, 8]] = -1.0
        state_0.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

        def substep():
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, _DT)
            wp.copy(state_0.joint_q, state_1.joint_q)
            wp.copy(state_0.joint_qd, state_1.joint_qd)
            wp.copy(state_0.body_q, state_1.body_q)
            wp.copy(state_0.body_qd, state_1.body_qd)

        if capture:
            substep()
            with wp.ScopedCapture(device=device) as graph:
                substep()
            for _ in range(steps - 1):
                wp.capture_launch(graph.graph)
        else:
            for _ in range(steps):
                substep()
        return state_0.body_q.numpy()[:2], state_0.body_qd.numpy()[:2]

    eager_q, eager_qd = run(
        SolverSteel(model, solve_mode=solve_mode, articulated_contact_response=response), 20, capture=False
    )
    captured_q, captured_qd = run(
        SolverSteel(model, solve_mode=solve_mode, articulated_contact_response=response), 20, capture=True
    )
    np.testing.assert_allclose(captured_q, eager_q, atol=1.0e-6)
    np.testing.assert_allclose(captured_qd, eager_qd, atol=1.0e-5)
    # Both worlds rest on the floor instead of falling through it.
    test.assertGreater(float(np.min(eager_q[:, 2])), 0.9)
    np.testing.assert_allclose(eager_q[0, 2], eager_q[1, 2], atol=1.0e-5)


def _jointed_kinematic_floor_model(device):
    """Two worlds with one box each on a global kinematic floor driven by a prismatic joint."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    for x in (-2.0, 2.0):
        world = newton.ModelBuilder(gravity=wp.vec3(0.0))
        body = world.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.999), wp.quat_identity()), mass=1.0)
        world.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
        builder.add_world(world)
    floor = builder.add_link(mass=1.0, is_kinematic=True)
    builder.add_articulation([builder.add_joint_prismatic(-1, floor, axis=newton.Axis.Z)])
    builder.add_shape_box(floor, hx=5.0, hy=5.0, hz=0.5)
    return builder.finalize(device=device)


def _construct(test, model, expect_warning, **kwargs):
    """Construct the solver, requiring exactly the unsolvable-global-contact warning or no warning."""
    if expect_warning:
        return _expect_one_warning(
            test, newton.exceptions.NewtonWarning, _UNSOLVABLE_GLOBAL_WARNING, lambda: SolverSteel(model, **kwargs)
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = SolverSteel(model, **kwargs)
    test.assertEqual(len(caught), 0, [f"{w.category.__name__}: {w.message}" for w in caught])
    return solver


def _expect_one_warning(test, category, pattern, call):
    """Return ``call()``, requiring it to emit exactly one warning, of ``category`` and matching ``pattern``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    test.assertEqual(len(caught), 1, [f"{w.category.__name__}: {w.message}" for w in caught])
    test.assertIs(caught[0].category, category)
    test.assertRegex(str(caught[0].message), pattern)
    return result


def test_jointed_global_kinematic_flags_other_world_contacts(
    test, device, solve_mode="matrix_free", response="immediate"
):
    """Solve a jointed kinematic global articulation in world 0 and flag its other-world contacts."""
    model = _jointed_kinematic_floor_model(device)
    solver = _construct(
        test, model, True, solve_mode=solve_mode, warn_constraint_overflow=False, articulated_contact_response=response
    )
    state, output = model.state(), model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    solver.step(state, output, model.control(), contacts, _DT)

    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    shape_body = model.shape_body.numpy()
    body_world = model.body_world.numpy()
    pairs = contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count]
    box_world = np.maximum(body_world[shape_body[pairs[0]]], body_world[shape_body[pairs[1]]])
    path = solver._contact_path.numpy()[:count]
    test.assertTrue(np.any(box_world == 0) and np.any(box_world == 1))
    np.testing.assert_array_equal(path[box_world == 0] >= 0, True)
    np.testing.assert_array_equal(path[box_world == 1], -1)
    np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [False, True, True])
    with test.assertRaisesRegex(RuntimeError, "cannot be solved per world"):
        solver.check_constraint_capacity()


def test_construction_warns_about_unsolvable_global_contacts(
    test, device, solve_mode="matrix_free", response="immediate"
):
    """Warn once at construction when shapes allow contacts coupling a global articulation with another world."""
    kwargs = {"solve_mode": solve_mode, "articulated_contact_response": response}
    # Dynamic global bodies and jointed kinematic global articulations warn.
    _construct(test, _floor_model(device, "dynamic"), True, **kwargs)
    _construct(test, _jointed_kinematic_floor_model(device), True, **kwargs)

    # World geometry and global kinematic free bodies are solvable in every world.
    _construct(test, _floor_model(device, "static"), False, **kwargs)
    _construct(test, _floor_model(device, "kinematic"), False, **kwargs)
    _construct(test, _floor_model(device, "kinematic", kinematic_world0=True), False, **kwargs)

    # A dynamic global body with a single world, or whose shapes cannot collide with the
    # worlds' bodies, cannot produce such contacts.
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body = builder.add_body(mass=1.0)
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()
    _construct(test, builder.finalize(device=device), False, **kwargs)
    model = _floor_model(device, "dynamic")
    flags = model.shape_flags.numpy()
    flags[model.shape_body.numpy() == 2] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
    model.shape_flags.assign(flags)
    _construct(test, model, False, **kwargs)
    model = _floor_model(device, "dynamic")
    groups = model.shape_collision_group.numpy()
    groups[model.shape_body.numpy() == 2] = 7
    model.shape_collision_group.assign(groups)
    _construct(test, model, False, **kwargs)


def _lift_box(model, world):
    """Move one world's box 3 m up, out of contact."""
    joint_q = model.joint_q.numpy()
    joint_q[7 * world + 2] += 3.0
    model.joint_q.assign(joint_q)


def test_dynamic_global_body_flags_other_world_contacts(test, device, solve_mode="matrix_free", response="immediate"):
    """Let a dynamic global body interact with world 0, and drop and flag its contacts with another world."""
    # Only world 1's box touches the global dynamic floor.
    model = _floor_model(device, "dynamic")
    _lift_box(model, 0)
    solver = _construct(
        test, model, True, solve_mode=solve_mode, warn_constraint_overflow=False, articulated_contact_response=response
    )
    v, contacts = _step_once(model, solver)
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    np.testing.assert_array_equal(solver._contact_path.numpy()[:count], -1)
    np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [False, True, True])
    with test.assertRaisesRegex(RuntimeError, r"worlds \[1\].*global"):
        solver.check_constraint_capacity()

    # World 0's box against the same body is solved in world 0.
    model = _floor_model(device, "dynamic")
    _lift_box(model, 1)
    solver = _construct(
        test, model, True, solve_mode=solve_mode, warn_constraint_overflow=False, articulated_contact_response=response
    )
    v, contacts = _step_once(model, solver)
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    np.testing.assert_array_equal(solver._contact_path.numpy()[:count] >= 0, True)
    np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [False, False, False])
    # The world-0 box is stopped and pushes the (heavier) global body down.
    test.assertGreater(float(v[0]), -0.1)
    test.assertLess(float(v[2]), 0.0)


def _global_overflow_model(device):
    """World 0 and world 1 hold a free box each; a global free box rests on the ground plane.

    With ``global_on_ground`` the global box fills world 0's free-body rows; the world boxes
    float above the ground.
    """
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    for x in (-3.0, 3.0):
        world = newton.ModelBuilder(gravity=wp.vec3(0.0))
        body = world.add_body(xform=wp.transform(wp.vec3(x, 0.0, 2.0), wp.quat_identity()), mass=1.0)
        world.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        builder.add_world(world)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.099), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _place(model, state, z):
    joint_q = model.joint_q.numpy().copy()
    for body, height in enumerate(z):
        joint_q[7 * body + 2] = height
    state.joint_q.assign(joint_q)
    state.joint_qd.zero_()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def test_global_overflow_has_its_own_reset_slot(test, device, solve_mode="matrix_free", response="immediate"):
    """Latch global row loss in the global entry, and clear only its own status with each reset-mask entry."""
    model = _global_overflow_model(device)
    # One contact's rows fit; the propagation response holds every contact row in a family
    # of free_body_constraint_max + articulation_constraint_max rows.
    capacity = {"free_body_constraint_max": 3}
    if response == "propagation":
        capacity = {"free_body_constraint_max": 1, "articulation_constraint_max": 2}
    solver = _construct(
        test,
        model,
        True,
        solve_mode=solve_mode,
        warn_constraint_overflow=False,
        articulated_contact_response=response,
        **capacity,
    )
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state, output = model.state(), model.state()
    control = model.control()

    def step(z):
        _place(model, state, z)
        pipeline.collide(state, contacts)
        solver.step(state, output, control, contacts, _DT)
        return solver.constraint_overflow.numpy().tolist()

    def reset(mask):
        solver.reset(state, wp.array(mask, dtype=wp.bool, device=device))
        return solver.constraint_overflow.numpy().tolist()

    # The global box shares world 0's row storage, so its loss flags world 0 and the global entry.
    test.assertEqual(step([2.0, 2.0, 0.099]), [True, False, True])
    test.assertEqual(reset([False, False, True]), [True, False, False])
    test.assertEqual(step([2.0, 2.0, 2.0]), [True, False, False])
    test.assertEqual(reset([True, False, False]), [False, False, False])
    test.assertEqual(step([2.0, 2.0, 0.099]), [True, False, True])
    test.assertEqual(reset([True, False, False]), [False, False, True])
    test.assertEqual(reset([False, False, True]), [False, False, False])

    # A loss in world 1 alone never reaches the global entry or world 0.
    test.assertEqual(step([2.0, 0.099, 2.0]), [False, True, False])
    test.assertEqual(reset([False, False, True]), [False, True, False])
    test.assertEqual(reset([False, True, False]), [False, False, False])

    if not wp.get_device(device).is_cuda:
        return
    # Captured reset + step replays with the mask reassigned between launches.
    mask = wp.array([False, False, True], dtype=wp.bool, device=device)
    _place(model, state, [2.0, 2.0, 0.099])
    pipeline.collide(state, contacts)
    with wp.ScopedCapture(device=device) as capture:
        solver.reset(state, mask)
        solver.step(state, output, control, contacts, _DT)
    solver.reset(state)
    wp.capture_launch(capture.graph)
    test.assertEqual(solver.constraint_overflow.numpy().tolist(), [True, False, True])
    mask.assign(np.array([False, False, True]))
    solver.constraint_overflow.assign(np.array([True, True, True]))
    _place(model, state, [2.0, 2.0, 2.0])
    pipeline.collide(state, contacts)
    wp.capture_launch(capture.graph)
    test.assertEqual(solver.constraint_overflow.numpy().tolist(), [True, True, False])
    mask.assign(np.array([False, True, False]))
    wp.capture_launch(capture.graph)
    test.assertEqual(solver.constraint_overflow.numpy().tolist(), [True, False, False])


def _few_articulation_models(device):
    """Models whose global status entry lies beyond their articulation count.

    Yields ``(name, model, z_contact, z_free)``: one global free box on the ground
    (one world, one articulation), and two worlds holding one articulation between them
    (world 0 has only static geometry, world 1 a box) plus a global box. ``z_contact``
    places the boxes on the ground, where their contacts overflow ``free_body_constraint_max=3``;
    ``z_free`` lifts them out of contact.
    """
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.099), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()
    yield "single_global_body", builder.finalize(device=device), [0.099], [2.0]

    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    world = newton.ModelBuilder(gravity=wp.vec3(0.0))
    world.add_shape_sphere(-1, xform=wp.transform(wp.vec3(0.0, 5.0, 5.0), wp.quat_identity()), radius=0.1)
    builder.add_world(world)
    world = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body = world.add_body(xform=wp.transform(wp.vec3(3.0, 0.0, 0.099), wp.quat_identity()), mass=1.0)
    world.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_world(world)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.099), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()
    yield "two_worlds_two_articulations", builder.finalize(device=device), [0.099, 0.099], [2.0, 2.0]


def _check_global_slot_reset(test, device, model, z_contact, z_free, solve_mode, response):
    entries = model.world_count + 1
    # A dynamic global box can only meet another world's body when there are several worlds.
    # One contact's rows fit; the propagation response holds every contact row in a family
    # of free_body_constraint_max + articulation_constraint_max rows.
    capacity = {"free_body_constraint_max": 3}
    if response == "propagation":
        capacity = {"free_body_constraint_max": 1, "articulation_constraint_max": 2}
    solver = _construct(
        test,
        model,
        model.world_count > 1,
        solve_mode=solve_mode,
        warn_constraint_overflow=False,
        articulated_contact_response=response,
        **capacity,
    )
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state, output = model.state(), model.state()
    control = model.control()
    every = [True] * entries
    clear = [False] * entries

    def step(z):
        _place(model, state, z)
        pipeline.collide(state, contacts)
        solver.step(state, output, control, contacts, _DT)
        return solver.constraint_overflow.numpy().tolist()

    def reset(mask):
        solver.reset(state, None if mask is None else wp.array(mask, dtype=wp.bool, device=device))
        return solver.constraint_overflow.numpy().tolist()

    # Actual contact overflow sets every entry: the world rows, and the global entry
    # because the global box shares world 0's row storage.
    test.assertEqual(step(z_contact), every)
    with test.assertRaisesRegex(RuntimeError, "global"):
        solver.check_constraint_capacity()
    test.assertEqual(reset(None), clear)
    solver.check_constraint_capacity()

    global_only = [False] * (entries - 1) + [True]
    worlds_only = [True] * (entries - 1) + [False]
    mixed = [False] * (entries - 2) + [True, True]
    for mask in (global_only, worlds_only, mixed, every):
        test.assertEqual(step(z_contact), every)
        test.assertEqual(reset(mask), [not selected for selected in mask], msg=f"mask {mask}")

    # Stepping again recomputes the status: a lossless step keeps it clear.
    test.assertEqual(step(z_free), clear)
    if not wp.get_device(device).is_cuda:
        return

    # Captured resets, unmasked and with the mask reassigned between replays.
    mask = wp.array(global_only, dtype=wp.bool, device=device)
    with wp.ScopedCapture(device=device) as capture_all:
        solver.reset(state)
    with wp.ScopedCapture(device=device) as capture_masked:
        solver.reset(state, mask)
    test.assertEqual(step(z_contact), every)
    wp.capture_launch(capture_all.graph)
    test.assertEqual(solver.constraint_overflow.numpy().tolist(), clear)
    test.assertEqual(step(z_contact), every)
    wp.capture_launch(capture_masked.graph)
    test.assertEqual(solver.constraint_overflow.numpy().tolist(), worlds_only)
    mask.assign(np.array(worlds_only))
    wp.capture_launch(capture_masked.graph)
    test.assertEqual(solver.constraint_overflow.numpy().tolist(), clear)


def test_global_slot_reset_with_few_articulations(test, device, solve_mode="matrix_free", response="immediate"):
    """Clear the global entry on every reset mask, even when articulations do not outnumber the status entries."""
    for name, model, z_contact, z_free in _few_articulation_models(device):
        with test.subTest(model=name):
            test.assertLessEqual(model.articulation_count, model.world_count)
            _check_global_slot_reset(test, device, model, z_contact, z_free, solve_mode, response)


class TestSteelGlobalWorld(unittest.TestCase):
    pass


devices = get_cuda_test_devices()


for _name, _func in (
    ("test_global_kinematic_floor_supports_every_world", test_global_kinematic_floor_supports_every_world),
    ("test_global_kinematic_floor_capture_replay", test_global_kinematic_floor_capture_replay),
    ("test_dynamic_global_body_flags_other_world_contacts", test_dynamic_global_body_flags_other_world_contacts),
    (
        "test_jointed_global_kinematic_flags_other_world_contacts",
        test_jointed_global_kinematic_flags_other_world_contacts,
    ),
    (
        "test_construction_warns_about_unsolvable_global_contacts",
        test_construction_warns_about_unsolvable_global_contacts,
    ),
    ("test_global_overflow_has_its_own_reset_slot", test_global_overflow_has_its_own_reset_slot),
    ("test_global_slot_reset_with_few_articulations", test_global_slot_reset_with_few_articulations),
):
    add_function_test(TestSteelGlobalWorld, _name, _func, devices=devices)
    split_devices = devices if _name == "test_global_kinematic_floor_capture_replay" else get_test_devices()
    add_function_test(TestSteelGlobalWorld, f"{_name}_split", _func, devices=split_devices, solve_mode="split")
    # The propagation response moves these free-body contacts to its body-space rows.
    add_function_test(TestSteelGlobalWorld, f"{_name}_propagation", _func, devices=devices, response="propagation")


def _floating_base_arm():
    """A free-floating base with one position-driven revolute joint, no gravity."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    base = builder.add_link(mass=5.0, inertia=wp.mat33(np.eye(3) * 0.1))
    builder.add_shape_box(base, hx=0.1, hy=0.1, hz=0.1)
    free = builder.add_joint_free(child=base)
    arm = builder.add_link(mass=0.5, inertia=wp.mat33(np.eye(3) * 0.01))
    builder.add_shape_box(arm, hx=0.1, hy=0.02, hz=0.02)
    hinge = builder.add_joint_revolute(
        parent=base,
        child=arm,
        axis=newton.Axis.Z,
        target_ke=200.0,
        target_kd=20.0,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    builder.add_articulation([free, hinge])
    return builder.finalize(), hinge


def test_floating_base_drive_reads_its_own_target(test, device, drive_mode, solve_mode="matrix_free"):
    """Read the target of a revolute drive after a free joint at joint_target_q_start, not its DOF index."""
    with wp.ScopedDevice(device):
        model, hinge = _floating_base_arm()
        solver = newton.solvers.SolverSteel(model, solve_mode=solve_mode, iterations=16, drive_mode=drive_mode)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        targets = control.joint_target_q.numpy()
        targets[int(model.joint_target_q_start.numpy()[hinge])] = 0.4
        control.joint_target_q.assign(targets)
        for _ in range(240):
            solver.step(state_0, state_1, control, None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        q = state_0.joint_q.numpy()[int(model.joint_q_start.numpy()[hinge])]
        test.assertAlmostEqual(float(q), 0.4, delta=0.03)


class TestSteelTargetLayout(unittest.TestCase):
    pass


for _drive_mode in ("augmented", "rows"):
    add_function_test(
        TestSteelTargetLayout,
        f"test_floating_base_drive_reads_its_own_target_{_drive_mode}",
        test_floating_base_drive_reads_its_own_target,
        devices=get_cuda_test_devices(),
        drive_mode=_drive_mode,
    )


add_function_test(
    TestSteelTargetLayout,
    "test_floating_base_drive_reads_its_own_target_split",
    test_floating_base_drive_reads_its_own_target,
    devices=get_test_devices(),
    drive_mode="augmented",
    solve_mode="split",
)


_PACKAGE_DIR = Path(__file__).parents[2] / "_src" / "solvers" / "steel"


_KERNEL_FACTORIES = (
    "_get_pack_mf_meta_kernel",
    "_get_pgs_solve_mf_gs_kernel",
    "_get_pgs_solve_local_owned_kernel",
    "_get_cholesky_kernel",
    "_get_crba_cholesky_kernel",
    "_get_crba_cholesky_warp_kernel",
    "_get_cholesky_inverse_tiled_kernel",
    "_get_inverse_cholesky_register_kernel",
    "_get_paired_hinv_jt_kernel",
    "_get_pgs_solve_paired_factor_kernel",
    "_get_triangular_solve_kernel",
    "_get_hinv_jt_kernel",
    "_get_hinv_jt_plain_kernel",
    "_get_joint_limit_warp_kernel",
    "_get_hinv_jt_fused_kernel",
    "_get_delassus_kernel",
    "_get_pgs_solve_tiled_row_kernel",
    "_get_pgs_solve_tiled_contact_kernel",
    "_get_pgs_solve_streaming_kernel",
    "_get_pgs_solve_mf_kernel",
    "_get_pgs_solve_propagation_contact_kernel",
    "_get_factor_propagation_tree_revolute_kernel",
    "_get_propagation_tree_body_response_revolute_kernel",
    "_get_propagate_tree_impulses_revolute_kernel",
    "_get_refresh_propagation_tree_body_qd_warp_kernel",
    "_get_mark_independent_sparse_contact_candidates_kernel",
    "_get_build_independent_sparse_contact_groups_kernel",
    "_get_pgs_solve_sparse_diagonal_kernel",
    "_get_propagation_tree_cached_response_kernel",
    "_get_propagation_cached_gemv_kernel",
    "_get_color_propagation_prebuild_kernel",
    "_get_colored_friction_setup_kernel",
    "_get_colored_payload_pack_kernel",
    "_get_pgs_solve_propagation_colored_warp_kernel",
    "_get_pgs_solve_propagation_colored_block_kernel",
)


class TestSteelPrivateApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        solver_path = _PACKAGE_DIR / "solver_steel.py"
        cls.solver_module = ast.parse(solver_path.read_text(encoding="utf-8"))
        cls.top_level_functions = {
            node.name: node for node in cls.solver_module.body if isinstance(node, ast.FunctionDef)
        }
        cls.solver_class = next(
            node for node in cls.solver_module.body if isinstance(node, ast.ClassDef) and node.name == "SolverSteel"
        )
        cls.solver_methods = {node.name: node for node in cls.solver_class.body if isinstance(node, ast.FunctionDef)}

    def test_solver_is_exported_once_from_newton_solvers(self):
        """Expose the solver through newton.solvers only."""
        self.assertIn("SolverSteel", newton.solvers.__all__)
        self.assertIs(newton.solvers.SolverSteel, SolverSteel)
        self.assertNotIn("SolverSteel", newton.__all__)

    def test_solver_is_marked_experimental(self):
        """Mark the public solver class with the Sphinx experimental directive."""
        self.assertIn(".. experimental::", SolverSteel.__doc__)

    def test_constructor_options_are_keyword_only(self):
        """Keep every option after the model keyword-only."""
        parameters = list(inspect.signature(SolverSteel.__init__).parameters.values())
        self.assertEqual([p.name for p in parameters[:2]], ["self", "model"])
        for parameter in parameters[2:]:
            with self.subTest(parameter=parameter.name):
                self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)

    def test_global_physics_switches_default_to_the_model(self):
        """Keep physical behavior local to the model unless a solver-wide override is requested.

        ``enable_joint_limits``, the experimental ``enable_joint_friction`` and ``contact_compliance``
        and the experimental ``friction_mode`` are opt-in. The solver-wide damping, friction and restitution overrides
        default to the values that leave the model's per-body and per-shape properties in charge.
        """
        parameters = inspect.signature(SolverSteel.__init__).parameters
        for opt_in in ("enable_joint_limits", "enable_joint_friction", "contact_compliance"):
            with self.subTest(opt_in=opt_in):
                self.assertIs(parameters[opt_in].default, False)
        self.assertEqual(parameters["friction_mode"].default, "fixed_normal")
        self.assertIn(".. experimental::", SolverSteel.__doc__)
        for option, default in (
            ("contact_friction_scale", 1.0),
            ("contact_friction_position_iterations", -1),
            ("enable_restitution", True),
        ):
            with self.subTest(option=option):
                self.assertEqual(parameters[option].default, default)
        self.assertNotIn("enable_contact_friction", parameters)
        self.assertNotIn("angular_damping", parameters)

    def test_pgs_mode_selects_only_the_supported_solves(self):
        """Offer exactly the matrix-free and split solves, with the matrix-free solve as the default."""
        parameter = inspect.signature(SolverSteel.__init__).parameters["solve_mode"]
        self.assertEqual(parameter.default, "matrix_free")
        self.assertEqual(typing.get_args(parameter.annotation), ("matrix_free", "split"))
        parameters = inspect.signature(SolverSteel.__init__).parameters
        self.assertNotIn("delassus_kernel", parameters)
        self.assertNotIn("pgs_kernel", parameters)
        self.assertNotIn("pgs_chunk_size", parameters)
        self.assertNotIn("tile_threads", parameters)
        self.assertNotIn("serial_kernel_block_dim", parameters)
        self.assertNotIn("nvtx", parameters)
        self.assertNotIn("pgs_debug", parameters)

    def test_construction_options_are_read_only(self):
        """Expose constructor options read-only, except the ones read at every step."""
        parameters = inspect.signature(SolverSteel.__init__).parameters
        step_options = {name for name in parameters if isinstance(vars(SolverSteel).get(name), _StepOption)}
        self.assertEqual(
            step_options,
            {
                "baumgarte",
                "relaxation",
                "velocity_drive_mode",
                "contact_warmstart_decay",
                "warn_constraint_overflow",
                "restitution_velocity_threshold",
                "contact_speculative_scale",
                "contact_gap_gate",
                "contact_friction_gap_max",
            },
        )
        for name in parameters:
            if name in ("self", "model"):
                continue
            with self.subTest(option=name):
                self.assertIsInstance(getattr(SolverSteel, name), (property, _ConstructionOption))

    def test_deterministic_option_follows_the_warp_mode(self):
        """Inherit ``wp.config.deterministic``, fall back from inherited unsupported modes and reject explicit ones."""
        mode = wp.DeterministicMode
        model, _ = _build_plane_model("cpu", separation=0.0, restitution=0.0)
        cuda_model = (
            _build_plane_model("cuda:0", separation=0.0, restitution=0.0)[0] if wp.is_cuda_available() else None
        )
        saved = wp.config.deterministic
        try:
            self.assertEqual(SolverSteel(model, solve_mode="split").deterministic, mode.NOT_GUARANTEED)
            wp.config.deterministic = mode.RUN_TO_RUN
            self.assertEqual(SolverSteel(model, solve_mode="split").deterministic, mode.RUN_TO_RUN)
            self.assertEqual(
                SolverSteel(model, solve_mode="split", deterministic=mode.NOT_GUARANTEED).deterministic,
                mode.NOT_GUARANTEED,
            )
            wp.config.deterministic = mode.GPU_TO_GPU
            with self.assertWarnsRegex(
                newton.exceptions.NewtonWarning, "is GPU_TO_GPU; .* supports at most RUN_TO_RUN"
            ):
                self.assertEqual(SolverSteel(model, solve_mode="split").deterministic, mode.RUN_TO_RUN)
            if cuda_model is not None:
                wp.config.deterministic = mode.RUN_TO_RUN
                for response in ("propagation", "propagation_colored"):
                    solver = SolverSteel(cuda_model, articulated_contact_response=response)
                    self.assertEqual(solver.deterministic, mode.RUN_TO_RUN)
        finally:
            wp.config.deterministic = saved
        with self.assertRaisesRegex(ValueError, "GPU_TO_GPU"):
            SolverSteel(model, solve_mode="split", deterministic=mode.GPU_TO_GPU)
        with self.assertRaises(TypeError):
            SolverSteel(model, solve_mode="split", deterministic=True)

    def test_step_option_assignment_validates_like_the_constructor(self):
        """Reject invalid step-option assignments, keep the previous value, and normalize valid ones."""
        model, _ = _build_plane_model("cpu", separation=0.0, restitution=0.0)
        solver = SolverSteel(model, solve_mode="split")
        invalid = {
            "velocity_drive_mode": "frozen",
            "contact_warmstart_decay": -1.0,
            "restitution_velocity_threshold": float("nan"),
            "contact_speculative_scale": float("inf"),
            "contact_gap_gate": -0.01,
            "baumgarte": "stiff",
            "relaxation": None,
        }
        for name, value in invalid.items():
            with self.subTest(option=name, value=value):
                before = getattr(solver, name)
                with self.assertRaises((ValueError, TypeError)):
                    setattr(solver, name, value)
                self.assertEqual(getattr(solver, name), before)
        solver.velocity_drive_mode = "active"
        solver.contact_gap_gate = None
        solver.contact_gap_gate = np.float32(0.02)
        solver.baumgarte = np.float32(0.25)
        solver.warn_constraint_overflow = 0
        self.assertEqual(solver.velocity_drive_mode, "active")
        self.assertIs(type(solver.contact_gap_gate), float)
        self.assertIs(type(solver.baumgarte), float)
        self.assertIs(solver.warn_constraint_overflow, False)
        with self.assertRaisesRegex(ValueError, "velocity_drive_mode must be 'freeze' or 'active'"):
            SolverSteel(model, solve_mode="split", velocity_drive_mode="frozen")

    def test_solver_reads_no_environment_variables(self):
        """Configure the solver only through its constructor; the environment only enables profiling ranges."""
        pattern = re.compile(r"\bos\.(environ|getenv)\b|STEEL_|IL_NEWTON")
        profiling = 'os.environ.get("NEWTON_PROFILE", "0")'
        for path in sorted(_PACKAGE_DIR.glob("*.py")):
            with self.subTest(file=path.name):
                self.assertIsNone(pattern.search(path.read_text(encoding="utf-8").replace(profiling, "")))

    def test_prescribed_response_is_not_a_public_execution_knob(self):
        """Select the kinematic-body response internally, not through a constructor option."""
        init_method = self.solver_methods["__init__"]
        parameters = {
            argument.arg
            for argument in [*init_method.args.posonlyargs, *init_method.args.args, *init_method.args.kwonlyargs]
        }
        self.assertNotIn("exclude_fully_kinematic_free_articulations", parameters)

    def test_kernel_factories_are_private_cached_functions(self):
        """Keep the size-specialized kernel factories private, cached and keyed by device architecture."""
        for helper_name in _KERNEL_FACTORIES:
            with self.subTest(helper_name=helper_name):
                self.assertIn(helper_name, self.top_level_functions)
                helper = self.top_level_functions[helper_name]
                decorator_names = {
                    decorator.id for decorator in helper.decorator_list if isinstance(decorator, ast.Name)
                }
                parameters = {arg.arg for arg in [*helper.args.posonlyargs, *helper.args.args, *helper.args.kwonlyargs]}
                self.assertIn("cache", decorator_names)
                self.assertIn("device_arch", parameters)

    def test_plain_hinv_jt_kernel_omits_diagonal_output(self):
        """Emit the row diagonal only from the diagonal-producing H^-1 J^T variant."""
        plain_factory = self.top_level_functions["_get_hinv_jt_plain_kernel"]
        diagonal_factory = self.top_level_functions["_get_hinv_jt_kernel"]
        plain_template = next(node for node in plain_factory.body if isinstance(node, ast.FunctionDef))
        diagonal_template = next(node for node in diagonal_factory.body if isinstance(node, ast.FunctionDef))

        diagonal_factory_parameters = {argument.arg for argument in diagonal_factory.args.args}
        plain_parameters = {argument.arg for argument in plain_template.args.args}
        diagonal_parameters = {argument.arg for argument in diagonal_template.args.args}
        plain_names = {node.id for node in ast.walk(plain_template) if isinstance(node, ast.Name)}
        diagonal_names = {node.id for node in ast.walk(diagonal_factory) if isinstance(node, ast.Name)}

        self.assertNotIn("diag_group", plain_parameters)
        self.assertNotIn("diag_tile", plain_names)
        self.assertIn("compute_diag", diagonal_factory_parameters)
        self.assertIn("COMPUTE_DIAG", diagonal_names)
        self.assertIn("diag_group", diagonal_parameters)
        self.assertIn("diag_tile", diagonal_names)

    def test_cholesky_and_triangular_kernels_are_built_for_every_group(self):
        """Build the factorization kernels independently of the dense row capacity."""
        init_method = self.solver_methods["_init_tiled_kernels"]
        size_group_loop = next(
            node
            for node in ast.walk(init_method)
            if isinstance(node, ast.For) and isinstance(node.iter, ast.Attribute) and node.iter.attr == "_size_groups"
        )
        calls = {
            child.func.id: child.lineno
            for child in ast.walk(size_group_loop)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        }
        first_continue = min(
            (child.lineno for child in ast.walk(size_group_loop) if isinstance(child, ast.Continue)), default=None
        )
        for helper_name in ("_get_cholesky_kernel", "_get_triangular_solve_kernel"):
            with self.subTest(helper_name=helper_name):
                self.assertIn(helper_name, calls)
                if first_continue is not None:
                    self.assertLess(calls[helper_name], first_continue)


if __name__ == "__main__":
    unittest.main()
