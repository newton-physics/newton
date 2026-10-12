# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver-wide physics overrides of SolverFeatherPGS: damping, friction and restitution switches."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs._fixtures import (
    DT,
    _build_contact_scene,
    _build_mixed_response_model,
    _build_plane_model,
    _build_sparse_contact_friction_model,
    _reset_state,
    _run,
    _set_box_velocities,
)
from newton.tests.feather_pgs._fixtures import run_fixture as run_compliance_fixture
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


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
    attributes = None if damping is None else {"physx:angular_damping": damping}
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
    solver = SolverFeatherPGS(model, **_split_or_default(device))
    state_in, state_out = model.state(), model.state()
    _spin(model, state_in)
    model.physx.angular_damping.fill_(0.0)
    solver.step(state_in, state_out, model.control(), None, DT)
    test.assertAlmostEqual(float(state_out.joint_qd.numpy()[5]), 4.0, delta=1.0e-5)


def test_physx_attributes_import_from_usd_and_are_read_every_step(test, device):
    """Import the PhysX body and collision USD attributes into the physx namespace, and step with replaced arrays."""
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
    SolverFeatherPGS.register_custom_attributes(builder)
    builder.add_usd(stage)
    model = builder.finalize(device=device)
    np.testing.assert_allclose(model.physx.angular_damping.numpy(), [0.3])
    np.testing.assert_allclose(model.physx.max_linear_velocity.numpy(), [2.0])
    np.testing.assert_allclose(model.physx.max_angular_velocity.numpy(), [0.5 * np.pi], rtol=1.0e-6)
    np.testing.assert_allclose(model.physx.max_depenetration_velocity.numpy(), [1.5])
    np.testing.assert_allclose(model.physx.min_torsional_patch_radius.numpy(), [0.004])
    # Contact torsion is CUDA-only; this check covers the body attributes.
    model.physx.min_torsional_patch_radius.zero_()

    solver = SolverFeatherPGS(model, **_split_or_default(device))
    model.physx.angular_damping = wp.zeros(1, dtype=wp.float32, device=device)
    model.physx.max_angular_velocity = wp.full(1, 100.0, dtype=wp.float32, device=device)
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


# ---------------------------------------------------------------------------------------------------------------
# Friction rows and contact_friction_scale

_SOLVES = {
    "matrix_free": {},
    "phased": {"row_schedule": "phased"},
    "propagation": {"articulated_contact_response": "propagation"},
    "propagation_fused": {"articulated_contact_response": "propagation_fused"},
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
    solver = SolverFeatherPGS(model, contact_friction_scale=0.25, **_split_or_default(device))
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
            SolverFeatherPGS(model, contact_friction_scale=value, **mode)


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
        SolverFeatherPGS(model, contact_friction_position_iterations=-2, **mode)
    if device.is_cuda:
        with test.assertRaisesRegex(NotImplementedError, "contact_warmstart"):
            SolverFeatherPGS(model, contact_friction_position_iterations=2, contact_warmstart=True)
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


def _bounce(device, scene, restitution=0.8, **options):
    """Return the post-impact velocity and pose of a sphere with ``restitution`` hitting a plane."""
    model, body = _build_plane_model(device, separation=0.0008, restitution=restitution, scene=scene)
    solver = SolverFeatherPGS(model, iterations=16, cfm=0.0, **options)
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
        SolverFeatherPGS(model, solve_mode="split")
    # A huge threshold is a physical parameter, not a second off switch.
    with test.assertRaisesRegex(NotImplementedError, "restitution"):
        SolverFeatherPGS(model, solve_mode="split", restitution_velocity_threshold=float(np.finfo(np.float32).max))
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


class TestFeatherPGSGlobalOptions(unittest.TestCase):
    pass


for _fn in (
    test_angular_damping_defaults_to_the_featherstone_value,
    test_angular_damping_decays_free_root_spin_analytically,
    test_angular_damping_follows_in_place_edits,
    test_physx_attributes_import_from_usd_and_are_read_every_step,
    test_zero_friction_matches_normal_rows_only,
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
    test_disabled_restitution_matches_inelastic_materials,
    test_disabled_restitution_lets_compliance_accept_restitution,
):
    add_function_test(TestFeatherPGSGlobalOptions, _fn.__name__, _fn, devices=get_cuda_test_devices())


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
