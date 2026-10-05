# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Passive joint springs and passive damping in SolverFeatherPGS.

FeatherPGS reads the MuJoCo spring attributes ``mujoco:dof_passive_stiffness``,
``mujoco:dof_springref`` and ``mujoco:dof_ref``. MuJoCo coordinates are offset from
Newton's by ``ref``, so the Newton rest coordinate is ``springref - ref``.
"""

import math
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS, SolverMuJoCo
from newton.tests.unittest_utils import USD_AVAILABLE, add_function_test, get_cuda_test_devices

DT = 1.0 / 240.0
SPRING_KEYS = ("mujoco:dof_passive_stiffness", "mujoco:dof_springref", "mujoco:dof_ref")


def _hinge_builder(spring_k: float, spring_ref: float, damping: float, limit_upper: float = 10.0):
    """A fixed-base link on a revolute Z joint, so gravity exerts no torque about the axis.

    The joint has no drive: motion toward the rest coordinate comes from the passive spring
    alone, and velocity decay from the passive damping alone.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverFeatherPGS.register_custom_attributes(builder)
    link = builder.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_box(link, hx=0.1, hy=0.02, hz=0.02)
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
        limit_lower=-10.0,
        limit_upper=limit_upper,
        damping=damping,
        custom_attributes={"mujoco:dof_passive_stiffness": spring_k, "mujoco:dof_springref": spring_ref},
    )
    builder.add_articulation([joint])
    return builder


def _run(model, steps: int = 1200, qd0: float = 0.0, **solver_kwargs):
    solver = SolverFeatherPGS(model, pgs_iterations=8, **solver_kwargs)
    state_0, state_1 = model.state(), model.state()
    if qd0 != 0.0:
        qd = state_0.joint_qd.numpy()
        qd[0] = qd0
        state_0.joint_qd.assign(qd)
    control = model.control()
    for _ in range(steps):
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
    return state_0


def _mjcf_hinge(
    *, angle: str, ref: float, springref: float, joint_type: str = "hinge", axis: str = "0 0 1", damping: float = 0.5
):
    return f"""<mujoco>
  <compiler angle="{angle}"/>
  <worldbody>
    <body name="link" pos="0 0 1">
      <joint name="j" type="{joint_type}" axis="{axis}" ref="{ref}" springref="{springref}" stiffness="2.0"
        damping="{damping}"/>
      <geom type="box" size="0.2 0.02 0.02" pos="0.2 0 0" mass="1"/>
    </body>
  </worldbody>
</mujoco>"""


def _import_mjcf(device, mjcf: str, *, register=(SolverFeatherPGS,), gravity=(0.0, 0.0, -9.81)):
    builder = newton.ModelBuilder(gravity=gravity)
    for solver_cls in register:
        solver_cls.register_custom_attributes(builder)
    builder.add_mjcf(mjcf)
    return builder.finalize(device=device)


def _revolute_usd(*, ref: float, springref: float, compiler_angle: str | None) -> str:
    angle = f'\n        uniform token mjc:compiler:angle = "{compiler_angle}"' if compiler_angle else ""
    return f"""#usda 1.0
(
    metersPerUnit = 1.0
    upAxis = "Z"
)

def PhysicsScene "physicsScene" (
    prepend apiSchemas = ["MjcSceneAPI"]
)
{{{angle}
}}

def Xform "Articulation" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI"]
)
{{
    def Cube "link" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsMassAPI"]
    )
    {{
        float physics:mass = 1.0
        double size = 0.1
        double3 xformOp:translate = (0, 0, 1)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}

    def PhysicsRevoluteJoint "hinge" (
        prepend apiSchemas = ["MjcJointAPI"]
    )
    {{
        token physics:axis = "Z"
        rel physics:body1 = </Articulation/link>
        point3f physics:localPos0 = (0, 0, 1)
        float mjc:ref = {ref}
        float mjc:springref = {springref}
        double mjc:stiffness = 2.0
    }}
}}
"""


def _import_usd(device, usd: str):
    from pxr import Usd

    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().ImportFromString(usd)
    builder = newton.ModelBuilder()
    SolverFeatherPGS.register_custom_attributes(builder)
    builder.add_usd(stage)
    return builder.finalize(device=device)


def _spring_arrays(model):
    solver = SolverFeatherPGS(model)
    return solver._passive_spring_stiffness.numpy(), solver._passive_spring_ref.numpy()


def test_spring_converges_to_reference(test, device):
    """Settle an undriven joint at its spring's rest coordinate."""
    model = _hinge_builder(spring_k=0.5, spring_ref=0.5, damping=0.05).finalize(device=device)
    test.assertAlmostEqual(float(_run(model).joint_q.numpy()[0]), 0.5, delta=0.02)


def test_springref_preloads_against_limit(test, device):
    """Press the joint against its limit when the rest coordinate lies beyond the joint range."""
    model = _hinge_builder(spring_k=0.5, spring_ref=2.62, damping=0.05, limit_upper=0.3).finalize(device=device)
    test.assertAlmostEqual(float(_run(model, enable_joint_limits=True).joint_q.numpy()[0]), 0.3, delta=0.03)


def test_passive_damping_decays_velocity(test, device):
    """Decay joint velocity through passive damping; without damping the joint keeps coasting."""
    damped = _run(_hinge_builder(0.0, 0.0, damping=0.02).finalize(device=device), steps=480, qd0=5.0)
    undamped = _run(_hinge_builder(0.0, 0.0, damping=0.0).finalize(device=device), steps=480, qd0=5.0)
    qd_damped = abs(float(damped.joint_qd.numpy()[0]))
    qd_undamped = abs(float(undamped.joint_qd.numpy()[0]))
    test.assertGreater(qd_undamped, 4.0, "undamped joint should keep coasting")
    test.assertLess(qd_damped, 0.2 * qd_undamped)


def test_imported_nonzero_ref_sets_rest_coordinate(test, device):
    """Use springref - ref as the Newton rest coordinate: ref 30 deg and springref 45 deg settle at 15 deg."""
    model = _import_mjcf(device, _mjcf_hinge(angle="degree", ref=30.0, springref=45.0))
    stiffness, rest = _spring_arrays(model)
    np.testing.assert_allclose(stiffness, [2.0])
    np.testing.assert_allclose(rest, [math.radians(15.0)], rtol=1.0e-6)
    q = float(_run(model, steps=2400).joint_q.numpy()[0])
    test.assertAlmostEqual(q, math.radians(15.0), delta=2.0e-3)


def test_degree_and_radian_imports_agree(test, device):
    """Give the same rest coordinate for MJCF degrees, MJCF radians and USD degree or radian stages."""
    expected = math.radians(15.0)
    variants = {
        "mjcf degree": _import_mjcf(device, _mjcf_hinge(angle="degree", ref=30.0, springref=45.0)),
        "mjcf radian": _import_mjcf(
            device, _mjcf_hinge(angle="radian", ref=math.radians(30.0), springref=math.radians(45.0))
        ),
    }
    if USD_AVAILABLE:
        variants["usd default (degree)"] = _import_usd(
            device, _revolute_usd(ref=30.0, springref=45.0, compiler_angle=None)
        )
        variants["usd radian"] = _import_usd(
            device, _revolute_usd(ref=math.radians(30.0), springref=math.radians(45.0), compiler_angle="radian")
        )
    for name, model in variants.items():
        with test.subTest(variant=name):
            stiffness, rest = _spring_arrays(model)
            np.testing.assert_allclose(stiffness, [2.0])
            np.testing.assert_allclose(rest, [expected], rtol=1.0e-5)


def test_prismatic_spring_with_nonzero_ref(test, device):
    """Settle a slide joint at springref - ref [m], without any angular conversion."""
    mjcf = _mjcf_hinge(angle="degree", ref=0.1, springref=0.25, joint_type="slide", axis="1 0 0", damping=3.0)
    model = _import_mjcf(device, mjcf)
    stiffness, rest = _spring_arrays(model)
    np.testing.assert_allclose(stiffness, [2.0])
    np.testing.assert_allclose(rest, [0.15], rtol=1.0e-6)
    q = float(_run(model, steps=2400).joint_q.numpy()[0])
    test.assertAlmostEqual(q, 0.15, delta=1.0e-3)


def test_springs_follow_newton_dof_order(test, device):
    """Apply each DOF's own stiffness and rest coordinate across hinge, slide and ball joints of one tree."""
    mjcf = """<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="a" pos="0 0 1">
      <joint name="ja" type="hinge" axis="0 0 1" ref="0.1" springref="0.4" stiffness="3"/>
      <geom type="sphere" size="0.05" mass="1"/>
      <body name="b" pos="0.3 0 0">
        <joint name="jb" type="slide" axis="1 0 0" ref="-0.05" springref="0.15" stiffness="7"/>
        <geom type="sphere" size="0.05" mass="1"/>
        <body name="c" pos="0.3 0 0">
          <joint name="jc" type="ball"/>
          <geom type="sphere" size="0.05" mass="1"/>
          <body name="d" pos="0.3 0 0">
            <joint name="jd" type="hinge" axis="0 1 0" ref="0.2" springref="-0.3" stiffness="11"/>
            <geom type="sphere" size="0.05" mass="1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""
    model = _import_mjcf(device, mjcf, gravity=(0.0, 0.0, 0.0))
    test.assertEqual(
        [newton.JointType(t).name for t in model.joint_type.numpy()],
        ["REVOLUTE", "PRISMATIC", "BALL", "REVOLUTE"],
    )
    expected_k = np.array([3.0, 7.0, 0.0, 0.0, 0.0, 11.0])
    expected_rest = np.array([0.3, 0.2, 0.0, 0.0, 0.0, -0.5])
    solver = SolverFeatherPGS(model)
    np.testing.assert_allclose(solver._passive_spring_stiffness.numpy(), expected_k)
    np.testing.assert_allclose(solver._passive_spring_ref.numpy(), expected_rest, atol=1.0e-6)
    # At q = 0 and rest without gravity the joint torque is exactly the spring torque k * rest.
    solver.step(model.state(), model.state(), model.control(), None, DT)
    np.testing.assert_allclose(solver.joint_tau.numpy(), expected_k * expected_rest, rtol=1.0e-5, atol=1.0e-6)


def test_spring_torque_is_applied_once(test, device):
    """Apply the spring torque exactly once whichever solver registers the attributes first."""
    mjcf = _mjcf_hinge(angle="radian", ref=0.0, springref=0.5).replace('damping="0.5"', 'damping="0"')
    expected_tau = 2.0 * 0.5
    for register in ((SolverFeatherPGS,), (SolverFeatherPGS, SolverMuJoCo), (SolverMuJoCo, SolverFeatherPGS)):
        with test.subTest(register=[cls.__name__ for cls in register]):
            model = _import_mjcf(device, mjcf, register=register)
            np.testing.assert_allclose(model.mujoco.dof_passive_stiffness.numpy(), [2.0])
            np.testing.assert_allclose(model.mujoco.dof_springref.numpy(), [0.5])
            solver = SolverFeatherPGS(model)
            state_0, state_1 = model.state(), model.state()
            solver.step(state_0, state_1, model.control(), None, DT)
            test.assertAlmostEqual(float(solver.joint_tau.numpy()[0]), expected_tau, places=5)
            # One explicit step: qd = dt * tau / I_axis, with I_axis the link's inertia about the hinge.
            inertia = float(solver.H_by_size[1].numpy()[0, 0, 0])
            test.assertAlmostEqual(float(state_1.joint_qd.numpy()[0]), DT * expected_tau / inertia, places=6)


def test_d6_springs_warn_and_are_ignored(test, device):
    """Warn at construction and apply no spring to D6 DOFs, here two hinges of one MJCF body."""
    mjcf = """<mujoco>
  <worldbody>
    <body name="link" pos="0 0 1">
      <joint name="j0" type="hinge" axis="0 0 1" springref="0.5" stiffness="2"/>
      <joint name="j1" type="hinge" axis="0 1 0" springref="0.5" stiffness="2"/>
      <geom type="box" size="0.2 0.02 0.02" pos="0.2 0 0" mass="1"/>
    </body>
  </worldbody>
</mujoco>"""
    model = _import_mjcf(device, mjcf, gravity=(0.0, 0.0, 0.0))
    test.assertEqual(model.joint_type.numpy().tolist(), [int(newton.JointType.D6)])
    with test.assertWarnsRegex(UserWarning, "2 DOF\\(s\\) of D6 joints"):
        solver = SolverFeatherPGS(model)
    np.testing.assert_array_equal(solver._passive_spring_stiffness.numpy(), [0.0, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        state = _run(model, steps=240)
    np.testing.assert_allclose(state.joint_q.numpy(), [0.0, 0.0], atol=1.0e-6)


def test_models_without_springs_do_not_warn(test, device):
    """Build models without the spring attributes, or with zero stiffness on D6 DOFs, silently."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plain = newton.ModelBuilder()
        link = plain.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        plain.add_articulation([plain.add_joint_revolute(-1, link, axis=newton.Axis.Z)])
        plain_model = plain.finalize(device=device)
        test.assertIsNone(getattr(getattr(plain_model, "mujoco", None), "dof_passive_stiffness", None))
        SolverFeatherPGS(plain_model)
        d6 = newton.ModelBuilder()
        SolverFeatherPGS.register_custom_attributes(d6)
        link = d6.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        d6.add_articulation(
            [d6.add_joint_d6(-1, link, angular_axes=[newton.ModelBuilder.JointDofConfig(axis=(0, 0, 1))])]
        )
        solver = SolverFeatherPGS(d6.finalize(device=device))
    np.testing.assert_array_equal(solver._passive_spring_stiffness.numpy(), [0.0])


def test_spring_attribute_changes_reach_captured_graphs(test, device):
    """Refresh springs on JOINT_DOF_PROPERTIES in place, so a captured step applies the new values."""
    model = _hinge_builder(spring_k=0.0, spring_ref=0.5, damping=0.0).finalize(device=device)
    solver = SolverFeatherPGS(model)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    solver.step(state_0, state_1, control, None, DT)
    with wp.ScopedCapture(device=device) as capture:
        solver.step(state_0, state_1, control, None, DT)
    wp.capture_launch(capture.graph)
    test.assertEqual(float(solver.joint_tau.numpy()[0]), 0.0)
    model.mujoco.dof_passive_stiffness.assign([4.0])
    model.mujoco.dof_ref.assign([0.25])
    solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
    wp.capture_launch(capture.graph)
    test.assertAlmostEqual(float(solver.joint_tau.numpy()[0]), 4.0 * (0.5 - 0.25), places=5)


def _slider_model(device):
    """A unit-mass slider without gravity or damping, whose joint torque is the spring torque alone."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverFeatherPGS.register_custom_attributes(builder)
    link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joint = builder.add_joint_prismatic(
        -1,
        link,
        axis=newton.Axis.X,
        damping=0.0,
        armature=0.0,
        custom_attributes=dict(zip(SPRING_KEYS, (2.0, 0.5, 0.1), strict=True)),
    )
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _check_spring_edits_match_fresh_solver(test, device, replace: bool):
    """Edit spring arrays after a step, notify, and compare the next eager or captured step to a fresh solver."""
    edits = {"dof_passive_stiffness": 4.0, "dof_springref": 0.9, "dof_ref": 0.15}
    for fields in [(name,) for name in edits] + [tuple(edits)]:
        for capture in (False, True):
            with test.subTest(fields=fields, capture=capture):
                model = _slider_model(device)
                solver = SolverFeatherPGS(model)
                state_0, state_1 = model.state(), model.state()
                state_0.joint_q.fill_(0.2)
                control = model.control()
                solver.step(state_0, state_1, control, None, DT)
                if capture:
                    with wp.ScopedCapture(device=device) as graph:
                        solver.step(state_0, state_1, control, None, DT)
                for name in fields:
                    if replace:
                        setattr(model.mujoco, name, wp.array([edits[name]], dtype=wp.float32, device=device))
                    else:
                        getattr(model.mujoco, name).assign([edits[name]])
                solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
                if capture:
                    wp.capture_launch(graph.graph)
                else:
                    solver.step(state_0, state_1, control, None, DT)

                fresh = SolverFeatherPGS(model)
                fresh_out = model.state()
                fresh.step(state_0, fresh_out, control, None, DT)
                np.testing.assert_allclose(solver.joint_tau.numpy(), fresh.joint_tau.numpy(), atol=1.0e-6)
                np.testing.assert_allclose(state_1.joint_qd.numpy(), fresh_out.joint_qd.numpy(), atol=1.0e-6)


def test_replaced_spring_arrays_match_fresh_solver(test, device):
    """Read spring arrays replaced on the model at the next JOINT_DOF_PROPERTIES notify."""
    _check_spring_edits_match_fresh_solver(test, device, replace=True)


def test_assigned_spring_arrays_match_fresh_solver(test, device):
    """Read spring arrays modified in place at the next JOINT_DOF_PROPERTIES notify."""
    _check_spring_edits_match_fresh_solver(test, device, replace=False)


def test_registration_matches_mujoco_definitions(test, device):
    """Register the spring attributes standalone with SolverMuJoCo's names, units and importer conversions."""
    fpgs = newton.ModelBuilder()
    SolverFeatherPGS.register_custom_attributes(fpgs)
    mujoco = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(mujoco)
    for key in SPRING_KEYS:
        with test.subTest(attribute=key):
            ours, theirs = fpgs.custom_attributes[key], mujoco.custom_attributes[key]
            for field in ("frequency", "dtype", "assignment", "namespace", "default", "usd_attribute_name"):
                test.assertEqual(getattr(ours, field), getattr(theirs, field))
            test.assertEqual(ours.mjcf_attribute_name, theirs.mjcf_attribute_name)
            test.assertIs(ours.mjcf_value_transformer, theirs.mjcf_value_transformer)
    # Registering both solvers on one builder, in either order, keeps one definition per key.
    for first, second in ((SolverFeatherPGS, SolverMuJoCo), (SolverMuJoCo, SolverFeatherPGS)):
        builder = newton.ModelBuilder()
        first.register_custom_attributes(builder)
        second.register_custom_attributes(builder)
        test.assertEqual(sum(key in builder.custom_attributes for key in SPRING_KEYS), 3)


class TestFeatherPGSSprings(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _name, _func in (
    ("test_spring_converges_to_reference", test_spring_converges_to_reference),
    ("test_springref_preloads_against_limit", test_springref_preloads_against_limit),
    ("test_passive_damping_decays_velocity", test_passive_damping_decays_velocity),
    ("test_imported_nonzero_ref_sets_rest_coordinate", test_imported_nonzero_ref_sets_rest_coordinate),
    ("test_degree_and_radian_imports_agree", test_degree_and_radian_imports_agree),
    ("test_prismatic_spring_with_nonzero_ref", test_prismatic_spring_with_nonzero_ref),
    ("test_springs_follow_newton_dof_order", test_springs_follow_newton_dof_order),
    ("test_spring_torque_is_applied_once", test_spring_torque_is_applied_once),
    ("test_d6_springs_warn_and_are_ignored", test_d6_springs_warn_and_are_ignored),
    ("test_models_without_springs_do_not_warn", test_models_without_springs_do_not_warn),
    ("test_spring_attribute_changes_reach_captured_graphs", test_spring_attribute_changes_reach_captured_graphs),
    ("test_replaced_spring_arrays_match_fresh_solver", test_replaced_spring_arrays_match_fresh_solver),
    ("test_assigned_spring_arrays_match_fresh_solver", test_assigned_spring_arrays_match_fresh_solver),
):
    add_function_test(TestFeatherPGSSprings, _name, _func, devices=devices)
add_function_test(
    TestFeatherPGSSprings,
    "test_registration_matches_mujoco_definitions",
    test_registration_matches_mujoco_definitions,
    devices=None,
)


if __name__ == "__main__":
    unittest.main()
