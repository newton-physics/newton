# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for narrow MuJoCo force-DOF property updates."""

import unittest
import warnings
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton import ModelFlags
from newton._src.solvers.mujoco.constants import SOLREF_MODE_FORCE_SPACE, SOLREF_MODE_MJCF_DEFAULT, SOLREF_MODE_RAW
from newton.solvers import SolverMuJoCo


@wp.kernel
def _publish_budget(values: wp.array[float], friction: wp.array[float], damping: wp.array[float]):
    dof = wp.tid()
    if dof % 2 == 0:
        friction[dof] = values[dof]
        damping[dof] = 2.0 * values[dof]


def _make_model(
    *, worlds: int = 2, dofs: int = 2, device="cpu", custom_attributes: bool = True, driven: bool = False
) -> newton.Model:
    """Build a replicated hinge chain with passive parameters and joint limits."""
    template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    if custom_attributes:
        SolverMuJoCo.register_custom_attributes(template)
    joints = []
    parent = -1
    for _ in range(dofs):
        body = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), com=wp.vec3(0.0))
        joints.append(
            template.add_joint_revolute(
                parent,
                body,
                target_ke=2.0 if driven else 0.0,
                target_kd=0.3 if driven else 0.0,
                armature=0.1,
                friction=0.0,
                damping=0.2,
                limit_lower=-1.0,
                limit_upper=1.0,
            )
        )
        parent = body
    template.add_articulation(joints)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    if custom_attributes:
        SolverMuJoCo.register_custom_attributes(builder)
    builder.replicate(template, worlds)
    return builder.finalize(device=device)


class TestMuJoCoForceProperties(unittest.TestCase):
    def test_inertial_and_reference_updates_preserve_pending_limit_parameters(self):
        """Recompute limits from published parameters until a force notification arrives."""
        for cpu in (True, False):
            if not cpu and not wp.get_cuda_device_count():
                continue
            for mode in (SOLREF_MODE_FORCE_SPACE, SOLREF_MODE_RAW, SOLREF_MODE_MJCF_DEFAULT):
                for flag in (ModelFlags.JOINT_DOF_INERTIAL_PROPERTIES, ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES):
                    with self.subTest(cpu=cpu, mode=mode, flag=flag):
                        models = [_make_model(worlds=1, device="cpu" if cpu else "cuda:0") for _ in range(2)]
                        for model in models:
                            model.joint_limit_ke.fill_(100.0)
                            model.joint_limit_kd.fill_(10.0)
                            model.mujoco.solreflimit_mode.fill_(mode)
                            model.mujoco.solreflimit.fill_(wp.vec2(0.02, 1.0))
                        solver, reference = [SolverMuJoCo(m, use_mujoco_cpu=cpu, disable_contacts=True) for m in models]
                        for model in models:
                            model.joint_armature.fill_(1.5)
                            model.mujoco.dof_ref.fill_(0.3)
                        model = models[0]
                        model.joint_limit_ke.fill_(10000.0)
                        model.joint_limit_kd.fill_(20.0)
                        model.mujoco.solreflimit.fill_(wp.vec2(0.2, 0.7))
                        if cpu:
                            solver.notify_model_changed(flag)
                        else:
                            with wp.ScopedCapture(device=model.device) as capture:
                                solver.notify_model_changed(flag)
                            wp.capture_launch(capture.graph)
                        reference.notify_model_changed(flag)
                        np.testing.assert_allclose(
                            solver.mjw_model.jnt_solref.numpy(), reference.mjw_model.jnt_solref.numpy()
                        )
                        np.testing.assert_array_equal(model.mujoco.solreflimit_mode.numpy(), mode)
                        before_force = solver.mjw_model.jnt_solref.numpy().copy()
                        models[1].joint_limit_ke.fill_(10000.0)
                        models[1].joint_limit_kd.fill_(20.0)
                        models[1].mujoco.solreflimit.fill_(wp.vec2(0.2, 0.7))
                        solver.notify_model_changed(ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
                        reference.notify_model_changed(ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
                        self.assertFalse(np.array_equal(solver.mjw_model.jnt_solref.numpy(), before_force))
                        np.testing.assert_allclose(
                            solver.mjw_model.jnt_solref.numpy(), reference.mjw_model.jnt_solref.numpy()
                        )

    def test_force_update_preserves_direct_actuator_damping(self):
        """Update joint targets without overwriting resolved direct-actuator damping."""
        mjcf = """
        <mujoco>
          <worldbody><body>
            <joint name="hinge" type="hinge"/>
            <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1"/>
          </body></worldbody>
          <actuator><position joint="hinge" kp="100" dampratio="1"/></actuator>
        </mujoco>
        """
        for cpu in (True, False):
            with self.subTest(cpu=cpu):
                builder = newton.ModelBuilder()
                builder.add_mjcf(mjcf, ctrl_direct=True)
                body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
                joint = builder.add_joint_revolute(-1, body, target_ke=2.0, target_kd=0.3)
                builder.add_articulation([joint])
                model = builder.finalize(device="cpu")
                solver = SolverMuJoCo(model, use_mujoco_cpu=cpu, disable_contacts=True)
                direct = solver.mjc_actuator_ctrl_source.numpy() == 1
                target = ~direct
                self.assertTrue(np.any(direct) and np.any(target))
                before = (
                    solver.mj_model.actuator_biasprm.copy()
                    if cpu
                    else solver.mjw_model.actuator_biasprm.numpy()[0].copy()
                )
                self.assertTrue(np.all(before[direct, 2] < 0))
                model.joint_target_ke.fill_(5.0)
                model.joint_target_kd.fill_(0.7)
                solver.notify_model_changed(ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
                after = solver.mj_model.actuator_biasprm if cpu else solver.mjw_model.actuator_biasprm.numpy()[0]
                np.testing.assert_array_equal(after[direct], before[direct])
                self.assertFalse(np.array_equal(after[target], before[target]))

    def test_joint_transform_update_preserves_pending_dof_properties(self):
        """Keep transform notifications cheap and leave pending DOF edits unpublished."""
        for cpu in (True, False):
            with self.subTest(cpu=cpu):
                model = _make_model(worlds=1)
                solver = SolverMuJoCo(model, use_mujoco_cpu=cpu, disable_contacts=True)
                before = {name: getattr(solver.mjw_model, name).numpy().copy() for name in ("dof_armature", "qpos0")}
                body_pos = solver.mjw_model.body_pos.numpy().copy()
                model.joint_armature.fill_(7.0)
                model.mujoco.dof_ref.fill_(0.2)
                transforms = model.joint_X_p.numpy()
                transforms[0, 0] += 0.1
                model.joint_X_p.assign(transforms)
                with patch.object(
                    solver, "_set_const_0_with_physical_meaninertia", side_effect=AssertionError("recomputed constants")
                ):
                    solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES)
                for name, values in before.items():
                    np.testing.assert_array_equal(getattr(solver.mjw_model, name).numpy(), values)
                self.assertFalse(np.array_equal(solver.mjw_model.body_pos.numpy(), body_pos))

    def test_constant_refresh_preserves_pending_armature(self):
        """Refresh constants without publishing an unnotified armature edit."""
        for cpu in (True, False):
            for kinematic in (False, True):
                with self.subTest(cpu=cpu, kinematic=kinematic):
                    model = _make_model(worlds=1)
                    if kinematic:
                        model.body_flags.fill_(int(newton.BodyFlags.KINEMATIC))
                    solver = SolverMuJoCo(model, use_mujoco_cpu=cpu, disable_contacts=True)
                    armature = solver.mjw_model.dof_armature.numpy().copy()
                    meaninertia = (
                        float(solver.mj_model.stat.meaninertia)
                        if cpu
                        else solver.mjw_model.stat.meaninertia.numpy().copy()
                    )
                    model.joint_armature.fill_(7.0)
                    solver.notify_model_changed(ModelFlags.ACTUATOR_PROPERTIES)
                    np.testing.assert_array_equal(solver.mjw_model.dof_armature.numpy(), armature)
                    actual = solver.mj_model.stat.meaninertia if cpu else solver.mjw_model.stat.meaninertia.numpy()
                    np.testing.assert_allclose(actual, meaninertia)

    def _cuda_device(self):
        """Skip capture tests when no CUDA device with a mempool is available."""
        if wp.get_cuda_device_count() == 0:
            self.skipTest("CUDA graph capture requires a CUDA device")
        device = wp.get_cuda_device(0)
        if not wp.is_mempool_enabled(device):
            self.skipTest("CUDA graph capture requires the CUDA mempool allocator")
        return device

    def test_captured_reference_updates_preserve_pending_armature(self):
        """Replay reference updates without publishing pending inertial or force edits."""
        device = self._cuda_device()
        model = _make_model(device=device)
        solver = SolverMuJoCo(model, disable_contacts=True)
        armature = solver.mjw_model.dof_armature.numpy().copy()
        damping = solver.mjw_model.dof_damping.numpy().copy()
        model.joint_armature.fill_(7.0)
        model.joint_damping.fill_(9.0)
        with wp.ScopedCapture(device=device) as capture:
            solver.notify_model_changed(ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES)
        for ref in (0.2, 0.4):
            model.mujoco.dof_ref.fill_(ref)
            model.mujoco.dof_springref.fill_(ref + 0.1)
            wp.capture_launch(capture.graph)
            np.testing.assert_allclose(solver.mjw_model.qpos0.numpy(), ref)
            np.testing.assert_allclose(solver.mjw_model.qpos_spring.numpy(), ref + 0.1)
            np.testing.assert_allclose(solver.mjw_model.jnt_range.numpy()[..., 0], ref - 1.0)
            np.testing.assert_allclose(solver.mjw_model.jnt_range.numpy()[..., 1], ref + 1.0)
            np.testing.assert_array_equal(solver.mjw_model.dof_armature.numpy(), armature)
            np.testing.assert_array_equal(solver.mjw_model.dof_damping.numpy(), damping)

        with (
            patch.object(
                solver, "_set_const_0_with_physical_meaninertia", side_effect=AssertionError("recomputed constants")
            ),
            patch.object(
                solver._mujoco_warp, "set_length_range", side_effect=AssertionError("recomputed length range")
            ),
            wp.ScopedCapture(device=device) as transforms,
        ):
            solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES)
        wp.capture_launch(transforms.graph)
        np.testing.assert_array_equal(solver.mjw_model.dof_armature.numpy(), armature)

    def test_eager_updates_preserve_armature_and_optional_parameters(self):
        """Publish force fields on both backends without modifying kinematic armature."""
        backends = [(True, "cpu"), (False, "cpu")]
        if wp.get_cuda_device_count():
            backends.append((False, "cuda:0"))
        for cpu, device in backends:
            for custom_attributes in (True, False):
                with self.subTest(cpu=cpu, device=device, custom_attributes=custom_attributes):
                    model = _make_model(worlds=1, device=device, custom_attributes=custom_attributes)
                    body_flags = model.body_flags.numpy()
                    body_flags[0] = int(newton.BodyFlags.KINEMATIC)
                    model.body_flags.assign(body_flags)
                    solver = SolverMuJoCo(model, use_mujoco_cpu=cpu, separate_worlds=False, disable_contacts=True)
                    mapping = solver.mjc_dof_to_newton_dof.numpy()
                    armature = solver.mjw_model.dof_armature.numpy().copy()
                    invweight = solver.mjw_model.dof_invweight0.numpy().copy()
                    initial_solref = solver.mjw_model.dof_solref.numpy().copy()
                    initial_solimp = solver.mjw_model.dof_solimp.numpy().copy()
                    friction = np.arange(1, model.joint_dof_count + 1, dtype=np.float32)
                    model.joint_friction.assign(friction)
                    model.joint_damping.assign(2.0 * friction)
                    model.joint_armature.fill_(7.0)
                    if custom_attributes:
                        solref = model.mujoco.solreffriction.numpy()
                        solref[:, 0] = -100.0 * friction
                        solref[:, 1] = -10.0
                        model.mujoco.solreffriction.assign(solref)
                        solimp = model.mujoco.solimpfriction.numpy()
                        solimp[:, 0] = np.linspace(0.8, 0.9, model.joint_dof_count)
                        model.mujoco.solimpfriction.assign(solimp)
                    solver.notify_model_changed(ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
                    expected = {
                        "dof_frictionloss": friction[mapping],
                        "dof_damping": 2.0 * friction[mapping],
                        "dof_solref": solref[mapping] if custom_attributes else initial_solref,
                        "dof_solimp": solimp[mapping] if custom_attributes else initial_solimp,
                        "dof_armature": armature,
                        "dof_invweight0": invweight,
                    }
                    for name, values in expected.items():
                        np.testing.assert_allclose(getattr(solver.mjw_model, name).numpy(), values, err_msg=name)
                        if cpu:
                            np.testing.assert_allclose(
                                getattr(solver.mj_model, name), values[0], rtol=1.0e-5, err_msg=name
                            )

    def test_force_flags_rearm_raw_limit_validation(self):
        """Validate raw limits for force updates while leaving transform-only updates independent."""
        for cpu in (True, False):
            with self.subTest(cpu=cpu):
                model = _make_model(worlds=1)
                model.mujoco.solreflimit_mode.fill_(SOLREF_MODE_RAW)
                model.mujoco.solreflimit.fill_(wp.vec2(0.02, 1.0))
                solver = SolverMuJoCo(model, use_mujoco_cpu=cpu, disable_contacts=True)
                initial_solref = solver.mjw_model.jnt_solref.numpy().copy()
                model.mujoco.solreflimit.fill_(wp.vec2(-1.0, 1.0))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES)
                self.assertFalse(any("invalid components" in str(w.message) for w in caught))
                np.testing.assert_array_equal(solver.mjw_model.jnt_solref.numpy(), initial_solref)
                for flags in (
                    ModelFlags.JOINT_DOF_FORCE_PROPERTIES,
                    ModelFlags.JOINT_DOF_FORCE_PROPERTIES | ModelFlags.JOINT_PROPERTIES,
                ):
                    with self.subTest(flags=flags), self.assertWarnsRegex(UserWarning, "invalid components"):
                        solver.notify_model_changed(flags)

    def test_captured_force_validation_remains_pending_after_transform_update(self):
        """Defer force-limit validation through capture and transform-only updates until an eager force update."""
        device = self._cuda_device()
        model = _make_model(device=device)
        model.mujoco.solreflimit_mode.fill_(SOLREF_MODE_RAW)
        model.mujoco.solreflimit.fill_(wp.vec2(0.02, 1.0))
        solver = SolverMuJoCo(model, disable_contacts=True)
        model.mujoco.solreflimit.fill_(wp.vec2(-1.0, 1.0))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with wp.ScopedCapture(device=device) as capture:
                solver.notify_model_changed(ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
            wp.capture_launch(capture.graph)
            solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES)
        self.assertFalse(any("invalid components" in str(w.message) for w in caught))
        self.assertFalse(solver._raw_solreflimit_validated)
        with self.assertWarnsRegex(UserWarning, "invalid components"):
            solver.notify_model_changed(ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
        self.assertTrue(solver._raw_solreflimit_validated)

    def test_model_without_dofs(self):
        """Accept force updates for a fixed-joint model without DOFs."""
        for cpu in (True, False):
            with self.subTest(cpu=cpu):
                builder = newton.ModelBuilder()
                body = builder.add_link()
                builder.add_shape_sphere(body, radius=0.1)
                joint = builder.add_joint_fixed(parent=-1, child=body)
                builder.add_articulation([joint])
                model = builder.finalize(device="cpu")
                solver = SolverMuJoCo(model, use_mujoco_cpu=cpu)
                solver.notify_model_changed(ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
                self.assertEqual(solver.mj_model.nv, 0)

    def test_captured_publication_preserves_unrelated_properties(self):
        """Replay distinct world budgets while preserving unowned DOFs and cached inertia."""
        device = self._cuda_device()
        model = _make_model(device=device)
        solver = SolverMuJoCo(model, disable_contacts=True, iterations=2)
        untouched = (
            "dof_armature",
            "dof_invweight0",
            "qpos0",
            "qpos_spring",
            "jnt_range",
            "jnt_solref",
            "actuator_gainprm",
        )
        before = {name: getattr(solver.mjw_model, name).numpy().copy() for name in untouched}
        model.joint_armature.fill_(7.0)
        model.mujoco.dof_ref.fill_(0.2)
        values = wp.zeros(model.joint_dof_count, dtype=float, device=model.device)
        mapping = solver.mjc_dof_to_newton_dof.numpy()
        with wp.ScopedCapture(device=model.device) as capture:
            wp.launch(
                _publish_budget,
                dim=model.joint_dof_count,
                inputs=[values, model.joint_friction, model.joint_damping],
                device=model.device,
            )
            solver.notify_model_changed(ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
        for scale in (1.0, 0.0, 3.0):
            values.assign(np.arange(1, model.joint_dof_count + 1, dtype=np.float32) * scale)
            solref = model.mujoco.solreffriction.numpy()
            solref[:, 0] = np.arange(1, model.joint_dof_count + 1) * -100.0
            solref[:, 1] = -10.0
            model.mujoco.solreffriction.assign(solref)
            solimp = model.mujoco.solimpfriction.numpy()
            solimp[:, 0] = np.linspace(0.8, 0.9, model.joint_dof_count)
            model.mujoco.solimpfriction.assign(solimp)
            wp.capture_launch(capture.graph)
            for target, source in (
                ("dof_frictionloss", model.joint_friction),
                ("dof_damping", model.joint_damping),
                ("dof_solref", model.mujoco.solreffriction),
                ("dof_solimp", model.mujoco.solimpfriction),
            ):
                np.testing.assert_allclose(getattr(solver.mjw_model, target).numpy(), source.numpy()[mapping])
            np.testing.assert_array_equal(model.joint_friction.numpy()[1::2], 0.0)
            np.testing.assert_allclose(model.joint_damping.numpy()[1::2], 0.2)
            for name in untouched:
                np.testing.assert_array_equal(getattr(solver.mjw_model, name).numpy(), before[name])
        np.testing.assert_allclose(solver.mjw_model.jnt_range.numpy()[..., 1], 1.0)
        solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
        np.testing.assert_allclose(solver.mjw_model.dof_armature.numpy(), 7.0)
        np.testing.assert_allclose(solver.mjw_model.dof_frictionloss.numpy(), model.joint_friction.numpy()[mapping])
        self.assertFalse(np.allclose(solver.mjw_model.dof_invweight0.numpy(), before["dof_invweight0"]))

    def test_force_fields_match_full_update_without_recomputing_constants(self):
        """Match full force publication, including driven joints and captured limit edits."""
        fields = (
            "dof_damping",
            "dof_frictionloss",
            "dof_solimp",
            "dof_solref",
            "jnt_stiffness",
            "jnt_margin",
            "jnt_range",
            "jnt_actfrcrange",
            "jnt_solimp",
            "jnt_solref",
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_lengthrange",
        )
        for cpu in (True, False) if wp.get_cuda_device_count() else (True,):
            with self.subTest(cpu=cpu):
                model = _make_model(worlds=1 if cpu else 2, device="cpu" if cpu else "cuda:0", driven=True)
                model.mujoco.dof_ref.fill_(0.2)
                reference = SolverMuJoCo(model, use_mujoco_cpu=cpu, disable_contacts=True)
                solver = SolverMuJoCo(model, use_mujoco_cpu=cpu, disable_contacts=True)
                before_inertia = solver.mjw_model.dof_invweight0.numpy().copy()
                self.assertGreater(solver.mj_model.nu, 0)
                with ExitStack() as stack:
                    for name in (
                        "_set_const_0_with_physical_meaninertia",
                        "_invalidate_contact_fast_path",
                        "_update_joint_dof_configuration_properties",
                        "_notify_connect_constraints_changed",
                    ):
                        stack.enter_context(patch.object(solver, name, side_effect=AssertionError(name)))
                    if not cpu:
                        with wp.ScopedCapture(device=model.device) as capture:
                            solver.notify_model_changed(ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
                    for scale in (1.0, 2.0, 0.5):
                        model.joint_target_ke.fill_(4.0 * scale)
                        model.joint_target_kd.fill_(0.6 * scale)
                        model.joint_damping.fill_(0.4 * scale)
                        model.joint_friction.fill_(0.1 * scale)
                        model.joint_effort_limit.fill_(3.0 * scale)
                        model.joint_limit_lower.fill_(-0.5 * scale)
                        model.joint_limit_upper.fill_(0.7 * scale)
                        model.joint_limit_ke.fill_(100.0 * scale)
                        model.joint_limit_kd.fill_(10.0 * scale)
                        model.mujoco.dof_passive_stiffness.fill_(0.2 * scale)
                        model.mujoco.limit_margin.fill_(0.01 * scale)
                        solimp = model.mujoco.solimplimit.numpy()
                        solimp[:, 1] = 0.8 + 0.05 * scale
                        model.mujoco.solimplimit.assign(solimp)
                        reference.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
                        if cpu:
                            solver.notify_model_changed(ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
                        else:
                            wp.capture_launch(capture.graph)
                        for name in fields:
                            np.testing.assert_allclose(
                                getattr(solver.mjw_model, name).numpy(),
                                getattr(reference.mjw_model, name).numpy(),
                                rtol=1.0e-5,
                                atol=1.0e-6,
                                err_msg=name,
                            )
                            if cpu:
                                np.testing.assert_allclose(
                                    getattr(solver.mj_model, name),
                                    getattr(reference.mj_model, name),
                                    rtol=1.0e-5,
                                    atol=1.0e-6,
                                    err_msg=name,
                                )
                        np.testing.assert_array_equal(solver.mjw_model.dof_invweight0.numpy(), before_inertia)

    def test_inertial_configuration_and_legacy_flags(self):
        """Preserve legacy integer flags and publish armature and reference changes separately."""
        self.assertEqual(int(ModelFlags.JOINT_DOF_PROPERTIES), 2)
        for cpu in (True, False):
            with self.subTest(cpu=cpu):
                model = _make_model(worlds=1)
                solver = SolverMuJoCo(model, use_mujoco_cpu=cpu, disable_contacts=True)
                initial_weight = (
                    solver.mj_model.dof_invweight0.copy() if cpu else solver.mjw_model.dof_invweight0.numpy().copy()
                )
                model.joint_armature.fill_(1.5)
                model.joint_friction.fill_(0.7)
                model.mujoco.dof_ref.fill_(0.2)
                model.mujoco.dof_springref.fill_(0.3)
                solver.notify_model_changed(ModelFlags.JOINT_DOF_INERTIAL_PROPERTIES)
                np.testing.assert_allclose(solver.mjw_model.dof_armature.numpy(), 1.5)
                np.testing.assert_allclose(solver.mjw_model.dof_frictionloss.numpy(), 0.0)
                np.testing.assert_allclose(solver.mjw_model.qpos0.numpy(), 0.0)
                weight = solver.mj_model.dof_invweight0 if cpu else solver.mjw_model.dof_invweight0.numpy()
                self.assertFalse(np.allclose(weight, initial_weight))
                solver.notify_model_changed(ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES)
                np.testing.assert_allclose(solver.mjw_model.qpos0.numpy(), 0.2)
                np.testing.assert_allclose(solver.mjw_model.qpos_spring.numpy(), 0.3)
                np.testing.assert_allclose(solver.mjw_model.jnt_range.numpy()[..., 0], -0.8)
                np.testing.assert_allclose(solver.mjw_model.jnt_range.numpy()[..., 1], 1.2)
                np.testing.assert_allclose(solver.mjw_model.dof_frictionloss.numpy(), 0.0)
                for flags in (
                    2,
                    511,
                    ModelFlags.ALL,
                    ModelFlags.JOINT_DOF_FORCE_PROPERTIES
                    | ModelFlags.JOINT_DOF_INERTIAL_PROPERTIES
                    | ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES,
                ):
                    model.joint_armature.fill_(2.0)
                    model.mujoco.dof_ref.fill_(0.4)
                    solver.notify_model_changed(flags)
                    np.testing.assert_allclose(solver.mjw_model.dof_armature.numpy(), 2.0)
                    np.testing.assert_allclose(solver.mjw_model.dof_frictionloss.numpy(), 0.7)
                    np.testing.assert_allclose(solver.mjw_model.qpos0.numpy(), 0.4)
                    np.testing.assert_allclose(solver.mjw_model.jnt_range.numpy()[..., 1], 1.4)
                    if cpu:
                        np.testing.assert_allclose(solver.mj_model.dof_armature, 2.0)
                        np.testing.assert_allclose(solver.mj_model.qpos0, 0.4)

    def test_matches_full_update_dynamics(self):
        """Match full-update trajectories with changing friction and damping on both backends."""
        for cpu in (True, False) if wp.get_cuda_device_count() else (True,):
            with self.subTest(cpu=cpu):
                model = _make_model(worlds=1 if cpu else 2, device="cpu" if cpu else "cuda:0")
                flags = (
                    ModelFlags.JOINT_DOF_PROPERTIES,
                    ModelFlags.JOINT_DOF_FORCE_PROPERTIES,
                )
                solvers = [SolverMuJoCo(model, use_mujoco_cpu=cpu, disable_contacts=True, iterations=10) for _ in flags]
                states = [[model.state(), model.state()] for _ in solvers]
                control = model.control()
                control.joint_f.fill_(0.5)
                for pair in states:
                    pair[0].joint_qd.fill_(0.3)
                    newton.eval_fk(model, pair[0].joint_q, pair[0].joint_qd, pair[0])
                for step in range(30):
                    # Include zero-to-positive friction after initially compiling with zero.
                    budget = np.arange(1, model.joint_dof_count + 1, dtype=np.float32) * (step % 3) * 0.2
                    model.joint_friction.assign(budget)
                    model.joint_damping.assign(budget + 0.1)
                    for solver, flag in zip(solvers, flags, strict=True):
                        solver.notify_model_changed(flag)
                    if cpu:
                        np.testing.assert_allclose(solvers[1].mj_model.dof_frictionloss, budget)
                    for solver, pair in zip(solvers, states, strict=True):
                        solver.step(pair[0], pair[1], control, None, 0.005)
                        pair.reverse()
                    for pair in states[1:]:
                        for field in ("joint_q", "joint_qd"):
                            np.testing.assert_allclose(
                                getattr(states[0][0], field).numpy(),
                                getattr(pair[0], field).numpy(),
                                atol=1.0e-6,
                                rtol=1.0e-5,
                            )

    def test_captured_update_wakes_sleeping_worlds(self):
        """Wake sleeping worlds after a captured force parameter change."""
        device = self._cuda_device()
        model = _make_model(device=device)
        sleep_policy = model.mujoco.sleep_policy.numpy()
        sleep_policy[::2] = int(SolverMuJoCo.SleepPolicy.INIT)
        model.mujoco.sleep_policy.assign(sleep_policy)
        solver = SolverMuJoCo(model, use_mujoco_contacts=True, enable_sleeping=True, solver="newton", iterations=2)
        np.testing.assert_array_equal(solver.mjw_data.ntree_awake.numpy(), 0)
        with wp.ScopedCapture(device=model.device) as capture:
            solver.notify_model_changed(ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
        wp.capture_launch(capture.graph)
        self.assertTrue(np.all(solver.mjw_data.tree_asleep.numpy() < 0))
        np.testing.assert_array_equal(solver.mjw_data.ntree_awake.numpy(), 1)
        np.testing.assert_array_equal(solver.mjw_data.nv_awake.numpy(), 2)

    def test_captured_publication_and_step_match_full_notification(self):
        """Consume changing friction budgets in the same graph as the simulation step."""
        device = self._cuda_device()
        model = _make_model(worlds=2, dofs=1, device=device)
        reference = SolverMuJoCo(model, disable_contacts=True, iterations=10)
        captured = SolverMuJoCo(model, disable_contacts=True, iterations=10)
        state_in = model.state()
        reference_out = model.state()
        captured_out = model.state()
        control = model.control()
        control.joint_f.fill_(0.5)
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        budget = wp.zeros(model.joint_dof_count, dtype=float, device=device)
        captured.step(state_in, captured_out, control, None, 0.005)
        with wp.ScopedCapture(device=device) as capture:
            wp.copy(model.joint_friction, budget)
            captured.notify_model_changed(ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
            captured.step(state_in, captured_out, control, None, 0.005)

        velocities = []
        for values in ([0.0, 0.0], [0.25, 2.0], [2.0, 0.25], [0.0, 0.0]):
            budget.assign(np.array(values, dtype=np.float32))
            model.joint_friction.assign(budget)
            reference.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
            reference.step(state_in, reference_out, control, None, 0.005)
            wp.capture_launch(capture.graph)
            for field in ("joint_q", "joint_qd"):
                np.testing.assert_allclose(
                    getattr(captured_out, field).numpy(),
                    getattr(reference_out, field).numpy(),
                    atol=1.0e-6,
                    rtol=1.0e-5,
                )
            velocities.append(captured_out.joint_qd.numpy().copy())
        self.assertGreater(float(velocities[0][1]), float(velocities[1][1]))


if __name__ == "__main__":
    unittest.main()
