# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import itertools
import os
import tempfile
import unittest
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import warp as wp

import newton
from newton import ModelFlags
from newton._src.core.types import vec5
from newton._src.solvers.mujoco.constants import (
    DEFAULT_LIMIT_SOLREF,
    MJ_MINVAL,
    SOLREF_MODE_FORCE_SPACE,
    SOLREF_MODE_MJCF_DEFAULT,
    SOLREF_MODE_RAW,
)
from newton._src.solvers.mujoco.kernels import convert_solref
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import USD_AVAILABLE
from newton.tests.utils.mujoco import _create_actuator_test_stage, _expected_positive_limit_solref


class TestMuJoCoSolverInvweightScaledSolref(unittest.TestCase):
    """Regression tests for joint-limit solref calibration.

    Newton users set ``ModelBuilder.add_joint_revolute`` ``limit_ke`` /
    ``limit_kd`` as *force-space* stiffness/damping (N·m/rad). MuJoCo's
    limit-constraint solver, however, applies effective stiffness
    ``k_eff = k / (invweight * (1 - dmax))``. Unless the Newton→MuJoCo
    ``jnt_solref`` conversion pre-multiplies the direct stiffness/damping
    pair by ``dof_invweight0 * (1 - dmax)`` before converting to MuJoCo's
    positive ``(timeconst, dampratio)`` convention, the simulated stiffness
    ends up scaled by the DOF inertia instead of matching the configured
    force-space limit response.

    The analogous scaling for ``shape_material_ke``/``kd`` on
    ``geom_solref`` is **not** applied, because MuJoCo mixes contact
    ``solref`` linearly in ``(timeconst, dampratio)`` space, which is
    non-linear in ``(ke, kd)`` and collapses dynamic-vs-static contact
    stiffness. The contact case therefore needs a separate contact-time
    mixing design.
    """

    @staticmethod
    def _build_tendon_limit_model(mass=1.0, worlds=1):
        """Build independent sliders with an authored native tendon limit."""
        template = newton.ModelBuilder()
        template.add_mjcf(f"""<mujoco><option gravity="0 0 0"/>
          <worldbody><body name="slider">
            <joint name="slide" type="slide" axis="1 0 0" limited="false"/>
            <geom type="sphere" size="0.05" mass="{mass}" contype="0" conaffinity="0"/>
          </body></worldbody>
          <tendon><fixed name="limit" limited="true" range="-0.1 0.1"
            solreflimit="-100 -20" solimplimit="0.95 0.95 0.001 0.5 2">
            <joint joint="slide" coef="1"/>
          </fixed></tendon>
        </mujoco>""")
        builder = newton.ModelBuilder()
        builder.replicate(template, worlds)
        return builder.finalize()

    def test_tendon_limit_force_gains_preserve_compliance(self):
        """Preserve static stiffness after mass edits and support zero gains and native restoration."""
        for use_cpu, mass in itertools.product((False, True), (1.0, 10.0)):
            with self.subTest(use_cpu=use_cpu, mass=mass):
                model = self._build_tendon_limit_model(mass)
                solver = SolverMuJoCo(model, use_mujoco_cpu=use_cpu, disable_contacts=True)
                attrs = model.mujoco

                def tendon_parameter(name, solver=solver):
                    """Read runtime parameters from the active backend."""
                    if solver.use_mujoco_cpu:
                        return getattr(solver.mj_model, name)
                    return getattr(solver.mjw_model, name).numpy()[0]

                np.testing.assert_allclose(solver.mj_model.tendon_solref_lim[0], [-100.0, -20.0])
                attrs.tendon_limit_ke.fill_(100.0)
                attrs.tendon_limit_kd.fill_(20.0)
                attrs.tendon_solref_limit_mode.fill_(SOLREF_MODE_FORCE_SPACE)
                solver.notify_model_changed(ModelFlags.TENDON_PROPERTIES)

                def step_at_equilibrium(model=model, solver=solver):
                    """Apply 3 N at 3 cm penetration, the equilibrium for 100 N/m."""
                    state_in, state_out = model.state(), model.state()
                    state_in.joint_q.fill_(0.13)
                    state_in.joint_qd.zero_()
                    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
                    control = model.control()
                    control.joint_f.fill_(3.0)
                    solver.step(state_in, state_out, control, None, 0.001)
                    return state_out.joint_qd.numpy()[0]

                self.assertAlmostEqual(float(step_at_equilibrium()), 0.0, delta=1.0e-7)
                model.body_mass.assign(model.body_mass.numpy() * 2.0)
                model.body_inertia.assign(model.body_inertia.numpy() * 2.0)
                solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)
                self.assertAlmostEqual(float(step_at_equilibrium()), 0.0, delta=1.0e-7)

                attrs.tendon_limit_kd.zero_()
                solver.notify_model_changed(ModelFlags.TENDON_PROPERTIES)
                self.assertAlmostEqual(float(step_at_equilibrium()), 0.0, delta=1.0e-7)
                self.assertEqual(float(tendon_parameter("tendon_solref_lim")[0, 1]), 0.0)

                attrs.tendon_limit_ke.zero_()
                solver.notify_model_changed(ModelFlags.TENDON_PROPERTIES)
                self.assertAlmostEqual(float(step_at_equilibrium()), 3.0 / (2.0 * mass) * 0.001, delta=1.0e-7)
                np.testing.assert_allclose(attrs.tendon_range.numpy()[0], [-0.1, 0.1])

                attrs.tendon_solref_limit_mode.fill_(SOLREF_MODE_RAW)
                solver.notify_model_changed(ModelFlags.TENDON_PROPERTIES)
                np.testing.assert_allclose(tendon_parameter("tendon_solref_lim")[0], [-100.0, -20.0])
                np.testing.assert_allclose(tendon_parameter("tendon_range")[0], [-0.1, 0.1])

                attrs.tendon_solref_limit_mode.fill_(SOLREF_MODE_MJCF_DEFAULT)
                solver.notify_model_changed(ModelFlags.TENDON_PROPERTIES)
                np.testing.assert_allclose(tendon_parameter("tendon_solref_lim")[0], [0.02, 1.0])
                np.testing.assert_allclose(attrs.tendon_solref_limit.numpy()[0], [-100.0, -20.0])

    def test_joint_force_updates_preserve_pending_tendon_limits(self):
        """Publish joint damping without consuming tendon edits in eager or captured updates."""
        modes = (SOLREF_MODE_FORCE_SPACE, SOLREF_MODE_RAW, SOLREF_MODE_MJCF_DEFAULT)
        for use_cpu, capture, mode in itertools.product((True, False), (False, True), modes):
            if capture and use_cpu:
                continue
            if not use_cpu and not wp.get_cuda_device_count():
                continue
            device = wp.get_device("cpu" if use_cpu else "cuda:0")
            if capture and not wp.is_mempool_enabled(device):
                continue
            with self.subTest(use_cpu=use_cpu, capture=capture, mode=mode), wp.ScopedDevice(device):
                model = self._build_tendon_limit_model(worlds=1 if use_cpu else 2)
                solver = SolverMuJoCo(model, use_mujoco_cpu=use_cpu, disable_contacts=True)

                def runtime_parameter(name, solver=solver):
                    """Read the active backend rather than the Warp backend's host template."""
                    if solver.use_mujoco_cpu:
                        return getattr(solver.mj_model, name).copy()
                    return getattr(solver.mjw_model, name).numpy().copy()

                initial_range = runtime_parameter("tendon_range")
                initial_solref = runtime_parameter("tendon_solref_lim")
                attrs = model.mujoco
                attrs.tendon_range.fill_(wp.vec2(-0.5, 0.5))
                attrs.tendon_limit_ke.fill_(100.0)
                attrs.tendon_limit_kd.fill_(20.0)
                attrs.tendon_solref_limit.fill_(wp.vec2(-200.0, -40.0))
                attrs.tendon_solref_limit_mode.fill_(mode)
                model.joint_damping.fill_(0.6)

                if capture:
                    with wp.ScopedCapture(device=device) as graph:
                        solver.notify_model_changed(ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
                    wp.capture_launch(graph.graph)
                else:
                    solver.notify_model_changed(ModelFlags.JOINT_DOF_FORCE_PROPERTIES)

                np.testing.assert_allclose(runtime_parameter("dof_damping"), 0.6)
                np.testing.assert_array_equal(runtime_parameter("tendon_range"), initial_range)
                np.testing.assert_array_equal(runtime_parameter("tendon_solref_lim"), initial_solref)

                solver.notify_model_changed(ModelFlags.TENDON_PROPERTIES)
                expected_range = np.broadcast_to([-0.5, 0.5], initial_range.shape)
                np.testing.assert_allclose(runtime_parameter("tendon_range"), expected_range)
                self.assertFalse(np.array_equal(runtime_parameter("tendon_solref_lim"), initial_solref))

    def test_tendon_limit_force_gains_select_worlds(self):
        """Keep native parameters in untouched worlds and update selected gains independently."""
        model = self._build_tendon_limit_model(worlds=2)
        self.assertFalse(hasattr(model.mujoco, "tendon_limit_gains_enabled"))
        model.mujoco.tendon_limit_ke.fill_(100.0)
        model.mujoco.tendon_limit_kd.fill_(20.0)
        model.mujoco.tendon_solref_limit_mode.assign(
            np.array([SOLREF_MODE_FORCE_SPACE, SOLREF_MODE_RAW], dtype=np.int32)
        )
        solver = SolverMuJoCo(model, disable_contacts=True)
        initial = solver.mjw_model.tendon_solref_lim.numpy().copy()
        np.testing.assert_allclose(initial[1, 0], [-100.0, -20.0])
        self.assertTrue(np.all(initial[0, 0] > 0.0))
        self.assertGreater(float(model.mujoco.tendon_limit_ke.numpy()[1]), 0.0)

        model.mujoco.tendon_limit_ke.assign(np.array([0.0, 100.0], dtype=np.float32))
        model.mujoco.tendon_limit_kd.fill_(20.0)
        model.mujoco.tendon_solref_limit_mode.fill_(SOLREF_MODE_FORCE_SPACE)
        solver.notify_model_changed(ModelFlags.TENDON_PROPERTIES)
        updated = solver.mjw_model.tendon_solref_lim.numpy()
        np.testing.assert_allclose(updated[1, 0], initial[0, 0])
        ranges = solver.mjw_model.tendon_range.numpy()
        self.assertTrue(np.isneginf(ranges[0, 0, 0]))
        self.assertTrue(np.isposinf(ranges[0, 0, 1]))
        np.testing.assert_allclose(ranges[1, 0], [-0.1, 0.1])

    def test_tendon_limit_updates_are_cuda_graph_capture_safe(self):
        """Replay per-world tendon gains, modes, and ranges through model notifications."""
        if wp.get_cuda_device_count() == 0:
            self.skipTest("CUDA graph capture requires a CUDA device")
        device = wp.get_cuda_device(0)
        if not wp.is_mempool_enabled(device):
            self.skipTest("CUDA graph capture requires the CUDA mempool allocator")

        for flags in (ModelFlags.TENDON_PROPERTIES, ModelFlags.BODY_INERTIAL_PROPERTIES, ModelFlags.ALL):
            with self.subTest(flags=flags), wp.ScopedDevice(device):
                model = self._build_tendon_limit_model(worlds=2)
                model.body_mass.assign(np.array([1.0, 3.0], dtype=np.float32))
                solver = SolverMuJoCo(model, use_mujoco_cpu=False, disable_contacts=True)
                attrs = model.mujoco
                with wp.ScopedCapture(device=device) as capture:
                    solver.notify_model_changed(flags)

                cases = (
                    ([SOLREF_MODE_FORCE_SPACE, SOLREF_MODE_FORCE_SPACE], [100.0, 240.0], [20.0, 60.0]),
                    ([SOLREF_MODE_FORCE_SPACE, SOLREF_MODE_FORCE_SPACE], [0.0, 180.0], [0.0, 0.0]),
                    ([SOLREF_MODE_RAW, SOLREF_MODE_MJCF_DEFAULT], [0.0, 0.0], [0.0, 0.0]),
                    ([SOLREF_MODE_FORCE_SPACE, SOLREF_MODE_FORCE_SPACE], [60.0, 350.0], [15.0, 70.0]),
                )
                for replay, (modes, ke, kd) in enumerate(cases, start=1):
                    with self.subTest(replay=replay):
                        authored_range = replay * np.array([[-0.2, 0.3], [-0.4, 0.5]], dtype=np.float32)
                        raw_solref = np.array([[-120.0, -12.0], [-300.0, -30.0]], dtype=np.float32)
                        attrs.tendon_solref_limit_mode.assign(np.array(modes, dtype=np.int32))
                        attrs.tendon_limit_ke.assign(np.array(ke, dtype=np.float32))
                        attrs.tendon_limit_kd.assign(np.array(kd, dtype=np.float32))
                        attrs.tendon_solref_limit.assign(raw_solref)
                        attrs.tendon_range.assign(authored_range)
                        if flags & ModelFlags.BODY_INERTIAL_PROPERTIES:
                            model.body_mass.assign(replay * np.array([2.0, 5.0], dtype=np.float32))
                        wp.capture_launch(capture.graph)

                        solref = solver.mjw_model.tendon_solref_lim.numpy()[:, 0]
                        ranges = solver.mjw_model.tendon_range.numpy()[:, 0]
                        invweight = solver.mjw_model.tendon_invweight0.numpy()[:, 0]
                        dmax = solver.mjw_model.tendon_solimp_lim.numpy()[:, 0, 1]
                        for world in range(2):
                            expected_range = authored_range[world]
                            if modes[world] == SOLREF_MODE_RAW:
                                expected_solref = raw_solref[world]
                            elif modes[world] == SOLREF_MODE_MJCF_DEFAULT:
                                expected_solref = DEFAULT_LIMIT_SOLREF
                            elif ke[world] == 0.0:
                                expected_range = [-np.inf, np.inf]
                                expected_solref = raw_solref[world]
                            else:
                                factor = invweight[world] * (1.0 - dmax[world])
                                expected_solref = (
                                    _expected_positive_limit_solref(ke[world], kd[world], factor)
                                    if kd[world] > 0.0
                                    else [-ke[world] * factor, 0.0]
                                )
                            np.testing.assert_allclose(solref[world], expected_solref, rtol=1.0e-5)
                            np.testing.assert_allclose(ranges[world], expected_range)
                        np.testing.assert_array_equal(attrs.tendon_range.numpy(), authored_range)

    def test_tendon_limit_force_gains_degenerate_scaling(self):
        """Preserve damped and undamped tendon gains when inverse-inertia scaling degenerates."""
        for use_cpu, boundary, kd in itertools.product((False, True), ("zero_invweight", "unit_dmax"), (0.0, 20.0)):
            with self.subTest(use_cpu=use_cpu, boundary=boundary, kd=kd):
                model = self._build_tendon_limit_model()
                solver = SolverMuJoCo(model, use_mujoco_cpu=use_cpu, disable_contacts=True)
                model.mujoco.tendon_limit_ke.fill_(100.0)
                model.mujoco.tendon_limit_kd.fill_(kd)
                model.mujoco.tendon_solref_limit_mode.fill_(SOLREF_MODE_FORCE_SPACE)
                if boundary == "zero_invweight":
                    # Exercise the conversion boundary without a subsequent inertia recomputation.
                    solver.mj_model.tendon_invweight0[:] = 0.0
                    solver.mjw_model.tendon_invweight0.zero_()
                    solver._update_tendon_limit_gains()
                else:
                    model.mujoco.tendon_solimp_limit.assign(np.array([[0.95, 1.0, 0.001, 0.5, 2.0]], dtype=np.float32))
                    solver.notify_model_changed(ModelFlags.TENDON_PROPERTIES)
                expected = _expected_positive_limit_solref(100.0, kd, 1.0) if kd else [-100.0, 0.0]
                if use_cpu:
                    np.testing.assert_allclose(solver.mj_model.tendon_solref_lim[0], expected, rtol=1.0e-5)
                np.testing.assert_allclose(solver.mjw_model.tendon_solref_lim.numpy()[0, 0], expected, rtol=1.0e-5)

    def _build_pendulum_model(
        self,
        mass: float,
        ke: float,
        kd: float,
        *,
        inertia: float = 0.5,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        register_mujoco_attrs: bool = False,
    ):
        builder = newton.ModelBuilder(gravity=gravity)
        if register_mujoco_attrs:
            SolverMuJoCo.register_custom_attributes(builder)
        inertia_tensor = wp.mat33(np.eye(3) * inertia)
        link = builder.add_link(mass=mass, com=wp.vec3(0.0, 0.0, 0.0), inertia=inertia_tensor)
        joint = builder.add_joint_revolute(
            parent=-1,
            child=link,
            axis=wp.vec3(0.0, 1.0, 0.0),
            limit_lower=-1.0,
            limit_upper=1.0,
            limit_ke=ke,
            limit_kd=kd,
        )
        builder.add_articulation([joint])
        return builder.finalize()

    def _build_solver_for_solref_case(
        self,
        *,
        ke: float,
        kd: float,
        use_mujoco_cpu: bool,
        inertia: float = 0.5,
        solimp: tuple[float, float, float, float, float] | None = None,
        solref: tuple[float, float] | None = None,
    ):
        model = self._build_pendulum_model(
            mass=2.0,
            ke=ke,
            kd=kd,
            inertia=inertia,
            register_mujoco_attrs=solimp is not None or solref is not None,
        )

        if solimp is not None:
            model.mujoco.solimplimit.assign(
                wp.array(np.array([solimp], dtype=np.float32), dtype=vec5, device=model.device)
            )
        if solref is not None:
            model.mujoco.solreflimit.assign(
                wp.array(np.array([solref], dtype=np.float32), dtype=wp.vec2, device=model.device)
            )
            model.mujoco.solreflimit_mode.assign(np.array([SOLREF_MODE_RAW], dtype=np.int32))

        return SolverMuJoCo(model, iterations=1, disable_contacts=True, use_mujoco_cpu=use_mujoco_cpu)

    def _expected_solref_from_solver(
        self,
        solver: SolverMuJoCo,
        ke: float,
        kd: float,
        solref: tuple[float, float] | None,
    ):
        if solref is not None:
            return np.array(solref, dtype=np.float64)
        if ke <= 0.0 or kd <= 0.0:
            return np.array([0.02, 1.0], dtype=np.float64)

        dof_invweight0 = float(solver.mj_model.dof_invweight0[0])
        dmax = float(solver.mj_model.jnt_solimp[0, 1])
        factor = dof_invweight0 * (1.0 - dmax) if dof_invweight0 > 0.0 and dmax < 1.0 else 1.0
        return _expected_positive_limit_solref(ke, kd, factor)

    def test_cpu_and_warp_joint_limit_solref_cases_match(self):
        """CPU and Warp backends must agree for each joint-limit ``solref`` branch."""
        cases = (
            {
                "name": "scale Newton force-space gains",
                "ke": 12000.0,
                "kd": 120.0,
                "solimp": (0.9, 0.95, 0.001, 0.5, 2.0),
                "solref": None,
            },
            {
                "name": "preserve MuJoCo-native solreflimit",
                "ke": 12000.0,
                "kd": 120.0,
                "solimp": None,
                "solref": (0.03, 0.7),
            },
            {
                "name": "preserve MuJoCo-native zero solreflimit",
                "ke": 12000.0,
                "kd": 120.0,
                "solimp": None,
                "solref": (0.0, 0.0),
            },
            {
                "name": "restore default solref when Newton limit is disabled",
                "ke": 0.0,
                "kd": 0.0,
                "solimp": None,
                "solref": None,
            },
            {
                "name": "fall back to unscaled gains when dmax is one",
                "ke": 700.0,
                "kd": 70.0,
                "solimp": (0.8, 1.0, 0.001, 0.5, 2.0),
                "solref": None,
            },
        )

        for case in cases:
            with self.subTest(case=case["name"]):
                solvers = (
                    self._build_solver_for_solref_case(
                        ke=case["ke"],
                        kd=case["kd"],
                        solimp=case["solimp"],
                        solref=case["solref"],
                        use_mujoco_cpu=False,
                    ),
                    self._build_solver_for_solref_case(
                        ke=case["ke"],
                        kd=case["kd"],
                        solimp=case["solimp"],
                        solref=case["solref"],
                        use_mujoco_cpu=True,
                    ),
                )

                expected = self._expected_solref_from_solver(
                    solvers[0],
                    case["ke"],
                    case["kd"],
                    case["solref"],
                )
                cpu_expected = self._expected_solref_from_solver(
                    solvers[1],
                    case["ke"],
                    case["kd"],
                    case["solref"],
                )
                np.testing.assert_allclose(cpu_expected, expected, rtol=1e-5, atol=1e-6)

                for solver in solvers:
                    mj_solref = np.array(solver.mj_model.jnt_solref[0], dtype=np.float64)
                    mjw_solref = np.array(solver.mjw_model.jnt_solref.numpy()[0, 0], dtype=np.float64)
                    np.testing.assert_allclose(mj_solref, expected, rtol=1e-5, atol=1e-6)
                    np.testing.assert_allclose(mjw_solref, expected, rtol=1e-5, atol=1e-6)

    def test_joint_limit_solref_uses_dof_invweight0(self):
        """``jnt_solref`` must use gains scaled by ``dof_invweight0 * (1 - dmax)``."""
        ke = 10000.0
        kd = 100.0
        # Use inertia = 0.5 so ``dof_invweight0 = 1/0.5 = 2`` and factor = 2 * 0.05 = 0.1,
        # which is far enough from 1.0 that the test distinguishes fixed from buggy behavior.
        model = self._build_pendulum_model(mass=2.0, ke=ke, kd=kd, inertia=0.5)

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        dof_invweight0 = float(solver.mjw_model.dof_invweight0.numpy()[0, 0])
        solimp = solver.mjw_model.jnt_solimp.numpy()[0, 0]
        dmax = float(solimp[1])
        factor = dof_invweight0 * (1.0 - dmax)

        actual_solref = solver.mjw_model.jnt_solref.numpy()[0, 0]
        expected_solref = _expected_positive_limit_solref(ke, kd, factor)
        rel_tol = 1e-4
        self.assertAlmostEqual(float(actual_solref[0]), expected_solref[0], delta=abs(expected_solref[0]) * rel_tol)
        self.assertAlmostEqual(float(actual_solref[1]), expected_solref[1], delta=abs(expected_solref[1]) * rel_tol)

    def test_joint_limit_solref_updates_after_mass_change(self):
        """Changing inertia and notifying the solver must recompute the ``invweight0`` factor."""
        ke = 8000.0
        kd = 80.0
        model = self._build_pendulum_model(mass=1.5, ke=ke, kd=kd, inertia=0.5)
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        new_mass = np.array([4.0], dtype=np.float32)
        new_inertia = np.array([np.eye(3, dtype=np.float32) * 0.2], dtype=np.float32)
        model.body_mass.assign(new_mass)
        model.body_inertia.assign(new_inertia)
        model.body_inv_mass.assign(1.0 / new_mass)
        model.body_inv_inertia.assign(np.linalg.inv(new_inertia))

        solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)

        dof_invweight0 = float(solver.mjw_model.dof_invweight0.numpy()[0, 0])
        solimp = solver.mjw_model.jnt_solimp.numpy()[0, 0]
        dmax = float(solimp[1])
        factor = dof_invweight0 * (1.0 - dmax)

        actual_solref = solver.mjw_model.jnt_solref.numpy()[0, 0]
        # ``actual_solref`` is stored in float32 on the GPU while ``expected_*``
        # is computed in float64 on the host. Use a relative tolerance so the
        # assertion passes for float32 round-off on multi-thousand-scale values
        # while still rejecting the buggy 10x-too-stiff solref.
        expected_solref = _expected_positive_limit_solref(ke, kd, factor)
        rel_tol = 1e-4
        self.assertAlmostEqual(float(actual_solref[0]), expected_solref[0], delta=abs(expected_solref[0]) * rel_tol)
        self.assertAlmostEqual(float(actual_solref[1]), expected_solref[1], delta=abs(expected_solref[1]) * rel_tol)

    @staticmethod
    def _build_mjcf_implicit_default_model(limit_ke=10000.0, limit_kd=10.0):
        """Build an MJCF model whose joint uses MuJoCo's implicit limit response."""
        builder = newton.ModelBuilder()
        builder.default_joint_cfg.limit_ke = limit_ke
        builder.default_joint_cfg.limit_kd = limit_kd
        builder.add_mjcf(
            """
            <mujoco>
              <compiler angle="radian"/>
              <worldbody>
                <body name="link">
                  <inertial pos="0 0 0" mass="2" diaginertia="0.5 0.5 0.5"/>
                  <joint name="hinge" type="hinge" axis="0 1 0" range="-1 1"/>
                  <geom type="sphere" size="0.05"/>
                </body>
              </worldbody>
            </mujoco>
            """,
            ignore_inertial_definitions=False,
        )
        return builder.finalize()

    def test_mjcf_configured_default_gains_apply_during_solver_construction(self):
        """Verify that configured builder gains override an implicit MuJoCo limit default."""
        for use_mujoco_cpu in (False, True):
            with self.subTest(use_mujoco_cpu=use_mujoco_cpu):
                model = self._build_mjcf_implicit_default_model(limit_ke=4321.0, limit_kd=76.0)

                np.testing.assert_allclose(model.mujoco.solreflimit.numpy()[0], [0.0, 0.0], rtol=0.0, atol=0.0)
                self.assertEqual(int(model.mujoco.solreflimit_mode.numpy()[0]), SOLREF_MODE_FORCE_SPACE)
                np.testing.assert_allclose(model.joint_limit_ke.numpy(), [4321.0], rtol=0.0, atol=0.0)
                np.testing.assert_allclose(model.joint_limit_kd.numpy(), [76.0], rtol=0.0, atol=0.0)

                solver = SolverMuJoCo(
                    model,
                    iterations=1,
                    disable_contacts=True,
                    use_mujoco_cpu=use_mujoco_cpu,
                )
                dof_invweight0 = float(solver.mjw_model.dof_invweight0.numpy()[0, 0])
                dmax = float(solver.mjw_model.jnt_solimp.numpy()[0, 0][1])
                factor = dof_invweight0 * (1.0 - dmax) if dof_invweight0 > 0.0 and dmax < 1.0 else 1.0
                expected_solref = _expected_positive_limit_solref(4321.0, 76.0, factor)
                actual_solref = np.array(solver.mjw_model.jnt_solref.numpy()[0, 0], dtype=np.float64)

                np.testing.assert_allclose(actual_solref, expected_solref, rtol=1e-5, atol=1e-6)

    def test_mjcf_gain_edit_before_solver_construction_promotes_force_space(self):
        """Verify that a post-import gain edit is detected when constructing the solver."""
        for use_mujoco_cpu in (False, True):
            with self.subTest(use_mujoco_cpu=use_mujoco_cpu):
                model = self._build_mjcf_implicit_default_model()
                self.assertEqual(int(model.mujoco.solreflimit_mode.numpy()[0]), SOLREF_MODE_MJCF_DEFAULT)

                ke = np.array([5000.0], dtype=np.float32)
                kd = np.array([50.0], dtype=np.float32)
                model.joint_limit_ke.assign(ke)
                model.joint_limit_kd.assign(kd)

                solver = SolverMuJoCo(
                    model,
                    iterations=1,
                    disable_contacts=True,
                    use_mujoco_cpu=use_mujoco_cpu,
                )

                self.assertEqual(int(model.mujoco.solreflimit_mode.numpy()[0]), SOLREF_MODE_FORCE_SPACE)
                dof_invweight0 = float(solver.mjw_model.dof_invweight0.numpy()[0, 0])
                dmax = float(solver.mjw_model.jnt_solimp.numpy()[0, 0][1])
                factor = dof_invweight0 * (1.0 - dmax) if dof_invweight0 > 0.0 and dmax < 1.0 else 1.0
                expected_solref = _expected_positive_limit_solref(ke[0], kd[0], factor)
                actual_solref = np.array(solver.mjw_model.jnt_solref.numpy()[0, 0], dtype=np.float64)

                np.testing.assert_allclose(actual_solref, expected_solref, rtol=1e-5, atol=1e-6)

    def test_mjcf_implicit_default_mode_promotes_after_gain_edit(self):
        """Verify that an imported MJCF default-gain edit permanently promotes force-space mode."""
        model = self._build_mjcf_implicit_default_model()
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        initial_ke = model.joint_limit_ke.numpy().copy()
        initial_kd = model.joint_limit_kd.numpy().copy()
        edited_ke = np.nextafter(initial_ke, np.full_like(initial_ke, np.inf))
        model.joint_limit_ke.assign(edited_ke)
        solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)

        self.assertEqual(int(model.mujoco.solreflimit_mode.numpy()[0]), SOLREF_MODE_FORCE_SPACE)

        model.joint_limit_ke.assign(initial_ke)
        model.joint_limit_kd.assign(initial_kd)
        solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)

        dof_invweight0 = float(solver.mjw_model.dof_invweight0.numpy()[0, 0])
        dmax = float(solver.mjw_model.jnt_solimp.numpy()[0, 0][1])
        factor = dof_invweight0 * (1.0 - dmax) if dof_invweight0 > 0.0 and dmax < 1.0 else 1.0
        expected_solref = _expected_positive_limit_solref(initial_ke[0], initial_kd[0], factor)
        actual_solref = np.array(solver.mjw_model.jnt_solref.numpy()[0, 0], dtype=np.float64)

        np.testing.assert_allclose(actual_solref, expected_solref, rtol=1e-5, atol=1e-6)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_usd_authored_zero_solreflimit_is_preserved_as_native_parameter(self):
        """USD-authored ``mjc:solreflimit = [0, 0]`` must set raw mode just like MJCF import."""
        from pxr import Sdf, Vt

        def set_zero_solref(joint):
            joint.GetPrim().CreateAttribute("mjc:solreflimit", Sdf.ValueTypeNames.DoubleArray, True).Set(
                Vt.DoubleArray([0.0, 0.0])
            )

        stage = _create_actuator_test_stage(extra_joint_attrs=set_zero_solref)
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage)
        model = builder.finalize()

        raw_dofs = np.flatnonzero(model.mujoco.solreflimit_mode.numpy() == SOLREF_MODE_RAW)
        self.assertEqual(len(raw_dofs), 1)
        raw_dof = int(raw_dofs[0])
        np.testing.assert_allclose(model.mujoco.solreflimit.numpy()[raw_dof], [0.0, 0.0], rtol=0.0, atol=0.0)

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
        jnt_to_dof = solver.mjc_jnt_to_newton_dof.numpy()[0]
        raw_mjc_joints = np.flatnonzero(jnt_to_dof == raw_dof)
        self.assertEqual(len(raw_mjc_joints), 1)
        np.testing.assert_allclose(
            solver.mjw_model.jnt_solref.numpy()[0, int(raw_mjc_joints[0])],
            [0.0, 0.0],
            rtol=0.0,
            atol=0.0,
        )

    def test_mjcf_authored_solreflimit_is_preserved_as_native_parameter(self):
        """Authored MJCF ``solreflimit`` still bypasses Newton force-space gain scaling."""
        builder = newton.ModelBuilder()
        builder.add_mjcf(
            """
            <mujoco>
              <compiler angle="radian"/>
              <worldbody>
                <body name="link">
                  <inertial pos="0 0 0" mass="2" diaginertia="0.5 0.5 0.5"/>
                  <joint name="hinge" type="hinge" axis="0 1 0" range="-1 1" solreflimit="0.03 0.7"/>
                  <geom type="sphere" size="0.05"/>
                </body>
              </worldbody>
            </mujoco>
            """,
            ignore_inertial_definitions=False,
        )
        model = builder.finalize()

        expected_solref = np.array([0.03, 0.7], dtype=np.float64)
        np.testing.assert_allclose(model.mujoco.solreflimit.numpy()[0], expected_solref, rtol=1e-6, atol=1e-6)
        self.assertEqual(int(model.mujoco.solreflimit_mode.numpy()[0]), SOLREF_MODE_RAW)

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
        np.testing.assert_allclose(solver.mjw_model.jnt_solref.numpy()[0, 0], expected_solref, rtol=1e-6, atol=1e-6)

    def test_mjcf_authored_zero_solreflimit_is_preserved_as_native_parameter(self):
        """Authored MJCF ``solreflimit="0 0"`` must not be confused with the sentinel."""
        builder = newton.ModelBuilder()
        builder.add_mjcf(
            """
            <mujoco>
              <compiler angle="radian"/>
              <worldbody>
                <body name="link">
                  <inertial pos="0 0 0" mass="2" diaginertia="0.5 0.5 0.5"/>
                  <joint name="hinge" type="hinge" axis="0 1 0" range="-1 1" solreflimit="0 0"/>
                  <geom type="sphere" size="0.05"/>
                </body>
              </worldbody>
            </mujoco>
            """,
            ignore_inertial_definitions=False,
        )
        model = builder.finalize()

        expected_solref = np.array([0.0, 0.0], dtype=np.float64)
        np.testing.assert_allclose(model.mujoco.solreflimit.numpy()[0], expected_solref, rtol=0.0, atol=0.0)
        self.assertEqual(int(model.mujoco.solreflimit_mode.numpy()[0]), SOLREF_MODE_RAW)

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
        np.testing.assert_allclose(solver.mjw_model.jnt_solref.numpy()[0, 0], expected_solref, rtol=0.0, atol=0.0)

    def test_saved_mjcf_and_mj_model_use_scaled_joint_limit_solref(self):
        """Force-space joints get the scaled ``jnt_solref`` in ``MjModel`` but no ``solreflimit`` in MJCF.

        ``save_to_mjcf`` writes ``solreflimit`` only for ``SOLREF_MODE_RAW``
        joints (issue #2009 review item I6): re-importing a scaled
        ``solreflimit`` would silently freeze the Newton force-space gains
        because the MJCF parser sets ``SOLREF_MODE_RAW`` on every authored
        ``solreflimit`` and the runtime ``jnt_solref`` would no longer
        follow ``joint_limit_ke``/``kd``.
        """
        ke = 10000.0
        kd = 100.0
        model = self._build_pendulum_model(mass=2.0, ke=ke, kd=kd, inertia=0.5)

        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            xml_path = f.name
        try:
            solver = SolverMuJoCo(model, iterations=1, disable_contacts=True, save_to_mjcf=xml_path)

            dof_invweight0 = float(solver.mj_model.dof_invweight0[0])
            dmax = float(solver.mj_model.jnt_solimp[0, 1])
            factor = dof_invweight0 * (1.0 - dmax)
            expected_solref = _expected_positive_limit_solref(ke, kd, factor)

            # Runtime ``MjModel`` still carries the derived solref so MuJoCo
            # produces the right force at this instant while retaining refsafety.
            np.testing.assert_allclose(solver.mj_model.jnt_solref[0], expected_solref, rtol=1e-6, atol=1e-6)

            # Exported MJCF must NOT author ``solreflimit`` for this
            # ``SOLREF_MODE_FORCE_SPACE`` joint (builder-API default).
            tree = ET.parse(xml_path)
            joint = next(tree.iter("joint"))
            self.assertIsNone(
                joint.get("solreflimit"),
                "FORCE_SPACE joints must not persist a derived solreflimit into the saved MJCF",
            )
        finally:
            os.unlink(xml_path)

    def test_saved_mjcf_does_not_write_limit_solref_to_unlimited_joints(self):
        """``solreflimit`` is exported only for RAW joints with active limits.

        After the I6 fix (issue #2009): ``SOLREF_MODE_FORCE_SPACE`` and
        ``SOLREF_MODE_MJCF_DEFAULT`` joints intentionally skip the
        ``solreflimit`` write to preserve round-trip semantics; unlimited
        joints are also skipped (they have no limit to author). Authored
        RAW solreflimit values still survive the export.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverMuJoCo.register_custom_attributes(builder)
        inertia = wp.mat33(np.eye(3) * 0.5)
        link_a = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=inertia, label="link_a")
        link_b = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=inertia, label="link_b")
        limited = builder.add_joint_revolute(
            parent=-1,
            child=link_a,
            axis=wp.vec3(0.0, 1.0, 0.0),
            limit_lower=-1.0,
            limit_upper=1.0,
            limit_ke=10000.0,
            limit_kd=100.0,
            label="limited",
        )
        unlimited = builder.add_joint_revolute(
            parent=link_a,
            child=link_b,
            axis=wp.vec3(0.0, 1.0, 0.0),
            limit_ke=10000.0,
            limit_kd=100.0,
            label="unlimited",
        )
        builder.add_articulation([limited, unlimited])
        model = builder.finalize()

        # Promote the limited joint to RAW with an authored solreflimit so
        # the export path has something to persist.
        mode = model.mujoco.solreflimit_mode.numpy()
        solref = model.mujoco.solreflimit.numpy()
        mode[0] = SOLREF_MODE_RAW
        solref[0] = np.array([0.03, 0.7], dtype=np.float32)
        model.mujoco.solreflimit_mode.assign(mode)
        model.mujoco.solreflimit.assign(wp.array(solref, dtype=wp.vec2, device=model.device))

        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            xml_path = f.name
        try:
            SolverMuJoCo(model, iterations=1, disable_contacts=True, save_to_mjcf=xml_path)

            tree = ET.parse(xml_path)
            joints = {joint.get("name"): joint for joint in tree.iter("joint")}
            self.assertIn("limited", joints)
            self.assertIn("unlimited", joints)
            self.assertIsNotNone(
                joints["limited"].get("solreflimit"),
                "RAW joint with authored solreflimit must be exported",
            )
            self.assertIsNone(
                joints["unlimited"].get("solreflimit"),
                "Unlimited joint must not author solreflimit",
            )
        finally:
            os.unlink(xml_path)

    def test_cpu_backend_solref_uses_updated_solimp(self):
        """CPU runtime solref scaling must use the latest ``solimplimit`` values."""
        ke = 8000.0
        kd = 80.0
        model = self._build_pendulum_model(
            mass=2.0,
            ke=ke,
            kd=kd,
            inertia=0.5,
            register_mujoco_attrs=True,
        )
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True, use_mujoco_cpu=True)

        solimp = np.array([[0.8, 0.8, 0.001, 0.5, 2.0]], dtype=np.float32)
        model.mujoco.solimplimit.assign(wp.array(solimp, dtype=vec5, device=model.device))

        solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)

        np.testing.assert_allclose(solver.mj_model.jnt_solimp[0], solimp[0], rtol=1e-6, atol=1e-6)
        dof_invweight0 = float(solver.mj_model.dof_invweight0[0])
        dmax = float(solimp[0, 1])
        factor = dof_invweight0 * (1.0 - dmax)
        expected_solref = _expected_positive_limit_solref(ke, kd, factor)
        np.testing.assert_allclose(solver.mj_model.jnt_solref[0], expected_solref, rtol=1e-6, atol=1e-6)

    def test_joint_limit_solref_resets_to_default_when_limit_ke_is_disabled(self):
        """Runtime ``ke -> 0`` updates must restore the same limit ``solref`` as a fresh ``ke=0`` model."""
        model = self._build_pendulum_model(mass=2.0, ke=10000.0, kd=100.0, inertia=0.5)
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        default_model = self._build_pendulum_model(mass=2.0, ke=0.0, kd=0.0, inertia=0.5)
        default_solver = SolverMuJoCo(default_model, iterations=1, disable_contacts=True)
        expected_solref = default_solver.mjw_model.jnt_solref.numpy()[0, 0]

        initial_solref = solver.mjw_model.jnt_solref.numpy()[0, 0]
        self.assertNotAlmostEqual(float(initial_solref[0]), float(expected_solref[0]), places=3)
        self.assertNotAlmostEqual(float(initial_solref[1]), float(expected_solref[1]), places=3)

        model.joint_limit_ke.assign(np.array([0.0], dtype=np.float32))
        model.joint_limit_kd.assign(np.array([0.0], dtype=np.float32))
        solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)

        actual_solref = solver.mjw_model.jnt_solref.numpy()[0, 0]
        self.assertAlmostEqual(float(actual_solref[0]), float(expected_solref[0]), places=6)
        self.assertAlmostEqual(float(actual_solref[1]), float(expected_solref[1]), places=6)

    def test_cpu_backend_scales_joint_limit_solref_after_set_const(self):
        """CPU ``MjModel.jnt_solref`` must use the same force-space scaling as the warp backend."""
        ke = 50000.0
        kd = 500.0
        model = self._build_pendulum_model(mass=1.0, ke=ke, kd=kd, inertia=0.5, gravity=(0.0, 0.0, 0.0))
        solver = SolverMuJoCo(model, iterations=50, disable_contacts=True, use_mujoco_cpu=True)

        def assert_scaled_solref(expected_invweight0):
            dof_invweight0 = float(solver.mj_model.dof_invweight0[0])
            self.assertAlmostEqual(dof_invweight0, expected_invweight0, delta=1e-4)
            dmax = float(solver.mj_model.jnt_solimp[0, 1])
            factor = dof_invweight0 * (1.0 - dmax)
            expected_solref = _expected_positive_limit_solref(ke, kd, factor)
            actual_solref = np.array(solver.mj_model.jnt_solref[0], dtype=np.float64)
            np.testing.assert_allclose(actual_solref, expected_solref, rtol=1e-6, atol=1e-6)

        assert_scaled_solref(expected_invweight0=2.0)

        new_mass = np.array([4.0], dtype=np.float32)
        new_inertia = np.array([np.eye(3, dtype=np.float32) * 0.2], dtype=np.float32)
        model.body_mass.assign(new_mass)
        model.body_inertia.assign(new_inertia)
        model.body_inv_mass.assign(1.0 / new_mass)
        model.body_inv_inertia.assign(np.linalg.inv(new_inertia))

        solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)

        assert_scaled_solref(expected_invweight0=5.0)

    def test_joint_limit_torque_matches_configured_stiffness(self):
        """Scaled ``jnt_solref`` produces a mass-independent constraint response.

        MuJoCo uses a soft constraint with impedance ``D = imp/(invweight*(1-imp))``
        and reference acceleration ``aref = -k_stored * pos``. After the
        ``invweight0 * (1 - dmax)`` scaling, ``k_stored = ke*invweight0*(1-dmax)``
        so the constraint-solver product ``D * k_stored = imp * ke`` is
        independent of the mass matrix — matching Newton's force-space ``ke``.

        For this pendulum (inertia=0.5, dof_invweight0=2, dmax=0.95, ke=50000,
        kd=500) the factor is ``2 * 0.05 = 0.1``. MuJoCo's single-DOF constraint
        solver then yields a one-step response of about ``qd_after ≈ -0.5 rad/s``
        with the fix. Without the fix the stored stiffness is ``ke`` directly,
        giving ``qd_after ≈ -5 rad/s`` — a 10x increase matching ``1 / factor``.
        The window below accepts the fix and rejects the 10x-too-stiff bug.
        """
        mass = 1.0
        inertia = 0.5
        ke = 50000.0
        kd = 500.0
        model = self._build_pendulum_model(mass=mass, ke=ke, kd=kd, inertia=inertia, gravity=(0.0, 0.0, 0.0))

        solver = SolverMuJoCo(model, iterations=50, disable_contacts=True)

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        overshoot = 0.1
        joint_q = state_0.joint_q.numpy()
        joint_q[0] = 1.0 + overshoot
        state_0.joint_q.assign(joint_q)
        joint_qd = state_0.joint_qd.numpy()
        joint_qd[:] = 0.0
        state_0.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

        dt = 1e-3
        solver.step(state_0, state_1, control, None, dt)

        qd_after = float(state_1.joint_qd.numpy()[0])
        # With the fix: qd_after ≈ -0.5 rad/s (stored stiffness ke*factor = 5000).
        # Without the fix: qd_after ≈ -5 rad/s (stored stiffness ke = 50000).
        self.assertLess(qd_after, -0.2)
        self.assertGreater(qd_after, -0.8)

    def test_saved_mjcf_default_solreflimit_round_trips(self):
        """MJCF without ``solreflimit`` must round-trip through ``save_to_mjcf``.

        ``spec.to_xml()`` serialises MuJoCo's default ``(0.02, 1.0)`` as a
        one-value ``solreflimit="0.02"`` shorthand. Before the fix the MJCF
        importer rejected single-value ``solreflimit`` with ``ValueError:
        ... expected 2 elements, got 3``; now the parser pads the missing
        ``dampratio`` from the MuJoCo default.
        """
        mjcf_source = """
            <mujoco>
              <compiler angle="radian"/>
              <worldbody>
                <body name="link">
                  <inertial pos="0 0 0" mass="2" diaginertia="0.5 0.5 0.5"/>
                  <joint name="hinge" type="hinge" axis="0 1 0" range="-1 1"/>
                  <geom type="sphere" size="0.05"/>
                </body>
              </worldbody>
            </mujoco>
            """
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_source, ignore_inertial_definitions=False)
        model = builder.finalize()

        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            xml_path = f.name
        try:
            SolverMuJoCo(model, iterations=1, disable_contacts=True, save_to_mjcf=xml_path)
            # Re-import must succeed (this is the round-trip failure repro).
            roundtrip_builder = newton.ModelBuilder()
            roundtrip_builder.add_mjcf(xml_path, ignore_inertial_definitions=False)
            roundtrip_builder.finalize()
        finally:
            os.unlink(xml_path)

    def test_solreflimit_mode_force_space_overrides_authored_raw_value(self):
        """``SOLREF_MODE_FORCE_SPACE`` must scale gains even when ``solreflimit`` is non-zero.

        The earlier ``raw_solref_is_set`` shortcut silently forwarded any
        non-zero ``mujoco.solreflimit`` regardless of the configured mode, so a
        user opting into Newton force-space scaling would still see the raw
        value at the constraint level. With the fix the explicit mode is
        authoritative.
        """
        for use_cpu in (False, True):
            with self.subTest(use_mujoco_cpu=use_cpu):
                ke = 10000.0
                kd = 100.0
                model = self._build_pendulum_model(
                    mass=2.0,
                    ke=ke,
                    kd=kd,
                    inertia=0.5,
                    register_mujoco_attrs=True,
                )
                model.mujoco.solreflimit.assign(
                    wp.array(np.array([[0.03, 0.7]], dtype=np.float32), dtype=wp.vec2, device=model.device)
                )
                model.mujoco.solreflimit_mode.assign(np.array([SOLREF_MODE_FORCE_SPACE], dtype=np.int32))

                solver = SolverMuJoCo(model, iterations=1, disable_contacts=True, use_mujoco_cpu=use_cpu)

                if use_cpu:
                    dof_invweight0 = float(solver.mj_model.dof_invweight0[0])
                    dmax = float(solver.mj_model.jnt_solimp[0, 1])
                    actual_solref = np.array(solver.mj_model.jnt_solref[0], dtype=np.float64)
                else:
                    dof_invweight0 = float(solver.mjw_model.dof_invweight0.numpy()[0, 0])
                    dmax = float(solver.mjw_model.jnt_solimp.numpy()[0, 0][1])
                    actual_solref = np.array(solver.mjw_model.jnt_solref.numpy()[0, 0], dtype=np.float64)
                factor = dof_invweight0 * (1.0 - dmax) if dof_invweight0 > 0.0 and dmax < 1.0 else 1.0
                expected_solref = _expected_positive_limit_solref(ke, kd, factor)

                self.assertFalse(np.allclose(actual_solref, [0.03, 0.7]))
                np.testing.assert_allclose(actual_solref, expected_solref, rtol=1e-5, atol=1e-6)

    def test_joint_limit_solref_scales_per_dof_on_multi_dof_joint(self):
        """Multi-DOF joints scale ``jnt_solref`` independently per DOF.

        Exercises the ``jnt_dofadr`` lookup and the per-``mjc_jnt`` →
        ``newton_dof`` mapping in ``update_jnt_solref_from_invweight0_kernel``
        for a D6 joint with three angular DOFs. Each DOF gets a distinct
        ``joint_limit_ke``/``kd`` pair so the per-DOF scaling can be observed
        in the resulting ``jnt_solref`` rows.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverMuJoCo.register_custom_attributes(builder)
        inertia = wp.mat33(np.eye(3) * 0.3)
        link = builder.add_link(
            mass=1.5,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=inertia,
            label="link",
        )
        # Three angular DOFs with distinct gains so per-DOF scaling is
        # observable in the output. Limits must be finite so MuJoCo emits a
        # ``jnt_limited`` slot per DOF.
        ke_x, kd_x = 5000.0, 50.0
        ke_y, kd_y = 12000.0, 120.0
        ke_z, kd_z = 8000.0, 80.0
        cfg_x = newton.ModelBuilder.JointDofConfig(
            axis=wp.vec3(1.0, 0.0, 0.0),
            limit_lower=-1.0,
            limit_upper=1.0,
            limit_ke=ke_x,
            limit_kd=kd_x,
        )
        cfg_y = newton.ModelBuilder.JointDofConfig(
            axis=wp.vec3(0.0, 1.0, 0.0),
            limit_lower=-1.0,
            limit_upper=1.0,
            limit_ke=ke_y,
            limit_kd=kd_y,
        )
        cfg_z = newton.ModelBuilder.JointDofConfig(
            axis=wp.vec3(0.0, 0.0, 1.0),
            limit_lower=-1.0,
            limit_upper=1.0,
            limit_ke=ke_z,
            limit_kd=kd_z,
        )
        j = builder.add_joint_d6(parent=-1, child=link, angular_axes=[cfg_x, cfg_y, cfg_z])
        builder.add_articulation([j])
        model = builder.finalize()

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # MuJoCo expands the D6 into three single-DOF joints. ``jnt_solref``
        # rows must reflect each DOF's own ke/kd, scaled by that DOF's
        # ``dof_invweight0`` and the per-joint ``solimp[1]``.
        jnt_solref = solver.mjw_model.jnt_solref.numpy()[0]
        dof_invweight0 = solver.mjw_model.dof_invweight0.numpy()[0]
        jnt_solimp = solver.mjw_model.jnt_solimp.numpy()[0]
        jnt_dofadr = solver.mjw_model.jnt_dofadr.numpy()

        expected_gains = [(ke_x, kd_x), (ke_y, kd_y), (ke_z, kd_z)]
        self.assertGreaterEqual(jnt_solref.shape[0], 3)
        for mjc_jnt, (ke, kd) in enumerate(expected_gains):
            dof_idx = int(jnt_dofadr[mjc_jnt])
            invw = float(dof_invweight0[dof_idx])
            dmax = float(jnt_solimp[mjc_jnt][1])
            factor = invw * (1.0 - dmax) if invw > 0.0 and dmax < 1.0 else 1.0
            expected = _expected_positive_limit_solref(ke, kd, factor)
            np.testing.assert_allclose(
                jnt_solref[mjc_jnt],
                expected,
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"D6 DOF {mjc_jnt}: expected positive solref {expected.tolist()}, "
                f"got {jnt_solref[mjc_jnt].tolist()}",
            )

        # Each DOF's solref is distinct (the three ke/kd pairs differ), which
        # implicitly verifies that the kernel uses ``jnt_dofadr`` rather than
        # broadcasting a single DOF's gain across the joint.
        self.assertFalse(np.allclose(jnt_solref[0], jnt_solref[1]))
        self.assertFalse(np.allclose(jnt_solref[1], jnt_solref[2]))

    def test_invalid_raw_solreflimit_warns_in_update_solref(self):
        """Warn once on invalid RAW solreflimit and re-arm on eager DOF notifications."""
        for use_mujoco_cpu in (True, False):
            with self.subTest(use_mujoco_cpu=use_mujoco_cpu):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                SolverMuJoCo.register_custom_attributes(builder)
                inertia = wp.mat33(np.eye(3) * 0.5)
                link = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=inertia)
                joint = builder.add_joint_revolute(
                    parent=-1,
                    child=link,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    limit_lower=-1.0,
                    limit_upper=1.0,
                    limit_ke=2500.0,
                    limit_kd=100.0,
                )
                builder.add_articulation([joint])
                model = builder.finalize()

                # Promote to RAW with a mixed-sign solref — MuJoCo would silently
                # treat the timeconst as 2500s and effectively disable the limit.
                model.mujoco.solreflimit_mode.assign(np.array([SOLREF_MODE_RAW], dtype=np.int32))
                model.mujoco.solreflimit.assign(
                    wp.array(np.array([[-2500.0, 1.0]], dtype=np.float32), dtype=wp.vec2, device=model.device)
                )

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    solver = SolverMuJoCo(model, iterations=1, disable_contacts=True, use_mujoco_cpu=use_mujoco_cpu)
                messages = [str(w.message) for w in caught]
                self.assertTrue(
                    any("invalid components" in m and "DOF indices" in m for m in messages),
                    f"Expected RAW solreflimit domain warning, got: {messages}",
                )

                # Second notify with no change must not re-warn — the one-shot flag
                # is sticky outside JOINT_DOF_PROPERTIES.
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)
                self.assertFalse(
                    any("invalid components" in str(w.message) for w in caught),
                    "RAW solreflimit warn must not re-fire on BODY_INERTIAL_PROPERTIES",
                )

                # Reassigned values must be validated on both backends.
                model.mujoco.solreflimit.fill_(wp.vec2(0.0, 1.0))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
                self.assertEqual(
                    sum("invalid components" in str(w.message) for w in caught),
                    1,
                    "JOINT_DOF_PROPERTIES must re-arm the RAW solreflimit validator on both backends",
                )

    def test_mjcf_invalid_solreflimit_emits_import_warning(self):
        """Verify that MJCF import warns for invalid authored ``solreflimit`` signs."""
        for joint_type in ("hinge", "ball"):
            with self.subTest(joint_type=joint_type):
                mjcf_source = f"""
                    <mujoco>
                      <compiler angle="radian"/>
                      <worldbody>
                        <body name="link">
                          <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
                          <joint name="joint" type="{joint_type}" solreflimit="-1 1"/>
                          <geom type="sphere" size="0.05"/>
                        </body>
                      </worldbody>
                    </mujoco>
                    """
                builder = newton.ModelBuilder()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    builder.add_mjcf(mjcf_source, ignore_inertial_definitions=False)
                messages = [str(w.message) for w in caught]
                self.assertTrue(
                    any("invalid solreflimit" in m and "joint" in m.lower() for m in messages),
                    f"Expected MJCF importer warning for invalid solreflimit, got: {messages}",
                )

    def test_mjcf_valid_solreflimit_does_not_emit_import_warning(self):
        """Verify that MJCF import accepts native standard, direct, and default solref pairs."""
        for joint_type, solreflimit in itertools.product(
            ("hinge", "ball"),
            ("0.03 0.7", "-100 -1", "0 0"),
        ):
            with self.subTest(joint_type=joint_type, solreflimit=solreflimit):
                mjcf_source = f"""
                    <mujoco>
                      <compiler angle="radian"/>
                      <worldbody>
                        <body name="link">
                          <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
                          <joint name="joint" type="{joint_type}" solreflimit="{solreflimit}"/>
                          <geom type="sphere" size="0.05"/>
                        </body>
                      </worldbody>
                    </mujoco>
                    """
                builder = newton.ModelBuilder()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    builder.add_mjcf(mjcf_source, ignore_inertial_definitions=False)
                messages = [str(w.message) for w in caught]
                self.assertFalse(
                    any("invalid solreflimit" in message for message in messages),
                    f"Expected no MJCF importer warning for valid solreflimit, got: {messages}",
                )

    def test_parse_vec_unknown_shorthand_warns(self):
        """``parse_vec`` warns when a non-whitelisted multi-component MJCF attribute gets a single value.

        ``solreflimit="0.04"`` (whitelisted) must round-trip silently; an
        unrelated multi-component attribute like ``actuatorfrcrange="1"``
        gets replicate semantics with a warning.
        """
        # Whitelisted: no warning, shorthand pads from default.
        mjcf_whitelisted = """
            <mujoco>
              <compiler angle="radian"/>
              <worldbody>
                <body name="link">
                  <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
                  <joint name="hinge" type="hinge" axis="0 1 0" range="-1 1" solreflimit="0.04"/>
                  <geom type="sphere" size="0.05"/>
                </body>
              </worldbody>
            </mujoco>
            """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            newton.ModelBuilder().add_mjcf(mjcf_whitelisted, ignore_inertial_definitions=False)
        self.assertFalse(
            any("provided a single value" in str(w.message) for w in caught),
            "Whitelisted solreflimit shorthand must not warn",
        )

        # Non-whitelisted multi-component shorthand: ``range`` is a vec2
        # joint attribute that's not on the solref whitelist. A single
        # value should trigger the warn-and-replicate fallback so the
        # semantic drift surfaces.
        mjcf_unexpected = """
            <mujoco>
              <compiler angle="radian"/>
              <worldbody>
                <body name="link">
                  <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
                  <joint name="hinge" type="hinge" axis="0 1 0" range="0.5"/>
                  <geom type="sphere" size="0.05"/>
                </body>
              </worldbody>
            </mujoco>
            """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            newton.ModelBuilder().add_mjcf(mjcf_unexpected, ignore_inertial_definitions=False)
        self.assertTrue(
            any("provided a single value" in str(w.message) and "range" in str(w.message) for w in caught),
            f"Non-whitelisted multi-component shorthand must warn; got: {[str(w.message) for w in caught]}",
        )


class TestMuJoCoSolverForceSpaceContactSolref(unittest.TestCase):
    """Force-space ``shape_material_ke``/``shape_material_kd`` for the MuJoCo solver.

    Issue #2009: MJCF/USD importers used to convert ``mjc:solref``
    (acceleration space) directly into Newton ``shape_material_ke`` /
    ``shape_material_kd`` (force space), corrupting the documented
    semantics. The fix stores raw ``mjc:solref`` in the
    ``mujoco.solref`` custom attribute and derives the per-contact
    ``solref`` from Newton ke/kd plus ``body_invweight0[A] +
    body_invweight0[B]`` in ``convert_newton_contacts_to_mjwarp_kernel``.

    Tests use ``use_mujoco_contacts=False`` explicitly so the
    Newton-contacts kernel (where the override lives) runs. They read
    ``mjw_data.contact.solref`` back directly and compare to the
    ``convert_solref`` conversion of ``(ke·factor, kd·factor)`` —
    mirroring the joint-limit suite
    pattern (:meth:`TestMuJoCoSolverInvweightScaledSolref
    .test_joint_limit_solref_uses_dof_invweight0`). ``use_mujoco_contacts
    =True`` and the MuJoCo CPU backend remain unsupported for
    ``FORCE_SPACE``: MuJoCo's internal ``contact_params`` operates on
    per-geom ``solref`` and cannot reproduce the two-body invweight sum.
    """

    @staticmethod
    def _make_force_space(model) -> None:
        mode = np.full(model.shape_count, SOLREF_MODE_FORCE_SPACE, dtype=np.int32)
        model.mujoco.solref_mode.assign(mode)

    def _build_box_on_plane(self, *, mass: float, ke: float, kd: float):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.ke = ke
        builder.default_shape_cfg.kd = kd
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.09), wp.quat_identity()), label="dyn")
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.ke = ke
        cfg.kd = kd
        cfg.density = mass / 0.008  # Box half-extents 0.1 → volume 0.008 m³.
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
        model = builder.finalize()
        self._make_force_space(model)
        return model, body

    def _run_to_first_contact(self, model, solver, *, max_substeps: int = 60, sim_dt: float = 1.0 / 240.0):
        """Step until at least one contact is generated, then return its index.

        Direct contact-injection via ``CollisionPipeline`` would also work,
        but stepping the same code path the user does keeps the test honest
        about the kernel actually firing.
        """
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        state_in, state_out, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        for _ in range(max_substeps):
            state_in.clear_forces()
            collision_pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, sim_dt)
            state_in, state_out = state_out, state_in
            if int(solver.mjw_data.nacon.numpy()[0]) > 0:
                return contacts
        self.fail("No contacts generated; the box never settled onto the plane.")

    def _expected_force_space_solref(self, solver, contact_idx: int, *, ke: float, kd: float):
        geom_pair = solver.mjw_data.contact.geom.numpy()[contact_idx]
        geom_a, geom_b = int(geom_pair[0]), int(geom_pair[1])
        geom_bodyid = solver.mjw_model.geom_bodyid.numpy()
        body_a, body_b = int(geom_bodyid[geom_a]), int(geom_bodyid[geom_b])
        body_invweight0 = solver.mjw_model.body_invweight0.numpy()
        invw_a = float(body_invweight0[0, body_a, 0]) if body_a >= 0 else 0.0
        invw_b = float(body_invweight0[0, body_b, 0]) if body_b >= 0 else 0.0
        dmax = float(solver.mjw_data.contact.solimp.numpy()[contact_idx, 1])
        factor = (invw_a + invw_b) * (1.0 - dmax)
        solref = convert_solref(max(ke * factor, MJ_MINVAL), max(kd * factor, MJ_MINVAL), 1.0, 1.0)
        return float(solref[0]), float(solref[1])

    def test_force_space_contact_solref_uses_invweight0_and_dmax(self):
        """Per-contact ``solref`` must equal the ``convert_solref`` conversion of
        ``(ke·factor, kd·factor)`` with ``factor = (invw_a + invw_b) * (1 - dmax)``."""
        ke, kd, mass = 1.0e4, 100.0, 5.0
        model, _ = self._build_box_on_plane(mass=mass, ke=ke, kd=kd)
        solver = SolverMuJoCo(model, use_mujoco_contacts=False, njmax=20, nconmax=20, iterations=10)
        self._run_to_first_contact(model, solver)

        nacon = int(solver.mjw_data.nacon.numpy()[0])
        self.assertGreater(nacon, 0)
        expected_ref, expected_damp = self._expected_force_space_solref(solver, 0, ke=ke, kd=kd)
        actual_solref = solver.mjw_data.contact.solref.numpy()[0]
        rel_tol = 1.0e-4
        self.assertAlmostEqual(float(actual_solref[0]), expected_ref, delta=abs(expected_ref) * rel_tol)
        self.assertAlmostEqual(float(actual_solref[1]), expected_damp, delta=abs(expected_damp) * rel_tol)

    def test_force_space_contact_solref_uses_two_body_invweight_sum(self):
        """Dynamic-vs-dynamic contact: factor uses the sum of both bodies' invweight0."""
        ke, kd = 1.0e4, 100.0
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.ke = ke
        builder.default_shape_cfg.kd = kd
        builder.add_ground_plane()
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.ke = ke
        cfg.kd = kd
        cfg.density = 1000.0  # 8 kg (0.008 m³ * 1000 kg/m³)
        bottom = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.09), wp.quat_identity()), label="bottom")
        builder.add_shape_box(bottom, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
        top = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.29), wp.quat_identity()), label="top")
        builder.add_shape_box(top, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
        model = builder.finalize()
        self._make_force_space(model)
        solver = SolverMuJoCo(model, use_mujoco_contacts=False, njmax=32, nconmax=30, iterations=10)
        self._run_to_first_contact(model, solver, max_substeps=120)

        # Find a contact between the two dynamic boxes (both bodies > 0).
        nacon = int(solver.mjw_data.nacon.numpy()[0])
        geom_bodyid = solver.mjw_model.geom_bodyid.numpy()
        geom_pairs = solver.mjw_data.contact.geom.numpy()[:nacon]
        dyn_dyn_idx = next(
            (
                i
                for i, (ga, gb) in enumerate(geom_pairs)
                if int(geom_bodyid[int(ga)]) > 0 and int(geom_bodyid[int(gb)]) > 0
            ),
            None,
        )
        if dyn_dyn_idx is None:
            self.skipTest("Stack did not settle into a top-bottom contact in time; not the kernel's fault.")
        expected_ref, expected_damp = self._expected_force_space_solref(solver, dyn_dyn_idx, ke=ke, kd=kd)
        # The dynamic-dynamic factor sums both invweight0; double-check it's
        # actually larger than the static-dynamic factor used in test_*_uses_invweight0.
        body_invweight0 = solver.mjw_model.body_invweight0.numpy()
        self.assertGreater(float(body_invweight0[0, int(geom_bodyid[int(geom_pairs[dyn_dyn_idx][0])]), 0]), 0.0)
        self.assertGreater(float(body_invweight0[0, int(geom_bodyid[int(geom_pairs[dyn_dyn_idx][1])]), 0]), 0.0)
        actual_solref = solver.mjw_data.contact.solref.numpy()[dyn_dyn_idx]
        rel_tol = 1.0e-4
        self.assertAlmostEqual(float(actual_solref[0]), expected_ref, delta=abs(expected_ref) * rel_tol)
        self.assertAlmostEqual(float(actual_solref[1]), expected_damp, delta=abs(expected_damp) * rel_tol)

    def test_force_space_contact_solref_updates_after_mass_change(self):
        """``BODY_INERTIAL_PROPERTIES`` notify must refresh the per-contact factor.

        Same-solver mass-change regression: catches stale ``body_invweight0``
        if the contact fast path is not invalidated after a mass update.
        """
        ke, kd = 1.0e4, 100.0
        model, body = self._build_box_on_plane(mass=2.0, ke=ke, kd=kd)
        solver = SolverMuJoCo(model, use_mujoco_contacts=False, njmax=20, nconmax=20, iterations=10)
        self._run_to_first_contact(model, solver)
        initial_solref_ref, _ = self._expected_force_space_solref(solver, 0, ke=ke, kd=kd)

        # Double the body mass and inertia; invweight0 should halve.
        body_mass = model.body_mass.numpy()
        body_mass[body] *= 4.0
        model.body_mass.assign(body_mass)
        body_inv_mass = model.body_inv_mass.numpy()
        body_inv_mass[body] /= 4.0
        model.body_inv_mass.assign(body_inv_mass)
        body_inertia = model.body_inertia.numpy()
        body_inertia[body] *= 4.0
        model.body_inertia.assign(body_inertia)
        body_inv_inertia = model.body_inv_inertia.numpy()
        body_inv_inertia[body] = np.linalg.inv(body_inertia[body])
        model.body_inv_inertia.assign(body_inv_inertia)
        solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)
        self._run_to_first_contact(model, solver)

        updated_solref_ref, _ = self._expected_force_space_solref(solver, 0, ke=ke, kd=kd)
        actual_solref = solver.mjw_data.contact.solref.numpy()[0]
        rel_tol = 1.0e-4
        self.assertAlmostEqual(float(actual_solref[0]), updated_solref_ref, delta=abs(updated_solref_ref) * rel_tol)
        # Heavier body → smaller factor → larger solref[0] (2 / (kd * factor)).
        self.assertGreater(updated_solref_ref, initial_solref_ref)

    def test_force_space_contact_mixed_mode_falls_through_to_geom_solref(self):
        """Mixed-mode (FORCE_SPACE + non-FORCE_SPACE) must bypass the override.

        The override only fires when *both* shapes in a contact are
        ``FORCE_SPACE``. A force-space box on a default-mode plane must
        leave the per-geom mixed ``solref`` untouched.
        """
        ke, kd = 1.0e4, 100.0
        model, _ = self._build_box_on_plane(mass=2.0, ke=ke, kd=kd)
        # Demote the ground-plane (shape 0) to MJCF_DEFAULT, leave the dynamic box at FORCE_SPACE.
        mode = model.mujoco.solref_mode.numpy()
        mode[0] = SOLREF_MODE_MJCF_DEFAULT
        model.mujoco.solref_mode.assign(mode)
        solver = SolverMuJoCo(model, use_mujoco_contacts=False, njmax=20, nconmax=20, iterations=10)
        self._run_to_first_contact(model, solver)
        actual_solref = solver.mjw_data.contact.solref.numpy()[0]
        # Override bypassed: solref must differ from the force-space conversion.
        override_ref, override_damp = self._expected_force_space_solref(solver, 0, ke=ke, kd=kd)
        self.assertFalse(
            np.isclose(float(actual_solref[0]), override_ref, rtol=1e-3)
            and np.isclose(float(actual_solref[1]), override_damp, rtol=1e-3),
            "mixed-mode contact must not use the force-space override solref",
        )

    def test_force_space_contact_solref_dmax_one_disables_override(self):
        """Guard boundary: ``dmax >= 1`` must skip the override (factor would be zero)."""
        ke, kd = 1.0e4, 100.0
        model, _ = self._build_box_on_plane(mass=2.0, ke=ke, kd=kd)
        # Pin solimp.dmax to 1.0 on every shape (the kernel reads dmax from the
        # mixed contact solimp; setting both shapes to 1.0 forces the mixed value to 1.0).
        solimp = model.mujoco.geom_solimp.numpy()
        solimp[:, 1] = 1.0
        model.mujoco.geom_solimp.assign(solimp)
        solver = SolverMuJoCo(model, use_mujoco_contacts=False, njmax=20, nconmax=20, iterations=10)
        self._run_to_first_contact(model, solver)
        actual_solref = solver.mjw_data.contact.solref.numpy()[0]
        # dmax=1 zeroes the factor → override skipped; solref[0] stays sub-1.0
        # instead of the ~2/MJ_MINVAL a fired override would produce.
        self.assertLess(float(actual_solref[0]), 1.0)

    def test_force_space_contact_stiff_contact_stays_finite(self):
        """A very stiff force-space contact must stay finite — the contact
        analogue of #3109, where a too-stiff constraint diverged."""
        # Low mass + very stiff ke make the contact too stiff for the step, so
        # the written timeconst falls below 2*dt and refsafe must engage.
        ke, kd, mass = 1.0e7, 1.0e3, 0.05
        sim_dt = 1.0 / 60.0
        model, _ = self._build_box_on_plane(mass=mass, ke=ke, kd=kd)
        solver = SolverMuJoCo(model, use_mujoco_contacts=False, njmax=20, nconmax=20, iterations=10)

        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        state_in, state_out, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

        contacted = False
        for _ in range(120):
            state_in.clear_forces()
            collision_pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, sim_dt)
            state_in, state_out = state_out, state_in
            if int(solver.mjw_data.nacon.numpy()[0]) > 0:
                contacted = True
                break
        self.assertTrue(contacted, "box never contacted the plane")

        # Written (unclamped) timeconst < 2*dt: the regime where the old negative
        # direct-mode solref bypassed refsafe and diverged.
        timeconst = float(solver.mjw_data.contact.solref.numpy()[0][0])
        self.assertLess(timeconst, 2.0 * sim_dt)

        # Step through the stiff contact; it must stay finite and bounded
        # (the #3109 joint analogue went non-finite around step 98).
        for _ in range(200):
            state_in.clear_forces()
            collision_pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, sim_dt)
            state_in, state_out = state_out, state_in

        qvel = solver.mjw_data.qvel.numpy()
        qfrc = solver.mjw_data.qfrc_constraint.numpy()
        body_qd = state_in.body_qd.numpy()
        self.assertTrue(np.all(np.isfinite(qvel)), "qvel went non-finite")
        self.assertTrue(np.all(np.isfinite(qfrc)), "qfrc_constraint went non-finite")
        self.assertTrue(np.all(np.isfinite(body_qd)), "body_qd went non-finite")
        self.assertLess(float(np.max(np.abs(body_qd))), 1.0e3, "contact response blew up")

    def test_force_space_with_mujoco_contacts_emits_startup_warning(self):
        """Opting into FORCE_SPACE while running ``use_mujoco_contacts=True`` must warn at startup.

        Force-space scaling silently regresses to ``convert_solref(ke,kd,1,1)``
        on the MuJoCo-contacts and CPU backends because the per-contact
        invweight0 override lives in ``convert_newton_contacts_to_mjwarp_kernel``,
        which only runs with ``use_mujoco_contacts=False``. Without an early
        warning users hit the original #2009 bug with no signal.
        """
        model, _ = self._build_box_on_plane(mass=2.0, ke=1.0e4, kd=100.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            SolverMuJoCo(model, use_mujoco_contacts=True, njmax=20, iterations=10)
        messages = [str(w.message) for w in caught if "SOLREF_MODE_FORCE_SPACE" in str(w.message)]
        self.assertEqual(len(messages), 1, f"expected one FORCE_SPACE warning, got {messages}")

    def test_force_space_on_newton_contacts_does_not_warn(self):
        """The Newton-contacts path supports FORCE_SPACE: no warning should fire."""
        model, _ = self._build_box_on_plane(mass=2.0, ke=1.0e4, kd=100.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            SolverMuJoCo(model, use_mujoco_contacts=False, njmax=20, nconmax=20, iterations=10)
        messages = [str(w.message) for w in caught if "SOLREF_MODE_FORCE_SPACE" in str(w.message)]
        self.assertEqual(len(messages), 0, f"unexpected FORCE_SPACE warning(s): {messages}")

    def test_force_space_contact_solref_heterogeneous_mix(self):
        """Heterogeneous ke/kd: the override must blend per-shape values via ``contact_params`` ``mix``.

        ``mix`` reproduces MuJoCo's solmix/priority weighting, so the
        force-space override has to reuse it; otherwise heterogeneous
        materials would combine inconsistently with friction/solimp.
        """
        ke_a, kd_a = 5.0e3, 50.0
        ke_b, kd_b = 2.0e4, 200.0
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.ke = ke_a
        builder.default_shape_cfg.kd = kd_a
        # Ground plane (shape 0) inherits the softer (ke_a, kd_a).
        builder.add_ground_plane()
        # Dynamic box uses the stiffer (ke_b, kd_b).
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.ke = ke_b
        cfg.kd = kd_b
        cfg.density = 2.0 / 0.008
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.09), wp.quat_identity()), label="dyn")
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
        model = builder.finalize()
        self._make_force_space(model)
        solver = SolverMuJoCo(model, use_mujoco_contacts=False, njmax=20, nconmax=20, iterations=10)
        self._run_to_first_contact(model, solver)

        actual_solref = solver.mjw_data.contact.solref.numpy()[0]
        # Per MuJoCo's solmix rule with equal priority and unit solmix on both
        # shapes, ``mix == 0.5`` exactly. Recover the mixed ke/kd, then assert
        # ``contact.solref`` equals the ``convert_solref`` conversion of the
        # blended pair.
        mix_ke = 0.5 * ke_a + 0.5 * ke_b
        mix_kd = 0.5 * kd_a + 0.5 * kd_b
        rel_tol = 1.0e-3
        expected_ref, expected_damp = self._expected_force_space_solref(solver, 0, ke=mix_ke, kd=mix_kd)
        self.assertAlmostEqual(float(actual_solref[0]), expected_ref, delta=abs(expected_ref) * rel_tol)
        self.assertAlmostEqual(float(actual_solref[1]), expected_damp, delta=abs(expected_damp) * rel_tol)
        # Sanity: the blended solref[0] (which tracks ``kd``) lies strictly
        # between the single-material endpoints, the smoke test for ``mix``
        # actually being used.
        ref_a, _ = self._expected_force_space_solref(solver, 0, ke=ke_a, kd=kd_a)
        ref_b, _ = self._expected_force_space_solref(solver, 0, ke=ke_b, kd=kd_b)
        lo, hi = sorted((ref_a, ref_b))
        self.assertGreater(float(actual_solref[0]), lo)
        self.assertLess(float(actual_solref[0]), hi)

    def test_mjcf_default_mode_preserved(self):
        """Unauthored MJCF geoms stay in ``SOLREF_MODE_MJCF_DEFAULT``.

        Force-space scaling is opt-in for shapes: imported MuJoCo geoms
        keep the compiled contact behavior until the user explicitly sets
        ``model.mujoco.solref_mode[shape] = SOLREF_MODE_FORCE_SPACE``.
        Unlike the joint-limit auto-promote (PR #2610), there is no
        automatic flip when ``shape_material_ke``/``kd`` drift: per-example
        overrides of ``ModelBuilder.default_shape_cfg.ke``/``kd`` are
        common and would otherwise produce spurious promotions that
        change contact dynamics for imported MuJoCo assets.
        """
        builder = newton.ModelBuilder()
        builder.add_mjcf(
            """
            <mujoco>
              <compiler angle="radian"/>
              <worldbody>
                <body name="link">
                  <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
                  <geom type="sphere" size="0.05"/>
                </body>
              </worldbody>
            </mujoco>
            """,
            ignore_inertial_definitions=False,
        )
        model = builder.finalize()
        np.testing.assert_array_equal(model.mujoco.solref_mode.numpy(), [SOLREF_MODE_MJCF_DEFAULT])
        np.testing.assert_array_equal(model.mujoco.solref.numpy()[0], [0.0, 0.0])

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
        self.assertEqual(int(model.mujoco.solref_mode.numpy()[0]), SOLREF_MODE_MJCF_DEFAULT)

        # Editing ``shape_material_ke`` alone does not promote the mode.
        ke = model.shape_material_ke.numpy()
        ke[:] = 5000.0
        model.shape_material_ke.assign(ke)
        solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)
        self.assertEqual(int(model.mujoco.solref_mode.numpy()[0]), SOLREF_MODE_MJCF_DEFAULT)

        # Force-space scaling is opt-in via an explicit mode write.
        mode = model.mujoco.solref_mode.numpy()
        mode[:] = SOLREF_MODE_FORCE_SPACE
        model.mujoco.solref_mode.assign(mode)
        solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)
        self.assertEqual(int(model.mujoco.solref_mode.numpy()[0]), SOLREF_MODE_FORCE_SPACE)

    def test_mjcf_authored_solref_preserved_as_raw(self):
        """Authored MJCF ``solref`` stays as ``SOLREF_MODE_RAW`` and forwards unscaled."""
        builder = newton.ModelBuilder()
        builder.add_mjcf(
            """
            <mujoco>
              <compiler angle="radian"/>
              <worldbody>
                <body name="link">
                  <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
                  <geom type="sphere" size="0.05" solref="0.05 0.5"/>
                </body>
              </worldbody>
            </mujoco>
            """,
            ignore_inertial_definitions=False,
        )
        model = builder.finalize()
        # ``mujoco.solref`` is stored as float32; allow a tight tolerance.
        np.testing.assert_allclose(model.mujoco.solref.numpy()[0], [0.05, 0.5], rtol=1e-6, atol=1e-6)
        self.assertEqual(int(model.mujoco.solref_mode.numpy()[0]), SOLREF_MODE_RAW)

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
        np.testing.assert_allclose(
            solver.mjw_model.geom_solref.numpy()[0, 0],
            [0.05, 0.5],
            rtol=1e-6,
            atol=1e-6,
            err_msg="RAW mode shapes must forward the authored mjc:solref into geom_solref unscaled",
        )

    def test_mjcf_authored_solref_does_not_promote_on_ke_edit(self):
        """RAW mode is sticky: editing Newton ke/kd does not flip it to FORCE_SPACE."""
        builder = newton.ModelBuilder()
        builder.add_mjcf(
            """
            <mujoco>
              <compiler angle="radian"/>
              <worldbody>
                <body name="link">
                  <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
                  <geom type="sphere" size="0.05" solref="0.05 0.5"/>
                </body>
              </worldbody>
            </mujoco>
            """,
            ignore_inertial_definitions=False,
        )
        model = builder.finalize()
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
        self.assertEqual(int(model.mujoco.solref_mode.numpy()[0]), SOLREF_MODE_RAW)

        ke = model.shape_material_ke.numpy()
        ke[:] = 5000.0
        model.shape_material_ke.assign(ke)
        solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)
        # Authored MuJoCo data should not silently switch to Newton scaling.
        self.assertEqual(int(model.mujoco.solref_mode.numpy()[0]), SOLREF_MODE_RAW)
