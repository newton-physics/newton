# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise native MuJoCo Warp determinism before construction-time constants."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mujoco import kernels, solver_mujoco
from newton.solvers import SolverMuJoCo
from newton.tests.determinism.test_solver_determinism import _run_isolated
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build_model(device):
    """Build two identical articulated contact scenes at the same local origin."""
    template = newton.ModelBuilder()
    root = template.add_link(xform=wp.transform((0.0, 0.0, 0.09), wp.quat_identity()))
    template.add_shape_box(root, hx=0.12, hy=0.08, hz=0.1)
    joints = [template.add_joint_free(child=root)]
    for y in (-0.2, 0.2):
        child = template.add_link()
        template.add_shape_box(child, hx=0.05, hy=0.06, hz=0.04)
        joints.append(
            template.add_joint_revolute(
                parent=root,
                child=child,
                axis=newton.Axis.Y,
                parent_xform=wp.transform((0.0, y, 0.1), wp.quat_identity()),
            )
        )
    template.add_articulation(joints)
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for _ in range(2):
        builder.add_world(template)
    return builder.finalize(device=device)


def test_native_configuration(test, device):
    """Enable the backend option before constants and keep solver options independent."""
    import mujoco_warp as mjw
    from mujoco_warp._src import smooth

    if not hasattr(mjw, "DeterminismType"):
        test.skipTest("MuJoCo Warp native determinism API is not available")
    for module in (smooth, kernels, solver_mujoco.convert_newton_contacts_to_mjwarp_kernel.module):
        test.addCleanup(wp.set_module_options, wp.get_module_options(module).copy(), module=module)
    # A backend-owned record bound must survive Newton construction and updates.
    wp.set_module_options({"deterministic_max_records": 17}, module=smooth)
    backend_options = wp.get_module_options(smooth).copy()
    observed = []
    original_crb = smooth.crb

    def trace_crb(model, data):
        """Observe the actual setting without replacing the mass-matrix calculation."""
        observed.append(model.opt.deterministic)
        return original_crb(model, data)

    global_mode = wp.config.deterministic
    global_records = wp.config.deterministic_max_records
    with wp.ScopedDevice(device):
        model = _build_model(device)
        with mock.patch.object(smooth, "crb", side_effect=trace_crb):
            solver = SolverMuJoCo(
                model,
                use_mujoco_contacts=True,
                deterministic=wp.DeterministicMode.RUN_TO_RUN,
                integrator="euler",
                jacobian="sparse",
                iterations=20,
                ls_iterations=50,
            )
        test.assertTrue(observed, "construction must exercise constant computation")
        test.assertTrue(all(value == int(mjw.DeterminismType.ALL) for value in observed), observed)
        test.assertEqual(solver.mjw_model.opt.deterministic, int(mjw.DeterminismType.ALL))
        test.assertEqual(wp.get_module_options(smooth), backend_options)
        for module in (kernels, solver_mujoco.convert_newton_contacts_to_mjwarp_kernel.module):
            test.assertEqual(wp.get_module_options(module)["deterministic"], wp.DeterministicMode.RUN_TO_RUN)
        for field in ("body_invweight0", "dof_invweight0"):
            values = getattr(solver.mjw_model, field).numpy()
            test.assertTrue(np.isfinite(values).all(), field)
            np.testing.assert_array_equal(values[0], values[1], err_msg=field)
        other = SolverMuJoCo(
            model,
            use_mujoco_contacts=True,
            deterministic=wp.DeterministicMode.NOT_GUARANTEED,
            integrator="euler",
            jacobian="sparse",
            iterations=20,
            ls_iterations=50,
        )
        test.assertEqual(other.mjw_model.opt.deterministic, int(mjw.DeterminismType.NONE))
        test.assertEqual(solver.mjw_model.opt.deterministic, int(mjw.DeterminismType.ALL))
        with test.assertRaisesRegex(ValueError, "GPU_TO_GPU.*from deterministic"):
            SolverMuJoCo(model, deterministic=wp.DeterministicMode.GPU_TO_GPU)
        try:
            wp.config.deterministic = wp.DeterministicMode.GPU_TO_GPU
            with test.assertRaisesRegex(ValueError, "GPU_TO_GPU.*from wp.config.deterministic"):
                SolverMuJoCo(model)
        finally:
            wp.config.deterministic = global_mode
        # Alternate solvers to check shared modules are reconfigured on entry.
        solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)
        for module in (kernels, solver_mujoco.convert_newton_contacts_to_mjwarp_kernel.module):
            test.assertEqual(wp.get_module_options(module)["deterministic"], wp.DeterministicMode.RUN_TO_RUN)
        test.assertEqual(solver.mjw_model.opt.deterministic, int(mjw.DeterminismType.ALL))
        test.assertEqual(wp.get_module_options(smooth), backend_options)
    test.assertEqual(wp.config.deterministic, global_mode)
    test.assertEqual(wp.config.deterministic_max_records, global_records)


def _check_native_rollout(device_name, use_mujoco_contacts):
    """Exercise one small contact workload in an isolated process."""

    test = unittest.TestCase()
    device = wp.get_device(device_name)

    def run(captured):
        """Construct fresh constants and execute ten physical steps."""
        with wp.ScopedDevice(device):
            model = _build_model(device)
            solver = SolverMuJoCo(
                model,
                use_mujoco_contacts=use_mujoco_contacts,
                deterministic=wp.DeterministicMode.RUN_TO_RUN,
                integrator="euler",
                jacobian="sparse",
                iterations=20,
                ls_iterations=50,
            )
            state0, state1 = model.state(), model.state()
            control = model.control()
            newton.eval_fk(model, state0.joint_q, state0.joint_qd, state0)

            pipeline = None if use_mujoco_contacts else newton.CollisionPipeline(model, deterministic=True)
            contacts = None if pipeline is None else pipeline.contacts()

            def step_pair():
                """Keep graph bindings fixed while advancing both state buffers."""
                if pipeline is not None:
                    pipeline.collide(state0, contacts)
                solver.step(state0, state1, control, contacts, 0.005)
                # Reuse this contact generation to exercise the cached conversion path.
                solver.step(state1, state0, control, contacts, 0.005)

            step_pair()
            graph = None
            if captured:
                with wp.ScopedCapture() as capture:
                    step_pair()
                graph = capture.graph
            snapshots = []
            for _ in range(4):
                if graph is None:
                    step_pair()
                else:
                    wp.capture_launch(graph)
                test.assertFalse(solver.mjw_data.overflow.numpy().any())
                values = {name: getattr(solver.mjw_data, name).numpy().copy() for name in ("qpos", "qvel", "qacc")}
                for name, value in values.items():
                    test.assertTrue(np.isfinite(value).all(), name)
                    np.testing.assert_array_equal(value[0], value[1], err_msg=name)
                snapshots.append(values)
            test.assertGreater(solver.mjw_data.nacon.numpy()[0], 0)
            return snapshots

    baseline = run(False)
    for captured in (False, True) if device.is_cuda else (False,):
        for first, second in zip(baseline, run(captured), strict=True):
            for name in first:
                np.testing.assert_array_equal(first[name], second[name], err_msg=name)


def test_native_rollout(test, device, use_mujoco_contacts=True):
    """Isolate deterministic compilation and runtime state from other test cases."""
    import mujoco_warp as mjw

    if not hasattr(mjw, "DeterminismType"):
        test.skipTest("MuJoCo Warp native determinism API is not available")
    _run_isolated(
        test,
        "_check_native_rollout",
        str(device),
        use_mujoco_contacts,
        module_name="newton.tests.mujoco.test_native_determinism",
    )


class TestMuJoCoNativeDeterminism(unittest.TestCase):
    """Check per-model determinism configuration on supported backends."""

    def test_scoped_configuration_restoration(self):
        """Restore ambient configuration after nested native and legacy failures."""
        native = object.__new__(SolverMuJoCo)
        native._native_mujoco_determinism = True
        legacy = object.__new__(SolverMuJoCo)
        legacy._native_mujoco_determinism = False
        legacy._deterministic = wp.DeterministicMode.RUN_TO_RUN
        legacy._deterministic_max_records = 123
        ambient = (wp.config.deterministic, wp.config.deterministic_max_records)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with legacy._scoped_deterministic_config():
                self.assertEqual(wp.config.deterministic, wp.DeterministicMode.RUN_TO_RUN)
                self.assertEqual(wp.config.deterministic_max_records, 123)
                with self.assertRaisesRegex(RuntimeError, "native"):
                    with native._scoped_deterministic_config():
                        self.assertEqual(wp.config.deterministic, wp.DeterministicMode.NOT_GUARANTEED)
                        self.assertEqual(wp.config.deterministic_max_records, 0)
                        raise RuntimeError("native")
                self.assertEqual(wp.config.deterministic, wp.DeterministicMode.RUN_TO_RUN)
                self.assertEqual(wp.config.deterministic_max_records, 123)
                raise RuntimeError("injected")
        self.assertEqual((wp.config.deterministic, wp.config.deterministic_max_records), ambient)


add_function_test(
    TestMuJoCoNativeDeterminism, "test_native_configuration", test_native_configuration, devices=get_test_devices()
)


add_function_test(TestMuJoCoNativeDeterminism, "test_native_rollout", test_native_rollout, devices=get_test_devices())


add_function_test(
    TestMuJoCoNativeDeterminism,
    "test_native_newton_contacts_rollout",
    test_native_rollout,
    devices=get_test_devices(),
    use_mujoco_contacts=False,
)


if __name__ == "__main__":
    unittest.main()
