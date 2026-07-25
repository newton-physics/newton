# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton.tests.unittest_utils
from newton._src.solvers.semi_implicit import kernels_particle as semi_implicit_particle_kernels
from newton._src.solvers.solver import _set_module_options_if_changed
from newton._src.solvers.vbd import particle_vbd_kernels, vbd_coupling_kernels
from newton._src.solvers.xpbd import kernels as xpbd_kernels
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

DETERMINISTIC_MODE = wp.DeterministicMode.RUN_TO_RUN


def _run_isolated(test, function_name, *args):
    # Deterministic kernels and GPU runtime state persist for the life of the process.
    call_args = ", ".join(repr(arg) for arg in args)
    code = f"from newton.tests.determinism.test_solver_determinism import {function_name}; {function_name}({call_args})"

    env = os.environ.copy()
    env.pop("PYTHONWARNINGS", None)
    warning_args = []
    if newton.tests.unittest_utils.strict_warnings:
        warning_args = ["-W", "error::DeprecationWarning"]
        code = f"import warnings; warnings.filterwarnings('error', module=r'newton(\\.|$)'); {code}"

    result = subprocess.run(
        [sys.executable, *warning_args, "-c", code],
        capture_output=True,
        env=env,
        text=True,
        timeout=600,
        check=False,
    )
    test.assertEqual(
        result.returncode,
        0,
        f"{function_name} subprocess failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
    )


def _snapshot(state, fields):
    return {field: np.array(getattr(state, field).numpy(), copy=True) for field in fields}


def _assert_snapshots_equal(first, second):
    if first.keys() != second.keys():
        raise AssertionError(f"snapshot fields differ: {first.keys()} != {second.keys()}")
    for field in first:
        np.testing.assert_array_equal(first[field], second[field], err_msg=f"non-deterministic {field}")


def _build_soft_body(device):
    builder = newton.ModelBuilder()
    builder.add_soft_grid(
        pos=wp.vec3(-0.2, -0.2, 0.5),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, -0.1),
        dim_x=3,
        dim_y=3,
        dim_z=3,
        cell_x=0.1,
        cell_y=0.1,
        cell_z=0.1,
        density=1.0e3,
        k_mu=5.0e4,
        k_lambda=5.0e4,
        k_damp=1.0e-2,
        fix_left=True,
        particle_radius=0.01,
    )
    builder.color()
    return builder.finalize(device=device)


def _make_particle_solver(model, solver_name):
    if solver_name == "xpbd":
        return newton.solvers.SolverXPBD(model, iterations=5, deterministic=DETERMINISTIC_MODE)
    if solver_name == "semi_implicit":
        return newton.solvers.SolverSemiImplicit(model, deterministic=DETERMINISTIC_MODE)
    if solver_name == "vbd":
        return newton.solvers.SolverVBD(
            model,
            iterations=5,
            particle_enable_self_contact=True,
            particle_enable_tile_solve=False,
            particle_vertex_contact_buffer_size=64,
            particle_edge_contact_buffer_size=64,
            deterministic=DETERMINISTIC_MODE,
        )
    raise ValueError(f"Unsupported particle solver: {solver_name}")


def _run_particle_rollout(device, solver_name):
    model = _build_soft_body(device)
    solver = _make_particle_solver(model, solver_name)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    contacts = newton.CollisionPipeline(model).contacts()

    for _ in range(20):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0

    return _snapshot(state_0, ("particle_q", "particle_qd"))


def _check_particle_determinism(device, solver_name):
    with wp.ScopedDevice(device):
        first = _run_particle_rollout(device, solver_name)
        second = _run_particle_rollout(device, solver_name)
    _assert_snapshots_equal(first, second)


def test_particle_determinism(test, device, solver_name):
    _run_isolated(test, "_check_particle_determinism", str(device), solver_name)


def _build_branching_articulation(device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    root = builder.add_link()
    builder.add_shape_box(root, hx=0.15, hy=0.1, hz=0.1)
    joints = [builder.add_joint_revolute(parent=-1, child=root, axis=newton.Axis.Z)]

    for y in (-0.45, -0.15, 0.15, 0.45):
        child = builder.add_link()
        builder.add_shape_box(child, hx=0.2, hy=0.05, hz=0.05)
        joints.append(
            builder.add_joint_revolute(
                parent=root,
                child=child,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(wp.vec3(0.0, y, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
            )
        )

    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    joint_q = np.linspace(-0.2, 0.2, model.joint_coord_count, dtype=np.float32)
    model.joint_q.assign(joint_q)
    return model


def _check_mujoco_sparse_articulation_construction(device):
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        joints = []
        parent = -1
        for _ in range(70):
            child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32)))
            joints.append(builder.add_joint_revolute(parent=parent, child=child, axis=newton.Axis.Z))
            parent = child
        builder.add_articulation(joints)

        model = builder.finalize(device=device)
        state_in, state_out = model.state(), model.state()
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        solver = newton.solvers.SolverMuJoCo(
            model,
            use_mujoco_cpu=False,
            disable_contacts=True,
            iterations=1,
            ls_iterations=1,
            deterministic=DETERMINISTIC_MODE,
        )

        solver.step(state_in, state_out, model.control(), None, 1.0 / 240.0)
        if not np.isfinite(state_out.joint_q.numpy()).all():
            raise AssertionError("MuJoCo sparse articulation produced non-finite joint coordinates")


def test_mujoco_sparse_articulation_construction(test, device):
    _run_isolated(test, "_check_mujoco_sparse_articulation_construction", str(device))


def _make_articulation_solver(model, solver_name):
    if solver_name == "featherstone":
        return newton.solvers.SolverFeatherstone(model, deterministic=DETERMINISTIC_MODE)
    if solver_name == "mujoco":
        return newton.solvers.SolverMuJoCo(
            model,
            use_mujoco_cpu=False,
            disable_contacts=True,
            iterations=10,
            ls_iterations=5,
            deterministic=DETERMINISTIC_MODE,
        )
    raise ValueError(f"Unsupported articulation solver: {solver_name}")


def _run_articulation_rollout(device, solver_name):
    model = _build_branching_articulation(device)
    solver = _make_articulation_solver(model, solver_name)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    control.joint_f.assign(np.linspace(1.0, 5.0, model.joint_dof_count, dtype=np.float32))
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    for _ in range(20):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0

    return _snapshot(state_0, ("body_q", "body_qd", "joint_q", "joint_qd"))


def _check_articulation_determinism(device, solver_name):
    with wp.ScopedDevice(device):
        first = _run_articulation_rollout(device, solver_name)
        second = _run_articulation_rollout(device, solver_name)
    _assert_snapshots_equal(first, second)


def test_articulation_determinism(test, device, solver_name):
    _run_isolated(test, "_check_articulation_determinism", str(device), solver_name)


class TestSolverDeterminism(unittest.TestCase):
    pass


class TestSolverDeterminismOptions(unittest.TestCase):
    def setUp(self):
        self._modules = (
            xpbd_kernels,
            particle_vbd_kernels,
            semi_implicit_particle_kernels,
            vbd_coupling_kernels,
        )
        self._saved_config = wp.config.deterministic
        self._saved_options = {
            module: {
                "deterministic": wp.get_module_options(module=module)["deterministic"],
                "deterministic_max_records": wp.get_module_options(module=module)["deterministic_max_records"],
            }
            for module in self._modules
        }

    def tearDown(self):
        wp.config.deterministic = self._saved_config
        for module, options in self._saved_options.items():
            wp.set_module_options(options, module=module)

    def test_xpbd_resets_inherited_module_options(self):
        with wp.ScopedDevice("cpu"):
            model = _build_soft_body("cpu")
            newton.solvers.SolverXPBD(model, deterministic=DETERMINISTIC_MODE)
            options = wp.get_module_options(module=xpbd_kernels)
            self.assertEqual(options["deterministic"], DETERMINISTIC_MODE)
            self.assertEqual(options["deterministic_max_records"], 0)

            wp.config.deterministic = wp.DeterministicMode.NOT_GUARANTEED
            newton.solvers.SolverXPBD(model)
            options = wp.get_module_options(module=xpbd_kernels)
            self.assertEqual(options["deterministic"], wp.DeterministicMode.NOT_GUARANTEED)
            self.assertEqual(options["deterministic_max_records"], 0)

    def test_matching_module_options_are_not_reapplied(self):
        options = {
            "deterministic": wp.get_module_options(module=xpbd_kernels)["deterministic"],
            "deterministic_max_records": wp.get_module_options(module=xpbd_kernels)["deterministic_max_records"],
        }
        with mock.patch.object(wp, "set_module_options") as set_module_options:
            _set_module_options_if_changed(options, module=xpbd_kernels)
        set_module_options.assert_not_called()

    def test_mujoco_generated_kernel_cache_tracks_determinism(self):
        from mujoco_warp._src import warp_util

        solver = object.__new__(newton.solvers.SolverMuJoCo)
        solver._deterministic = DETERMINISTIC_MODE
        solver._deterministic_max_records = 17
        cache = warp_util._KERNEL_CACHE
        saved_cache = cache.copy()
        saved_options = newton.solvers.SolverMuJoCo._generated_kernel_deterministic_options
        sentinel = object()

        try:
            cache.clear()
            cache["sentinel"] = sentinel
            newton.solvers.SolverMuJoCo._generated_kernel_deterministic_options = None

            solver._prepare_generated_kernels()
            self.assertEqual(cache, {})

            cache["sentinel"] = sentinel
            solver._prepare_generated_kernels()
            self.assertIs(cache["sentinel"], sentinel)

            solver._deterministic_max_records += 1
            solver._prepare_generated_kernels()
            self.assertEqual(cache, {})
        finally:
            cache.clear()
            cache.update(saved_cache)
            newton.solvers.SolverMuJoCo._generated_kernel_deterministic_options = saved_options

    def test_current_solver_skips_module_option_checks(self):
        model = newton.ModelBuilder().finalize(device="cpu")
        solver = newton.solvers.SolverXPBD(model)
        with mock.patch.object(wp, "get_module_options") as get_module_options:
            solver._apply_module_options()
        get_module_options.assert_not_called()

    def test_live_solvers_reapply_module_options_before_step(self):
        with wp.ScopedDevice("cpu"):
            model = newton.ModelBuilder().finalize(device="cpu")
            deterministic_solver = newton.solvers.SolverXPBD(model, deterministic=DETERMINISTIC_MODE)
            default_solver = newton.solvers.SolverXPBD(
                model,
                deterministic=wp.DeterministicMode.NOT_GUARANTEED,
            )
            state_0, state_1 = model.state(), model.state()

            deterministic_solver.step(state_0, state_1, None, None, 1.0 / 60.0)
            self.assertEqual(wp.get_module_options(module=xpbd_kernels)["deterministic"], DETERMINISTIC_MODE)

            default_solver.step(state_1, state_0, None, None, 1.0 / 60.0)
            self.assertEqual(
                wp.get_module_options(module=xpbd_kernels)["deterministic"],
                wp.DeterministicMode.NOT_GUARANTEED,
            )

    def test_shared_semi_implicit_modules_reset_record_budget(self):
        with wp.ScopedDevice("cpu"):
            model = _build_soft_body("cpu")
            for solver_type in (newton.solvers.SolverSemiImplicit, newton.solvers.SolverFeatherstone):
                with self.subTest(solver_type=solver_type.__name__):
                    wp.set_module_options(
                        {"deterministic_max_records": 7},
                        module=semi_implicit_particle_kernels,
                    )
                    solver_type(model, deterministic=DETERMINISTIC_MODE)
                    options = wp.get_module_options(module=semi_implicit_particle_kernels)
                    self.assertEqual(options["deterministic"], DETERMINISTIC_MODE)
                    self.assertEqual(options["deterministic_max_records"], 0)

    def test_vbd_resets_inherited_module_options(self):
        with wp.ScopedDevice("cpu"):
            model = _build_soft_body("cpu")
            newton.solvers.SolverVBD(
                model,
                particle_enable_self_contact=True,
                particle_enable_tile_solve=False,
                particle_vertex_contact_buffer_size=64,
                particle_edge_contact_buffer_size=64,
                deterministic=DETERMINISTIC_MODE,
            )
            options = wp.get_module_options(module=particle_vbd_kernels)
            self.assertEqual(options["deterministic"], DETERMINISTIC_MODE)
            records_per_buffer = (64 + particle_vbd_kernels.NUM_THREADS_PER_COLLISION_PRIMITIVE - 1) // (
                particle_vbd_kernels.NUM_THREADS_PER_COLLISION_PRIMITIVE
            )
            self.assertEqual(options["deterministic_max_records"], 8 * records_per_buffer)

            wp.config.deterministic = wp.DeterministicMode.NOT_GUARANTEED
            newton.solvers.SolverVBD(
                model,
                particle_enable_self_contact=False,
                particle_enable_tile_solve=False,
            )
            options = wp.get_module_options(module=particle_vbd_kernels)
            self.assertEqual(options["deterministic"], wp.DeterministicMode.NOT_GUARANTEED)
            self.assertEqual(options["deterministic_max_records"], 0)

    def test_vbd_coupling_hook_reapplies_deterministic_options(self):
        with wp.ScopedDevice("cpu"):
            model = _build_soft_body("cpu")
            deterministic_solver = newton.solvers.SolverVBD(
                model,
                particle_enable_self_contact=True,
                particle_enable_tile_solve=False,
                particle_vertex_contact_buffer_size=64,
                particle_edge_contact_buffer_size=64,
                deterministic=DETERMINISTIC_MODE,
            )
            newton.solvers.SolverVBD(
                model,
                particle_enable_self_contact=False,
                particle_enable_tile_solve=False,
                deterministic=wp.DeterministicMode.NOT_GUARANTEED,
            )

            deterministic_solver.coupling_notify_input_state_update(
                model.state(),
                newton.StateFlags.BODY_Q,
            )

            options = wp.get_module_options(module=vbd_coupling_kernels)
            records_per_buffer = (64 + particle_vbd_kernels.NUM_THREADS_PER_COLLISION_PRIMITIVE - 1) // (
                particle_vbd_kernels.NUM_THREADS_PER_COLLISION_PRIMITIVE
            )
            self.assertEqual(options["deterministic"], DETERMINISTIC_MODE)
            self.assertEqual(options["deterministic_max_records"], 5 * records_per_buffer)


devices = get_cuda_test_devices(mode="basic")
for solver_name in ("xpbd", "semi_implicit", "vbd"):
    add_function_test(
        TestSolverDeterminism,
        f"test_particle_determinism_{solver_name}",
        test_particle_determinism,
        devices=devices,
        solver_name=solver_name,
        check_output=False,
    )

for solver_name in ("featherstone", "mujoco"):
    add_function_test(
        TestSolverDeterminism,
        f"test_articulation_determinism_{solver_name}",
        test_articulation_determinism,
        devices=devices,
        solver_name=solver_name,
        check_output=False,
    )

add_function_test(
    TestSolverDeterminism,
    "test_mujoco_sparse_articulation_construction",
    test_mujoco_sparse_articulation_construction,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
