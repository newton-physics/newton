# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import builtins
import os
import re
import subprocess
import sys
import unittest
import warnings
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton.tests.unittest_utils
from newton._src.solvers.semi_implicit import kernels_particle as semi_implicit_particle_kernels
from newton._src.solvers.solver import _set_module_options_if_changed
from newton._src.solvers.vbd import particle_vbd_kernels, vbd_coupling_kernels
from newton._src.solvers.xpbd import kernels as xpbd_kernels
from newton._src.solvers.xpbd import tendon_kernels as xpbd_tendon_kernels
from newton.tests.unittest_utils import (
    add_function_test,
    get_cuda_test_devices,
    warp_version_at_least,
)

DETERMINISTIC_MODE = wp.DeterministicMode.RUN_TO_RUN

# Warp 1.19 rejects MuJoCo Warp kernels that read and atomically update one array
# (_accumulate_subtreemass, _qLD_acc) in deterministic mode.
_MUJOCO_WARP_DETERMINISM_UNSUPPORTED = warp_version_at_least("1.19.0.dev0")


def _skip_unsupported_mujoco_determinism(test):
    if _MUJOCO_WARP_DETERMINISM_UNSUPPORTED:
        test.skipTest("MuJoCo Warp does not compile in Warp 1.19 deterministic mode")


def _run_isolated(
    test,
    function_name,
    *args,
    module_name="newton.tests.determinism.test_solver_determinism",
    deterministic_mode=None,
):
    # Deterministic kernels and GPU runtime state persist for the life of the process.
    call_args = ", ".join(repr(arg) for arg in args)
    code = f"from {module_name} import {function_name}; {function_name}({call_args})"
    if deterministic_mode is not None:
        # Dependency guards import backends while registering tests.
        # Configure Warp before importing the test module in the fresh child.
        code = (
            "import sys; import warp as wp; assert 'mujoco_warp' not in sys.modules; "
            f"wp.config.deterministic = wp.DeterministicMode.{deterministic_mode.name}; " + code
        )

    env = os.environ.copy()
    env.pop("PYTHONWARNINGS", None)
    warning_args = newton.tests.unittest_utils.get_strict_warning_args()
    if newton.tests.unittest_utils.strict_warnings:
        policy = "import warnings; warnings.filterwarnings('error', module=r'newton(\\.|$)'); "
        for message in newton.tests.unittest_utils.allowed_deprecation_warnings:
            policy += (
                f"warnings.filterwarnings('default', message={re.escape(message)!r}, category=DeprecationWarning); "
            )
        code = policy + code

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
    sys.stderr.write(result.stderr)


def _snapshot(state, fields):
    return {field: np.array(getattr(state, field).numpy(), copy=True) for field in fields}


def _assert_snapshots_equal(first, second):
    """Compare snapshots bitwise, so signed zeros and NaN payloads must match too."""
    if first.keys() != second.keys():
        raise AssertionError(f"snapshot fields differ: {first.keys()} != {second.keys()}")
    for field in first:
        a, b = first[field], second[field]
        if a.dtype != b.dtype or a.shape != b.shape:
            raise AssertionError(f"{field} layout differs: {a.dtype}{a.shape} != {b.dtype}{b.shape}")
        if a.tobytes() != b.tobytes():
            np.testing.assert_array_equal(a, b, err_msg=f"non-deterministic {field}")
            raise AssertionError(f"non-deterministic {field}: values match but bit patterns differ")


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
    _skip_unsupported_mujoco_determinism(test)
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
    if solver_name == "mujoco":
        _skip_unsupported_mujoco_determinism(test)
    _run_isolated(test, "_check_articulation_determinism", str(device), solver_name)


def _build_rigid_pile(device, requires_grad=False):
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    rng = np.random.default_rng(42)
    half = 0.1
    for k in range(3):
        for j in range(3):
            for i in range(3):
                body = builder.add_body(
                    xform=wp.transform(
                        wp.vec3(i * 0.19 + 0.02 * k, j * 0.19, half + k * 0.21),
                        wp.quat_rpy(*rng.uniform(-0.3, 0.3, size=3)),
                    )
                )
                if (i + j + k) % 2:
                    builder.add_shape_box(body, hx=half, hy=half, hz=half)
                else:
                    builder.add_shape_sphere(body, radius=half)
    return builder.finalize(device=device, requires_grad=requires_grad)


def _make_rigid_solver(model, solver_name):
    if solver_name == "xpbd":
        return newton.solvers.SolverXPBD(model, iterations=4, deterministic=DETERMINISTIC_MODE)
    if solver_name == "semi_implicit":
        return newton.solvers.SolverSemiImplicit(model, deterministic=DETERMINISTIC_MODE)
    if solver_name == "featherstone":
        return newton.solvers.SolverFeatherstone(model, deterministic=DETERMINISTIC_MODE)
    raise ValueError(f"Unsupported rigid solver: {solver_name}")


def _run_rigid_pile_rollout(device, solver_name, use_graph):
    model = _build_rigid_pile(device)
    solver = _make_rigid_solver(model, solver_name)
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    observables = None
    if solver_name == "xpbd":
        # Cover XPBD's per-contact impulse accumulation through its contact-force output.
        observables = solver.observables({newton.solvers.SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    dt = 1.0 / 600.0

    def step_pair():
        nonlocal state_0, state_1
        for _ in range(2):
            state_0.clear_forces()
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt, observables=observables)
            state_0, state_1 = state_1, state_0

    # Step eagerly first so Warp checks its deterministic record buffers for overflow.
    step_pair()
    if use_graph:
        with wp.ScopedCapture() as capture:
            step_pair()
        for _ in range(39):
            wp.capture_launch(capture.graph)
    else:
        for _ in range(40):
            step_pair()

    if contacts.rigid_contact_count.numpy()[0] == 0:
        raise AssertionError("rigid pile produced no contacts")
    snapshot = _snapshot(state_0, ("body_q", "body_qd"))
    if observables is not None:
        contact_f = np.array(observables.contact_f.numpy(), copy=True)
        if not np.any(contact_f):
            raise AssertionError("rigid pile produced no XPBD contact forces")
        snapshot["contact_f"] = contact_f
    return snapshot


def _check_rigid_pile_determinism(device, solver_name, use_graph):
    with wp.ScopedDevice(device):
        first = _run_rigid_pile_rollout(device, solver_name, use_graph)
        second = _run_rigid_pile_rollout(device, solver_name, use_graph)
    _assert_snapshots_equal(first, second)


def test_rigid_pile_determinism(test, device, solver_name, use_graph):
    if use_graph and not wp.get_device(device).is_cuda:
        test.skipTest("CUDA graph capture requires a CUDA device")
    _run_isolated(test, "_check_rigid_pile_determinism", str(device), solver_name, use_graph)


class TestSolverDeterminism(unittest.TestCase):
    pass


def _emit_warning_for_policy_test(category_name, message):
    warnings.warn_explicit(message, getattr(builtins, category_name), __file__, 1, module=__name__)


@wp.kernel
def _copy_spatial_vectors(src: wp.array[wp.spatial_vector], dst: wp.array[wp.spatial_vector]):
    tid = wp.tid()
    dst[tid] = src[tid]


def _check_contact_gather_backward(device, gather_name):
    """Differentiate a per-body contact gather for a body with more sides than Warp's record floor.

    Gradients must reach both the contact-side values and the body values accumulated before the gather.
    ``gather_name`` selects the SemiImplicit/Featherstone wrench gather or the XPBD delta gather.
    """
    from newton._src.solvers.contact_gather import gather_contact_side_wrenches  # noqa: PLC0415

    side_count = 2000
    with wp.ScopedDevice(device):
        # Body 0 owns sides [0, side_count), body 1 owns the last side.
        body_start = wp.array([0, side_count], dtype=wp.int32)
        body_end = wp.array([side_count, side_count + 1], dtype=wp.int32)
        sides = wp.array(np.arange(side_count + 1, dtype=np.int32), dtype=wp.int32)
        side_wrench = wp.ones(side_count + 1, dtype=wp.spatial_vector, requires_grad=True)
        # The gather adds to a body wrench that earlier kernels already wrote.
        prior_wrench = wp.full(2, value=wp.spatial_vector(0.5, 0.5, 0.5, 0.5, 0.5, 0.5), requires_grad=True)
        body_wrench = wp.zeros(2, dtype=wp.spatial_vector, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(_copy_spatial_vectors, dim=2, inputs=[prior_wrench], outputs=[body_wrench])
            if gather_name == "wrench":
                wp.launch(
                    gather_contact_side_wrenches,
                    dim=2,
                    inputs=[body_start, body_end, sides, side_wrench],
                    outputs=[body_wrench],
                )
            else:
                contact_active = wp.ones((side_count + 2) // 2, dtype=wp.int32)
                contact_inv_weight = wp.zeros(2, dtype=float)
                wp.launch(
                    xpbd_kernels.gather_body_contact_deltas,
                    dim=2,
                    inputs=[body_start, body_end, sides, side_wrench, contact_active],
                    outputs=[body_wrench, contact_inv_weight],
                )
        np.testing.assert_array_equal(body_wrench.numpy()[:, 0], [side_count + 0.5, 1.5])
        seed_np = np.array([[1.0] * 6, [2.0] * 6], dtype=np.float32)
        seed = wp.array(seed_np, dtype=wp.spatial_vector)
        expected_side = np.ones((side_count + 1, 6), dtype=np.float32)
        expected_side[side_count] = 2.0

        def check_gradients(mode):
            np.testing.assert_array_equal(side_wrench.grad.numpy(), expected_side, err_msg=f"{mode} side gradient")
            np.testing.assert_array_equal(prior_wrench.grad.numpy(), seed_np, err_msg=f"{mode} prior gradient")

        tape.backward(grads={body_wrench: seed})
        check_gradients("eager")

        tape.zero()
        with wp.ScopedCapture() as capture:
            tape.backward(grads={body_wrench: seed})
        tape.zero()
        wp.capture_launch(capture.graph)
        check_gradients("captured")


def test_contact_gather_backward(test, device, gather_name):
    _run_isolated(
        test, "_check_contact_gather_backward", str(device), gather_name, deterministic_mode=DETERMINISTIC_MODE
    )


@wp.kernel
def _body_height_loss(body_q: wp.array[wp.transform], loss: wp.array[float]):
    tid = wp.tid()
    wp.atomic_add(loss, 0, wp.transform_get_translation(body_q[tid])[2])


def _check_rigid_contact_backward(device, solver_name):
    """Differentiate a deterministic step through the rigid-contact gather."""
    with wp.ScopedDevice(device):
        model = _build_rigid_pile(device, requires_grad=True)
        solver = _make_rigid_solver(model, solver_name)
        pipeline = newton.CollisionPipeline(model, deterministic=True)
        contacts = pipeline.contacts()
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        pipeline.collide(state_0, contacts)
        if contacts.rigid_contact_count.numpy()[0] == 0:
            raise AssertionError("rigid pile produced no contacts")
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        tape = wp.Tape()
        with tape:
            solver.step(state_0, state_1, model.control(), contacts, 1.0 / 600.0)
            wp.launch(_body_height_loss, dim=model.body_count, inputs=[state_1.body_q], outputs=[loss])
        tape.backward(loss)
        grad = state_0.body_qd.grad.numpy()
        if not np.all(np.isfinite(grad)) or not np.any(grad):
            raise AssertionError(f"invalid body velocity gradient: {grad}")


def test_rigid_contact_backward(test, device, solver_name):
    _run_isolated(test, "_check_rigid_contact_backward", str(device), solver_name)


class TestSnapshotComparison(unittest.TestCase):
    def test_snapshots_compare_bitwise(self):
        """Values that compare equal but differ in bits, such as signed zeros, count as divergence."""
        snapshot = {"x": np.array([0.0, 1.0], dtype=np.float32)}
        _assert_snapshots_equal(snapshot, {"x": snapshot["x"].copy()})
        with self.assertRaisesRegex(AssertionError, "bit patterns differ"):
            _assert_snapshots_equal(snapshot, {"x": np.array([-0.0, 1.0], dtype=np.float32)})


class TestSolverDeterminismWarnings(unittest.TestCase):
    def test_allowlisted_deprecations_override_newton_error_filter(self):
        """Allow only acknowledged deprecations ahead of the Newton error filter."""
        allowed_prefix = "dependency.old_api is deprecated"
        with (
            mock.patch.object(newton.tests.unittest_utils, "strict_warnings", True),
            mock.patch.object(newton.tests.unittest_utils, "allowed_deprecation_warnings", (allowed_prefix,)),
        ):
            _run_isolated(self, "_emit_warning_for_policy_test", "DeprecationWarning", f"{allowed_prefix}; use new_api")
            for category, message in (
                ("DeprecationWarning", "unexpected deprecation"),
                ("UserWarning", allowed_prefix),
            ):
                with self.subTest(category=category), self.assertRaisesRegex(AssertionError, category):
                    _run_isolated(self, "_emit_warning_for_policy_test", category, message)


class TestSolverDeterminismOptions(unittest.TestCase):
    def setUp(self):
        self._modules = (
            xpbd_kernels,
            xpbd_tendon_kernels,
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
            tendon_options = wp.get_module_options(module=xpbd_tendon_kernels)
            self.assertEqual(tendon_options["deterministic"], DETERMINISTIC_MODE)
            self.assertEqual(tendon_options["deterministic_max_records"], 0)

            wp.config.deterministic = wp.DeterministicMode.NOT_GUARANTEED
            newton.solvers.SolverXPBD(model)
            options = wp.get_module_options(module=xpbd_kernels)
            self.assertEqual(options["deterministic"], wp.DeterministicMode.NOT_GUARANTEED)
            self.assertEqual(options["deterministic_max_records"], 0)
            tendon_options = wp.get_module_options(module=xpbd_tendon_kernels)
            self.assertEqual(tendon_options["deterministic"], wp.DeterministicMode.NOT_GUARANTEED)
            self.assertEqual(tendon_options["deterministic_max_records"], 0)

    def test_matching_module_options_are_not_reapplied(self):
        options = {
            "deterministic": wp.get_module_options(module=xpbd_kernels)["deterministic"],
            "deterministic_max_records": wp.get_module_options(module=xpbd_kernels)["deterministic_max_records"],
        }
        with mock.patch.object(wp, "set_module_options") as set_module_options:
            _set_module_options_if_changed(options, module=xpbd_kernels)
        set_module_options.assert_not_called()

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
            self.assertEqual(
                wp.get_module_options(module=xpbd_tendon_kernels)["deterministic"],
                DETERMINISTIC_MODE,
            )

            default_solver.step(state_1, state_0, None, None, 1.0 / 60.0)
            self.assertEqual(
                wp.get_module_options(module=xpbd_kernels)["deterministic"],
                wp.DeterministicMode.NOT_GUARANTEED,
            )
            self.assertEqual(
                wp.get_module_options(module=xpbd_tendon_kernels)["deterministic"],
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

for solver_name in ("xpbd", "semi_implicit", "featherstone"):
    for use_graph in (False, True):
        add_function_test(
            TestSolverDeterminism,
            f"test_rigid_pile_determinism_{solver_name}{'_graph' if use_graph else ''}",
            test_rigid_pile_determinism,
            devices=devices,
            solver_name=solver_name,
            use_graph=use_graph,
            check_output=False,
        )

for gather_name in ("wrench", "xpbd_delta"):
    add_function_test(
        TestSolverDeterminism,
        f"test_contact_gather_backward_{gather_name}",
        test_contact_gather_backward,
        devices=devices,
        gather_name=gather_name,
        check_output=False,
    )

for solver_name in ("xpbd", "semi_implicit"):
    add_function_test(
        TestSolverDeterminism,
        f"test_rigid_contact_backward_{solver_name}",
        test_rigid_contact_backward,
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
