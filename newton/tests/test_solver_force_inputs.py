# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests that solvers treat ``State.body_f`` and ``State.particle_f`` as read-only inputs."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import MUJOCO_MODULES, add_function_test, get_test_devices, requires


class TestSolverForceInputs(unittest.TestCase):
    pass


_BOX_HALF = 0.1


def _add_box_on_ground(builder: newton.ModelBuilder) -> int:
    """Add a free box that slightly penetrates the ground plane."""
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, _BOX_HALF - 0.005), wp.quat_identity()))
    builder.add_shape_box(body, hx=_BOX_HALF, hy=_BOX_HALF, hz=_BOX_HALF)
    return body


def _build_rigid_model(device) -> newton.Model:
    """Build a box resting on the ground and a two-link revolute chain."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    _add_box_on_ground(builder)

    link_half = 0.2
    links = []
    for i in range(2):
        link = builder.add_link(
            xform=wp.transform(wp.vec3(1.0, 0.0, 1.5 - (2 * i + 1) * link_half), wp.quat_identity())
        )
        builder.add_shape_box(link, hx=0.05, hy=0.05, hz=link_half)
        links.append(link)
    joints = [
        builder.add_joint_revolute(
            parent=-1,
            child=links[0],
            parent_xform=wp.transform(wp.vec3(1.0, 0.0, 1.5), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, link_half), wp.quat_identity()),
            axis=newton.Axis.Y,
        ),
        builder.add_joint_revolute(
            parent=links[0],
            child=links[1],
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, -link_half), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, link_half), wp.quat_identity()),
            axis=newton.Axis.Y,
        ),
    ]
    builder.add_articulation(joints)
    builder.color()
    return builder.finalize(device=device)


def _build_cloth_model(device, requires_grad: bool = False, with_box: bool = True) -> newton.Model:
    """Build a spring cloth draped over a free box resting on the ground."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    if with_box:
        _add_box_on_ground(builder)
    builder.add_cloth_grid(
        pos=wp.vec3(-0.075, -0.075, 2.0 * _BOX_HALF + 0.01),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=3,
        dim_y=3,
        cell_x=0.05,
        cell_y=0.05,
        mass=0.01,
        add_springs=True,
        particle_radius=0.02,
    )
    builder.color()
    return builder.finalize(device=device, requires_grad=requires_grad)


def _fill_sentinels(array: wp.array | None, scale: float) -> np.ndarray | None:
    """Fill ``array`` with small distinct values and return them."""
    if array is None:
        return None
    values = np.zeros((array.shape[0], wp.types.type_size(array.dtype)), dtype=np.float32)
    values.flat[:] = scale * (1.0 + np.arange(values.size, dtype=np.float32) % 7.0)
    array.assign(values)
    return values


def _run_force_input_contract(test, device, solver_fn, model, alias_forces: bool):
    """Step a model with sentinel force inputs and assert they are unchanged."""
    solver = solver_fn(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    control = model.control()
    if control.joint_f is not None:
        control.joint_f.fill_(0.1)

    state_in, state_out = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    if alias_forces:
        state_out.body_f = state_in.body_f
        state_out.particle_f = state_in.particle_f

    states = (state_in, state_out)
    expected = {}
    for index, state in enumerate(states):
        for name in ("body_f", "particle_f"):
            array = getattr(state, name)
            if array is None or (alias_forces and index == 1):
                continue
            expected[(index, name)] = _fill_sentinels(array, 1.0e-3 * (index + 1))

    for _ in range(3):
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, 1.0e-3)
        state_in, state_out = state_out, state_in

    if model.body_count:
        test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
    if model.particle_count:
        test.assertGreater(int(contacts.soft_contact_count.numpy()[0]), 0)
    for (index, name), values in expected.items():
        np.testing.assert_array_equal(
            getattr(states[index], name).numpy().reshape(values.shape),
            values,
            err_msg=f"{name} of state {index} was modified by the solver",
        )


def test_rigid_force_inputs_unchanged(test, device, solver_fn):
    """Keep body_f unchanged in both states across steps with contacts and joint forces."""
    for alias_forces in (False, True):
        with test.subTest(alias_forces=alias_forces):
            model = _build_rigid_model(device)
            _run_force_input_contract(test, device, solver_fn, model, alias_forces)


def test_cloth_force_inputs_unchanged(test, device, solver_fn):
    """Keep body_f and particle_f unchanged across steps with elastic and contact forces."""
    for alias_forces in (False, True):
        with test.subTest(alias_forces=alias_forces):
            model = _build_cloth_model(device)
            _run_force_input_contract(test, device, solver_fn, model, alias_forces)


def _simulate_substeps(model, solver, pipeline, contacts, body_f, particle_f, write_once: bool, use_graph: bool):
    """Run substeps with forces written once per step or cleared and reapplied per substep."""
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    if write_once:
        state_1.body_f = state_0.body_f
        state_1.particle_f = state_0.particle_f
        state_0.body_f.assign(body_f)
        state_0.particle_f.assign(particle_f)
    pipeline.collide(state_0, contacts)

    def simulate():
        nonlocal state_0, state_1
        for _ in range(4):
            if not write_once:
                state_0.clear_forces()
                state_0.body_f.assign(body_f)
                state_0.particle_f.assign(particle_f)
            solver.step(state_0, state_1, None, contacts, 1.0e-3)
            state_0, state_1 = state_1, state_0

    if use_graph:
        with wp.ScopedCapture(device=model.device) as capture:
            simulate()
        wp.capture_launch(capture.graph)
    else:
        simulate()
    return state_0


def test_write_once_forces_match_reapplied(test, device, solver_fn):
    """Match clear-and-reapply substepping when shared forces are written once per step."""
    model = _build_cloth_model(device)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    body_f = wp.array([wp.spatial_vector(0.5, 0.0, 0.0, 0.0, 0.0, 0.05)], dtype=wp.spatial_vector, device=device)
    particle_f = wp.full(model.particle_count, wp.vec3(0.0, 0.0, 0.05), dtype=wp.vec3, device=device)

    solver = solver_fn(model)
    reference = _simulate_substeps(model, solver, pipeline, contacts, body_f, particle_f, False, False)
    graph_modes = (False, True) if wp.get_device(device).is_cuda else (False,)
    for use_graph in graph_modes:
        with test.subTest(use_graph=use_graph):
            solver = solver_fn(model)
            result = _simulate_substeps(model, solver, pipeline, contacts, body_f, particle_f, True, use_graph)
            for name in ("body_q", "body_qd", "particle_q", "particle_qd"):
                np.testing.assert_allclose(
                    getattr(result, name).numpy(), getattr(reference, name).numpy(), rtol=1.0e-5, atol=1.0e-6
                )


@wp.kernel
def _rollout_loss_kernel(
    body_q: wp.array[wp.transform],
    particle_q: wp.array[wp.vec3],
    loss: wp.array[float],
):
    tid = wp.tid()
    if tid < body_q.shape[0]:
        rot = wp.transform_get_rotation(body_q[tid])
        value = wp.length_sq(wp.transform_get_translation(body_q[tid]))
        value += 0.1 * rot[0] + 0.2 * rot[1] + 0.3 * rot[2]
        wp.atomic_add(loss, 0, value)
    if tid < particle_q.shape[0]:
        wp.atomic_add(loss, 0, wp.length_sq(particle_q[tid]))


def _rollout_gradients(model, solvers, contacts):
    """Return loss gradients with respect to each step's force inputs and the initial velocities."""
    states = [model.state() for _ in range(len(solvers) + 1)]
    newton.eval_fk(model, model.joint_q, model.joint_qd, states[0])
    if model.body_count:
        qd = states[0].body_qd.numpy()
        qd[0] = (0.2, 0.1, 0.0, 0.0, 0.0, 2.0)
        states[0].body_qd.assign(qd)
    # Distinct inputs per step make each step's force work buffer hold different values.
    for i, state in enumerate(states):
        if state.body_f is not None:
            state.body_f.fill_(wp.spatial_vector(0.1 * (i + 1), 0.0, 0.0, 5.0 * (i + 1), -3.0 * i, 2.0))
        state.particle_f.fill_(wp.vec3(0.01 * i, 0.0, 0.01 * (i + 1)))
    loss = wp.zeros(1, dtype=float, device=model.device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        for i, solver in enumerate(solvers):
            solver.step(states[i], states[i + 1], None, contacts, 5.0e-3)
        wp.launch(
            _rollout_loss_kernel,
            dim=max(model.body_count, model.particle_count),
            inputs=[states[-1].body_q, states[-1].particle_q],
            outputs=[loss],
            device=model.device,
        )
    tape.backward(loss)
    gradients = {}
    for name in ("body_f", "particle_f", "body_qd", "particle_qd"):
        for i, state in enumerate(states[:-1]):
            array = getattr(state, name)
            if array is not None and (i == 0 or name.endswith("_f")):
                gradients[f"{name}[{i}]"] = array.grad.numpy().copy()
    return gradients


def test_force_work_buffers_keep_gradients(test, device, solver_fn, with_box):
    """Differentiate a rollout that reuses one solver as if each step had its own solver.

    Each differentiable step must keep its own force work buffer; a shared buffer would
    replay the last step's forces in the adjoints of earlier steps.
    """
    model = _build_cloth_model(device, requires_grad=True, with_box=with_box)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    pipeline.collide(state, contacts)

    step_count = 4
    shared = _rollout_gradients(model, [solver_fn(model)] * step_count, contacts)
    separate = _rollout_gradients(model, [solver_fn(model) for _ in range(step_count)], contacts)
    test.assertEqual(shared.keys(), separate.keys())
    for name, gradient in separate.items():
        test.assertGreater(np.abs(gradient).max(), 0.0, msg=f"zero gradient for {name}")
        scale = np.abs(gradient).max()
        np.testing.assert_allclose(shared[name], gradient, rtol=1.0e-3, atol=1.0e-4 * scale, err_msg=name)


devices = get_test_devices()

rigid_solvers = {
    "featherstone": newton.solvers.SolverFeatherstone,
    "kamino": newton.solvers.SolverKamino,
    "mujoco_cpu": requires(*MUJOCO_MODULES)(lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True)),
    "mujoco_warp": requires(*MUJOCO_MODULES)(lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False)),
    "semi_implicit": newton.solvers.SolverSemiImplicit,
    "steel": newton.solvers.SolverSteel,
    "vbd": newton.solvers.SolverVBD,
    "xpbd": newton.solvers.SolverXPBD,
}
cloth_solvers = {
    "featherstone": newton.solvers.SolverFeatherstone,
    "semi_implicit": newton.solvers.SolverSemiImplicit,
    "vbd": newton.solvers.SolverVBD,
    "xpbd": newton.solvers.SolverXPBD,
}
# Featherstone keeps mass-matrix buffers on the solver, so its gradient check reuses
# one solver only for a particle model.
gradient_solvers = {
    "featherstone": (newton.solvers.SolverFeatherstone, False),
    "semi_implicit": (newton.solvers.SolverSemiImplicit, True),
}

for device in devices:
    for solver_name, solver_fn in rigid_solvers.items():
        if (device.is_cuda and solver_name == "mujoco_cpu") or (device.is_cpu and solver_name == "mujoco_warp"):
            continue
        add_function_test(
            TestSolverForceInputs,
            f"test_rigid_force_inputs_unchanged_{solver_name}",
            test_rigid_force_inputs_unchanged,
            devices=[device],
            solver_fn=solver_fn,
        )
    for solver_name, solver_fn in cloth_solvers.items():
        add_function_test(
            TestSolverForceInputs,
            f"test_cloth_force_inputs_unchanged_{solver_name}",
            test_cloth_force_inputs_unchanged,
            devices=[device],
            solver_fn=solver_fn,
        )
        add_function_test(
            TestSolverForceInputs,
            f"test_write_once_forces_match_reapplied_{solver_name}",
            test_write_once_forces_match_reapplied,
            devices=[device],
            solver_fn=solver_fn,
        )
    for solver_name, (solver_fn, with_box) in gradient_solvers.items():
        add_function_test(
            TestSolverForceInputs,
            f"test_force_work_buffers_keep_gradients_{solver_name}",
            test_force_work_buffers_keep_gradients,
            devices=[device],
            solver_fn=solver_fn,
            with_box=with_box,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
