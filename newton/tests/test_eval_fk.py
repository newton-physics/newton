# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.articulation import eval_articulation_fk
from newton._src.sim.articulation_cuda import (
    FK_TILE_MAX_LEVEL_WIDTH,
    TILE_BLOCK_DIM,
    create_eval_articulation_fk_tile,
)
from newton.tests.unittest_utils import (
    add_function_test,
    assert_np_equal,
    get_selected_cuda_test_devices,
    get_test_devices,
)


class TestEvalFK(unittest.TestCase):
    pass


@wp.kernel
def _body_loss(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body: int,
    loss: wp.array[float],
):
    position = wp.transform_get_translation(body_q[body])
    velocity = body_qd[body]
    loss[0] = wp.dot(position, position) + wp.dot(velocity, velocity)


def _add_chain(builder: newton.ModelBuilder, length: int, x_offset: float) -> None:
    joints = []
    parent = -1
    for i in range(length):
        child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        joint = builder.add_joint_revolute(
            parent=parent,
            child=child,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(wp.vec3(x_offset, 0.1 * i, 0.0), wp.quat_identity()),
        )
        joints.append(joint)
        parent = child
    builder.add_articulation(joints)


def _build_heterogeneous_model(device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    _add_chain(builder, 3, -1.0)

    root = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joints = [builder.add_joint_revolute(parent=-1, child=root, axis=newton.Axis.Z)]
    for i in range(40):
        child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        joints.append(
            builder.add_joint_revolute(
                parent=root,
                child=child,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(wp.vec3(0.01 * i, 0.0, 0.0), wp.quat_identity()),
            )
        )
    builder.add_articulation(joints)

    _add_chain(builder, 7, 1.0)
    model = builder.finalize(device=device)
    model.joint_q.assign(np.linspace(-0.4, 0.6, model.joint_coord_count, dtype=np.float32))
    model.joint_qd.assign(np.linspace(0.7, -0.3, model.joint_dof_count, dtype=np.float32))
    return model


def _build_gradient_model(device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    joints = []
    parent = -1
    for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z):
        child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        joints.append(
            builder.add_joint_revolute(
                parent=parent,
                child=child,
                axis=axis,
                parent_xform=wp.transform(wp.vec3(0.2, 0.1, 0.3), wp.quat_identity()),
            )
        )
        parent = child
    builder.add_articulation(joints)
    return builder.finalize(device=device, requires_grad=True), parent


def _build_loop_model(device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    root = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    root_joint = builder.add_joint_revolute(parent=-1, child=root, axis=newton.Axis.Z)
    child_joint = builder.add_joint_revolute(parent=root, child=child, axis=newton.Axis.Y)
    builder.add_articulation([root_joint, child_joint])
    builder.add_joint_fixed(parent=child, child=root)
    model = builder.finalize(device=device)
    model.joint_q.assign(np.array([0.3, -0.7], dtype=np.float32))
    return model


def _eval_fk_serial(model, state, mask=None, indices=None, body_flag_filter=newton.BodyFlags.ALL):
    count = len(indices) if indices is not None else model.articulation_count
    wp.launch(
        eval_articulation_fk,
        dim=count,
        inputs=[
            model.articulation_start,
            model.articulation_end,
            model.articulation_count,
            mask,
            indices,
            model.joint_articulation,
            state.joint_q,
            state.joint_qd,
            model.joint_q_start,
            model.joint_qd_start,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_axis,
            model.joint_dof_dim,
            model.body_com,
            model.body_flags,
            int(body_flag_filter),
        ],
        outputs=[state.body_q, state.body_qd],
        device=model.device,
    )


def _eval_fk_parallel(model, state, mask=None, indices=None, body_flag_filter=newton.BodyFlags.ALL):
    count = len(indices) if indices is not None else model.articulation_count
    block_dim = TILE_BLOCK_DIM if model.device.is_cuda else 1
    kernel = create_eval_articulation_fk_tile(
        model._fk_level_capacity,
        body_flag_filter == newton.BodyFlags.ALL,
        model._has_cable_joints,
    )
    wp.launch_tiled(
        kernel,
        dim=[count],
        block_dim=block_dim,
        inputs=[
            model._fk_articulation_level_start,
            model._fk_level_joint_start,
            model._fk_level_joints,
            model._fk_level_parent_pos,
            model.articulation_count,
            mask,
            indices,
            state.joint_q,
            state.joint_qd,
            model.joint_q_start,
            model.joint_qd_start,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_axis,
            model.joint_dof_dim,
            model.body_com,
            model.body_flags,
            int(body_flag_filter),
        ],
        outputs=[state.body_q, state.body_qd],
        device=model.device,
    )


def test_heterogeneous_wide_articulations(test, device):
    model = _build_heterogeneous_model(device)
    joint_child = model.joint_child.numpy()
    child_to_joint = {child: joint for joint, child in enumerate(joint_child)}
    expected_ancestor = np.array(
        [child_to_joint.get(parent, -1) for parent in model.joint_parent.numpy()], dtype=np.int32
    )
    assert_np_equal(model.joint_ancestor.numpy(), expected_ancestor)

    articulation_level_start = model._fk_articulation_level_start.numpy()
    level_joint_start = model._fk_level_joint_start.numpy()
    level_parent_pos = model._fk_level_parent_pos.numpy()
    test.assertEqual(level_parent_pos[0], -1)
    test.assertEqual(level_parent_pos[1], 0)
    wide_articulation_levels = range(articulation_level_start[1], articulation_level_start[2])
    test.assertEqual(
        max(level_joint_start[level + 1] - level_joint_start[level] for level in wide_articulation_levels), 40
    )
    variants = [
        (None, None),
        (wp.array([True, False, True], dtype=bool, device=device), None),
        (None, wp.array([1, 2], dtype=int, device=device)),
    ]
    for mask, indices in variants:
        state_reference = model.state()
        state_parallel = model.state()
        _eval_fk_serial(model, state_reference, mask=mask, indices=indices)
        _eval_fk_parallel(model, state_parallel, mask=mask, indices=indices)
        assert_np_equal(state_parallel.body_q.numpy(), state_reference.body_q.numpy(), tol=1.0e-6)
        assert_np_equal(state_parallel.body_qd.numpy(), state_reference.body_qd.numpy(), tol=1.0e-6)


def test_deep_articulation(test, device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    joints = []
    parent = -1
    for _ in range(41):
        child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        joints.append(
            builder.add_joint_revolute(
                parent=parent,
                child=child,
                axis=newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(0.02, 0.01, 0.0), wp.quat_identity()),
            )
        )
        parent = child
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    model.joint_q.assign(np.linspace(-0.4, 0.6, model.joint_coord_count, dtype=np.float32))
    model.joint_qd.assign(np.linspace(0.7, -0.3, model.joint_dof_count, dtype=np.float32))

    articulation_level_start = model._fk_articulation_level_start.numpy()
    test.assertEqual(articulation_level_start[1] - articulation_level_start[0], 41)
    test.assertEqual(model._fk_level_capacity, 1)
    test.assertEqual(model._fk_level_parent_pos.numpy()[-1], 0)

    state_reference = model.state()
    state_parallel = model.state()
    _eval_fk_serial(model, state_reference)
    _eval_fk_parallel(model, state_parallel)
    assert_np_equal(state_parallel.body_q.numpy(), state_reference.body_q.numpy(), tol=2.0e-5)
    assert_np_equal(state_parallel.body_qd.numpy(), state_reference.body_qd.numpy(), tol=2.0e-5)


def test_empty_articulation_selection(test, device):
    model = _build_heterogeneous_model(device)
    state = model.state()
    body_q = state.body_q.numpy()
    body_qd = state.body_qd.numpy()

    newton.eval_fk(
        model,
        state.joint_q,
        state.joint_qd,
        state,
        indices=wp.empty(0, dtype=int, device=device),
    )

    assert_np_equal(state.body_q.numpy(), body_q)
    assert_np_equal(state.body_qd.numpy(), body_qd)


def test_noncontiguous_bodies_and_external_parent(test, device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    external_parent = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    root = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    root_joint = builder.add_joint_revolute(parent=external_parent, child=root, axis=newton.Axis.Z)
    child_joint = builder.add_joint_revolute(parent=root, child=child, axis=newton.Axis.Y)
    builder.add_articulation([root_joint, child_joint])
    model = builder.finalize(device=device)
    model.joint_q.assign(np.array([0.4, -0.2], dtype=np.float32))
    model.joint_qd.assign(np.array([0.7, -0.3], dtype=np.float32))

    state_reference = model.state()
    state_parallel = model.state()
    _eval_fk_serial(model, state_reference)
    _eval_fk_parallel(model, state_parallel)
    assert_np_equal(state_parallel.body_q.numpy(), state_reference.body_q.numpy(), tol=1.0e-6)
    assert_np_equal(state_parallel.body_qd.numpy(), state_reference.body_qd.numpy(), tol=1.0e-6)


def test_body_flag_filter(test, device):
    model = _build_heterogeneous_model(device)
    body_flags = model.body_flags.numpy()
    body_flags[model.joint_child.numpy()[1]] = int(newton.BodyFlags.KINEMATIC)
    model.body_flags.assign(body_flags)

    state_reference = model.state()
    state_parallel = model.state()
    _eval_fk_serial(model, state_reference, body_flag_filter=newton.BodyFlags.DYNAMIC)
    _eval_fk_parallel(model, state_parallel, body_flag_filter=newton.BodyFlags.DYNAMIC)
    assert_np_equal(state_parallel.body_q.numpy(), state_reference.body_q.numpy(), tol=1.0e-6)
    assert_np_equal(state_parallel.body_qd.numpy(), state_reference.body_qd.numpy(), tol=1.0e-6)


def test_cable_pose_preserved(test, device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cable_body = builder.add_link(
        xform=wp.transform(wp.vec3(0.4, -0.3, 0.2), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3)),
    )
    child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    cable_joint = builder.add_joint_cable(parent=-1, child=cable_body)
    child_joint = builder.add_joint_revolute(parent=cable_body, child=child, axis=newton.Axis.Z)
    builder.add_articulation([cable_joint, child_joint])
    model = builder.finalize(device=device)
    model.joint_q.assign(np.array([0.6], dtype=np.float32))
    model.joint_qd.assign(np.array([0.0, 0.0, -0.2], dtype=np.float32))

    state_reference = model.state()
    state_parallel = model.state()
    _eval_fk_serial(model, state_reference)
    _eval_fk_parallel(model, state_parallel)
    assert_np_equal(state_parallel.body_q.numpy(), state_reference.body_q.numpy(), tol=1.0e-6)
    assert_np_equal(state_parallel.body_qd.numpy(), state_reference.body_qd.numpy(), tol=1.0e-6)


def test_empty_model(test, device):
    model = newton.ModelBuilder().finalize(device=device)
    test.assertIsNone(model._fk_articulation_level_start)
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def test_loop_closing_joint(test, device):
    model = _build_loop_model(device)
    assert_np_equal(model.joint_ancestor.numpy(), np.array([-1, 2, 1], dtype=np.int32))
    assert_np_equal(model._fk_articulation_level_start.numpy(), np.array([0, 2], dtype=np.int32))
    assert_np_equal(model._fk_level_joint_start.numpy(), np.array([0, 1, 2], dtype=np.int32))
    assert_np_equal(model._fk_level_parent_pos.numpy(), np.array([-1, 0], dtype=np.int32))

    state_reference = model.state()
    state_parallel = model.state()
    _eval_fk_serial(model, state_reference)
    newton.eval_fk(model, state_parallel.joint_q, state_parallel.joint_qd, state_parallel)
    assert_np_equal(state_parallel.body_q.numpy(), state_reference.body_q.numpy(), tol=1.0e-6)
    assert_np_equal(state_parallel.body_qd.numpy(), state_reference.body_qd.numpy(), tol=1.0e-6)


def test_duplicate_child_serial_fallback(test, device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    root = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    root_joint = builder.add_joint_revolute(parent=-1, child=root, axis=newton.Axis.Z)
    child_joint = builder.add_joint_revolute(parent=root, child=child, axis=newton.Axis.Y)
    duplicate_joint = builder.add_joint_fixed(parent=root, child=child)
    builder.add_articulation([root_joint, child_joint, duplicate_joint])
    model = builder.finalize(device=device)
    model.joint_q.assign(np.array([0.3, -0.7], dtype=np.float32))

    test.assertIsNone(model._fk_articulation_level_start)

    state_reference = model.state()
    state_public = model.state()
    _eval_fk_serial(model, state_reference)
    newton.eval_fk(model, state_public.joint_q, state_public.joint_qd, state_public)
    assert_np_equal(state_public.body_q.numpy(), state_reference.body_q.numpy(), tol=1.0e-6)
    assert_np_equal(state_public.body_qd.numpy(), state_reference.body_qd.numpy(), tol=1.0e-6)


def test_cyclic_articulation_serial_fallback(test, device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body_a = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    body_b = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joint_a = builder.add_joint_revolute(parent=body_b, child=body_a, axis=newton.Axis.Z)
    joint_b = builder.add_joint_revolute(parent=body_a, child=body_b, axis=newton.Axis.Y)
    builder.add_articulation([joint_a, joint_b])
    model = builder.finalize(device=device)
    model.joint_q.assign(np.array([0.3, -0.7], dtype=np.float32))

    test.assertIsNone(model._fk_articulation_level_start)

    state_reference = model.state()
    state_public = model.state()
    _eval_fk_serial(model, state_reference)
    newton.eval_fk(model, state_public.joint_q, state_public.joint_qd, state_public)
    assert_np_equal(state_public.body_q.numpy(), state_reference.body_q.numpy(), tol=1.0e-6)
    assert_np_equal(state_public.body_qd.numpy(), state_reference.body_qd.numpy(), tol=1.0e-6)


def test_wide_level_serial_fallback(test, device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    root = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joints = [builder.add_joint_revolute(parent=-1, child=root, axis=newton.Axis.Z)]
    for i in range(FK_TILE_MAX_LEVEL_WIDTH + 1):
        child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        joints.append(
            builder.add_joint_revolute(
                parent=root,
                child=child,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(wp.vec3(0.001 * i, 0.0, 0.0), wp.quat_identity()),
            )
        )
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    model.joint_q.assign(np.linspace(-0.1, 0.1, model.joint_coord_count, dtype=np.float32))

    test.assertIsNone(model._fk_articulation_level_start)

    state_reference = model.state()
    state_public = model.state()
    _eval_fk_serial(model, state_reference)
    newton.eval_fk(model, state_public.joint_q, state_public.joint_qd, state_public)
    assert_np_equal(state_public.body_q.numpy(), state_reference.body_q.numpy(), tol=1.0e-6)
    assert_np_equal(state_public.body_qd.numpy(), state_reference.body_qd.numpy(), tol=1.0e-6)


def _eval_fk_gradients(device, use_public):
    model, body = _build_gradient_model(device)
    state = model.state(requires_grad=True)
    state.joint_q.assign(np.array([0.2, -0.4, 0.7], dtype=np.float32))
    state.joint_qd.assign(np.array([0.5, -0.3, 0.8], dtype=np.float32))
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    with wp.Tape() as tape:
        if use_public:
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        else:
            _eval_fk_serial(model, state)
        wp.launch(_body_loss, dim=1, inputs=[state.body_q, state.body_qd, body], outputs=[loss], device=device)
    tape.backward(loss)
    return tape.gradients[state.joint_q].numpy(), tape.gradients[state.joint_qd].numpy()


def test_public_gradients(test, device):
    serial_q, serial_qd = _eval_fk_gradients(device, use_public=False)
    public_q, public_qd = _eval_fk_gradients(device, use_public=True)
    assert_np_equal(public_q, serial_q, tol=2.0e-5)
    assert_np_equal(public_qd, serial_qd, tol=2.0e-5)


add_function_test(
    TestEvalFK,
    "test_heterogeneous_wide_articulations",
    test_heterogeneous_wide_articulations,
    get_test_devices(),
)
add_function_test(TestEvalFK, "test_deep_articulation", test_deep_articulation, get_test_devices())
add_function_test(
    TestEvalFK, "test_empty_articulation_selection", test_empty_articulation_selection, get_test_devices()
)
add_function_test(TestEvalFK, "test_empty_model", test_empty_model, get_test_devices())
add_function_test(
    TestEvalFK,
    "test_noncontiguous_bodies_and_external_parent",
    test_noncontiguous_bodies_and_external_parent,
    get_test_devices(),
)
add_function_test(TestEvalFK, "test_body_flag_filter", test_body_flag_filter, get_test_devices())
add_function_test(TestEvalFK, "test_cable_pose_preserved", test_cable_pose_preserved, get_test_devices())
add_function_test(TestEvalFK, "test_loop_closing_joint", test_loop_closing_joint, get_test_devices())
add_function_test(
    TestEvalFK,
    "test_duplicate_child_serial_fallback",
    test_duplicate_child_serial_fallback,
    get_test_devices(),
)
add_function_test(
    TestEvalFK,
    "test_cyclic_articulation_serial_fallback",
    test_cyclic_articulation_serial_fallback,
    get_test_devices(),
)
add_function_test(
    TestEvalFK,
    "test_wide_level_serial_fallback",
    test_wide_level_serial_fallback,
    get_selected_cuda_test_devices(),
)
add_function_test(
    TestEvalFK,
    "test_public_gradients",
    test_public_gradients,
    get_selected_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
