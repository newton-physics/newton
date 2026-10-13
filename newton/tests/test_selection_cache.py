# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import gc
import unittest
import weakref

import numpy as np
import warp as wp

import newton
from newton.actuators import Actuator, DrivePD
from newton.selection import ArticulationView
from newton.tests.unittest_utils import add_function_test, assert_np_equal, get_test_devices


class TestSelectionCacheLifetime(unittest.TestCase):
    def attribute_sources_release_with_view(self, device):
        """Release abandoned attribute sources while another view remains usable."""
        for layout in ("dense", "indexed"):
            with self.subTest(layout=layout):
                live_model, live_view = self.make_view(layout, device=device)
                live_model.joint_q.fill_(17.0)
                live_buffer = live_view.get_dof_positions(live_model)
                live_values = live_buffer.numpy().copy()

                model, view = self.make_view(layout, device=device)
                state, control = model.state(), model.control()
                view.get_dof_positions(model)
                view.get_dof_positions(state)
                view.get_dof_forces(control)
                refs = {
                    name: weakref.ref(obj)
                    for name, obj in (("view", view), ("model", model), ("state", state), ("control", control))
                }
                del model, view, state, control
                gc.collect()

                assert_np_equal(live_buffer.numpy(), live_values)
                self.assertIs(live_view.get_dof_positions(live_model), live_buffer)
                assert_np_equal(live_view.get_dof_positions(live_model).numpy(), live_values)
                for name, ref in refs.items():
                    with self.subTest(source=name):
                        self.assertIsNone(ref(), f"Abandoned {name} retained for {layout}")

    def actuator_sources_release_with_view(self, device):
        """Release abandoned actuator mappings independently of a second live view."""
        for layout in ("dense", "indexed"):
            with self.subTest(layout=layout):
                live_model, live_view = self.make_view(layout, device=device)
                live_actuator = self.make_actuator(live_model, 17.0)
                live_buffer = live_view.get_actuator_parameter(live_actuator, live_actuator.drive, "kp")
                live_values = live_buffer.numpy().copy()

                model, view = self.make_view(layout, device=device)
                actuator = self.make_actuator(model, 3.0)
                values = view.get_actuator_parameter(actuator, actuator.drive, "kp").numpy()
                assert_np_equal(values, np.full(values.shape, 3.0))
                refs = {
                    name: weakref.ref(obj) for name, obj in (("view", view), ("model", model), ("actuator", actuator))
                }
                del model, view, actuator
                gc.collect()

                assert_np_equal(live_buffer.numpy(), live_values)
                assert_np_equal(
                    live_view.get_actuator_parameter(live_actuator, live_actuator.drive, "kp").numpy(), live_values
                )
                for name, ref in refs.items():
                    with self.subTest(source=name):
                        self.assertIsNone(ref(), f"Abandoned {name} retained for {layout}")

    def attribute_sources_and_slices_remain_distinct(self, device):
        """Keep model, state, control, root slices, and full selections independent."""
        for layout in ("dense", "indexed"):
            with self.subTest(layout=layout):
                model, view = self.make_view(layout, device=device)
                states = [model.state(), model.state()]
                controls = [model.control(), model.control()]
                sources = [model, *states]
                for value, source in enumerate(sources, 1):
                    source.joint_q.fill_(float(value))
                for value, control in enumerate(controls, 4):
                    control.joint_f.fill_(float(value))
                buffers = [view.get_dof_positions(source) for source in sources]
                force_buffers = [view.get_dof_forces(control) for control in controls]
                for value, (source, buffer) in enumerate(zip(sources, buffers, strict=True), 1):
                    root = view.get_root_transforms(source)
                    self.assertEqual(root.shape, (view.world_count, 1))
                    assert_np_equal(root.numpy(), np.full((*root.shape, 7), value))
                    assert_np_equal(buffer.numpy(), np.full(buffer.shape, value))
                    self.assertIs(view.get_dof_positions(source), buffer)
                    assert_np_equal(view.get_dof_positions(source).numpy(), buffer.numpy())
                for value, (control, buffer) in enumerate(zip(controls, force_buffers, strict=True), 4):
                    assert_np_equal(buffer.numpy(), np.full(buffer.shape, value))
                    self.assertIs(view.get_dof_forces(control), buffer)
                    assert_np_equal(view.get_dof_forces(control).numpy(), buffer.numpy())

    def returned_array_owns_backing_allocation(self, device):
        """Keep zero-copy data and gradient allocations alive beyond their view."""
        for requires_grad in (False, True):
            with self.subTest(requires_grad=requires_grad):
                model, view = self.make_view("dense", requires_grad=requires_grad, device=device)
                model.joint_q.fill_(5.0)
                source_ref = weakref.ref(model.joint_q)
                model_ref, view_ref = weakref.ref(model), weakref.ref(view)
                buffer = view.get_dof_positions(model)
                if requires_grad:
                    model.joint_q.grad.fill_(7.0)
                    grad_ref = weakref.ref(model.joint_q.grad)
                    gradient = buffer.grad
                del model, view
                gc.collect()

                self.assertIsNone(view_ref())
                self.assertIsNone(model_ref())
                self.assertIsNotNone(source_ref())
                assert_np_equal(buffer.numpy(), np.full(buffer.shape, 5.0))
                del buffer
                gc.collect()
                if requires_grad:
                    self.assertIsNotNone(grad_ref())
                    assert_np_equal(gradient.numpy(), np.full(gradient.shape, 7.0))
                    del gradient
                    gc.collect()
                    self.assertIsNone(grad_ref())
                self.assertIsNone(source_ref())

    def view_holds_sources_and_actuators_weakly(self, device):
        """Release states, controls, and actuators that are dropped while their view stays alive."""
        model, view = self.make_view("dense", device=device)
        state, control = model.state(), model.control()
        actuator = self.make_actuator(model, 3.0)
        view.get_dof_positions(state)
        view.get_dof_forces(control)
        view.get_actuator_parameter(actuator, actuator.drive, "kp")
        refs = {
            name: weakref.ref(obj) for name, obj in (("state", state), ("control", control), ("actuator", actuator))
        }
        del state, control, actuator
        gc.collect()

        for name, ref in refs.items():
            with self.subTest(source=name):
                self.assertIsNone(ref(), f"{name} retained by a live view")
        model.joint_q.fill_(2.0)
        positions = view.get_dof_positions(model)
        assert_np_equal(positions.numpy(), np.full(positions.shape, 2.0))

    def getters_follow_replaced_source_arrays(self, device):
        """Read and write the current source arrays after they are replaced."""
        for layout in ("dense", "indexed"):
            with self.subTest(layout=layout):
                model, view = self.make_view(layout, device=device)
                state, control = model.state(), model.control()
                view.get_dof_positions(state)
                view.get_root_transforms(state)
                view.get_dof_forces(control)
                old_q, old_f = state.joint_q, control.joint_f
                old_q_values, old_f_values = old_q.numpy().copy(), old_f.numpy().copy()

                state.joint_q = wp.full(old_q.shape, 4.0, dtype=old_q.dtype, device=device)
                control.joint_f = wp.full(old_f.shape, 6.0, dtype=old_f.dtype, device=device)
                q = view.get_dof_positions(state)
                assert_np_equal(q.numpy(), np.full(q.shape, 4.0))
                root = view.get_root_transforms(state)
                assert_np_equal(root.numpy(), np.full((*root.shape, 7), 4.0))
                forces = view.get_dof_forces(control)
                assert_np_equal(forces.numpy(), np.full(forces.shape, 6.0))

                view.set_dof_positions(state, np.full(q.shape, 9.0, dtype=np.float32))
                view.set_dof_forces(control, np.full(forces.shape, 8.0, dtype=np.float32))
                assert_np_equal(view.get_dof_positions(state).numpy(), np.full(q.shape, 9.0))
                assert_np_equal(view.get_dof_forces(control).numpy(), np.full(forces.shape, 8.0))
                # unselected coordinates of the indexed layout keep the replacement's value
                self.assertEqual(set(np.unique(state.joint_q.numpy())), {9.0} if layout == "dense" else {4.0, 9.0})
                assert_np_equal(old_q.numpy(), old_q_values)
                assert_np_equal(old_f.numpy(), old_f_values)

    def getters_follow_replaced_gradient(self, device):
        """Bind the current gradient after a source array's gradient is replaced."""
        model, view = self.make_view("dense", requires_grad=True, device=device)
        self.assertEqual(view.get_dof_positions(model).grad.ptr, model.joint_q.grad.ptr)

        model.joint_q.grad = wp.full_like(model.joint_q, 3.0)
        gradient = view.get_dof_positions(model).grad
        self.assertEqual(gradient.ptr, model.joint_q.grad.ptr)
        assert_np_equal(gradient.numpy(), np.full(gradient.shape, 3.0))

    def getters_follow_arrays_replaced_by_xpbd(self, device):
        """Read and write the arrays XPBD assigns to a differentiable output state."""
        model, view = self.make_view("dense", requires_grad=True, device=device)
        state_in, state_out = model.state(), model.state()
        view.get_link_transforms(state_out)
        view.get_link_velocities(state_out)
        old_q, old_qd = state_out.body_q, state_out.body_qd

        newton.solvers.SolverXPBD(model).step(state_in, state_out, model.control(), None, 1e-3)
        self.assertIsNot(state_out.body_q, old_q)
        self.assertIsNot(state_out.body_qd, old_qd)
        # stale reads would return these values
        old_q.fill_(wp.transform((7.0, 7.0, 7.0), wp.quat_identity()))
        old_qd.fill_(wp.spatial_vector(7.0, 7.0, 7.0, 7.0, 7.0, 7.0))
        old_q_values = old_q.numpy()
        transforms = view.get_link_transforms(state_out)
        velocities = view.get_link_velocities(state_out)
        assert_np_equal(transforms.numpy().reshape(-1, 7), state_out.body_q.numpy())
        assert_np_equal(velocities.numpy().reshape(-1, 6), state_out.body_qd.numpy())

        target = wp.transform((1.0, 2.0, 3.0), wp.quat_identity())
        view.set_attribute("body_q", state_out, wp.full(transforms.shape, target, dtype=wp.transform, device=device))
        assert_np_equal(state_out.body_q.numpy(), np.tile(np.array(target, dtype=np.float32), (model.body_count, 1)))
        assert_np_equal(old_q.numpy(), old_q_values)

    def native_slices_reuse_cached_arrays(self, device):
        """Cache arrays selected by native slices, which are unhashable before Python 3.12."""
        model, view = self.make_view("dense", device=device)
        model.joint_q.fill_(2.0)
        values = view._get_attribute_values("joint_q", model, _slice=slice(0, 3))
        self.assertEqual(values.shape, (view.world_count, view.count_per_world, 3))
        assert_np_equal(values.numpy(), np.full(values.shape, 2.0))
        self.assertIs(view._get_attribute_array("joint_q", model, _slice=slice(0, 3)), values)

    def observable_reads_follow_replaced_arrays(self, device):
        """Read and write the current solver observable arrays, including replaced and unrequested ones."""
        model, view = self.make_view("dense", device=device)
        flag = newton.solvers.SolverBase.ObservableKind.BODY_PARENT_F
        observables = newton.solvers.SolverFeatherstone(model).observables({flag})
        observables.body_parent_f.fill_(wp.spatial_vector(2.0, 2.0, 2.0, 2.0, 2.0, 2.0))
        forces = view.get_attribute("body_parent_f", observables)
        assert_np_equal(forces.numpy(), np.full((*forces.shape, 6), 2.0))
        self.assertIs(view.get_attribute("body_parent_f", observables), forces)

        old_forces = observables.body_parent_f
        observables.body_parent_f = wp.full_like(old_forces, wp.spatial_vector(4.0, 4.0, 4.0, 4.0, 4.0, 4.0))
        forces = view.get_attribute("body_parent_f", observables)
        assert_np_equal(forces.numpy(), np.full((*forces.shape, 6), 4.0))
        target = wp.spatial_vector(5.0, 5.0, 5.0, 5.0, 5.0, 5.0)
        view.set_attribute("body_parent_f", observables, wp.full(forces.shape, target, device=device))
        assert_np_equal(observables.body_parent_f.numpy(), np.full((model.body_count, 6), 5.0))
        assert_np_equal(old_forces.numpy(), np.full((model.body_count, 6), 2.0))

        observables.body_parent_f = None
        with self.assertRaisesRegex(ValueError, "not requested"):
            view.get_attribute("body_parent_f", observables)

    def observable_cache_entries_release_with_observables(self, device):
        """Cache observable views per container and drop them with the container."""
        model, view = self.make_view("dense", device=device)
        flag = newton.solvers.SolverBase.ObservableKind.BODY_PARENT_F
        observables = newton.solvers.SolverFeatherstone(model).observables({flag})
        selection = observables.select({flag})
        forces = view.get_attribute("body_parent_f", observables)
        selected_forces = view.get_attribute("body_parent_f", selection)
        self.assertIsNot(selected_forces, forces)
        self.assertIs(view.get_attribute("body_parent_f", observables), forces)
        self.assertIs(view.get_attribute("body_parent_f", selection), selected_forces)
        self.assertEqual(len(view._attribute_array_cache), 2)

        refs = {"observables": weakref.ref(observables), "selection": weakref.ref(selection)}
        del observables, selection, forces, selected_forces
        gc.collect()

        for name, ref in refs.items():
            with self.subTest(source=name):
                self.assertIsNone(ref(), f"{name} retained by a live view")
        self.assertEqual(len(view._attribute_array_cache), 0)

    @staticmethod
    def make_view(layout, requires_grad=False, device="cpu"):
        """Build regular or indexed selections on the requested device."""
        robot = newton.ModelBuilder()
        parent = robot.add_link(label="robot/root")
        joints = [robot.add_joint_free(child=parent, label="robot/root_joint")]
        for index in range(3):
            child = robot.add_link(label=f"robot/link_{index}")
            joints.append(robot.add_joint_revolute(parent=parent, child=child, label=f"robot/joint_{index}"))
            parent = child
        robot.add_articulation(joints, label="robot")
        scene = newton.ModelBuilder()
        for _ in range(2):
            scene.add_world(robot)
        model = scene.finalize(device=device, requires_grad=requires_grad)
        selected = ["root_joint", "joint_0", "joint_2"] if layout == "indexed" else None
        return model, ArticulationView(model, "robot", include_joints=selected, verbose=False)

    @staticmethod
    def make_actuator(model, value):
        """Build an actuator covering every model DOF."""
        count = model.joint_dof_count
        return Actuator(
            indices=wp.array(np.arange(count), dtype=wp.uint32, device=model.device),
            drive=DrivePD(kp=wp.full(count, value, device=model.device), kd=wp.zeros(count, device=model.device)),
        )


devices = get_test_devices()
add_function_test(
    TestSelectionCacheLifetime,
    "test_attribute_sources_release_with_view",
    TestSelectionCacheLifetime.attribute_sources_release_with_view,
    devices=devices,
)
add_function_test(
    TestSelectionCacheLifetime,
    "test_actuator_sources_release_with_view",
    TestSelectionCacheLifetime.actuator_sources_release_with_view,
    devices=devices,
)
add_function_test(
    TestSelectionCacheLifetime,
    "test_attribute_sources_and_slices_remain_distinct",
    TestSelectionCacheLifetime.attribute_sources_and_slices_remain_distinct,
    devices=devices,
)
add_function_test(
    TestSelectionCacheLifetime,
    "test_returned_array_owns_backing_allocation",
    TestSelectionCacheLifetime.returned_array_owns_backing_allocation,
    devices=devices,
)
add_function_test(
    TestSelectionCacheLifetime,
    "test_view_holds_sources_and_actuators_weakly",
    TestSelectionCacheLifetime.view_holds_sources_and_actuators_weakly,
    devices=devices,
)
add_function_test(
    TestSelectionCacheLifetime,
    "test_getters_follow_replaced_source_arrays",
    TestSelectionCacheLifetime.getters_follow_replaced_source_arrays,
    devices=devices,
)
add_function_test(
    TestSelectionCacheLifetime,
    "test_getters_follow_replaced_gradient",
    TestSelectionCacheLifetime.getters_follow_replaced_gradient,
    devices=devices,
)
add_function_test(
    TestSelectionCacheLifetime,
    "test_getters_follow_arrays_replaced_by_xpbd",
    TestSelectionCacheLifetime.getters_follow_arrays_replaced_by_xpbd,
    devices=devices,
)
add_function_test(
    TestSelectionCacheLifetime,
    "test_native_slices_reuse_cached_arrays",
    TestSelectionCacheLifetime.native_slices_reuse_cached_arrays,
    devices=devices,
)
add_function_test(
    TestSelectionCacheLifetime,
    "test_observable_reads_follow_replaced_arrays",
    TestSelectionCacheLifetime.observable_reads_follow_replaced_arrays,
    devices=devices,
)
add_function_test(
    TestSelectionCacheLifetime,
    "test_observable_cache_entries_release_with_observables",
    TestSelectionCacheLifetime.observable_cache_entries_release_with_observables,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
