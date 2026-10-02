# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for batched deformable state access and indexed writes."""

import unittest

import numpy as np
import warp as wp

from newton.selection import (
    DeformableCurveView,
    DeformableSurfaceView,
)
from newton.tests._selection_deformable_test_utils import (
    _irregular_model,
    _replicated_model,
)


class TestDeformableStateReads(unittest.TestCase):
    """Read regular and irregular state layouts, including captured gathers."""

    def test_cloth_view_selects_and_batches_across_worlds(self):
        """A replicated cloth selects one deformable object per world with batched particle state."""
        model = _replicated_model(3)
        state = model.state()

        view = DeformableSurfaceView(model, "/World/Cloth")
        self.assertEqual((view.count, view.world_count, view.count_per_world), (3, 3, 1))
        self.assertEqual(view.worlds, [0, 1, 2])
        self.assertEqual(view.particles_per_deformable_object, 4)

        positions = view.get_particle_positions(state)
        self.assertEqual(positions.shape, (3, 4))

        # Round-trip: lift each world's cloth by its world index and read it back.
        lifted = positions.numpy().copy()
        for g in range(3):
            lifted[g, :, 2] += float(g + 1)
        view.set_particle_positions(state, wp.array(lifted, dtype=wp.vec3))
        np.testing.assert_allclose(view.get_particle_positions(state).numpy(), lifted, atol=1e-6)

        # Velocities go through the same path.
        velocities = np.full((3, 4, 3), 2.5, dtype=np.float32)
        view.set_particle_velocities(state, wp.array(velocities, dtype=wp.vec3))
        np.testing.assert_allclose(view.get_particle_velocities(state).numpy(), velocities, atol=1e-6)

    def test_cable_view_batches_body_transforms(self):
        """A replicated cable selects per world and round-trips its segment transforms."""
        model = _replicated_model(2)
        state = model.state()

        view = DeformableCurveView(model, "/World/Cable")
        self.assertEqual((view.count, view.bodies_per_deformable_object), (2, 3))

        transforms = view.get_body_transforms(state)
        self.assertEqual(transforms.shape, (2, 3))
        shifted = transforms.numpy().copy()
        shifted[:, :, 1] += 5.0  # translate all segments in y
        view.set_body_transforms(state, wp.array(shifted, dtype=wp.transform))
        np.testing.assert_allclose(view.get_body_transforms(state).numpy(), shifted, atol=1e-6)

        velocities = view.get_body_velocities(state)
        self.assertEqual(velocities.shape, (2, 3))

    def test_view_round_trip_on_cpu(self):
        """The gather/scatter path works on a CPU-finalized model, not just CUDA."""
        model = _replicated_model(2, device="cpu")
        state = model.state()

        view = DeformableSurfaceView(model, "/World/Cloth")
        positions = view.get_particle_positions(state)
        self.assertTrue(positions.device.is_cpu)
        before = state.particle_q.numpy().copy()
        lifted = positions.numpy().copy()
        lifted[..., 2] += 1.0
        np.testing.assert_array_equal(state.particle_q.numpy(), before)
        view.set_particle_positions(state, wp.array(lifted, dtype=wp.vec3, device="cpu"))
        np.testing.assert_allclose(view.get_particle_positions(state).numpy(), lifted, atol=1e-6)

    def test_regular_getter_returns_a_live_state_view(self):
        """A regular selection aliases state data like an ArticulationView getter."""
        model = _replicated_model(2, device="cpu")
        state = model.state()
        view = DeformableSurfaceView(model, "/World/Cloth")
        positions = view.get_particle_positions(state)
        moved = positions.numpy().copy()
        moved[..., 2] += 1.0

        view.set_particle_positions(state, wp.array(moved, dtype=wp.vec3, device=model.device))

        np.testing.assert_allclose(positions.numpy(), moved, atol=1.0e-6)

    def test_irregular_getter_reuses_internal_staging(self):
        """An indexed selection reuses its internally owned contiguous result."""
        model = _irregular_model(device="cpu")
        state = model.state()
        view = DeformableSurfaceView(model, "selected_cloth_*")
        first = view.get_particle_positions(state)
        moved = first.numpy().copy()
        moved[..., 2] += 1.0

        view.set_particle_positions(state, wp.array(moved, dtype=wp.vec3, device=model.device))
        second = view.get_particle_positions(state)

        self.assertIs(first, second)
        np.testing.assert_allclose(first.numpy(), moved, atol=1.0e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA graph capture")
    def test_irregular_getters_capture_all_state_dtypes(self):
        """Internally staged vec3, transform, and spatial-vector reads replay in a CUDA graph."""
        device = wp.get_device("cuda:0")
        model = _irregular_model(device=device)
        state = model.state()
        cloth = DeformableSurfaceView(model, "selected_cloth_*")
        cable = DeformableCurveView(model, "selected_cable_*")

        for name, getter, setter, dtype in (
            ("particle_positions", cloth.get_particle_positions, cloth.set_particle_positions, wp.vec3),
            ("particle_velocities", cloth.get_particle_velocities, cloth.set_particle_velocities, wp.vec3),
            ("body_transforms", cable.get_body_transforms, cable.set_body_transforms, wp.transform),
            (
                "body_velocities",
                cable.get_body_velocities,
                cable.set_body_velocities,
                wp.spatial_vector,
            ),
        ):
            with self.subTest(getter=name):
                result = getter(state)  # warm up and allocate indexed staging
                with wp.ScopedCapture(device) as capture:
                    captured_result = getter(state)
                self.assertIs(result, captured_result)

                expected = result.numpy().copy()
                expected[..., 0] += 0.25
                setter(state, wp.array(expected, dtype=dtype, device=device))
                wp.capture_launch(capture.graph)
                np.testing.assert_allclose(result.numpy(), expected, atol=1.0e-6)


class TestDeformableStateWrites(unittest.TestCase):
    """Validate selectors and write state on CPU and under CUDA graph capture."""

    def test_repeated_device_index_writes_preserve_gradients(self):
        """Preserve each write's gradients when one view updates separate states."""

        def differentiate_writes(setter, states, values, indices, sources, seeds):
            with wp.Tape() as tape:
                for state, value, index in zip(states, values, indices, strict=True):
                    setter(state, value, deformable_object_indices=index, source_indices=sources)
            tape.backward(grads=seeds)
            return tape

        for device in wp.get_devices():
            model = _replicated_model(3, device=device)
            cloth = DeformableSurfaceView(model, "/World/Cloth")
            cable = DeformableCurveView(model, "/World/Cable")
            for getter, setter, attribute in (
                (cloth.get_particle_positions, cloth.set_particle_positions, "particle_q"),
                (cloth.get_particle_velocities, cloth.set_particle_velocities, "particle_qd"),
                (cable.get_body_transforms, cable.set_body_transforms, "body_q"),
                (cable.get_body_velocities, cable.set_body_velocities, "body_qd"),
            ):
                for source_rows, row_gradients in (
                    (None, [0, 1, 1, 0, 0]),
                    ([2, 0, 0, 1, 1], [2, 0, 0]),
                    ([2, 3, 0, 1, 1], [1, 0, 0]),
                ):
                    with self.subTest(device=device, setter=setter.__name__, source_rows=source_rows):
                        states = [model.state(requires_grad=True) for _ in range(2)]
                        targets = [getattr(state, attribute) for state in states]
                        values = [
                            wp.array(
                                np.repeat(getter(state).numpy()[:1], len(row_gradients), axis=0),
                                dtype=target.dtype,
                                device=device,
                                requires_grad=True,
                            )
                            for state, target in zip(states, targets, strict=True)
                        ]
                        # Only the last duplicate and object 2 are written. The
                        # second call must not erase the first call's winners.
                        indices = [wp.array([i, i, 2, -1, 3], dtype=wp.int32, device=device) for i in range(2)]
                        sources = None if source_rows is None else wp.array(source_rows, dtype=wp.int32, device=device)
                        seeds = {target: wp.full_like(target, i + 1) for i, target in enumerate(targets)}
                        expected = [
                            np.broadcast_to(
                                (i + 1) * np.array(row_gradients, dtype=np.float32)[:, None, None], value.numpy().shape
                            )
                            for i, value in enumerate(values)
                        ]

                        tape = differentiate_writes(setter, states, values, indices, sources, seeds)
                        for value, gradient in zip(values, expected, strict=True):
                            np.testing.assert_array_equal(value.grad.numpy(), gradient)

                        if device.is_cuda:
                            tape.zero()
                            with wp.ScopedCapture(device) as capture:
                                tape = differentiate_writes(setter, states, values, indices, sources, seeds)
                            for _ in range(2):
                                tape.zero()
                                wp.capture_launch(capture.graph)
                                for value, gradient in zip(values, expected, strict=True):
                                    np.testing.assert_array_equal(value.grad.numpy(), gradient)

    def test_indexed_partial_writes_touch_only_selected_objects(self):
        """deformable_object_indices= scatters into selected deformable objects only, from host and device index
        forms, and cable body velocities round-trip through an indexed write."""
        model = _replicated_model(3)
        state = model.state()

        cloth = DeformableSurfaceView(model, "/World/Cloth")
        before = cloth.get_particle_positions(state).numpy().copy()
        moved = before[[1]].copy()
        moved[..., 2] += 5.0
        cloth.set_particle_positions(state, wp.array(moved, dtype=wp.vec3), deformable_object_indices=[1])
        after = cloth.get_particle_positions(state).numpy()
        np.testing.assert_array_equal(after[[0, 2]], before[[0, 2]])
        np.testing.assert_allclose(after[1], moved[0], atol=1e-6)

        cable = DeformableCurveView(model, "/World/Cable")
        velocities = np.zeros((1, cable.bodies_per_deformable_object, 6), dtype=np.float32)
        velocities[..., 3] = 2.0
        device_indices = wp.array([2], dtype=wp.int32, device=model.device)
        cable.set_body_velocities(
            state, wp.array(velocities, dtype=wp.spatial_vector), deformable_object_indices=device_indices
        )
        out = cable.get_body_velocities(state).numpy()
        np.testing.assert_allclose(out[2], velocities[0], atol=1e-6)
        np.testing.assert_array_equal(out[:2], np.zeros_like(out[:2]))

        with self.assertRaisesRegex(ValueError, "must be in"):
            cloth.set_particle_positions(state, wp.array(moved, dtype=wp.vec3), deformable_object_indices=[3])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            cloth.set_particle_positions(
                state,
                wp.array(np.repeat(moved, 2, axis=0), dtype=wp.vec3),
                deformable_object_indices=[1, 1],
            )

    def test_source_indices_map_value_rows_to_objects(self):
        """Source rows can be mapped independently to destination deformable objects."""
        model = _replicated_model(3, device="cpu")
        state = model.state()
        cloth = DeformableSurfaceView(model, "/World/Cloth")
        before = cloth.get_particle_positions(state).numpy().copy()
        values = np.zeros((4, cloth.particles_per_deformable_object, 3), dtype=np.float32)
        values[1].fill(10.0)
        values[3].fill(30.0)

        cloth.set_particle_positions(
            state,
            wp.array(values, dtype=wp.vec3, device=model.device),
            deformable_object_indices=[0, 2],
            source_indices=[3, 1],
        )

        after = cloth.get_particle_positions(state).numpy()
        np.testing.assert_array_equal(after[0], values[3])
        np.testing.assert_array_equal(after[1], before[1])
        np.testing.assert_array_equal(after[2], values[1])

    def test_setters_reject_values_sharing_target_storage(self):
        """Reject overlapping input before a remapped write changes any state."""
        for device in wp.get_devices():
            model = _replicated_model(3, device=device)
            state = model.state()
            cloth = DeformableSurfaceView(model, "/World/Cloth")
            cable = DeformableCurveView(model, "/World/Cable")
            for getter, setter, dtype in (
                (cloth.get_particle_positions, cloth.set_particle_positions, wp.vec3),
                (cloth.get_particle_velocities, cloth.set_particle_velocities, wp.vec3),
                (cable.get_body_transforms, cable.set_body_transforms, wp.transform),
                (cable.get_body_velocities, cable.set_body_velocities, wp.spatial_vector),
            ):
                before = getter(state).numpy().copy()
                before[..., 0] = np.arange(3)[:, None]
                initial = wp.array(before, dtype=dtype, device=device)
                setter(state, initial)
                live = getter(state)
                inputs = [live, live[1:], live[::2]]
                if device.is_cpu:
                    inputs.append(live.numpy())
                for values in inputs:
                    with self.subTest(device=device, setter=setter.__name__, input_type=type(values), rows=len(values)):
                        with self.assertRaisesRegex(ValueError, "overlap.*wp.clone"):
                            setter(state, values, deformable_object_indices=[0, 1], source_indices=[1, 0])
                        np.testing.assert_array_equal(getter(state).numpy(), before)

                # Independent input makes the row exchange well-defined on every device.
                setter(state, wp.clone(live), deformable_object_indices=[0, 1], source_indices=[1, 0])
                expected = before.copy()
                expected[:2] = before[[1, 0]]
                np.testing.assert_array_equal(getter(state).numpy(), expected)

    def test_source_indices_map_particle_velocity_rows(self):
        """Particle velocity writes use the same source-to-deformable-object mapping."""
        model = _replicated_model(3, device="cpu")
        state = model.state()
        cloth = DeformableSurfaceView(model, "/World/Cloth")
        values = np.zeros((4, cloth.particles_per_deformable_object, 3), dtype=np.float32)
        values[0].fill(2.0)
        values[3].fill(7.0)

        cloth.set_particle_velocities(
            state,
            wp.array(values, dtype=wp.vec3, device=model.device),
            deformable_object_indices=[1, 2],
            source_indices=[3, 0],
        )

        after = cloth.get_particle_velocities(state).numpy()
        np.testing.assert_array_equal(after[0], np.zeros_like(after[0]))
        np.testing.assert_array_equal(after[1], values[3])
        np.testing.assert_array_equal(after[2], values[0])

    def test_source_indices_map_body_transform_rows(self):
        """Cable transform writes map source rows independently."""
        model = _replicated_model(3, device="cpu")
        state = model.state()
        cable = DeformableCurveView(model, "/World/Cable")
        before = cable.get_body_transforms(state).numpy().copy()
        values = np.repeat(before[[0]], 4, axis=0)
        values[1, :, 0] += 2.0
        values[3, :, 1] -= 4.0

        cable.set_body_transforms(
            state,
            wp.array(values, dtype=wp.transform, device=model.device),
            deformable_object_indices=[0, 2],
            source_indices=[3, 1],
        )

        after = cable.get_body_transforms(state).numpy()
        np.testing.assert_allclose(after[0], values[3], atol=1.0e-6)
        np.testing.assert_array_equal(after[1], before[1])
        np.testing.assert_allclose(after[2], values[1], atol=1.0e-6)

    def test_source_indices_map_body_velocity_rows(self):
        """Cable velocity writes map source rows independently."""
        model = _replicated_model(3, device="cpu")
        state = model.state()
        cable = DeformableCurveView(model, "/World/Cable")
        values = np.zeros((4, cable.bodies_per_deformable_object, 6), dtype=np.float32)
        values[0, :, 0] = 2.0
        values[3, :, 4] = -5.0

        cable.set_body_velocities(
            state,
            wp.array(values, dtype=wp.spatial_vector, device=model.device),
            deformable_object_indices=[1, 2],
            source_indices=[3, 0],
        )

        after = cable.get_body_velocities(state).numpy()
        np.testing.assert_array_equal(after[0], np.zeros_like(after[0]))
        np.testing.assert_array_equal(after[1], values[3])
        np.testing.assert_array_equal(after[2], values[0])

    def test_source_indices_validate_host_contract(self):
        """Host source rows are integral, in range, and aligned with destinations."""
        model = _replicated_model(3, device="cpu")
        state = model.state()
        cloth = DeformableSurfaceView(model, "/World/Cloth")
        values = wp.zeros((4, cloth.particles_per_deformable_object), dtype=wp.vec3, device=model.device)

        for invalid_indices in ([-1], [4]):
            with self.subTest(source_indices=invalid_indices):
                with self.assertRaisesRegex(ValueError, "source_indices"):
                    cloth.set_particle_positions(
                        state,
                        values,
                        deformable_object_indices=[0],
                        source_indices=invalid_indices,
                    )

        for invalid_indices in ([1.5], ["1"], [True]):
            with self.subTest(source_indices=invalid_indices):
                with self.assertRaisesRegex(TypeError, "source_indices"):
                    cloth.set_particle_positions(
                        state,
                        values,
                        deformable_object_indices=[0],
                        source_indices=invalid_indices,
                    )

        with self.assertRaisesRegex(ValueError, "source_indices length"):
            cloth.set_particle_positions(
                state,
                values,
                deformable_object_indices=[0, 1],
                source_indices=[0],
            )

        repeated = np.full((1, cloth.particles_per_deformable_object, 3), 6.0, dtype=np.float32)
        cloth.set_particle_positions(
            state,
            wp.array(repeated, dtype=wp.vec3, device=model.device),
            deformable_object_indices=[0, 2],
            source_indices=[0, 0],
        )
        after = cloth.get_particle_positions(state).numpy()
        np.testing.assert_array_equal(after[0], repeated[0])
        np.testing.assert_array_equal(after[2], repeated[0])

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA graph capture")
    def test_source_indices_capture_and_replay(self):
        """Device source and destination rows can change between captured writes."""
        device = wp.get_device("cuda:0")
        model = _replicated_model(3, device=device)
        state = model.state()
        cloth = DeformableSurfaceView(model, "/World/Cloth")
        initial = cloth.get_particle_positions(state).numpy().copy()
        values_np = np.zeros((4, cloth.particles_per_deformable_object, 3), dtype=np.float32)
        values_np[0].fill(2.0)
        values_np[1].fill(4.0)
        values_np[2].fill(6.0)
        values_np[3].fill(8.0)
        values = wp.array(values_np, dtype=wp.vec3, device=device)
        objects = wp.array([0, 2], dtype=wp.int32, device=device)
        sources = wp.array([3, 1], dtype=wp.int32, device=device)

        cloth.set_particle_positions(
            state,
            values,
            deformable_object_indices=objects,
            source_indices=sources,
        )
        state.particle_q.assign(model.particle_q)

        with wp.ScopedCapture(device) as capture:
            cloth.set_particle_positions(
                state,
                values,
                deformable_object_indices=objects,
                source_indices=sources,
            )

        wp.capture_launch(capture.graph)
        expected = initial.copy()
        expected[0] = values_np[3]
        expected[2] = values_np[1]
        np.testing.assert_array_equal(cloth.get_particle_positions(state).numpy(), expected)

        objects.assign(np.array([1, 2], dtype=np.int32))
        sources.assign(np.array([0, 2], dtype=np.int32))
        wp.capture_launch(capture.graph)
        expected[1] = values_np[0]
        expected[2] = values_np[2]
        np.testing.assert_array_equal(cloth.get_particle_positions(state).numpy(), expected)

        before_invalid = cloth.get_particle_positions(state).numpy().copy()
        sources.assign(np.array([-1, values.shape[0]], dtype=np.int32))
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(cloth.get_particle_positions(state).numpy(), before_invalid)

    def test_host_selectors_reject_numpy_booleans(self):
        """Reject NumPy Boolean source and destination selectors without changing state."""
        model = _replicated_model(3, device="cpu")
        cloth = DeformableSurfaceView(model, "/World/Cloth")
        values = wp.full((2, cloth.particles_per_deformable_object), 9.0, dtype=wp.vec3, device="cpu")

        for argument in ("deformable_object_indices", "source_indices"):
            for invalid_indices in ([np.bool_(False)], [np.bool_(True)], np.array([False]), np.array([True])):
                with self.subTest(argument=argument, invalid_indices=invalid_indices):
                    state = model.state()
                    before = state.particle_q.numpy().copy()
                    selectors = {"deformable_object_indices": [0], "source_indices": [0]}
                    selectors[argument] = invalid_indices

                    with self.assertRaisesRegex(TypeError, argument):
                        cloth.set_particle_positions(state, values, **selectors)

                    np.testing.assert_array_equal(state.particle_q.numpy(), before)

    def test_host_indices_require_integral_values(self):
        """Host selectors reject lossy coercions before they can write another deformable object."""
        model = _replicated_model(3, device="cpu")
        cloth = DeformableSurfaceView(model, "/World/Cloth")
        values = wp.full((1, cloth.particles_per_deformable_object), 9.0, dtype=wp.vec3, device="cpu")

        for invalid_indices in ([-0.2], [1.9], ["1"], [False], [True]):
            with self.subTest(invalid_indices=invalid_indices):
                state = model.state()
                before = cloth.get_particle_positions(state).numpy().copy()
                with self.assertRaisesRegex(TypeError, "deformable_object_indices"):
                    cloth.set_particle_positions(state, values, deformable_object_indices=invalid_indices)
                np.testing.assert_array_equal(cloth.get_particle_positions(state).numpy(), before)

        state = model.state()
        before = cloth.get_particle_positions(state).numpy().copy()
        cloth.set_particle_positions(state, values, deformable_object_indices=[np.int64(1)])
        after = cloth.get_particle_positions(state).numpy()
        np.testing.assert_array_equal(after[[0, 2]], before[[0, 2]])
        np.testing.assert_array_equal(after[1], np.full_like(after[1], 9.0))

        state = model.state()
        before = cloth.get_particle_positions(state).numpy().copy()
        empty_values = wp.empty((0, cloth.particles_per_deformable_object), dtype=wp.vec3, device="cpu")
        cloth.set_particle_positions(state, empty_values, deformable_object_indices=[])
        np.testing.assert_array_equal(cloth.get_particle_positions(state).numpy(), before)

    def test_invalid_device_object_index_does_not_write(self):
        """An out-of-range device deformable object index cannot address another deformable object's state."""
        model = _replicated_model(3)
        state = model.state()
        cloth = DeformableSurfaceView(model, "/World/Cloth")
        before = cloth.get_particle_positions(state).numpy().copy()
        values = np.full((1, cloth.particles_per_deformable_object, 3), 17.0, dtype=np.float32)

        cloth.set_particle_positions(
            state,
            wp.array(values, dtype=wp.vec3, device=model.device),
            deformable_object_indices=wp.array([cloth.count], dtype=wp.int32, device=model.device),
        )

        np.testing.assert_array_equal(cloth.get_particle_positions(state).numpy(), before)

    def test_device_deformable_object_indices_must_be_one_dimensional(self):
        """Device index arrays fail clearly before reaching one-dimensional kernel inputs."""
        model = _replicated_model(3)
        state = model.state()
        cloth = DeformableSurfaceView(model, "/World/Cloth")
        values = wp.zeros((1, cloth.particles_per_deformable_object), dtype=wp.vec3, device=model.device)
        indices = wp.zeros((1, 2), dtype=wp.int32, device=model.device)

        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            cloth.set_particle_positions(state, values, deformable_object_indices=indices)
        with self.assertRaisesRegex(ValueError, "source_indices.*one-dimensional"):
            cloth.set_particle_positions(state, values, deformable_object_indices=[0], source_indices=indices)

    def test_device_selectors_validate_dtype_and_device(self):
        """Warp selectors must be int32 arrays on the view's device."""
        model = _replicated_model(3, device="cpu")
        state = model.state()
        cloth = DeformableSurfaceView(model, "/World/Cloth")
        values = wp.zeros((1, cloth.particles_per_deformable_object), dtype=wp.vec3, device="cpu")

        indices = wp.array([0], dtype=wp.int64, device="cpu")
        with self.assertRaisesRegex(ValueError, "deformable_object_indices dtype int32"):
            cloth.set_particle_positions(state, values, deformable_object_indices=indices)
        with self.assertRaisesRegex(ValueError, "source_indices dtype int32"):
            cloth.set_particle_positions(state, values, deformable_object_indices=[0], source_indices=indices)

        if wp.is_cuda_available():
            indices = wp.array([0], dtype=wp.int32, device="cuda:0")
            with self.assertRaisesRegex(ValueError, "deformable_object_indices on device cpu"):
                cloth.set_particle_positions(state, values, deformable_object_indices=indices)
            with self.assertRaisesRegex(ValueError, "source_indices on device cpu"):
                cloth.set_particle_positions(state, values, deformable_object_indices=[0], source_indices=indices)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA graph capture")
    def test_device_index_writes_capture_and_replay(self):
        """Captured indexed setters reuse changing device values without unsafe writes."""
        device = wp.get_device("cuda:0")
        model = _replicated_model(3, device=device)
        cloth = DeformableSurfaceView(model, "/World/Cloth")

        state = model.state()
        initial = cloth.get_particle_positions(state).numpy().copy()
        values_np = np.zeros((2, cloth.particles_per_deformable_object, 3), dtype=np.float32)
        values = wp.array(values_np, dtype=wp.vec3, device=device)
        indices = wp.array([cloth.count, cloth.count], dtype=wp.int32, device=device)

        # Compile every captured operation without changing state.
        cloth.set_particle_positions(state, values, deformable_object_indices=indices)

        indices.assign(np.array([1, 1], dtype=np.int32))
        values_np[0].fill(3.0)
        values_np[1].fill(8.0)
        values.assign(values_np)
        with wp.ScopedCapture(device) as capture:
            cloth.set_particle_positions(state, values, deformable_object_indices=indices)

        wp.capture_launch(capture.graph)
        expected = initial.copy()
        expected[1].fill(8.0)
        np.testing.assert_allclose(cloth.get_particle_positions(state).numpy(), expected, atol=1e-6)

        indices.assign(np.array([0, 2], dtype=np.int32))
        values_np[0].fill(4.0)
        values_np[1].fill(9.0)
        values.assign(values_np)
        wp.capture_launch(capture.graph)
        expected[0].fill(4.0)
        expected[2].fill(9.0)
        np.testing.assert_allclose(cloth.get_particle_positions(state).numpy(), expected, atol=1e-6)

        before_invalid = cloth.get_particle_positions(state).numpy().copy()
        indices.assign(np.array([-1, cloth.count], dtype=np.int32))
        values_np.fill(12.0)
        values.assign(values_np)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(cloth.get_particle_positions(state).numpy(), before_invalid)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA graph capture")
    def test_cable_device_writes_capture_and_replay(self):
        """Captured cable setters replay transform and spatial-vector device writes."""
        device = wp.get_device("cuda:0")
        model = _replicated_model(3, device=device)
        cable = DeformableCurveView(model, "/World/Cable")

        for name, getter, setter, dtype in (
            ("transforms", cable.get_body_transforms, cable.set_body_transforms, wp.transform),
            ("velocities", cable.get_body_velocities, cable.set_body_velocities, wp.spatial_vector),
        ):
            with self.subTest(values=name):
                state = model.state()
                initial = getter(state).numpy().copy()
                first = initial[[0]].copy()
                second = initial[[0]].copy()
                if dtype is wp.transform:
                    first[..., 0] += 1.25
                    second[..., 1] -= 2.5
                else:
                    first.fill(0.0)
                    first[..., 0] = 1.25
                    first[..., 3] = 2.5
                    second.fill(0.0)
                    second[..., 1] = -3.0
                    second[..., 4] = 4.0

                values = wp.array(first, dtype=dtype, device=device)
                indices = wp.array([cable.count], dtype=wp.int32, device=device)

                # Warm the captured operations with an ignored index.
                setter(state, values, deformable_object_indices=indices)

                indices.assign(np.array([1], dtype=np.int32))
                with wp.ScopedCapture(device) as capture:
                    setter(state, values, deformable_object_indices=indices)

                wp.capture_launch(capture.graph)
                expected = initial.copy()
                expected[1] = first[0]
                np.testing.assert_allclose(getter(state).numpy(), expected, atol=1e-6)

                indices.assign(np.array([2], dtype=np.int32))
                values.assign(second)
                wp.capture_launch(capture.graph)
                expected[2] = second[0]
                np.testing.assert_allclose(getter(state).numpy(), expected, atol=1e-6)

    def test_duplicate_device_object_index_uses_last_value(self):
        """Device duplicate indices are deterministic: the last row wins."""
        model = _replicated_model(3)
        state = model.state()
        cloth = DeformableSurfaceView(model, "/World/Cloth")
        before = cloth.get_particle_positions(state).numpy().copy()
        values = np.empty((2, cloth.particles_per_deformable_object, 3), dtype=np.float32)
        values[0].fill(3.0)
        values[1].fill(8.0)

        cloth.set_particle_positions(
            state,
            wp.array(values, dtype=wp.vec3, device=model.device),
            deformable_object_indices=wp.array([1, 1], dtype=wp.int32, device=model.device),
        )

        after = cloth.get_particle_positions(state).numpy()
        np.testing.assert_array_equal(after[[0, 2]], before[[0, 2]])
        np.testing.assert_array_equal(after[1], values[1])

    def test_single_object_per_world_uses_world_ids(self):
        """Environment IDs use the flat deformable object axis when each world has one match."""
        model = _replicated_model(3, device="cpu")
        state = model.state()
        cloth = DeformableSurfaceView(model, "/World/Cloth")
        before = cloth.get_particle_positions(state).numpy().copy()
        moved = before[[2]].copy()
        moved[..., 2] += 4.0

        cloth.set_particle_positions(
            state,
            wp.array(moved, dtype=wp.vec3, device=model.device),
            deformable_object_indices=wp.array([2], dtype=wp.int32, device=model.device),
        )

        after = cloth.get_particle_positions(state).numpy()
        np.testing.assert_array_equal(after[:2], before[:2])
        np.testing.assert_allclose(after[2], moved[0], atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
