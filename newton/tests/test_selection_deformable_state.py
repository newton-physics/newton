# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for batched deformable state access and masked writes."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.selection import (
    DeformableCurveView,
    DeformableSurfaceView,
)
from newton.tests._selection_deformable_test_utils import (
    _add_test_cable,
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


def _state_accessors(model):
    cloth = DeformableSurfaceView(model, "/World/Cloth")
    cable = DeformableCurveView(model, "/World/Cable")
    return (
        (cloth.get_particle_positions, cloth.set_particle_positions, "particle_q"),
        (cloth.get_particle_velocities, cloth.set_particle_velocities, "particle_qd"),
        (cable.get_body_transforms, cable.set_body_transforms, "body_q"),
        (cable.get_body_velocities, cable.set_body_velocities, "body_qd"),
    )


class TestDeformableStateWrites(unittest.TestCase):
    """Validate masked writes, capture, and gradients through public setters."""

    def test_indices_replace_mask_without_reordering_values(self):
        """Convert unsorted and repeated object indices into matching-row resets."""
        for device in wp.get_devices():
            model = _replicated_model(4, device=device)
            cloth = DeformableSurfaceView(model, "/World/Cloth")
            backing = wp.ones(8, dtype=wp.bool, device=device)
            mask = backing[::2]
            cloth.set_mask_from_indices(mask, [2, np.int64(0), 2])
            np.testing.assert_array_equal(backing.numpy(), [True, True, False, True, True, True, False, True])
            state = model.state()
            before = state.particle_q.numpy().copy()
            defaults = cloth.get_particle_positions(model).numpy().copy()
            defaults[:, :, 2] += np.array([10, 20, 30, 40])[:, None]
            cloth.set_particle_positions(state, defaults, mask=mask)
            expected = before.copy()
            expected[:4] = defaults[0]
            expected[8:12] = defaults[2]
            np.testing.assert_array_equal(state.particle_q.numpy(), expected)
            cloth.set_mask_from_indices(mask, [])
            np.testing.assert_array_equal(backing.numpy(), [False, True] * 4)

    def test_mask_conversion_validates_without_mutation(self):
        """Reject invalid host indices and array layouts before replacing a mask."""
        for device in wp.get_devices():
            model = _replicated_model(3, device=device)
            view = DeformableSurfaceView(model, "/World/Cloth")
            mask = wp.array([True, False, True], dtype=wp.bool, device=device)
            for indices, error in (
                ([-0.2], TypeError),
                ([1.9], TypeError),
                (["1"], TypeError),
                ([True], TypeError),
                ([np.bool_(False)], TypeError),
                (None, TypeError),
                ([-1], ValueError),
                ([3], ValueError),
                ([0, 3], ValueError),
                (wp.zeros((1, 3), dtype=wp.int32, device=device), ValueError),
                (wp.zeros(1, dtype=wp.int64, device=device), ValueError),
            ):
                with self.subTest(device=device, indices=indices):
                    with self.assertRaises(error):
                        view.set_mask_from_indices(mask, indices)
                    np.testing.assert_array_equal(mask.numpy(), [True, False, True])
            other = "cpu" if device.is_cuda else "cuda:0"
            if wp.is_cuda_available():
                with self.assertRaisesRegex(ValueError, "device"):
                    view.set_mask_from_indices(mask, wp.zeros(1, dtype=wp.int32, device=other))
                np.testing.assert_array_equal(mask.numpy(), [True, False, True])

            for output in (None, [True] * 3):
                with self.assertRaisesRegex(TypeError, "caller-owned"):
                    view.set_mask_from_indices(output, [0])
            for dtype, shape in ((wp.int32, (3,)), (wp.bool, (1, 3)), (wp.bool, (2,))):
                output = wp.ones(shape, dtype=dtype, device=device)
                before = output.numpy().copy()
                with self.assertRaisesRegex(ValueError, "mask"):
                    view.set_mask_from_indices(output, [0])
                np.testing.assert_array_equal(output.numpy(), before)
            overlapping = wp.array(ptr=mask.ptr, shape=(3,), strides=(0,), dtype=wp.bool, device=device)
            with self.assertRaisesRegex(ValueError, "overlap"):
                view.set_mask_from_indices(overlapping, [0])
            np.testing.assert_array_equal(mask.numpy(), [True, False, True])

    def test_mask_conversion_uses_view_rows_in_uneven_and_global_scenes(self):
        """Keep object membership independent of model-world partition sizes."""
        for device in wp.get_devices():
            for global_only in (False, True):
                builder = newton.ModelBuilder()
                for count in (1, 0, 2):
                    if not global_only:
                        builder.begin_world()
                    if count == 0:
                        builder.add_particle(wp.vec3(0.0), wp.vec3(0.0), 1.0)
                    for _ in range(count):
                        _add_test_cable(builder)
                    if not global_only:
                        builder.end_world()
                model = builder.finalize(device=device)
                cables = DeformableCurveView(model, "cable")
                self.assertEqual(cables.worlds, [-1, -1, -1] if global_only else [0, 2, 2])
                mask = wp.zeros(3, dtype=wp.bool, device=device)
                indices = range(1, 3) if global_only else range(*cables.deformable_object_ranges()[2])
                cables.set_mask_from_indices(mask, indices)
                np.testing.assert_array_equal(mask.numpy(), [False, True, True])

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA graph capture")
    def test_conversion_and_reset_replay_changing_membership(self):
        """Replay duplicate and padded device selectors without retaining old selections."""
        device = wp.get_device("cuda:0")
        model = _replicated_model(3, device=device)
        for getter, setter, attribute in _state_accessors(model):
            view = setter.__self__
            state = model.state()
            initial = getter(state).numpy().copy()
            defaults = initial.copy()
            defaults[..., 0] += 10.0
            values = wp.array(defaults, dtype=getattr(state, attribute).dtype, device=device)
            indices = wp.array([2, 0, 2, -1], dtype=wp.int32, device=device)
            mask = wp.zeros(3, dtype=wp.bool, device=device)
            view.set_mask_from_indices(mask, indices)
            setter(state, values, mask=mask)
            with wp.ScopedMempool(device, enable=False):
                with wp.ScopedCapture(device) as capture:
                    view.set_mask_from_indices(mask, indices)
                    setter(state, values, mask=mask)
            for selected, expected_mask in (
                ([2, 0, 2, -1], [True, False, True]),
                ([1, -1, 3, 1], [False, True, False]),
                ([-1, 3, -1, 99], [False] * 3),
            ):
                setter(state, initial)
                indices.assign(np.array(selected, dtype=np.int32))
                wp.capture_launch(capture.graph)
                expected = initial.copy()
                expected[expected_mask] = defaults[expected_mask]
                np.testing.assert_array_equal(mask.numpy(), expected_mask)
                np.testing.assert_array_equal(getter(state).numpy(), expected)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA graph capture")
    def test_conversion_rejects_host_indices_during_capture(self):
        """Leave a caller-owned mask intact when host conversion is attempted in capture."""
        device = wp.get_device("cuda:0")
        view = DeformableSurfaceView(_replicated_model(3, device=device), "/World/Cloth")
        mask = wp.ones(3, dtype=wp.bool, device=device)
        for pooled in (False, True):
            for indices in ([0], []):
                with wp.ScopedMempool(device, enable=pooled):
                    with wp.ScopedCapture(device):
                        with self.assertRaisesRegex(RuntimeError, "before CUDA capture"):
                            view.set_mask_from_indices(mask, indices)
                np.testing.assert_array_equal(mask.numpy(), [True] * 3)

    def test_masked_writes_keep_full_buffer_row_order(self):
        """Copy matching full-buffer rows and preserve every unselected row."""
        for device in wp.get_devices():
            model = _replicated_model(4, device=device)
            for getter, setter, attribute in _state_accessors(model):
                for selection in (None, [True, False, True, False], [False] * 4):
                    with self.subTest(device=device, setter=setter.__name__, selection=selection):
                        state = model.state()
                        before = getter(state).numpy().copy()
                        defaults = before.copy()
                        for i in range(4):
                            defaults[i, ..., 0] += 10.0 + i
                        values = wp.array(defaults, dtype=getattr(state, attribute).dtype, device=device)
                        expected = before.copy()
                        chosen = np.ones(4, dtype=bool) if selection is None else selection
                        expected[chosen] = defaults[chosen]

                        # Every second entry belongs to the mask. Other entries are sentinels.
                        backing_np = np.ones(8, dtype=bool)
                        backing_np[::2] = chosen
                        backing = wp.array(backing_np, dtype=wp.bool, device=device)
                        mask = None if selection is None else backing[::2]
                        setter(state, values, mask=mask)
                        np.testing.assert_array_equal(getter(state).numpy(), expected)
                        np.testing.assert_array_equal(values.numpy(), defaults)
                        np.testing.assert_array_equal(backing.numpy(), backing_np)

                        # Host inputs follow the same mapping outside capture.
                        state = model.state()
                        setter(state, defaults, mask=selection)
                        np.testing.assert_array_equal(getter(state).numpy(), expected)

    def test_masks_validate_before_writing(self):
        """Reject malformed masks without changing the destination."""
        model = _replicated_model(3, device="cpu")
        for getter, setter, attribute in _state_accessors(model):
            state = model.state()
            before = getattr(state, attribute).numpy().copy()
            values = wp.clone(getter(state))
            invalid = [
                [0, 2],
                [0, 1, 0],
                [True, 1, False],
                ["", "yes", ""],
                [True],
                [[True, False, True]],
                [],
                wp.zeros(3, dtype=wp.int32, device="cpu"),
                wp.zeros((1, 3), dtype=wp.bool, device="cpu"),
            ]
            if wp.is_cuda_available():
                invalid.append(wp.zeros(3, dtype=wp.bool, device="cuda:0"))
            for mask in invalid:
                with self.subTest(setter=setter.__name__, mask=mask):
                    with self.assertRaisesRegex(ValueError, "mask"):
                        setter(state, values, mask=mask)
                    np.testing.assert_array_equal(getattr(state, attribute).numpy(), before)

    def test_all_false_masks_still_validate_values(self):
        """Validate full values and reject aliases even when nothing is selected."""
        model = _replicated_model(3, device="cpu")
        for getter, setter, attribute in _state_accessors(model):
            state = model.state()
            before = getattr(state, attribute).numpy().copy()
            live = getter(state)
            invalid = [
                (wp.zeros((2, live.shape[1]), dtype=live.dtype, device="cpu"), "shape"),
                (wp.zeros(live.shape, dtype=float, device="cpu"), "dtype"),
                (live, "overlap.*clone"),
            ]
            if wp.is_cuda_available():
                invalid.append((wp.zeros(live.shape, dtype=live.dtype, device="cuda:0"), "device"))
            for values, message in invalid:
                with self.subTest(setter=setter.__name__, message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        setter(state, values, mask=[False] * 3)
                    np.testing.assert_array_equal(getattr(state, attribute).numpy(), before)
            with self.assertRaisesRegex(ValueError, "overlap.*clone"):
                setter(state, live)
            setter(state, wp.clone(live))

    def test_separate_masked_writes_preserve_gradients(self):
        """Keep both writes' gradients when one view updates separate states."""

        def differentiate(setter, states, values, masks, seeds, selectors):
            with wp.Tape() as tape:
                for i, (state, value, mask) in enumerate(zip(states, values, masks, strict=True)):
                    if selectors is not None:
                        setter.__self__.set_mask_from_indices(mask, selectors[i])
                    setter(state, value, mask=mask)
            tape.backward(grads=seeds)
            return tape

        for device in wp.get_devices():
            model = _replicated_model(3, device=device)
            for convert_indices in (False, True):
                for getter, setter, attribute in _state_accessors(model):
                    with self.subTest(device=device, setter=setter.__name__, convert_indices=convert_indices):
                        states = [model.state(requires_grad=True) for _ in range(2)]
                        targets = [getattr(state, attribute) for state in states]
                        values = [
                            wp.array(getter(state).numpy(), dtype=target.dtype, device=device, requires_grad=True)
                            for state, target in zip(states, targets, strict=True)
                        ]
                        masks_np = ([True, False, True], [False, True, True])
                        masks = [wp.array(mask, dtype=wp.bool, device=device) for mask in masks_np]
                        selectors = (
                            [wp.array(rows, dtype=wp.int32, device=device) for rows in ([2, 0, 2], [1, 2, 1])]
                            if convert_indices
                            else None
                        )
                        seeds = {target: wp.full_like(target, i + 1) for i, target in enumerate(targets)}
                        expected = [
                            np.broadcast_to((i + 1) * np.array(mask)[:, None, None], value.numpy().shape)
                            for i, (mask, value) in enumerate(zip(masks_np, values, strict=True))
                        ]
                        tape = differentiate(setter, states, values, masks, seeds, selectors)
                        for value, gradient in zip(values, expected, strict=True):
                            np.testing.assert_array_equal(value.grad.numpy(), gradient)

                        if device.is_cuda:
                            tape.zero()
                            with wp.ScopedCapture(device) as capture:
                                tape = differentiate(setter, states, values, masks, seeds, selectors)
                            for _ in range(2):
                                tape.zero()
                                wp.capture_launch(capture.graph)
                                for value, gradient in zip(values, expected, strict=True):
                                    np.testing.assert_array_equal(value.grad.numpy(), gradient)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA graph capture")
    def test_masked_setters_capture_without_allocations(self):
        """Replay all setter types with changing masks and gradient-enabled values."""
        device = wp.get_device("cuda:0")
        model = _replicated_model(3, device=device)
        for getter, setter, attribute in _state_accessors(model):
            for use_mask in (False, True):
                with self.subTest(setter=setter.__name__, use_mask=use_mask):
                    state = model.state(requires_grad=True)
                    initial = getter(state).numpy().copy()
                    values = wp.array(initial, dtype=getattr(state, attribute).dtype, device=device, requires_grad=True)
                    mask = wp.zeros(3, dtype=wp.bool, device=device) if use_mask else None
                    setter(state, values, mask=mask)
                    # No active tape: requires_grad alone must not trigger hidden storage.
                    with wp.ScopedMempool(device, enable=False):
                        with wp.ScopedCapture(device) as capture:
                            setter(state, values, mask=mask)
                    expected = initial.copy()
                    for step, chosen in enumerate(([True, False, True], [False] * 3, [True] * 3)):
                        next_values = initial.copy()
                        next_values[..., 0] += step + 1
                        values.assign(next_values)
                        if mask is not None:
                            mask.assign(np.array(chosen, dtype=bool))
                        wp.capture_launch(capture.graph)
                        rows = chosen if use_mask else [True] * 3
                        expected[rows] = next_values[rows]
                        np.testing.assert_array_equal(getter(state).numpy(), expected)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA graph capture")
    def test_host_inputs_rejected_during_capture(self):
        """Reject host conversions independently of CUDA memory-pool settings."""
        device = wp.get_device("cuda:0")
        model = _replicated_model(3, device=device)
        for getter, setter, attribute in _state_accessors(model):
            state = model.state()
            values = wp.clone(getter(state))
            host_values = values.numpy()
            before = getattr(state, attribute).numpy().copy()
            for pooled in (False, True):
                for value, mask in ((host_values, None), (values, [False] * 3), ([], None), (values, [])):
                    with self.subTest(setter=setter.__name__, pooled=pooled):
                        with wp.ScopedMempool(device, enable=pooled):
                            with wp.ScopedCapture(device):
                                with self.assertRaisesRegex(RuntimeError, "before CUDA capture"):
                                    setter(state, value, mask=mask)
                        np.testing.assert_array_equal(getattr(state, attribute).numpy(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
