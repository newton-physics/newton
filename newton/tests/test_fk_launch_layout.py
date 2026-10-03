# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check that public FK launch layout preserves its numerical and selection API."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.sim import articulation
from newton.tests.test_ik_fk_kernels import _randomize_joint_q
from newton.tests.test_kinematics import _build_dynamic_and_kinematic_single_joint_model
from newton.tests.unittest_utils import add_function_test, get_test_devices

BLOCKS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _mixed_model(device, *, requires_grad=True):
    """Mix branched and serial articulations in a batch that leaves partial blocks.

    Gradient-enabled models take the serial per-articulation FK launch on every
    device; ordinary CUDA tree models take the tiled route instead, which has a
    fixed block size and is covered by ``test_fk_layout_preserves_tiled_dispatch``.
    """
    builder = newton.ModelBuilder()
    for art in range(35):
        bodies, joints = [], []
        parents = (-1, 0, 1, 0, 3, 4) if art % 2 else (-1, 0, 1, 2, 3, 4)
        for index, kind in enumerate(("free", "revolute", "prismatic", "ball", "d6", "fixed")):
            body = builder.add_link(mass=1.0, com=wp.vec3(0.03, -0.02, 0.04), inertia=wp.mat33(np.eye(3)))
            options = {
                "parent": -1 if parents[index] < 0 else bodies[parents[index]],
                "child": body,
                "parent_xform": wp.transform(wp.vec3(0.2, -0.1, 0.3), wp.quat_rpy(0.1, -0.2, 0.3)),
                "child_xform": wp.transform(wp.vec3(-0.02, 0.04, 0.01), wp.quat_rpy(-0.1, 0.2, 0.05)),
            }
            if kind in ("revolute", "prismatic"):
                options["axis"] = newton.Axis.Y
            elif kind == "d6":
                axes = [newton.ModelBuilder.JointDofConfig(axis=axis) for axis in newton.Axis]
                options.update(linear_axes=axes, angular_axes=axes)
            joints.append(getattr(builder, f"add_joint_{kind}")(**options))
            bodies.append(body)
        builder.add_articulation(joints)
    model = builder.finalize(device=device, requires_grad=requires_grad)
    _randomize_joint_q(model, seed=17)
    model.joint_qd.assign(np.linspace(-0.7, 0.8, model.joint_dof_count, dtype=np.float32))
    return model


def _fk(model, state, block, **selection):
    """Exercise the public wrapper, overriding only the kernel launch block size."""
    launch = wp.launch

    def configured(*args, **kwargs):
        if kwargs.get("kernel") is not articulation.eval_articulation_fk:
            raise AssertionError("Unexpected kernel in public FK")
        return launch(*args, **dict(kwargs, block_dim=block))

    with (
        patch.object(wp, "launch_tiled", side_effect=AssertionError("Unexpected tiled FK dispatch")),
        patch.object(wp, "launch", side_effect=configured) as launched,
    ):
        newton.eval_fk(model, model.joint_q, model.joint_qd, state, **selection)
    # An empty selection launches nothing; otherwise the override must apply.
    expected_calls = 0 if "indices" in selection and len(selection["indices"]) == 0 else 1
    if launched.call_count != expected_calls:
        raise AssertionError(f"Expected {expected_calls} serial FK launch(es), got {launched.call_count}")


def _sentinel_state(model):
    state = model.state()
    # Route models built without gradients through the serial launch as well.
    state.body_q.requires_grad = True
    state.body_q.fill_(wp.transform(wp.vec3(7.0, -3.0, 2.0), wp.quat_identity()))
    state.body_qd.fill_(wp.spatial_vector(2.0, 3.0, 4.0, 5.0, 6.0, 7.0))
    return state


def test_fk_layout_selection(test, device):
    """Preserve exact mixed-tree poses, velocities, masks and indexed partial updates."""
    with wp.ScopedDevice(device):
        model = _mixed_model(device)
        inputs = (model.joint_q.numpy().copy(), model.joint_qd.numpy().copy())
        selected = np.arange(model.articulation_count) % 2 == 0
        choices = (
            ({}, np.ones(model.articulation_count, dtype=bool)),
            ({"mask": wp.array(selected, dtype=wp.bool, device=device)}, selected),
            (
                {"indices": wp.empty(0, dtype=wp.int32, device=device)},
                np.zeros(model.articulation_count, dtype=bool),
            ),
            (
                {"indices": wp.array([-1, 100, *np.flatnonzero(selected)[::-1]], dtype=wp.int32, device=device)},
                selected,
            ),
        )
        child_bodies = model.joint_child.numpy()
        joint_articulations = model.joint_articulation.numpy()
        for selection, active in choices:
            reference = _sentinel_state(model)
            untouched = (reference.body_q.numpy().copy(), reference.body_qd.numpy().copy())
            _fk(model, reference, 256, **selection)
            expected = (reference.body_q.numpy(), reference.body_qd.numpy())
            excluded = child_bodies[~active[joint_articulations]]
            for before, after in zip(untouched, expected, strict=True):
                np.testing.assert_array_equal(after[excluded], before[excluded])
            for block in BLOCKS:
                with test.subTest(block=block, selection=tuple(selection)):
                    state = _sentinel_state(model)
                    _fk(model, state, block, **selection)
                    np.testing.assert_array_equal(state.body_q.numpy(), expected[0])
                    np.testing.assert_array_equal(state.body_qd.numpy(), expected[1])
        np.testing.assert_array_equal(model.joint_q.numpy(), inputs[0])
        np.testing.assert_array_equal(model.joint_qd.numpy(), inputs[1])


def test_fk_layout_body_flags(test, device):
    """Retain unmatched dynamic or kinematic body state at every block size."""
    with wp.ScopedDevice(device):
        model = _build_dynamic_and_kinematic_single_joint_model(device)
        model.joint_q.assign(np.array([0.3, -0.4], dtype=np.float32))
        model.joint_qd.assign(np.array([0.7, -0.8], dtype=np.float32))
        for flag in (newton.BodyFlags.DYNAMIC, newton.BodyFlags.KINEMATIC):
            reference = _sentinel_state(model)
            before = (reference.body_q.numpy().copy(), reference.body_qd.numpy().copy())
            _fk(model, reference, 256, body_flag_filter=int(flag))
            excluded = (model.body_flags.numpy() & int(flag)) == 0
            expected = (reference.body_q.numpy(), reference.body_qd.numpy())
            for initial, final in zip(before, expected, strict=True):
                np.testing.assert_array_equal(final[excluded], initial[excluded])
            for block in BLOCKS:
                state = _sentinel_state(model)
                _fk(model, state, block, body_flag_filter=int(flag))
                np.testing.assert_array_equal(state.body_q.numpy(), expected[0])
                np.testing.assert_array_equal(state.body_qd.numpy(), expected[1])


def test_fk_layout_gradients(test, device):
    """Preserve position and velocity adjoints across independent-articulation layouts."""
    with wp.ScopedDevice(device):
        model = _mixed_model(device, requires_grad=True)
        reference = None
        for block in (256, *BLOCKS[:-1]):
            state = model.state()
            with wp.Tape() as tape:
                _fk(model, state, block)
            tape.backward(grads={state.body_q: wp.ones_like(state.body_q), state.body_qd: wp.ones_like(state.body_qd)})
            gradients = (model.joint_q.grad.numpy().copy(), model.joint_qd.grad.numpy().copy())
            if reference is None:
                reference = gradients
                for gradient in gradients:
                    test.assertTrue(np.isfinite(gradient).all())
                    test.assertGreater(float(np.max(np.abs(gradient))), 0.01)
            else:
                for actual, expected in zip(gradients, reference, strict=True):
                    np.testing.assert_array_equal(actual, expected)
            tape.zero()


def test_fk_layout_graph_replay(test, device):
    """Replay captured FK layouts after changing joint state without recapturing."""
    if not device.is_cuda:
        test.skipTest("CUDA graphs require CUDA")
    with wp.ScopedDevice(device):
        model = _mixed_model(device)
        states, graphs = [], []
        mask = wp.array(np.arange(model.articulation_count) % 2 == 0, dtype=wp.bool, device=device)
        for block in BLOCKS:
            state = _sentinel_state(model)
            _fk(model, state, block, mask=mask)
            with wp.ScopedCapture(device=device) as capture:
                _fk(model, state, block, mask=mask)
            states.append(state)
            graphs.append(capture.graph)
        _randomize_joint_q(model, seed=29)
        model.joint_qd.fill_(0.35)
        reference = _sentinel_state(model)
        _fk(model, reference, 256, mask=mask)
        for state, graph in zip(states, graphs, strict=True):
            wp.capture_launch(graph)
            np.testing.assert_array_equal(state.body_q.numpy(), reference.body_q.numpy())
            np.testing.assert_array_equal(state.body_qd.numpy(), reference.body_qd.numpy())


def test_fk_layout_preserves_tiled_dispatch(test, device):
    """Keep ordinary CUDA tree models on the tiled FK route."""
    if not device.is_cuda:
        test.skipTest("Tiled FK dispatch requires CUDA")
    with wp.ScopedDevice(device):
        model = _mixed_model(device, requires_grad=False)
        state = model.state()
        with (
            patch.object(articulation, "_fk_block_dim", side_effect=AssertionError("Unexpected serial dispatch")),
            patch.object(wp, "launch_tiled", wraps=wp.launch_tiled) as launch,
        ):
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        launch.assert_called_once()
        test.assertTrue(np.isfinite(state.body_q.numpy()).all())
        test.assertTrue(np.isfinite(state.body_qd.numpy()).all())


class TestFKLaunchLayout(unittest.TestCase):
    def test_launch_policy_power_of_two_boundaries(self):
        """Round down at batch-size thresholds and retain CPU, singleton and cap behavior."""
        for count, expected in ((0, 256), (1, 256), (2, 1), (255, 1), (256, 1), (511, 1), (512, 2)):
            with self.subTest(count=count, sm_count=128):
                self.assertEqual(articulation._fk_block_dim(count, 128, 8), expected)
        for sm_count in (-1, 0):
            for count in (0, 1, 4096, 1 << 30):
                self.assertEqual(articulation._fk_block_dim(count, sm_count, 8), 256)
        for sm_count in (8, 128, 152, 188, 256):
            for block in (2, 4, 8, 16):
                threshold = 2 * sm_count * block
                with self.subTest(sm_count=sm_count, threshold=threshold):
                    self.assertEqual(articulation._fk_block_dim(threshold - 1, sm_count, 8), block // 2)
                    self.assertEqual(articulation._fk_block_dim(threshold, sm_count, 8), block)
                    self.assertEqual(articulation._fk_block_dim(threshold + 1, sm_count, 8), block)
            self.assertEqual(articulation._fk_block_dim(1 << 30, sm_count, 8), 16)
        self.assertEqual(articulation._fk_block_dim(4096, 152, 8), 8)
        self.assertEqual(articulation._fk_block_dim(4096, 188, 8), 8)

    def test_launch_policy_retains_packed_tiny_articulations(self):
        """Retain the previous layout through four joints and switch policy at five."""
        for max_joints in (0, 1, 2, 3, 4):
            for sm_count in (8, 128, 256):
                for count in (0, 1, 2, 16, 17, 4096, 1 << 30):
                    with self.subTest(max_joints=max_joints, sm_count=sm_count, count=count):
                        expected = 16 if count > 16 else 256
                        self.assertEqual(articulation._fk_block_dim(count, sm_count, max_joints), expected)
            self.assertEqual(articulation._fk_block_dim(4096, 0, max_joints), 256)
        for count, expected in ((0, 256), (1, 256), (2, 1), (16, 1), (17, 1), (512, 2), (65536, 16)):
            with self.subTest(max_joints=5, count=count):
                self.assertEqual(articulation._fk_block_dim(count, 128, 5), expected)

    def test_launch_policy_scales_with_device_parallelism(self):
        """Expose more independent blocks for the same batch on a larger GPU."""
        model = _mixed_model("cpu")
        state = model.state()
        blocks = []
        for sm_count in (8, 256):
            selected_model = SimpleNamespace(**vars(model))
            selected_model.articulation_count = 4096
            selected_model.device = SimpleNamespace(is_cuda=True, sm_count=sm_count)
            # Synthetic launch metadata only: no kernel accesses these CPU arrays.
            # The gradient-enabled model selects the serial launch, which owns
            # the per-articulation layout; a tiled dispatch would bypass it.
            with (
                patch.object(wp, "launch_tiled", side_effect=AssertionError("Unexpected tiled FK dispatch")),
                patch.object(wp, "launch") as launch,
            ):
                newton.eval_fk(selected_model, model.joint_q, model.joint_qd, state)
            launch.assert_called_once()
            self.assertEqual(launch.call_args.kwargs["dim"], 4096)
            blocks.append(launch.call_args.kwargs.get("block_dim", 256))
        self.assertLess(blocks[1], blocks[0])

    def test_launch_policy_uses_selected_articulation_count(self):
        """Select layouts deterministically from host metadata without reading array contents."""
        model = _mixed_model("cpu")
        state = model.state()
        choices = [
            ({}, 35),
            ({"mask": wp.ones(35, dtype=wp.bool, device="cpu")}, 35),
            ({"mask": wp.zeros(35, dtype=wp.bool, device="cpu")}, 35),
        ]
        choices.extend(
            ({"indices": wp.array(np.arange(count), dtype=wp.int32, device="cpu")}, count)
            for count in (0, 1, 16, 17, 35)
        )
        for sm_count in (0, 8, 80, 148, 192):
            # Only host dispatch is inspected; all arrays remain on CPU and no
            # kernel launches against this stand-in device.
            selected_model = SimpleNamespace(**vars(model))
            selected_model.device = SimpleNamespace(is_cuda=sm_count > 0, sm_count=sm_count)
            observed = {}
            for selection, count in choices:
                with self.subTest(sm_count=sm_count, selection=tuple(selection), count=count):
                    with (
                        patch.object(wp.array, "numpy", side_effect=AssertionError("Unexpected array readback")),
                        patch.object(wp, "Event", side_effect=AssertionError("Unexpected runtime timing")),
                        patch.object(wp, "capture_launch", side_effect=AssertionError("Unexpected runtime profiling")),
                        patch.object(
                            wp, "synchronize_device", side_effect=AssertionError("Unexpected synchronization")
                        ),
                        patch.object(wp, "launch_tiled", side_effect=AssertionError("Unexpected tiled FK dispatch")),
                        patch.object(wp, "launch") as launch,
                    ):
                        for _ in range(2):
                            newton.eval_fk(selected_model, model.joint_q, model.joint_qd, state, **selection)
                        # An indexed subset must choose the same layout as a
                        # full launch of that size, regardless of the loaded count.
                        same_count_model = SimpleNamespace(**vars(selected_model))
                        same_count_model.articulation_count = count
                        newton.eval_fk(same_count_model, model.joint_q, model.joint_qd, state)
                    if count == 0:
                        # An empty index array selects no articulations, so
                        # eval_fk returns before any launch: a zero-dim launch
                        # would do no work and only add host overhead.
                        launch.assert_not_called()
                        continue
                    self.assertEqual(launch.call_count, 3)
                    for call in launch.call_args_list:
                        self.assertEqual(call.kwargs["dim"], count)
                        block = call.kwargs.get("block_dim", 256)
                        self.assertIn(block, BLOCKS)
                        if sm_count > 0 and count > 1:
                            # Independent of the policy helper: cap the serial
                            # block and expose at least two blocks per SM.
                            self.assertLessEqual(block, 16)
                            self.assertGreaterEqual((count + block - 1) // block, min(count, 2 * sm_count))
                        self.assertEqual(
                            block, articulation._fk_block_dim(count, sm_count, model.max_joints_per_articulation)
                        )
                        if sm_count == 0:
                            self.assertEqual(block, 256)
                        self.assertEqual(block, observed.setdefault(count, block))


for function in (
    test_fk_layout_selection,
    test_fk_layout_body_flags,
    test_fk_layout_gradients,
    test_fk_layout_graph_replay,
    test_fk_layout_preserves_tiled_dispatch,
):
    add_function_test(TestFKLaunchLayout, function.__name__, function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
