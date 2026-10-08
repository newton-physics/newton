# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Qualify inline friction against the unfused shoe law without external assets."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from projects.digital_shoe.friction_adapter import FrictionAdapter
from projects.digital_shoe.friction_parameter_adapter import FrictionParameterAdapter
from projects.digital_shoe.runtime import FoundationConfig, MidsoleFoundation, ShoeMaterial, SurroundConfig
from projects.impedance_instron.cartesian.gpu.foundation import FoundationFused

_DT = 0.00025
_METHODS = (0, 1, 4, 5, 6, 7, 8, 9)
_NORMAL = """z_free q_state peq_prev compression base_pressure column_pressed contact_point
normal_force cop_moment active max_compression pressed_force partial_cop partial_normal
partial_pressed partial_max partial_active surround_compression surround_scratch
surround_previous surround_rate""".split()
_TANGENTIAL = """column_force ground_force tangent_anchor tangent_stuck tangent_dwell
tangent_deflection tangent_maxwell_force resultant_force resultant_moment_origin contact_power
partial_force partial_torque partial_moment partial_power""".split()
_ADAPTER = """deflection sliding_distance maxwell_force stored_energy column_diagnostics
step_diagnostics partial_diagnostics totals scratch_anchor scratch_stuck scratch_dwell""".split()


class _ParameterSubclass(FrictionParameterAdapter):
    """Require delegation even when an adapter inherits all parameter-adapter behavior."""


def _parameters(methods):
    rows = np.tile([0, 0.5, 0.7, 0.6, 0.15, 2 * _DT, 0, 0.25, 0.12, 5e4, 0.0002, 0.004], (len(methods), 1))
    w = np.arange(len(methods))
    rows[:, 0] = methods
    rows[:, 1:4] += w[:, None] * [0.04, 0.1, 0.1]
    rows[:, 4] = np.where(np.isin(methods, (7, 8)), 0.0, 0.15)
    rows[:, 7] += 0.02 * w
    rows[:, 11] += 0.001 * w
    return rows.astype(np.float32)


def _motion(pair, height=0.012, speed=0.4, frame=0):
    for foundation, state in pair:
        w = np.arange(foundation.world_count)
        poses = np.zeros((len(w), 7), dtype=np.float32)
        poses[:, :3] = np.stack([0.02 * w + frame * 0.00015, -0.003 * w, height + 0.0003 * w], axis=1)
        poses[:, 5:7] = np.stack([np.sin(0.04 + 0.005 * w), np.cos(0.04 + 0.005 * w)], axis=1)
        velocity = np.tile([speed, -0.04, 0.003, 0.5, -0.3, 1.1], (len(w), 1)).astype(np.float32)
        velocity[:, 0] *= 1 + 0.1 * w
        state.body_q.assign(poses)
        state.body_qd.assign(velocity)


def _pair(
    columns,
    device,
    methods=_METHODS,
    *,
    surround=True,
    sweeps=3,
    plane=True,
    adapter_type=FrictionParameterAdapter,
    configured_law=None,
):
    """Build distinct material/settings worlds and a chain spanning columns 511 and 512."""
    rng = np.random.default_rng(columns)
    worlds = len(methods)
    anchors = np.zeros((columns, 3))
    anchors[:, 0] = (np.arange(columns) % 32 - 16) * 0.0004
    anchors[:, 1] = (np.arange(columns) // 32 - 16) * 0.0004
    anchors[:, 2] = -0.02
    rest, area = rng.uniform(0.019, 0.023, columns), rng.uniform(0.8e-5, 1.2e-5, columns)
    neighbors = np.full((columns, 4), -1, dtype=np.int32)
    neighbors[1:, 0], neighbors[:-1, 1] = np.arange(columns - 1), np.arange(1, columns)
    driven = np.arange(columns) % 7 == 0
    if columns > 511:
        driven[511] = True
    material = ShoeMaterial(
        74000.0, 0.22, 0.7, 1500.0, instantaneous_shear_modulus_2_pa=18000.0, hyperfoam_exponent_2=3.1
    )
    config = FoundationConfig(
        ground_height_m=0.0 if plane else None,
        friction_model=configured_law or "legacy",
        friction_stiffness=10000.0,
        friction=8.0,
        mu=0.8,
        normal_damping=0.15,
    )
    passive = SurroundConfig(driven=driven, sweeps=sweeps, carrier_bond=True) if surround else None
    materials = [
        replace(material, equilibrium_fraction=0.4 + 0.05 * w, maxwell_relaxation_time_s=0.02 + 0.003 * w)
        for w in range(worlds)
    ]
    pair = []
    for cls in (MidsoleFoundation, FoundationFused):
        state = SimpleNamespace(
            body_q=wp.zeros(worlds, dtype=wp.transform, device=device),
            body_qd=wp.zeros(worlds, dtype=wp.spatial_vector, device=device),
            body_f=wp.zeros(worlds, dtype=wp.spatial_vector, device=device),
        )
        com = wp.array([[0.002, -0.001, 0.004]] * worlds, dtype=wp.vec3, device=device)
        geometry = (anchors, np.zeros(columns), rest, area, neighbors, 0.005)
        foundation = cls(*geometry, material, np.arange(worlds), com, config, device, passive, world_count=worlds)
        foundation.set_world_materials(materials)
        if configured_law is not None:
            assert foundation.friction_solver.is_default
        elif adapter_type is FrictionAdapter:
            adapter_type(foundation, wp.zeros(worlds, dtype=wp.spatial_matrix, device=device), mode="deflection")
        else:
            adapter_type(foundation, worlds, initial_parameters=_parameters(methods))
        pair.append((foundation, state))
    _motion(pair)
    return pair


def _snapshot(foundation, state):
    """Flatten each world's persistent state, retaining world boundaries for isolation checks."""
    arrays = {
        name: getattr(foundation, name)
        for name in _NORMAL + _TANGENTIAL
        if not name.startswith("surround_") or foundation.surround is not None
    }
    adapter = foundation.friction_solver
    arrays.update({"adapter." + name: getattr(adapter, name) for name in _ADAPTER if hasattr(adapter, name)})
    arrays["body_f"] = state.body_f
    result = {name: arr.numpy().reshape(foundation.world_count, -1) for name, arr in arrays.items()}
    for name in ("column_force", "ground_force"):
        result[name + "_z"] = result[name][:, 2::3]
    return result


class TestDigitalShoeFrictionFullFusion(unittest.TestCase):
    """Protect fused execution, physical parity, and graph-resident mutable settings."""

    @classmethod
    def setUpClass(cls):
        """Initialize Warp for CPU and optional CUDA coverage."""
        wp.init()

    def _cuda(self):
        if not wp.is_cuda_available():
            self.skipTest("CUDA is required for full fusion")
        return wp.get_device("cuda:0")

    def _assert_pair(self, pair, world=None):
        reference, actual = [_snapshot(*entry) for entry in pair]
        self.assertEqual(reference.keys(), actual.keys())
        for name in reference:
            a, b = actual[name], reference[name]
            if world is not None:
                a, b = a[world], b[world]
            if name in _NORMAL or name.endswith("_z") or name == "tangent_stuck":
                np.testing.assert_array_equal(a, b, err_msg=name)
            else:
                atol = 1e-9 if "tangent_" in name or name.startswith("adapter.") else 1e-7
                np.testing.assert_allclose(a, b, rtol=2e-6, atol=atol, err_msg=name)

    def _observed_apply(self, foundation, state, full=True, **kwargs):
        adapter = foundation.friction_solver
        with (
            patch.object(wp, "launch", wraps=wp.launch) as launch,
            patch.object(wp, "launch_tiled", wraps=wp.launch_tiled) as tiled,
            patch.object(adapter, "apply", wraps=adapter.apply) as delegate,
        ):
            foundation.apply(state, _DT, **{"clear_body_force": True, **kwargs})

        def keys(mock):
            return [(call.args[0] if call.args else call.kwargs["kernel"]).key for call in mock.call_args_list]

        if full:
            delegate.assert_not_called()
            expected = ["_apply_contact_world", "_reduce_contact_world"]
            if foundation.free_column_count:
                expected.insert(0, "_surround_fused")
            self.assertEqual(keys(tiled), expected)
            # Warp versions differ in whether launch_tiled calls the public launch alias.
            self.assertIn(keys(launch), ([], expected))
        else:
            delegate.assert_called_once_with(state, _DT)
            self.assertNotIn("_apply_world", keys(launch) + keys(tiled))
            self.assertNotIn("_apply_contact_world", keys(launch) + keys(tiled))

    def test_all_methods_load_unload_recontact_and_reversal(self):
        """Match all histories and reductions through short dropouts, release, and reversed slip."""
        device = self._cuda()
        trajectory = [(0.015, 0.4), (0.012, 0.8), (0.032, 0.8), (0.012, 0.8)]
        trajectory += [(0.032, 0.0)] * 3 + [(0.012, -1.2), (0.014, -1.2)]
        for columns in (31, 513):
            with self.subTest(columns=columns):
                pair = _pair(columns, device)
                for frame, (height, speed) in enumerate(trajectory):
                    _motion(pair, height, speed, frame)
                    for foundation, state in pair:
                        foundation.apply(state, _DT, clear_body_force=True)
                    self._assert_pair(pair)
                    if height > 0.02:
                        np.testing.assert_array_equal(pair[1][0].ground_force.numpy(), 0.0)
                    else:
                        self.assertGreater(np.linalg.norm(pair[1][1].body_f.numpy()[:, :2]), 0.0)
                if columns > 512:
                    self.assertTrue(np.all(pair[1][0].surround_compression.numpy().reshape(8, columns)[:, 512] > 0))
                self._observed_apply(*pair[1])

    def test_boundaries_and_large_bed_fallback(self):
        """Use fused stages through 1024 columns and preserve delegation at 1025 columns."""
        device = self._cuda()
        for columns in (1, 31, 511, 512, 513, 1024, 1025):
            with self.subTest(columns=columns):
                pair = _pair(columns, device, (7, 8, 9), sweeps=2 + columns % 2)
                for foundation, state in pair:
                    foundation.apply(state, _DT, clear_body_force=True)
                pair[0][0].apply(pair[0][1], _DT, clear_body_force=True)
                self._observed_apply(*pair[1], full=columns <= 1024)
                self._assert_pair(pair)

    def test_no_surround(self):
        """Fuse every supported law when no passive-surround configuration exists."""
        pair = _pair(31, self._cuda(), surround=False)
        for foundation, state in pair:
            foundation.apply(state, _DT, clear_body_force=True)
        pair[0][0].apply(pair[0][1], _DT, clear_body_force=True)
        self._observed_apply(*pair[1])
        self._assert_pair(pair)

    def test_configured_defaults_and_material_refresh(self):
        """Fuse automatically installed laws and honor refreshed material-derived parameters."""
        device = self._cuda()
        for law, method in (("elastic_coulomb", 9), ("maxwell", 7), ("column_maxwell", 8)):
            with self.subTest(law=law):
                pair = _pair(513, device, (method,) * 3, configured_law=law)
                for foundation, state in pair:
                    foundation.apply(state, _DT, clear_body_force=True)
                    foundation.set_world_material(
                        1,
                        ShoeMaterial(96000.0, 0.22, 0.6, 1500.0, maxwell_relaxation_time_s=0.007),
                    )
                    foundation.reset()
                    foundation.apply(state, _DT, clear_body_force=True)
                self._assert_pair(pair)
                pair[0][0].apply(pair[0][1], _DT, clear_body_force=True)
                self._observed_apply(*pair[1])
                self._assert_pair(pair)

    def test_cpu_fallback(self):
        """Delegate all methods on CPU while preserving normal and tangential state."""
        pair = _pair(31, wp.get_device("cpu"))
        for foundation, state in pair:
            foundation.apply(state, _DT, clear_body_force=True)
        pair[0][0].apply(pair[0][1], _DT, clear_body_force=True)
        self._observed_apply(*pair[1], full=False)
        self._assert_pair(pair)

    def test_generic_additive_and_custom_adapter_fallbacks(self):
        """Honor generic contact, preexisting wrenches, and non-exact adapter types."""
        device = self._cuda()
        for plane, clear, adapter_type in (
            (False, True, FrictionParameterAdapter),
            (True, False, FrictionParameterAdapter),
            (True, True, _ParameterSubclass),
            (True, True, FrictionAdapter),
        ):
            with self.subTest(plane=plane, clear=clear, adapter=adapter_type.__name__):
                pair = _pair(31, device, (7, 9), plane=plane, adapter_type=adapter_type)
                for foundation, state in pair:
                    foundation.apply(state, _DT, clear_body_force=True)
                    state.body_f.fill_(wp.spatial_vector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
                pair[0][0].apply(pair[0][1], _DT, clear_body_force=clear)
                self._observed_apply(*pair[1], full=False, clear_body_force=clear)
                self._assert_pair(pair)
                if not clear:
                    expected = pair[1][0].resultant_force.numpy() + np.array([1, 2, 3])
                    np.testing.assert_allclose(pair[1][1].body_f.numpy()[:, :3], expected, rtol=2e-6, atol=1e-7)

    def test_graph_reset_replay_and_settings_without_recapture(self):
        """Replay captured resets and read changed methods and coefficients from resident settings."""
        device = self._cuda()
        pair = _pair(513, device)
        graphs = []
        for foundation, state in pair:
            foundation.apply(state, _DT, clear_body_force=True)
            with wp.ScopedCapture(device=device) as capture:
                foundation.reset()
                for _ in range(3):
                    foundation.apply(state, _DT, clear_body_force=True)
            graphs.append(capture.graph)
        pointers = [foundation.friction_solver.settings.ptr for foundation, _ in pair]
        previous = None
        for methods in (_METHODS, tuple(np.roll(_METHODS, 1))):
            params = _parameters(methods)
            if previous is not None:
                params[:, 1] *= 0.7
                params[:, 7] *= 0.7
                params[:, 2:4] *= 1.4
            for foundation, _ in pair:
                foundation.friction_solver.set_parameters(params)
            first = None
            for _ in range(2):
                for graph in graphs:
                    wp.capture_launch(graph)
                self._assert_pair(pair)
                current = _snapshot(*pair[1])
                if first is not None:
                    for name in current:
                        np.testing.assert_array_equal(current[name], first[name], err_msg=name)
                first = current
            if previous is not None:
                self.assertFalse(np.array_equal(current["body_f"], previous["body_f"]))
            previous = current
        self.assertEqual(pointers, [foundation.friction_solver.settings.ptr for foundation, _ in pair])

    def test_diagnostic_clock_and_disabled_worlds(self):
        """Tick once even with world zero disabled and leave disabled histories and diagnostics untouched."""
        device = self._cuda()
        pair = _pair(513, device, (7, 8, 9))
        fused, state = pair[1]
        groups = 32  # Match Engine's fixed allocation, including unused groups.
        rest = fused.rest_len.numpy().astype(np.float64)
        maxima = wp.zeros((3, groups), dtype=wp.vec2d, device=device)
        caps = wp.zeros((3, groups), dtype=int, device=device)
        invalid = wp.zeros_like(caps)
        clock = wp.zeros(1, dtype=int, device=device)
        fused.diagnostics = (wp.array(rest, dtype=wp.float64, device=device), 0.01, maxima, caps, invalid, clock)
        self.assertTrue(fused.fused_diagnostics)
        maxima.fill_(wp.vec2d(99.0, 99.0))
        caps.fill_(99)
        invalid.fill_(1)
        for foundation, body_state in pair:
            foundation.apply(body_state, _DT, clear_body_force=True)
        before = _snapshot(fused, state)
        diagnostic_before = [arr.numpy() for arr in (maxima, caps, invalid)]
        fused.enabled.assign(np.array([0, 1, 0], dtype=np.int32))
        _motion(pair, 0.01, -0.9, 5)
        for tick in (False, True):
            pair[0][0].apply(pair[0][1], _DT, clear_body_force=True)
            self._observed_apply(fused, state, tick=tick)
            self._assert_pair(pair, world=1)
            self.assertEqual(clock.numpy()[0], int(tick))
        for name, value in _snapshot(fused, state).items():
            np.testing.assert_array_equal(value[[0, 2]], before[name][[0, 2]], err_msg=name)
        for arr, old in zip((maxima, caps, invalid), diagnostic_before, strict=True):
            np.testing.assert_array_equal(arr.numpy()[[0, 2]], old[[0, 2]])
        fraction = fused.compression.numpy().reshape(3, -1).astype(np.float64) / rest
        driven = fused.driven.numpy() != 0
        expected = np.stack(
            [np.where(driven, fraction, 0).max(axis=1), np.where(~driven, fraction, 0).max(axis=1)], axis=1
        )
        np.testing.assert_array_equal(maxima.numpy().max(axis=1), expected)
        np.testing.assert_array_equal(caps.numpy().sum(axis=1), ((fraction >= 0.01 - 1e-6) & ~driven).sum(axis=1))
        np.testing.assert_array_equal(invalid.numpy(), 0)
        self.assertGreater(caps.numpy().sum(), 0)


if __name__ == "__main__":
    unittest.main()
