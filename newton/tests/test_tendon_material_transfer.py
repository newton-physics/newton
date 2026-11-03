# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check packed linear material projection with current guide routes."""

import unittest
from functools import partial

import numpy as np
import warp as wp

import newton
from newton._src.solvers.tendon_material import TendonMaterialStatus
from newton._src.solvers.tendon_material_state import (
    _create_tendon_material_workspace,
    _project_tendon_material,
    _TendonMaterialWorkspace,
)
from newton.tests.test_tendon_capstan import _loaded_route_arc_length, build_stiff_pinhole_capstan
from newton.tests.test_tendon_material_direct import _reference_taut
from newton.tests.unittest_utils import NewtonTestCase, add_function_test, get_test_devices


@wp.kernel(enable_backward=False)
def _project_routes(
    workspace: _TendonMaterialWorkspace,
    starts: wp.array[int],
    kinds: wp.array[int],
    guide_active: wp.array[bool],
    cone_l: wp.array[int],
    cone_r: wp.array[int],
    caps: wp.array[float],
    segment_active: wp.array[int],
    raw_compliance: wp.array[float],
    compliance: wp.array[float],
    damping: wp.array[float],
    length: wp.array[float],
    rolling_l: wp.array[float],
    rolling_r: wp.array[float],
    rolling: int,
    pinhole: int,
    stretch: wp.array[float],
    transfer: wp.array[float],
    accepted: wp.array[bool],
):
    tendon = wp.tid()
    accepted[tendon] = _project_tendon_material(
        workspace,
        tendon,
        starts[tendon],
        starts[tendon + 1],
        starts[tendon] - tendon,
        starts[tendon + 1] - tendon - 1,
        kinds,
        guide_active,
        cone_l,
        cone_r,
        caps,
        segment_active,
        raw_compliance,
        compliance,
        damping,
        length,
        rolling_l,
        rolling_r,
        rolling,
        pinhole,
        1.0e-6,
        stretch,
        transfer,
    )


class _MaterialFixture:
    """Project real solver arrays at prescribed, frozen material trials."""

    def __init__(self, device, kinds, *, copies=1):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for copy in range(copies):
            guides = []
            for i, kind in enumerate(kinds):
                guides.append(
                    newton.TendonGuide(
                        body=-1,
                        offset=(float(copy), 0.0, float(i)),
                        guide_type=kind,
                        axis=(0.0, 1.0, 0.0),
                        radius=0.1 if kind == newton.TendonGuideType.ROLLER else 0.0,
                        mu=0.3,
                        **({"compliance": 0.001, "rest_length": 0.1} if i else {}),
                    )
                )
            builder.add_tendon(guides)
        self.model = builder.finalize(device=device)
        self.solver = newton.solvers.SolverXPBD(self.model, iterations=1)
        self.workspace = _create_tendon_material_workspace(self.model)
        self.transfer = wp.full(self.model.tendon_guide_count, -77.0, dtype=float, device=device)
        self.accepted = wp.zeros(copies, dtype=bool, device=device)
        solver = self.solver
        solver.tendon_segment_length.fill_(0.1)
        solver.tendon_segment_active.fill_(1)
        solver.tendon_segment_active_compliance.fill_(0.001)
        solver.tendon_segment_damping_tension.zero_()
        solver.tendon_segment_rolling_delta_l.zero_()
        solver.tendon_segment_rolling_delta_r.zero_()
        solver.tendon_guide_active.fill_(True)
        solver.tendon_guide_cap_ratio.fill_(2.0)
        n = len(kinds) - 1
        cone_l = np.full(copies * (n + 1), -1, dtype=np.int32)
        cone_r = cone_l.copy()
        for copy in range(copies):
            for guide in range(1, n):
                if kinds[guide] != newton.TendonGuideType.ANCHOR:
                    cone_l[copy * (n + 1) + guide] = copy * n + guide - 1
                    cone_r[copy * (n + 1) + guide] = copy * n + guide
        solver.tendon_guide_cone_seg_l.assign(cone_l)
        solver.tendon_guide_cone_seg_r.assign(cone_r)

    def set_extension(self, extension):
        self.solver.tendon_segment_stretch.assign(np.asarray(extension, dtype=np.float32))

    def set_motion(self, differential, common=0.0):
        self.solver.tendon_segment_rolling_delta_l.assign(np.array([0.0, common - differential], dtype=np.float32))
        self.solver.tendon_segment_rolling_delta_r.assign(np.array([common + differential, 0.0], dtype=np.float32))

    def launch(self, *, rolling=True, pinhole=True):
        model, solver = self.model, self.solver
        wp.launch(
            _project_routes,
            dim=model.tendon_count,
            inputs=[
                self.workspace,
                model.tendon_start,
                model.tendon_guide_type,
                solver.tendon_guide_active,
                solver.tendon_guide_cone_seg_l,
                solver.tendon_guide_cone_seg_r,
                solver.tendon_guide_cap_ratio,
                solver.tendon_segment_active,
                model.tendon_segment_compliance,
                solver.tendon_segment_active_compliance,
                solver.tendon_segment_damping_tension,
                solver.tendon_segment_length,
                solver.tendon_segment_rolling_delta_l,
                solver.tendon_segment_rolling_delta_r,
                int(rolling),
                int(pinhole),
                solver.tendon_segment_stretch,
                self.transfer,
                self.accepted,
            ],
            device=model.device,
        )


def test_rolling_trial_and_slide(test, device):
    """Project both slip directions after full rolling and signed arc exchange."""
    a, r = newton.TendonGuideType.ANCHOR, newton.TendonGuideType.ROLLER
    fixture = _MaterialFixture(device, [a, r, a])
    for differential, common in ((0.001, 0.0), (-0.008, 0.0), (0.008, 0.0), (0.11, 0.0), (0.001, 0.0005)):
        with test.subTest(differential=differential, common=common):
            initial = np.array([0.01, 0.01])
            fixture.set_extension(initial)
            fixture.set_motion(differential, common)
            fixture.launch()
            test.assertTrue(fixture.accepted.numpy()[0])
            trial = initial - common + np.array([-differential, differential])
            expected, _ = _reference_taut(trial, [0.001, 0.001], [2.0])
            result = fixture.solver.tendon_segment_stretch.numpy()
            np.testing.assert_allclose(result, expected, atol=1.0e-8, rtol=0)
            test.assertAlmostEqual(float(result.sum()), float(initial.sum() - 2.0 * common), delta=2.0e-8)
            transfer = fixture.transfer.numpy()
            test.assertAlmostEqual(float(transfer[1]), float(trial[0] - expected[0]), delta=1.0e-8)
            np.testing.assert_array_equal(transfer[[0, 2]], 0.0)


def test_damped_pinhole_and_disabled_connections(test, device):
    """Project total linear-law tension and honor disabled material connections."""
    a, p = newton.TendonGuideType.ANCHOR, newton.TendonGuideType.PINHOLE
    fixture = _MaterialFixture(device, [a, p, a])
    initial = np.array([0.003, -0.001])
    damping = np.array([3.0, -1.0], dtype=np.float32)
    fixture.solver.tendon_segment_damping_tension.assign(damping)
    fixture.set_extension(initial)
    fixture.launch()
    test.assertTrue(fixture.accepted.numpy()[0])
    effective, _ = _reference_taut(initial + 0.001 * damping, [0.001, 0.001], [2.0])
    np.testing.assert_allclose(fixture.solver.tendon_segment_stretch.numpy(), effective - 0.001 * damping, atol=1.0e-9)
    fixture.set_extension(initial)
    fixture.launch(pinhole=False)
    test.assertTrue(fixture.accepted.numpy()[0])
    np.testing.assert_allclose(fixture.solver.tendon_segment_stretch.numpy(), initial, atol=1.0e-9)
    np.testing.assert_array_equal(fixture.transfer.numpy(), 0.0)


def test_damping_preserves_sticking_material(test, device):
    """Keep isolated and frictionally sticking spans unchanged under strong damping."""
    a, p = newton.TendonGuideType.ANCHOR, newton.TendonGuideType.PINHOLE
    for kinds in ([a, a], [a, p, a, p, a]):
        fixture = _MaterialFixture(device, kinds)
        initial = np.full(len(kinds) - 1, 0.001, dtype=np.float32)
        fixture.solver.tendon_segment_damping_tension.fill_(100000.0)
        fixture.set_extension(initial)
        fixture.launch()
        test.assertTrue(fixture.accepted.numpy()[0])
        np.testing.assert_array_equal(fixture.solver.tendon_segment_stretch.numpy(), initial)
        np.testing.assert_array_equal(fixture.transfer.numpy(), 0.0)


def test_damped_sliding_conserves_material(test, device):
    """Apply equal and opposite material transfers in taut and slack damped components."""
    a, p = newton.TendonGuideType.ANCHOR, newton.TendonGuideType.PINHOLE
    fixture = _MaterialFixture(device, [a, p, a])
    fixture.solver.tendon_segment_length.fill_(2.0)
    initial = np.full(2, 0.001, dtype=np.float32)
    compliance = fixture.solver.tendon_segment_active_compliance.numpy().astype(np.float64)
    fixture.set_extension(initial)
    fixture.launch()
    advance = fixture.launch
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            fixture.launch()
        advance = partial(wp.capture_launch, capture.graph)
    for damping in ([1000.0, 1.0], [1.0, 1000.0], [-10.0, 2.0]):
        fixture.set_extension(initial)
        fixture.solver.tendon_segment_damping_tension.assign(np.array(damping, dtype=np.float32))
        advance()
        test.assertTrue(fixture.accepted.numpy()[0])
        effective = fixture.workspace.initial.numpy().astype(np.float64)
        if effective.sum() > 0.0:
            if effective[0] > 2.0 * effective[1]:
                flow = effective[0] - 2.0 * effective.sum() / 3.0
            else:
                flow = effective[0] - effective.sum() / 3.0
        else:
            flow = -max(effective[1], 0.0)
        expected = initial.astype(np.float64) + np.array([-flow, flow])
        actual = fixture.solver.tendon_segment_stretch.numpy()
        np.testing.assert_array_equal(actual, expected.astype(np.float32))
        rounding = np.sum(np.abs(np.spacing(actual)), dtype=np.float64)
        test.assertAlmostEqual(
            float(actual.sum(dtype=np.float64)), float(initial.sum(dtype=np.float64)), delta=rounding
        )
        test.assertAlmostEqual(float(fixture.transfer.numpy()[1]), flow, delta=abs(float(np.spacing(np.float32(flow)))))
        # The projected force is still the damped force, not just elastic tension.
        actual_force = np.maximum(actual.astype(np.float64) / compliance + damping, 0.0)
        if effective.sum() > 0.0:
            high, low = sorted(actual_force, reverse=True)
            test.assertAlmostEqual(high, 2.0 * low, delta=1e-4)
        else:
            np.testing.assert_allclose(actual_force, 0.0, atol=1e-6)


def test_damped_isolated_span_preserves_rest_history(test, device):
    """Preserve anchored material through repeated damped XPBD and VBD steps."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body(xform=wp.transform((0.1, 0.0, 0.0), wp.quat_identity()), is_kinematic=True)
    builder.add_tendon(
        [
            newton.TendonGuide(body=-1),
            newton.TendonGuide(body=body, compliance=0.1, damping=1000.0, rest_length=0.099),
        ]
    )
    builder.color()
    model = builder.finalize(device=device)
    expected = model.tendon_segment_rest_length.numpy()
    for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
        with test.subTest(solver=solver_type.__name__):
            solver = solver_type(model, iterations=1, tendon_material_direct=True)
            a, b = model.state(), model.state()
            control = model.control()
            for i in range(16):
                pose = a.body_q.numpy()
                pose[0, 0] = 0.1 + 0.001 * i
                a.body_q.assign(pose)
                a.body_qd.assign(np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
                solver.step(a, b, control, None, 0.001)
                a, b = b, a
            solver.check_tendon_material(a)
            np.testing.assert_array_equal(a.tendon_segment_rest_length.numpy(), expected)


def test_components_and_tendons_are_independent(test, device):
    """Keep anchor-separated components and packed tendon offsets independent."""
    a, r, p = newton.TendonGuideType.ANCHOR, newton.TendonGuideType.ROLLER, newton.TendonGuideType.PINHOLE
    fixture = _MaterialFixture(device, [a, r, a, p, a], copies=2)
    initial = np.array([0.02, 0.0, 0.001, 0.029, 0.0, 0.006, 0.004, 0.004], dtype=np.float32)
    fixture.set_extension(initial)
    fixture.launch()
    np.testing.assert_array_equal(fixture.accepted.numpy(), True)
    expected = np.array([0.04 / 3, 0.02 / 3, 0.01, 0.02, 0.002, 0.004, 0.004, 0.004])
    np.testing.assert_allclose(fixture.solver.tendon_segment_stretch.numpy(), expected, atol=1.0e-8, rtol=0)
    transfer = fixture.transfer.numpy()
    np.testing.assert_array_equal(transfer[[0, 2, 4, 5, 7, 9]], 0.0)
    test.assertEqual(float(transfer[8]), 0.0)


def test_inactive_slots_and_reactivation(test, device):
    """Invalidate packed face caches as a bypassed guide re-enters the route."""
    a, r = newton.TendonGuideType.ANCHOR, newton.TendonGuideType.ROLLER
    fixture = _MaterialFixture(device, [a, r, r, r, a])
    initial = np.array([0.02, 0.007, 0.001, 0.002], dtype=np.float32)
    solver = fixture.solver
    fixture.set_extension(initial)
    solver.tendon_guide_active.assign(np.array([True, True, False, True, True]))
    solver.tendon_segment_active.assign(np.array([1, 0, 1, 1], dtype=np.int32))
    solver.tendon_guide_cone_seg_l.assign(np.array([-1, 0, -1, 2, -1], dtype=np.int32))
    solver.tendon_guide_cone_seg_r.assign(np.array([-1, 2, -1, 3, -1], dtype=np.int32))
    fixture.launch()
    test.assertTrue(fixture.accepted.numpy()[0])
    expected, _ = _reference_taut(initial[[0, 2, 3]], [0.001] * 3, [2.0] * 2)
    np.testing.assert_allclose(solver.tendon_segment_stretch.numpy()[[0, 2, 3]], expected, atol=1.0e-8)
    test.assertEqual(float(solver.tendon_segment_stretch.numpy()[1]), float(initial[1]))
    test.assertEqual(float(fixture.transfer.numpy()[2]), 0.0)
    fixture.set_extension(initial)
    solver.tendon_guide_active.fill_(True)
    solver.tendon_segment_active.fill_(1)
    solver.tendon_guide_cone_seg_l.assign(np.array([-1, 0, 1, 2, -1], dtype=np.int32))
    solver.tendon_guide_cone_seg_r.assign(np.array([-1, 1, 2, 3, -1], dtype=np.int32))
    fixture.launch()
    test.assertTrue(fixture.accepted.numpy()[0])
    expected, _ = _reference_taut(initial, [0.001] * 4, [2.0] * 3)
    np.testing.assert_allclose(solver.tendon_segment_stretch.numpy(), expected, atol=1.0e-8)
    test.assertEqual(int(fixture.workspace.count.numpy()[0]), 4)


def test_failure_does_not_publish_partial_tendon(test, device):
    """Preserve stretch and slip when a later component fails, then permit retry."""
    a, r = newton.TendonGuideType.ANCHOR, newton.TendonGuideType.ROLLER
    fixture = _MaterialFixture(device, [a, r, a, r, a])
    initial = np.array([0.02, 0.0, 0.1, 0.1], dtype=np.float32)
    fixture.set_extension(initial)
    fixture.solver.tendon_segment_rolling_delta_r.assign(np.array([0.001, 0.0, 0.0, 0.0], dtype=np.float32))
    fixture.solver.tendon_segment_rolling_delta_l.assign(np.array([0.0, -0.001, 0.0, 0.0], dtype=np.float32))
    fixture.launch()
    test.assertFalse(fixture.accepted.numpy()[0])
    test.assertEqual(int(fixture.workspace.failed_component.numpy()[0]), 2)
    test.assertEqual(int(fixture.workspace.status.numpy()[2]), int(TendonMaterialStatus.BOUND_INCOMPATIBLE))
    np.testing.assert_array_equal(fixture.solver.tendon_segment_stretch.numpy(), initial)
    np.testing.assert_array_equal(fixture.transfer.numpy(), -77.0)
    initial[2:] = 0.01
    fixture.set_extension(initial)
    fixture.launch()
    test.assertTrue(fixture.accepted.numpy()[0])
    test.assertEqual(int(fixture.workspace.failed_component.numpy()[0]), -1)
    np.testing.assert_allclose(
        fixture.solver.tendon_segment_stretch.numpy(), [0.04 / 3, 0.02 / 3, 0.01, 0.01], atol=1e-8
    )


def test_invalid_route_and_material_are_rejected(test, device):
    """Reject inconsistent span connectivity and unsupported material inputs."""
    a, r = newton.TendonGuideType.ANCHOR, newton.TendonGuideType.ROLLER
    fixture = _MaterialFixture(device, [a, r, a])
    initial = np.array([0.01, 0.01], dtype=np.float32)
    fixture.set_extension(initial)
    fixture.solver.tendon_guide_cone_seg_r.assign(np.array([-1, 2, -1], dtype=np.int32))
    fixture.launch()
    test.assertFalse(fixture.accepted.numpy()[0])
    test.assertEqual(int(fixture.workspace.status.numpy()[0]), int(TendonMaterialStatus.INVALID_ROUTE))
    np.testing.assert_array_equal(fixture.solver.tendon_segment_stretch.numpy(), initial)
    fixture.solver.tendon_guide_cone_seg_r.assign(np.array([-1, 1, -1], dtype=np.int32))
    fixture.solver.tendon_guide_cap_ratio.fill_(5.0)
    fixture.launch()
    test.assertFalse(fixture.accepted.numpy()[0])
    test.assertEqual(int(fixture.workspace.status.numpy()[0]), int(TendonMaterialStatus.BAD_INPUT))
    np.testing.assert_array_equal(fixture.transfer.numpy(), -77.0)
    fixture.solver.tendon_guide_cap_ratio.fill_(2.0)
    fixture.model.tendon_segment_compliance.zero_()
    fixture.launch()
    test.assertFalse(fixture.accepted.numpy()[0])
    test.assertEqual(int(fixture.workspace.status.numpy()[0]), int(TendonMaterialStatus.BAD_INPUT))
    with test.assertRaisesRegex(ValueError, "compliance"):
        _create_tendon_material_workspace(fixture.model)


def test_graph_replay_uses_current_trial(test, device):
    """Replay projection with changed input, including slip reversal and slack."""
    a, r = newton.TendonGuideType.ANCHOR, newton.TendonGuideType.ROLLER
    fixture = _MaterialFixture(device, [a, r, a])
    fixture.set_extension([0.01, 0.01])
    fixture.launch()
    advance = fixture.launch
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            fixture.launch()
        advance = partial(wp.capture_launch, capture.graph)
    for initial, expected in (
        ([0.02, 0.0], [0.04 / 3, 0.02 / 3]),
        ([0.0, 0.02], [0.02 / 3, 0.04 / 3]),
        ([-0.01, -0.003], [-0.01, -0.003]),
        ([0.005, 0.005], [0.005, 0.005]),
    ):
        fixture.set_extension(initial)
        advance()
        test.assertTrue(fixture.accepted.numpy()[0])
        np.testing.assert_allclose(fixture.solver.tendon_segment_stretch.numpy(), expected, atol=1.0e-8)


def test_workspace_limits_each_anchor_component(test, device):
    """Apply the 32-span limit per anchor component, not to the entire tendon."""
    a, p = newton.TendonGuideType.ANCHOR, newton.TendonGuideType.PINHOLE
    for component_sizes, valid in (((32, 32), True), ((33,), False)):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        guides = [newton.TendonGuide(body=-1)]
        for count in component_sizes:
            for span in range(count):
                guides.append(
                    newton.TendonGuide(
                        body=-1,
                        offset=(0.0, 0.0, float(len(guides))),
                        guide_type=a if span == count - 1 else p,
                        compliance=0.001,
                        mu=0.3,
                    )
                )
        builder.add_tendon(guides)
        model = builder.finalize(device=device)
        if valid:
            workspace = _create_tendon_material_workspace(model)
            test.assertEqual(workspace.initial.size, sum(component_sizes))
            test.assertEqual(workspace.faces.size, sum(component_sizes))
        else:
            with test.assertRaisesRegex(ValueError, "32 authored spans"):
                _create_tendon_material_workspace(model)


def _world_material_model(device, *, worlds=1, mu=0.0):
    """Build two orthogonal fixed spans with unequal initial material allocation."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for _ in range(worlds):
        builder.begin_world()
        builder.add_tendon(
            [
                newton.TendonGuide(body=-1, offset=(-1.0, 0.0, 0.0)),
                newton.TendonGuide(
                    body=-1,
                    guide_type=newton.TendonGuideType.PINHOLE,
                    compliance=0.01,
                    damping=0.3,
                    rest_length=0.8,
                    mu=mu,
                ),
                newton.TendonGuide(body=-1, offset=(0.0, 1.0, 0.0), compliance=0.01, damping=0.3, rest_length=1.0),
            ]
        )
        builder.end_world()
    builder.color()
    return builder.finalize(device=device)


def test_direct_steps_and_state_replay(test, device):
    """Equilibrate real XPBD/VBD steps and keep material history with each State."""
    model = _world_material_model(device)
    for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
        with test.subTest(solver=solver_type.__name__):
            solver = solver_type(model, iterations=2, tendon_material_direct=True)
            initial, state, output, saved = (model.state() for _ in range(4))
            control = model.control()

            def advance(solver=solver, state=state, output=output, control=control):
                solver.step(state, output, control, None, 0.01)
                solver.step(output, state, control, None, 0.01)

            advance()
            solver.check_tendon_material(state)
            np.testing.assert_allclose(state.tendon_segment_rest_length.numpy(), [0.9, 0.9], atol=1e-6)
            np.testing.assert_allclose(solver.tendon_segment_material_tension.numpy(), [10.0, 10.0], atol=1e-4)
            saved.assign(state)
            state.tendon_segment_rest_length.fill_(0.7)
            advance()
            np.testing.assert_allclose(solver.tendon_segment_material_tension.numpy(), [30.0, 30.0], atol=1e-4)
            state.assign(saved)
            advance()
            np.testing.assert_allclose(solver.tendon_segment_material_tension.numpy(), [10.0, 10.0], atol=1e-4)

            if wp.get_device(device).is_cuda:
                with wp.ScopedCapture(device=device) as capture:
                    advance()
                state.assign(initial)
                wp.capture_launch(capture.graph)
                solver.check_tendon_material(state)
                np.testing.assert_allclose(state.tendon_segment_rest_length.numpy(), [0.9, 0.9], atol=1e-6)
                state.tendon_segment_rest_length.fill_(0.7)
                wp.capture_launch(capture.graph)
                np.testing.assert_allclose(solver.tendon_segment_material_tension.numpy(), [30.0, 30.0], atol=1e-4)

            model.tendon_segment_compliance.fill_(0.02)
            solver.notify_model_changed(newton.ModelFlags.TENDON_PROPERTIES)
            state.assign(saved)
            advance()
            np.testing.assert_allclose(solver.tendon_segment_material_tension.numpy(), [5.0, 5.0], atol=1e-4)
            model.tendon_segment_compliance.fill_(0.01)


def test_direct_rest_edit_and_partial_reset(test, device):
    """Refresh authored rest lengths without changing history or captured array addresses."""
    for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
        with test.subTest(solver=solver_type.__name__):
            model = _world_material_model(device, worlds=2, mu=0.2)
            solver = solver_type(model, iterations=2, tendon_material_direct=True)
            a, b, saved = (model.state() for _ in range(3))
            control = model.control()

            def advance(solver=solver, a=a, b=b, control=control):
                solver.step(a, b, control, None, 0.01)
                solver.step(b, a, control, None, 0.01)

            advance()
            saved.assign(a)
            rest = a.tendon_segment_rest_length.numpy().copy()
            graph = None
            if wp.get_device(device).is_cuda:
                with wp.ScopedCapture(device=device) as capture:
                    advance()
                graph = capture.graph
            a.assign(saved)
            model.tendon_segment_rest_length.fill_(0.7)
            solver.notify_model_changed(newton.ModelFlags.TENDON_PROPERTIES)
            if graph is None:
                advance()
            else:
                wp.capture_launch(graph)
            np.testing.assert_allclose(a.tendon_segment_rest_length.numpy(), rest, atol=1e-6)

            mask = wp.array([True, False, False], dtype=bool, device=device)
            solver.reset(a, flags=0, world_mask=mask)
            if graph is None:
                advance()
            else:
                wp.capture_launch(graph)
            solver.check_tendon_material(a)
            np.testing.assert_allclose(a.tendon_segment_rest_length.numpy()[:2], [0.7, 0.7], atol=1e-6)
            np.testing.assert_allclose(a.tendon_segment_rest_length.numpy()[2:], rest[2:], atol=1e-6)
            np.testing.assert_allclose(solver.tendon_segment_material_tension.numpy()[:2], [30.0, 30.0], atol=1e-4)


def test_accepted_failure_follows_state(test, device):
    """Latch accepted failures across steps/copies/graphs without poisoning other trajectories."""
    test.expectOutputRegex(
        r"^Error: direct tendon material solve rejected accepted pose "
        r"\(tendon=[01], component=[02], status=-1\)\. "
        r"Discard this trajectory and call solver\.check_tendon_material\(state\)\.$",
        stream="stdout",
    )
    model = _world_material_model(device, worlds=2, mu=1.0)
    for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
        with test.subTest(solver=solver_type.__name__):
            model.tendon_guide_mu.assign(np.array([0.0, 1.0, 0.0] * 2, dtype=np.float32))
            solver = solver_type(model, iterations=2, tendon_material_direct=True)
            a, b, clean, copied = (model.state() for _ in range(4))
            control = model.control()
            solver.step(a, b, control, None, 0.01)
            with test.assertRaisesRegex(RuntimeError, "BAD_INPUT"):
                solver.check_tendon_material(b)
            failure = b._tendon_material_failure.numpy().copy()
            np.testing.assert_array_equal(failure[:, 0], int(TendonMaterialStatus.BAD_INPUT))
            copied.assign(b)
            np.testing.assert_array_equal(copied._tendon_material_failure.numpy(), failure)
            solver.check_tendon_material(clean)

            # Fix the parameter, not the trajectory: its earlier rejection must remain visible.
            model.tendon_guide_mu.fill_(0.1)
            solver.notify_model_changed(newton.ModelFlags.TENDON_PROPERTIES)
            solver.step(b, a, control, None, 0.01)
            np.testing.assert_array_equal(a._tendon_material_failure.numpy(), failure)
            solver.step(clean, b, control, None, 0.01)
            solver.check_tendon_material(b)
            np.testing.assert_array_equal(b._tendon_material_failure.numpy()[:, 0], 0)
            with test.assertRaisesRegex(RuntimeError, "BAD_INPUT"):
                solver.check_tendon_material(copied)

            mask = wp.array([True, False, False], dtype=bool, device=device)
            solver.reset(copied, world_mask=mask, flags=0)
            np.testing.assert_array_equal(
                copied._tendon_material_failure.numpy()[:, 0], [0, int(TendonMaterialStatus.BAD_INPUT)]
            )
            solver.reset(copied, flags=0)
            solver.check_tendon_material(copied)

            if wp.get_device(device).is_cuda:
                with wp.ScopedCapture(device=device) as capture:
                    solver.step(clean, b, control, None, 0.01)
                    solver.step(b, clean, control, None, 0.01)
                model.tendon_guide_mu.fill_(1.0)
                wp.capture_launch(capture.graph)
                with test.assertRaisesRegex(RuntimeError, "BAD_INPUT"):
                    solver.check_tendon_material(clean)
                model.tendon_guide_mu.fill_(0.1)
                wp.capture_launch(capture.graph)
                with test.assertRaisesRegex(RuntimeError, "BAD_INPUT"):
                    solver.check_tendon_material(clean)


def test_provisional_material_failure_can_retry(test, device):
    """Retry rejected intermediate iterates without publishing or latching their errors."""
    model = _world_material_model(device, mu=1.0)
    solver = newton.solvers.SolverXPBD(model, tendon_material_direct=True)
    state = model.state()
    solver._load_tendon_history(state)
    solver._update_tendon_cone_rows(model, state.body_q, False)
    before = solver.tendon_segment_rest_length.numpy().copy()
    solver._solve_tendon_material(None, None, None, 0.01, damping_from_pose_delta=False, accepted_pose=False)
    np.testing.assert_array_equal(solver.tendon_segment_rest_length.numpy(), before)
    np.testing.assert_array_equal(solver._tendon_material_failure.numpy()[:, 0], 0)
    test.assertLess(int(solver._tendon_material_workspace.status.numpy()[0]), 0)
    solver.tendon_guide_cap_ratio.fill_(1.0)
    solver._solve_tendon_material(None, None, None, 0.01, damping_from_pose_delta=False, accepted_pose=True)
    np.testing.assert_allclose(solver.tendon_segment_rest_length.numpy(), [0.9, 0.9], atol=1e-6)
    np.testing.assert_array_equal(solver._tendon_material_failure.numpy()[:, 0], 0)


def test_direct_dynamics_match_converged_sweeps(test, device):
    """Match a moving off-center frictional roller's pose, spin, rest lengths, and damped forces."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)), com=(0.1, 0.0, 0.0))
    builder.add_tendon(
        [
            newton.TendonGuide(body=-1, offset=(-1.0, 0.0, -1.0)),
            newton.TendonGuide(
                body=body,
                guide_type=newton.TendonGuideType.ROLLER,
                radius=0.2,
                axis=(0.0, 1.0, 0.0),
                mu=0.2,
                compliance=0.01,
                damping=0.3,
                rest_length=1.1,
            ),
            newton.TendonGuide(body=-1, offset=(1.0, 0.0, -1.0), compliance=0.02, damping=0.3, rest_length=1.3),
        ]
    )
    builder.color()
    model = builder.finalize(device=device)
    for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
        with test.subTest(solver=solver_type.__name__):
            results = []
            for direct in (False, True):
                options = (
                    {"deterministic": wp.DeterministicMode.RUN_TO_RUN}
                    if solver_type is newton.solvers.SolverXPBD
                    else {}
                )
                solver = solver_type(
                    model, iterations=8, tendon_material_direct=direct, tendon_settle_tol=1e-6, **options
                )
                a, b = model.state(), model.state()
                control = model.control()
                a.body_qd.assign(np.array([[0.1, 0.0, -0.2, 0.0, 0.4, 0.0]], dtype=np.float32))
                for _ in range(8):
                    solver.step(a, b, control, None, 0.002)
                    a, b = b, a
                solver.check_tendon_material(a)
                results.append(
                    [
                        a.body_q.numpy().copy(),
                        a.body_qd.numpy().copy(),
                        a.tendon_segment_rest_length.numpy().copy(),
                        solver.tendon_segment_material_tension.numpy().copy(),
                        solver.tendon_segment_damping_tension.numpy().copy(),
                    ]
                )
                if direct:
                    saved, expected = model.state(), model.state()
                    saved.assign(a)
                    solver.step(a, b, control, None, 0.002)
                    expected.assign(b)
                    # Change the last material cache before restoring the saved trajectory.
                    solver.step(b, a, control, None, 0.002)
                    solver.reset(a, flags=0)
                    a.assign(saved)
                    solver.step(a, b, control, None, 0.002)
                    for name in ("body_q", "body_qd", "tendon_segment_rest_length"):
                        if solver_type is newton.solvers.SolverXPBD:
                            np.testing.assert_array_equal(getattr(b, name).numpy(), getattr(expected, name).numpy())
                        else:
                            np.testing.assert_allclose(
                                getattr(b, name).numpy(), getattr(expected, name).numpy(), atol=1e-6, rtol=1e-6
                            )
            test.assertGreater(float(np.max(results[1][3])), 1.0)
            test.assertGreater(float(np.linalg.norm(results[1][0][:, :3])), 1e-4)
            for expected, actual in zip(*results, strict=True):
                np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=5e-4)


def test_direct_dynamic_route_reactivation(test, device):
    """Preserve cable material when a world roller repeatedly activates and detaches."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    endpoints = builder.add_body(mass=0.0)
    builder.add_tendon(
        [
            newton.TendonGuide(body=endpoints, offset=(0.0, 0.0, -0.5), axis=(0.0, 1.0, 0.0)),
            newton.TendonGuide(
                body=-1,
                guide_type=newton.TendonGuideType.ROLLER,
                offset=(0.25, 0.0, 0.0),
                axis=(0.0, 1.0, 0.0),
                radius=0.1,
                dynamic=True,
                mu=0.2,
                compliance=0.01,
            ),
            newton.TendonGuide(body=endpoints, offset=(0.0, 0.0, 0.5), axis=(0.0, 1.0, 0.0), compliance=0.01),
        ]
    )
    builder.color()
    model = builder.finalize(device=device)
    for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
        with test.subTest(solver=solver_type.__name__):
            solver = solver_type(model, iterations=2, tendon_material_direct=True)
            a, b = model.state(), model.state()
            control = model.control()
            baseline = float(
                np.sum(solver.tendon_segment_rest_length.numpy()[solver.tendon_segment_active.numpy() != 0])
            )
            for x, active in ((0.2, True), (0.0, False), (0.2, True)):
                pose = a.body_q.numpy()
                pose[0, 0] = x
                a.body_q.assign(pose)
                solver.step(a, b, control, None, 0.01)
                a, b = b, a
                solver.check_tendon_material(a)
                test.assertEqual(bool(a.tendon_guide_active.numpy()[1]), active)
                material = float(
                    np.sum(a.tendon_segment_rest_length.numpy()[solver.tendon_segment_active.numpy() != 0])
                )
                if active:
                    material += _loaded_route_arc_length(solver, model, 1)
                test.assertAlmostEqual(material, baseline, delta=1e-6)


def test_direct_input_contract(test, device):
    """Reject unsupported material edits, gradient states, and mutable solver selection."""
    model = _world_material_model(device)
    for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
        solver = solver_type(model, iterations=1, tendon_material_direct=True)
        model.tendon_segment_compliance.zero_()
        with test.assertRaisesRegex(ValueError, "compliance"):
            solver.notify_model_changed(newton.ModelFlags.TENDON_PROPERTIES)
        with test.assertRaisesRegex(ValueError, "compliance"):
            solver_type(model, tendon_material_direct=True)
        model.tendon_segment_compliance.fill_(0.01)
        solver.tendon_material_direct = False
        with test.assertRaisesRegex(ValueError, "Reconstruct"):
            solver.step(model.state(), model.state(), model.control(), None, 0.01)
    for options in ({"iterations": 0}, {"integrate_with_external_rigid_solver": True}):
        with test.assertRaisesRegex(ValueError, "Direct tendon"):
            newton.solvers.SolverVBD(model, tendon_material_direct=True, **options)

    # State.requires_grad is derived from body/particle arrays, so use a body here.
    builder = newton.ModelBuilder()
    builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
    builder.color()
    grad_model = builder.finalize(device=device)
    for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
        solver = solver_type(grad_model, tendon_material_direct=True)
        a, b = grad_model.state(requires_grad=True), grad_model.state()
        for state_in, state_out in ((a, b), (b, a)):
            with test.assertRaisesRegex(NotImplementedError, "not differentiable"):
                solver.step(state_in, state_out, grad_model.control(), None, 0.01)


def test_stiff_pinhole_projection_matches_reference(test, device):
    """Resolve tiny strains on long spans before rest-length rounding loses force precision."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    points = [(-0.005, 0.0, -0.15)]
    points.extend((0.005 * np.cos(a), 0.0, 0.005 * np.sin(a)) for a in np.linspace(np.pi, 0.0, 9))
    points.append((0.005, 0.0, -0.15))
    builder.add_tendon(
        [
            newton.TendonGuide(
                body=-1,
                offset=point,
                guide_type=newton.TendonGuideType.ANCHOR
                if i in (0, len(points) - 1)
                else newton.TendonGuideType.PINHOLE,
                mu=0.25,
                **({"compliance": 2e-8} if i else {}),
            )
            for i, point in enumerate(points)
        ]
    )
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverXPBD(model, tendon_material_direct=True)
    solver._update_tendon_cone_rows(model, None, False)
    lengths = solver.tendon_segment_length.numpy().copy()
    rest = lengths.copy()
    rest[-1] -= np.float32(4e-6)
    solver.tendon_segment_route_rest_length.assign(rest)
    solver.tendon_segment_rest_length_step.assign(rest)
    compliance = solver.tendon_segment_active_compliance.numpy().astype(np.float64)
    trial = (lengths - rest).astype(np.float64)
    caps = solver.tendon_guide_cap_ratio.numpy()[1:-1].astype(np.float64)
    expected, _ = _reference_taut(trial, compliance, caps)
    solver._solve_tendon_material(None, None, None, 1 / 2400, damping_from_pose_delta=False, accepted_pose=True)
    projected = solver.tendon_segment_stretch.numpy().astype(np.float64)
    np.testing.assert_allclose(projected / compliance, expected / compliance, atol=1e-5, rtol=1e-6)
    test.assertAlmostEqual(float(projected.sum()), float(trial.sum()), delta=5e-12)
    # Rest history is float32; its round trip must be within the actual per-span ULP.
    rebuilt = (lengths - solver.tendon_segment_rest_length.numpy()).astype(np.float64)
    bound = np.spacing(rest).astype(np.float64) * 0.51
    test.assertTrue(np.all(abs(rebuilt - projected) <= bound))
    np.testing.assert_array_equal(solver._tendon_material_failure.numpy()[:, 0], 0)


def test_stiff_pinhole_body_force_balance(test, device):
    """Balance a 10 N load after body convergence rather than accepting a near-static pose alone."""
    with wp.ScopedDevice(device):
        for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
            with test.subTest(solver=solver_type.__name__):
                model, slider = build_stiff_pinhole_capstan(n_pinholes=9, mu=0.25)
                model.body_color_groups = [wp.array([body], dtype=int) for body in range(model.body_count)]
                options = {"joint_linear_relaxation": 0.8} if solver_type is newton.solvers.SolverXPBD else {}
                solver = solver_type(model, iterations=128, tendon_material_direct=True, **options)
                a, b = model.state(), model.state()
                control = model.control()
                force = np.zeros((model.body_count, 6), dtype=np.float32)
                force[slider, 2] = -10.0
                # Both buffers keep the same external pull, including captured replay.
                a.body_f.assign(force)
                b.body_f.assign(force)
                dt = 1 / 2400
                for _ in range(16):
                    solver.step(a, b, control, None, dt)
                    a, b = b, a
                solver.check_tendon_material(a)
                tension = solver.tendon_segment_material_tension.numpy()
                compliance = solver.tendon_segment_active_compliance.numpy()
                resolution = float(np.spacing(a.tendon_segment_rest_length.numpy()[-1]) / compliance[-1])
                test.assertLess(resolution, 0.75)
                test.assertAlmostEqual(float(tension[-1]), 10.0, delta=resolution)
                test.assertGreater(float(tension[0]), 0.4 * float(tension[-1]))
                test.assertLess(float(tension[0]), 0.6 * float(tension[-1]))


class TestTendonMaterialTransfer(NewtonTestCase):
    def setUp(self):
        """Allow Warp's module-loading diagnostics while checking all other output."""
        self.allowOutputRegex(r"^Module .* load on device .*", stream="stdout")


for function in (
    test_rolling_trial_and_slide,
    test_damped_pinhole_and_disabled_connections,
    test_damping_preserves_sticking_material,
    test_damped_sliding_conserves_material,
    test_damped_isolated_span_preserves_rest_history,
    test_components_and_tendons_are_independent,
    test_inactive_slots_and_reactivation,
    test_failure_does_not_publish_partial_tendon,
    test_invalid_route_and_material_are_rejected,
    test_graph_replay_uses_current_trial,
    test_workspace_limits_each_anchor_component,
    test_direct_steps_and_state_replay,
    test_direct_rest_edit_and_partial_reset,
    test_accepted_failure_follows_state,
    test_provisional_material_failure_can_retry,
    test_direct_dynamics_match_converged_sweeps,
    test_direct_dynamic_route_reactivation,
    test_direct_input_contract,
    test_stiff_pinhole_projection_matches_reference,
    test_stiff_pinhole_body_force_balance,
):
    add_function_test(TestTendonMaterialTransfer, function.__name__, function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
