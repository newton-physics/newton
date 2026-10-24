# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverVBD ``rigid_rod_direct_solve`` on near-inextensible rods.

For a capsule-chain rod with segment length ``l`` the stretch row is far stiffer than the bend row: the ratio
``EA * l**2 / (4 * EI)`` is ~1e3-1e6 for threads, cables and textile tapes. These tests cover that regime:

- a horizontally clamped rod under its own weight must settle to the static equilibrium of its own discretization
  (an exact discrete reference, see :func:`discrete_cantilever_tip`);
- a tip load ramped along the rod must be carried with the elastic strain ``F / EA``;
- the state must stay finite: a non-finite state fails the test within the 30 steps where it first appears.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices

# SolverVBD options under test: the direct (linear-time, per-rod) solve of the rod joint rows.
ROD_SOLVER_KWARGS: dict = {"rigid_rod_direct_solve": True}

GRAVITY = 9.81
# A cable-scale section: the regime is set by the stiffness ratio alone, and segments this size keep their true inertia
# (finalize() raises an inertia whose smallest principal moment is below 1e-10 kg m^2).
EI = 1.0e-4  # N m^2
LINEAR_DENSITY = 2.0e-2  # kg/m
RADIUS = 2.5e-3  # m
STIFFNESS_RATIOS = (1.0e3, 1.0e6)  # EA l^2 / (4 EI): both ends of the slender-rod range


def discrete_cantilever_tip(
    n: int, seg: float, joint_ke: float, seg_mass: float, g: float = GRAVITY
) -> tuple[float, float]:
    """Static tip (dx, dz) of the discrete model itself: ``n`` rigid segments, the first hinged to the clamp, a
    bend spring of stiffness ``joint_ke`` at every joint with energy ``1/2 k (2 tan(dphi/2))^2`` (the DER curvature
    binormal of SolverVBD rods), each segment's weight at its midpoint. Planar Newton iteration on the angles."""
    phi = np.zeros(n)
    lever = (n - np.arange(n) - 0.5) * seg_mass * g * seg  # d(gravity energy)/d(sin phi_i) = -lever_i
    for _ in range(200):
        dphi = np.diff(np.concatenate([[0.0], phi]))
        kappa = 2.0 * np.tan(0.5 * dphi)
        dk = 1.0 / np.cos(0.5 * dphi) ** 2  # d kappa / d dphi
        m = joint_ke * kappa * dk  # dE/d dphi at each joint
        grad = m - np.concatenate([m[1:], [0.0]]) - lever * np.cos(phi)
        d2 = joint_ke * (dk * dk + kappa * dk * np.tan(0.5 * dphi))
        hess = np.diag(d2 + np.concatenate([d2[1:], [0.0]]) + lever * np.sin(phi))
        off = -d2[1:]
        hess[np.arange(n - 1), np.arange(1, n)] = off
        hess[np.arange(1, n), np.arange(n - 1)] = off
        step = np.linalg.solve(hess, -grad)
        phi += step
        if np.max(np.abs(step)) < 1e-14:
            break
    return float(seg * np.sum(np.cos(phi))), float(-seg * np.sum(np.sin(phi)))


def _segment_end(body_q: np.ndarray, half: float) -> np.ndarray:
    """+Z end of a COM-origin capsule body (the rod's forward direction)."""
    q = body_q[3:7]
    return np.asarray(body_q[:3]) + np.asarray(wp.quat_rotate(wp.quat(*q), wp.vec3(0.0, 0.0, half)))


class _ClampedRod:
    """A straight rod along +X whose first segment is a kinematic clamp; ``n_free`` dynamic segments follow."""

    def __init__(
        self,
        device,
        n_free: int,
        seg: float,
        ea: float,
        damping_time: float = 0.05,
        bend_xy: tuple[float, float] | None = None,
    ):
        self.seg, self.n_free = seg, n_free
        points = [wp.vec3((i - 1) * seg, 0.0, 0.0) for i in range(n_free + 2)]
        rod = newton.Rod(
            points, radius=RADIUS, stretch_rigidity=ea, shear_rigidity=ea, bend_rigidity=EI, twist_rigidity=EI
        )
        volume = math.pi * RADIUS**2 * seg + 4.0 / 3.0 * math.pi * RADIUS**3
        cfg = newton.ModelBuilder.ShapeConfig(density=LINEAR_DENSITY * seg / volume)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -GRAVITY))
        joint_ke = EI / seg
        bodies, joints = builder.add_rod(
            rod=rod, cfg=cfg, bend_damping=joint_ke * damping_time, twist_damping=joint_ke * damping_time
        )
        self.bodies = [int(b) for b in bodies]
        if bend_xy is not None:
            self._set_bend_axes(builder, joints, bend_xy, seg, damping_time)
        clamp = self.bodies[0]
        builder.body_mass[clamp] = 0.0
        builder.body_inv_mass[clamp] = 0.0
        builder.body_inertia[clamp] = wp.mat33(0.0)
        builder.body_inv_inertia[clamp] = wp.mat33(0.0)
        builder.body_flags[clamp] = int(newton.BodyFlags.KINEMATIC)
        builder.color()
        self.model = builder.finalize(device=device)
        self.joint_ke, self.seg_mass = joint_ke, LINEAR_DENSITY * seg

    @staticmethod
    def _set_bend_axes(builder, joints, bend_xy: tuple[float, float], seg: float, damping_time: float) -> None:
        """Flat section: bend rigidity (EI_x, EI_y) about the segment frame's x and y axes, per ROD joint dof
        (axis order shear x, shear y, stretch z, bend x, bend y, twist z)."""
        for joint in joints:
            start = builder.joint_qd_start[int(joint)]
            for axis, ei in zip((3, 4), bend_xy, strict=True):
                builder.joint_target_ke[start + axis] = ei / seg
                builder.joint_target_kd[start + axis] = ei / seg * damping_time

    def run(
        self,
        test: unittest.TestCase,
        seconds: float,
        dt: float,
        tip_force=None,
        iterations: int = 10,
        check_every: int = 30,
    ):
        """Step the rod; fail within ``check_every`` steps of the first non-finite state (each check syncs the
        device). tip_force(t) -> +X force on the last body [N]."""
        solver = newton.solvers.SolverVBD(self.model, iterations=iterations, **ROD_SOLVER_KWARGS)
        s0, s1, control = self.model.state(), self.model.state(), self.model.control()
        forces = np.zeros((self.model.body_count, 6), dtype=np.float32)
        steps = round(seconds / dt)
        for k in range(steps):
            s0.clear_forces()
            if tip_force is not None:
                forces[self.bodies[-1], 0] = tip_force((k + 1) * dt)
                s0.body_f.assign(forces)
            solver.step(s0, s1, control, None, dt)
            s0, s1 = s1, s0
            if (k + 1) % check_every and k + 1 < steps:
                continue
            if not np.all(np.isfinite(s0.body_q.numpy())):
                load = 0.0 if tip_force is None else tip_force((k + 1) * dt)
                test.fail(
                    f"non-finite rod state by step {k + 1}/{steps} (t = {(k + 1) * dt:.4f} s, tip load {load:.4g} N)"
                )
        return s0.body_q.numpy()

    def tip(self, body_q: np.ndarray) -> np.ndarray:
        return _segment_end(body_q[self.bodies[-1]], 0.5 * self.seg)

    def joint_strain(self, body_q: np.ndarray) -> np.ndarray:
        """Axial gap between consecutive segment ends over the segment length (the stretch rows' strain)."""
        half = 0.5 * self.seg
        q = body_q[self.bodies]
        ends = np.array([_segment_end(qi, half) for qi in q[:-1]])
        starts = np.array([_segment_end(qi, -half) for qi in q[1:]])
        return np.linalg.norm(starts - ends, axis=1) / self.seg


def _overhang_and_seg(n_free: int) -> tuple[float, float]:
    """Peirce overhang 2c (ASTM D1388 geometry, chord ~41.5 deg) split into n_free segments."""
    c = (EI / (LINEAR_DENSITY * GRAVITY)) ** (1.0 / 3.0)
    return 2.0 * c, 2.0 * c / n_free


def test_heavy_cantilever_settles_to_discrete_equilibrium(test, device, n_free: int = 12):
    """Settle a self-weight cantilever to its exact discrete equilibrium at both ends of the slender-rod range."""
    _, seg = _overhang_and_seg(n_free)
    for ratio in STIFFNESS_RATIOS:
        with test.subTest(stiffness_ratio=ratio):
            ea = ratio * 4.0 * EI / seg**2
            rod = _ClampedRod(device, n_free, seg, ea)
            body_q = rod.run(test, seconds=2.0, dt=1.0 / 120.0)
            tip = rod.tip(body_q)
            ref_dx, ref_dz = discrete_cantilever_tip(n_free, seg, rod.joint_ke, rod.seg_mass)
            test.assertAlmostEqual(
                tip[2] / ref_dz, 1.0, delta=0.01, msg=f"tip dz {tip[2]:.5f} m vs discrete equilibrium {ref_dz:.5f} m"
            )
            test.assertAlmostEqual(tip[0] / ref_dx, 1.0, delta=0.01)


def test_flat_section_droops_about_its_loaded_axis(
    test, device, n_free: int = 10, ratio: float = 1.0e4, overhang: float = 0.11, stiff_over_soft: float = 10.0
):
    """Droop a flat section by its loaded-axis bend rigidity alone, whichever axis is the stiff one.

    Gravity bends this rod about its segments' y axis, so the droop follows EI_y. The overhang stays below the
    lateral-torsional buckling length.
    """
    seg = overhang / n_free
    ea = ratio * 4.0 * EI / seg**2
    for bend_xy in ((EI, stiff_over_soft * EI), (stiff_over_soft * EI, EI)):
        with test.subTest(bend_xy=bend_xy):
            rod = _ClampedRod(device, n_free, seg, ea, bend_xy=bend_xy)
            tip = rod.tip(rod.run(test, seconds=1.5, dt=1.0 / 120.0))
            _, ref_dz = discrete_cantilever_tip(n_free, seg, bend_xy[1] / seg, rod.seg_mass)
            test.assertAlmostEqual(
                tip[2] / ref_dz,
                1.0,
                delta=0.01,
                msg=f"tip dz {tip[2]:.6f} m vs discrete equilibrium {ref_dz:.6f} m for EI_y",
            )
            test.assertLess(abs(tip[1]), 1.0e-3 * overhang)  # no lateral (out-of-plane) drift


def test_ramped_tip_load_is_carried(test, device, n_free: int = 12, max_load: float = 20.0):
    """Carry a ramped axial tip load with the elastic strain F / EA at both ends of the slender-rod range."""
    overhang, seg = _overhang_and_seg(n_free)
    for ratio in STIFFNESS_RATIOS:
        with test.subTest(stiffness_ratio=ratio):
            ea = ratio * 4.0 * EI / seg**2
            load = min(max_load, 0.04 * ea)  # at most 4 % strain
            rod = _ClampedRod(device, n_free, seg, ea)
            body_q = rod.run(test, seconds=1.5, dt=1.0 / 120.0, tip_force=lambda t, f=load: f * min(1.0, t / 1.0))
            strain = rod.joint_strain(body_q)
            test.assertAlmostEqual(
                float(np.mean(strain)) * ea / load,
                1.0,
                delta=0.02,
                msg=f"mean joint strain {np.mean(strain):.4g} vs F/EA {load / ea:.4g}",
            )
            test.assertGreater(rod.tip(body_q)[0], overhang)  # pulled straight, not folded back


devices = get_test_devices()


class TestRodDirectRegime(unittest.TestCase):
    pass


add_function_test(
    TestRodDirectRegime,
    "test_heavy_cantilever_settles_to_discrete_equilibrium",
    test_heavy_cantilever_settles_to_discrete_equilibrium,
    devices=devices,
)
add_function_test(
    TestRodDirectRegime, "test_ramped_tip_load_is_carried", test_ramped_tip_load_is_carried, devices=devices
)
add_function_test(
    TestRodDirectRegime,
    "test_flat_section_droops_about_its_loaded_axis",
    test_flat_section_droops_about_its_loaded_axis,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
