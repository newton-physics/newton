# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dynamic elastic-foundation midsole example."""

import json
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.viewer
from projects.digital_instron_v2 import core, dynamics, workflow
from projects.digital_instron_v2.example import Example
from projects.digital_instron_v2.geometry import build_column_grid, load_mesh

MANIFEST = "DigitalInstron/manifest_v2.json"


def _run_mode(mode: str, num_frames: int) -> Example:
    """Instantiate and step the midsole example headlessly for ``num_frames``."""

    class _Args:
        pass

    args = _Args()
    args.mode = mode
    args.manifest = MANIFEST
    viewer = newton.viewer.ViewerNull(num_frames=num_frames)
    example = Example(viewer, args)
    for _ in range(num_frames):
        example.step()
    return example


class TestFoundationGeometry(unittest.TestCase):
    def test_column_bed(self):
        """Sample a nonempty column bed with valid clearances, thicknesses, and neighbours."""
        geo = dynamics.build_foundation_geometry(MANIFEST)
        column_count = len(geo.slack_m)
        self.assertGreater(column_count, 100)
        self.assertTrue(np.all(geo.slack_m > 0.0))
        self.assertTrue(np.all(geo.gap0_m >= 0.0))
        self.assertTrue(np.all(geo.z_free_m > geo.z_bottom_m))
        self.assertEqual(geo.neighbors.shape, (column_count, 4))
        self.assertTrue(np.all(geo.neighbors < column_count))
        self.assertTrue(np.all(geo.neighbors >= -2))

    def test_pasternak_neighbours_match_calibration_laplacian(self):
        """Reconstruct the compression Laplacian from the neighbour table and match the calibration operator."""
        geo = dynamics.build_foundation_geometry(MANIFEST)
        grid = build_column_grid(load_mesh(geo.midsole_mesh_path, 0.001), geo.spacing_m)
        rng = np.random.default_rng(0)
        compression = rng.random(len(geo.slack_m))

        laplacian = np.empty_like(compression)
        for i in range(len(compression)):
            total = -4.0 * compression[i]
            for side in range(4):
                j = geo.neighbors[i, side]
                if j >= 0:
                    total += compression[j]
                elif j == -1:
                    total += compression[i]
            laplacian[i] = total / geo.spacing_m**2

        reference = workflow.compression_laplacian(compression[None, :], geo.uv_m, grid.uv_m, geo.spacing_m)[0]
        np.testing.assert_allclose(laplacian, reference, atol=1.0e-10)


class TestMidsoleFoundation(unittest.TestCase):
    def test_kernel_reproduces_calibrated_model(self):
        """Sweep the Warp foundation through the Instron cycle and match core.predict to float precision.

        The live per-substep Hyperfoam-Maxwell-Pasternak force integration must reproduce
        the calibrated periodic-fixed-point model that was fitted to the bench data.
        """
        base = Path("DigitalInstron")
        config = json.loads((base / "manifest_v2.json").read_text())
        midsole = load_mesh(base / config["midsole_mesh"], 0.001)
        grid = build_column_grid(midsole, config["grid"]["coarse_spacing_m"])
        trials, _, _ = workflow.prepare_trials(base, config, grid, midsole)
        trial = next(t for t in trials if t.name == "fullfoot_185ms")

        material = dynamics.load_fitted_material(MANIFEST)
        predicted = core.predict(trial, material)

        geo = dynamics.build_foundation_geometry(MANIFEST)
        device = wp.get_preferred_device()
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        model = builder.finalize()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        anchor = np.column_stack([geo.uv_m[:, 0], geo.uv_m[:, 1], geo.surface_m])
        foundation = dynamics.MidsoleFoundation(
            anchor,
            geo.z_free_m,
            geo.slack_m,
            np.full(len(geo.slack_m), geo.area_m2),
            geo.neighbors,
            geo.spacing_m,
            material,
            body,
            model.body_com,
            dynamics.FoundationConfig(stretch_floor=1.0e-3),
            device,
        )

        displacement = np.asarray(trial.displacement_m)
        dt = np.asarray(trial.dt_s)
        collected = np.zeros_like(displacement)
        for cycle in range(6):
            for k in range(len(displacement)):
                state.body_q.assign(np.array([[0.0, 0.0, -displacement[k], 0.0, 0.0, 0.0, 1.0]], np.float32))
                state.body_qd.zero_()
                state.clear_forces()
                foundation.apply(state, float(dt[k]))
                if cycle == 5:
                    collected[k] = foundation.diagnostics()["normal_force_n"]

        peak = float(predicted.max())
        self.assertLess(np.sqrt(np.mean((collected - predicted) ** 2)) / peak, 5.0e-3)
        self.assertLess(abs(collected.max() - peak) / peak, 2.0e-2)

    def test_bristle_friction_sticks_below_cone_and_slips_above(self):
        """Verify anchored stick-slip friction: a planted patch resists a static offset, saturating at the cone.

        A purely viscous friction law produces zero tangential force at zero slip velocity,
        so the static-offset probes below would all read zero and fail without the anchored
        bristle model.
        """
        geo = dynamics.build_foundation_geometry(MANIFEST)
        material = dynamics.load_fitted_material(MANIFEST)
        device = wp.get_preferred_device()
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        model = builder.finalize()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        anchor = np.column_stack([geo.uv_m[:, 0], geo.uv_m[:, 1], geo.surface_m])
        kt, mu, depth = 2.0e4, 1.0, 0.006
        foundation = dynamics.MidsoleFoundation(
            anchor,
            geo.z_free_m,
            geo.slack_m,
            np.full(len(geo.slack_m), geo.area_m2),
            geo.neighbors,
            geo.spacing_m,
            material,
            body,
            model.body_com,
            dynamics.FoundationConfig(stretch_floor=1.0e-3, friction_stiffness=kt, friction=0.0, mu=mu),
            device,
        )

        def probe(dx: float) -> tuple[float, float]:
            # Hold the bed a fixed depth into the ground and offset it by ``dx`` tangentially with
            # zero velocity, returning the tangential reaction opposing the offset and the normal load.
            state.body_q.assign(np.array([[dx, 0.0, -depth, 0.0, 0.0, 0.0, 1.0]], np.float32))
            state.body_qd.zero_()
            state.clear_forces()
            foundation.apply(state, 1.0e-3)
            return -float(state.body_f.numpy()[0][0]), foundation.diagnostics()["normal_force_n"]

        seat_force, normal_force = probe(0.0)  # fresh contact seats the bristles with no pre-stretch
        cone = mu * normal_force
        self.assertGreater(normal_force, 100.0)
        self.assertLess(abs(seat_force), 1.0e-3 * cone)

        near_force, _ = probe(2.0e-5)  # 0.02 mm static offset -> static shear with zero slip velocity
        far_force, _ = probe(4.0e-5)  # 0.04 mm static offset -> larger elastic build-up, still stuck
        self.assertGreater(near_force, 0.05 * cone)
        self.assertLess(far_force, cone)
        self.assertGreater(far_force, near_force)

        foundation.reset()  # release the bristles, then re-seat before driving well past the cone
        probe(0.0)
        slip_force, slip_normal = probe(0.02)  # 20 mm offset saturates the whole patch at mu * fn
        self.assertAlmostEqual(slip_force, mu * slip_normal, delta=0.03 * mu * slip_normal)


class TestMidsoleExample(unittest.TestCase):
    def test_instron_hysteresis(self):
        """Run the digital Instron mode and audit its dissipative hysteresis loop."""
        example = _run_mode("instron", 190)
        example.test_final()

    def test_settle_supports_mass_with_friction(self):
        """Run the massive-midsole mode and audit stable weight support and lateral grip."""
        example = _run_mode("settle", 150)
        example.test_final()

    def test_stride_ground_reaction(self):
        """Run the synthetic-stride mode and audit its ground-reaction force and center-of-pressure roll."""
        example = _run_mode("stride", 40)
        example.test_final()

    def test_attached_dynamic_stride(self):
        """Run the attached, foot-mounted dynamic shoe and audit its stance/flight ground reaction.

        The dynamic shoe carries mass and inertia and stays coupled to the foot the whole
        stride, so the run must remain finite (no flight blow-up), develop a real stance
        ground reaction, and unload to near zero in flight without the shoe separating.
        """
        example = _run_mode("attached", 95)
        example.test_final()


if __name__ == "__main__":
    unittest.main()
