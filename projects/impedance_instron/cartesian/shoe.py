# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Attach the shared shoe foundation to a planar foot without a pitch servo."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import warp as wp

import newton
from projects.digital_shoe.artifact import load_artifact
from projects.digital_shoe.rendering import carried_column_endpoints
from projects.digital_shoe.runtime import FoundationConfig, MidsoleFoundation, SurroundConfig


class Shoe:
    """Evaluate contact at an ankle-centered foot pose supplied by the limb solver.

    The fullfoot last and the bed's rigid fixture-footprint backing share one
    carrier, with a passive outer region. Fixed attachment offsets retain the
    calibrated assembly; no second mesh-collision force or upper sliding contact
    is introduced. The
    planar angle increases from +X toward +Z (rotation about Newton's -Y axis).

    Args:
        artifact_path: Identified portable shoe artifact.
        mount_m: Ankle location in the intrinsic shoe frame [m], shape [3].
        static_pitch_rad: Measured static foot-axis angle [rad]. The intrinsic
            shoe is level in the static calibration, not aligned to skin markers.
        device: Warp device. CPU avoids device round trips in this reference solver.
    """

    def __init__(
        self,
        artifact_path: str | Path,
        mount_m,
        static_pitch_rad: float,
        device: str = "cpu",
        *,
        friction_model: str = "elastic_coulomb",
    ):
        self.artifact_path = Path(artifact_path).resolve()
        self.shoe = load_artifact(self.artifact_path)
        self.mount_m = np.asarray(mount_m, dtype=float)
        self.static_pitch_rad = float(static_pitch_rad)
        if self.mount_m.shape != (3,) or not np.isfinite(self.mount_m).all():
            raise ValueError("Shoe mount must be a finite three-vector in metres")
        if not np.isfinite(self.static_pitch_rad):
            raise ValueError("Static foot pitch must be finite")
        coordinate = self.shoe.raw["coordinate_system"]
        if coordinate.get("up_axis") != "+Z" or coordinate.get("length_unit") != "m":
            raise ValueError("The shoe must use the +Z-up metre convention")
        self.device = wp.get_device(device)
        bed = self.shoe.column_bed
        fixture = self.shoe.instron_fixture("fullfoot_last")
        lookup = {tuple(np.round(p, 8)): i for i, p in enumerate(bed.anchor_bottom_m[:, :2])}
        fixture_keys = [tuple(np.round(point, 8)) for point in fixture.carrier_anchor_m[:, :2]]
        if len(lookup) != len(bed.rest_length_m) or len(set(fixture_keys)) != len(fixture_keys):
            raise ValueError("Shoe bed and fixture footprint must have unique planar column coordinates")
        driven = np.zeros(len(bed.rest_length_m), dtype=np.int32)
        for key in fixture_keys:
            if key not in lookup:
                raise ValueError("Fixture footprint does not match the intrinsic shoe bed")
            driven[lookup[key]] = 1
        supported = np.asarray([lookup[key] for key in fixture_keys])
        self.anchor_local_m = bed.anchor_bottom_m - self.mount_m
        self.attachment_local_m = self.anchor_local_m.copy()
        self.attachment_local_m[:, 2] += bed.rest_length_m
        sites = fixture.carrier_anchor_m.copy()
        sites[:, 2] += bed.anchor_bottom_m[supported, 2] - fixture.foam_bottom_m
        self.attachment_local_m[supported] = sites - self.mount_m
        gap = sites[:, 2] - (bed.anchor_bottom_m[supported, 2] + bed.rest_length_m[supported])
        mesh = self.shoe.visual_mesh("fullfoot_last")
        self.last_vertices_local_m = mesh.vertices_m - self.mount_m
        self.last_triangles = mesh.triangles.copy()
        builder = newton.ModelBuilder()
        builder.add_body(mass=1.0, com=wp.vec3(0.0), inertia=wp.mat33(np.eye(3)), label="fullfoot_last_carrier")
        builder.add_shape_mesh(
            0,
            mesh=newton.Mesh(self.last_vertices_local_m.astype(np.float32), self.last_triangles.ravel()),
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False),
            color=(0.72, 0.77, 0.82),
            label="fullfoot_last",
        )
        # As in the previous rig, the mesh and all column sites share this body.
        # Only the foundation supplies contact; enabling triangle collision here
        # would add an unintended parallel force path.
        self.model = builder.finalize(device=self.device)
        self.state = self.model.state()
        self.foundation = MidsoleFoundation(
            self.anchor_local_m,
            np.zeros(len(bed.rest_length_m)),
            bed.rest_length_m,
            bed.area_m2,
            bed.neighbors,
            bed.spacing_m,
            self.shoe.material,
            0,
            self.model.body_com,
            FoundationConfig(
                ground_height_m=0.0,
                normal_damping=0.0,
                friction_stiffness=10000.0
                if friction_model == "legacy"
                else (1000.0 if friction_model == "maxwell" else 0.0),
                friction=10.0 if friction_model in ("legacy", "maxwell") else 0.0,
                mu=0.8,
                friction_model=friction_model,
            ),
            self.device,
            SurroundConfig(driven=driven, carrier_bond=True),
        )
        self.foundation.reset()
        self._bottoms = wp.zeros(len(driven), dtype=wp.vec3, device=self.device)
        self._tops = wp.zeros(len(driven), dtype=wp.vec3, device=self.device)
        self.metadata = {
            "path": str(self.artifact_path),
            "sha256": hashlib.sha256(self.artifact_path.read_bytes()).hexdigest(),
            "shoe_id": self.shoe.shoe_id,
            "artifact_provenance": self.shoe.provenance,
            "constitutive_model": self.shoe.raw["constitutive_model"],
            "mount_m": self.mount_m.tolist(),
            "static_pitch_rad": self.static_pitch_rad,
            "registration": "rigid placement only; no geometry scaling or material refit",
            "friction_model": friction_model,
            "friction_equilibrium_stiffness_n_m": (
                10000.0 if friction_model == "legacy" else (1000.0 if friction_model == "maxwell" else None)
            ),
            "friction_viscosity_ns_m": (
                10.0
                if friction_model in ("legacy", "maxwell")
                else (None if friction_model == "column_maxwell" else 0.0)
            ),
            "friction_mu": 0.8,
            "friction_relaxation_time_s": self.shoe.material.maxwell_relaxation_time_s
            if friction_model in ("maxwell", "column_maxwell")
            else None,
            "friction_stiffness_source": (
                "material_shear_modulus_times_column_area_over_rest_length"
                if friction_model in ("column_maxwell", "elastic_coulomb")
                else "configured_per_column"
            ),
            "attachment": "fullfoot last and driven spring tops share one rigid carrier with fixed assembly offsets",
            "column_count": len(driven),
            "driven_columns": int(driven.sum()),
            "passive_columns": int(len(driven) - driven.sum()),
            "last_vertex_count": len(self.last_vertices_local_m),
            "last_triangle_count": len(self.last_triangles),
            "mesh_collision_enabled": False,
            "fixture_offset_median_m": float(np.median(gap)),
            "fixture_offset_max_m": float(np.max(gap)),
            "limitations": "original fixture registration retained, not a new gap-aware upper contact solve; recorded shoe may differ",
            "side": "intrinsic artifact retained; sagittal projection, no certified anatomical side",
            "friction": {
                "mu": 0.8,
                "model": friction_model,
                "per_column_stiffness_n_m": (
                    10000.0 if friction_model == "legacy" else (1000.0 if friction_model == "maxwell" else None)
                ),
                "per_column_damping_n_s_m": (
                    10.0
                    if friction_model in ("legacy", "maxwell")
                    else (None if friction_model == "column_maxwell" else 0.0)
                ),
                "stiffness_source": (
                    "material_shear_modulus_times_column_area_over_rest_length"
                    if friction_model in ("column_maxwell", "elastic_coulomb")
                    else "configured_per_column"
                ),
                "equilibrium_shear_modulus_pa": self.shoe.material.equilibrium_shear_modulus_pa,
                "instantaneous_shear_modulus_pa": (
                    self.shoe.material.instantaneous_shear_modulus_pa
                    + self.shoe.material.instantaneous_shear_modulus_2_pa
                ),
                "relaxation_time_s": self.shoe.material.maxwell_relaxation_time_s
                if friction_model in ("maxwell", "column_maxwell")
                else None,
                "source": "declared contact assumptions, not identified by normal Instron loading",
            },
        }
        points = bed.anchor_bottom_m[:, (0, 2)] - self.mount_m[[0, 2]]
        # A sagittal outline is an undeformed registration diagnostic, not a
        # claim that rigid columns remain undeformed during contact.
        unique = sorted(set(map(tuple, points)))
        lower, upper = [], []
        for chain, ordered in ((lower, unique), (upper, reversed(unique))):
            for point in ordered:
                while len(chain) >= 2:
                    a, b = np.subtract(chain[-1], chain[-2]), np.subtract(point, chain[-1])
                    if a[0] * b[1] - a[1] * b[0] > 0:
                        break
                    chain.pop()
                chain.append(point)
        self.outline_local = np.asarray(lower[:-1] + upper[:-1])

    def geometry(self) -> dict[str, np.ndarray]:
        """Return fixed last and spring-site geometry in the ankle-centered frame [m]."""
        bed = self.shoe.column_bed
        return {
            "last_vertices_local_m": self.last_vertices_local_m.copy(),
            "last_triangles": self.last_triangles.copy(),
            "anchor_local_m": self.anchor_local_m.copy(),
            "attachment_local_m": self.attachment_local_m.copy(),
            "rest_length_m": bed.rest_length_m.copy(),
            "area_m2": bed.area_m2.copy(),
            "driven": self.foundation.driven.numpy().copy(),
            "static_pitch_rad": np.asarray(self.static_pitch_rad),
        }

    def column_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Copy per-column compression [m] and external ground normal pressure [Pa]."""
        compression = self.foundation.compression.numpy().copy()
        pressure = self.foundation.ground_force.numpy()[:, 2] / self.shoe.column_bed.area_m2
        return compression, pressure.astype(np.float32)

    def snapshot(self) -> dict[str, np.ndarray]:
        """Copy the loaded bed and reconstruct its endpoints without advancing physics."""
        wp.launch(
            carried_column_endpoints,
            dim=self.foundation.column_count,
            inputs=[
                0,
                self.state.body_q,
                self.foundation.anchor_local,
                self.foundation.rest_len,
                self.foundation.compression,
                self.foundation.driven,
                0.0,
                self._bottoms,
                self._tops,
            ],
            device=self.device,
        )
        compression, pressure = self.column_state()
        return {
            "bottom_m": self._bottoms.numpy().copy(),
            "top_m": self._tops.numpy().copy(),
            "compression_m": compression,
            "pressure_pa": pressure,
        }

    def outline(self, ankle_m, pitch_rad: float) -> np.ndarray:
        """Return the undeformed registered sagittal outline [m], shape [N, 2]."""
        angle = pitch_rad - self.static_pitch_rad
        c, s = np.cos(angle), np.sin(angle)
        return self.outline_local @ np.array([[c, s], [-s, c]]) + ankle_m

    def apply(self, ankle_m, velocity_m_s, pitch_rad: float, angular_velocity_rad_s: float, dt: float):
        """Advance contact once and return ankle wrench [N, N, N m] and compression [m]."""
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Contact timestep must be finite and positive")
        angle = pitch_rad - self.static_pitch_rad
        position = np.array([[ankle_m[0], 0.0, ankle_m[1], 0.0, -np.sin(angle / 2), 0.0, np.cos(angle / 2)]])
        velocity = np.array([[velocity_m_s[0], 0.0, velocity_m_s[1], 0.0, -angular_velocity_rad_s, 0.0]])
        if not np.isfinite(position).all() or not np.isfinite(velocity).all():
            raise ValueError("Foot pose and velocity must be finite")
        self.state.body_q.assign(position.astype(np.float32))
        self.state.body_qd.assign(velocity.astype(np.float32))
        self.foundation.apply(self.state, dt, clear_body_force=True)
        wrench = self.state.body_f.numpy()[0].astype(float)
        # Newton +Y torque is opposite to the mathematical X/Z angle.
        return np.array([wrench[0], wrench[2], -wrench[4]]), float(self.foundation.max_compression.numpy()[0])
