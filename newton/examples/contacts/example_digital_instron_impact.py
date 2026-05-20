# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Digital Instron Impact
#
# Replays a calibrated Digital Instron v2 contact trace in the Newton viewer.
#
# Command: python -m newton.examples digital_instron_impact
#
###########################################################################

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from projects.digital_instron_v2.manifest import load_manifest
from projects.digital_instron_v2.workflow import (
    _load_averaged_cycle,
    _load_spring_grid,
    _rearfoot_mask,
    _spring_state_for_trial_frame,
    _trial_contact_surface_cache,
)


def _colors_from_compression(compression: np.ndarray, max_compression: float) -> np.ndarray:
    normalized = np.clip(compression / max(max_compression, 1.0e-9), 0.0, 1.0)
    colors = np.empty((len(compression), 3), dtype=np.float32)
    colors[:, 0] = 0.12 + 0.88 * normalized
    colors[:, 1] = 0.55 * (1.0 - normalized) + 0.12
    colors[:, 2] = 0.95 * (1.0 - normalized) + 0.10
    return colors


def _line_segments_from_neighbors(points: np.ndarray, neighbors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = []
    ends = []
    for index, point in enumerate(points):
        for neighbor in neighbors[index, (1, 3)]:
            if neighbor >= 0:
                starts.append(point)
                ends.append(points[neighbor])
    if not starts:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    return np.asarray(starts, dtype=np.float32), np.asarray(ends, dtype=np.float32)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.frame_index = 0

        self.manifest = load_manifest(args.manifest)
        self.output_dir = Path(args.output_dir) if args.output_dir else self.manifest.cache_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.trial = self._select_trial(args.trial)
        self.trace = _load_averaged_cycle(self.trial.averaged_cycle_path)
        frame_limit = max(2, int(args.trace_frames))
        sample_count = min(frame_limit, len(self.trace["time_s"]))
        self.trace_indices = np.unique(
            np.linspace(0, len(self.trace["time_s"]) - 1, sample_count, dtype=np.int64)
        )
        self.frame_count = len(self.trace_indices)

        self.spring_grid, vertices = _load_spring_grid(self.manifest, self.output_dir)
        self.contact_surfaces = _trial_contact_surface_cache(self.manifest, self.spring_grid)
        self.rearfoot_mask = _rearfoot_mask(self.manifest, self.spring_grid, vertices)

        self.xy = np.asarray(self.spring_grid.grid_uv_m, dtype=np.float64)
        self.xy_center = np.mean(self.xy, axis=0)
        self.xy = self.xy - self.xy_center
        self.z_offset = -float(np.min(self.spring_grid.bottom_m))
        self.bottom_z = np.asarray(self.spring_grid.bottom_m, dtype=np.float64) + self.z_offset
        self.top_z = np.asarray(self.spring_grid.top_m, dtype=np.float64) + self.z_offset
        self.max_displacement = float(np.max(np.maximum(self.trace["displacement_m"], 0.0)))
        self.point_radius = max(float(self.spring_grid.spacing_m) * 0.22, 0.0008)

        self.bottom_points = self._points_at(self.bottom_z)
        self.top_points = self._points_at(self.top_z)
        self.mesh_starts, self.mesh_ends = _line_segments_from_neighbors(self.top_points, self.spring_grid.neighbors)
        self.bottom_colors = wp.full(len(self.bottom_points), wp.vec3(0.24, 0.26, 0.28), dtype=wp.vec3)
        self.contact_color = wp.vec3(0.05, 0.95, 0.52)
        self.mesh_color = (0.18, 0.32, 0.52)

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.contacts = self.model.contacts()
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.25, -0.45, 0.22), pitch=-22.0, yaw=145.0)

    def _select_trial(self, name: str | None):
        fit_trials = [trial for trial in self.manifest.trials if trial.include_in_fit and trial.averaged_cycle_path]
        if not fit_trials:
            raise ValueError("The manifest does not contain any fit trials with averaged_cycle_path")
        if name is None:
            return fit_trials[0]
        for trial in fit_trials:
            if trial.name == name:
                return trial
        choices = ", ".join(trial.name for trial in fit_trials)
        raise ValueError(f"Unknown trial {name!r}. Available fit trials: {choices}")

    def _points_at(self, z_values: np.ndarray) -> np.ndarray:
        return np.column_stack((self.xy[:, 0], self.xy[:, 1], z_values)).astype(np.float32)

    def _frame_state(self) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        trace_index = int(self.trace_indices[self.frame_index % self.frame_count])
        displacement = float(self.trace["displacement_m"][trace_index])
        velocity = float(self.trace["velocity_m_s"][trace_index])
        current_length, _spring_velocity = _spring_state_for_trial_frame(
            self.spring_grid,
            self.trial,
            self.rearfoot_mask,
            self.contact_surfaces,
            displacement,
            velocity,
        )
        compression = np.maximum(self.spring_grid.slack_length_m - current_length, 0.0)
        deformed_top_z = self.top_z - compression
        deformed_points = self._points_at(deformed_top_z)

        contact_z = self.top_z - max(displacement, 0.0)
        contact_mask = np.ones_like(compression, dtype=bool)
        if self.trial.fixture == "rearfoot_punch":
            contact_mask = self.rearfoot_mask
        elif self.trial.fixture == "fullfoot_last" and self.trial.name in self.contact_surfaces:
            contact_surface_0, valid = self.contact_surfaces[self.trial.name]
            contact_z = np.asarray(contact_surface_0, dtype=np.float64) + self.z_offset - max(displacement, 0.0)
            contact_mask = valid

        spring_starts = self._points_at(self.bottom_z)
        spring_ends = deformed_points
        active = compression > 1.0e-6
        if np.any(active):
            spring_starts = spring_starts[active]
            spring_ends = spring_ends[active]
        else:
            spring_starts = spring_starts[:1]
            spring_ends = spring_ends[:1]

        contact_points = self._points_at(contact_z)[contact_mask]
        if len(contact_points) == 0:
            contact_points = self._points_at(contact_z)[:1]

        return displacement, velocity, compression, deformed_points, spring_starts, spring_ends, contact_points

    def step(self):
        self.frame_index = (self.frame_index + 1) % self.frame_count
        self.sim_time += self.frame_dt

    def render(self):
        displacement, velocity, compression, deformed_points, spring_starts, spring_ends, contact_points = (
            self._frame_state()
        )
        colors = _colors_from_compression(compression, self.max_displacement)
        spring_colors = colors[compression > 1.0e-6]
        if len(spring_colors) == 0:
            spring_colors = colors[:1]

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_points(
            "/digital_instron/bottom_surface",
            wp.array(self.bottom_points, dtype=wp.vec3),
            self.point_radius,
            self.bottom_colors,
        )
        self.viewer.log_lines(
            "/digital_instron/undeformed_top_mesh",
            wp.array(self.mesh_starts, dtype=wp.vec3),
            wp.array(self.mesh_ends, dtype=wp.vec3),
            self.mesh_color,
        )
        self.viewer.log_points(
            "/digital_instron/deformed_top",
            wp.array(deformed_points, dtype=wp.vec3),
            self.point_radius * 1.25,
            wp.array(colors, dtype=wp.vec3),
        )
        self.viewer.log_lines(
            "/digital_instron/active_springs",
            wp.array(spring_starts, dtype=wp.vec3),
            wp.array(spring_ends, dtype=wp.vec3),
            wp.array(spring_colors, dtype=wp.vec3),
        )
        self.viewer.log_points(
            "/digital_instron/contact_surface",
            wp.array(contact_points, dtype=wp.vec3),
            self.point_radius * 1.8,
            wp.full(len(contact_points), self.contact_color, dtype=wp.vec3),
        )
        self.viewer.log_array(
            "/digital_instron/trace",
            np.asarray(
                [
                    self.sim_time,
                    displacement,
                    velocity,
                    float(np.max(compression)) if len(compression) else 0.0,
                ],
                dtype=np.float32,
            ),
        )
        self.viewer.end_frame()

    def test_final(self):
        if self.frame_count < 2:
            raise ValueError("Digital Instron impact scene did not load enough frames")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--manifest", default="DigitalInstron/manifest_v2.json", help="Path to the v2 manifest")
        parser.add_argument("--output-dir", default=None, help="Directory for conditioned mesh cache")
        parser.add_argument("--trial", default=None, help="Trial name to replay; defaults to the first fit trial")
        parser.add_argument("--trace-frames", type=int, default=240, help="Maximum averaged-cycle frames to replay")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
