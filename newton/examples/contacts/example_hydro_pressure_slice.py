# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Hydro Pressure Slice
#
# Visualize a hydroelastic-style pressure-density field by slicing an SDF
# shape and rendering one continuous heat-map section.
#
# Command: python -m newton.examples hydro_pressure_slice --shape box
#
###########################################################################

from __future__ import annotations

from collections import deque

import numpy as np
import warp as wp

import newton
import newton.examples


@wp.func
def sample_sdf_extrapolated_local(
    sparse_sdf_ptr: wp.uint64,
    coarse_sdf_ptr: wp.uint64,
    center: wp.vec3,
    half_extents: wp.vec3,
    sparse_voxel_size: wp.vec3,
    background_value: float,
    sdf_pos: wp.vec3,
) -> float:
    """Sample SDF using sparse->coarse->extrapolated fallback."""
    lower = center - half_extents
    upper = center + half_extents

    inside_extent = (
        sdf_pos[0] >= lower[0]
        and sdf_pos[0] <= upper[0]
        and sdf_pos[1] >= lower[1]
        and sdf_pos[1] <= upper[1]
        and sdf_pos[2] >= lower[2]
        and sdf_pos[2] <= upper[2]
    )

    if inside_extent:
        sparse_idx = wp.volume_world_to_index(sparse_sdf_ptr, sdf_pos)
        sparse_dist = wp.volume_sample_f(sparse_sdf_ptr, sparse_idx, wp.Volume.LINEAR)
        if sparse_dist >= background_value * 0.99 or wp.isnan(sparse_dist):
            coarse_idx = wp.volume_world_to_index(coarse_sdf_ptr, sdf_pos)
            return wp.volume_sample_f(coarse_sdf_ptr, coarse_idx, wp.Volume.LINEAR)
        return sparse_dist

    eps = 1.0e-2 * sparse_voxel_size
    clamped_pos = wp.min(wp.max(sdf_pos, lower + eps), upper - eps)
    dist_to_boundary = wp.length(sdf_pos - clamped_pos)
    coarse_idx = wp.volume_world_to_index(coarse_sdf_ptr, clamped_pos)
    boundary_dist = wp.volume_sample_f(coarse_sdf_ptr, coarse_idx, wp.Volume.LINEAR)
    return boundary_dist + dist_to_boundary


@wp.kernel
def build_pressure_slice_field(
    resolution: int,
    axis: int,
    axis_position: float,
    plane_scale: float,
    sparse_sdf_ptr: wp.uint64,
    coarse_sdf_ptr: wp.uint64,
    sdf_center: wp.vec3,
    sdf_half_extents: wp.vec3,
    sparse_voxel_size: wp.vec3,
    background_value: float,
    global_pressure_depth: float,
    out_points: wp.array(dtype=wp.vec3),
    out_density: wp.array(dtype=wp.float32),
    out_inside_count: wp.array(dtype=wp.int32),
):
    """Sample one axis-aligned section and write mesh points + heat values."""
    tid = wp.tid()
    total = resolution * resolution
    if tid >= total:
        return

    u_idx = tid % resolution
    v_idx = tid // resolution

    u = float(u_idx) / float(resolution - 1)
    v = float(v_idx) / float(resolution - 1)

    hx = sdf_half_extents[0] * plane_scale
    hy = sdf_half_extents[1] * plane_scale
    hz = sdf_half_extents[2] * plane_scale

    sample = wp.vec3(0.0, 0.0, 0.0)
    if axis == 0:
        sample = wp.vec3(
            sdf_center[0] + axis_position,
            sdf_center[1] + (2.0 * u - 1.0) * hy,
            sdf_center[2] + (2.0 * v - 1.0) * hz,
        )
    elif axis == 1:
        sample = wp.vec3(
            sdf_center[0] + (2.0 * u - 1.0) * hx,
            sdf_center[1] + axis_position,
            sdf_center[2] + (2.0 * v - 1.0) * hz,
        )
    else:
        sample = wp.vec3(
            sdf_center[0] + (2.0 * u - 1.0) * hx,
            sdf_center[1] + (2.0 * v - 1.0) * hy,
            sdf_center[2] + axis_position,
        )

    out_points[tid] = sample

    sdf_val = sample_sdf_extrapolated_local(
        sparse_sdf_ptr,
        coarse_sdf_ptr,
        sdf_center,
        sdf_half_extents,
        sparse_voxel_size,
        background_value,
        sample,
    )

    if sdf_val > 0.0:
        out_density[tid] = -1.0
        return

    # Raw penetration extent proxy from SDF depth. Smoothing/remapping to the
    # immutable pressure potential is done on the full slice in Python.
    out_density[tid] = wp.clamp((-sdf_val) / wp.max(global_pressure_depth, 1.0e-8), 0.0, 1.0)
    wp.atomic_add(out_inside_count, 0, 1)


def density_to_rgb_image(density_grid: np.ndarray) -> np.ndarray:
    """Map density [0, 1] to RGB with a monotonic perceptual colormap."""
    d = np.clip(density_grid.astype(np.float32), 0.0, 1.0)

    # Viridis-like anchor table (normalized RGB) to avoid false contour bands.
    t_knots = np.array([0.00, 0.13, 0.25, 0.38, 0.50, 0.63, 0.75, 0.88, 1.00], dtype=np.float32)
    c_knots = np.array(
        [
            [68, 1, 84],
            [71, 44, 122],
            [59, 81, 139],
            [44, 113, 142],
            [33, 144, 141],
            [39, 173, 129],
            [92, 200, 99],
            [170, 220, 50],
            [253, 231, 37],
        ],
        dtype=np.float32,
    ) / 255.0

    seg = np.searchsorted(t_knots, d, side="right") - 1
    seg = np.clip(seg, 0, len(t_knots) - 2)
    t0 = t_knots[seg]
    t1 = t_knots[seg + 1]
    alpha = (d - t0) / np.maximum(t1 - t0, 1.0e-8)
    rgb = (1.0 - alpha[..., None]) * c_knots[seg] + alpha[..., None] * c_knots[seg + 1]

    outside = density_grid < 0.0
    rgb[outside] = 0.0

    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def harmonic_pressure_potential(
    depth_grid: np.ndarray,
    gamma: float,
    iterations: int,
) -> np.ndarray:
    """Build a smooth slice-intrinsic pressure potential.

    Solves a Poisson problem on the slice interior with zero Dirichlet boundary
    on the shape boundary and outside. This yields a smooth field that is not
    tied to axis-depth scaling and avoids piecewise nearest-face transitions.
    """
    inside = depth_grid >= 0.0
    if not np.any(inside):
        return depth_grid.copy()

    # SDF narrow-band truncation can create enclosed "outside" islands inside the
    # true shape interior. Fill holes topologically so immutable pressure remains
    # continuous through the full cross-section.
    outside = ~inside
    exterior_outside = np.zeros_like(outside, dtype=bool)
    h, w = outside.shape
    q: deque[tuple[int, int]] = deque()

    for x in range(w):
        if outside[0, x]:
            exterior_outside[0, x] = True
            q.append((0, x))
        if outside[h - 1, x] and not exterior_outside[h - 1, x]:
            exterior_outside[h - 1, x] = True
            q.append((h - 1, x))
    for y in range(h):
        if outside[y, 0] and not exterior_outside[y, 0]:
            exterior_outside[y, 0] = True
            q.append((y, 0))
        if outside[y, w - 1] and not exterior_outside[y, w - 1]:
            exterior_outside[y, w - 1] = True
            q.append((y, w - 1))

    while q:
        y, x = q.popleft()
        if y > 0 and outside[y - 1, x] and not exterior_outside[y - 1, x]:
            exterior_outside[y - 1, x] = True
            q.append((y - 1, x))
        if y + 1 < h and outside[y + 1, x] and not exterior_outside[y + 1, x]:
            exterior_outside[y + 1, x] = True
            q.append((y + 1, x))
        if x > 0 and outside[y, x - 1] and not exterior_outside[y, x - 1]:
            exterior_outside[y, x - 1] = True
            q.append((y, x - 1))
        if x + 1 < w and outside[y, x + 1] and not exterior_outside[y, x + 1]:
            exterior_outside[y, x + 1] = True
            q.append((y, x + 1))

    hole_cells = outside & (~exterior_outside)
    if np.any(hole_cells):
        inside = inside | hole_cells
        outside = ~inside

    outside_up = np.ones_like(outside, dtype=bool)
    outside_down = np.ones_like(outside, dtype=bool)
    outside_left = np.ones_like(outside, dtype=bool)
    outside_right = np.ones_like(outside, dtype=bool)

    outside_up[1:, :] = outside[:-1, :]
    outside_down[:-1, :] = outside[1:, :]
    outside_left[:, 1:] = outside[:, :-1]
    outside_right[:, :-1] = outside[:, 1:]

    boundary = inside & (outside_up | outside_down | outside_left | outside_right)
    free = inside & (~boundary)

    potential = np.zeros_like(depth_grid, dtype=np.float32)

    if np.any(free):
        for _ in range(max(int(iterations), 1)):
            up = np.zeros_like(potential)
            down = np.zeros_like(potential)
            left = np.zeros_like(potential)
            right = np.zeros_like(potential)

            up[1:, :] = potential[:-1, :]
            down[:-1, :] = potential[1:, :]
            left[:, 1:] = potential[:, :-1]
            right[:, :-1] = potential[:, 1:]

            # Discrete Poisson solve: -L u = 1 on free interior cells.
            update = 0.25 * (up + down + left + right + 1.0)
            potential[free] = update[free]

    max_val = float(np.max(potential[inside]))
    if max_val <= 1.0e-8:
        mapped = np.zeros_like(depth_grid, dtype=np.float32)
        mapped[inside] = np.clip(depth_grid[inside], 0.0, 1.0)
    else:
        mapped = np.zeros_like(depth_grid, dtype=np.float32)
        mapped[inside] = np.clip(potential[inside] / max_val, 0.0, 1.0)

    mapped[inside] = np.power(mapped[inside], max(gamma, 1.0e-6)).astype(np.float32)
    mapped[~inside] = -1.0
    return mapped


def build_regular_grid_indices(resolution: int) -> np.ndarray:
    """Create triangle index buffer for a regular resolution x resolution grid."""
    tris: list[int] = []
    for v in range(resolution - 1):
        row0 = v * resolution
        row1 = (v + 1) * resolution
        for u in range(resolution - 1):
            i00 = row0 + u
            i10 = row0 + (u + 1)
            i01 = row1 + u
            i11 = row1 + (u + 1)
            tris.extend([i00, i10, i11, i00, i11, i01])
    return np.asarray(tris, dtype=np.int32)


def build_regular_grid_uvs(resolution: int) -> np.ndarray:
    """Create UVs for a regular resolution x resolution grid."""
    u = np.linspace(0.0, 1.0, resolution, dtype=np.float32)
    v = np.linspace(0.0, 1.0, resolution, dtype=np.float32)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    return np.stack([uu.reshape(-1), vv.reshape(-1)], axis=-1)


class Example:
    def __init__(self, viewer, args):
        if not wp.get_device().is_cuda:
            raise RuntimeError(
                "hydro_pressure_slice requires CUDA (Mesh.build_sdf uses wp.Volume, which is CUDA-only)."
            )

        self.viewer = viewer
        self.device = wp.get_device()

        self.fps = 60.0
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.shape_name = args.shape
        self.slice_axis = {"x": 0, "y": 1, "z": 2}[args.slice_axis]
        self.slice_position_normalized = float(args.slice_position)
        self.animate_slice = bool(args.animate_slice)
        self.slice_speed = float(args.slice_speed)
        self.plane_scale = float(args.plane_scale)
        self.resolution = int(args.resolution)
        self.pressure_gamma = float(args.pressure_gamma)
        self.harmonic_iterations = int(args.harmonic_iterations)
        self.show_shape = False
        self.show_slice = True

        self.mesh = self._create_shape_mesh(self.shape_name)
        self.mesh.build_sdf(
            max_resolution=int(args.sdf_resolution),
            narrow_band_range=(-float(args.narrow_band), float(args.narrow_band)),
            margin=float(args.sdf_margin),
        )
        assert self.mesh.sdf is not None
        sdf_data = self.mesh.sdf.data

        self.sdf_center = wp.vec3(sdf_data.center)
        self.sdf_half_extents = wp.vec3(sdf_data.half_extents)
        self.sparse_voxel_size = wp.vec3(sdf_data.sparse_voxel_size)
        self.background_value = float(sdf_data.background_value)
        self.sparse_sdf_ptr = wp.uint64(int(sdf_data.sparse_sdf_ptr))
        self.coarse_sdf_ptr = wp.uint64(int(sdf_data.coarse_sdf_ptr))

        self.global_pressure_depth = max(
            1.0e-4,
            min(
                float(self.sdf_half_extents[0]),
                float(self.sdf_half_extents[1]),
                float(self.sdf_half_extents[2]),
            ),
        )

        self.capacity = self.resolution * self.resolution
        self.slice_points = wp.zeros(self.capacity, dtype=wp.vec3, device=self.device)
        self.slice_density = wp.zeros(self.capacity, dtype=wp.float32, device=self.device)
        self.slice_inside_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.last_slice_count = 0

        self.slice_indices = wp.array(build_regular_grid_indices(self.resolution), dtype=wp.int32, device=self.device)
        self.slice_uvs = wp.array(build_regular_grid_uvs(self.resolution), dtype=wp.vec2, device=self.device)
        self.slice_texture = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)

        self.shape_xforms = wp.array([wp.transform_identity()], dtype=wp.transform, device=self.device)
        self.shape_colors = wp.array([wp.vec3(0.72, 0.72, 0.76)], dtype=wp.vec3, device=self.device)
        self.shape_materials = wp.array([wp.vec4(0.75, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=self.device)

        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(2.5, -2.0, 1.6), -20.0, 50.0)

        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.render_ui, position="side")

        self._slice_dirty = True

    def _create_shape_mesh(self, shape: str) -> newton.Mesh:
        if shape == "sphere":
            return newton.Mesh.create_sphere(radius=0.6, compute_inertia=False)
        if shape == "box":
            return newton.Mesh.create_box(hx=0.65, hy=0.45, hz=0.3, compute_inertia=False)
        if shape == "capsule":
            return newton.Mesh.create_capsule(
                radius=0.28,
                half_height=0.5,
                up_axis=newton.Axis.Z,
                compute_inertia=False,
            )
        if shape == "cylinder":
            return newton.Mesh.create_cylinder(
                radius=0.4,
                half_height=0.5,
                up_axis=newton.Axis.Z,
                compute_inertia=False,
            )
        if shape == "cone":
            return newton.Mesh.create_cone(
                radius=0.45,
                half_height=0.55,
                up_axis=newton.Axis.Z,
                compute_inertia=False,
            )
        if shape == "ellipsoid":
            return newton.Mesh.create_ellipsoid(rx=0.7, ry=0.45, rz=0.3, compute_inertia=False)
        raise ValueError(f"Unsupported shape '{shape}'")

    def _axis_half_extent(self) -> float:
        return float(self.sdf_half_extents[self.slice_axis])

    def _update_slice(self):
        if not self._slice_dirty:
            return

        self.slice_inside_count.zero_()
        axis_position = self.slice_position_normalized * self._axis_half_extent()

        wp.launch(
            kernel=build_pressure_slice_field,
            dim=self.capacity,
            inputs=[
                self.resolution,
                self.slice_axis,
                axis_position,
                self.plane_scale,
                self.sparse_sdf_ptr,
                self.coarse_sdf_ptr,
                self.sdf_center,
                self.sdf_half_extents,
                self.sparse_voxel_size,
                self.background_value,
                self.global_pressure_depth,
            ],
            outputs=[self.slice_points, self.slice_density, self.slice_inside_count],
            device=self.device,
        )

        self.last_slice_count = int(self.slice_inside_count.numpy()[0])
        density_grid = self.slice_density.numpy().reshape(self.resolution, self.resolution)
        immutable_field = harmonic_pressure_potential(
            density_grid,
            gamma=self.pressure_gamma,
            iterations=self.harmonic_iterations,
        )
        self.slice_texture = density_to_rgb_image(immutable_field)
        self._slice_dirty = False

    def step(self):
        if self.animate_slice:
            self.slice_position_normalized = float(np.sin(self.sim_time * self.slice_speed))
            self._slice_dirty = True
        self.sim_time += self.frame_dt

    def render(self):
        self._update_slice()

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_shapes(
            name="/slice/shape",
            geo_type=newton.GeoType.MESH,
            geo_scale=(1.0, 1.0, 1.0),
            xforms=self.shape_xforms,
            colors=self.shape_colors,
            materials=self.shape_materials,
            geo_src=self.mesh,
            hidden=not self.show_shape,
        )

        self.viewer.log_mesh(
            name="/slice/pressure_heatmap",
            points=self.slice_points,
            indices=self.slice_indices,
            uvs=self.slice_uvs,
            texture=self.slice_texture,
            hidden=not self.show_slice,
            backface_culling=False,
        )

        self.viewer.end_frame()

    def render_ui(self, imgui):
        imgui.text("Hydro Pressure Slice")
        imgui.text(f"Shape: {self.shape_name}")
        imgui.text("Immutable harmonic pressure field.")
        imgui.text("Smooth interior potential from slice Poisson solve.")
        imgui.text(f"Nonlinear curve: field^gamma, gamma={self.pressure_gamma:.2f}")

        _changed, self.show_shape = imgui.checkbox("Show Shape", self.show_shape)
        _changed, self.show_slice = imgui.checkbox("Show Slice", self.show_slice)
        changed, self.animate_slice = imgui.checkbox("Animate Slice", self.animate_slice)
        if changed:
            self._slice_dirty = True

        if not self.animate_slice:
            changed, self.slice_position_normalized = imgui.slider_float(
                "Slice Position",
                self.slice_position_normalized,
                -1.0,
                1.0,
            )
            if changed:
                self._slice_dirty = True

        imgui.text("Slice Axis")
        if imgui.radio_button("X", self.slice_axis == 0):
            self.slice_axis = 0
            self._slice_dirty = True
        if imgui.radio_button("Y", self.slice_axis == 1):
            self.slice_axis = 1
            self._slice_dirty = True
        if imgui.radio_button("Z", self.slice_axis == 2):
            self.slice_axis = 2
            self._slice_dirty = True

        imgui.text(f"Inside Samples: {self.last_slice_count}")

    def test_final(self):
        self._update_slice()
        assert self.last_slice_count > 0, "Expected non-empty heat-map section for the selected shape."


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=600)
    parser.add_argument(
        "--shape",
        type=str,
        choices=["sphere", "box", "capsule", "cylinder", "cone", "ellipsoid"],
        default="box",
        help="Shape to section and visualize.",
    )
    parser.add_argument(
        "--slice-axis",
        type=str,
        choices=["x", "y", "z"],
        default="z",
        help="Axis normal of the slice plane.",
    )
    parser.add_argument(
        "--slice-position",
        type=float,
        default=0.0,
        help="Initial normalized slice position in [-1, 1].",
    )
    parser.add_argument(
        "--animate-slice",
        action="store_true",
        help="Animate slice position sinusoidally through the shape.",
    )
    parser.add_argument(
        "--slice-speed",
        type=float,
        default=1.5,
        help="Animation speed [rad/s] for --animate-slice.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="Slice grid resolution per axis (total samples = resolution^2).",
    )
    parser.add_argument(
        "--pressure-gamma",
        type=float,
        default=2.2,
        help="Nonlinear pressure curve exponent (density^gamma).",
    )
    parser.add_argument(
        "--harmonic-iterations",
        type=int,
        default=160,
        help="Jacobi iterations for slice harmonic pressure potential.",
    )
    parser.add_argument(
        "--plane-scale",
        type=float,
        default=1.02,
        help="Scale factor for section sampling extents relative to SDF bounds.",
    )
    parser.add_argument(
        "--sdf-resolution",
        type=int,
        default=96,
        help="SDF max resolution (must be divisible by 8).",
    )
    parser.add_argument(
        "--narrow-band",
        type=float,
        default=0.1,
        help="Half-width [m] for SDF narrow band range (-band, +band).",
    )
    parser.add_argument(
        "--sdf-margin",
        type=float,
        default=0.1,
        help="Padding [m] around the shape during SDF build.",
    )

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
