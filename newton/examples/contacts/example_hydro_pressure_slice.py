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
# Visualize the immutable pressure field produced by HydroelasticSDF by
# slicing the loaded shape and rendering one continuous heat-map section.
#
# Command: uv run -m newton.examples.contacts.example_hydro_pressure_slice --shape box
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples


@wp.kernel
def build_slice_points(
    resolution: int,
    axis: int,
    axis_position: float,
    plane_scale: float,
    sdf_center: wp.vec3,
    sdf_half_extents: wp.vec3,
    out_points: wp.array(dtype=wp.vec3),
):
    """Sample one axis-aligned section in object-local coordinates."""
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


@wp.kernel
def sample_pressure_on_slice(
    pressure_volume_id: wp.uint64,
    pressure_epsilon: float,
    points: wp.array(dtype=wp.vec3),
    out_inside_flag: wp.array(dtype=wp.float32),
    out_pressure: wp.array(dtype=wp.float32),
    out_inside_count: wp.array(dtype=wp.int32),
):
    """Sample immutable pressure volume and classify pressure-support interior."""
    tid = wp.tid()
    sample = points[tid]
    pressure_idx = wp.volume_world_to_index(pressure_volume_id, sample)
    pressure = wp.volume_sample_f(pressure_volume_id, pressure_idx, wp.Volume.LINEAR)
    if wp.isnan(pressure):
        pressure = 0.0
    pressure = wp.max(pressure, 0.0)
    if pressure <= pressure_epsilon:
        out_inside_flag[tid] = -1.0
        out_pressure[tid] = -1.0
        return

    out_inside_flag[tid] = 1.0
    out_pressure[tid] = pressure
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


def validate_slice_field(pressure_grid: np.ndarray) -> tuple[bool, float, float, float]:
    """Return validation metrics proving smooth interior pressure continuity."""
    inside = pressure_grid >= 0.0
    if not np.any(inside):
        return False, 0.0, 0.0, 0.0

    p_inside = pressure_grid[inside]
    max_p = float(np.max(p_inside))
    if max_p <= 1.0e-8:
        return False, 0.0, 0.0, 0.0

    p_norm = np.zeros_like(pressure_grid, dtype=np.float32)
    p_norm[inside] = pressure_grid[inside] / max_p

    # One-pixel morphological erosion marks deep interior support.
    deep = inside.copy()
    deep[1:, :] &= inside[:-1, :]
    deep[:-1, :] &= inside[1:, :]
    deep[:, 1:] &= inside[:, :-1]
    deep[:, :-1] &= inside[:, 1:]
    if not np.any(deep):
        deep = inside
    deep_p05 = float(np.percentile(p_norm[deep], 5.0)) if np.any(deep) else 0.0
    zero_fraction = float(np.mean(p_norm[inside] <= 1.0e-6))

    ys, xs = np.where(inside)
    y0, y1 = int(np.min(ys)), int(np.max(ys))
    x0, x1 = int(np.min(xs)), int(np.max(xs))
    sub_inside = inside[y0 : y1 + 1, x0 : x1 + 1]
    outside = ~sub_inside
    exterior = np.zeros_like(outside, dtype=bool)
    q: list[tuple[int, int]] = []

    h, w = outside.shape
    for x in range(w):
        if outside[0, x] and not exterior[0, x]:
            exterior[0, x] = True
            q.append((0, x))
        if outside[h - 1, x] and not exterior[h - 1, x]:
            exterior[h - 1, x] = True
            q.append((h - 1, x))
    for y in range(h):
        if outside[y, 0] and not exterior[y, 0]:
            exterior[y, 0] = True
            q.append((y, 0))
        if outside[y, w - 1] and not exterior[y, w - 1]:
            exterior[y, w - 1] = True
            q.append((y, w - 1))

    while q:
        y, x = q.pop()
        if y > 0 and outside[y - 1, x] and not exterior[y - 1, x]:
            exterior[y - 1, x] = True
            q.append((y - 1, x))
        if y + 1 < h and outside[y + 1, x] and not exterior[y + 1, x]:
            exterior[y + 1, x] = True
            q.append((y + 1, x))
        if x > 0 and outside[y, x - 1] and not exterior[y, x - 1]:
            exterior[y, x - 1] = True
            q.append((y, x - 1))
        if x + 1 < w and outside[y, x + 1] and not exterior[y, x + 1]:
            exterior[y, x + 1] = True
            q.append((y, x + 1))

    hole_fraction = float(np.sum(outside & (~exterior))) / float(np.sum(inside))

    valid = bool(deep_p05 > 0.05 and zero_fraction < 0.10 and hole_fraction < 1.0e-3)
    return valid, deep_p05, zero_fraction, hole_fraction


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
        self.show_shape = False
        self.show_slice = True

        self.mesh = self._create_shape_mesh(self.shape_name)
        self._build_hydroelastic_reference(args)

        self.capacity = self.resolution * self.resolution
        self.slice_points = wp.zeros(self.capacity, dtype=wp.vec3, device=self.device)
        self.slice_inside_flag = wp.zeros(self.capacity, dtype=wp.float32, device=self.device)
        self.slice_pressure = wp.zeros(self.capacity, dtype=wp.float32, device=self.device)
        self.slice_inside_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.last_slice_count = 0
        self.last_deep_p05 = 0.0
        self.last_zero_fraction = 0.0
        self.last_hole_fraction = 0.0
        self.last_validation_ok = False

        self.slice_indices = wp.array(build_regular_grid_indices(self.resolution), dtype=wp.int32, device=self.device)
        self.slice_uvs = wp.array(build_regular_grid_uvs(self.resolution), dtype=wp.vec2, device=self.device)
        self.slice_texture = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)

        self.shape_xforms = wp.array([wp.transform_identity()], dtype=wp.transform, device=self.device)
        self.shape_colors = wp.array([wp.vec3(0.72, 0.72, 0.76)], dtype=wp.vec3, device=self.device)
        self.shape_materials = wp.array([wp.vec4(0.75, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=self.device)

        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(2.5, -2.0, 1.6), -20.0, 145.0)

        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.render_ui, position="side")

        self._slice_dirty = True

    def _build_hydroelastic_reference(self, args):
        """Build a tiny hydroelastic scene and fetch the exact immutable field volume."""
        self.mesh.build_sdf(
            max_resolution=int(args.sdf_resolution),
            narrow_band_range=(-float(args.narrow_band), float(args.narrow_band)),
            margin=float(args.sdf_margin),
        )

        cfg_mesh = newton.ModelBuilder.ShapeConfig(
            hydroelastic_type=newton.HydroelasticType.COMPLIANT,
            gap=0.02,
            kh=2.0e8,
            margin=1.0e-5,
        )
        cfg_primitive = newton.ModelBuilder.ShapeConfig(
            hydroelastic_type=newton.HydroelasticType.COMPLIANT,
            sdf_max_resolution=int(args.sdf_resolution),
            sdf_narrow_band_range=(-float(args.narrow_band), float(args.narrow_band)),
            gap=0.02,
            kh=2.0e8,
            margin=1.0e-5,
        )

        builder = newton.ModelBuilder(gravity=0.0)
        body_main = builder.add_body(xform=wp.transform_identity(), label="main_shape")
        self.main_shape_index = builder.add_shape_mesh(body=body_main, mesh=self.mesh, cfg=cfg_mesh)

        # Add a distant compliant shape so hydroelastic pair tables are instantiated.
        body_dummy = builder.add_body(xform=wp.transform(wp.vec3(10.0, 0.0, 0.0), wp.quat_identity()), label="dummy")
        builder.add_shape_sphere(body=body_dummy, radius=0.1, cfg=cfg_primitive)

        model = builder.finalize(device=self.device)
        pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
        hydro = pipeline.hydroelastic_sdf
        if hydro is None:
            raise RuntimeError("Failed to construct hydroelastic pipeline for immutable pressure field visualization.")
        self._hydro_model = model
        self._hydro_pipeline = pipeline
        self._hydro = hydro

        main_sdf_idx = int(model.shape_sdf_index.numpy()[self.main_shape_index])
        if main_sdf_idx < 0:
            raise RuntimeError("Main shape has no SDF index.")

        self.pressure_volume = hydro.pressure_field_volume[main_sdf_idx]
        if self.pressure_volume is None:
            raise RuntimeError("Hydroelastic immutable pressure volume is missing for main shape.")
        self.pressure_volume_id = wp.uint64(int(self.pressure_volume.id))

        sdf_data = model.sdf_data.numpy()[main_sdf_idx]
        self.sdf_center = wp.vec3(sdf_data["center"])
        self.sdf_half_extents = wp.vec3(sdf_data["half_extents"])
        self.mesh.finalize(device=self.device)

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
            kernel=build_slice_points,
            dim=self.capacity,
            inputs=[
                self.resolution,
                self.slice_axis,
                axis_position,
                self.plane_scale,
                self.sdf_center,
                self.sdf_half_extents,
            ],
            outputs=[self.slice_points],
            device=self.device,
        )

        wp.launch(
            kernel=sample_pressure_on_slice,
            dim=self.capacity,
            inputs=[
                self.pressure_volume_id,
                1.0e-8,
                self.slice_points,
            ],
            outputs=[self.slice_inside_flag, self.slice_pressure, self.slice_inside_count],
            device=self.device,
        )

        self.last_slice_count = int(self.slice_inside_count.numpy()[0])
        pressure_grid = self.slice_pressure.numpy().reshape(self.resolution, self.resolution)

        inside = pressure_grid >= 0.0
        normalized = np.full_like(pressure_grid, -1.0, dtype=np.float32)
        if np.any(inside):
            max_p = float(np.max(pressure_grid[inside]))
            if max_p > 1.0e-8:
                normalized[inside] = pressure_grid[inside] / max_p
            else:
                normalized[inside] = 0.0

        self.last_validation_ok, self.last_deep_p05, self.last_zero_fraction, self.last_hole_fraction = (
            validate_slice_field(pressure_grid)
        )
        self.slice_texture = density_to_rgb_image(normalized)
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
        imgui.text("Source: Hydroelastic immutable pressure volume.")
        imgui.text("This is the same field used by contact isosurface extraction.")
        imgui.text("Validation: deep interior percentile, zero-fraction, interior-hole fraction.")

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
        imgui.text(f"Deep Interior P05 (norm): {self.last_deep_p05:.4f}")
        imgui.text(f"Inside Zero Fraction: {self.last_zero_fraction:.4f}")
        imgui.text(f"Interior Hole Fraction: {self.last_hole_fraction:.4f}")
        imgui.text(f"Validation: {'PASS' if self.last_validation_ok else 'FAIL'}")

    def test_final(self):
        self._update_slice()
        assert self.last_slice_count > 0, "Expected non-empty heat-map section for the selected shape."
        assert self.last_validation_ok, "Immutable hydroelastic pressure slice validation failed."


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
