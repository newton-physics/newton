# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Water simulation and reconstruction used only by the bottle example.

BottleFluid uses XPBD density constraints with one-way shell contact, in
cm/kg/s. BottleSurface clips Newton's native ParticleSurface reconstruction
against the deformed shell, in meters. A classification cap leaves the actual
neck open for both particle contact and surface reconstruction.

Fluid kernels follow Position Based Fluids (2013) and XPBD (2016), equation 18:
https://mmacklin.com/pbf_sig_preprint.pdf and https://mmacklin.com/xpbd.pdf.
"""

import math

import numpy as np
import warp as wp

from newton.geometry import ParticleSurface
from newton.viewer import ViewerRTX


@wp.func
def _poly6(delta: wp.vec3, h: float):
    r2 = wp.dot(delta, delta)
    value = float(0.0)
    if r2 < h * h:
        a = h * h - r2
        value = 315.0 / (64.0 * wp.pi * wp.pow(h, 9.0)) * a * a * a
    return value


@wp.func
def _spiky_gradient(delta: wp.vec3, h: float):
    r = wp.length(delta)
    value = wp.vec3(0.0)
    if r > 1.0e-6 and r < h:
        a = h - r
        value = -45.0 / (wp.pi * wp.pow(h, 6.0)) * a * a * delta / r
    return value


@wp.kernel
def _predict(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    dt: float,
    predicted: wp.array[wp.vec3],
):
    i = wp.tid()
    predicted[i] = positions[i] + dt * velocities[i] + dt * dt * wp.vec3(0.0, 0.0, -981.0)


@wp.kernel
def _density_multipliers(
    grid: wp.uint64,
    positions: wp.array[wp.vec3],
    shell_mesh: wp.uint64,
    h: float,
    volume: float,
    compliance: float,
    multipliers: wp.array[float],
    multiplier_deltas: wp.array[float],
    densities: wp.array[float],
    boundary_gradients: wp.array[wp.vec3],
):
    i = wp.tid()
    p = positions[i]
    density = float(0.0)
    grad_sum = wp.vec3(0.0)
    grad_sq = float(0.0)
    for j in wp.hash_grid_query(grid, p, h):
        delta = p - positions[j]
        density += volume * _poly6(delta, h)
        grad = volume * _spiky_gradient(delta, h)
        grad_sum += grad
        grad_sq += wp.dot(grad, grad)
    boundary_gradient = wp.vec3(0.0)
    query = wp.mesh_query_point_no_sign(shell_mesh, p, h)
    if query.result:
        closest = wp.mesh_eval_position(shell_mesh, query.face, query.u, query.v)
        normal = wp.mesh_eval_face_normal(shell_mesh, query.face)
        offset = p - closest
        distance = wp.length(offset)
        tangent = offset - wp.dot(offset, normal) * normal
        if distance > 1.0e-6 and wp.length(tangent) < 0.1 * h:
            s = wp.clamp(distance / h, 0.0, 1.0)
            # Integrate the Poly6 kernel over the wall's missing half-space.
            # Fixed virtual water completes boundary density without forcing
            # fluid next to a wall to compress to twice its rest density.
            primitive = s - 4.0 / 3.0 * wp.pow(s, 3.0) + 6.0 / 5.0 * wp.pow(s, 5.0)
            primitive += -4.0 / 7.0 * wp.pow(s, 7.0) + wp.pow(s, 9.0) / 9.0
            density += 0.5 - 315.0 / 256.0 * primitive
            boundary_gradient = -315.0 / (256.0 * h) * wp.pow(1.0 - s * s, 4.0) * offset / distance
            grad_sum += boundary_gradient
    grad_sq += wp.dot(grad_sum, grad_sum)
    # Clamp the density constraint at the free surface, as in PBF. Releasing
    # a hard constraint on underfilled samples would pull their neighbors in.
    constraint = wp.max(density - 1.0, 0.0)
    previous = multipliers[i]
    multiplier_delta = -(constraint + compliance * previous) / (grad_sq + compliance + 0.01 / (h * h))
    current = wp.min(previous + multiplier_delta, 0.0)
    multipliers[i] = current
    multiplier_deltas[i] = current - previous
    densities[i] = density
    boundary_gradients[i] = boundary_gradient


@wp.kernel
def _density_corrections(
    grid: wp.uint64,
    positions: wp.array[wp.vec3],
    multiplier_deltas: wp.array[float],
    boundary_gradients: wp.array[wp.vec3],
    h: float,
    volume: float,
    corrections: wp.array[wp.vec3],
):
    i = wp.tid()
    p = positions[i]
    correction = multiplier_deltas[i] * boundary_gradients[i]
    for j in wp.hash_grid_query(grid, p, h):
        if i != j:
            delta = p - positions[j]
            correction += volume * (multiplier_deltas[i] + multiplier_deltas[j]) * _spiky_gradient(delta, h)
    # Limit each Jacobi update to avoid overshooting neighboring constraints.
    length = wp.length(correction)
    if length > 0.2 * h:
        correction *= 0.2 * h / length
    corrections[i] = correction


@wp.kernel
def _shell_contacts(
    mesh: wp.uint64,
    shell_indices: wp.array[int],
    shell_previous: wp.array[wp.vec3],
    previous: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    corrections: wp.array[wp.vec3],
    inside_bottle: wp.array[int],
    mouth_height: float,
    radius: float,
    search_radius: float,
    floor: float,
):
    i = wp.tid()
    p = positions[i] + corrections[i]
    motion = p - previous[i]
    travel = wp.length(motion)
    if travel > 1.0e-6:
        direction = motion / travel
        ray = wp.mesh_query_ray(mesh, previous[i], direction, travel)
        if ray.result:
            normal_ray = wp.mesh_eval_face_normal(mesh, ray.face)
            side_ray = wp.where(wp.dot(direction, normal_ray) > 0.0, -1.0, 1.0)
            target = previous[i] + ray.t * direction + radius * side_ray * normal_ray
            p = target
    if p[2] > mouth_height + radius:
        inside_bottle[i] = 0
    query = wp.mesh_query_point_no_sign(mesh, p, search_radius)
    if query.result:
        a = shell_indices[3 * query.face]
        b = shell_indices[3 * query.face + 1]
        c = shell_indices[3 * query.face + 2]
        u = query.u
        v = query.v
        w = 1.0 - u - v
        closest = wp.mesh_eval_position(mesh, query.face, u, v)
        normal = wp.mesh_eval_face_normal(mesh, query.face)
        closest_previous = u * shell_previous[a] + v * shell_previous[b] + w * shell_previous[c]
        old_delta = previous[i] - closest_previous
        normal_previous = wp.normalize(
            wp.cross(shell_previous[b] - shell_previous[a], shell_previous[c] - shell_previous[a])
        )
        side = wp.where(wp.dot(old_delta, normal_previous) >= 0.0, 1.0, -1.0)
        # A local closest-face sign is ambiguous at concave bottle ribs.
        # Interior particles retain their side until they cross the open mouth.
        if inside_bottle[i] != 0:
            side = -1.0
        delta = p - closest
        signed_distance = wp.dot(delta, normal)
        tangent = delta - signed_distance * normal
        # Only the finite triangle blocks a crossing; the open mouth stays open.
        if wp.length(tangent) < radius and side * signed_distance < radius:
            correction = (radius - side * signed_distance) * side * normal
            p += correction
    p[2] = wp.max(p[2], floor + radius)
    positions[i] = p


@wp.kernel
def _update_velocities(
    previous: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    dt: float,
    velocities: wp.array[wp.vec3],
):
    i = wp.tid()
    velocities[i] = (positions[i] - previous[i]) / dt


@wp.kernel
def _xsph(
    grid: wp.uint64,
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    h: float,
    volume: float,
    dt: float,
    floor: float,
    radius: float,
    output: wp.array[wp.vec3],
):
    i = wp.tid()
    velocity = velocities[i]
    change = wp.vec3(0.0)
    for j in wp.hash_grid_query(grid, positions[i], h):
        change += volume * _poly6(positions[i] - positions[j], h) * (velocities[j] - velocity)
    # XSPH is a numerical velocity filter, not a calibrated water viscosity.
    velocity += wp.min(0.12, 48.0 * dt) * change
    if positions[i][2] < floor + radius + 0.001:
        # An empirical wet-table drag arrests escaped droplets and puddles.
        velocity[0] /= 1.0 + 4.0 * dt
        velocity[1] /= 1.0 + 4.0 * dt
        velocity[2] = wp.max(velocity[2], 0.0)
    output[i] = velocity


class BottleFluid:
    """Keep fluid state separate from the VBD shell model.

    The shell prescribes the fluid boundary (one-way coupling); pressure does
    not feed back onto the shell. Lengths use cm, masses kg and time seconds.
    The bulk modulus is supplied in Pa; infinity gives zero compliance.
    """

    def __init__(
        self,
        positions,
        spacing,
        shell_positions,
        indices,
        *,
        floor,
        mouth_height,
        iterations,
        device,
        bulk_modulus=math.inf,
    ):
        if math.isnan(bulk_modulus) or bulk_modulus <= 0.0:
            raise ValueError("bulk_modulus must be positive or infinity [Pa]")
        self.device = device
        self.iterations = iterations
        self.spacing = spacing
        self.h = 2.6 * spacing
        self.volume = spacing**3
        self.mass = 1000.0 / 100.0**3 * self.volume
        # Uniform masses factor out of XPBD. Convert K [Pa] to kg/(cm s^2).
        self.compliance = self.mass / (0.01 * bulk_modulus * self.volume)
        self.radius = 0.4 * spacing
        self.floor = floor
        self.mouth_height = mouth_height
        self.inside_bottle = wp.ones(len(positions), dtype=int, device=device)
        self.positions = wp.array(positions, dtype=wp.vec3, device=device)
        self.predicted = wp.empty_like(self.positions)
        self.velocities = wp.zeros_like(self.positions)
        self.velocity_temp = wp.empty_like(self.positions)
        self.corrections = wp.empty_like(self.positions)
        self.multipliers = wp.zeros(len(positions), dtype=float, device=device)
        self.multiplier_deltas = wp.empty_like(self.multipliers)
        self.densities = wp.empty_like(self.multipliers)
        self.boundary_gradients = wp.empty_like(self.positions)
        self.grid = wp.HashGrid(64, 64, 128, device=device)
        self.grid.reserve(len(positions))
        self.shell_indices = wp.array(np.asarray(indices).flatten(), dtype=int, device=device)
        self.shell_points = wp.clone(shell_positions)
        self.shell_previous = wp.clone(shell_positions)
        self.shell_mesh = wp.Mesh(points=self.shell_points, indices=self.shell_indices)
        # Sphere volume agrees with the fluid's physical particle volume.
        self.surface_radii = wp.full(
            len(positions),
            (3.0 * self.volume / (4.0 * math.pi)) ** (1.0 / 3.0) * 0.01,
            dtype=float,
            device=device,
        )

    def step(self, shell_positions, dt):
        wp.copy(self.shell_previous, self.shell_points)
        wp.copy(self.shell_points, shell_positions)
        self.shell_mesh.refit()
        count = len(self.positions)
        self.multipliers.zero_()
        wp.launch(_predict, count, inputs=[self.positions, self.velocities, dt, self.predicted], device=self.device)
        for _ in range(self.iterations):
            self.grid.build(self.predicted, self.h)
            wp.launch(
                _density_multipliers,
                count,
                inputs=[
                    self.grid.id,
                    self.predicted,
                    self.shell_mesh.id,
                    self.h,
                    self.volume,
                    self.compliance / (dt * dt),
                    self.multipliers,
                    self.multiplier_deltas,
                    self.densities,
                    self.boundary_gradients,
                ],
                device=self.device,
            )
            wp.launch(
                _density_corrections,
                count,
                inputs=[
                    self.grid.id,
                    self.predicted,
                    self.multiplier_deltas,
                    self.boundary_gradients,
                    self.h,
                    self.volume,
                    self.corrections,
                ],
                device=self.device,
            )
            wp.launch(
                _shell_contacts,
                count,
                inputs=[
                    self.shell_mesh.id,
                    self.shell_indices,
                    self.shell_previous,
                    self.positions,
                    self.predicted,
                    self.corrections,
                    self.inside_bottle,
                    self.mouth_height,
                    self.radius,
                    2.0 * self.h,
                    self.floor,
                ],
                device=self.device,
            )
        wp.launch(
            _update_velocities,
            count,
            inputs=[self.positions, self.predicted, dt, self.velocities],
            device=self.device,
        )
        self.grid.build(self.predicted, self.h)
        wp.launch(
            _xsph,
            count,
            inputs=[
                self.grid.id,
                self.predicted,
                self.velocities,
                self.h,
                self.volume,
                dt,
                self.floor,
                self.radius,
                self.velocity_temp,
            ],
            device=self.device,
        )
        wp.copy(self.velocities, self.velocity_temp)
        wp.copy(self.positions, self.predicted)


def compute_normals(vertices: wp.array[wp.vec3], indices: wp.array[int], normals: wp.array[wp.vec3]):
    """Reuse GPU output buffers for area-weighted bottle and liquid normals."""
    normals.zero_()
    wp.launch(_accumulate_normals, len(indices) // 3, inputs=[vertices, indices, normals], device=vertices.device)
    wp.launch(_normalize_normals, len(normals), inputs=[normals], device=vertices.device)


@wp.kernel
def _particle_sides(mesh: wp.uint64, positions: wp.array[wp.vec3], sides: wp.array[float]):
    i = wp.tid()
    query = wp.mesh_query_point_sign_winding_number(mesh, positions[i], 0.05)
    sides[i] = wp.where(query.result, query.sign, 1.0)


@wp.kernel
def _confine_field(
    volume: wp.uint64,
    coordinates: wp.array[wp.vec3i],
    field: wp.array[float],
    mesh: wp.uint64,
    shell_mesh: wp.uint64,
    shell_triangle_count: int,
    grid: wp.uint64,
    positions: wp.array[wp.vec3],
    sides: wp.array[float],
    voxel_size: float,
    kernel_radius: float,
    search_radius: float,
    margin: float,
    floor: float,
):
    i = wp.tid()
    ijk = coordinates[i]
    index = wp.volume_lookup_index(volume, ijk[0], ijk[1], ijk[2])
    if index < 0:
        return
    point = voxel_size * wp.vec3(float(ijk[0]), float(ijk[1]), float(ijk[2]))
    value = wp.max(field[index], (floor - point[2]) / kernel_radius)
    field[index] = value
    query = wp.mesh_query_point_no_sign(shell_mesh, point, search_radius)
    if not query.result:
        return
    signed_query = wp.mesh_query_point_sign_winding_number(mesh, point, search_radius)
    if signed_query.face >= shell_triangle_count:
        # The actual opening moves and tilts with the shell. Its temporary
        # classification cap must never act as a water boundary.
        return
    closest = wp.mesh_eval_position(shell_mesh, query.face, query.u, query.v)
    distance = wp.length(point - closest)
    # Far inside the bottle, or far outside the liquid, the zero contour cannot
    # intersect the wall. Avoid unnecessary neighbor queries in those regions.
    if distance > margin + 2.0 * voxel_size and (signed_query.sign < 0.0 or value > 0.0):
        return
    nearest = int(-1)
    nearest_distance = search_radius * search_radius
    for particle in wp.hash_grid_query(grid, point, search_radius):
        delta = point - positions[particle]
        squared_distance = wp.dot(delta, delta)
        if squared_distance < nearest_distance:
            nearest = particle
            nearest_distance = squared_distance
    if nearest >= 0:
        # Negative field values denote liquid. Intersect with the interior for
        # contained particles and with the exterior for actual spilled water.
        boundary = (margin - sides[nearest] * signed_query.sign * distance) / kernel_radius
        field[index] = wp.max(value, boundary)


class BottleSurface:
    """Clip an anisotropic density field before marching cubes, in SI units.

    A temporary neck cap supplies a robust winding-number sign. It contributes
    no collision or rendered geometry, and its faces never clip the liquid.
    The nearest fluid particle determines which side of the PET wall to keep.
    """

    def __init__(
        self,
        shell_positions,
        shell_triangles,
        *,
        segments,
        spacing,
        voxel_size,
        margin,
        floor,
        max_grid_cells,
        device,
    ):
        self.device = device
        self.voxel_size = voxel_size
        self.kernel_radius = 3.0 * spacing
        self.search_radius = 4.0 * spacing + 2.0 * voxel_size
        # Resolve the inset even when the reconstruction voxels are coarser
        # than the physical wall thickness.
        self.margin = max(margin, 0.5 * voxel_size)
        self.floor = floor
        self.shell_triangle_count = len(shell_triangles)
        top = len(shell_positions) - segments
        cap = np.array([[top, top + j, top + j + 1] for j in range(1, segments - 1)], dtype=np.int32)
        indices = np.concatenate((shell_triangles, cap)).flatten()
        self.mesh = wp.Mesh(
            points=shell_positions,
            indices=wp.array(indices, dtype=int, device=device),
            support_winding_number=True,
        )
        self.shell_mesh = wp.Mesh(
            points=shell_positions,
            indices=wp.array(np.asarray(shell_triangles).flatten(), dtype=int, device=device),
        )
        self.grid = wp.HashGrid(64, 64, 128, device=device)
        self.sides = None
        self.coordinates = None
        self.surface = ParticleSurface(
            voxel_size=voxel_size,
            kernel_radius=self.kernel_radius,
            threshold=0.25,
            smooth_lambda=0.0,
            anisotropic=True,
            anisotropy_min_neighbors=8,
            anisotropy_ratio=4.0,
            anisotropy_binning=True,
            anisotropy_strength=0.6,
            kernel_scale=0.65,
            field_mode="sdf",
            field_smooth_iterations=1,
            # Smoothing after clipping would move vertices back through the wall.
            mesh_smooth_iterations=0,
            max_grid_cells=max_grid_cells,
            device=device,
        )

    def extract(self, positions, radii):
        self.mesh.refit()
        self.shell_mesh.refit()
        if self.sides is None or len(self.sides) != len(positions):
            self.sides = wp.empty(len(positions), dtype=float, device=self.device)
            self.grid.reserve(len(positions))
        wp.launch(_particle_sides, len(positions), inputs=[self.mesh.id, positions, self.sides], device=self.device)
        self.grid.build(positions, self.kernel_radius)
        field = self.surface.update_field(positions, radii)
        if field is not None:
            count = field.volume.get_voxel_count()
            if self.coordinates is None or len(self.coordinates) < count:
                self.coordinates = wp.empty(2 * count, dtype=wp.vec3i, device=self.device)
            field.volume.get_voxels(out=self.coordinates)
            wp.launch(
                _confine_field,
                count,
                inputs=[
                    field.volume.id,
                    self.coordinates,
                    field.voxel_data,
                    self.mesh.id,
                    self.shell_mesh.id,
                    self.shell_triangle_count,
                    self.grid.id,
                    positions,
                    self.sides,
                    self.voxel_size,
                    self.kernel_radius,
                    self.search_radius,
                    self.margin,
                    self.floor,
                ],
                device=self.device,
            )
        mesh = self.surface.resurface()
        vertices, indices, normals = mesh.to_arrays()
        if vertices is not None:
            # A sub-voxel crease can cross a marching-cubes triangle even when
            # all field nodes are on the correct side. Guard the final vertices.
            wp.launch(
                _confine_vertices,
                len(vertices),
                inputs=[
                    self.mesh.id,
                    self.shell_mesh.id,
                    self.shell_triangle_count,
                    self.grid.id,
                    positions,
                    self.sides,
                    vertices,
                    self.search_radius,
                    self.margin,
                    self.floor,
                ],
                device=self.device,
            )
            compute_normals(vertices, indices, normals)
        return vertices, indices, normals


@wp.kernel
def _confine_vertices(
    mesh: wp.uint64,
    shell_mesh: wp.uint64,
    shell_triangle_count: int,
    grid: wp.uint64,
    positions: wp.array[wp.vec3],
    sides: wp.array[float],
    vertices: wp.array[wp.vec3],
    search_radius: float,
    margin: float,
    floor: float,
):
    i = wp.tid()
    point = vertices[i]
    point[2] = wp.max(point[2], floor)
    nearest = int(-1)
    nearest_distance = search_radius * search_radius
    for particle in wp.hash_grid_query(grid, point, search_radius):
        squared_distance = wp.dot(point - positions[particle], point - positions[particle])
        if squared_distance < nearest_distance:
            nearest = particle
            nearest_distance = squared_distance
    if nearest < 0:
        vertices[i] = point
        return
    side = sides[nearest]
    # Keep the emitting component fixed while correcting the vertex. Choosing
    # its side again after displacement can switch to a different fluid source.
    for _ in range(4):
        query = wp.mesh_query_point_no_sign(shell_mesh, point, search_radius)
        if not query.result:
            break
        signed_query = wp.mesh_query_point_sign_winding_number(mesh, point, search_radius)
        if signed_query.face >= shell_triangle_count:
            break
        closest = wp.mesh_eval_position(shell_mesh, query.face, query.u, query.v)
        offset = point - closest
        distance = wp.length(offset)
        if side * signed_query.sign * distance >= margin:
            break
        normal = wp.mesh_eval_face_normal(shell_mesh, query.face)
        if distance > 1.0e-8:
            normal = signed_query.sign * offset / distance
        point = closest + side * (margin + 1.0e-6) * normal
        point[2] = wp.max(point[2], floor)
    vertices[i] = point


@wp.kernel
def _accumulate_normals(vertices: wp.array[wp.vec3], indices: wp.array[int], normals: wp.array[wp.vec3]):
    i = wp.tid()
    a = indices[3 * i]
    b = indices[3 * i + 1]
    c = indices[3 * i + 2]
    normal = wp.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
    wp.atomic_add(normals, a, normal)
    wp.atomic_add(normals, b, normal)
    wp.atomic_add(normals, c, normal)


@wp.kernel
def _normalize_normals(normals: wp.array[wp.vec3]):
    i = wp.tid()
    normals[i] = wp.normalize(normals[i])


def bind_water_material(viewer, water_color):
    """Bind dielectric water before the RTX stage is first loaded."""
    if not isinstance(viewer, ViewerRTX):
        return False

    from pxr import Sdf, UsdShade  # noqa: PLC0415

    water = viewer.stage.GetPrimAtPath("/root/water/surface")
    if not water:
        return False

    # MDL transmission traces light through the closed liquid volume.
    # Preview-surface opacity only removes coverage; it does not model refraction.
    material = UsdShade.Material.Define(viewer.stage, "/root/Materials/BottleWater")
    shader = UsdShade.Shader.Define(viewer.stage, "/root/Materials/BottleWater/Glass")
    shader.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(water_color)
    shader.CreateInput("reflection_color", Sdf.ValueTypeNames.Color3f).Set(water_color)
    shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.333)
    shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(0.02)
    shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(False)
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    material.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    material.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    UsdShade.MaterialBindingAPI.Apply(water).Bind(material)
    # The path tracer does not resolve focused caustics for these droplets.
    # Let direct illumination through instead of casting opaque glass shadows.
    water.CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
    return True
