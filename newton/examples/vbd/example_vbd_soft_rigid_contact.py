# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD Soft Contact
#
# Each supported rigid shape type gets its own cloth bag so that
# particle-rigid (soft) contacts can be inspected in isolation.
# The bags are laid out in a row; bodies settle under gravity and are
# then lifted, making it easy to spot which shape type causes
# instability.
#
# Command: python -m newton.examples vbd_soft_rigid_contact
#
###########################################################################

import os

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
from newton import ParticleFlags

PARAMS = {
    "shape_names": ["mesh", "cone", "sphere", "box", "capsule", "cylinder"],
    "shape_size": 0.012,
    "shape_margin": 0.005,
    "soft_contact_creation_margin": 0.01,
    "bag_width": 0.06,
    "bag_depth": 0.06,
    "bag_height": 0.10,
    "bag_res": 10,
    "bag_elevation": 0.30,
    "bag_spacing": 0.10,
    "lift_speed": 0.10,
    "settle_frames": 120,
    "lift_frames": 180,
    "particle_radius": 0.003,
    "fps": 60,
    "sim_substeps": 5,
    "solver_iterations": 15,
    "cloth_density": 0.08,
    "cloth_tri_ke": 1e5,
    "cloth_tri_ka": 1e5,
    "cloth_tri_kd": 1e2,
    "cloth_edge_ke": 50.0,
    "cloth_edge_kd": 5e-1,
    "shape_density": 100.0,
    "shape_ke": 1e5,
    "shape_kd": 1e1,
    "shape_mu": 0.5,
    "soft_contact_ke": 1e5,
    "soft_contact_kd": 1e2,
    "soft_contact_mu": 0.8,
    "gravity": (0.0, 0.0, -9.8),
    "initial_paused": False,
    "body_drop_offset": 0.04,
    "rigid_body_particle_contact_buffer_size": 1024,
    "particle_self_contact_radius_scale": 1.0,
    "particle_self_contact_margin_scale": 2.0,
    "particle_topological_contact_filter_threshold": 3,
}


def _generate_box_bag(half_x, half_y, height, res, z_base):
    """Generate a box-shaped bag (5 faces, open top) as a single merged mesh."""
    cell_x = 2.0 * half_x / res
    cell_y = 2.0 * half_y / res
    cell_z = height / res

    vertex_map = {}
    vertices = []
    faces = []

    def get_or_add_vertex(x, y, z):
        key = (round(x, 6), round(y, 6), round(z, 6))
        if key not in vertex_map:
            vertex_map[key] = len(vertices)
            vertices.append([x, y, z])
        return vertex_map[key]

    def add_quad(v00, v10, v01, v11):
        faces.extend([v00, v10, v01])
        faces.extend([v10, v11, v01])

    # Bottom face
    for i in range(res):
        for j in range(res):
            x0, x1 = -half_x + i * cell_x, -half_x + (i + 1) * cell_x
            y0, y1 = -half_y + j * cell_y, -half_y + (j + 1) * cell_y
            add_quad(
                get_or_add_vertex(x0, y0, z_base),
                get_or_add_vertex(x1, y0, z_base),
                get_or_add_vertex(x0, y1, z_base),
                get_or_add_vertex(x1, y1, z_base),
            )

    # Side walls
    sides = [
        lambda i, j: (-half_x + i * cell_x, -half_y, z_base + j * cell_z, cell_x, 0, cell_z, 0),
        lambda i, j: (-half_x + i * cell_x, half_y, z_base + j * cell_z, cell_x, 0, cell_z, 1),
        lambda i, j: (-half_x, -half_y + i * cell_y, z_base + j * cell_z, 0, cell_y, cell_z, 2),
        lambda i, j: (half_x, -half_y + i * cell_y, z_base + j * cell_z, 0, cell_y, cell_z, 3),
    ]
    for side_fn in sides:
        for i in range(res):
            for j in range(res):
                x0, y0, z0, dx, dy, dz, side = side_fn(i, j)
                if side == 0:
                    add_quad(
                        get_or_add_vertex(x0, y0, z0),
                        get_or_add_vertex(x0 + dx, y0, z0),
                        get_or_add_vertex(x0, y0, z0 + dz),
                        get_or_add_vertex(x0 + dx, y0, z0 + dz),
                    )
                elif side == 1:
                    add_quad(
                        get_or_add_vertex(x0 + dx, y0, z0),
                        get_or_add_vertex(x0, y0, z0),
                        get_or_add_vertex(x0 + dx, y0, z0 + dz),
                        get_or_add_vertex(x0, y0, z0 + dz),
                    )
                elif side == 2:
                    add_quad(
                        get_or_add_vertex(x0, y0 + dy, z0),
                        get_or_add_vertex(x0, y0, z0),
                        get_or_add_vertex(x0, y0 + dy, z0 + dz),
                        get_or_add_vertex(x0, y0, z0 + dz),
                    )
                elif side == 3:
                    add_quad(
                        get_or_add_vertex(x0, y0, z0),
                        get_or_add_vertex(x0, y0 + dy, z0),
                        get_or_add_vertex(x0, y0, z0 + dz),
                        get_or_add_vertex(x0, y0 + dy, z0 + dz),
                    )

    return np.array(vertices, dtype=np.float32), faces


def _load_bear_mesh(target_size):
    bear_path = os.path.join(newton.examples.get_asset_directory(), "bear.usd")
    stage = Usd.Stage.Open(bear_path)
    geom = UsdGeom.Mesh(stage.GetPrimAtPath("/root/bear/bear"))

    points = np.array(geom.GetPointsAttr().Get(), dtype=np.float32)
    indices = np.array(geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)

    center = (points.max(axis=0) + points.min(axis=0)) / 2.0
    points -= center
    extent = (points.max(axis=0) - points.min(axis=0)).max()
    points *= (target_size * 2.0) / extent

    return points, indices.tolist()


@wp.kernel
def lift_pinned_vertices(
    pinned_indices: wp.array[wp.int32],
    original_positions: wp.array[wp.vec3],
    dz: float,
    pos_0: wp.array[wp.vec3],
    pos_1: wp.array[wp.vec3],
):
    tid = wp.tid()
    vi = pinned_indices[tid]
    p = original_positions[tid]
    new_p = wp.vec3(p[0], p[1], p[2] + dz)
    pos_0[vi] = new_p
    pos_1[vi] = new_p


def build_model(builder, params, seed=42):
    shape_names = params["shape_names"]
    n_bags = len(shape_names)
    bag_half_x = params["bag_width"] / 2
    bag_half_y = params["bag_depth"] / 2
    bag_height = params["bag_height"]
    bag_res = params["bag_res"]
    elev = params["bag_elevation"]
    spacing = params["bag_spacing"]
    pr = params["particle_radius"]
    r = params["shape_size"]
    margin = params["shape_margin"]

    # Center the row of bags around x=0
    total_width = (n_bags - 1) * spacing
    x_start = -total_width / 2

    all_top_indices = []
    body_indices = []
    bear_mesh = None

    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = params["shape_density"]
    cfg.ke = params["shape_ke"]
    cfg.kd = params["shape_kd"]
    cfg.mu = params["shape_mu"]
    cfg.has_particle_collision = True
    cfg.margin = margin

    for bag_i, name in enumerate(shape_names):
        bag_cx = x_start + bag_i * spacing

        # Generate and offset bag vertices
        bag_verts, bag_faces = _generate_box_bag(bag_half_x, bag_half_y, bag_height, bag_res, elev)
        bag_verts[:, 0] += bag_cx

        bag_start_particle = len(builder.particle_q)

        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=bag_verts.tolist(),
            indices=bag_faces,
            density=params["cloth_density"],
            tri_ke=params["cloth_tri_ke"],
            tri_ka=params["cloth_tri_ka"],
            tri_kd=params["cloth_tri_kd"],
            edge_ke=params["cloth_edge_ke"],
            edge_kd=params["cloth_edge_kd"],
            particle_radius=pr,
        )

        # Pin top-edge vertices
        z_top = elev + bag_height
        top_mask = np.abs(bag_verts[:, 2] - z_top) < 0.001
        top_global = np.where(top_mask)[0] + bag_start_particle
        all_top_indices.append(top_global)

        # Drop one rigid body centered in this bag
        drop_z = elev + params["body_drop_offset"]
        body = builder.add_body(xform=wp.transform(wp.vec3(bag_cx, 0.0, drop_z), wp.quat_identity()))
        body_indices.append(body)

        if name == "sphere":
            builder.add_shape_sphere(body, radius=r, cfg=cfg)
        elif name == "box":
            builder.add_shape_box(body, hx=r, hy=r, hz=r, cfg=cfg)
        elif name == "capsule":
            builder.add_shape_capsule(body, radius=r * 0.7, half_height=r, cfg=cfg)
        elif name == "cylinder":
            builder.add_shape_cylinder(body, radius=r, half_height=r * 0.5, cfg=cfg)
        elif name == "cone":
            builder.add_shape_cone(body, radius=r, half_height=r, cfg=cfg)
        elif name == "mesh":
            if bear_mesh is None:
                bear_pts, bear_idx = _load_bear_mesh(r)
                bear_mesh = newton.Mesh(bear_pts, np.array(bear_idx, dtype=np.int32))
            builder.add_shape_mesh(body, mesh=bear_mesh, cfg=cfg)

    top_global_indices = np.concatenate(all_top_indices)

    builder.color(include_bending=True)

    return {
        "top_global_indices": top_global_indices,
        "body_indices": body_indices,
        "particle_radius": pr,
    }


def setup_sim(builder, info, params):
    model = builder.finalize()
    model.soft_contact_ke = params["soft_contact_ke"]
    model.soft_contact_kd = params["soft_contact_kd"]
    model.soft_contact_mu = params["soft_contact_mu"]

    top_idx = info["top_global_indices"]
    flags = model.particle_flags.numpy()
    for vi in top_idx:
        flags[vi] = flags[vi] & ~int(ParticleFlags.ACTIVE)
    model.particle_flags = wp.array(flags, dtype=wp.int32)

    pq = model.state().particle_q.numpy()
    pinned_indices = wp.array(top_idx.astype(np.int32), dtype=wp.int32)
    pinned_original = wp.array(pq[top_idx].copy(), dtype=wp.vec3)

    pr = info["particle_radius"]
    solver = newton.solvers.SolverVBD(
        model=model,
        iterations=params["solver_iterations"],
        rigid_body_particle_contact_buffer_size=params["rigid_body_particle_contact_buffer_size"],
        particle_enable_self_contact=False,
        particle_self_contact_radius=pr * params["particle_self_contact_radius_scale"],
        particle_self_contact_margin=pr * params["particle_self_contact_margin_scale"],
        particle_topological_contact_filter_threshold=params["particle_topological_contact_filter_threshold"],
    )

    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_margin=params["soft_contact_creation_margin"]
    )

    return model, solver, pipeline, pinned_indices, pinned_original


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.params = PARAMS
        self.sim_time = 0.0
        self.fps = self.params["fps"]
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = self.params["sim_substeps"]
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.frame = 0
        self.total_frames = self.params["settle_frames"] + self.params["lift_frames"]

        seed = getattr(args, "seed", 42)
        builder = newton.ModelBuilder(gravity=self.params["gravity"])
        self.info = build_model(builder, self.params, seed=seed)
        self.model, self.solver, self.pipeline, self.pinned_indices, self.pinned_original = setup_sim(
            builder, self.info, self.params
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.pipeline.contacts()

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_wireframe = True
        if hasattr(self.viewer, "_paused"):
            self.viewer._paused = self.params["initial_paused"]
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(0.0, -0.8, 0.6), -20.0, 90.0)

    def simulate(self):
        dz = 0.0
        if self.frame > self.params["settle_frames"]:
            dz = self.params["lift_speed"] * (self.frame - self.params["settle_frames"]) * self.frame_dt

        for _ in range(self.sim_substeps):
            wp.launch(
                lift_pinned_vertices,
                dim=self.pinned_indices.shape[0],
                inputs=[self.pinned_indices, self.pinned_original, dz],
                outputs=[self.state_0.particle_q, self.state_1.particle_q],
            )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.frame += 1
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        body_indices = self.info["body_indices"]
        shape_names = self.params["shape_names"]
        elev = self.params["bag_elevation"]

        stable = 0
        for i, bi in enumerate(body_indices):
            z = body_q[bi][2]
            name = shape_names[i]
            if np.isnan(z) or z < elev * 0.5:
                print(f"  UNSTABLE: {name} (body {bi}) z={z:.4f}")
            else:
                stable += 1
                print(f"  ok: {name} (body {bi}) z={z:.4f}")

        assert stable >= len(body_indices) - 1, f"Only {stable}/{len(body_indices)} rigid bodies remained stable"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--seed", type=int, default=42)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
