# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD Rigid Drop
#
# Rigid bodies of all supported shape types (sphere, box, capsule,
# cylinder, cone, mesh bear) are dropped into a static rigid box
# container and settle under gravity.  This exercises pure rigid-rigid
# contact handling in the VBD solver at a small (centimeter) scale.
#
# Command: python -m newton.examples vbd_rigid_rigid_contact
#
###########################################################################

import os

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples

PARAMS = {
    "shape_names": ["mesh", "cone", "sphere", "box", "capsule", "cylinder"],
    "shape_size": 0.042,
    "shape_margin": 0.005,
    "box_width_scale": 10.0,
    "box_depth_scale": 10.0,
    "box_height_scale": 5.0,
    "box_elevation": 0.30,
    "box_wall_thickness": 0.005,
    "fps": 60,
    "sim_substeps": 5,
    "solver_iterations": 15,
    "settle_frames": 300,
    "shape_density": 100.0,
    "shape_ke": 1e3,
    "shape_kd": 0,
    "shape_mu": 0.5,
    "container_ke": 1e3,
    "container_kd": 0,
    "container_mu": 0.8,
    "gravity": (0.0, 0.0, -9.8),
    "initial_paused": True,
    "body_drop_offset_scale": 4.0,
    "body_drop_spacing_scale": 3.0,
    "rigid_body_contact_buffer_size": 256,
}


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


def build_model(builder, params, seed=42):
    rng = np.random.default_rng(seed)

    r = params["shape_size"]
    hx = r * params["box_width_scale"] / 2
    hy = r * params["box_depth_scale"] / 2
    hz = r * params["box_height_scale"]
    elev = params["box_elevation"]
    t = params["box_wall_thickness"]

    container_cfg = newton.ModelBuilder.ShapeConfig()
    container_cfg.ke = params["container_ke"]
    container_cfg.kd = params["container_kd"]
    container_cfg.mu = params["container_mu"]
    container_cfg.margin = params["shape_margin"]

    # Floor
    builder.add_shape_box(
        -1,
        wp.transform(wp.vec3(0.0, 0.0, elev - t / 2), wp.quat_identity()),
        hx=hx + t,
        hy=hy + t,
        hz=t / 2,
        cfg=container_cfg,
    )
    # Front wall (-Y)
    builder.add_shape_box(
        -1,
        wp.transform(wp.vec3(0.0, -(hy + t / 2), elev + hz / 2), wp.quat_identity()),
        hx=hx + t,
        hy=t / 2,
        hz=hz / 2,
        cfg=container_cfg,
    )
    # Back wall (+Y)
    builder.add_shape_box(
        -1,
        wp.transform(wp.vec3(0.0, hy + t / 2, elev + hz / 2), wp.quat_identity()),
        hx=hx + t,
        hy=t / 2,
        hz=hz / 2,
        cfg=container_cfg,
    )
    # Left wall (-X)
    builder.add_shape_box(
        -1,
        wp.transform(wp.vec3(-(hx + t / 2), 0.0, elev + hz / 2), wp.quat_identity()),
        hx=t / 2,
        hy=hy,
        hz=hz / 2,
        cfg=container_cfg,
    )
    # Right wall (+X)
    builder.add_shape_box(
        -1,
        wp.transform(wp.vec3(hx + t / 2, 0.0, elev + hz / 2), wp.quat_identity()),
        hx=t / 2,
        hy=hy,
        hz=hz / 2,
        cfg=container_cfg,
    )

    # Rigid bodies
    margin = params["shape_margin"]
    interior_x = hx - r - margin * 2
    interior_y = hy - r - margin * 2
    min_spacing = r * 2 + margin * 3
    body_indices = []
    positions = []

    bear_mesh = None
    shape_names = params["shape_names"]

    for i in range(len(shape_names)):
        if shape_names[i] == "mesh":
            positions.append((0.0, 0.0))
        else:
            for _ in range(200):
                x = rng.uniform(-interior_x, interior_x)
                y = rng.uniform(-interior_y, interior_y)
                ok = all(np.sqrt((x - px) ** 2 + (y - py) ** 2) >= min_spacing for px, py in positions)
                if ok:
                    positions.append((x, y))
                    break
            else:
                positions.append((x, y))

    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = params["shape_density"]
    cfg.ke = params["shape_ke"]
    cfg.kd = params["shape_kd"]
    cfg.mu = params["shape_mu"]
    cfg.margin = margin

    for i, name in enumerate(shape_names):
        px, py = positions[i]
        drop_z = elev + r * params["body_drop_offset_scale"] + i * r * params["body_drop_spacing_scale"]

        body = builder.add_body(xform=wp.transform(wp.vec3(px, py, drop_z), wp.quat_identity()))
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

    builder.color()

    return {
        "body_indices": body_indices,
    }


def setup_sim(builder, params):
    model = builder.finalize()

    solver = newton.solvers.SolverVBD(
        model=model,
        iterations=params["solver_iterations"],
        rigid_body_contact_buffer_size=params["rigid_body_contact_buffer_size"],
        # rigid_contact_hard=False,
    )

    return model, solver


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

        seed = getattr(args, "seed", 42)
        builder = newton.ModelBuilder(gravity=self.params["gravity"])
        self.info = build_model(builder, self.params, seed=seed)
        self.model, self.solver = setup_sim(builder, self.params)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_wireframe = True
        if hasattr(self.viewer, "_paused"):
            self.viewer._paused = self.params["initial_paused"]
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(0.35, -0.35, 0.55), -25.0, 135.0)

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.frame += 1
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        body_indices = self.info["body_indices"]
        elev = self.params["box_elevation"]
        box_height = self.params["shape_size"] * self.params["box_height_scale"]

        settled = 0
        for bi in body_indices:
            z = body_q[bi][2]
            if not np.isnan(z) and z > elev and z < elev + box_height:
                settled += 1

        assert settled >= len(body_indices) - 1, (
            f"Only {settled}/{len(body_indices)} rigid bodies settled inside the box"
        )

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
