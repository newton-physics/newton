#!/usr/bin/env python3
"""Render tendon/cable example scenes headlessly and save frames as JPGs."""

import functools
import os

import numpy as np
import warp as wp

import newton
from newton._src.sim.builder import Axis
from newton._src.sim.tendon import TendonLinkType
from newton.viewer import ViewerGL

os.environ["DISPLAY"] = ":99"
wp.init()

OUTPUT_DIR = os.path.expanduser("~/reports/cable-sim-research")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WIDTH, HEIGHT = 960, 720


def save_frame(viewer, path):
    from PIL import Image

    frame = viewer.get_frame()
    img = Image.fromarray(frame.numpy(), mode="RGB")
    img.save(path, quality=92)
    print(f"  saved {path}")


def get_cable_lines(solver, model, state):
    """Build line segment arrays for cable visualization."""
    att_l = solver.tendon_seg_attachment_l.numpy()
    att_r = solver.tendon_seg_attachment_r.numpy()
    n = model.tendon_segment_count

    starts_list = []
    ends_list = []
    for i in range(n):
        starts_list.append(att_l[i])
        ends_list.append(att_r[i])

    # also draw wrap arcs as polylines
    tendon_start = model.tendon_start.numpy()
    link_type = model.tendon_link_type.numpy()
    link_body = model.tendon_link_body.numpy()
    link_offset = model.tendon_link_offset.numpy()
    link_axis = model.tendon_link_axis.numpy()
    body_q = state.body_q.numpy()

    seg = 0
    for t in range(model.tendon_count):
        start = tendon_start[t]
        end = tendon_start[t + 1]
        num_links = end - start
        for i in range(start + 1, end - 1):
            if link_type[i] == int(TendonLinkType.ROLLING):
                b = link_body[i]
                pose = body_q[b]
                p = pose[:3]
                q = pose[3:]
                off = link_offset[i]
                ax = link_axis[i]
                t2 = 2.0 * np.cross(q[:3], off)
                center = off + q[3] * t2 + np.cross(q[:3], t2) + p
                t2n = 2.0 * np.cross(q[:3], ax)
                normal = ax + q[3] * t2n + np.cross(q[:3], t2n)

                seg_left = seg + (i - start) - 1
                seg_right = seg + (i - start)
                pt_dep = att_r[seg_left]
                pt_arr = att_l[seg_right]

                # draw arc from departure to arrival
                r_dep = pt_dep - center
                r_arr = pt_arr - center
                cross_val = np.dot(np.cross(r_dep, r_arr), normal)
                dot_val = np.dot(r_dep, r_arr)
                total_angle = np.arctan2(cross_val, dot_val)

                n_arc = max(8, int(abs(total_angle) / 0.2))
                for j in range(n_arc):
                    frac0 = j / n_arc
                    frac1 = (j + 1) / n_arc
                    angle0 = frac0 * total_angle
                    angle1 = frac1 * total_angle
                    # Rodrigues rotation of r_dep around normal
                    c0, s0 = np.cos(angle0), np.sin(angle0)
                    p0 = center + r_dep * c0 + np.cross(normal, r_dep) * s0
                    c1, s1 = np.cos(angle1), np.sin(angle1)
                    p1 = center + r_dep * c1 + np.cross(normal, r_dep) * s1
                    starts_list.append(p0)
                    ends_list.append(p1)

        seg += num_links - 1

    starts = wp.array(np.array(starts_list, dtype=np.float32), dtype=wp.vec3)
    ends = wp.array(np.array(ends_list, dtype=np.float32), dtype=wp.vec3)
    return starts, ends


def render_scene(viewer, solver, model, state, sim_time, cable_color=(1.0, 0.3, 0.1)):
    viewer.begin_frame(sim_time)
    viewer.log_state(state)
    starts, ends = get_cable_lines(solver, model, state)
    viewer.log_lines("cable", starts, ends, colors=cable_color, width=0.008)
    viewer.end_frame()


# ─────────────────────────────────────────────────────────────────────
# Example 1: Simple cable — single tendon between kinematic anchor and
# dynamic weight
# ─────────────────────────────────────────────────────────────────────


def build_simple_cable():
    builder = newton.ModelBuilder(up_axis=Axis.Y, gravity=-9.81)

    anchor = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 2.5, 0.0), q=wp.quat_identity()),
        mass=0.0,
        is_kinematic=True,
    )
    builder.add_shape_sphere(anchor, radius=0.04)

    # offset horizontally to create pendulum swing
    weight = builder.add_body(
        xform=wp.transform(p=wp.vec3(1.2, 2.5, 0.0), q=wp.quat_identity()),
        mass=2.0,
    )
    builder.add_shape_box(weight, hx=0.1, hy=0.1, hz=0.1)

    axis = (0.0, 0.0, 1.0)
    builder.add_tendon()
    builder.add_tendon_link(
        body=anchor,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.0),
        axis=axis,
    )
    builder.add_tendon_link(
        body=weight,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.1, 0.0),
        axis=axis,
        compliance=1.0e-5,
        damping=0.1,
        rest_length=-1.0,
    )

    return builder.finalize()


# ─────────────────────────────────────────────────────────────────────
# Example 2: Atwood machine — two weights over a frictionless pulley
# ─────────────────────────────────────────────────────────────────────


def build_atwood():
    builder = newton.ModelBuilder(up_axis=Axis.Y, gravity=-9.81)

    pulley_radius = 0.15
    pulley = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 2.5, 0.0), q=wp.quat_identity()),
        mass=0.0,
        is_kinematic=True,
    )
    builder.add_shape_cylinder(pulley, radius=pulley_radius, half_height=0.04)

    left = builder.add_body(
        xform=wp.transform(p=wp.vec3(-0.5, 1.0, 0.0), q=wp.quat_identity()),
        mass=1.0,
    )
    builder.add_shape_box(left, hx=0.08, hy=0.08, hz=0.08)

    right = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.5, 1.0, 0.0), q=wp.quat_identity()),
        mass=3.0,
    )
    builder.add_shape_box(right, hx=0.12, hy=0.12, hz=0.12)

    axis = (0.0, 0.0, 1.0)
    builder.add_tendon()
    builder.add_tendon_link(
        body=left,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.08, 0.0),
        axis=axis,
    )
    builder.add_tendon_link(
        body=pulley,
        link_type=int(TendonLinkType.ROLLING),
        radius=pulley_radius,
        orientation=-1,
        mu=0.0,
        offset=(0.0, 0.0, 0.0),
        axis=axis,
        compliance=1.0e-5,
        damping=0.1,
        rest_length=-1.0,
    )
    builder.add_tendon_link(
        body=right,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.12, 0.0),
        axis=axis,
        compliance=1.0e-5,
        damping=0.1,
        rest_length=-1.0,
    )

    return builder.finalize()


# ─────────────────────────────────────────────────────────────────────
# Example 3: Compound pulley — two pulleys at different heights
# ─────────────────────────────────────────────────────────────────────


def build_compound():
    builder = newton.ModelBuilder(up_axis=Axis.Y, gravity=-9.81)

    r1 = 0.12
    r2 = 0.10
    pulley1 = builder.add_body(
        xform=wp.transform(p=wp.vec3(-0.4, 2.8, 0.0), q=wp.quat_identity()),
        mass=0.0,
        is_kinematic=True,
    )
    builder.add_shape_cylinder(pulley1, radius=r1, half_height=0.04)

    pulley2 = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.4, 2.4, 0.0), q=wp.quat_identity()),
        mass=0.0,
        is_kinematic=True,
    )
    builder.add_shape_cylinder(pulley2, radius=r2, half_height=0.04)

    left = builder.add_body(
        xform=wp.transform(p=wp.vec3(-0.7, 1.0, 0.0), q=wp.quat_identity()),
        mass=1.5,
    )
    builder.add_shape_box(left, hx=0.09, hy=0.09, hz=0.09)

    right = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.7, 1.0, 0.0), q=wp.quat_identity()),
        mass=4.0,
    )
    builder.add_shape_box(right, hx=0.13, hy=0.13, hz=0.13)

    axis = (0.0, 0.0, 1.0)
    builder.add_tendon()
    builder.add_tendon_link(
        body=left,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.09, 0.0),
        axis=axis,
    )
    builder.add_tendon_link(
        body=pulley1,
        link_type=int(TendonLinkType.ROLLING),
        radius=r1,
        orientation=-1,
        mu=0.0,
        offset=(0.0, 0.0, 0.0),
        axis=axis,
        compliance=1.0e-5,
        damping=0.1,
        rest_length=-1.0,
    )
    builder.add_tendon_link(
        body=pulley2,
        link_type=int(TendonLinkType.ROLLING),
        radius=r2,
        orientation=-1,
        mu=0.0,
        offset=(0.0, 0.0, 0.0),
        axis=axis,
        compliance=1.0e-5,
        damping=0.1,
        rest_length=-1.0,
    )
    builder.add_tendon_link(
        body=right,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.13, 0.0),
        axis=axis,
        compliance=1.0e-5,
        damping=0.1,
        rest_length=-1.0,
    )

    return builder.finalize()


@functools.cache
def get_viewer():
    return ViewerGL(width=WIDTH, height=HEIGHT, headless=True)


# ─────────────────────────────────────────────────────────────────────
# Simulation + rendering
# ─────────────────────────────────────────────────────────────────────


def simulate_and_render(
    name, model, cam_pos, cam_pitch, cam_yaw, n_frames=120, substeps=16, cable_color=(1.0, 0.3, 0.1)
):
    print(f"\n=== {name} ===")
    solver = newton.solvers.SolverXPBD(model, iterations=8, joint_linear_relaxation=0.8)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    fps = 60
    frame_dt = 1.0 / fps
    sim_dt = frame_dt / substeps

    viewer = get_viewer()
    viewer.set_model(model)
    viewer.set_camera(pos=cam_pos, pitch=cam_pitch, yaw=cam_yaw)
    viewer.renderer.show_wireframe_overlay = True

    capture_frames = [0, n_frames // 3, 2 * n_frames // 3, n_frames - 1]

    for frame in range(n_frames):
        for _ in range(substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

        if frame in capture_frames:
            render_scene(viewer, solver, model, state_0, frame * frame_dt, cable_color)
            idx = capture_frames.index(frame)
            labels = ["t0", "t1", "t2", "t3"]
            path = os.path.join(OUTPUT_DIR, f"{name}_{labels[idx]}.jpg")
            save_frame(viewer, path)

    q = state_0.body_q.numpy()
    print("  final body positions:")
    for i in range(model.body_count):
        print(f"    body {i}: ({q[i][0]:.3f}, {q[i][1]:.3f}, {q[i][2]:.3f})")


if __name__ == "__main__":
    # Example 1: Simple cable (pendulum)
    model = build_simple_cable()
    simulate_and_render(
        "cable",
        model,
        cam_pos=wp.vec3(0.0, 1.5, 5.0),
        cam_pitch=-5.0,
        cam_yaw=-90.0,
        n_frames=120,
        cable_color=(1.0, 0.4, 0.1),
    )

    # Example 2: Atwood machine
    model = build_atwood()
    simulate_and_render(
        "atwood",
        model,
        cam_pos=wp.vec3(0.0, 1.5, 4.0),
        cam_pitch=-5.0,
        cam_yaw=-90.0,
        n_frames=120,
        cable_color=(0.9, 0.2, 0.2),
    )

    # Example 3: Compound pulley
    model = build_compound()
    simulate_and_render(
        "compound",
        model,
        cam_pos=wp.vec3(0.0, 1.8, 4.5),
        cam_pitch=-5.0,
        cam_yaw=-90.0,
        n_frames=120,
        cable_color=(0.2, 0.7, 1.0),
    )

    print(f"\nAll renders saved to {OUTPUT_DIR}/")
