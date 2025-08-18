#!/usr/bin/env python3

"""
Example: Viewer demo with basic geometries, lines, and live updates.

This script demonstrates how to:
- Build a minimal Newton Model (with a ground plane)
- Create a viewer backend (OpenGL by default; optionally USD)
- Add basic meshes (box, sphere, plane, cone, capsule, cylinder)
- Create and update instanced transforms/colors/materials over time
- Render animated lines (axes, grid, connections, rotating spokes)
- Render in a loop using begin_frame(time)/end_frame()
"""

from __future__ import annotations

import argparse
import math
import time

import warp as wp

import newton


def create_model() -> newton.Model:
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    return builder.finalize()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--viewer", choices=["gl", "usd", "rerun"], default="gl", help="Viewer backend to use.")
    args = parser.parse_args()

    # Create a minimal model and viewer
    model = create_model()
    if args.viewer == "usd":
        from newton.viewer import ViewerUSD  # noqa: PLC0415

        viewer = ViewerUSD(model, output_path="example_viewer.usd", num_frames=600)
    elif args.viewer == "rerun":
        from newton.viewer import ViewerRerun  # noqa: PLC0415

        viewer = ViewerRerun(model, server=True, launch_viewer=True)
    else:
        from newton.viewer import ViewerGL  # noqa: PLC0415

        viewer = ViewerGL(model)

    # No explicit mesh creation; we'll use viewer.log_shapes() below

    # Colors and materials per instance
    col_sphere = wp.array([wp.vec3(1.0, 0.1, 0.1)], dtype=wp.vec3)
    col_box = wp.array([wp.vec3(0.1, 1.0, 0.1)], dtype=wp.vec3)
    col_cone = wp.array([wp.vec3(0.1, 0.4, 1.0)], dtype=wp.vec3)
    col_capsule = wp.array([wp.vec3(1.0, 1.0, 0.1)], dtype=wp.vec3)
    col_cylinder = wp.array([wp.vec3(0.8, 0.5, 0.2)], dtype=wp.vec3)

    # material = (metallic, roughness, checker, unused)
    mat_default = wp.array([wp.vec4(0.0, 0.7, 0.0, 0.0)], dtype=wp.vec4)

    if args.viewer == "gl":
        print("Viewer running. WASD/Arrow keys to move, drag to orbit, scroll to zoom. Close window to exit.")

    start = time.time()
    frame = 0
    try:
        while viewer.is_running():
            t = time.time() - start

            # Begin frame with time
            viewer.begin_frame(t)

            # Render model-driven content (ground plane)
            viewer.log_model(model.state())

            # Animate transforms
            qy = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * t)
            qx = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.3 * t)

            # Sphere: orbiting
            p_s = wp.vec3(
                2.0 + math.sin(t) * 0.5,
                0.0,
                1.8 + math.cos(2.0 * t) * 0.3,
            )
            x_sphere_anim = wp.array([wp.transform(p_s, qy)], dtype=wp.transform)

            # Box: rocking
            x_box_anim = wp.array([wp.transform([-2.0, 0.0, 1.5], qx)], dtype=wp.transform)

            # Cone: circular motion
            p_c = wp.vec3(
                math.cos(0.8 * t) * 1.5,
                math.sin(0.8 * t) * 1.5,
                1.8,
            )
            x_cone_anim = wp.array([wp.transform(p_c, qy)], dtype=wp.transform)

            # Cylinder: slow spin
            x_cyl_anim = wp.array([wp.transform([0.0, 0.0, 1.8], qy)], dtype=wp.transform)

            # Capsule: bounce
            p_cap = wp.vec3(0.0, -2.0, 1.8 + 0.2 * abs(math.sin(t)))
            x_cap_anim = wp.array([wp.transform(p_cap, wp.quat_identity())], dtype=wp.transform)

            # Update instances via log_shapes
            viewer.log_shapes(
                "/sphere_instance",
                newton.GEO_SPHERE,
                0.5,
                x_sphere_anim,
                col_sphere,
                mat_default,
            )
            viewer.log_shapes(
                "/box_instance",
                newton.GEO_BOX,
                (0.5, 0.3, 0.8),
                x_box_anim,
                col_box,
                mat_default,
            )
            viewer.log_shapes(
                "/cone_instance",
                newton.GEO_CONE,
                (0.4, 1.2),
                x_cone_anim,
                col_cone,
                mat_default,
            )
            viewer.log_shapes(
                "/cylinder_instance",
                newton.GEO_CYLINDER,
                (0.35, 1.0),
                x_cyl_anim,
                col_cylinder,
                mat_default,
            )
            viewer.log_shapes(
                "/capsule_instance",
                newton.GEO_CAPSULE,
                (0.3, 1.0),
                x_cap_anim,
                col_capsule,
                mat_default,
            )

            # Demonstrate log_lines() with animated debug/visualization lines
            axis_eps = 0.01
            axis_length = 2.0
            axes_begins = wp.array(
                [
                    wp.vec3(0.0, 0.0, axis_eps),  # X axis start
                    wp.vec3(0.0, 0.0, axis_eps),  # Y axis start
                    wp.vec3(0.0, 0.0, axis_eps),  # Z axis start
                ],
                dtype=wp.vec3,
            )

            axes_ends = wp.array(
                [
                    wp.vec3(axis_length, 0.0, axis_eps),  # X axis end
                    wp.vec3(0.0, axis_length, axis_eps),  # Y axis end
                    wp.vec3(0.0, 0.0, axis_length + axis_eps),  # Z axis end
                ],
                dtype=wp.vec3,
            )

            axes_colors = wp.array(
                [
                    wp.vec3(1.0, 0.0, 0.0),  # Red X
                    wp.vec3(0.0, 1.0, 0.0),  # Green Y
                    wp.vec3(0.0, 0.0, 1.0),  # Blue Z
                ],
                dtype=wp.vec3,
            )

            viewer.log_lines("coordinate_axes", axes_begins, axes_ends, axes_colors)

            # End frame (process events, render, present)
            viewer.end_frame()

            frame += 1

    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
