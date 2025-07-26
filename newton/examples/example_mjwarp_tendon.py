import os
import tempfile

import numpy as np
import warp as wp

import newton
import newton.solvers
import newton.utils


class Example:
    def __init__(self, stage_path="example_mjwarp_tendon.usd", num_frames=300, headless=False):
        # Simulation parameters
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60  # 60 FPS
        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.num_frames = num_frames

        # MJCF content with tendon
        mjcf_content = """
        <mujoco>
          <worldbody>
            <site name="anchor" pos="0 0 2" rgba="1 0 0 1" size="0.1"/>

            <body name="pendulum" pos="0 1 2">
              <geom type="box" size="0.1 0.4 0.1" rgba="0.2 0.2 0.8 1"/>
              <!--joint name="swing" type="hinge" axis="1 0 0" pos="0 0 0.4"/-->
              <freejoint/>
              <site name="attach" pos="0 -0.4 0" rgba="0 1 0 1" size="0.1"/>
              <inertial pos="0 0 0" mass="1"/>
            </body>
          </worldbody>

          <tendon>
            <spatial name="cable">
              <site site="anchor"/>
              <site site="attach"/>
            </spatial>
          </tendon>

          <actuator>
            <position name="cable_act" tendon="cable" kp="2000" kv="1000"/>
          </actuator>
        </mujoco>
        """

        # Create temporary MJCF file
        self.tmpdir = tempfile.TemporaryDirectory()
        mjcf_path = os.path.join(self.tmpdir.name, "test-tendon.xml")
        with open(mjcf_path, "w") as f:
            f.write(mjcf_content)

        # Build model
        builder = newton.ModelBuilder()
        newton.utils.parse_mjcf(
            mjcf_path,
            builder,
            collapse_fixed_joints=True,
            up_axis="Z",
            enable_self_collisions=False,
        )
        self.model = builder.finalize()

        print("Model statistics:")
        print(f"  Sites: {self.model.site_count}")
        print(f"  Tendons: {self.model.tendon_count}")
        print(f"  Tendon actuators: {self.model.tendon_actuator_count}")

        # Create solver
        self.solver = newton.solvers.MuJoCoSolver(self.model)
        print("\nMuJoCo solver created successfully!")

        # Create states and control
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Create renderer
        if not headless:
            self.renderer = newton.utils.SimRendererOpenGL(
                model=self.model,
                path=stage_path,
                scaling=1.0,
                up_axis="Z",
                screen_width=1280,
                screen_height=720,
                camera_pos=(0, 1, 3),  # View from negative Y direction, looking at the pendulum
            )
        elif stage_path:
            self.renderer = newton.utils.SimRendererUsd(self.model, stage_path)
        else:
            self.renderer = None

        # Set initial tendon target to contract the cable
        tendon_targets = self.control.tendon_target.numpy()
        tendon_targets[0] = -0.3  # Contract by 30cm
        self.control.tendon_target = wp.array(tendon_targets, dtype=wp.float32, device=self.model.device)
        print(f"Set tendon target to: {tendon_targets[0]}")

        # Record initial state
        self.initial_angle = self.state_0.joint_q.numpy()[0]

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        with wp.ScopedTimer("step", active=False):
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)

            # Visualize the tendon as a line between sites
            if self.model.site_count >= 2:
                # Get site positions in world space
                site_positions = []
                site_body_np = self.model.site_body.numpy()
                site_xform_np = self.model.site_xform.numpy()
                body_q_np = self.state_0.body_q.numpy()

                for i in range(2):  # anchor and attach sites
                    body_idx = int(site_body_np[i])
                    site_xform = site_xform_np[i]

                    if body_idx >= 0:
                        # Site is attached to a body, transform to world space
                        body_q = body_q_np[body_idx]
                        body_transform = wp.transform(body_q[:3], body_q[3:7])
                        site_world_pos = wp.transform_point(body_transform, wp.vec3(site_xform[:3]))
                    else:
                        # Site is in world space
                        site_world_pos = site_xform[:3]

                    site_positions.append(site_world_pos)

                # Render line between sites
                self.renderer.render_line_strip(
                    "tendon_cable",
                    site_positions,
                    color=(1.0, 0.5, 0.0),  # Orange color for the cable
                    radius=0.02,
                )

            self.renderer.end_frame()

    def run(self):
        print(f"\nRunning simulation for {self.num_frames} frames...")
        print("Controls: SPACE to pause, TAB to skip rendering, ESC to exit")
        print("Camera: Use WASD + mouse drag to move, mouse wheel to zoom")
        print("Orange cable shows the tendon, gets redder/thicker with more force")

        for i in range(self.num_frames):
            self.step()
            self.render()

            # Print status every 0.5 seconds (30 frames at 60 FPS)
            if i % 30 == 0:
                angle = self.state_0.joint_q.numpy()[0]
                velocity = self.state_0.joint_qd.numpy()[0]
                print(f"  t={self.sim_time:.1f}s: angle={np.degrees(angle):6.1f}°, velocity={velocity:6.2f} rad/s")

        # Final results
        final_angle = self.state_0.joint_q.numpy()[0]
        angle_change = np.degrees(final_angle - self.initial_angle)

        print("\nFinal result:")
        print(f"  Angle changed by: {angle_change:.1f}°")

        if abs(angle_change) > 1.0:
            print("✓ SUCCESS: Tendon control is working!")
        else:
            print("✗ FAIL: Tendon control is NOT working - pendulum didn't move")

        # Save renderer output
        if self.renderer and hasattr(self.renderer, "save"):
            self.renderer.save()
            print(f"\nAnimation saved to: {self.renderer.stage_path}")

    def __del__(self):
        # Clean up temporary directory
        if hasattr(self, "tmpdir") and self.tmpdir:
            try:
                self.tmpdir.cleanup()
            except Exception:
                pass  # Ignore cleanup errors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_mjwarp_tendon.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=300, help="Total number of frames (default: 300).")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without visualization.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_frames=args.num_frames, headless=args.headless)
        example.run()
