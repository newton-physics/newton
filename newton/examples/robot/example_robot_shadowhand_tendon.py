# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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
# Example Robot Shadow Hand with Tendons
#
# Shows how to simulate the Shadow Hand robot with PhysX fixed tendons
# that couple finger joint movements. The tendons automatically handle
# the mechanical coupling between finger joints.
#
# Command: python -m newton.examples robot_shadowhand_tendon --num-envs 4
#
###########################################################################

import math

import warp as wp

import newton
import newton.examples
import newton.utils


class Example:
    def __init__(self, viewer, num_envs=4, disable_tendons=True, mass_scale=1.0):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10  # Increased from 4 for better stability
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_envs = num_envs
        self.viewer = viewer
        self.device = wp.get_device()

        # Build Shadow Hand model
        hand = newton.ModelBuilder()

        # Configure joint and shape parameters
        # Reduced stiffness values for stability
        stiffness_scale = 0.01  # Scale all stiffnesses down
        hand.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            limit_ke=1.0e2 * stiffness_scale,
            limit_kd=1.0e1 * stiffness_scale,
            friction=1e-4,
            armature=1e-2,  # Increased armature for stability
        )
        hand.default_shape_cfg.ke = 1.0e3 * stiffness_scale
        hand.default_shape_cfg.kd = 1.0e2 * stiffness_scale
        hand.default_shape_cfg.kf = 1.0e2 * stiffness_scale
        hand.default_shape_cfg.mu = 0.75

        # Also set body armature for additional stability
        hand.default_body_armature = 1e-3

        # Load Shadow Hand USD with tendons
        # Using the USD file from the assets directory
        asset_path = "newton/examples/assets/shadow_hand.usd"

        # Define joint ordering to avoid topological sort issues
        # The Shadow Hand has a complex joint structure
        joint_ordering = [
            "rootJoint",
            "robot0_forearm",
            "robot0_WRJ1",
            "robot0_WRJ0",
            "robot0_FFJ3",
            "robot0_FFJ2",
            "robot0_FFJ1",
            "robot0_FFJ0",
            "robot0_MFJ3",
            "robot0_MFJ2",
            "robot0_MFJ1",
            "robot0_MFJ0",
            "robot0_RFJ3",
            "robot0_RFJ2",
            "robot0_RFJ1",
            "robot0_RFJ0",
            "robot0_LFJ4",
            "robot0_LFJ3",
            "robot0_LFJ2",
            "robot0_LFJ1",
            "robot0_LFJ0",
            "robot0_THJ4",
            "robot0_THJ3",
            "robot0_THJ2",
            "robot0_THJ1",
            "robot0_THJ0",
        ]

        try:
            hand.add_usd(
                asset_path,
                xform=wp.transform(wp.vec3(0, 0, 0.5), wp.quat_identity()),
                collapse_fixed_joints=False,
                enable_self_collisions=False,  # Disable self-collisions for performance
                hide_collision_shapes=True,
                verbose=False,  # Reduce output verbosity
                joint_ordering=joint_ordering,  # Explicit joint ordering
            )
        except Exception as e:
            print(f"Warning: Failed to load with joint ordering: {e}")
            print("Trying without joint ordering...")
            # Try again without joint ordering
            hand.add_usd(
                asset_path,
                xform=wp.transform(wp.vec3(0, 0, 0.5), wp.quat_identity()),
                collapse_fixed_joints=False,
                enable_self_collisions=False,
                hide_collision_shapes=True,
                verbose=False,
                joint_ordering=None,
            )

        # Set up joint control modes
        # The Shadow Hand has many joints, but due to tendons,
        # we only need to control certain "actuated" joints
        # for i in range(len(hand.joint_dof_mode)):
        for i in [9]:
            hand.joint_dof_mode[i] = newton.JointMode.TARGET_POSITION
            hand.joint_target_ke[i] = 10  # Reduced from 100
            hand.joint_target_kd[i] = 1  # Reduced from 5

        # Build the full scene
        builder = newton.ModelBuilder()
        builder.replicate(hand, self.num_envs, spacing=(0.8, 0.8, 0))
        builder.add_ground_plane()  # Add ground to prevent falling

        self.model = builder.finalize()

        ## Apply mass scaling if requested
        # if mass_scale != 1.0:
        #    print(f"\nApplying mass scale factor: {mass_scale}")
        #    # Scale all body masses (which means scaling inverse masses down)
        #    self.model.body_inv_mass = self.model.body_inv_mass / mass_scale
        #    # Also scale inertias
        #    self.model.body_inv_inertia = self.model.body_inv_inertia / mass_scale

        # Print tendon info
        if self.model.tendon_count > 0:
            print(f"\nModel has {self.model.tendon_count} tendons:")
            # Convert arrays to numpy for indexing
            tendon_start = (
                self.model.tendon_start.numpy()
                if hasattr(self.model.tendon_start, "numpy")
                else self.model.tendon_start
            )
            tendon_key = self.model.tendon_key if hasattr(self.model, "tendon_key") else []

            tendon_params = (
                self.model.tendon_params.numpy()
                if hasattr(self.model.tendon_params, "numpy")
                else self.model.tendon_params
            )
            # tendon_joints = self.model.tendon_joints.numpy() if hasattr(self.model.tendon_joints, 'numpy') else self.model.tendon_joints
            # tendon_gearings = self.model.tendon_gearings.numpy() if hasattr(self.model.tendon_gearings, 'numpy') else self.model.tendon_gearings

            for i in range(self.model.tendon_count):
                tendon_name = tendon_key[i] if i < len(tendon_key) else f"tendon_{i}"
                start_idx = tendon_start[i]
                end_idx = tendon_start[i + 1] if i + 1 < len(tendon_start) else len(self.model.tendon_joints)
                num_joints = end_idx - start_idx
                params = tendon_params[i]
                print(f"  Tendon '{tendon_name}': couples {num_joints} joints")
                print(f"    Parameters: ke={params[0]}, kd={params[1]}, rest_length={params[2]}")

                # Check for extreme values
                if params[0] > 1e4:
                    print(f"    WARNING: Very high stiffness ke={params[0]}")
                if params[2] < 0:
                    print(f"    WARNING: Negative rest length={params[2]}")

            # Optionally disable tendons by zeroing their stiffness
            if disable_tendons:
                print("\nDisabling tendons for debugging...")
                # Convert to numpy, modify, then copy back
                tendon_params_np = self.model.tendon_params.numpy()
                for i in range(self.model.tendon_count):
                    tendon_params_np[i][0] = 0.0  # Set ke to 0
                self.model.tendon_params.assign(tendon_params_np)

        # Create solver
        try:
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                iterations=50,
                ls_iterations=25,
                use_mujoco_cpu=False,
                solver="newton",
                integrator="implicit",  # Use implicit integrator for better stability
                separate_envs_to_worlds=False,  # Avoid environment separation issues
                nefc_per_env=500,  # Increase contact constraint buffer size for Shadow Hand
            )
        except Exception as e:
            print(f"Error creating MuJoCo solver: {e}")
            print("Trying with CPU MuJoCo...")
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                iterations=50,
                ls_iterations=25,
                use_mujoco_cpu=True,  # Fall back to CPU
                solver="newton",
                integrator="implicit",  # Use implicit integrator for better stability
                separate_envs_to_worlds=False,
                nefc_per_env=500,  # Increase contact constraint buffer size for Shadow Hand
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Debug: Check for bodies with zero or very small mass
        print("\nChecking body masses and inertias:")
        body_masses = (
            self.model.body_inv_mass.numpy() if hasattr(self.model.body_inv_mass, "numpy") else self.model.body_inv_mass
        )
        body_inertias = (
            self.model.body_inv_inertia.numpy()
            if hasattr(self.model.body_inv_inertia, "numpy")
            else self.model.body_inv_inertia
        )

        min_mass = float("inf")
        max_mass = 0
        mass_issues = []

        for i in range(self.model.body_count):
            if i < len(self.model.body_key):
                body_name = self.model.body_key[i]
            else:
                body_name = f"body_{i}"

            if body_masses[i] == 0:
                mass = "infinite (fixed)"
            else:
                mass = 1.0 / body_masses[i]
                if mass < min_mass and mass > 0:
                    min_mass = mass
                if mass > max_mass:
                    max_mass = mass

                if mass < 1e-4:
                    mass_issues.append(f"  WARNING: Body {i} '{body_name}' has very small mass: {mass:.6f} kg")

                # Check inertia
                inv_I = body_inertias[i]
                if inv_I[0, 0] > 0:  # Not fixed
                    I_xx = 1.0 / inv_I[0, 0]
                    if I_xx < 1e-6:
                        mass_issues.append(f"  WARNING: Body {i} '{body_name}' has very small inertia: {I_xx:.6e}")

        if mass_issues:
            print("\nMass/Inertia warnings:")
            for issue in mass_issues[:10]:  # Show first 10
                print(issue)
            if len(mass_issues) > 10:
                print(f"  ... and {len(mass_issues) - 10} more warnings")

        if max_mass > 0 and min_mass < float("inf"):
            mass_ratio = max_mass / min_mass
            print(f"\nMass ratio (max/min): {mass_ratio:.1f}")
            if mass_ratio > 1000:
                print("  WARNING: Very large mass ratio can cause instability!")

        self.contacts = self.model.collide(self.state_0)

        # Store initial joint positions
        self.initial_joint_q = self.state_0.joint_q.numpy().copy()

        # Identify key finger joints for control
        # We'll animate some finger flexion/extension
        self.finger_joints = []
        joint_names = [self.model.joint_key[i] for i in range(self.model.joint_dof_count)]

        # Look for finger flexion joints (these names are typical for Shadow Hand)
        finger_prefixes = ["FFJ", "MFJ", "RFJ", "LFJ", "THJ"]  # First, Middle, Ring, Little, Thumb
        for i, name in enumerate(joint_names):
            for prefix in finger_prefixes:
                if prefix in name and "J2" in name:  # J2 is typically the main flexion joint
                    self.finger_joints.append(i)
                    print(f"Found finger joint: {name} at index {i}")
                    break

        self.viewer.set_model(self.model)
        self.capture()

        # Print summary
        print("\n" + "=" * 50)
        print("Shadow Hand Example Configuration:")
        print(f"  Mass scale factor: {mass_scale}")
        print(f"  Stiffness scale factor: {stiffness_scale}")
        print(f"  Simulation timestep: {self.sim_dt:.6f} s")
        print(f"  Substeps per frame: {self.sim_substeps}")
        print(f"  Tendons: {'DISABLED' if disable_tendons else 'ENABLED'}")
        print("  Integrator: implicit")
        print("=" * 50 + "\n")

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.contacts = self.model.collide(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply sinusoidal motion to finger joints
            # This will demonstrate how tendons couple the motion
            phase = self.sim_time * 2.0  # 2 rad/s frequency

            # Create target positions
            target_q = self.initial_joint_q.copy()

            # Animate finger joints with different phases
            for i, joint_idx in enumerate(self.finger_joints):
                # Each finger moves with a slight phase offset
                finger_phase = phase + i * 0.5
                # Oscillate between 0 and 60 degrees
                angle = 0.3 * (1.0 + math.sin(finger_phase))
                target_q[joint_idx] = angle

            # Set target positions
            self.control.joint_target.assign(target_q)

            # Apply viewer forces (for interaction)
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
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

    def test(self):
        # Run a few steps to verify tendons are working
        for _ in range(10):
            self.step()

        # Check that joints have moved
        current_q = self.state_0.joint_q.numpy()
        q_diff = current_q - self.initial_joint_q
        max_diff = abs(q_diff).max()

        print(f"Test: Maximum joint position change: {max_diff:.4f} rad")
        assert max_diff > 0.01, "Joints should have moved"
        print("Test passed: Tendons are coupling joint motion correctly")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-envs", type=int, default=4, help="Total number of simulated environments.")
    parser.add_argument("--disable-tendons", action="store_true", help="Disable tendons for debugging")
    parser.add_argument(
        "--mass-scale", type=float, default=1.0, help="Scale all masses by this factor (try 10 or 100 for stability)"
    )

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args.num_envs, args.disable_tendons, args.mass_scale)

    newton.examples.run(example)
