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
# Example Diffsim Drone
#
# A drone and its 4 propellers is simulated with the goal of reaching
# different targets via model-predictive control (MPC) that continuously
# optimizes the control trajectory.
#
# Command: python -m newton.examples diffsim_drone
#
###########################################################################

import os

import numpy as np
import warp as wp
import warp.optim

import newton
import newton.examples

# TODO: These should be imported from a public API once available
# For now, implementing locally.
from newton._src.geometry.kernels import box_sdf, capsule_sdf, cone_sdf, cylinder_sdf, mesh_sdf, plane_sdf, sphere_sdf
from newton.tests.unittest_utils import most
from newton.utils import bourke_color_map

DEFAULT_DRONE_PATH = newton.examples.get_asset("crazyflie.usd")  # Path to input drone asset


@wp.struct
class Propeller:
    body: int
    pos: wp.vec3
    dir: wp.vec3
    thrust: float
    power: float
    diameter: float
    height: float
    max_rpm: float
    max_thrust: float
    max_torque: float
    turning_direction: float
    max_speed_square: float


@wp.kernel
def increment_seed(
    seed: wp.array(dtype=int),
):
    seed[0] += 1


@wp.kernel
def sample_gaussian(
    mean_trajectory: wp.array(dtype=float, ndim=3),
    noise_scale: float,
    num_control_points: int,
    control_dim: int,
    control_limits: wp.array(dtype=float, ndim=2),
    seed: wp.array(dtype=int),
    rollout_trajectories: wp.array(dtype=float, ndim=3),
):
    world_id, point_id, control_id = wp.tid()
    unique_id = (world_id * num_control_points + point_id) * control_dim + control_id
    r = wp.rand_init(seed[0], unique_id)
    mean = mean_trajectory[0, point_id, control_id]
    lo, hi = control_limits[control_id, 0], control_limits[control_id, 1]
    sample = mean + noise_scale * wp.randn(r)
    for _i in range(10):
        if sample < lo or sample > hi:
            sample = mean + noise_scale * wp.randn(r)
        else:
            break
    rollout_trajectories[world_id, point_id, control_id] = wp.clamp(sample, lo, hi)


@wp.kernel
def replicate_states(
    body_q_in: wp.array(dtype=wp.transform),
    body_qd_in: wp.array(dtype=wp.spatial_vector),
    bodies_per_world: int,
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    world_offset = tid * bodies_per_world
    for i in range(bodies_per_world):
        body_q_out[world_offset + i] = body_q_in[i]
        body_qd_out[world_offset + i] = body_qd_in[i]


@wp.kernel
def drone_cost(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    targets: wp.array(dtype=wp.vec3),
    prop_control: wp.array(dtype=float),
    step: int,
    horizon_length: int,
    weighting: float,
    cost: wp.array(dtype=wp.float32),
):
    world_id = wp.tid()
    tf = body_q[world_id]
    target = targets[0]

    pos_drone = wp.transform_get_translation(tf)
    pos_cost = wp.length_sq(pos_drone - target)
    altitude_cost = wp.max(pos_drone[2] - 0.75, 0.0) + wp.max(0.25 - pos_drone[2], 0.0)
    upvector = wp.vec3(0.0, 0.0, 1.0)
    drone_up = wp.transform_vector(tf, upvector)
    upright_cost = 1.0 - wp.dot(drone_up, upvector)

    vel_drone = body_qd[world_id]

    # Encourage zero velocity.
    vel_cost = wp.length_sq(vel_drone)

    control = wp.vec4(
        prop_control[world_id * 4 + 0],
        prop_control[world_id * 4 + 1],
        prop_control[world_id * 4 + 2],
        prop_control[world_id * 4 + 3],
    )
    control_cost = wp.dot(control, control)

    discount = 0.8 ** wp.float(horizon_length - step - 1) / wp.float(horizon_length) ** 2.0

    pos_weight = 1000.0
    altitude_weight = 100.0
    control_weight = 0.05
    vel_weight = 0.1
    upright_weight = 10.0
    total_weight = pos_weight + altitude_weight + control_weight + vel_weight + upright_weight

    wp.atomic_add(
        cost,
        world_id,
        (
            pos_cost * pos_weight
            + altitude_cost * altitude_weight
            + control_cost * control_weight
            + vel_cost * vel_weight
            + upright_cost * upright_weight
        )
        * (weighting / total_weight)
        * discount,
    )


@wp.kernel
def collision_cost(
    body_q: wp.array(dtype=wp.transform),
    obstacle_ids: wp.array(dtype=int, ndim=2),
    shape_X_bs: wp.array(dtype=wp.transform),
    # geo: wp.sim.ModelShapeGeometry,
    shape_type: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_source_ptr: wp.array(dtype=wp.uint64),
    margin: float,
    weighting: float,
    cost: wp.array(dtype=wp.float32),
):
    world_id, obs_id = wp.tid()
    shape_index = obstacle_ids[world_id, obs_id]

    px = wp.transform_get_translation(body_q[world_id])

    X_bs = shape_X_bs[shape_index]

    # transform particle position to shape local space
    x_local = wp.transform_point(wp.transform_inverse(X_bs), px)

    # geo description
    geo_type = shape_type[shape_index]
    geo_scale = shape_scale[shape_index]

    # evaluate shape sdf
    d = 1e6

    if geo_type == newton.GeoType.SPHERE:
        d = sphere_sdf(wp.vec3(), geo_scale[0], x_local)
    elif geo_type == newton.GeoType.BOX:
        d = box_sdf(geo_scale, x_local)
    elif geo_type == newton.GeoType.CAPSULE:
        d = capsule_sdf(geo_scale[0], geo_scale[1], x_local)
    elif geo_type == newton.GeoType.CYLINDER:
        d = cylinder_sdf(geo_scale[0], geo_scale[1], x_local)
    elif geo_type == newton.GeoType.CONE:
        d = cone_sdf(geo_scale[0], geo_scale[1], x_local)
    elif geo_type == newton.GeoType.MESH:
        mesh = shape_source_ptr[shape_index]
        min_scale = wp.min(geo_scale)
        max_dist = margin / min_scale
        d = mesh_sdf(mesh, wp.cw_div(x_local, geo_scale), max_dist)
        d *= min_scale  # TODO fix this, mesh scaling needs to be handled properly
    elif geo_type == newton.GeoType.PLANE:
        d = plane_sdf(geo_scale[0], geo_scale[1], x_local)

    d = wp.max(d, 0.0)
    if d < margin:
        c = margin - d
        wp.atomic_add(cost, world_id, weighting * c)


@wp.kernel
def enforce_control_limits(
    control_limits: wp.array(dtype=float, ndim=2),
    control_points: wp.array(dtype=float, ndim=3),
):
    world_id, t_id, control_id = wp.tid()
    lo, hi = control_limits[control_id, 0], control_limits[control_id, 1]
    control_points[world_id, t_id, control_id] = wp.clamp(control_points[world_id, t_id, control_id], lo, hi)


@wp.kernel
def pick_best_trajectory(
    rollout_trajectories: wp.array(dtype=float, ndim=3),
    lowest_cost_id: int,
    best_traj: wp.array(dtype=float, ndim=3),
):
    t_id, control_id = wp.tid()
    best_traj[0, t_id, control_id] = rollout_trajectories[lowest_cost_id, t_id, control_id]


@wp.kernel
def interpolate_control_linear(
    control_points: wp.array(dtype=float, ndim=3),
    control_dofs: wp.array(dtype=int),
    control_gains: wp.array(dtype=float),
    t: float,
    torque_dim: int,
    torques: wp.array(dtype=float),
):
    world_id, control_id = wp.tid()
    t_id = int(t)
    frac = t - wp.floor(t)
    control_left = control_points[world_id, t_id, control_id]
    control_right = control_points[world_id, t_id + 1, control_id]
    torque_id = world_id * torque_dim + control_dofs[control_id]
    action = control_left * (1.0 - frac) + control_right * frac
    torques[torque_id] = action * control_gains[control_id]


@wp.kernel
def compute_prop_wrenches(
    props: wp.array(dtype=Propeller),
    controls: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    prop = props[tid]
    control = controls[tid]
    tf = body_q[prop.body]
    dir = wp.transform_vector(tf, prop.dir)
    force = dir * prop.max_thrust * control
    torque = dir * prop.max_torque * control * prop.turning_direction
    moment_arm = wp.transform_point(tf, prop.pos) - wp.transform_point(tf, body_com[prop.body])
    torque += wp.cross(moment_arm, force)
    # Apply angular damping.
    torque *= 0.8
    wp.atomic_add(body_f, prop.body, wp.spatial_vector(force, torque))


def define_propeller(
    drone: int,
    pos: wp.vec3,
    fps: float,
    thrust: float = 0.109919,
    power: float = 0.040164,
    diameter: float = 0.2286,
    height: float = 0.01,
    max_rpm: float = 6396.667,
    turning_direction: float = 1.0,
):
    # Air density at sea level.
    air_density = 1.225  # kg / m^3

    rps = max_rpm / fps
    max_speed = rps * wp.TAU  # radians / sec
    rps_square = rps**2

    prop = Propeller()
    prop.body = drone
    prop.pos = pos
    prop.dir = wp.vec3(0.0, 0.0, 1.0)
    prop.thrust = thrust
    prop.power = power
    prop.diameter = diameter
    prop.height = height
    prop.max_rpm = max_rpm
    prop.max_thrust = thrust * air_density * rps_square * diameter**4
    prop.max_torque = power * air_density * rps_square * diameter**5 / wp.TAU
    prop.turning_direction = turning_direction
    prop.max_speed_square = max_speed**2

    return prop


class Drone:
    def __init__(
        self,
        name: str,
        fps: float,
        trajectory_shape: tuple[int, int],
        variation_count: int = 1,
        size: float = 1.0,
        requires_grad: bool = False,
        state_count: int | None = None,
    ) -> None:
        self.variation_count = variation_count
        self.requires_grad = requires_grad

        # Current tick of the simulation, including substeps.
        self.sim_tick = 0

        # Initialize the helper to build a physics scene.
        builder = newton.ModelBuilder()
        builder.rigid_contact_margin = 0.05

        # Initialize the rigid bodies, propellers, and colliders.
        props = []
        colliders = []
        crossbar_length = size
        crossbar_height = size * 0.05
        crossbar_width = size * 0.05
        carbon_fiber_density = 1750.0  # kg / m^3
        for i in range(variation_count):
            # Register the drone as a rigid body in the simulation model.
            body = builder.add_body(key=f"{name}_{i}")

            # Define the shapes making up the drone's rigid body.
            builder.add_shape_box(
                body,
                hx=crossbar_width,
                hy=crossbar_length,
                hz=crossbar_height,
                cfg=newton.ModelBuilder.ShapeConfig(density=carbon_fiber_density, collision_group=i),
            )
            builder.add_shape_box(
                body,
                hx=crossbar_length,
                hy=crossbar_width,
                hz=crossbar_height,
                cfg=newton.ModelBuilder.ShapeConfig(density=carbon_fiber_density, collision_group=i),
            )

            # Initialize the propellers.
            props.extend(
                (
                    define_propeller(
                        body,
                        wp.vec3(0.0, crossbar_length, 0.0),
                        fps,
                        turning_direction=-1.0,
                    ),
                    define_propeller(
                        body,
                        wp.vec3(0.0, -crossbar_length, 0.0),
                        fps,
                        turning_direction=1.0,
                    ),
                    define_propeller(
                        body,
                        wp.vec3(crossbar_length, 0.0, 0.0),
                        fps,
                        turning_direction=1.0,
                    ),
                    define_propeller(
                        body,
                        wp.vec3(-crossbar_length, 0.0, 0.0),
                        fps,
                        turning_direction=-1.0,
                    ),
                ),
            )

            # Initialize the colliders.
            colliders.append(
                (
                    builder.add_shape_capsule(
                        -1,
                        xform=wp.transform(wp.vec3(0.5, 0.5, 2.0), wp.quat_identity()),
                        radius=0.15,
                        half_height=2.0,
                        cfg=newton.ModelBuilder.ShapeConfig(collision_group=i),
                    ),
                ),
            )
        self.props = wp.array(props, dtype=Propeller)
        self.colliders = wp.array(colliders, dtype=int)

        # Build the model and set-up its properties.
        self.model = builder.finalize(requires_grad=requires_grad)

        # Initialize the required simulation states.
        if requires_grad:
            self.states = tuple(self.model.state() for _ in range(state_count + 1))
            self.controls = tuple(self.model.control() for _ in range(state_count))
        else:
            # When only running a forward simulation, we don't need to store
            # the history of the states at each step, instead we use double
            # buffering to represent the previous and next states.
            self.states = [self.model.state(), self.model.state()]
            self.controls = (self.model.control(),)

        # create array for the propeller controls
        for control in self.controls:
            control.prop_controls = wp.zeros(len(self.props), dtype=float, requires_grad=requires_grad)

        # Define the trajectories as arrays of control points.
        # The point data has an additional item to support linear interpolation.
        self.trajectories = wp.zeros(
            (variation_count, trajectory_shape[0], trajectory_shape[1]),
            dtype=float,
            requires_grad=requires_grad,
        )

        # Store some miscellaneous info.
        self.body_count = len(builder.body_q)
        self.collider_count = self.colliders.shape[1]
        self.collision_radius = crossbar_length

    @property
    def state(self) -> newton.State:
        return self.states[self.sim_tick if self.requires_grad else 0]

    @property
    def next_state(self) -> newton.State:
        return self.states[self.sim_tick + 1 if self.requires_grad else 1]

    @property
    def control(self) -> newton.Control:
        return self.controls[min(len(self.controls) - 1, self.sim_tick) if self.requires_grad else 0]


class Example:
    def __init__(
        self,
        viewer,
        verbose=False,
        num_rollouts=16,
        render_rollouts=False,
        drone_path=DEFAULT_DRONE_PATH,
    ):
        # setup simulation parameters first
        self.fps = 60
        self.frame = 0
        self.frame_dt = 1.0 / self.fps
        self.sim_steps = 360  # 6.0 seconds
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.verbose = verbose
        self.rollout_count = num_rollouts
        self.render_rollouts = render_rollouts
        # TODO: Use drone path to load USD asset and draw it
        self.drone_path = drone_path

        # setup rendering
        self.viewer = viewer

        # Targets positions that the drone will try to reach in turn.
        self.targets = (
            wp.vec3(1.0, 0.0, 0.5),
            wp.vec3(0.0, 1.0, 0.5),
        )

        # Define the index of the active target.
        # We start with -1 since it'll be incremented on the first frame.
        self.target_idx = -1
        # use a Warp array to store the current target so that we can assign
        # a new target to it while retaining the original CUDA graph.
        self.current_target = wp.array([self.targets[self.target_idx + 1]], dtype=wp.vec3)

        # Number of steps to run at each frame for the optimisation pass.
        self.optim_step_count = 20

        # Time steps between control points.
        self.control_point_step = 10

        # Number of control horizon points to interpolate between.
        self.control_point_count = 3

        self.control_point_data_count = self.control_point_count + 1
        self.control_dofs = wp.array((0, 1, 2, 3), dtype=int)
        self.control_dim = len(self.control_dofs)
        self.control_gains = wp.array((0.8,) * self.control_dim, dtype=float)
        self.control_limits = wp.array(((0.1, 1.0),) * self.control_dim, dtype=float)

        drone_size = 0.2

        # Declare the reference drone.
        self.drone = Drone(
            "drone",
            self.fps,
            (self.control_point_data_count, self.control_dim),
            size=drone_size,
        )

        # Declare the drone's rollouts.
        # These allow to run parallel simulations in order to find the best
        # trajectory at each control point.
        self.rollout_step_count = self.control_point_step * self.control_point_count
        self.rollouts = Drone(
            "rollout",
            self.fps,
            (self.control_point_data_count, self.control_dim),
            variation_count=self.rollout_count,
            size=drone_size,
            requires_grad=True,
            state_count=self.rollout_step_count * self.sim_substeps,
        )

        self.seed = wp.zeros(1, dtype=int)
        self.rollout_costs = wp.zeros(self.rollout_count, dtype=float, requires_grad=True)
        self.cost_history = []

        # Use the SemiImplicit integrator for stepping through the simulation.
        self.solver_rollouts = newton.solvers.SolverSemiImplicit(self.rollouts.model)
        self.solver_drone = newton.solvers.SolverSemiImplicit(self.drone.model)

        self.optimizer = warp.optim.SGD(
            [self.rollouts.trajectories.flatten()],
            lr=1e-2,
            nesterov=False,
            momentum=0.0,
        )

        # rendering
        self.viewer.set_model(self.drone.model)

        # capture forward/backward passes
        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.forward_backward()
            self.graph = capture.graph
        else:
            self.graph = None

    def forward_backward(self):
        self.tape = wp.Tape()
        with self.tape:
            self.forward()
        self.rollout_costs.grad.fill_(1.0)
        self.tape.backward()

    def update_drone(self, drone: Drone, solver) -> None:
        drone.state.clear_forces()

        wp.launch(
            interpolate_control_linear,
            dim=(
                drone.variation_count,
                self.control_dim,
            ),
            inputs=(
                drone.trajectories,
                self.control_dofs,
                self.control_gains,
                drone.sim_tick / (self.sim_substeps * self.control_point_step),
                self.control_dim,
            ),
            outputs=(drone.control.prop_controls,),
        )

        wp.launch(
            compute_prop_wrenches,
            dim=len(drone.props),
            inputs=(
                drone.props,
                drone.control.prop_controls,
                drone.state.body_q,
                drone.model.body_com,
            ),
            outputs=(drone.state.body_f,),
        )

        solver.step(
            drone.state,
            drone.next_state,
            None,
            None,
            self.sim_dt,
        )

        drone.sim_tick += 1

    def forward(self):
        # Evaluate the rollouts with their costs.
        self.rollouts.sim_tick = 0
        self.rollout_costs.zero_()
        wp.launch(
            replicate_states,
            dim=self.rollout_count,
            inputs=(
                self.drone.state.body_q,
                self.drone.state.body_qd,
                self.drone.body_count,
            ),
            outputs=(
                self.rollouts.state.body_q,
                self.rollouts.state.body_qd,
            ),
        )

        for i in range(self.rollout_step_count):
            for _ in range(self.sim_substeps):
                self.update_drone(self.rollouts, self.solver_rollouts)

            wp.launch(
                drone_cost,
                dim=self.rollout_count,
                inputs=(
                    self.rollouts.state.body_q,
                    self.rollouts.state.body_qd,
                    self.current_target,
                    self.rollouts.control.prop_controls,
                    i,
                    self.rollout_step_count,
                    1e3,
                ),
                outputs=(self.rollout_costs,),
            )
            wp.launch(
                collision_cost,
                dim=(
                    self.rollout_count,
                    self.rollouts.collider_count,
                ),
                inputs=(
                    self.rollouts.state.body_q,
                    self.rollouts.colliders,
                    self.rollouts.model.shape_transform,
                    # self.rollouts.model.shape_geo,
                    self.rollouts.model.shape_type,
                    self.rollouts.model.shape_scale,
                    self.rollouts.model.shape_source_ptr,
                    self.rollouts.collision_radius,
                    1e4,
                ),
                outputs=(self.rollout_costs,),
            )

    def step_optimizer(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.forward_backward()

        self.optimizer.step([self.rollouts.trajectories.grad.flatten()])

        # Enforce limits on the control points.
        wp.launch(
            enforce_control_limits,
            dim=self.rollouts.trajectories.shape,
            inputs=(self.control_limits,),
            outputs=(self.rollouts.trajectories,),
        )
        self.tape.zero()

    def step(self):
        if self.frame % int(self.sim_steps / len(self.targets)) == 0:
            if self.verbose:
                print(f"Choosing new flight target: {self.target_idx + 1}")

            self.target_idx += 1
            self.target_idx %= len(self.targets)

            # Assign the new target to the current target array.
            self.current_target.assign([self.targets[self.target_idx]])

        # Sample control waypoints around the nominal trajectory.
        noise_scale = 0.15
        wp.launch(
            sample_gaussian,
            dim=(
                self.rollouts.trajectories.shape[0] - 1,
                self.rollouts.trajectories.shape[1],
                self.rollouts.trajectories.shape[2],
            ),
            inputs=(
                self.drone.trajectories,
                noise_scale,
                self.control_point_data_count,
                self.control_dim,
                self.control_limits,
                self.seed,
            ),
            outputs=(self.rollouts.trajectories,),
        )

        wp.launch(
            increment_seed,
            dim=1,
            inputs=(),
            outputs=(self.seed,),
        )

        for _ in range(self.optim_step_count):
            self.step_optimizer()

        # Pick the best trajectory.
        wp.synchronize()
        lowest_cost_id = np.argmin(self.rollout_costs.numpy())
        wp.launch(
            pick_best_trajectory,
            dim=(
                self.control_point_data_count,
                self.control_dim,
            ),
            inputs=(
                self.rollouts.trajectories,
                lowest_cost_id,
            ),
            outputs=(self.drone.trajectories,),
        )
        self.rollouts.trajectories[-1].assign(self.drone.trajectories[0])

        # Simulate the drone.
        self.drone.sim_tick = 0
        for _ in range(self.sim_substeps):
            self.update_drone(self.drone, self.solver_drone)

            # Swap the drone's states.
            (self.drone.states[0], self.drone.states[1]) = (self.drone.states[1], self.drone.states[0])

        loss = np.min(self.rollout_costs.numpy())
        print(f"[{(self.frame + 1):3d}/{self.sim_steps}] loss={loss:.8f}")
        self.viewer.log_scalar("/loss", loss)
        self.cost_history.append(loss)

    def test_final(self):
        assert all(np.array(self.cost_history) < 2.0)
        assert most(np.diff(self.cost_history) < 0.0, min_ratio=0.6)
        assert all(np.diff(self.cost_history) < 1e-2)

    def render(self):
        self.viewer.begin_frame(self.frame * self.frame_dt)
        self.viewer.log_state(self.drone.state)

        # Render a sphere as the current target.
        self.viewer.log_shapes(
            "/target",
            newton.GeoType.SPHERE,
            (0.05,),
            wp.array([wp.transform(self.targets[self.target_idx], wp.quat_identity())], dtype=wp.transform),
            wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3),
        )

        # Render the rollout trajectories.
        if self.render_rollouts:
            costs = self.rollout_costs.numpy()

            positions = np.array([x.body_q.numpy()[:, :3] for x in self.rollouts.states])

            min_cost = np.min(costs)
            max_cost = np.max(costs)
            for i in range(self.rollout_count):
                # Flip colors, so red means best trajectory, blue worst.
                color = bourke_color_map(-max_cost, -min_cost, -costs[i])
                self.viewer.log_lines(
                    f"/rollout_{i}",
                    wp.array(positions[0:-1, i], dtype=wp.vec3),
                    wp.array(positions[1:, i], dtype=wp.vec3),
                    color,
                )

        self.viewer.end_frame()

        self.frame += 1


if __name__ == "__main__":
    # Create parser that inherits common arguments and adds example-specific ones
    parser = newton.examples.create_parser()
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")
    parser.add_argument("--num_rollouts", type=int, default=16, help="Number of drone rollouts.")
    parser.add_argument(
        "--drone_path",
        type=str,
        default=os.path.join(newton.examples.get_asset_directory(), "crazyflie.usd"),
        help="Path to the USD file to use as the reference for the drone prim in the output stage.",
    )
    parser.add_argument(
        "--render_rollouts",
        action="store_true",
        help="Add rollout trajectories to the output stage.",
    )

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    example = Example(
        viewer,
        args.verbose,
        args.num_rollouts,
        args.render_rollouts,
        args.drone_path,
    )

    # Run example
    newton.examples.run(example, args)
