# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Used for benchmarking MjWarp.
#
# This script allows us to choose between several predefined robots and
# provides a large range of customizable options.
#
###########################################################################


import time

import numpy as np
import warp as wp

wp.config.enable_backward = False

import newton
import newton.examples
import newton.utils
from newton.sensors import SensorContact
from newton.utils import EventTracer

_NEW_LAYOUT_AVAILABLE = hasattr(newton, "use_coord_layout_targets")


def _target_q(owner):
    """Resolve the joint-position-target array across pre/post #2556 layouts.

    On pre-PR Newton (no ``joint_target_q``) falls back to ``joint_target_pos``.
    Used by the benchmark harness so ``asv compare`` works against both refs.
    """
    target = getattr(owner, "joint_target_q", None)
    if target is None:
        target = getattr(owner, "joint_target_pos", None)
    return target


# Slew targets toward per-joint random waypoints instead of teleporting them:
# unbounded setpoint jumps create solves the MuJoCo Newton solver cannot finish
# within any iteration budget (exposed by mujoco_warp#1374), so the benchmark
# measured the iteration cap instead of converged stepping.
TARGET_SLEW_PER_FRAME = 0.01  # fraction of each target's range per frame
TARGET_WAYPOINT_FRAMES = 50  # frames between waypoint draws


@wp.kernel
def _waypoint_control(
    seed: int,
    frame: wp.array[int],
    lo: wp.array[float],
    hi: wp.array[float],
    slew_per_frame: float,
    waypoint_frames: int,
    joint_target: wp.array[float],
):
    # The waypoint is a pure function of (seed, leg, tid): deterministic, and
    # no state beyond the frame counter, so the kernel is graph-capturable.
    tid = wp.tid()
    leg = frame[0] // waypoint_frames
    state = wp.rand_init(seed, leg * joint_target.shape[0] + tid)
    span = hi[tid] - lo[tid]
    waypoint = lo[tid] + wp.randf(state) * span
    slew = slew_per_frame * span
    joint_target[tid] = joint_target[tid] + wp.clamp(waypoint - joint_target[tid], -slew, slew)


@wp.kernel
def _advance_frame(frame: wp.array[int]):
    frame[0] = frame[0] + 1


def _target_bounds(model):
    """Per-entry bounds for randomized position targets in the ``_target_q``
    layout. Joints whose coords map 1:1 to dofs get their limits intersected
    with the historical [-1, 1] command range (sentinel/unlimited sides fall
    back to +-1); free/ball/distance coordinates keep the initial target.
    """
    init = _target_q(model).numpy().astype(np.float32)
    lo, hi = init.copy(), init.copy()
    joint_types = model.joint_type.numpy()
    q_starts = model.joint_q_start.numpy()
    qd_starts = model.joint_qd_start.numpy()
    limit_lo = model.joint_limit_lower.numpy()
    limit_hi = model.joint_limit_upper.numpy()
    dof_total = limit_lo.shape[0]
    coord_layout = init.shape[0] == model.joint_coord_count
    quat_joints = (int(newton.JointType.FREE), int(newton.JointType.BALL), int(newton.JointType.DISTANCE))
    for j, jt in enumerate(joint_types):
        if jt in quat_joints:
            continue
        d0 = int(qd_starts[j])
        d1 = int(qd_starts[j + 1]) if j + 1 < len(qd_starts) else dof_total
        for d in range(d0, d1):
            i = int(q_starts[j]) + (d - d0) if coord_layout else d
            lower_raw = float(limit_lo[d])
            upper_raw = float(limit_hi[d])
            lower = max(lower_raw, -1.0) if lower_raw > -1.0e6 else -1.0
            upper = min(upper_raw, 1.0) if upper_raw < 1.0e6 else 1.0
            if lower >= upper and -1.0e6 < lower_raw < upper_raw < 1.0e6:
                lower, upper = lower_raw, upper_raw  # bounded range outside [-1, 1]
            if lower < upper:
                lo[i], hi[i] = lower, upper
    return lo, hi


ROBOT_CONFIGS = {
    "humanoid": {
        "solver": "newton",
        "integrator": "implicitfast",
        "njmax": 80,
        "nconmax": 25,
        "cone": "pyramidal",
        "sensing_bodies": ["*thigh*", "*shin*", "*foot*", "*arm*"],
    },
    "g1": {
        "solver": "newton",
        "integrator": "implicitfast",
        "njmax": 210,
        "nconmax": 35,
        "cone": "pyramidal",
    },
    "h1": {
        "solver": "newton",
        "integrator": "implicitfast",
        "njmax": 65,
        "nconmax": 15,
        "cone": "pyramidal",
    },
    "cartpole": {
        "solver": "newton",
        "integrator": "implicitfast",
        "njmax": 5,
        "nconmax": 0,
        "cone": "pyramidal",
    },
    "ant": {
        "solver": "newton",
        "integrator": "implicitfast",
        "njmax": 38,
        "nconmax": 15,
        "cone": "pyramidal",
    },
    "quadruped": {
        "solver": "newton",
        "integrator": "implicitfast",
        "njmax": 75,
        "nconmax": 50,
        "cone": "pyramidal",
    },
    "allegro": {
        "solver": "newton",
        "integrator": "implicitfast",
        "njmax": 60,
        "nconmax": 40,
        "cone": "elliptic",
    },
    "kitchen": {
        "setup_builder": lambda x: _setup_kitchen(x),  # noqa: PLW0108  # lambda defers lookup
        "njmax": 3800,
        "nconmax": 900,
    },
    "tabletop": {
        "setup_builder": lambda x: _setup_tabletop(x),  # noqa: PLW0108  # lambda defers lookup
        "njmax": 100,
        "nconmax": 20,
    },
}


def _setup_humanoid(articulation_builder):
    articulation_builder.add_mjcf(
        newton.examples.get_asset("nv_humanoid.xml"),
        ignore_names=["floor", "ground"],
        up_axis="Z",
        parse_sites=False,  # AD: remove once asset is fixed
        enable_self_collisions=False,  # Keep False for consistent benchmark performance
    )

    # Setting root pose
    root_dofs = 7
    articulation_builder.joint_q[:3] = [0.0, 0.0, 1.5]

    return root_dofs


def _setup_g1(articulation_builder):
    articulation_builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5
    )
    articulation_builder.default_shape_cfg.ke = 5.0e4
    articulation_builder.default_shape_cfg.kd = 5.0e2
    articulation_builder.default_shape_cfg.kf = 1.0e3
    articulation_builder.default_shape_cfg.mu = 0.75

    asset_path = newton.utils.download_asset("unitree_g1")

    articulation_builder.add_usd(
        str(asset_path / "usd" / "g1_isaac.usd"),
        xform=wp.transform(wp.vec3(0, 0, 0.8)),
        collapse_fixed_joints=True,
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )

    for i in range(6, articulation_builder.joint_dof_count):
        articulation_builder.joint_target_ke[i] = 1000.0
        articulation_builder.joint_target_kd[i] = 5.0

    # approximate meshes for faster collision detection
    articulation_builder.approximate_meshes("bounding_box")

    root_dofs = 7

    return root_dofs


def _setup_h1(articulation_builder):
    articulation_builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5
    )
    articulation_builder.default_shape_cfg.ke = 5.0e4
    articulation_builder.default_shape_cfg.kd = 5.0e2
    articulation_builder.default_shape_cfg.kf = 1.0e3
    articulation_builder.default_shape_cfg.mu = 0.75

    asset_path = newton.utils.download_asset("unitree_h1")
    asset_file = str(asset_path / "usd" / "h1_minimal.usda")
    articulation_builder.add_usd(
        asset_file,
        ignore_paths=["/GroundPlane"],
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )
    # approximate meshes for faster collision detection
    articulation_builder.approximate_meshes("bounding_box")

    for i in range(articulation_builder.joint_dof_count):
        articulation_builder.joint_target_ke[i] = 150
        articulation_builder.joint_target_kd[i] = 5

    root_dofs = 7

    return root_dofs


def _setup_cartpole(articulation_builder):
    articulation_builder.default_shape_cfg.density = 100.0
    articulation_builder.default_joint_cfg.armature = 0.1

    articulation_builder.add_usd(
        newton.examples.get_asset("cartpole_single_pendulum.usda"),
        enable_self_collisions=False,
        collapse_fixed_joints=True,
    )
    armature_inertia = wp.mat33(np.eye(3, dtype=np.float32)) * 0.1
    for i in range(articulation_builder.body_count):
        articulation_builder.body_inertia[i] = articulation_builder.body_inertia[i] + armature_inertia
    # set initial joint positions (cartpole has 2 joints: prismatic slider + revolute pole)
    # joint_q[0] = slider position, joint_q[1] = pole angle
    articulation_builder.joint_q[0] = 0.0  # slider at origin
    articulation_builder.joint_q[1] = 0.3  # pole tilted

    # Setting root pose
    root_dofs = 1

    return root_dofs


def _setup_ant(articulation_builder):
    articulation_builder.add_usd(
        newton.examples.get_asset("ant.usda"),
        collapse_fixed_joints=True,
    )

    # Setting root pose
    root_dofs = 7
    articulation_builder.joint_q[:3] = [0.0, 0.0, 1.5]

    return root_dofs


def _setup_quadruped(articulation_builder):
    articulation_builder.default_joint_cfg.armature = 0.01
    articulation_builder.default_shape_cfg.ke = 1.0e4
    articulation_builder.default_shape_cfg.kd = 1.0e2
    articulation_builder.default_shape_cfg.kf = 1.0e2
    articulation_builder.default_shape_cfg.mu = 1.0
    articulation_builder.add_urdf(
        newton.examples.get_asset("quadruped.urdf"),
        xform=wp.transform([0.0, 0.0, 0.7], wp.quat_identity()),
        floating=True,
        enable_self_collisions=False,
    )
    armature_inertia = wp.mat33(np.eye(3, dtype=np.float32)) * 0.01
    for i in range(articulation_builder.body_count):
        articulation_builder.body_inertia[i] = articulation_builder.body_inertia[i] + armature_inertia
    root_dofs = 7

    return root_dofs


def _setup_allegro(articulation_builder):
    asset_path = newton.utils.download_asset("wonik_allegro")
    asset_file = str(asset_path / "usd" / "allegro_left_hand_with_cube.usda")
    articulation_builder.rigid_gap = 0.0
    articulation_builder.add_usd(
        asset_file,
        xform=wp.transform(wp.vec3(0, 0, 0.5)),
        enable_self_collisions=True,
        ignore_paths=[".*Dummy", ".*CollisionPlane", ".*goal", ".*DexCube/visuals"],
    )

    # set joint targets and joint drive gains
    for i in range(articulation_builder.joint_dof_count):
        articulation_builder.joint_target_ke[i] = 150
        articulation_builder.joint_target_kd[i] = 20
    if _NEW_LAYOUT_AVAILABLE:
        articulation_builder.joint_target_q[:] = articulation_builder.joint_q
    else:
        for i in range(articulation_builder.joint_dof_count):
            articulation_builder.joint_target_pos[i] = 0.0
    root_dofs = 1

    return root_dofs


def _setup_kitchen(articulation_builder):
    asset_path = newton.utils.download_asset("kitchen")
    asset_file = str(asset_path / "mjcf" / "kitchen.xml")
    articulation_builder.add_mjcf(
        asset_file,
        collapse_fixed_joints=True,
        enable_self_collisions=False,  # Keep False for consistent benchmark performance
    )

    # Change pose of the robot to minimize overlap
    articulation_builder.joint_q[:2] = [1.5, -1.5]


def _setup_tabletop(articulation_builder):
    articulation_builder.add_mjcf(
        newton.examples.get_asset("tabletop.xml"),
        collapse_fixed_joints=True,
    )


class Example:
    def __init__(
        self,
        robot="humanoid",
        environment="None",
        stage_path=None,
        world_count=1,
        use_cuda_graph=True,
        use_mujoco_cpu=False,
        randomize=False,
        headless=False,
        actuation="None",
        solver=None,
        integrator=None,
        solver_iteration=None,
        ls_iteration=None,
        njmax=None,
        nconmax=None,
        builder=None,
        cone=None,
    ):
        if _NEW_LAYOUT_AVAILABLE:
            newton.use_coord_layout_targets = True
        fps = 600
        self.sim_time = 0.0
        self.benchmark_time = 0.0
        self.frame_dt = 1.0 / fps
        self.sim_substeps = 10
        self.contacts = None
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.world_count = world_count
        self.use_cuda_graph = use_cuda_graph
        self.use_mujoco_cpu = use_mujoco_cpu
        self.actuation = actuation

        self.seed = 123

        if not stage_path:
            stage_path = "example_" + robot + ".usd"

        if builder is None:
            builder = Example.create_model_builder(robot, world_count, environment, randomize, self.seed)

        # finalize model
        self.model = builder.finalize()

        self.solver = Example.create_solver(
            self.model,
            robot,
            use_mujoco_cpu=use_mujoco_cpu,
            environment=environment,
            solver=solver,
            integrator=integrator,
            solver_iteration=solver_iteration,
            ls_iteration=ls_iteration,
            njmax=njmax,
            nconmax=nconmax,
            cone=cone,
        )

        if stage_path and not headless:
            self.renderer = newton.viewer.ViewerGL()
            self.renderer.set_model(self.model)
            self.renderer.set_world_offsets((4.0, 4.0, 0.0))
        else:
            self.renderer = None

        self.control = self.model.control()
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        if self.actuation == "random":
            self.init_waypoint_control()

        self.sensor_contact = None
        sensing_bodies = ROBOT_CONFIGS.get(robot, {}).get("sensing_bodies", None)
        if sensing_bodies is not None:
            self.sensor_contact = SensorContact(self.model, sensing_bodies=sensing_bodies, counterpart_bodies="*")
            self.contacts = newton.Contacts(
                self.solver.get_max_contact_count(),
                0,
                device=self.model.device,
                requested_attributes=self.model.get_requested_contact_attributes(),
            )

        self.graph = None
        if self.use_cuda_graph:
            # simulate() allocates memory via a clone, so we can't use graph capture if the device does not support mempools
            cuda_graph_comp = wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device())
            if not cuda_graph_comp:
                print("Cannot use graph capture. Graph capture is disabled.")
            else:
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        if self.sensor_contact is not None:
            self.solver.update_contacts(self.contacts, self.state_0)
            self.sensor_contact.update(self.state_0, self.contacts)

    def init_waypoint_control(self):
        lo, hi = _target_bounds(self.model)
        self._target_lo = wp.array(lo, dtype=wp.float32)
        self._target_hi = wp.array(hi, dtype=wp.float32)
        self._target_frame = wp.zeros(1, dtype=int)

    def apply_waypoint_control(self):
        """Enqueue one frame of waypoint target tracking; safe to graph-capture."""
        target_q = _target_q(self.control)
        wp.launch(
            _waypoint_control,
            dim=(target_q.shape[0],),
            inputs=[
                self.seed,
                self._target_frame,
                self._target_lo,
                self._target_hi,
                TARGET_SLEW_PER_FRAME,
                TARGET_WAYPOINT_FRAMES,
            ],
            outputs=[target_q],
        )
        wp.launch(_advance_frame, dim=1, inputs=[self._target_frame])

    def step(self):
        if self.actuation == "random":
            self.apply_waypoint_control()

        wp.synchronize_device()
        start_time = time.time()
        if self.use_cuda_graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        wp.synchronize_device()
        end_time = time.time()

        self.benchmark_time += end_time - start_time
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        self.renderer.begin_frame(self.sim_time)
        self.renderer.log_state(self.state_0)
        self.renderer.end_frame()

    @staticmethod
    def create_model_builder(robot, world_count, environment="None", randomize=False, seed=123) -> newton.ModelBuilder:
        rng = np.random.default_rng(seed)

        articulation_builder = newton.ModelBuilder()
        articulation_builder.default_shape_cfg.ke = 1.0e3
        articulation_builder.default_shape_cfg.kd = 1.0e2
        newton.solvers.SolverMuJoCo.register_custom_attributes(articulation_builder)
        if robot == "humanoid":
            root_dofs = _setup_humanoid(articulation_builder)
        elif robot == "g1":
            root_dofs = _setup_g1(articulation_builder)
        elif robot == "h1":
            root_dofs = _setup_h1(articulation_builder)
        elif robot == "cartpole":
            root_dofs = _setup_cartpole(articulation_builder)
        elif robot == "ant":
            root_dofs = _setup_ant(articulation_builder)
        elif robot == "quadruped":
            root_dofs = _setup_quadruped(articulation_builder)
        elif robot == "allegro":
            root_dofs = _setup_allegro(articulation_builder)
        else:
            raise ValueError(f"Name of the provided robot not recognized: {robot}")

        custom_setup_fn = ROBOT_CONFIGS.get(environment, {}).get("setup_builder", None)
        if custom_setup_fn is not None:
            custom_setup_fn(articulation_builder)

        builder = newton.ModelBuilder()
        builder.rigid_gap = articulation_builder.rigid_gap
        builder.replicate(articulation_builder, world_count)
        if randomize:
            njoint = len(articulation_builder.joint_q)
            for i in range(world_count):
                istart = i * njoint
                builder.joint_q[istart + root_dofs : istart + njoint] = rng.uniform(
                    -1.0, 1.0, size=(njoint - root_dofs)
                ).tolist()
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        if robot != "cartpole":
            # Disable all collisions for the cartpole benchmark
            builder.add_ground_plane()
        return builder

    @staticmethod
    def create_solver(
        model,
        robot,
        *,
        use_mujoco_cpu=False,
        environment="None",
        solver=None,
        integrator=None,
        solver_iteration=None,
        ls_iteration=None,
        njmax=None,
        nconmax=None,
        cone=None,
    ):
        solver_iteration = solver_iteration if solver_iteration is not None else 100
        ls_iteration = ls_iteration if ls_iteration is not None else 50
        solver = solver if solver is not None else ROBOT_CONFIGS[robot]["solver"]
        integrator = integrator if integrator is not None else ROBOT_CONFIGS[robot]["integrator"]
        njmax = njmax if njmax is not None else ROBOT_CONFIGS[robot]["njmax"]
        nconmax = nconmax if nconmax is not None else ROBOT_CONFIGS[robot]["nconmax"]
        cone = cone if cone is not None else ROBOT_CONFIGS[robot]["cone"]

        njmax += ROBOT_CONFIGS.get(environment, {}).get("njmax", 0)
        nconmax += ROBOT_CONFIGS.get(environment, {}).get("nconmax", 0)

        return newton.solvers.SolverMuJoCo(
            model,
            use_mujoco_cpu=use_mujoco_cpu,
            solver=solver,
            integrator=integrator,
            iterations=solver_iteration,
            ls_iterations=ls_iteration,
            njmax=njmax,
            nconmax=nconmax,
            cone=cone,
        )


def print_trace(trace, indent, steps):
    if indent == 0:
        print("================= Profiling =================")
    for k, v in trace.items():
        times, sub_trace = v
        print("  " * indent + f"{k}: {times / steps:.4f}")
        print_trace(sub_trace, indent + 1, steps)
    if indent == 0:
        step_time = trace["step"][0]
        step_trace = trace["step"][1]
        mujoco_warp_step_time = step_trace["_mujoco_warp_step"][0]
        overhead = 100.0 * (step_time - mujoco_warp_step_time) / step_time
        print("---------------------------------------------")
        print(f"Newton overhead:\t{overhead:.2f} %")
        print("=============================================")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--robot", type=str, default="humanoid", help="Name of the robot to simulate.")
    parser.add_argument("--env", type=str, default="None", help="Name of the environment where the robot is located.")
    parser.add_argument(
        "--event-trace", default=False, action=argparse.BooleanOptionalAction, help="Print profiling information."
    )
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default=None,
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=12000, help="Total number of frames.")
    parser.add_argument("--world-count", type=int, default=1, help="Total number of simulated worlds.")
    parser.add_argument("--use-cuda-graph", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--use-mujoco-cpu",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use Mujoco-C CPU (Not yet supported).",
    )
    parser.add_argument(
        "--headless", default=False, action=argparse.BooleanOptionalAction, help="Run the simulation in headless mode."
    )

    parser.add_argument(
        "--random-init", default=False, action=argparse.BooleanOptionalAction, help="Randomize initial pose."
    )
    parser.add_argument(
        "--actuation",
        type=str,
        default="None",
        choices=["None", "random"],
        help="Type of action to apply at each step.",
    )

    parser.add_argument(
        "--solver", type=str, default=None, choices=["cg", "newton"], help="Mujoco model constraint solver used."
    )
    parser.add_argument(
        "--integrator", type=str, default=None, choices=["euler", "rk4", "implicit"], help="Mujoco integrator used."
    )
    parser.add_argument("--solver-iteration", type=int, default=None, help="Number of solver iterations.")
    parser.add_argument("--ls-iteration", type=int, default=None, help="Number of linesearch iterations.")
    parser.add_argument("--njmax", type=int, default=None, help="Maximum number of constraints per world.")
    parser.add_argument("--nconmax", type=int, default=None, help="Maximum number of collision per world.")
    parser.add_argument("--cone", type=str, default=None, choices=["pyramidal", "elliptic"], help="Friction cone type.")

    args = parser.parse_known_args()[0]

    if args.use_mujoco_cpu:
        args.use_mujoco_cpu = False
        print("The option ``use-mujoco-cpu`` is not yet supported. Disabling it.")

    trace = {}
    with EventTracer(enabled=args.event_trace) as tracer:
        with wp.ScopedDevice(args.device):
            example = Example(
                robot=args.robot,
                environment=args.env,
                stage_path=args.stage_path,
                world_count=args.world_count,
                use_cuda_graph=args.use_cuda_graph,
                use_mujoco_cpu=args.use_mujoco_cpu,
                randomize=args.random_init,
                headless=args.headless,
                actuation=args.actuation,
                solver=args.solver,
                integrator=args.integrator,
                solver_iteration=args.solver_iteration,
                ls_iteration=args.ls_iteration,
                njmax=args.njmax,
                nconmax=args.nconmax,
                cone=args.cone,
            )

            # Print simulation configuration summary
            LABEL_WIDTH = 25
            TOTAL_WIDTH = 45
            title = " Simulation Configuration "
            print(f"\n{title.center(TOTAL_WIDTH, '=')}")
            print(f"{'Simulation Steps':<{LABEL_WIDTH}}: {args.num_frames * example.sim_substeps}")
            print(f"{'World Count':<{LABEL_WIDTH}}: {args.world_count}")
            print(f"{'Robot Type':<{LABEL_WIDTH}}: {args.robot}")
            print(f"{'Timestep (dt)':<{LABEL_WIDTH}}: {example.sim_dt:.6f}s")
            print(f"{'Randomize Initial Pose':<{LABEL_WIDTH}}: {args.random_init!s}")
            print("-" * TOTAL_WIDTH)

            # Map MuJoCo solver enum back to string
            solver_value = example.solver.mj_model.opt.solver
            solver_map = {0: "PGS", 1: "CG", 2: "Newton"}  # mjSOL_PGS = 0, mjSOL_CG = 1, mjSOL_NEWTON = 2
            actual_solver = solver_map.get(solver_value, f"unknown({solver_value})")
            # Map MuJoCo integrator enum back to string
            integrator_map = {
                0: "Euler",
                1: "RK4",
                2: "Implicit",
                3: "Implicitfast",
            }  # mjINT_EULER = 0, mjINT_RK4 = 1, mjINT_IMPLICIT = 2, mjINT_IMPLICITFAST = 3
            actual_integrator = integrator_map.get(example.solver.mj_model.opt.integrator, "unknown")
            # Map MuJoCo cone enum back to string
            cone_value = example.solver.mj_model.opt.cone
            cone_map = {0: "pyramidal", 1: "elliptic"}  # mjCONE_PYRAMIDAL = 0, mjCONE_ELLIPTIC = 1
            actual_cone = cone_map.get(cone_value, f"unknown({cone_value})")
            # Get actual max constraints and contacts from MuJoCo Warp data
            actual_njmax = example.solver.mjw_data.njmax
            actual_nconmax = (
                example.solver.mjw_data.naconmax // args.world_count
                if args.world_count > 0
                else example.solver.mjw_data.naconmax
            )
            print(f"{'Solver':<{LABEL_WIDTH}}: {actual_solver}")
            print(f"{'Integrator':<{LABEL_WIDTH}}: {actual_integrator}")
            print(f"{'Cone':<{LABEL_WIDTH}}: {actual_cone}")
            print(f"{'Solver Iterations':<{LABEL_WIDTH}}: {example.solver.mj_model.opt.iterations}")
            print(f"{'Line Search Iterations':<{LABEL_WIDTH}}: {example.solver.mj_model.opt.ls_iterations}")
            print(f"{'Max Constraints / world':<{LABEL_WIDTH}}: {actual_njmax}")
            print(f"{'Max Contacts / world':<{LABEL_WIDTH}}: {actual_nconmax}")
            print(f"{'Joint DOFs':<{LABEL_WIDTH}}: {example.model.joint_dof_count}")
            print(f"{'Body Count':<{LABEL_WIDTH}}: {example.model.body_count}")
            print("-" * TOTAL_WIDTH)

            print(f"{'Execution Device':<{LABEL_WIDTH}}: {wp.get_device()}")
            print(f"{'Use CUDA Graph':<{LABEL_WIDTH}}: {example.use_cuda_graph!s}")
            print("=" * TOTAL_WIDTH + "\n")

            for _ in range(args.num_frames):
                example.step()
                example.render()
                if args.event_trace:
                    trace = tracer.add_trace(trace, tracer.trace())

    if args.event_trace:
        print_trace(trace, 0, args.num_frames * example.sim_substeps)
