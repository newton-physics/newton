# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Warp script to run simulations on USD files
#
# Simulates the stage of the input USD file as described by the USD Physics
# definitions.
#
###########################################################################

from enum import Enum
from typing import Optional
from pathlib import Path

from pxr import Usd

import warp as wp
import newton
from newton.utils import parse_usd
import numpy as np


class IntegratorType(Enum):
    EULER = "euler"
    XPBD = "xpbd"
    VBD = "vbd"
    MJWARP = "mjwarp"

    def __str__(self):
        return self.value


class Simulator:
    # TODO: make logic for the case when attributes can be specified in multiple places
    #       eg: fps specified on the stage or physxScene:timeStepsPerSecond for substeps

    MODEL_ATTRIBUTES = {
        "newton:joint_attach_kd": "joint_attach_kd",
        "newton:joint_attach_ke": "joint_attach_ke",
        "newton:soft_contact_kd": "soft_contact_kd",
        "newton:soft_contact_ke": "soft_contact_ke",
    }
    SOLVER_ATTRIBUTES = {
        "newton:collide_on_substeps": "collide_on_substeps",
        "newton:fps": "fps",
        "newton:integrator": "integrator_type",
        "newton:integrator_iterations": "integrator_iterations",
        "newton:substeps": "substeps",
    }
    INTEGRATOR_ATTRIBUTES = {
        IntegratorType.EULER: {
            "newton:euler:angular_damping": "angular_damping",
            "newton:euler:friction_smoothing": "friction_smoothing",
        },
        IntegratorType.VBD: {"newton:vbd:friction_epsilon": "friction_epsilon"},
        IntegratorType.XPBD: {
            "newton:xpbd:soft_body_relaxation": "soft_body_relaxation",
            "newton:xpbd:soft_contact_relaxation": "soft_contact_relaxation",
            "newton:xpbd:joint_linear_relaxation": "joint_linear_relaxation",
            "newton:xpbd:joint_angular_relaxation": "joint_angular_relaxation",
            "newton:xpbd:rigid_contact_relaxation": "rigid_contact_relaxation",
            "newton:xpbd:rigid_contact_con_weighting": "rigid_contact_con_weighting",
            "newton:xpbd:angular_damping": "angular_damping",
            "newton:xpbd:enable_restitution": "enable_restitution",
        },
        IntegratorType.MJWARP: {
            "newton:mjwarp:use_mujoco_cpu": "use_mujoco_cpu",
            "newton:mjwarp:solver": "solver",
            "newton:mjwarp:integrator": "integrator",
            "newton:mjwarp:iterations": "iterations",
            "newton:mjwarp:ls_iterations": "ls_iterations",
            "newton:mjwarp:save_to_mjcf": "save_to_mjcf",
            "newton:mjwarp:contact_stiffness_time_const": "contact_stiffness_time_const",
        },
    }
    MODEL_ATTRIBUTES_KEYS = MODEL_ATTRIBUTES.keys()
    SOLVER_ATTRIBUTES_KEYS = SOLVER_ATTRIBUTES.keys()

    def __init__(self, input_path, output_path, integrator: Optional[IntegratorType] = None):
        def create_stage_from_path(input_path) -> Usd.Stage:
            stage = Usd.Stage.Open(input_path, Usd.Stage.LoadAll)
            flattened = stage.Flatten()
            out_stage = Usd.Stage.Open(flattened.identifier)
            return out_stage

        self.sim_time = 0.0
        self.profiler = {}

        self.in_stage = create_stage_from_path(input_path)

        builder = newton.ModelBuilder()
        builder.up_axis = newton.Axis.Z
        results = parse_usd(
            self.in_stage,
            builder,
            invert_rotations=True,
            collapse_fixed_joints=True,
        )

        scene_attributes = results["scene_attributes"]
        self._apply_solver_attributes(scene_attributes)

        if integrator:
            self.integrator_type = integrator

        if self.integrator_type == IntegratorType.VBD:
            builder.color()
        self.model = builder.finalize()
        self.builder_results = results
        orig_orientation_1 = self.model.body_q.numpy().flatten()[3:]
        print(f"__init__ orig_orientation_1 = {orig_orientation_1}")

        self.path_body_map = self.builder_results["path_body_map"]
        collapse_results = self.builder_results["collapse_results"]
        self.path_body_relative_transform = self.builder_results["path_body_relative_transform"]
        if collapse_results:
            self.body_remap = collapse_results["body_remap"]
            self.body_merged_parent = collapse_results["body_merged_parent"]
            self.body_merged_transform = collapse_results["body_merged_transform"]
        else:
            self.body_remap = None
            self.body_merged_parent = None
            self.body_merged_transform = None

        self._apply_model_attributes(scene_attributes)
        self._setup_integrator(scene_attributes)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # NB: body_q will be modified, so initial state will be slightly altered
        if self.model.joint_count:
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0, mask=None)

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

        self.renderer = newton.viewer.RendererUsd(
            self.model,
            stage=output_path,
            source_stage=input_path,
            path_body_relative_transform=self.path_body_relative_transform,
            path_body_map=self.path_body_map,
            builder_results=self.builder_results,
        )

    def _apply_solver_attributes(self, scene_attributes: dict):
        """Apply scene attributes parsed from the stage to self."""

        # Defaults
        self.fps = 60
        self.sim_substeps = 32
        self.integrator_type = IntegratorType.XPBD
        self.integrator_iterations = 100
        self.collide_on_substeps = True

        # Loading attributes
        set_attrs = set(scene_attributes.keys())
        solver_attrs = set_attrs.intersection(self.SOLVER_ATTRIBUTES_KEYS)

        for attr in solver_attrs:
            self.__dict__[self.SOLVER_ATTRIBUTES[attr]] = scene_attributes[attr]

        # Derived/computed attributes that depend on the above
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.integrator_type = IntegratorType(self.integrator_type)

    def _apply_model_attributes(self, scene_attributes: dict):
        """Apply scene attributes parsed from the stage to the model."""

        # Defaults
        self.model.ground = False
        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = 1.0e2

        # Loading attributes
        set_attrs = set(scene_attributes.keys())
        model_attrs = set_attrs.intersection(self.MODEL_ATTRIBUTES_KEYS)

        for attr in model_attrs:
            self.model.__dict__[self.MODEL_ATTRIBUTES[attr]] = scene_attributes[attr]

    def _setup_integrator(self, scene_attributes: dict):
        """Set up the integrator, and apply attributes parsed from the stage."""

        # TODO: add Euler integrator
        # if self.integrator_type == IntegratorType.EULER:
        #     self.integrator = newton.SemiImplicitIntegrator()
        if self.integrator_type == IntegratorType.XPBD:
            solver_args = {"iterations": scene_attributes.get("newton:xpbd:iterations", self.integrator_iterations)}
            self.integrator = newton.solvers.XPBDSolver(self.model, **solver_args)
        elif self.integrator_type == IntegratorType.MJWARP:
            solver_args = {
                "use_mujoco_cpu": scene_attributes.get("newton:mjwarp:use_mujoco_cpu", False),
                "solver": scene_attributes.get("newton:mjwarp:solver", "newton"),
                "integrator": scene_attributes.get("newton:mjwarp:integrator", "euler"),
                "iterations": scene_attributes.get("newton:mjwarp:iterations", self.integrator_iterations),
                "ls_iterations": scene_attributes.get("newton:mjwarp:ls_iterations", 5),
                "save_to_mjcf": scene_attributes.get("newton:mjwarp:save_to_mjcf", "sim_usd_mjcf.xml"),
                "contact_stiffness_time_const": scene_attributes.get(
                    "newton:mjwarp:contact_stiffness_time_const", 0.02
                ),
            }
            self.integrator = newton.solvers.SolverMuJoCo(
                self.model,
                **solver_args,
            )
        else:
            self.integrator = newton.solvers.VBDIntegrator(self.model, iterations=self.integrator_iterations)

        # Loading attributes
        set_attrs = set(scene_attributes.keys())
        cur_integrator = self.INTEGRATOR_ATTRIBUTES[self.integrator_type]
        integrator_attrs = set_attrs.intersection(cur_integrator.keys())
        for attr in integrator_attrs:
            self.integrator.__dict__[cur_integrator[attr]] = scene_attributes[attr]

    def simulate(self):
        rigid_contact_margin = 0.1
        if not self.collide_on_substeps:
            self.contacts = self.model.collide(self.state_0, rigid_contact_margin=rigid_contact_margin)

        for _ in range(self.sim_substeps):
            if self.collide_on_substeps:
                self.contacts = self.model.collide(self.state_0, rigid_contact_margin=rigid_contact_margin)

            self.state_0.clear_forces()
            self.integrator.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)

            # swap states
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        with wp.ScopedTimer("step", dict=self.profiler):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        with wp.ScopedTimer("render", dict=self.profiler):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_update_stage(self.state_0)
            self.renderer.end_frame()

    def save(self):
        self.renderer.save()


def print_time_profiler(simulator):
    frame_times = simulator.profiler["step"]
    render_times = simulator.profiler["render"]
    print("\nAverage frame sim time: {:.2f} ms".format(sum(frame_times) / len(frame_times)))
    print("\nAverage frame render time: {:.2f} ms".format(sum(render_times) / len(render_times)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "stage_path",
        help="Path to the input USD file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output USD file.",
    )
    parser.add_argument("-d", "--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("-n", "--num_frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument(
        "-i",
        "--integrator",
        help="Type of integrator",
        type=IntegratorType,
        choices=list(IntegratorType),
        default=None,
    )

    args = parser.parse_known_args()[0]

    if not args.output:
        path = Path(args.stage_path)
        base_path = path.parent / "output"
        base_path.mkdir(parents=True, exist_ok=True)
        args.output = str(base_path / path.name)
        print(f'Output path not specified (-o flag). Writing to "{args.output}".')

    with wp.ScopedDevice(args.device):
        simulator = Simulator(input_path=args.stage_path, output_path=args.output, integrator=args.integrator)

        for i in range(args.num_frames):
            print(f"frame {i}")
            simulator.step()
            simulator.render()

        print_time_profiler(simulator)

        simulator.save()
