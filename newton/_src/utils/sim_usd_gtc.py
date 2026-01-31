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
# Newton script to run simulations on USD files
#
# Simulates the stage of the input USD file as described by the USD Physics
# definitions.
#
###########################################################################

import inspect
import itertools
import os
from enum import Enum
from pathlib import Path
from typing import ClassVar, Optional

import numpy as np
import warp as wp

# wp.config.verify_cuda = True
# wp.config.verify_fp = True
from pxr import Usd, UsdGeom, UsdPhysics

import newton
from newton._src.core.spatial import angvel_between_quats
from newton._src.utils.import_usd import parse_usd
from newton._src.usd.schema_resolver import (
    SchemaAttribute as Attribute,
    PrimType,
    SchemaResolver,
    SchemaResolverManager,
)
from newton._src.utils.update_usd import UpdateUsd


def parse_xform(prim, time=Usd.TimeCode.Default(), return_mat=False):
    from pxr import UsdGeom

    xform = UsdGeom.Xform(prim)
    mat = np.array(xform.ComputeLocalToWorldTransform(time), dtype=np.float32)
    if return_mat:
        return mat
    rot = wp.quat_from_matrix(wp.mat33(mat[:3, :3].T.flatten()))
    pos = mat[3, :3]
    return wp.transform(pos, rot)


def _build_solver_args_from_resolver(
    resolver_mgr, prim, prim_type, solver_cls, defaults: dict[str, object] | None = None
):
    defaults = defaults or {}
    sig = inspect.signature(solver_cls.__init__)
    solver_args = {}
    for name, param in sig.parameters.items():
        # skip self, model, and var positional/keyword args: *args/**kwargs
        if name in ("self", "model") or param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        value = resolver_mgr.get_value(prim, prim_type, name, defaults.get(name))
        if value is not None:
            solver_args[name] = value
    return solver_args


class IntegratorType(Enum):
    EULER = "euler"
    XPBD = "xpbd"
    VBD = "vbd"
    MJWARP = "mjwarp"
    COUPLED_MPM = "cmpm"

    def __str__(self):
        return self.value


class SchemaResolverSimUsd(SchemaResolver):
    name: ClassVar[str] = "sim_usd"
    mapping: ClassVar[dict[PrimType, dict[str, Attribute]]] = {
        PrimType.SCENE: {
            # model attributes
            "joint_attach_kd": Attribute("newton:joint_attach_kd", 2718.0),
            "joint_attach_ke": Attribute("newton:joint_attach_ke", 2718.0),
            "soft_contact_ke": Attribute("newton:soft_contact_ke", 1.0e4),
            "soft_contact_kd": Attribute("newton:soft_contact_kd", 1.0e2),
            # solver attributes
            "fps": Attribute("newton:fps", 60),
            "sim_substeps": Attribute("newton:substeps", 12),
            "integrator_type": Attribute("newton:integrator", "xpbd"),
            "integrator_iterations": Attribute("newton:integrator_iterations", 32),
            "collide_on_substeps": Attribute("newton:collide_on_substeps", True),
        },
        PrimType.BODY: {
            "kinematic_collider": Attribute("physics:kinematicEnabled", False),
        },
        PrimType.SHAPE: {
            "kinematic_collider": Attribute("physics:kinematicEnabled", False),
        },
    }


class SchemaResolverEuler(SchemaResolver):
    name: ClassVar[str] = "euler"
    mapping: ClassVar[dict[PrimType, dict[str, Attribute]]] = {
        PrimType.SCENE: {
            "angular_damping": Attribute("newton:euler:angular_damping", 2718.0),
            "friction_smoothing": Attribute("newton:euler:friction_smoothing", 2718.0),
        },
    }


class SchemaResolverVBD(SchemaResolver):
    name: ClassVar[str] = "vbd"
    mapping: ClassVar[dict[PrimType, dict[str, Attribute]]] = {
        PrimType.SCENE: {
            "friction_epsilon": Attribute("newton:vbd:friction_epsilon", 2718.0),
        },
    }


class SchemaResolverXPBD(SchemaResolver):
    name: ClassVar[str] = "xpbd"
    mapping: ClassVar[dict[PrimType, dict[str, Attribute]]] = {
        PrimType.SCENE: {
            "soft_body_relaxation": Attribute("newton:xpbd:soft_body_relaxation", 0.9),
            "soft_contact_relaxation": Attribute("newton:xpbd:soft_contact_relaxation", 0.9),
            "joint_linear_relaxation": Attribute("newton:xpbd:joint_linear_relaxation", 1.0),
            "joint_angular_relaxation": Attribute("newton:xpbd:joint_angular_relaxation", 1.0),
            "rigid_contact_relaxation": Attribute("newton:xpbd:rigid_contact_relaxation", 1.0),
            "rigid_contact_con_weighting": Attribute("newton:xpbd:rigid_contact_con_weighting", True),
            "angular_damping": Attribute("newton:xpbd:angular_damping", 0.2),
            "enable_restitution": Attribute("newton:xpbd:enable_restitution", False),
        },
    }


class SchemaResolverMJWarp(SchemaResolver):
    name: ClassVar[str] = "mjwarp"
    mapping: ClassVar[dict[PrimType, dict[str, Attribute]]] = {
        PrimType.SCENE: {
            "use_mujoco_cpu": Attribute("newton:mjwarp:use_mujoco_cpu", False),
            "solver": Attribute("newton:mjwarp:solver", "newton"),
            "integrator": Attribute("newton:mjwarp:integrator", "implicitfast"),
            "iterations": Attribute("newton:mjwarp:iterations", 30),
            "ls_iterations": Attribute("newton:mjwarp:ls_iterations", 15),
            "save_to_mjcf": Attribute("newton:mjwarp:save_to_mjcf", "sim_usd_mjcf.xml"),
            "contact_stiffness_time_const": Attribute("newton:mjwarp:contact_stiffness_time_const", 0.02),
            "ncon_per_world": Attribute("newton:mjwarp:ncon_per_world", 150),
            "njmax": Attribute("newton:mjwarp:njmax", 300),
        },
    }


class SchemaResolverCoupledMPM(SchemaResolver):
    name: ClassVar[str] = "cmpm"
    mapping: ClassVar[dict[PrimType, dict[str, Attribute]]] = {
        PrimType.SCENE: {},
    }


class SchemaResolverMPM(SchemaResolver):
    name: ClassVar[str] = "mpm"
    mapping: ClassVar[dict[PrimType, dict[str, Attribute]]] = {
        PrimType.SCENE: {
            "voxel_size": Attribute("newton:mpm:voxel_size", 0.05),
            "grid_type": Attribute("newton:mpm:grid_type", "sparse"),
            "max_iterations": Attribute("newton:mpm:max_iterations", 250),
        },
    }


class CoupledMPMIntegrator(newton.solvers.SolverBase):
    """Integrator for coupled MPM and rigid body solvers."""

    def __init__(self, model: newton.Model, particles_dict: dict[str, np.ndarray], **kwargs):
        super().__init__(model)

        self.particles_dict = particles_dict

        rigid_solver_kwargs = {}

        mpm_options = newton.solvers.SolverImplicitMPM.Options()
        for key, value in kwargs.items():
            if hasattr(mpm_options, key):
                setattr(mpm_options, key, value)
            else:
                rigid_solver_kwargs[key] = value

        mpm_model = self._build_mpm_model(model, mpm_options)

        self.mpm_solver = newton.solvers.SolverImplicitMPM(mpm_model, mpm_options)

        self.mpm_state_0 = mpm_model.model.state()
        self.mpm_state_1 = mpm_model.model.state()

        self.mpm_solver.enrich_state(self.mpm_state_0)
        self.mpm_solver.enrich_state(self.mpm_state_1)

        self.rigid_solver = newton.solvers.SolverXPBD(model, **rigid_solver_kwargs)

        self._initialized = False

        self.particle_render_colors = wp.full(
            mpm_model.model.particle_count, value=wp.vec3(0.7, 0.6, 0.4), dtype=wp.vec3, device=mpm_model.model.device
        )

    def step(
        self,
        state_0: newton.State,
        state_1: newton.State,
        control: newton.Control,
        contacts: newton.Contacts,
        dt: float,
    ):
        if not self._initialized:
            self.sand_body_forces = wp.zeros_like(state_0.body_f)
            self.collider_impulses = None
            self.collider_impulse_pos = None
            self.collider_impulse_ids = None
            self.collect_collider_impulses(self.mpm_state_0)
            self._initialized = True

        has_compliance_bodies = self.mpm_state_0.body_q is not None

        if has_compliance_bodies:
            wp.launch(
                compute_body_forces,
                dim=self.collider_impulse_ids.shape[0],
                inputs=[
                    dt,
                    self.collider_impulse_ids,
                    self.collider_impulses,
                    self.collider_impulse_pos,
                    self.collider_body_id,
                    state_0.body_q,
                    self.model.body_com,
                    self.model.body_mass,
                    state_0.body_f,
                ],
            )

        self.sand_body_forces.assign(state_0.body_f)
        self.rigid_solver.step(state_0, state_1, control, contacts, dt)

        # Subtract previously applied impulses from body velocities
        if has_compliance_bodies:
            wp.launch(
                substract_body_force,
                dim=self.mpm_state_0.body_q.shape,
                inputs=[
                    dt,
                    state_1.body_q,
                    state_1.body_qd,
                    self.sand_body_forces,
                    self.model.body_inv_inertia,
                    self.model.body_inv_mass,
                    self.mpm_state_0.body_q,
                    self.mpm_state_0.body_qd,
                ],
            )

        self.mpm_solver.step(self.mpm_state_0, self.mpm_state_1, None, None, dt)

        self.collect_collider_impulses(self.mpm_state_1)

        self.mpm_state_0, self.mpm_state_1 = self.mpm_state_1, self.mpm_state_0

    def _build_mpm_model(self, model, mpm_options):
        sand_builder = newton.ModelBuilder()
        self._add_particles(sand_builder, mpm_options.voxel_size)
        sand_model = sand_builder.finalize()

        # basic particle material params
        sand_model.particle_mu = 0.48
        sand_model.particle_ke = 1.0e15
        sand_model.particle_kd = 0.0
        sand_model.particle_adhesion = 0.0
        sand_model.particle_cohesion = 0.0

        mpm_model = newton.solvers.SolverImplicitMPM.Model(sand_model, mpm_options)

        self._setup_mpm_collider(model, mpm_model)

        return mpm_model

    def _setup_mpm_collider(self, model: newton.Model, mpm_model: newton.solvers.SolverImplicitMPM.Model):
        collider_body_id = np.arange(-1, model.body_count)

        mpm_model.setup_collider(
            model=self.model,
            collider_body_ids=collider_body_id,
            collider_friction=[0.5 for _ in collider_body_id],
            collider_adhesion=[0.0 for _ in collider_body_id],
            # body_mass=wp.zeros_like(self.model.body_mass),  # so that the bodies are considered as kinematic,
            ground_height=-10.0,
        )

        self.collider_body_id = wp.array(collider_body_id, dtype=int)

    def _add_particles(self, sand_builder: newton.ModelBuilder, voxel_size: float):
        density = 2500

        if self.particles_dict:
            psize = voxel_size / 2.0
            radius = float(psize * 0.5)
            mass = float(psize**3 * density)

            points = np.vstack([pts for pts in self.particles_dict.values()])
        else:
            # ------------------------------------------
            # Add sand bed (1m x 2m x 0.5m) above ground
            # ------------------------------------------
            particles_per_cell = 3.0

            bed_lo = np.array([3.0, -1.0, 0.0])
            bed_hi = np.array([4.0, 1.0, 0.5])
            bed_res = np.array(np.ceil(particles_per_cell * (bed_hi - bed_lo) / voxel_size), dtype=int)

            # spawn particles on a jittered grid
            Nx, Ny, Nz = bed_res
            px = np.linspace(bed_lo[0], bed_hi[0], Nx + 1)
            py = np.linspace(bed_lo[1], bed_hi[1], Ny + 1)
            pz = np.linspace(bed_lo[2], bed_hi[2], Nz + 1)
            points = np.stack(np.meshgrid(px, py, pz)).reshape(3, -1).T

            cell_size = (bed_hi - bed_lo) / bed_res
            cell_volume = np.prod(cell_size)
            radius = float(np.max(cell_size) * 0.5)
            mass = float(np.prod(cell_volume) * density)

            rng = np.random.default_rng()
            points += 2.0 * radius * (rng.random(points.shape) - 0.5)

        sand_builder.particle_q = points
        sand_builder.particle_qd = np.zeros_like(points)
        sand_builder.particle_mass = np.full(points.shape[0], mass)
        sand_builder.particle_radius = np.full(points.shape[0], radius)
        sand_builder.particle_flags = np.ones(points.shape[0], dtype=int)

    def collect_collider_impulses(self, mpm_state):
        self.collider_impulses, self.collider_impulse_pos, self.collider_impulse_ids = (
            self.mpm_solver.collect_collider_impulses(mpm_state)
        )


@wp.kernel
def compute_body_forces(
    dt: float,
    collider_ids: wp.array(dtype=int),
    collider_impulses: wp.array(dtype=wp.vec3),
    collider_impulse_pos: wp.array(dtype=wp.vec3),
    body_ids: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_mass: wp.array(dtype=float),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    i = wp.tid()

    cid = collider_ids[i]
    if cid >= 0 and cid < body_ids.shape[0]:
        body_index = body_ids[cid]
        if body_index >= 0:
            m = body_mass[body_index]

            f_world = collider_impulses[i] / dt
            max_f = 0.0 * m
            if wp.length(f_world) > max_f:
                f_world = f_world * max_f / wp.length(f_world)

            X_wb = body_q[body_index]
            X_com = body_com[body_index]
            r = collider_impulse_pos[i] - wp.transform_point(X_wb, X_com)
            wp.atomic_add(body_f, body_index, wp.spatial_vector(f_world, wp.cross(r, f_world)))


@wp.kernel
def substract_body_force(
    dt: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=float),
    body_q_res: wp.array(dtype=wp.transform),
    body_qd_res: wp.array(dtype=wp.spatial_vector),
):
    body_id = wp.tid()

    # Remove previously applied force
    f = body_f[body_id]
    delta_v = dt * body_inv_mass[body_id] * wp.spatial_top(f)
    r = wp.transform_get_rotation(body_q[body_id])

    delta_w = dt * wp.quat_rotate(r, body_inv_inertia[body_id] * wp.quat_rotate_inv(r, wp.spatial_bottom(f)))

    body_q_res[body_id] = body_q[body_id]
    body_qd_res[body_id] = body_qd[body_id] - wp.spatial_vector(delta_v, delta_w)


@wp.kernel
def extract_rotation_and_scales(
    transform: wp.array(dtype=wp.mat33),
    rotation: wp.array(dtype=wp.vec4h),
    scales: wp.array(dtype=wp.vec3),
):
    i = wp.tid()

    A = transform[i]

    Q = wp.mat33()
    R = wp.mat33()
    wp.qr3(A, Q, R)

    q = wp.quat_from_matrix(Q)

    rotation[i] = wp.vec4h(wp.vec4(q[0], q[1], q[2], q[3]))
    scales[i] = wp.vec3(wp.length(R[0]), wp.length(R[1]), wp.length(R[2]))


def extract_particle_rotation_and_scales(state: newton.State):
    particle_transform_rotation = wp.empty(state.particle_count, dtype=wp.vec4h)
    particle_transform_scales = wp.empty(state.particle_count, dtype=wp.vec3)

    wp.launch(
        extract_rotation_and_scales,
        dim=state.particle_count,
        inputs=[
            state.particle_transform,
            particle_transform_rotation,
            particle_transform_scales,
        ],
    )

    return particle_transform_rotation, particle_transform_scales


class Simulator:
    def __init__(
        self,
        input_path,
        output_path,
        num_frames: int,
        integrator: Optional[IntegratorType] = None,
        sim_time: float = 0.0,
        sim_frame: int = 0,
        record_path: str = "",
        render_folder: str = "",
        usd_offset: wp.vec3 = wp.vec3(0.0, 0.0, 0.0),
        use_unified_collision_pipeline: bool = True,
        use_coacd: bool = False,
        enable_timers: bool = False,
        load_visual_shapes: bool = False,
    ):
        def create_stage_from_path(input_path) -> Usd.Stage:
            stage = Usd.Stage.Open(input_path, Usd.Stage.LoadAll)
            flattened = stage.Flatten()
            out_stage = Usd.Stage.Open(flattened.identifier)
            return out_stage

        self.profiler = {}
        self.usd_offset = usd_offset
        self.enable_timers = enable_timers
        self.record_path = record_path

        self.in_stage = create_stage_from_path(input_path)

        builder = newton.ModelBuilder()
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
        builder.default_joint_cfg.armature = 0.001
        builder.default_joint_cfg.friction = 0.0
        builder.up_axis = newton.Axis.Z
        builder.default_shape_cfg.density = 1000.0
        builder.default_shape_cfg.ke = 1.0e4
        builder.default_shape_cfg.kd = 5.0e1
        builder.default_shape_cfg.thickness = 0.001
        results = parse_usd(
            builder,
            self.in_stage,
            collapse_fixed_joints=False,  # ! keep it disabled for the lanterns to not disrupt the USD body mapping
            # ignore_paths=[".*BDXDroid"],
            xform=wp.transform(self.usd_offset, wp.quat_identity()),
            # hide_collision_shapes=False,
            skip_mesh_approximation=True,
            ignore_paths=[
                ".*proxy",
                # ".*proxy/Ground_Collider/mesh_0",
            ],
        )
        self.R = SchemaResolverManager([SchemaResolverSimUsd(), SchemaResolverMJWarp()])
        self.physics_prim = next(iter([prim for prim in self.in_stage.Traverse() if prim.IsA(UsdPhysics.Scene)]), None)

        # set up pair-wise filters for the BDX Droid shapes to disable self collisions
        droid_shapes: list[int] = []
        for i, key in enumerate(builder.shape_key):
            if "BDXDroid" in key:
                droid_shapes.append(i)
        for shape1, shape2 in itertools.combinations(droid_shapes, 2):
            builder.shape_collision_filter_pairs.append((shape1, shape2))

        self._setup_solver_attributes()
        self.sim_time = max(sim_time, sim_frame * self.frame_dt)

        # create custom attributes for the recorder
        if self.record_path:
            self._bdxlanterndemo_add_custom_attributes_for_recorder(builder)

        if integrator:
            self.integrator_type = integrator
        # TODO: this is where we want people to be able to write their own code. Need to go to self.prebuilder_finalize(). Other option is to use lambda/callback function.
        # disable collisions for the lantern chains
        chain_shapes = [i for i, key in enumerate(builder.shape_key) if "chain" in key.lower()]
        for shape1, shape2 in itertools.product(chain_shapes, range(builder.shape_count)):
            builder.shape_collision_filter_pairs.append((shape1, shape2))
            builder.shape_collision_filter_pairs.append((shape2, shape1))
        if use_unified_collision_pipeline:
            # convert all collision meshes to convex hulls but keep terrain high-res meshes (just for visualization)
            shape_indices = [
                i
                for i, (flags, key) in enumerate(zip(builder.shape_flags, builder.shape_key))
                if flags & newton.ShapeFlags.COLLIDE_SHAPES
                and "terrainMaincol" not in key
                and "ground_plane" not in key
            ]

            lantern_shapes = [i for i in shape_indices if "HangingLantern" in builder.shape_key[i]]
            other_shapes = [i for i in shape_indices if i not in lantern_shapes]

            if use_coacd:
                builder.approximate_meshes(
                    "coacd",
                    lantern_shapes,
                    keep_visual_shapes=True,
                    threshold=0.15,
                )
                builder.approximate_meshes(
                    "convex_hull",
                    other_shapes,
                    keep_visual_shapes=True,
                )
            else:
                builder.approximate_meshes(
                    "convex_hull",
                    lantern_shapes + other_shapes,
                    keep_visual_shapes=False,
                )

        self._collect_animated_colliders(builder, results["path_body_map"])
        if self.integrator_type == IntegratorType.VBD:
            builder.color()
        self.model = builder.finalize()
        self.builder_results = results

        self.path_body_map = self.builder_results["path_body_map"]
        self.path_shape_map = results["path_shape_map"]
        self.body_path_map = {idx: path for path, idx in self.path_body_map.items()}
        self.shape_path_map = {idx: path for path, idx in self.path_shape_map.items()}
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

        self._setup_model_attributes()
        self._setup_integrator()

        self.droid_is_walking = True
        self.droid_is_walking_wp = wp.array([self.droid_is_walking], dtype=wp.bool)
        self.current_frame = 0
        self.num_frames = num_frames

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # animation data
        self.collider_body_q = None
        self.collider_body_qd = None
        self.time_step_wp = wp.zeros(1, dtype=wp.int32)
        self._process_collider_animations(num_frames=num_frames)
        self._update_animated_colliders()
        # TODO: double check whether this is specific to MJWarp or XPBD
        if use_unified_collision_pipeline:
            self.collision_pipeline = newton.CollisionPipelineUnified.from_model(
                self.model,
                rigid_contact_max_per_pair=20,
                broad_phase_mode=newton.BroadPhaseMode.EXPLICIT,
            )
        else:
            self.collision_pipeline = newton.CollisionPipeline.from_model(
                self.model,
                rigid_contact_max_per_pair=20,
            )

        if self.integrator_type != IntegratorType.MJWARP:
            self.contacts = self.model.collide(
                self.state_0,
                collision_pipeline=self.collision_pipeline,
            )
        else:
            # use MuJoCo's own collision handling
            self.contacts = None

        # NB: body_q will be modified, so initial state will be slightly altered
        if self.model.joint_count:
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0, mask=None)

        # Setup USD updater to write into
        self.usd_updater = UpdateUsd(
            stage=output_path,
            source_stage=input_path,
            path_body_relative_transform=self.path_body_relative_transform,
            path_body_map=self.path_body_map,
            builder_results=self.builder_results,
            up_axis="Z",
            global_offset=self.usd_offset,
        )
        self.usd_updater.configure_body_mapping(
            path_body_map=self.path_body_map,
            path_body_relative_transform=self.path_body_relative_transform,
            builder_results=self.builder_results,
        )

        # TODO: has an option to configure Newton viewer?? Have a flag to turn on/off the viewer and to support USD Camera.
        self.show_viewer = True
        self.render_folder = ""
        if self.show_viewer:
            self.viewer = newton.viewer.ViewerGL()

            self.viewer.set_model(self.model)

            # add a UI callback to toggle the droid is walking
            def toggle_droid_is_walking(imgui):
                changed, droid_is_walking = imgui.checkbox("Droid is walking", self.droid_is_walking)
                if changed:
                    self.droid_is_walking = droid_is_walking
                    self.droid_is_walking_wp.fill_(self.droid_is_walking)

                imgui.text(f"Current frame: {self.current_frame} / {self.num_frames}")
                imgui.progress_bar(self.current_frame / self.num_frames)

            self.viewer.register_ui_callback(toggle_droid_is_walking, position="side")

            if self.record_path:
                self.viewer.start_recording(self.record_path)

            # Setup frame rendering to PNG if render_folder is specified
            self.render_folder = render_folder
            if self.render_folder:
                self.render_folder_path = Path(self.render_folder)
                self.render_folder_path.mkdir(parents=True, exist_ok=True)
                print(f"Saving rendered frames to: {self.render_folder_path}")

                # Pre-allocate frame buffer for efficiency
                h, w = self.viewer.renderer._screen_height, self.viewer.renderer._screen_width
                self.frame_buffer = wp.empty(shape=(h, w, 3), dtype=wp.uint8, device=self.viewer.device)

                # Try to import PIL/Pillow
                try:
                    from PIL import Image

                    self.image_module = Image
                    self.use_pil = True
                except ImportError:
                    print("Warning: PIL/Pillow not available. Trying imageio...")
                    try:
                        import imageio.v3 as iio

                        self.image_module = iio
                        self.use_pil = False
                    except ImportError:
                        print("Error: Neither PIL nor imageio available. Frame saving will be disabled.")
                        print("Install with: pip install Pillow")
                        self.render_folder = ""  # Disable frame saving
            else:
                self.render_folder = ""

        self.use_cuda_graph = wp.get_device().is_cuda
        self.is_mujoco_cpu_mode = self.integrator_type == IntegratorType.MJWARP and self.integrator.use_mujoco_cpu
        self.graph = None
        if self.use_cuda_graph and not self.is_mujoco_cpu_mode:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def _setup_solver_attributes(self):
        """Apply scene attributes parsed from the stage to self."""

        self.fps = self.R.get_value(self.physics_prim, PrimType.SCENE, "fps")
        self.sim_substeps = self.R.get_value(self.physics_prim, PrimType.SCENE, "sim_substeps")
        self.integrator_type = self.R.get_value(self.physics_prim, PrimType.SCENE, "integrator_type")
        self.integrator_iterations = self.R.get_value(self.physics_prim, PrimType.SCENE, "integrator_iterations")
        self.collide_on_substeps = self.R.get_value(self.physics_prim, PrimType.SCENE, "collide_on_substeps")

        # Derived/computed attributes that depend on the above
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.integrator_type = IntegratorType(self.integrator_type)
        self.rigid_contact_margin = self.R.get_value(self.physics_prim, PrimType.SCENE, "contact_margin", 0.001)

    def _setup_model_attributes(self):
        """Apply scene attributes parsed from the stage to the model."""

        # Defaults
        # TODO: set self.model.ground from the resolver manager
        self.model.soft_contact_kd = self.R.get_value(self.physics_prim, PrimType.SCENE, "soft_contact_kd")
        self.model.soft_contact_ke = self.R.get_value(self.physics_prim, PrimType.SCENE, "soft_contact_ke")

    def _setup_integrator(self):
        """Set up the integrator, and apply attributes parsed from the stage."""

        if self.integrator_type == IntegratorType.XPBD:
            res = SchemaResolverXPBD()
            R = SchemaResolverManager([res])
            solver_args = _build_solver_args_from_resolver(
                resolver_mgr=R,
                prim=self.physics_prim,
                prim_type=PrimType.SCENE,
                solver_cls=newton.solvers.SolverXPBD,
                defaults={"iterations": self.integrator_iterations},
            )
            self.integrator = newton.solvers.SolverXPBD(self.model, **solver_args)

        elif self.integrator_type == IntegratorType.MJWARP:
            res = SchemaResolverMJWarp()
            R = SchemaResolverManager([res])
            solver_args = _build_solver_args_from_resolver(
                resolver_mgr=R,
                prim=self.physics_prim,
                prim_type=PrimType.SCENE,
                solver_cls=newton.solvers.SolverMuJoCo,
                defaults={"iterations": self.integrator_iterations},
            )
            self.integrator = newton.solvers.SolverMuJoCo(
                self.model, **solver_args, ls_parallel=True, cone="elliptic", impratio=3.0
            )

        elif self.integrator_type == IntegratorType.COUPLED_MPM:
            res = SchemaResolverCoupledMPM()
            R = SchemaResolverManager([SchemaResolverMPM(), SchemaResolverXPBD()])
            solver_args = _build_solver_args_from_resolver(
                resolver_mgr=R,
                prim=self.physics_prim,
                prim_type=PrimType.SCENE,
                solver_cls=newton.solvers.SolverXPBD,
                defaults={"iterations": self.integrator_iterations},
            )
            mpm_solver_args = _build_solver_args_from_resolver(
                resolver_mgr=R,
                prim=self.physics_prim,
                prim_type=PrimType.SCENE,
                solver_cls=newton.solvers.SolverImplicitMPM.Options,
                defaults={},
            )

            particles_dict = {}
            for prim in self.in_stage.Traverse():
                if prim.GetTypeName() == "PointInstancer":
                    pi = UsdGeom.PointInstancer(prim)
                    particles_dict[prim.GetPath()] = np.array(pi.GetPositionsAttr().Get())

            self.integrator = CoupledMPMIntegrator(self.model, particles_dict, **solver_args, **mpm_solver_args)
        else:  # VBD
            res = SchemaResolverVBD()
            R = SchemaResolverManager([res])
            solver_args = _build_solver_args_from_resolver(
                resolver_mgr=R,
                prim=self.physics_prim,
                prim_type=PrimType.SCENE,
                solver_cls=newton.solvers.VBDIntegrator,
                defaults={"iterations": self.integrator_iterations},
            )
            self.integrator = newton.solvers.VBDIntegrator(self.model, **solver_args)

        # Iterate resolver-defined keys (these are your internal integrator attribute names)
        var_map = res.mapping.get(PrimType.SCENE, {})
        for key in var_map.keys():
            value = R.get_value(self.physics_prim, PrimType.SCENE, key)
            if value is not None and hasattr(self.integrator, key):
                setattr(self.integrator, key, value)

    def _collect_animated_colliders(self, builder, path_body_map):
        """
        Go through the builder mass array and set the inverse mass and inertia to 0 for kinematic bodies.
        """
        self.animated_colliders_body_ids = []
        self.animated_colliders_paths = []
        self.animated_colliders_joint_q_start = []  # start indices of joint_q of the free joints corresponding to the animated colliders
        self.animated_colliders_joint_qd_start = []  # start indices of joint_qd of the free joints corresponding to the animated colliders
        R = SchemaResolverManager([SchemaResolverSimUsd()])
        for path, body_id in path_body_map.items():
            kinematic_collider = R.get_value(self.in_stage.GetPrimAtPath(path), PrimType.BODY, "kinematic_collider")
            if kinematic_collider:
                builder.body_mass[body_id] = 0.0
                builder.body_inv_mass[body_id] = 0.0
                builder.body_inv_inertia[body_id] = wp.mat33(0.0)

                self.animated_colliders_body_ids.append(body_id)
                self.animated_colliders_paths.append(path)
                joint_id = builder.joint_child.index(body_id)
                self.animated_colliders_joint_q_start.append(builder.joint_q_start[joint_id])
                self.animated_colliders_joint_qd_start.append(builder.joint_qd_start[joint_id])
                # Mujoco requires nonzero inertia
                if self.integrator_type == IntegratorType.MJWARP:
                    m = 0.1
                    builder.body_mass[body_id] = m
                    builder.body_inv_mass[body_id] = 1 / m
                    builder.body_inertia[body_id] = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
                    builder.body_inv_inertia[body_id] = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        self.animated_colliders_joint_q_start_wp = wp.array(self.animated_colliders_joint_q_start, dtype=wp.int32)
        self.animated_colliders_joint_qd_start_wp = wp.array(self.animated_colliders_joint_qd_start, dtype=wp.int32)
        self.animated_colliders_body_ids_wp = wp.array(self.animated_colliders_body_ids, dtype=wp.int32)

    def _process_collider_animations(self, num_frames: int):
        xform_mats = []
        starting_frame = int(self.sim_time / self.frame_dt)

        for frame in range(starting_frame, starting_frame + num_frames + 1):
            sim_time = frame * self.frame_dt
            for substep in range(self.sim_substeps):
                time = self.fps * (sim_time + self.sim_dt / self.sim_substeps * float(substep))
                mats = []
                for path in self.animated_colliders_paths:
                    prim = self.in_stage.GetPrimAtPath(path)
                    mat = parse_xform(prim, time, return_mat=True)
                    mats.append(mat.T)
                xform_mats.append(mats)

        mats = wp.array2d(xform_mats, dtype=wp.mat44)

        num_bodies = len(self.animated_colliders_paths)
        self.collider_body_q = wp.empty((num_frames * self.sim_substeps, num_bodies), dtype=wp.transform)
        self.collider_body_qd = wp.empty((num_frames * self.sim_substeps, num_bodies), dtype=wp.spatial_vector)

        @wp.kernel(module="unique")
        def compute_collider_coordinates_kernel(
            xform_mats: wp.array2d(dtype=wp.mat44),
            usd_offset: wp.vec3,
            # outputs
            collider_body_q: wp.array2d(dtype=wp.transform),
            collider_body_qd: wp.array2d(dtype=wp.spatial_vector),
        ):
            frame, body_id = wp.tid()
            mat = xform_mats[frame, body_id]
            mat_next = xform_mats[frame + wp.static(self.sim_substeps), body_id]
            tf = wp.transform_from_matrix(mat)
            tf_next = wp.transform_from_matrix(mat_next)
            vel = (tf_next.p - tf.p) / wp.static(self.frame_dt)
            ang = angvel_between_quats(tf.q, tf_next.q, wp.static(self.frame_dt))
            tf.p += usd_offset
            collider_body_q[frame, body_id] = tf
            collider_body_qd[frame, body_id] = wp.spatial_vector(vel, ang)

        wp.launch(
            compute_collider_coordinates_kernel,
            dim=self.collider_body_q.shape,
            inputs=[mats, self.usd_offset],
            outputs=[self.collider_body_q, self.collider_body_qd],
        )

    @wp.kernel
    def _update_animated_colliders_kernel(
        time_step: wp.array(dtype=wp.int32),
        collider_body_q: wp.array2d(dtype=wp.transform),
        collider_body_qd: wp.array2d(dtype=wp.spatial_vector),
        colliders_joint_q_start: wp.array(dtype=wp.int32),
        colliders_joint_qd_start: wp.array(dtype=wp.int32),
        collider_body_ids: wp.array(dtype=int),
        # outputs
        body_q: wp.array(dtype=wp.transform),
        body_qd: wp.array(dtype=wp.spatial_vector),
        joint_q: wp.array(dtype=wp.float32),
        joint_qd: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        step = min(time_step[0], len(collider_body_q) - 1)
        body_id = collider_body_ids[i]
        q = collider_body_q[step, i]
        qd = collider_body_qd[step, i]

        # update maximal coordinates
        body_q[body_id] = q
        body_qd[body_id] = qd

        # update generalized coordinates (necessary for MuJoCo)
        q_start = colliders_joint_q_start[i]
        qd_start = colliders_joint_qd_start[i]
        for j in range(7):
            joint_q[q_start + j] = q[j]
        for j in range(6):
            joint_qd[qd_start + j] = qd[j]

    @wp.kernel
    def _advance_substep_time_kernel(
        time_step: wp.array(dtype=wp.int32),
        droid_is_walking: wp.array(dtype=bool),
    ):
        if droid_is_walking[0]:
            time_step[0] += 1

    def _advance_substep_time(self):
        wp.launch(
            self._advance_substep_time_kernel,
            dim=1,
            inputs=[self.time_step_wp, self.droid_is_walking_wp],
        )

    def _update_animated_colliders(self):
        wp.launch(
            self._update_animated_colliders_kernel,
            dim=len(self.animated_colliders_body_ids),
            inputs=[
                self.time_step_wp,
                self.collider_body_q,
                self.collider_body_qd,
                self.animated_colliders_joint_q_start_wp,
                self.animated_colliders_joint_qd_start_wp,
                self.animated_colliders_body_ids_wp,
            ],
            outputs=[self.state_0.body_q, self.state_0.body_qd, self.state_0.joint_q, self.state_0.joint_qd],
        )

    def _bdxlanterndemo_add_custom_attributes_for_recorder(self, builder):
        builder.add_custom_attribute(
            "mouse_position",
            newton.ModelAttributeFrequency.ONCE,
            default=wp.vec2(0.0, 0.0),
            dtype=wp.vec2,
            assignment=newton.ModelAttributeAssignment.STATE,
        )
        builder.add_custom_attribute(
            "droid_is_walking",
            newton.ModelAttributeFrequency.ONCE,
            default=False,
            dtype=bool,
            assignment=newton.ModelAttributeAssignment.STATE,
        )
        builder.add_custom_attribute(
            "camera_position",
            newton.ModelAttributeFrequency.ONCE,
            default=wp.vec3(0.0, 0.0, 0.0),
            dtype=wp.vec3,
            assignment=newton.ModelAttributeAssignment.STATE,
        )
        builder.add_custom_attribute(
            "camera_fov",
            newton.ModelAttributeFrequency.ONCE,
            default=45.0,
            dtype=float,
            assignment=newton.ModelAttributeAssignment.STATE,
        )
        builder.add_custom_attribute(
            "camera_pitch",
            newton.ModelAttributeFrequency.ONCE,
            default=0.0,
            dtype=float,
            assignment=newton.ModelAttributeAssignment.STATE,
        )
        builder.add_custom_attribute(
            "camera_yaw",
            newton.ModelAttributeFrequency.ONCE,
            default=-180.0,
            dtype=float,
            assignment=newton.ModelAttributeAssignment.STATE,
        )
        builder.add_custom_attribute(
            "picking_line",
            newton.ModelAttributeFrequency.ONCE,
            default=wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            dtype=wp.spatial_vector,
            assignment=newton.ModelAttributeAssignment.STATE,
        )

    def _bdxlanterndemo_setup_viewer_for_recording(self):
        if self.viewer.file_playback is None:
            # write camera, mouse coordinates to the state so we can record it to a file
            self.state_0.mouse_position.fill_(self.viewer.mouse_position)
            self.state_0.mouse_button.fill_(self.viewer.mouse_button)
            self.state_0.camera_position.fill_(wp.vec3(*self.viewer.camera.pos))
            self.state_0.camera_fov.fill_(self.viewer.camera.fov)
            self.state_0.camera_pitch.fill_(self.viewer.camera.pitch)
            self.state_0.camera_yaw.fill_(self.viewer.camera.yaw)
            self.state_0.droid_is_walking.fill_(self.droid_is_walking)
            if not self.viewer.picking.is_picking() or self.viewer.picking.pick_body.numpy()[0] < 0:
                self.state_0.picking_line.fill_(wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            else:
                pick_state = self.viewer.picking.pick_state.numpy()
                self.state_0.picking_line.fill_(wp.spatial_vector(*pick_state[8:14]))
        else:
            # Render picking line from playback
            pick_state = self.state_0.picking_line.numpy()[0]
            picked_point = wp.vec3(*pick_state[:3])
            pick_target = wp.vec3(*pick_state[3:])
            starts = wp.array([picked_point], dtype=wp.vec3, device=self.viewer.device)
            ends = wp.array([pick_target], dtype=wp.vec3, device=self.viewer.device)
            colors = wp.array([wp.vec3(0.0, 1.0, 1.0)], dtype=wp.vec3, device=self.viewer.device)
            self.viewer.log_lines("picking_line_playback", starts, ends, colors, hidden=False)

    def simulate(self):
        if not self.collide_on_substeps and self.integrator_type != IntegratorType.MJWARP:
            self.contacts = self.model.collide(
                self.state_0,
                collision_pipeline=self.collision_pipeline,
            )

        for _ in range(self.sim_substeps):
            self._update_animated_colliders()
            self._advance_substep_time()

            if self.collide_on_substeps and self.integrator_type != IntegratorType.MJWARP:
                self.contacts = self.model.collide(
                    self.state_0,
                    collision_pipeline=self.collision_pipeline,
                )

            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.integrator.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)

            # swap states
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        with wp.ScopedTimer("step", dict=self.profiler, active=self.enable_timers, synchronize=True):
            if self.graph is not None:
                try:
                    wp.capture_launch(self.graph)
                except Exception as e:
                    print(f"Error launching graph: {e}. Falling back to regular execution, which may be slower.")
                    self.graph = None
                    self.simulate()
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        # if self.integrator_type == IntegratorType.MJWARP:
        #     newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0, mask=None)

        with wp.ScopedTimer("render", dict=self.profiler, active=self.enable_timers, synchronize=True):
            if self.usd_updater is not None:
                self.usd_updater.begin_frame(self.sim_time)
                with wp.ScopedTimer("update_usd", dict=self.profiler, active=self.enable_timers, synchronize=True):
                    self.usd_updater.update_usd(self.state_0)
                if self.integrator_type == IntegratorType.COUPLED_MPM:
                    rot, scale = extract_particle_rotation_and_scales(self.integrator.mpm_state_0)
                    self.usd_updater.render_points(
                        path="/particles",
                        points=self.integrator.mpm_state_0.particle_q,
                        rotations=rot,
                        scales=scale,
                        radius=float(self.integrator.mpm_solver.mpm_model.model.particle_radius.numpy()[0]),
                    )
                self.usd_updater.end_frame()

            if self.show_viewer:
                if self.record_path:
                    self._bdxlanterndemo_setup_viewer_for_recording()

                # print(f"Mouse position: {self.state_0.mouse_position.numpy()[0]}, button: {self.state_0.mouse_button.numpy()[0]}")

                # if self.integrator_type == IntegratorType.MJWARP:
                #     self.integrator.render_mujoco_viewer()
                self.viewer.begin_frame(self.sim_time)
                with wp.ScopedTimer("log_state", dict=self.profiler, active=self.enable_timers, synchronize=True):
                    self.viewer.log_state(self.state_0)
                with wp.ScopedTimer("log_contacts", dict=self.profiler, active=self.enable_timers, synchronize=True):
                    self.viewer.log_contacts(self.contacts, self.state_0)
                if self.integrator_type == IntegratorType.COUPLED_MPM:
                    self.viewer.log_points(
                        "sand",
                        points=self.integrator.mpm_state_0.particle_q,
                        radii=self.integrator.mpm_solver.mpm_model.model.particle_radius,
                        colors=self.integrator.particle_render_colors,
                        hidden=False,
                    )
                    impulses, pos, cid = self.integrator.mpm_solver.collect_collider_impulses(
                        self.integrator.mpm_state_0
                    )
                    self.viewer.log_lines(
                        "impulses",
                        starts=pos,
                        ends=pos + impulses,
                        colors=wp.full(pos.shape[0], value=wp.vec3(1.0, 0.0, 0.0), dtype=wp.vec3),
                    )
                self.viewer.end_frame()

                # Save frame as PNG if render_folder is specified
                if self.render_folder:
                    self._save_frame_as_png()

                # Increment frame counter after rendering
                self.current_frame += 1

    def _save_frame_as_png(self):
        """Save the current viewer frame as a PNG file."""
        try:
            # Capture frame from viewer
            self.viewer.get_frame(target_image=self.frame_buffer, render_ui=False)
            frame_np = self.frame_buffer.numpy()

            # Generate filename with zero-padded frame number
            filename = self.render_folder_path / f"frame_{self.current_frame:04d}.png"

            # Save using PIL or imageio
            if self.use_pil:
                img = self.image_module.fromarray(frame_np, mode="RGB")
                img.save(filename)
            else:
                self.image_module.imwrite(filename, frame_np)

        except Exception as e:
            print(f"Warning: Failed to save frame {self.current_frame}: {e}")

    def save(self):
        if self.usd_updater is not None:
            self.usd_updater.close()
        self.viewer.close()


def print_time_profiler(simulator):
    frame_times = simulator.profiler.get("step", None)
    render_times = simulator.profiler.get("render", None)
    if frame_times:
        print(f"\nAverage frame sim time: {sum(frame_times) / len(frame_times):.2f} ms")
    if render_times:
        print(f"\nAverage frame render time: {sum(render_times) / len(render_times):.2f} ms")


def render_mouse_state(x, y, button, width, height):
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    colors = [
        (255, 255, 255),
        (255, 0, 0),
        (0, 255, 255),
        (0, 255, 0),
        (0, 0, 255),
    ]
    px, py = int(x * width), int(y * height)
    y_start = max(py - 1, 0)
    y_end = min(py + 1, height - 1) + 1  # +1 because slice end is exclusive
    x_start = max(px - 1, 0)
    x_end = min(px + 1, width - 1) + 1
    if button.any():
        color = colors[button.argmax()]
    else:
        color = (255, 255, 255)
    rgb[y_start:y_end, x_start:x_end] = color
    return rgb


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
        default="",
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
    parser.add_argument(
        "-t",
        "--sim_time",
        help="Simulation time",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "-s",
        "--sim_frame",
        help="Simulation frame",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-p",
        "--pause",
        help="Pause droid walking sequence at the given frame number",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-r",
        "--record",
        help="File path to where to record the interaction (needs to be a .bin file)",
        type=str,
        default="",
    )
    parser.add_argument(
        "-l",
        "--load",
        help="File path of a recording (.bin) file to load for playback",
        type=str,
        default="",
    )
    parser.add_argument(
        "-f",
        "--render_folder",
        help="Folder path to where to store rendered PNG frames",
        type=str,
        default="",
    )
    parser.add_argument(
        "--usd_offset",
        help="USD offset as three floats: x y z (e.g., '0.0 0.0 0.0')",
        type=str,
        default="0.0 0.0 0.0",
    )
    parser.add_argument(
        "--use_unified_collision_pipeline",
        help="Use the unified collision pipeline",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--use_coacd",
        help="Use COACD for collision mesh approximation",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--enable_timers",
        help="Enable timers",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--load_visual_shapes",
        help="Load visual shapes",
        type=bool,
        default=False,
    )

    args = parser.parse_known_args()[0]

    # Parse usd_offset argument
    try:
        offset_values = [float(x) for x in args.usd_offset.split()]
        if len(offset_values) != 3:
            raise ValueError("usd_offset must have exactly 3 values")
        usd_offset = wp.vec3(*offset_values)
    except (ValueError, AttributeError) as e:
        print(f"Error parsing usd_offset: {e}. Using default (0.0, 0.0, 0.0)")
        usd_offset = wp.vec3(0.0, 0.0, 0.0)

    if not args.output:
        from pathlib import Path

        path = Path(args.stage_path)
        base_path = path.parent / "output"
        base_path.mkdir(parents=True, exist_ok=True)
        args.output = str(base_path / path.name)
        print(f'Output path not specified (-o flag). Writing to "{args.output}".')

    with wp.ScopedDevice(args.device):
        simulator = Simulator(
            input_path=args.stage_path,
            output_path=args.output,
            integrator=args.integrator,
            num_frames=args.num_frames,
            sim_time=args.sim_time,
            sim_frame=args.sim_frame,
            record_path=args.record,
            render_folder=args.render_folder,
            usd_offset=usd_offset,
            use_unified_collision_pipeline=args.use_unified_collision_pipeline,
            use_coacd=args.use_coacd,
        )

        i = 0
        while i < args.num_frames:
            print(f"frame {i}")
            simulator.step()
            simulator.render()
            i += 1

        print_time_profiler(simulator)

        simulator.save()

