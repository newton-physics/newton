# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mujoco.constants import MJ_MINVAL


class _MuJoCoSolverPropertiesFixture:
    """Multi-world scene shared by the MuJoCo solver property tests."""

    def setUp(self):
        """Set up a model with multiple worlds, each with a free body and an articulated tree."""
        self.seed = 123
        self.rng = np.random.default_rng(self.seed)

        world_count = 2
        self.debug_stage_path = "newton/tests/test_mujoco_render.usda"

        template_builder = newton.ModelBuilder()
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)  # Define ShapeConfig

        # --- Free-floating body (e.g., a box) ---
        free_body_initial_pos = wp.transform((0.5, 0.5, 0.0), wp.quat_identity())
        free_body_idx = template_builder.add_body(mass=0.2, xform=free_body_initial_pos)
        template_builder.add_shape_box(
            body=free_body_idx,
            xform=wp.transform(),  # Shape at body's local origin
            hx=0.1,
            hy=0.1,
            hz=0.1,
            cfg=shape_cfg,
        )

        # --- Articulated tree (3 bodies) ---
        link_radius = 0.05
        link_half_length = 0.15
        tree_root_initial_pos_y = link_half_length * 2.0
        tree_root_initial_transform = wp.transform((0.0, tree_root_initial_pos_y, 0.0), wp.quat_identity())

        body1_idx = template_builder.add_link(mass=0.1)
        template_builder.add_shape_capsule(
            body=body1_idx,
            xform=wp.transform(),  # Shape at body's local origin
            radius=link_radius,
            half_height=link_half_length,
            cfg=shape_cfg,
        )
        joint1 = template_builder.add_joint_free(child=body1_idx, parent_xform=tree_root_initial_transform)

        body2_idx = template_builder.add_link(mass=0.1)
        template_builder.add_shape_capsule(
            body=body2_idx,
            xform=wp.transform(),  # Shape at body's local origin
            radius=link_radius,
            half_height=link_half_length,
            cfg=shape_cfg,
        )
        joint2 = template_builder.add_joint_revolute(
            parent=body1_idx,
            child=body2_idx,
            parent_xform=wp.transform((0.0, link_half_length, 0.0), wp.quat_identity()),
            child_xform=wp.transform((0.0, -link_half_length, 0.0), wp.quat_identity()),
            axis=(0.0, 0.0, 1.0),
        )

        body3_idx = template_builder.add_link(mass=0.1)
        template_builder.add_shape_capsule(
            body=body3_idx,
            xform=wp.transform(),  # Shape at body's local origin
            radius=link_radius,
            half_height=link_half_length,
            cfg=shape_cfg,
        )
        joint3 = template_builder.add_joint_revolute(
            parent=body2_idx,
            child=body3_idx,
            parent_xform=wp.transform((0.0, link_half_length, 0.0), wp.quat_identity()),
            child_xform=wp.transform((0.0, -link_half_length, 0.0), wp.quat_identity()),
            axis=(1.0, 0.0, 0.0),
        )

        template_builder.add_articulation([joint1, joint2, joint3])

        self.builder = newton.ModelBuilder()
        self.builder.add_shape_plane()

        for i in range(world_count):
            world_transform = wp.transform((i * 2.0, 0.0, 0.0), wp.quat_identity())
            self.builder.add_world(template_builder, xform=world_transform)

        try:
            if self.builder.world_count == 0 and world_count > 0:
                self.builder.world_count = world_count
            self.model = self.builder.finalize()
            if self.model.world_count != world_count:
                print(
                    f"Warning: Model.world_count ({self.model.world_count}) does not match expected world_count ({world_count})."
                )
        except Exception as e:
            self.fail(f"Model finalization failed: {e}")

        self.state_in = self.model.state()
        self.state_out = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.collision_pipeline.collide(self.state_in, self.contacts)


def _add_deprecated_mimic_constraint(
    builder: newton.ModelBuilder,
    *,
    joint0: int,
    joint1: int,
    coef0: float = 0.0,
    coef1: float = 1.0,
    enabled: bool = True,
    label: str | None = None,
) -> int:
    """Add a sparse mimic constraint while acknowledging its deprecation."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return builder.add_constraint_mimic(
            joint0=joint0,
            joint1=joint1,
            coef0=coef0,
            coef1=coef1,
            enabled=enabled,
            label=label,
        )


def _expected_positive_limit_solref(ke: float, kd: float, factor: float) -> np.ndarray:
    direct_stiffness = max(float(ke) * float(factor), MJ_MINVAL)
    direct_damping = max(float(kd) * float(factor), MJ_MINVAL)
    return np.array(
        [2.0 / direct_damping, direct_damping / (2.0 * math.sqrt(direct_stiffness))],
        dtype=np.float64,
    )


def _create_actuator_test_stage(extra_joint_attrs=None, extra_actuator_attrs=None):
    """Create a minimal USD stage with one revolute joint and one MjcActuator.

    Returns the stage. Caller can add additional attributes before building.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.Scene.Define(stage, "/physicsScene")

    base = UsdGeom.Cube.Define(stage, "/World/base")
    base_prim = base.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(base_prim)
    UsdPhysics.ArticulationRootAPI.Apply(base_prim)
    UsdPhysics.CollisionAPI.Apply(base_prim)

    link = UsdGeom.Cube.Define(stage, "/World/link")
    link_prim = link.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(link_prim)
    UsdPhysics.CollisionAPI.Apply(link_prim)

    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/joint")
    joint.CreateAxisAttr().Set("Z")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/World/base")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/link")])

    if extra_joint_attrs:
        extra_joint_attrs(joint)

    act = stage.DefinePrim("/World/actuator", "MjcActuator")
    act.CreateRelationship("mjc:target", True).SetTargets([Sdf.Path("/World/joint")])

    if extra_actuator_attrs:
        extra_actuator_attrs(act)

    return stage


def _large_nv_mjcf(cone: str) -> str:
    """Build an MJCF whose free joints put it over MuJoCo Warp's nv > 500 threshold."""
    bodies = "".join(
        f'<body name="b{i}" pos="{i * 0.5} 0 1"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>'
        for i in range(100)
    )
    return f'<mujoco><option cone="{cone}"/><worldbody>{bodies}</worldbody></mujoco>'
