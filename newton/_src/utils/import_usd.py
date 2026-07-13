# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import collections
import copy
import datetime
import hashlib
import inspect
import itertools
import logging
import math
import os
import posixpath
import re
import warnings
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urljoin, urlparse

if TYPE_CHECKING:
    from pxr import Usd

    from ..geometry.types import TetMesh

    UsdStage = Usd.Stage
else:
    UsdStage = Any

import numpy as np
import warp as wp

from ..core import quat_between_axes
from ..core.types import Axis, Transform
from ..geometry import GeoType, Mesh, ShapeFlags, compute_inertia_shape, compute_inertia_sphere
from ..sim.builder import ModelBuilder
from ..sim.enums import JointTargetMode
from ..sim.model import Model
from ..solvers.mujoco.constants import SOLREF_MODE_FORCE_SPACE, SOLREF_MODE_MJCF_DEFAULT, SOLREF_MODE_RAW
from ..solvers.mujoco.enums import EqType
from ..solvers.mujoco.equality import _add_equality_constraint
from ..solvers.mujoco.utils import (
    mjc_add_equality_loop_joint,
    mjc_add_equality_mimic,
    mjc_polycoef_has_higher_order,
)
from ..usd import require_newton_usd_schemas
from ..usd import utils as usd
from ..usd.schema_resolver import PrimType, SchemaResolver, SchemaResolverManager
from ..usd.schemas import SchemaResolverNewton
from .import_usd_deformable_attachments import (
    _deformable_import_attachments,
    _deformable_import_element_collision_filters,
    _deformable_remap_collapsed,
)
from .import_usd_deformable_cable import _deformable_import_cable, _deformable_import_cable_graphs
from .import_usd_deformable_cloth import _deformable_import_cloth
from .import_usd_deformable_utils import _DeformableImportContext, _scout_deformable_prims
from .import_usd_deformable_volume import _deformable_import_volume
from .import_utils import should_show_collider

logger = logging.getLogger("newton")

AttributeFrequency = Model.AttributeFrequency

_NEWTON_SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir)) + os.sep

# Stiffness used for a hard joint limit (NewtonJointAPI newton:limitStiffness == +inf).
_HARD_LIMIT_KE = 1.0e8


def _resolve_newton_limit_ke(
    limit_ke: float | None,
    fallback: float,
    fallback_source: str,
    builder_default: float,
) -> tuple[float, str]:
    """Resolve a NewtonJointAPI ``newton:limitStiffness`` value.

    ``limit_ke`` is ``None`` when the attribute is not authored, ``-inf`` when
    authored as the engine-default sentinel, ``+inf`` for a hard limit, or a
    finite stiffness value.

    ``fallback`` is the per-DOF stiffness resolved from lower-priority schemas
    (PhysX/MuJoCo).  ``builder_default`` is the ModelBuilder engine default.

    An explicit ``-inf`` takes precedence over the per-DOF fallback and selects
    the builder default so that a lower-priority schema cannot override an
    authored Newton sentinel.

    Returns (resolved_value, source) where source is ``"force"`` when Newton
    broadcast values are used, or the original ``fallback_source`` otherwise.
    """
    if limit_ke is None:
        return fallback, fallback_source
    if limit_ke == float("-inf"):
        return builder_default, "force"
    if limit_ke == float("inf"):
        return _HARD_LIMIT_KE, "force"
    return limit_ke, "force"


def _resolve_newton_limit_kd(
    limit_ke: float | None,
    limit_kd: float | None,
    fallback: float,
    fallback_source: str,
    builder_default: float,
) -> tuple[float, str]:
    """Resolve a NewtonJointAPI ``newton:limitDamping`` value.

    Hard limits (``limit_ke`` or ``limit_kd`` == ``+inf``) have no damping.
    An authored ``-inf`` selects the builder default (engine default), taking
    precedence over per-DOF fallbacks from lower-priority schemas.
    When neither Newton attribute is authored (``None``), the per-DOF ``fallback``
    from other resolvers is used.

    Returns (resolved_value, source) where source is ``"force"`` when Newton
    broadcast values are used, or the original ``fallback_source`` otherwise.
    """
    # Hard (rigid) limit: infinite ke or kd means no dissipation is needed.
    if limit_ke is not None and limit_ke == float("inf"):
        return 0.0, "force"
    if limit_kd is not None and limit_kd == float("inf"):
        return 0.0, "force"
    # Not authored → lower-priority per-DOF fallback.
    if limit_kd is None:
        return fallback, fallback_source
    # Authored -inf → builder default.
    if limit_kd == float("-inf"):
        return builder_default, "force"
    return limit_kd, "force"


def _validate_https_usd_url(url: str) -> None:
    """Reject non-HTTPS URLs before USD asset downloads."""
    if urlparse(url).scheme != "https":
        raise ValueError(f"USD URL downloads require HTTPS: {url}")


def _cache_path_for_absolute_usd_reference(url: str) -> str:
    """Return a safe cache-relative path for an absolute USD reference URL."""
    parsed = urlparse(url)
    basename = posixpath.basename(parsed.path) or "reference.usd"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return posixpath.join("_external_usd", digest, basename)


def _external_stacklevel() -> int:
    """Return a ``stacklevel`` that points past all ``newton._src`` frames."""
    frame = inspect.currentframe()
    if frame is None:
        return 2
    frame = frame.f_back
    stacklevel = 1
    try:
        while frame is not None and os.path.normpath(frame.f_code.co_filename).startswith(_NEWTON_SRC_DIR):
            frame = frame.f_back
            stacklevel += 1
        return stacklevel
    finally:
        del frame


def parse_usd(
    builder: ModelBuilder,
    source: str | UsdStage,
    *,
    xform: Transform | None = None,
    floating: bool | None = None,
    base_joint: dict | None = None,
    parent_body: int = -1,
    only_load_enabled_rigid_bodies: bool = False,
    only_load_enabled_joints: bool = True,
    joint_drive_gains_scaling: float = 1.0,
    verbose: bool = False,
    ignore_paths: list[str] | None = None,
    collapse_fixed_joints: bool = False,
    enable_self_collisions: bool = True,
    apply_up_axis_from_stage: bool = False,
    root_path: str = "/",
    joint_ordering: Literal["bfs", "dfs"] | None = "dfs",
    bodies_follow_joint_ordering: bool = True,
    skip_mesh_approximation: bool = False,
    load_sites: bool = True,
    load_visual_shapes: bool = True,
    hide_collision_shapes: bool = False,
    force_show_colliders: bool = False,
    parse_mujoco_options: bool = True,
    mesh_maxhullvert: int | None = None,
    schema_resolvers: list[SchemaResolver] | None = None,
    force_position_velocity_actuation: bool = False,
    convert_mjc_equality_constraints: bool = True,
    override_root_xform: bool = False,
    legacy_margin_gap: bool = False,
    return_deformable_results: bool = False,
) -> dict[str, Any]:
    """Parses a Universal Scene Description (USD) stage and adds rigid bodies, soft bodies, shapes, and joints to the given ModelBuilder.

    The USD description has to be either a path (file name or URL), or an existing USD stage instance that implements the `Stage <https://openusd.org/dev/api/class_usd_stage.html>`_ interface.

    See :ref:`usd_parsing` for more information.

    Args:
        builder: The :class:`ModelBuilder` to add the bodies and joints to.
        source: The file path to the USD file, or an existing USD stage instance.
        xform: The transform to apply to the entire scene.
        override_root_xform: If ``True``, the articulation root's world-space
            transform is replaced by ``xform`` instead of being composed with it,
            preserving only the internal structure (relative body positions). Useful
            for cloning articulations at explicit positions. Not intended for sources
            containing multiple articulations, as all roots would be placed at the
            same ``xform``. Defaults to ``False``.
        floating: Controls the base joint type for the root body (bodies not connected as
            a child to any joint).

            - ``None`` (default): Uses format-specific default (creates a FREE joint for USD bodies without joints).
            - ``True``: Creates a FREE joint with 6 DOF (3 translation + 3 rotation). Only valid when
              ``parent_body == -1`` since FREE joints must connect to world frame.
            - ``False``: Creates a FIXED joint (0 DOF).

            Cannot be specified together with ``base_joint``.
        base_joint: Custom joint specification for connecting the root body to the world
            (or to ``parent_body`` if specified). This parameter enables hierarchical composition with
            custom mobility. Dictionary with joint parameters as accepted by
            :meth:`ModelBuilder.add_joint` (e.g., joint type, axes, limits, stiffness).

            Cannot be specified together with ``floating``.
        parent_body: Parent body index for hierarchical composition. If specified, attaches the
            imported root body to this existing body, making them part of the same kinematic articulation.
            The connection type is determined by ``floating`` or ``base_joint``. If ``-1`` (default),
            the root connects to the world frame. **Restriction**: Only the most recently added
            articulation can be used as parent; attempting to attach to an older articulation will raise
            a ``ValueError``.

            .. note::
               Valid combinations of ``floating``, ``base_joint``, and ``parent_body``:

               .. list-table::
                  :header-rows: 1
                  :widths: 15 15 15 55

                  * - floating
                    - base_joint
                    - parent_body
                    - Result
                  * - ``None``
                    - ``None``
                    - ``-1``
                    - Format default (USD: FREE joint for bodies without joints)
                  * - ``True``
                    - ``None``
                    - ``-1``
                    - FREE joint to world (6 DOF)
                  * - ``False``
                    - ``None``
                    - ``-1``
                    - FIXED joint to world (0 DOF)
                  * - ``None``
                    - ``{dict}``
                    - ``-1``
                    - Custom joint to world (e.g., D6)
                  * - ``False``
                    - ``None``
                    - ``body_idx``
                    - FIXED joint to parent body
                  * - ``None``
                    - ``{dict}``
                    - ``body_idx``
                    - Custom joint to parent body (e.g., D6)
                  * - *explicitly set*
                    - *explicitly set*
                    - *any*
                    - ❌ Error: mutually exclusive (cannot specify both)
                  * - ``True``
                    - ``None``
                    - ``body_idx``
                    - ❌ Error: FREE joints require world frame

        only_load_enabled_rigid_bodies: If True, only rigid bodies which do not have `physics:rigidBodyEnabled` set to False are loaded.
        only_load_enabled_joints: If True, only joints which do not have `physics:jointEnabled` set to False are loaded.
        joint_drive_gains_scaling: The default scaling of the PD control gains (stiffness and damping), if not set in the PhysicsScene with as "newton:joint_drive_gains_scaling".
        verbose: If True, print additional information about the parsed USD file. Default is False.
        ignore_paths: A list of regular expressions matching prim paths to ignore.
        collapse_fixed_joints: If True, fixed joints are removed and the respective bodies are merged. Only considered if not set on the PhysicsScene as "newton:collapse_fixed_joints".
        enable_self_collisions: Default for whether self-collisions are enabled for all shapes within an articulation. Resolved via the schema resolver from ``newton:selfCollisionEnabled`` (NewtonArticulationRootAPI) or ``physxArticulation:enabledSelfCollisions``; if neither is authored, this value takes precedence.
        apply_up_axis_from_stage: If True, the up axis of the stage will be used to set :attr:`newton.ModelBuilder.up_axis`. Otherwise, the stage will be rotated such that its up axis aligns with the builder's up axis. Default is False.
        root_path: The USD path to import, defaults to "/".
        joint_ordering: The ordering of the joints in the simulation. Can be either "bfs" or "dfs" for breadth-first or depth-first search, or ``None`` to keep joints in the order in which they appear in the USD. Default is "dfs".
        bodies_follow_joint_ordering: If True, the bodies are added to the builder in the same order as the joints (parent then child body). Otherwise, bodies are added in the order they appear in the USD. Default is True.
        skip_mesh_approximation: If True, mesh approximation is skipped. Otherwise, meshes are approximated according to the ``physics:approximation`` attribute defined on the UsdPhysicsMeshCollisionAPI (if it is defined). Default is False.
        load_sites: If True, sites (prims with ``NewtonSiteAPI`` or ``MjcSiteAPI``) are loaded as non-colliding reference points. If False, sites are ignored. Default is True.
        load_visual_shapes: If True, non-physics visual geometry is loaded. If False, visual-only shapes are ignored (sites are still controlled by ``load_sites``). Default is True.
        hide_collision_shapes: If True, collision shapes on bodies that already
            have visual-only geometry are hidden unconditionally, regardless of
            whether the collider has authored PBR material data. Collision
            shapes on bodies without visual-only geometry remain visible as a
            rendering fallback. Default is False.
        force_show_colliders: If True, collision shapes get the VISIBLE flag
            regardless of whether visual shapes exist on the same body. Note that
            ``hide_collision_shapes=True`` still suppresses the VISIBLE flag for
            colliders on bodies with visual-only geometry. Default is False.
        parse_mujoco_options: Whether MuJoCo solver options from the PhysicsScene should be parsed. If False, solver options are not loaded and custom attributes retain their default values. Default is True.
        convert_mjc_equality_constraints: Whether MuJoCo equality schemas should be converted to Newton loop
            joints or mimic constraints while preserving MuJoCo equality metadata for SolverMuJoCo. If False,
            equality constraints are preserved in the ``mujoco:equality_constraint`` custom-attribute namespace
            and finalize under ``model.mujoco.equality_constraint_*``.
        mesh_maxhullvert: Maximum vertices for convex hull approximation of meshes. Note that an authored ``newton:maxHullVertices`` attribute on any shape with a ``NewtonMeshCollisionAPI`` will take priority over this value.
        schema_resolvers: Resolver instances in priority order. Default is to only parse Newton-specific attributes.
            Schema resolvers collect per-prim "solver-specific" attributes, see :ref:`schema_resolvers` for more information.
            These include namespaced attributes such as ``newton:*``, ``physx*``
            (e.g., ``physxScene:*``, ``physxRigidBody:*``, ``physxSDFMeshCollision:*``), and ``mjc:*`` that
            are authored in the USD but not strictly required to build the simulation. This is useful for
            inspection, experimentation, or custom pipelines that read these values via
            ``result["schema_attrs"]`` returned from ``parse_usd()``.

            .. experimental::

                The ``schema_resolvers`` argument may change without prior notice.
        force_position_velocity_actuation: If True and both stiffness (kp) and damping (kd)
            are non-zero, joints use :attr:`~newton.JointTargetMode.POSITION_VELOCITY` actuation mode.
            If False (default), actuator modes are inferred per joint via :func:`newton.JointTargetMode.from_gains`:
            :attr:`~newton.JointTargetMode.POSITION` if stiffness > 0, :attr:`~newton.JointTargetMode.VELOCITY` if only
            damping > 0, :attr:`~newton.JointTargetMode.EFFORT` if a drive is present but both gains are zero
            (direct torque control), or :attr:`~newton.JointTargetMode.NONE` if no drive/actuation is applied.
        legacy_margin_gap: If True, restore pre-MuJoCo-3.9 import behavior
            where ``shape_margin`` is computed as ``mjc_margin - mjc_gap``.
            Use for USD files authored against MuJoCo <= 3.8. Defaults to
            False (identity translation matching MuJoCo 3.9 semantics).

        return_deformable_results: If True, include the experimental deformable entries in the
            returned mapping (``path_cable_map`` / ``path_cloth_map`` / ``path_soft_map`` /
            ``path_attachment_map`` and the matching ``path_*_attrs``). Off by default, so the
            default return shape carries no deformable additions.

    Returns:
        .. experimental::

           ``return_deformable_results`` and its conditional result entries are experimental and
           may change or be removed without prior notice.

        When ``return_deformable_results=True``, imported deformable (cable/cloth/volume) element
        ranges are returned by prim path in the ``path_cable_map`` / ``path_cloth_map`` /
        ``path_soft_map`` entries below, and the material attributes as authored in the
        matching ``path_*_attrs`` entries. The map entries are build-time snapshots of the
        builder immediately after this call (already remapped when this call collapses fixed
        joints); they are not live selections, and a later ``replicate()``, ``add_builder()``,
        or other structural mutation is outside their contract. The ``path_*_attrs`` entries
        hold authored or resolved source values (``material`` as authored,
        ``resolved_density`` as used), while the map entries and ``joint_indices`` inside
        ``path_attachment_attrs`` are realized builder indices; ``unsupported_reason`` is
        diagnostic text, not a stable code, and a prim absent from a realized map may still
        appear in the authored metadata.

        The returned mapping has the following entries:

        .. list-table::
            :widths: 25 75

            * - ``"fps"``
              - USD stage frames per second
            * - ``"duration"``
              - Difference between end time code and start time code of the USD stage
            * - ``"up_axis"``
              - :class:`Axis` representing the stage's up axis ("X", "Y", or "Z")
            * - ``"path_body_map"``
              - Mapping from prim path (str) of a rigid body prim (e.g. that implements the PhysicsRigidBodyAPI) to the respective body index in :class:`~newton.ModelBuilder`
            * - ``"path_joint_map"``
              - Mapping from prim path (str) of a joint prim (e.g. that implements the PhysicsJointAPI) to the respective joint index in :class:`~newton.ModelBuilder`
            * - ``"path_shape_map"``
              - Mapping from prim path (str) of the UsdGeom to the respective shape index in :class:`~newton.ModelBuilder`
            * - ``"path_shape_scale"``
              - Mapping from prim path (str) of the UsdGeom to its respective 3D world scale
            * - ``"path_cable_map"``
              - Mapping from prim path (str) of a curve deformable (cable) to its ``(body_indices, joint_indices)`` lists. Curves welded into a rod graph report empty joints (the joints belong to the shared graph articulation). Present only with ``return_deformable_results=True``.
            * - ``"path_cloth_map"``
              - Mapping from prim path (str) of a surface deformable (cloth) to its ``[start, end)`` index ranges, keyed ``"particle"`` / ``"tri"`` / ``"edge"``. Present only with ``return_deformable_results=True``.
            * - ``"path_soft_map"``
              - Mapping from prim path (str) of a soft body (a volume deformable, or a legacy bare TetMesh) to its ``[start, end)`` index ranges, keyed ``"particle"`` / ``"tet"``. Present only with ``return_deformable_results=True``.
            * - ``"path_cable_attrs"``
              - Mapping from prim path (str) of a curve deformable (cable) to its as-authored, solver-neutral attributes (``material`` moduli, ``resolved_density``, ``closed``); includes moduli the imported rod cannot express (e.g. shear / twist). ``graph_component`` is present only for curves successfully welded into the same rod graph; curves in one graph share the component identifier. Present only with ``return_deformable_results=True``.
            * - ``"path_cloth_attrs"``
              - Mapping from prim path (str) of a surface deformable (cloth) to its as-authored, solver-neutral attributes (``material`` moduli, ``resolved_density``). Present only with ``return_deformable_results=True``.
            * - ``"path_soft_attrs"``
              - Mapping from prim path (str) of a soft body (a volume deformable, or a legacy bare TetMesh) to its as-authored, solver-neutral attributes (``resolved_density``). Present only with ``return_deformable_results=True``.
            * - ``"path_attachment_map"``
              - Mapping from prim path (str) of a supported ``PhysicsAttachment`` prim to the created joint indices. Curve-to-curve ``point``->``point`` junctions are consumed as rod-graph topology and are absent from this mapping. Present only with ``return_deformable_results=True``.
            * - ``"path_attachment_attrs"``
              - Mapping from prim path (str) of a ``PhysicsAttachment`` prim to its parsed, solver-neutral attributes and any unsupported reason. Junctions consumed as rod-graph topology are absent here as well. Present only with ``return_deformable_results=True``.
            * - ``"mass_unit"``
              - The stage's Kilograms Per Unit (KGPU) definition (1.0 by default)
            * - ``"linear_unit"``
              - The stage's Meters Per Unit (MPU) definition (1.0 by default)
            * - ``"scene_attributes"``
              - Dictionary of all attributes applied to the PhysicsScene prim
            * - ``"collapse_results"``
              - Dictionary returned by :meth:`newton.ModelBuilder.collapse_fixed_joints` if ``collapse_fixed_joints`` is True, otherwise None.
            * - ``"physics_dt"``
              - The resolved physics scene time step (float or None)
            * - ``"schema_attrs"``
              - Dictionary of collected per-prim schema attributes (dict)
            * - ``"max_solver_iterations"``
              - The resolved maximum solver iterations (int or None)
            * - ``"path_body_relative_transform"``
              - Mapping from prim path to relative transform for bodies merged via ``collapse_fixed_joints``
            * - ``"path_original_body_map"``
              - Mapping from prim path to original body index before ``collapse_fixed_joints``
            * - ``"actuator_count"``
              - Number of external actuators parsed from the USD stage
    """
    # Early validation of base joint parameters
    builder._validate_base_joint_params(floating, base_joint, parent_body)

    if mesh_maxhullvert is None:
        mesh_maxhullvert = Mesh.MAX_HULL_VERTICES

    if schema_resolvers is None:
        schema_resolvers = [SchemaResolverNewton()]
    collect_schema_attrs = len(schema_resolvers) > 0

    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    except ImportError as e:
        raise ImportError("Failed to import pxr. Please install USD (e.g. via `pip install usd-core`).") from e
    require_newton_usd_schemas(Usd)

    from .topology import topological_sort_undirected  # noqa: PLC0415

    @dataclass
    class PhysicsMaterial:
        staticFriction: float = builder.default_shape_cfg.mu
        dynamicFriction: float = builder.default_shape_cfg.mu
        torsionalFriction: float = builder.default_shape_cfg.mu_torsional
        rollingFriction: float = builder.default_shape_cfg.mu_rolling
        restitution: float = builder.default_shape_cfg.restitution
        density: float = builder.default_shape_cfg.density
        ke: float | None = None
        kd: float | None = None
        kf: float | None = None
        ka: float | None = None

    # load joint defaults
    default_joint_friction = builder.default_joint_cfg.friction
    default_joint_damping = builder.default_joint_cfg.damping
    default_joint_limit_ke = builder.default_joint_cfg.limit_ke
    default_joint_limit_kd = builder.default_joint_cfg.limit_kd
    default_joint_armature = builder.default_joint_cfg.armature
    default_joint_velocity_limit = builder.default_joint_cfg.velocity_limit

    # load shape defaults
    default_shape_density = builder.default_shape_cfg.density

    # mapping from physics:approximation attribute (lower case) to remeshing method
    approximation_to_remeshing_method = {
        "convexdecomposition": "coacd",
        "convexhull": "convex_hull",
        "boundingsphere": "bounding_sphere",
        "boundingcube": "bounding_box",
        "meshsimplification": "quadratic",
    }
    # mapping from remeshing method to a list of shape indices
    remeshing_queue = {}

    if ignore_paths is None:
        ignore_paths = []

    usd_axis_to_axis = {
        UsdPhysics.Axis.X: Axis.X,
        UsdPhysics.Axis.Y: Axis.Y,
        UsdPhysics.Axis.Z: Axis.Z,
    }

    if isinstance(source, str):
        stage = Usd.Stage.Open(source, Usd.Stage.LoadAll)
        _raise_on_stage_errors(stage, source)
    else:
        stage = source
        _raise_on_stage_errors(stage, "provided stage")

    DegreesToRadian = float(np.pi / 180)
    mass_unit = 1.0

    try:
        if UsdPhysics.StageHasAuthoredKilogramsPerUnit(stage):
            mass_unit = UsdPhysics.GetStageKilogramsPerUnit(stage)
    except Exception as e:
        if verbose:
            print(f"Failed to get mass unit: {e}")
    if not math.isclose(mass_unit, 1.0):
        warnings.warn(
            "USD stages with non-unit mass units are not supported. "
            f"Set kilogramsPerUnit to 1.0 before import. Found kilogramsPerUnit={mass_unit}.",
            stacklevel=_external_stacklevel(),
        )
    linear_unit = 1.0
    try:
        if UsdGeom.StageHasAuthoredMetersPerUnit(stage):
            linear_unit = UsdGeom.GetStageMetersPerUnit(stage)
    except Exception as e:
        if verbose:
            print(f"Failed to get linear unit: {e}")
    if not math.isclose(linear_unit, 1.0):
        warnings.warn(
            "USD stages with non-unit linear units are not supported. "
            f"Set metersPerUnit to 1.0 before import. Found metersPerUnit={linear_unit}.",
            stacklevel=_external_stacklevel(),
        )

    non_regex_ignore_paths = [path for path in ignore_paths if ".*" not in path]
    # One scouting walk classifies every deformable candidate prim; it runs before the
    # native loader so deformable-owned geometry (simulation prims and their colliders)
    # can be excluded from rigid parsing, and the buckets are reused by the deformable
    # passes below instead of re-traversing the stage.
    root_prim = stage.GetPrimAtPath(root_path)
    _deformable_prims = _scout_deformable_prims(root_prim, ignore_paths)
    native_exclude_paths = list(
        dict.fromkeys([*non_regex_ignore_paths, *_deformable_prims.native_physics_exclude_paths])
    )
    ret_dict = UsdPhysics.LoadUsdPhysicsFromRange(stage, [root_path], excludePaths=native_exclude_paths)

    # Initialize schema resolver according to precedence
    R = SchemaResolverManager(schema_resolvers)

    # Vendor namespaces (e.g. omniphysics, physxDeformableBody) accepted as a
    # fallback to the canonical physics: deformable schema. Empty unless a
    # resolver declaring them (e.g. SchemaResolverPhysx) is active, so a default
    # import parses the AOUSD proposal as written.
    deformable_compat_ns = R.deformable_compat_namespaces()
    # Resolver-owned deformable read (physics: first, then opted-in vendor namespaces).
    deformable_read = R.read_deformable_attr

    # Validate solver-specific custom attributes are registered
    for resolver in schema_resolvers:
        resolver.validate_custom_attributes(builder)
    mjc_resolver = next((resolver for resolver in schema_resolvers if resolver.name == "mjc"), None)
    solreflimit_mode_key = "mujoco:solreflimit_mode"

    # mapping from prim path to body index in ModelBuilder
    path_body_map: dict[str, int] = {}
    # mapping from prim path to shape index in ModelBuilder
    path_shape_map: dict[str, int] = {}
    path_shape_scale: dict[str, wp.vec3] = {}
    # mapping from prim path to joint index in ModelBuilder
    path_joint_map: dict[str, int] = {}
    # Import-internal deformable index maps (not returned): the attachment and collapse passes
    # look up a curve/cloth/soft prim's element indices by path while building. The equivalent
    # per-group index ranges are recorded on the builder/Model registries for callers.
    path_cable_map: dict[str, tuple[list[int], list[int]]] = {}
    path_cloth_map: dict[str, dict[str, tuple[int, int]]] = {}
    path_soft_map: dict[str, dict[str, tuple[int, int]]] = {}
    # Solver-neutral deformable attributes per prim path: the parsed material moduli
    # (including ones the VBD build ignores) and the resolved density, so a non-VBD
    # consumer can rebuild the deformable without re-parsing the stage.
    path_cable_attrs: dict[str, dict[str, Any]] = {}
    path_cloth_attrs: dict[str, dict[str, Any]] = {}
    path_soft_attrs: dict[str, dict[str, Any]] = {}
    path_attachment_map: dict[str, list[int]] = {}
    # Attachment attributes are preserved even when the current builder cannot lower
    # the attachment faithfully (e.g. cloth/volume feature attachments).
    path_attachment_attrs: dict[str, dict[str, Any]] = {}
    # Internal cable maps used by the PhysicsAttachment post-pass. Proposal
    # point/segment indices are flattened across each BasisCurves prim in curve order.
    path_cable_point_anchors: dict[str, dict[int, list[tuple[int, wp.vec3]]]] = {}
    path_cable_segments: dict[str, dict[int, tuple[int, float]]] = {}
    # DOF offset within a merged D6 joint for each original prim path (only populated for merged joints)
    merged_dof_offset: dict[str, int] = {}
    # cache for resolved material properties (keyed by prim path)
    material_props_cache: dict[str, dict[str, Any]] = {}
    # cache for mesh data loaded from USD prims
    mesh_cache: dict[tuple[str, bool, bool], Mesh] = {}
    # cache for TetMesh data loaded from USD prims
    tetmesh_cache: dict[str, TetMesh] = {}

    physics_scene_prim = None
    physics_dt = None
    max_solver_iters = None

    visual_shape_cfg = ModelBuilder.ShapeConfig(
        density=0.0,
        has_shape_collision=False,
        has_particle_collision=False,
    )

    # Create a cache for world transforms to avoid recomputing them for each prim.
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    def _is_enabled_collider(prim: Usd.Prim) -> bool:
        if collider := UsdPhysics.CollisionAPI(prim):
            return collider.GetCollisionEnabledAttr().Get()
        return False

    def _xform_to_mat44(xform: wp.transform) -> wp.mat44:
        return wp.transform_compose(xform.p, xform.q, wp.vec3(1.0))

    def _get_material_props_cached(prim: Usd.Prim) -> dict[str, Any]:
        """Get material properties with caching to avoid repeated traversal."""
        prim_path = str(prim.GetPath())
        if prim_path not in material_props_cache:
            material_props_cache[prim_path] = usd.resolve_material_properties_for_prim(prim)
        return material_props_cache[prim_path]

    def _get_mesh_cached(prim: Usd.Prim, *, load_uvs: bool = False, load_normals: bool = False) -> Mesh:
        """Load and cache mesh data to avoid repeated expensive USD mesh extraction."""
        prim_path = str(prim.GetPath())
        key = (prim_path, load_uvs, load_normals)
        if key in mesh_cache:
            return mesh_cache[key]

        # A mesh loaded with more data is a superset of simpler representations.
        for cached_key in [
            (prim_path, True, True),
            (prim_path, load_uvs, True),
            (prim_path, True, load_normals),
        ]:
            if cached_key != key and cached_key in mesh_cache:
                return mesh_cache[cached_key]

        mesh = usd.get_mesh(prim, load_uvs=load_uvs, load_normals=load_normals)
        mesh_cache[key] = mesh
        return mesh

    def _has_api_schema(prim: Usd.Prim, schema_name: str) -> bool:
        return bool(prim and prim.IsValid() and usd.has_applied_api_schema(prim, schema_name))

    def _should_write_solreflimit_mode() -> bool:
        return mjc_resolver is not None and solreflimit_mode_key in builder.custom_attributes

    # Keep source tracking local until schema applicability and provenance are modeled globally (#3307).
    def _get_mjc_joint_limit_default(prim: Usd.Prim, key: str) -> float | None:
        if mjc_resolver is None or not _has_api_schema(prim, "MjcJointAPI"):
            return None
        spec = mjc_resolver.mapping.get(PrimType.JOINT, {}).get(key)
        if spec is None or spec.default is None:
            return None
        if spec.usd_value_transformer is not None:
            return spec.usd_value_transformer(spec.default)
        return spec.default

    def _resolve_joint_limit_gain(
        prim: Usd.Prim, key: str, builder_default: float
    ) -> tuple[float, Literal["force", "mjc_authored", "mjc_default"]]:
        """Resolve a limit gain and report the semantics of its source."""
        for resolver in R.resolvers:
            spec = resolver.mapping.get(PrimType.JOINT, {}).get(key)
            if spec is None:
                continue

            if resolver.name == "mjc":
                raw_value = usd.get_attribute(prim, spec.name)
                if raw_value is None:
                    continue
                R._collect_on_first_use(resolver, prim)
                authored_value = (
                    spec.usd_value_transformer(raw_value) if spec.usd_value_transformer is not None else raw_value
                )
                if authored_value is not None:
                    return authored_value, "mjc_authored"
                mjc_default = _get_mjc_joint_limit_default(prim, key)
                if mjc_default is not None:
                    return mjc_default, "mjc_authored"
                return builder_default, "mjc_authored"

            authored_value = resolver.get_value(prim, PrimType.JOINT, key)
            if authored_value is not None:
                R._collect_on_first_use(resolver, prim)
                return authored_value, "force"

        if mjc_resolver is not None:
            mjc_default = _get_mjc_joint_limit_default(prim, key)
            if mjc_default is not None:
                return mjc_default, "mjc_default"
        return builder_default, "force"

    def _joint_limit_solref_mode(ke_source: str, kd_source: str) -> int:
        """Choose MuJoCo limit-solref semantics from the resolved gain sources."""
        if ke_source == kd_source == "mjc_authored":
            return SOLREF_MODE_RAW
        if ke_source == kd_source == "mjc_default":
            return SOLREF_MODE_MJCF_DEFAULT
        return SOLREF_MODE_FORCE_SPACE

    def _get_rigid_body_ancestor_path(prim: Usd.Prim) -> str | None:
        current = prim
        while current and current.IsValid():
            current_path = str(current.GetPath())
            if current_path in path_body_map:
                return current_path
            current = current.GetParent()
        return None

    def _is_world_target(target_path: str) -> bool:
        """Return whether the target path represents the world body."""
        if target_path in ("", "/"):
            return True

        default_prim = stage.GetDefaultPrim()
        return bool(
            default_prim
            and default_prim.IsValid()
            and target_path == str(default_prim.GetPath())
            and target_path not in path_body_map
        )

    def _get_target_body_and_local_pos(target_path: str) -> tuple[int, wp.vec3] | None:
        """Resolve a target to its body index and body-local position."""
        if _is_world_target(target_path):
            return (-1, wp.vec3())

        target_prim = stage.GetPrimAtPath(target_path)
        if not target_prim or not target_prim.IsValid():
            return None

        body_path = _get_rigid_body_ancestor_path(target_prim)
        if body_path is None:
            return None

        body_idx = path_body_map.get(body_path, -1)
        if body_idx < 0:
            return None

        if target_path == body_path:
            return (body_idx, wp.vec3())

        body_prim = stage.GetPrimAtPath(body_path)
        body_world = usd.get_transform(body_prim, local=False, xform_cache=xform_cache)
        target_world = usd.get_transform(target_prim, local=False, xform_cache=xform_cache)
        local_tf = wp.transform_inverse(body_world) * target_world
        return (body_idx, local_tf.p)

    def _get_first_target(prim: Usd.Prim, rel_name: str) -> str:
        """Return the first target path of *rel_name* on *prim*, or ``""`` for world."""
        rel = prim.GetRelationship(rel_name)
        targets = rel.GetTargets() if rel else []
        return str(targets[0]) if targets else ""

    def _resolve_equality_bodies(
        joint_prim: Usd.Prim,
        joint_path: str,
        schema_name: str,
    ) -> tuple[tuple[int, wp.vec3] | None, tuple[int, wp.vec3] | None]:
        """Resolve body0 and body1 for a Connect/Weld equality joint prim.

        Returns ``(body0_info, body1_info)`` where each is
        ``(body_index, local_position)`` or ``None`` on failure.
        An empty target list is interpreted as the world body (index -1).
        """
        target0 = _get_first_target(joint_prim, "physics:body0")
        target1 = _get_first_target(joint_prim, "physics:body1")

        if target0 == "" and target1 == "":
            warnings.warn(
                f"{schema_name} on '{joint_path}' has no physics:body0 or physics:body1 targets; skipping.",
                stacklevel=3,
            )
            return None, None

        # Empty target means world body (index -1).
        body0_info = _get_target_body_and_local_pos(target0) if target0 else (-1, wp.vec3())
        body1_info = _get_target_body_and_local_pos(target1) if target1 else (-1, wp.vec3())

        if body0_info is None or body1_info is None:
            failed_targets = []
            if body0_info is None:
                failed_targets.append(f"physics:body0='{target0}'")
            if body1_info is None:
                failed_targets.append(f"physics:body1='{target1}'")
            warnings.warn(
                f"{schema_name} on '{joint_path}' references unresolved body target(s) "
                f"{', '.join(failed_targets)}; skipping.",
                stacklevel=3,
            )
            return None, None

        return body0_info, body1_info

    def _get_mesh_with_visual_material(prim: Usd.Prim, *, path_name: str) -> Mesh:
        """Load a renderable mesh without changing physics mass properties."""
        material_props = _get_material_props_cached(prim)
        texture = material_props.get("texture")
        physics_mesh = _get_mesh_cached(prim)
        if texture is not None:
            render_mesh = _get_mesh_cached(prim, load_uvs=True)
            # Texture UV expansion is render-only. Preserve the collision mesh's
            # mass/inertia so visibility changes do not perturb simulation.
            mesh = Mesh(
                render_mesh.vertices,
                render_mesh.indices,
                normals=render_mesh.normals,
                uvs=render_mesh.uvs,
                compute_inertia=False,
                is_solid=physics_mesh.is_solid,
                maxhullvert=physics_mesh.maxhullvert,
                sdf=physics_mesh.sdf,
            )
            mesh.mass = physics_mesh.mass
            mesh.com = physics_mesh.com
            mesh.inertia = physics_mesh.inertia
            mesh.has_inertia = physics_mesh.has_inertia
        else:
            mesh = physics_mesh.copy(recompute_inertia=False)
        if texture is not None:
            mesh.texture = texture
        if mesh.texture is not None and mesh.uvs is None:
            logger.info("Mesh %s: dropping texture because UVs could not be recovered.", path_name)
            mesh.texture = None
        if material_props.get("color") is not None and mesh.texture is None:
            mesh.color = material_props["color"]
        if material_props.get("roughness") is not None:
            mesh.roughness = material_props["roughness"]
        if material_props.get("metallic") is not None:
            mesh.metallic = material_props["metallic"]
        return mesh

    def _get_face_material_subsets(prim: Usd.Prim) -> list[Usd.Prim]:
        """Return face-based material subsets authored directly under a mesh prim."""
        subsets = []
        for child in prim.GetChildren():
            try:
                is_subset = child.IsA(UsdGeom.Subset)
            except Exception:
                is_subset = False
            if not is_subset:
                continue

            subset = UsdGeom.Subset(child)
            element_type = subset.GetElementTypeAttr().Get()
            if element_type != UsdGeom.Tokens.face:
                continue
            family_name = subset.GetFamilyNameAttr().Get()
            if family_name and family_name != "materialBind":
                continue
            indices = subset.GetIndicesAttr().Get()
            if not indices:
                continue
            subsets.append(child)
        return subsets

    def _get_subset_uvs(prim: Usd.Prim, used_vertices: np.ndarray, expected_count: int) -> np.ndarray | None:
        """Return UVs for a material subset when a matching primvar is authored."""
        max_used_vertex = int(np.max(used_vertices, initial=-1))
        full_mesh_uvs = None
        for primvar in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
            name = primvar.GetBaseName()
            if not name.startswith("st"):
                continue
            values = primvar.Get()
            if values is None:
                continue
            uvs = np.asarray(values, dtype=np.float32)
            if primvar.IsIndexed():
                indices = primvar.GetIndices()
                if indices is None:
                    continue
                indices = np.asarray(indices, dtype=np.int32)
                if len(indices) == expected_count:
                    uvs = uvs[indices]
                    if len(uvs) == expected_count:
                        return uvs
                    continue
                if len(indices) > max_used_vertex:
                    uvs = uvs[indices]
                else:
                    continue
            if len(uvs) == expected_count:
                return uvs
            if full_mesh_uvs is None and len(uvs) > max_used_vertex:
                full_mesh_uvs = uvs[used_vertices]
        return full_mesh_uvs

    def _make_visual_submesh(
        mesh: Mesh,
        triangle_indices: np.ndarray,
        material_props: dict[str, Any],
        *,
        prim: Usd.Prim,
        path_name: str,
    ) -> Mesh | None:
        """Create a render-only mesh slice for the selected triangle rows."""
        if len(triangle_indices) == 0:
            return None

        triangles = mesh.indices.reshape(-1, 3)[triangle_indices]
        used_vertices = np.unique(triangles)
        vertex_remap = np.full(len(mesh.vertices), -1, dtype=np.int32)
        vertex_remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int32)

        normals = None
        if mesh.normals is not None and len(mesh.normals) == len(mesh.vertices):
            normals = mesh.normals[used_vertices]

        uvs = None
        if mesh.uvs is not None and len(mesh.uvs) == len(mesh.vertices):
            uvs = mesh.uvs[used_vertices]
        elif material_props.get("texture") is not None:
            uvs = _get_subset_uvs(prim, used_vertices, len(used_vertices))

        submesh = Mesh(
            mesh.vertices[used_vertices],
            vertex_remap[triangles].reshape(-1),
            normals=normals,
            uvs=uvs,
            compute_inertia=False,
            is_solid=mesh.is_solid,
            maxhullvert=mesh.maxhullvert,
        )

        texture = material_props.get("texture")
        if texture:
            submesh.texture = texture
        if submesh.texture is not None and submesh.uvs is None:
            logger.info("Mesh material subset %s: dropping texture because UVs could not be recovered.", path_name)
            submesh.texture = None

        color = material_props.get("color")
        if color is not None:
            submesh.color = color
        elif submesh.texture is not None:
            submesh.color = (1.0, 1.0, 1.0)
        if material_props.get("roughness") is not None:
            submesh.roughness = material_props["roughness"]
        if material_props.get("metallic") is not None:
            submesh.metallic = material_props["metallic"]
        return submesh

    def _get_visual_material_subset_meshes(prim: Usd.Prim) -> list[tuple[str, Mesh]]:
        """Load one render mesh per USD material subset when subsets are authored."""
        subsets = _get_face_material_subsets(prim)
        if not subsets:
            return []

        mesh_schema = UsdGeom.Mesh(prim)
        face_counts = mesh_schema.GetFaceVertexCountsAttr().Get()
        if face_counts is None:
            return []
        face_counts = np.asarray(face_counts, dtype=np.int32)
        if len(face_counts) == 0 or np.any(face_counts < 3):
            return []

        subset_props = [(str(subset.GetPath()), usd.resolve_material_properties_for_prim(subset)) for subset in subsets]
        mesh = _get_mesh_cached(prim)
        triangle_face_indices = np.repeat(np.arange(len(face_counts), dtype=np.int32), face_counts - 2)
        covered_faces = np.zeros(len(face_counts), dtype=bool)

        submeshes = []
        for subset_path, material_props in subset_props:
            # `resolve_material_properties_for_prim` does not fall back from a subset to its parent mesh
            # (see `newton/_src/usd/utils.py` resolve_material_properties_for_prim). If a subset binds no
            # visible material, let the uncovered-faces fallback below apply the parent mesh material
            # instead of producing a materialless submesh and hiding the parent material on those faces.
            if not any(value is not None for value in material_props.values()):
                continue
            subset = UsdGeom.Subset(stage.GetPrimAtPath(subset_path))
            subset_indices = np.asarray(subset.GetIndicesAttr().Get(), dtype=np.int32)
            valid = (subset_indices >= 0) & (subset_indices < len(face_counts))
            if not np.all(valid):
                logger.info(
                    "Mesh material subset %s: face indices outside the mesh face range; "
                    "out-of-range indices will be ignored.",
                    subset_path,
                )
                subset_indices = subset_indices[valid]
            if len(subset_indices) == 0:
                continue

            face_mask = np.zeros(len(face_counts), dtype=bool)
            face_mask[subset_indices] = True
            triangle_indices = np.nonzero(face_mask[triangle_face_indices])[0]
            submesh = _make_visual_submesh(mesh, triangle_indices, material_props, prim=prim, path_name=subset_path)
            if submesh is None:
                continue
            covered_faces[subset_indices] = True
            submeshes.append((subset_path, submesh))

        if not submeshes:
            return []

        uncovered_faces = np.nonzero(~covered_faces)[0]
        if len(uncovered_faces) > 0:
            face_mask = np.zeros(len(face_counts), dtype=bool)
            face_mask[uncovered_faces] = True
            triangle_indices = np.nonzero(face_mask[triangle_face_indices])[0]
            fallback_mesh = _make_visual_submesh(
                mesh,
                triangle_indices,
                _get_material_props_cached(prim),
                prim=prim,
                path_name=str(prim.GetPath()),
            )
            if fallback_mesh is not None:
                submeshes.insert(0, (str(prim.GetPath()), fallback_mesh))

        return submeshes

    def _get_tetmesh_cached(prim: Usd.Prim) -> TetMesh:
        """Load and cache TetMesh data to avoid repeated USD extraction."""
        prim_path = str(prim.GetPath())
        if prim_path not in tetmesh_cache:
            # Pass the resolver-declared namespaces explicitly (never None), so the importer keeps the
            # canonical physics: default and does not trip get_tetmesh()'s legacy-default deprecation.
            compat_ns = deformable_compat_ns
            if not compat_ns and usd._material_authors_legacy_deformable_attrs(prim):
                # Without this deprecation window, a vendor-only material would silently
                # import with default stiffness/density instead of its authored values.
                warnings.warn(
                    f"{prim_path}: the bound material authors legacy vendor-namespaced deformable "
                    f"material attributes (omniphysics: / physxDeformableBody:) without "
                    f"PhysicsVolumeDeformableMaterialAPI. add_usd() still reads them, but this is "
                    f"deprecated: author the canonical physics: attributes with the material API, or "
                    f"pass schema_resolvers=[..., SchemaResolverPhysx()] to keep vendor namespaces "
                    f"explicitly.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                compat_ns = usd.DEFORMABLE_LEGACY_NAMESPACES
            tetmesh_cache[prim_path] = usd.get_tetmesh(prim, compat_namespaces=compat_ns)
        return tetmesh_cache[prim_path]

    def _has_visual_material_properties(material_props: dict[str, Any]) -> bool:
        # Require PBR-like material cues to avoid promoting generic displayColor-only colliders.
        return any(material_props.get(key) is not None for key in ("texture", "roughness", "metallic"))

    def _is_effectively_visible(prim: Usd.Prim) -> bool:
        """Return whether ``prim`` is effectively visible in USD.

        A prim is effectively visible only when it is a :class:`UsdGeom.Imageable`
        whose inherited visibility is not ``invisible``. Non-imageable prims are
        not renderable in USD, so they are treated as not effectively visible.
        """
        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            return False
        return imageable.ComputeVisibility() != UsdGeom.Tokens.invisible

    bodies_with_visual_shapes: set[int] = set()

    def _get_prim_world_mat(prim, articulation_root_xform, incoming_world_xform):
        prim_world_mat = usd.get_transform_matrix(prim, local=False, xform_cache=xform_cache)
        if articulation_root_xform is not None:
            rebase_mat = _xform_to_mat44(wp.transform_inverse(articulation_root_xform))
            prim_world_mat = rebase_mat @ prim_world_mat
        if incoming_world_xform is not None:
            # Apply the incoming world transform in model space (static shapes or when using body_xform).
            incoming_mat = _xform_to_mat44(incoming_world_xform)
            prim_world_mat = incoming_mat @ prim_world_mat
        return prim_world_mat

    def _load_visual_shapes_impl(
        parent_body_id: int,
        prim: Usd.Prim,
        body_xform: wp.transform | None = None,
        articulation_root_xform: wp.transform | None = None,
    ):
        """Load visual-only shapes (non-physics) for a prim subtree.

        Args:
            parent_body_id: ModelBuilder body id to attach shapes to. Use -1 for
                static shapes that are not bound to any rigid body.
            prim: USD prim to inspect for visual geometry and recurse into.
            body_xform: Rigid body transform actually used by the builder.
                This matches any physics-authored pose, scene-level transforms,
                and incoming transforms that were applied when the body was created.
            articulation_root_xform: The articulation root's world-space transform,
                passed when override_root_xform=True. Strips the root's original
                pose from visual prim transforms to match the rebased body transforms.
        """
        if _is_enabled_collider(prim) or prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return
        path_name = str(prim.GetPath())
        if any(re.match(path, path_name) for path in ignore_paths):
            return

        prim_world_mat = _get_prim_world_mat(
            prim,
            articulation_root_xform,
            incoming_world_xform if (parent_body_id == -1 or body_xform is not None) else None,
        )
        if body_xform is not None:
            # Use the body transform used by the builder to avoid USD/physics pose mismatches.
            body_world_mat = _xform_to_mat44(body_xform)
            rel_mat = wp.inverse(body_world_mat) @ prim_world_mat
        else:
            rel_mat = prim_world_mat

        xform_pos, xform_rot, scale = wp.transform_decompose(rel_mat)
        xform = wp.transform(xform_pos, xform_rot)

        if prim.IsInstance():
            proto = prim.GetPrototype()
            for child in proto.GetChildren():
                # remap prototype child path to this instance's path (instance proxy)
                inst_path = child.GetPath().ReplacePrefix(proto.GetPath(), prim.GetPath())
                inst_child = stage.GetPrimAtPath(inst_path)
                _load_visual_shapes_impl(parent_body_id, inst_child, body_xform, articulation_root_xform)
            return
        type_name = str(prim.GetTypeName()).lower()
        if type_name.endswith("joint"):
            return

        shape_id = -1

        is_site = usd.has_applied_api_schema(prim, "NewtonSiteAPI") or usd.has_applied_api_schema(prim, "MjcSiteAPI")

        # Skip based on granular loading flags
        if is_site and not load_sites:
            return
        if not is_site and not load_visual_shapes:
            return

        visual_shape_cfg_for_prim = copy.copy(visual_shape_cfg)
        visual_shape_cfg_for_prim.is_visible = is_site or _is_effectively_visible(prim)
        material_props = _get_material_props_cached(prim)
        shape_color = material_props.get("color")

        if path_name not in path_shape_map:
            if type_name == "cube":
                size = usd.get_float(prim, "size", 2.0)
                side_lengths = scale * size
                shape_id = builder.add_shape_box(
                    parent_body_id,
                    xform=xform,
                    hx=side_lengths[0] / 2,
                    hy=side_lengths[1] / 2,
                    hz=side_lengths[2] / 2,
                    cfg=visual_shape_cfg_for_prim,
                    color=shape_color,
                    as_site=is_site,
                    label=path_name,
                )
            elif type_name == "sphere":
                if not (scale[0] == scale[1] == scale[2]):
                    print("Warning: Non-uniform scaling of spheres is not supported.")
                radius = usd.get_float(prim, "radius", 1.0) * max(scale)
                shape_id = builder.add_shape_sphere(
                    parent_body_id,
                    xform=xform,
                    radius=radius,
                    cfg=visual_shape_cfg_for_prim,
                    color=shape_color,
                    as_site=is_site,
                    label=path_name,
                )
            elif type_name == "plane":
                axis = usd.get_gprim_axis(prim)
                plane_xform = xform
                # Apply axis rotation to transform
                xform = wp.transform(xform.p, xform.q * quat_between_axes(Axis.Z, axis))
                width = usd.get_float(prim, "width", 0.0) * scale[0]
                length = usd.get_float(prim, "length", 0.0) * scale[1]
                shape_id = builder.add_shape_plane(
                    body=parent_body_id,
                    xform=plane_xform,
                    width=width,
                    length=length,
                    cfg=visual_shape_cfg_for_prim,
                    color=shape_color,
                    label=path_name,
                )
            elif type_name == "capsule":
                axis = usd.get_gprim_axis(prim)
                radius = usd.get_float(prim, "radius", 0.5) * scale[0]
                half_height = usd.get_float(prim, "height", 2.0) / 2 * scale[1]
                # Apply axis rotation to transform
                xform = wp.transform(xform.p, xform.q * quat_between_axes(Axis.Z, axis))
                shape_id = builder.add_shape_capsule(
                    parent_body_id,
                    xform=xform,
                    radius=radius,
                    half_height=half_height,
                    cfg=visual_shape_cfg_for_prim,
                    color=shape_color,
                    as_site=is_site,
                    label=path_name,
                )
            elif type_name == "cylinder":
                axis = usd.get_gprim_axis(prim)
                radius = usd.get_float(prim, "radius", 0.5) * scale[0]
                half_height = usd.get_float(prim, "height", 2.0) / 2 * scale[1]
                # Apply axis rotation to transform
                xform = wp.transform(xform.p, xform.q * quat_between_axes(Axis.Z, axis))
                shape_id = builder.add_shape_cylinder(
                    parent_body_id,
                    xform=xform,
                    radius=radius,
                    half_height=half_height,
                    cfg=visual_shape_cfg_for_prim,
                    color=shape_color,
                    as_site=is_site,
                    label=path_name,
                )
            elif type_name == "cone":
                axis = usd.get_gprim_axis(prim)
                radius = usd.get_float(prim, "radius", 0.5) * scale[0]
                half_height = usd.get_float(prim, "height", 2.0) / 2 * scale[1]
                # Apply axis rotation to transform
                xform = wp.transform(xform.p, xform.q * quat_between_axes(Axis.Z, axis))
                shape_id = builder.add_shape_cone(
                    parent_body_id,
                    xform=xform,
                    radius=radius,
                    half_height=half_height,
                    cfg=visual_shape_cfg_for_prim,
                    color=shape_color,
                    as_site=is_site,
                    label=path_name,
                )
            elif type_name == "mesh":
                subset_meshes = _get_visual_material_subset_meshes(prim)
                if subset_meshes:
                    for subset_path, subset_mesh in subset_meshes:
                        subset_shape_id = builder.add_shape_mesh(
                            parent_body_id,
                            xform=xform,
                            scale=scale,
                            mesh=subset_mesh,
                            cfg=visual_shape_cfg_for_prim,
                            color=None,
                            label=subset_path,
                        )
                        path_shape_map[subset_path] = subset_shape_id
                        path_shape_scale[subset_path] = scale
                        if shape_id < 0:
                            shape_id = subset_shape_id
                        if verbose:
                            print(
                                f"Added visual shape {subset_path} ({type_name} material subset) "
                                f"with id {subset_shape_id}."
                            )
                else:
                    mesh = _get_mesh_with_visual_material(prim, path_name=path_name)
                    shape_id = builder.add_shape_mesh(
                        parent_body_id,
                        xform=xform,
                        scale=scale,
                        mesh=mesh,
                        cfg=visual_shape_cfg_for_prim,
                        color=shape_color,
                        label=path_name,
                    )
            elif type_name == "particlefield3dgaussiansplat":
                gaussian = usd.get_gaussian(prim)
                shape_id = builder.add_shape_gaussian(
                    parent_body_id,
                    gaussian=gaussian,
                    xform=xform,
                    scale=scale,
                    cfg=visual_shape_cfg_for_prim,
                    color=shape_color,
                    label=path_name,
                )
            elif (
                len(type_name) > 0
                and type_name not in {"geomsubset", "material", "scope", "shader", "xform", "tetmesh"}
                and verbose
            ):
                print(f"Warning: Unsupported geometry type {type_name} at {path_name} while loading visual shapes.")

            if shape_id >= 0:
                path_shape_map[path_name] = shape_id
                path_shape_scale[path_name] = scale
                if not is_site and visual_shape_cfg_for_prim.is_visible:
                    bodies_with_visual_shapes.add(parent_body_id)
                if verbose:
                    print(f"Added visual shape {path_name} ({type_name}) with id {shape_id}.")

        for child in prim.GetChildren():
            _load_visual_shapes_impl(parent_body_id, child, body_xform, articulation_root_xform)

    def add_body(
        prim: Usd.Prim,
        xform: wp.transform,
        label: str,
        articulation_root_xform: wp.transform | None = None,
        is_kinematic: bool = False,
    ) -> int:
        """Add a rigid body to the builder and optionally load its visual shapes and sites among the body prim's children. Returns the resulting body index."""
        # Extract custom attributes for this body
        body_custom_attrs = usd.get_custom_attribute_values(
            prim, builder_custom_attr_body, context={"builder": builder}
        )

        b = builder.add_link(
            xform=xform,
            label=label,
            is_kinematic=is_kinematic,
            custom_attributes=body_custom_attrs,
        )
        path_body_map[label] = b
        if load_sites or load_visual_shapes:
            for child in prim.GetChildren():
                _load_visual_shapes_impl(b, child, body_xform=xform, articulation_root_xform=articulation_root_xform)
        return b

    def parse_body(
        rigid_body_desc: UsdPhysics.RigidBodyDesc,
        prim: Usd.Prim,
        incoming_xform: wp.transform | None = None,
        add_body_to_builder: bool = True,
        articulation_root_xform: wp.transform | None = None,
    ) -> int | dict[str, Any]:
        """Parses a rigid body description.
        If `add_body_to_builder` is True, adds it to the builder and returns the resulting body index.
        Otherwise returns a dictionary of body data that can be passed to ModelBuilder.add_body()."""
        nonlocal path_body_map
        nonlocal physics_scene_prim

        if not rigid_body_desc.rigidBodyEnabled and only_load_enabled_rigid_bodies:
            return -1

        rot = rigid_body_desc.rotation
        origin = wp.transform(rigid_body_desc.position, usd.value_to_warp(rot))
        if incoming_xform is not None:
            origin = wp.mul(incoming_xform, origin)
        path = str(prim.GetPath())

        is_kinematic = rigid_body_desc.kinematicBody

        if add_body_to_builder:
            return add_body(
                prim,
                origin,
                path,
                articulation_root_xform=articulation_root_xform,
                is_kinematic=is_kinematic,
            )
        else:
            result = {
                "prim": prim,
                "xform": origin,
                "label": path,
                "is_kinematic": is_kinematic,
            }
            if articulation_root_xform is not None:
                result["articulation_root_xform"] = articulation_root_xform
            return result

    def resolve_joint_parent_child(
        joint_desc: UsdPhysics.JointDesc,
        body_index_map: dict[str, int],
        get_transforms: bool = True,
    ):
        """Resolve the parent and child of a joint and return their parent + child transforms if requested."""
        if get_transforms:
            parent_tf = wp.transform(joint_desc.localPose0Position, usd.value_to_warp(joint_desc.localPose0Orientation))
            child_tf = wp.transform(joint_desc.localPose1Position, usd.value_to_warp(joint_desc.localPose1Orientation))
        else:
            parent_tf = None
            child_tf = None

        parent_path = str(joint_desc.body0)
        child_path = str(joint_desc.body1)
        parent_id = body_index_map.get(parent_path, -1)
        child_id = body_index_map.get(child_path, -1)
        # If child_id is -1, swap parent and child
        if child_id == -1:
            if parent_id == -1:
                raise ValueError(f"Unable to parse joint {joint_desc.primPath}: both bodies unresolved")
            parent_id, child_id = child_id, parent_id
            if get_transforms:
                parent_tf, child_tf = child_tf, parent_tf
            if verbose:
                print(f"Joint {joint_desc.primPath} connects {parent_path} to world")
        if get_transforms:
            return parent_id, child_id, parent_tf, child_tf
        else:
            return parent_id, child_id

    def parse_joint(
        joint_desc: UsdPhysics.JointDesc,
        incoming_xform: wp.transform | None = None,
    ) -> int | None:
        """Parse a joint description and add it to the builder. Returns the resulting joint index if successful, None otherwise."""
        if not joint_desc.jointEnabled and only_load_enabled_joints:
            return None
        key = joint_desc.type
        joint_path = str(joint_desc.primPath)
        joint_prim = stage.GetPrimAtPath(joint_desc.primPath)
        # collect engine-specific attributes on the joint prim if requested
        if collect_schema_attrs:
            R.collect_prim_attrs(joint_prim)
        parent_id, child_id, parent_tf, child_tf = resolve_joint_parent_child(  # pyright: ignore[reportAssignmentType]
            joint_desc, path_body_map, get_transforms=True
        )

        if incoming_xform is not None:
            parent_tf = incoming_xform * parent_tf

        joint_armature = R.get_value(
            joint_prim, prim_type=PrimType.JOINT, key="armature", default=default_joint_armature, verbose=verbose
        )
        joint_friction = R.get_value(
            joint_prim, prim_type=PrimType.JOINT, key="friction", default=default_joint_friction, verbose=verbose
        )
        _joint_damping_usd = R.get_value(
            joint_prim, prim_type=PrimType.JOINT, key="damping", default=None, verbose=verbose
        )
        joint_damping_authored = _joint_damping_usd is not None
        joint_damping = _joint_damping_usd if joint_damping_authored else default_joint_damping
        joint_velocity_limit = R.get_value(
            joint_prim,
            prim_type=PrimType.JOINT,
            key="velocity_limit",
            default=None,
            verbose=verbose,
        )
        # NewtonJointAPI uses +inf for "unlimited"; treat it as the builder default below.
        if joint_velocity_limit == float("inf"):
            joint_velocity_limit = None
        limit_ke = R.get_value(joint_prim, prim_type=PrimType.JOINT, key="limit_ke", default=None, verbose=verbose)
        limit_kd = R.get_value(joint_prim, prim_type=PrimType.JOINT, key="limit_kd", default=None, verbose=verbose)

        # Extract custom attributes for this joint
        joint_custom_attrs = usd.get_custom_attribute_values(
            joint_prim, builder_custom_attr_joint, context={"builder": builder}
        )
        joint_params = {
            "parent": parent_id,
            "child": child_id,
            "parent_xform": parent_tf,
            "child_xform": child_tf,
            "label": joint_path,
            "collision_filter_parent": parent_id != -1 and not joint_desc.collisionEnabled,
            "enabled": joint_desc.jointEnabled,
            "custom_attributes": joint_custom_attrs,
        }

        joint_index: int | None = None
        if key == UsdPhysics.ObjectType.FixedJoint:
            joint_index = builder.add_joint_fixed(**joint_params)
        elif key == UsdPhysics.ObjectType.RevoluteJoint or key == UsdPhysics.ObjectType.PrismaticJoint:
            # we need to scale the builder defaults for the joint limits to degrees for revolute joints
            if key == UsdPhysics.ObjectType.RevoluteJoint:
                limit_gains_scaling = DegreesToRadian
            else:
                limit_gains_scaling = 1.0

            limit_key = "limit_angular" if key == UsdPhysics.ObjectType.RevoluteJoint else "limit_linear"
            fallback_limit_ke, limit_ke_source = _resolve_joint_limit_gain(
                joint_prim,
                f"{limit_key}_ke",
                default_joint_limit_ke * limit_gains_scaling,
            )
            fallback_limit_kd, limit_kd_source = _resolve_joint_limit_gain(
                joint_prim,
                f"{limit_key}_kd",
                default_joint_limit_kd * limit_gains_scaling,
            )
            current_joint_limit_ke, limit_ke_source = _resolve_newton_limit_ke(
                limit_ke, fallback_limit_ke, limit_ke_source, default_joint_limit_ke * limit_gains_scaling
            )
            current_joint_limit_kd, limit_kd_source = _resolve_newton_limit_kd(
                limit_ke, limit_kd, fallback_limit_kd, limit_kd_source, default_joint_limit_kd * limit_gains_scaling
            )
            if _should_write_solreflimit_mode():
                joint_custom_attrs[solreflimit_mode_key] = _joint_limit_solref_mode(limit_ke_source, limit_kd_source)
            joint_params["axis"] = usd_axis_to_axis[joint_desc.axis]
            joint_params["limit_lower"] = joint_desc.limit.lower
            joint_params["limit_upper"] = joint_desc.limit.upper
            joint_params["limit_ke"] = current_joint_limit_ke
            joint_params["limit_kd"] = current_joint_limit_kd
            joint_params["armature"] = joint_armature
            joint_params["friction"] = joint_friction
            joint_params["damping"] = joint_damping
            joint_params["velocity_limit"] = joint_velocity_limit
            if joint_desc.drive.enabled:
                target_vel = joint_desc.drive.targetVelocity
                target_pos = joint_desc.drive.targetPosition
                target_ke = joint_desc.drive.stiffness
                target_kd = joint_desc.drive.damping

                joint_params["target_vel"] = target_vel
                joint_params["target_pos"] = target_pos
                joint_params["target_ke"] = target_ke
                joint_params["target_kd"] = target_kd
                joint_params["effort_limit"] = joint_desc.drive.forceLimit

                joint_params["actuator_mode"] = JointTargetMode.from_gains(
                    target_ke, target_kd, force_position_velocity_actuation, has_drive=True
                )
            else:
                joint_params["actuator_mode"] = JointTargetMode.NONE

            # Read initial joint state BEFORE creating/overwriting USD attributes
            initial_position = None
            initial_velocity = None
            dof_type = "linear" if key == UsdPhysics.ObjectType.PrismaticJoint else "angular"

            # Resolve initial joint state from schema resolver
            if dof_type == "angular":
                initial_position = R.get_value(
                    joint_prim, PrimType.JOINT, "angular_position", default=None, verbose=verbose
                )
                initial_velocity = R.get_value(
                    joint_prim, PrimType.JOINT, "angular_velocity", default=None, verbose=verbose
                )
            else:  # linear
                initial_position = R.get_value(
                    joint_prim, PrimType.JOINT, "linear_position", default=None, verbose=verbose
                )
                initial_velocity = R.get_value(
                    joint_prim, PrimType.JOINT, "linear_velocity", default=None, verbose=verbose
                )

            if key == UsdPhysics.ObjectType.PrismaticJoint:
                joint_index = builder.add_joint_prismatic(**joint_params)
            else:
                if joint_desc.drive.enabled:
                    joint_params["target_pos"] *= DegreesToRadian
                    joint_params["target_vel"] *= DegreesToRadian
                    joint_params["target_kd"] /= DegreesToRadian / joint_drive_gains_scaling
                    joint_params["target_ke"] /= DegreesToRadian / joint_drive_gains_scaling

                joint_params["limit_lower"] *= DegreesToRadian
                joint_params["limit_upper"] *= DegreesToRadian
                joint_params["limit_ke"] /= DegreesToRadian
                joint_params["limit_kd"] /= DegreesToRadian
                if joint_damping_authored:
                    joint_params["damping"] /= DegreesToRadian
                if joint_params["velocity_limit"] is not None:
                    joint_params["velocity_limit"] *= DegreesToRadian

                joint_index = builder.add_joint_revolute(**joint_params)
        elif key == UsdPhysics.ObjectType.SphericalJoint:
            joint_index = builder.add_joint_ball(**joint_params)
        elif key == UsdPhysics.ObjectType.D6Joint:
            linear_axes = []
            angular_axes = []
            num_dofs = 0
            # Store initial state for D6 joints
            d6_initial_positions = {}
            d6_initial_velocities = {}
            # Track which axes were added as DOFs (in order)
            d6_dof_axes = []
            linear_solref_modes: list[int] = []
            angular_solref_modes: list[int] = []
            # print(joint_desc.jointLimits, joint_desc.jointDrives)
            # print(joint_desc.body0)
            # print(joint_desc.body1)
            # print(joint_desc.jointLimits)
            # print("Limits")
            # for limit in joint_desc.jointLimits:
            #     print("joint_path :", joint_path, limit.first, limit.second.lower, limit.second.upper)
            # print("Drives")
            # for drive in joint_desc.jointDrives:
            #     print("joint_path :", joint_path, drive.first, drive.second.targetPosition, drive.second.targetVelocity)

            for limit in joint_desc.jointLimits:
                dof = limit.first
                if limit.second.enabled:
                    limit_lower = limit.second.lower
                    limit_upper = limit.second.upper
                else:
                    limit_lower = builder.default_joint_cfg.limit_lower
                    limit_upper = builder.default_joint_cfg.limit_upper

                free_axis = limit_lower < limit_upper

                def define_joint_targets(dof, joint_desc):
                    target_pos = 0.0  # TODO: parse target from state:*:physics:appliedForce usd attribute when no drive is present
                    target_vel = 0.0
                    target_ke = 0.0
                    target_kd = 0.0
                    effort_limit = np.inf
                    has_drive = False
                    for drive in joint_desc.jointDrives:
                        if drive.first != dof:
                            continue
                        if drive.second.enabled:
                            has_drive = True
                            target_vel = drive.second.targetVelocity
                            target_pos = drive.second.targetPosition
                            target_ke = drive.second.stiffness
                            target_kd = drive.second.damping
                            effort_limit = drive.second.forceLimit
                    actuator_mode = JointTargetMode.from_gains(
                        target_ke, target_kd, force_position_velocity_actuation, has_drive=has_drive
                    )
                    return target_pos, target_vel, target_ke, target_kd, effort_limit, actuator_mode

                target_pos, target_vel, target_ke, target_kd, effort_limit, actuator_mode = define_joint_targets(
                    dof, joint_desc
                )

                _trans_axes = {
                    UsdPhysics.JointDOF.TransX: (1.0, 0.0, 0.0),
                    UsdPhysics.JointDOF.TransY: (0.0, 1.0, 0.0),
                    UsdPhysics.JointDOF.TransZ: (0.0, 0.0, 1.0),
                }
                _rot_axes = {
                    UsdPhysics.JointDOF.RotX: (1.0, 0.0, 0.0),
                    UsdPhysics.JointDOF.RotY: (0.0, 1.0, 0.0),
                    UsdPhysics.JointDOF.RotZ: (0.0, 0.0, 1.0),
                }
                _rot_names = {
                    UsdPhysics.JointDOF.RotX: "rotX",
                    UsdPhysics.JointDOF.RotY: "rotY",
                    UsdPhysics.JointDOF.RotZ: "rotZ",
                }
                if free_axis and dof in _trans_axes:
                    # Per-axis translation names: transX/transY/transZ
                    trans_name = {
                        UsdPhysics.JointDOF.TransX: "transX",
                        UsdPhysics.JointDOF.TransY: "transY",
                        UsdPhysics.JointDOF.TransZ: "transZ",
                    }[dof]
                    # Store initial state for this axis
                    d6_initial_positions[trans_name] = R.get_value(
                        joint_prim,
                        PrimType.JOINT,
                        f"{trans_name}_position",
                        default=None,
                        verbose=verbose,
                    )
                    d6_initial_velocities[trans_name] = R.get_value(
                        joint_prim,
                        PrimType.JOINT,
                        f"{trans_name}_velocity",
                        default=None,
                        verbose=verbose,
                    )
                    fallback_limit_ke, limit_ke_source = _resolve_joint_limit_gain(
                        joint_prim,
                        f"limit_{trans_name}_ke",
                        default_joint_limit_ke,
                    )
                    fallback_limit_kd, limit_kd_source = _resolve_joint_limit_gain(
                        joint_prim,
                        f"limit_{trans_name}_kd",
                        default_joint_limit_kd,
                    )
                    current_joint_limit_ke, limit_ke_source = _resolve_newton_limit_ke(
                        limit_ke, fallback_limit_ke, limit_ke_source, default_joint_limit_ke
                    )
                    current_joint_limit_kd, limit_kd_source = _resolve_newton_limit_kd(
                        limit_ke, limit_kd, fallback_limit_kd, limit_kd_source, default_joint_limit_kd
                    )
                    linear_axes.append(
                        ModelBuilder.JointDofConfig(
                            axis=_trans_axes[dof],
                            limit_lower=limit_lower,
                            limit_upper=limit_upper,
                            limit_ke=current_joint_limit_ke,
                            limit_kd=current_joint_limit_kd,
                            target_pos=target_pos,
                            target_vel=target_vel,
                            target_ke=target_ke,
                            target_kd=target_kd,
                            damping=joint_damping,
                            armature=joint_armature,
                            effort_limit=effort_limit,
                            velocity_limit=joint_velocity_limit
                            if joint_velocity_limit is not None
                            else default_joint_velocity_limit,
                            friction=joint_friction,
                            actuator_mode=actuator_mode,
                        )
                    )
                    linear_solref_modes.append(_joint_limit_solref_mode(limit_ke_source, limit_kd_source))
                    # Track that this axis was added as a DOF
                    d6_dof_axes.append(trans_name)
                elif free_axis and dof in _rot_axes:
                    # Resolve per-axis rotational gains
                    rot_name = _rot_names[dof]
                    # Store initial state for this axis
                    d6_initial_positions[rot_name] = R.get_value(
                        joint_prim,
                        PrimType.JOINT,
                        f"{rot_name}_position",
                        default=None,
                        verbose=verbose,
                    )
                    d6_initial_velocities[rot_name] = R.get_value(
                        joint_prim,
                        PrimType.JOINT,
                        f"{rot_name}_velocity",
                        default=None,
                        verbose=verbose,
                    )
                    fallback_limit_ke, limit_ke_source = _resolve_joint_limit_gain(
                        joint_prim,
                        f"limit_{rot_name}_ke",
                        default_joint_limit_ke * DegreesToRadian,
                    )
                    fallback_limit_kd, limit_kd_source = _resolve_joint_limit_gain(
                        joint_prim,
                        f"limit_{rot_name}_kd",
                        default_joint_limit_kd * DegreesToRadian,
                    )
                    current_joint_limit_ke, limit_ke_source = _resolve_newton_limit_ke(
                        limit_ke,
                        fallback_limit_ke,
                        limit_ke_source,
                        default_joint_limit_ke * DegreesToRadian,
                    )
                    current_joint_limit_kd, limit_kd_source = _resolve_newton_limit_kd(
                        limit_ke,
                        limit_kd,
                        fallback_limit_kd,
                        limit_kd_source,
                        default_joint_limit_kd * DegreesToRadian,
                    )

                    angular_axes.append(
                        ModelBuilder.JointDofConfig(
                            axis=_rot_axes[dof],
                            limit_lower=limit_lower * DegreesToRadian,
                            limit_upper=limit_upper * DegreesToRadian,
                            limit_ke=current_joint_limit_ke / DegreesToRadian,
                            limit_kd=current_joint_limit_kd / DegreesToRadian,
                            target_pos=target_pos * DegreesToRadian,
                            target_vel=target_vel * DegreesToRadian,
                            target_ke=target_ke / DegreesToRadian / joint_drive_gains_scaling,
                            target_kd=target_kd / DegreesToRadian / joint_drive_gains_scaling,
                            damping=joint_damping / DegreesToRadian if joint_damping_authored else joint_damping,
                            armature=joint_armature,
                            effort_limit=effort_limit,
                            velocity_limit=joint_velocity_limit * DegreesToRadian
                            if joint_velocity_limit is not None
                            else default_joint_velocity_limit,
                            friction=joint_friction,
                            actuator_mode=actuator_mode,
                        )
                    )
                    angular_solref_modes.append(_joint_limit_solref_mode(limit_ke_source, limit_kd_source))
                    # Track that this axis was added as a DOF
                    d6_dof_axes.append(rot_name)
                    num_dofs += 1

            if _should_write_solreflimit_mode():
                joint_custom_attrs[solreflimit_mode_key] = linear_solref_modes + angular_solref_modes

            joint_index = builder.add_joint_d6(**joint_params, linear_axes=linear_axes, angular_axes=angular_axes)
        elif key == UsdPhysics.ObjectType.DistanceJoint:
            if joint_desc.limit.enabled and joint_desc.minEnabled:
                min_dist = joint_desc.limit.lower
            else:
                min_dist = -1.0  # no limit
            if joint_desc.limit.enabled and joint_desc.maxEnabled:
                max_dist = joint_desc.limit.upper
            else:
                max_dist = -1.0
            joint_index = builder.add_joint_distance(**joint_params, min_distance=min_dist, max_distance=max_dist)
        else:
            raise NotImplementedError(f"Unsupported joint type {key}")

        if joint_index is None:
            raise ValueError(f"Failed to add joint {joint_path}")

        # map the joint path to the index at insertion time
        path_joint_map[joint_path] = joint_index

        # Apply saved initial joint state after joint creation
        if key in (UsdPhysics.ObjectType.RevoluteJoint, UsdPhysics.ObjectType.PrismaticJoint):
            if initial_position is not None:
                q_start = builder.joint_q_start[joint_index]
                if key == UsdPhysics.ObjectType.RevoluteJoint:
                    builder.joint_q[q_start] = initial_position * DegreesToRadian
                else:
                    builder.joint_q[q_start] = initial_position
                if verbose:
                    joint_type_str = "revolute" if key == UsdPhysics.ObjectType.RevoluteJoint else "prismatic"
                    print(
                        f"Set {joint_type_str} joint {joint_index} position to {initial_position} ({'rad' if key == UsdPhysics.ObjectType.RevoluteJoint else 'm'})"
                    )
            if initial_velocity is not None:
                qd_start = builder.joint_qd_start[joint_index]
                if key == UsdPhysics.ObjectType.RevoluteJoint:
                    builder.joint_qd[qd_start] = initial_velocity  # velocity is already in rad/s
                else:
                    builder.joint_qd[qd_start] = initial_velocity
                if verbose:
                    joint_type_str = "revolute" if key == UsdPhysics.ObjectType.RevoluteJoint else "prismatic"
                    print(f"Set {joint_type_str} joint {joint_index} velocity to {initial_velocity} rad/s")
        elif key == UsdPhysics.ObjectType.D6Joint:
            # Apply D6 joint initial state
            q_start = builder.joint_q_start[joint_index]
            qd_start = builder.joint_qd_start[joint_index]

            # Get joint coordinate and DOF ranges
            if joint_index + 1 < len(builder.joint_q_start):
                q_end = builder.joint_q_start[joint_index + 1]
                qd_end = builder.joint_qd_start[joint_index + 1]
            else:
                q_end = len(builder.joint_q)
                qd_end = len(builder.joint_qd)

            # Apply initial values for each axis that was actually added as a DOF
            for dof_idx, axis_name in enumerate(d6_dof_axes):
                if dof_idx >= (qd_end - qd_start):
                    break

                is_rot = axis_name.startswith("rot")
                pos = d6_initial_positions.get(axis_name)
                vel = d6_initial_velocities.get(axis_name)

                if pos is not None and q_start + dof_idx < q_end:
                    coord_val = pos * DegreesToRadian if is_rot else pos
                    builder.joint_q[q_start + dof_idx] = coord_val
                    if verbose:
                        print(f"Set D6 joint {joint_index} {axis_name} position to {pos} ({'deg' if is_rot else 'm'})")

                if vel is not None and qd_start + dof_idx < qd_end:
                    vel_val = vel  # D6 velocities are already in correct units
                    builder.joint_qd[qd_start + dof_idx] = vel_val
                    if verbose:
                        print(f"Set D6 joint {joint_index} {axis_name} velocity to {vel} rad/s")

        return joint_index

    def parse_merged_joints(
        joint_paths: list[str],
        incoming_xform: wp.transform | None = None,
    ) -> int | None:
        """Combine multiple single-DOF joints between the same two bodies into one D6 joint.

        This handles USD files where multi-DOF MuJoCo joints are represented as
        separate PhysicsRevoluteJoint / PhysicsPrismaticJoint prims connecting the
        same parent and child bodies.  The individual joints are merged into a
        single :func:`~newton.ModelBuilder.add_joint_d6` call, following the same
        pattern used by the MJCF importer.

        Args:
            joint_paths: Prim paths of the joints to merge (all must share the
                same body pair).
            incoming_xform: Optional world-space transform applied to the parent
                frame of the first joint.

        Returns:
            The builder joint index of the newly created D6 joint, or ``None`` if
            all joints in the group are disabled.
        """
        linear_axes: list[ModelBuilder.JointDofConfig] = []
        angular_axes: list[ModelBuilder.JointDofConfig] = []
        # Track prim paths and initial state separately for linear/angular DOFs
        # because add_joint_d6 orders linear DOFs first, then angular
        linear_prim_paths: list[str] = []
        angular_prim_paths: list[str] = []
        linear_initial_pos: list[float | None] = []
        linear_initial_vel: list[float | None] = []
        angular_initial_pos: list[float | None] = []
        angular_initial_vel: list[float | None] = []
        enabled_count = 0
        collision_filter_parent = False

        # Find the first enabled joint to use as representative for transforms and metadata
        first_desc = None
        first_prim = None
        for jp in joint_paths:
            jd = joint_descriptions[jp]
            if not jd.jointEnabled and only_load_enabled_joints:
                continue
            first_desc = jd
            first_prim = stage.GetPrimAtPath(jd.primPath)
            break
        if first_desc is None:
            return None  # all joints disabled

        parent_id, child_id, parent_tf, child_tf = resolve_joint_parent_child(  # pyright: ignore[reportAssignmentType]
            first_desc, path_body_map, get_transforms=True
        )
        if incoming_xform is not None:
            parent_tf = incoming_xform * parent_tf

        # Warn if any sibling joint has a different anchor position.
        # Different local rotations are expected (they encode different DOF axis directions)
        # and are handled by remapping axes into the representative frame.
        for jp in joint_paths:
            jd = joint_descriptions[jp]
            if jd is first_desc:
                continue
            _, _, other_parent_tf, other_child_tf = resolve_joint_parent_child(  # pyright: ignore[reportAssignmentType]
                jd, path_body_map, get_transforms=True
            )
            parent_pos_match = np.allclose(parent_tf.p, other_parent_tf.p, atol=1e-6)
            child_pos_match = np.allclose(child_tf.p, other_child_tf.p, atol=1e-6)
            if not (parent_pos_match and child_pos_match):
                warnings.warn(
                    f"Merged joint {jp} has different anchor positions than representative "
                    f"{first_desc.primPath}; using representative positions for the D6 joint.",
                    stacklevel=2,
                )
                break

        # Split custom attributes into joint-level (one value per joint) and
        # per-DOF (one value per DOF).  Joint-level attrs come from the
        # representative prim; per-DOF attrs are collected from each sibling.
        joint_freq_attrs = [a for a in builder_custom_attr_joint if a.frequency == AttributeFrequency.JOINT]
        dof_freq_attrs = [
            a
            for a in builder_custom_attr_joint
            if a.frequency in (AttributeFrequency.JOINT_DOF, AttributeFrequency.JOINT_COORD)
        ]
        joint_custom_attrs = usd.get_custom_attribute_values(first_prim, joint_freq_attrs, context={"builder": builder})
        # Per-DOF custom attributes accumulated separately for linear / angular
        # so we can reorder to D6 DOF order (linear first, then angular).
        linear_dof_custom: list[dict[str, Any]] = []
        angular_dof_custom: list[dict[str, Any]] = []

        # Cache the representative parent-side rotation for axis remapping
        rep_parent_rot = np.array(parent_tf.q, dtype=float)

        for jp in joint_paths:
            jd = joint_descriptions[jp]
            if not jd.jointEnabled and only_load_enabled_joints:
                continue
            collision_filter_parent = collision_filter_parent or not jd.collisionEnabled
            jp_prim = stage.GetPrimAtPath(jd.primPath)
            if collect_schema_attrs:
                R.collect_prim_attrs(jp_prim)

            key = jd.type
            if key not in (UsdPhysics.ObjectType.RevoluteJoint, UsdPhysics.ObjectType.PrismaticJoint):
                raise ValueError(
                    f"Cannot merge joint {jp} of type {key} into a D6 joint. "
                    "Only RevoluteJoint and PrismaticJoint are supported for merging."
                )

            is_revolute = key == UsdPhysics.ObjectType.RevoluteJoint
            if is_revolute:
                limit_gains_scaling = DegreesToRadian
            else:
                limit_gains_scaling = 1.0

            j_armature = R.get_value(
                jp_prim, prim_type=PrimType.JOINT, key="armature", default=default_joint_armature, verbose=verbose
            )
            j_friction = R.get_value(
                jp_prim, prim_type=PrimType.JOINT, key="friction", default=default_joint_friction, verbose=verbose
            )
            _j_damping_usd = R.get_value(
                jp_prim, prim_type=PrimType.JOINT, key="damping", default=None, verbose=verbose
            )
            j_damping_authored = _j_damping_usd is not None
            j_damping = _j_damping_usd if j_damping_authored else default_joint_damping
            j_velocity_limit = R.get_value(
                jp_prim, prim_type=PrimType.JOINT, key="velocity_limit", default=None, verbose=verbose
            )
            if j_velocity_limit == float("inf"):
                j_velocity_limit = None

            limit_key = "limit_angular" if is_revolute else "limit_linear"
            j_newton_limit_ke = R.get_value(
                jp_prim, prim_type=PrimType.JOINT, key="limit_ke", default=None, verbose=verbose
            )
            j_newton_limit_kd = R.get_value(
                jp_prim, prim_type=PrimType.JOINT, key="limit_kd", default=None, verbose=verbose
            )
            fallback_limit_ke, limit_ke_source = _resolve_joint_limit_gain(
                jp_prim,
                f"{limit_key}_ke",
                default_joint_limit_ke * limit_gains_scaling,
            )
            fallback_limit_kd, limit_kd_source = _resolve_joint_limit_gain(
                jp_prim,
                f"{limit_key}_kd",
                default_joint_limit_kd * limit_gains_scaling,
            )
            limit_ke, limit_ke_source = _resolve_newton_limit_ke(
                j_newton_limit_ke,
                fallback_limit_ke,
                limit_ke_source,
                default_joint_limit_ke * limit_gains_scaling,
            )
            limit_kd, limit_kd_source = _resolve_newton_limit_kd(
                j_newton_limit_ke,
                j_newton_limit_kd,
                fallback_limit_kd,
                limit_kd_source,
                default_joint_limit_kd * limit_gains_scaling,
            )

            limit_lower = jd.limit.lower
            limit_upper = jd.limit.upper

            # Build drive params
            target_pos = 0.0
            target_vel = 0.0
            target_ke = 0.0
            target_kd = 0.0
            effort_limit = np.inf
            actuator_mode = JointTargetMode.NONE
            if jd.drive.enabled:
                target_vel = jd.drive.targetVelocity
                target_pos = jd.drive.targetPosition
                target_ke = jd.drive.stiffness
                target_kd = jd.drive.damping
                effort_limit = jd.drive.forceLimit
                actuator_mode = JointTargetMode.from_gains(
                    target_ke, target_kd, force_position_velocity_actuation, has_drive=True
                )

            # Read initial joint state
            initial_position = None
            initial_velocity = None
            if is_revolute:
                initial_position = R.get_value(
                    jp_prim, PrimType.JOINT, "angular_position", default=None, verbose=verbose
                )
                initial_velocity = R.get_value(
                    jp_prim, PrimType.JOINT, "angular_velocity", default=None, verbose=verbose
                )
            else:
                initial_position = R.get_value(
                    jp_prim, PrimType.JOINT, "linear_position", default=None, verbose=verbose
                )
                initial_velocity = R.get_value(
                    jp_prim, PrimType.JOINT, "linear_velocity", default=None, verbose=verbose
                )

            # Unit conversion for revolute joints
            if is_revolute:
                limit_lower *= DegreesToRadian
                limit_upper *= DegreesToRadian
                limit_ke /= DegreesToRadian
                limit_kd /= DegreesToRadian
                if j_damping_authored:
                    j_damping /= DegreesToRadian
                if jd.drive.enabled:
                    target_pos *= DegreesToRadian
                    target_vel *= DegreesToRadian
                    target_kd /= DegreesToRadian / joint_drive_gains_scaling
                    target_ke /= DegreesToRadian / joint_drive_gains_scaling
                if j_velocity_limit is not None:
                    j_velocity_limit *= DegreesToRadian
                if initial_position is not None:
                    initial_position *= DegreesToRadian

            # Compute the DOF axis in the representative joint's frame.
            # Each USD joint may have a different localRot that orients its fixed axis
            # (X, Y, or Z) to the physical DOF direction.  We remap into the rep frame.
            _, _, jp_parent_tf, _ = resolve_joint_parent_child(  # pyright: ignore[reportAssignmentType]
                jd, path_body_map, get_transforms=True
            )
            jp_parent_rot = np.array(jp_parent_tf.q, dtype=float)
            # q and -q represent the same rotation
            if abs(np.dot(rep_parent_rot, jp_parent_rot)) > 1.0 - 1e-6:
                # Same rotation — use the original axis directly
                dof_axis = usd_axis_to_axis[jd.axis]
            else:
                # Different rotation — transform axis into rep frame
                rep_q_inv = wp.quat_inverse(wp.quat(*rep_parent_rot.tolist()))
                jp_q = wp.quat(*jp_parent_rot.tolist())
                relative_q = wp.mul(rep_q_inv, jp_q)
                # Axis enum value: 0=X, 1=Y, 2=Z → unit vector
                axis_idx = int(usd_axis_to_axis[jd.axis])
                axis_unit = [0.0, 0.0, 0.0]
                axis_unit[axis_idx] = 1.0
                rotated = wp.quat_rotate(relative_q, wp.vec3(axis_unit[0], axis_unit[1], axis_unit[2]))
                dof_axis = (float(rotated[0]), float(rotated[1]), float(rotated[2]))

            ax = ModelBuilder.JointDofConfig(
                axis=dof_axis,
                limit_lower=limit_lower,
                limit_upper=limit_upper,
                limit_ke=limit_ke,
                limit_kd=limit_kd,
                target_pos=target_pos,
                target_vel=target_vel,
                target_ke=target_ke,
                target_kd=target_kd,
                damping=j_damping,
                armature=j_armature,
                friction=j_friction,
                effort_limit=effort_limit,
                velocity_limit=j_velocity_limit if j_velocity_limit is not None else default_joint_velocity_limit,
                actuator_mode=actuator_mode,
            )

            # Collect per-DOF custom attributes from this sibling prim
            sibling_dof_attrs = usd.get_custom_attribute_values(jp_prim, dof_freq_attrs, context={"builder": builder})
            if _should_write_solreflimit_mode():
                sibling_dof_attrs[solreflimit_mode_key] = _joint_limit_solref_mode(limit_ke_source, limit_kd_source)

            if is_revolute:
                angular_axes.append(ax)
                angular_prim_paths.append(jp)
                angular_initial_pos.append(initial_position)
                angular_initial_vel.append(initial_velocity)
                angular_dof_custom.append(sibling_dof_attrs)
            else:
                linear_axes.append(ax)
                linear_prim_paths.append(jp)
                linear_initial_pos.append(initial_position)
                linear_initial_vel.append(initial_velocity)
                linear_dof_custom.append(sibling_dof_attrs)

            enabled_count += 1

        if enabled_count == 0:
            return None

        # D6 DOF order: linear first, then angular
        dof_prim_paths = linear_prim_paths + angular_prim_paths
        dof_initial_pos = linear_initial_pos + angular_initial_pos
        dof_initial_vel = linear_initial_vel + angular_initial_vel
        ordered_dof_custom = linear_dof_custom + angular_dof_custom

        # Merge per-DOF custom attributes into DOF-indexed dicts for add_joint_d6.
        # Each entry in ordered_dof_custom is a dict of {attr_key: value} from one sibling prim.
        # We assemble {attr_key: {dof_index: value}} so _process_joint_custom_attributes
        # assigns each DOF its own value instead of broadcasting from the representative.
        for dof_idx, dof_attrs in enumerate(ordered_dof_custom):
            for attr_key, value in dof_attrs.items():
                if attr_key not in joint_custom_attrs:
                    joint_custom_attrs[attr_key] = {}
                existing = joint_custom_attrs[attr_key]
                if not isinstance(existing, dict):
                    # First per-DOF value for an attr that was already set as a scalar
                    # from the representative — convert to a dict to allow per-DOF override.
                    joint_custom_attrs[attr_key] = {dof_idx: value}
                else:
                    existing[dof_idx] = value

        # Use the representative (first enabled) joint path as the D6 joint label
        label = str(first_desc.primPath)

        # Register original prim paths as DOF labels so MjcActuator targets resolve correctly
        if "mujoco:joint_dof_label" in builder.custom_attributes:
            joint_custom_attrs["mujoco:joint_dof_label"] = dof_prim_paths

        joint_index = builder.add_joint_d6(
            parent=parent_id,
            child=child_id,
            linear_axes=linear_axes if linear_axes else None,
            angular_axes=angular_axes if angular_axes else None,
            parent_xform=parent_tf,
            child_xform=child_tf,
            label=label,
            collision_filter_parent=parent_id != -1 and collision_filter_parent,
            enabled=first_desc.jointEnabled,
            custom_attributes=joint_custom_attrs,
        )

        # Register all original joint prim paths in path_joint_map and track per-path DOF offsets
        for jp in joint_paths:
            path_joint_map[jp] = joint_index
        for dof_idx, dof_path in enumerate(dof_prim_paths):
            merged_dof_offset[dof_path] = dof_idx

        # Apply initial positions/velocities
        q_start = builder.joint_q_start[joint_index]
        qd_start = builder.joint_qd_start[joint_index]
        for dof_idx, (pos, vel) in enumerate(zip(dof_initial_pos, dof_initial_vel, strict=True)):
            if pos is not None:
                builder.joint_q[q_start + dof_idx] = pos
            if vel is not None:
                builder.joint_qd[qd_start + dof_idx] = vel

        if verbose:
            print(
                f"Merged {len(joint_paths)} joints into D6 joint {joint_index}: "
                f"{len(linear_axes)} linear + {len(angular_axes)} angular DOFs"
            )

        return joint_index

    # Looking for and parsing the attributes on PhysicsScene prims
    scene_attributes = {}
    physics_scene_prim = None
    if UsdPhysics.ObjectType.Scene in ret_dict:
        paths, scene_descs = ret_dict[UsdPhysics.ObjectType.Scene]
        if len(paths) > 1 and verbose:
            print("Only the first PhysicsScene is considered")
        path, scene_desc = paths[0], scene_descs[0]
        if verbose:
            print("Found PhysicsScene:", path)
            print("Gravity direction:", scene_desc.gravityDirection)
            print("Gravity magnitude:", scene_desc.gravityMagnitude)
        builder.gravity = -scene_desc.gravityMagnitude

        # Storing Physics Scene attributes
        physics_scene_prim = stage.GetPrimAtPath(path)
        for a in physics_scene_prim.GetAttributes():
            scene_attributes[a.GetName()] = a.Get()

        # Parse custom attribute declarations from PhysicsScene prim
        # This must happen before processing any other prims
        declarations = usd.get_custom_attribute_declarations(physics_scene_prim)
        for attr in declarations.values():
            builder.add_custom_attribute(attr)

        # Updating joint_drive_gains_scaling if set of the PhysicsScene
        joint_drive_gains_scaling = usd.get_float(
            physics_scene_prim, "newton:joint_drive_gains_scaling", joint_drive_gains_scaling
        )

        time_steps_per_second = R.get_value(
            physics_scene_prim, prim_type=PrimType.SCENE, key="time_steps_per_second", default=1000, verbose=verbose
        )
        physics_dt = (1.0 / time_steps_per_second) if time_steps_per_second > 0 else 0.001

        gravity_enabled = R.get_value(
            physics_scene_prim, prim_type=PrimType.SCENE, key="gravity_enabled", default=True, verbose=verbose
        )
        if not gravity_enabled:
            builder.gravity = 0.0
        max_solver_iters = R.get_value(
            physics_scene_prim, prim_type=PrimType.SCENE, key="max_solver_iterations", default=None, verbose=verbose
        )

    stage_up_axis = Axis.from_string(str(UsdGeom.GetStageUpAxis(stage)))

    if apply_up_axis_from_stage:
        builder.up_axis = stage_up_axis
        axis_xform = wp.transform_identity()
        if verbose:
            print(f"Using stage up axis {stage_up_axis} as builder up axis")
    else:
        axis_xform = wp.transform(wp.vec3(0.0), quat_between_axes(stage_up_axis, builder.up_axis))
        if verbose:
            print(f"Rotating stage to align its up axis {stage_up_axis} with builder up axis {builder.up_axis}")
    if override_root_xform and xform is None:
        raise ValueError("override_root_xform=True requires xform to be set")

    if xform is None:
        incoming_world_xform = axis_xform
    else:
        incoming_world_xform = wp.transform(*xform) * axis_xform

    if verbose:
        print(
            f"Scaling PD gains by (joint_drive_gains_scaling / DegreesToRadian) = {joint_drive_gains_scaling / DegreesToRadian}, default scale for joint_drive_gains_scaling=1 is 1.0/DegreesToRadian = {1.0 / DegreesToRadian}"
        )

    # Process custom attributes defined for different kinds of prim.
    # Note that at this time we may have more custom attributes than before since they may have been
    # declared on the PhysicsScene prim.
    builder_custom_attr_shape: list[ModelBuilder.CustomAttribute] = builder.get_custom_attributes_by_frequency(
        [AttributeFrequency.SHAPE]
    )
    builder_custom_attr_body: list[ModelBuilder.CustomAttribute] = builder.get_custom_attributes_by_frequency(
        [AttributeFrequency.BODY]
    )
    builder_custom_attr_joint: list[ModelBuilder.CustomAttribute] = builder.get_custom_attributes_by_frequency(
        [AttributeFrequency.JOINT, AttributeFrequency.JOINT_DOF, AttributeFrequency.JOINT_COORD]
    )
    builder_custom_attr_eq: list[ModelBuilder.CustomAttribute] = builder.get_custom_attributes_by_frequency(
        ["mujoco:equality_constraint"]
    )
    builder_custom_attr_articulation: list[ModelBuilder.CustomAttribute] = builder.get_custom_attributes_by_frequency(
        [AttributeFrequency.ARTICULATION]
    )

    if physics_scene_prim is not None:
        # Collect schema-defined attributes from the scene prim for inspection (e.g., mjc:* attributes)
        if collect_schema_attrs:
            R.collect_prim_attrs(physics_scene_prim)

        # Extract custom attributes for model (ONCE and WORLD frequency) from the PhysicsScene prim
        # WORLD frequency attributes use index 0 here; they get remapped during add_world()
        builder_custom_attr_model: list[ModelBuilder.CustomAttribute] = [
            attr
            for attr in builder.custom_attributes.values()
            if attr.frequency in (AttributeFrequency.ONCE, AttributeFrequency.WORLD)
        ]

        # Filter out MuJoCo attributes if parse_mujoco_options is False
        if not parse_mujoco_options:
            builder_custom_attr_model = [attr for attr in builder_custom_attr_model if attr.namespace != "mujoco"]

        # Read custom attribute values from the PhysicsScene prim
        scene_custom_attrs = usd.get_custom_attribute_values(
            physics_scene_prim, builder_custom_attr_model, context={"builder": builder}
        )
        scene_attributes.update(scene_custom_attrs)

        # Set values on builder's custom attributes
        for key, value in scene_custom_attrs.items():
            if key in builder.custom_attributes:
                builder.custom_attributes[key].values[0] = value

    joint_descriptions = {}
    # stores physics spec for every RigidBody in the selected range
    body_specs = {}
    # set of prim paths of rigid bodies that are ignored
    # (to avoid repeated regex evaluations)
    ignored_body_paths = set()
    material_specs = {}
    # maps from articulation_id to list of body_ids
    articulation_bodies = {}

    # TODO: uniform interface for iterating
    def data_for_key(physics_utils_results, key):
        if key not in physics_utils_results:
            return
        if verbose:
            print(physics_utils_results[key])

        yield from zip(*physics_utils_results[key], strict=False)

    # Setting up the default material
    material_specs[""] = PhysicsMaterial()

    def warn_invalid_desc(path, descriptor) -> bool:
        if not descriptor.isValid:
            warnings.warn(
                f'Warning: Invalid {type(descriptor).__name__} descriptor for prim at path "{path}".',
                stacklevel=2,
            )
            return True
        return False

    # Parsing physics materials from the stage
    for sdf_path, desc in data_for_key(ret_dict, UsdPhysics.ObjectType.RigidBodyMaterial):
        if warn_invalid_desc(sdf_path, desc):
            continue
        prim = stage.GetPrimAtPath(sdf_path)

        def _resolve_contact_attr(key, _prim=prim):
            val = R.get_value(_prim, prim_type=PrimType.MATERIAL, key=key, verbose=verbose)
            if val is None:
                return None
            return float(val)

        material_specs[str(sdf_path)] = PhysicsMaterial(
            staticFriction=desc.staticFriction,
            dynamicFriction=desc.dynamicFriction,
            restitution=desc.restitution,
            torsionalFriction=R.get_value(
                prim,
                prim_type=PrimType.MATERIAL,
                key="mu_torsional",
                default=builder.default_shape_cfg.mu_torsional,
                verbose=verbose,
            ),
            rollingFriction=R.get_value(
                prim,
                prim_type=PrimType.MATERIAL,
                key="mu_rolling",
                default=builder.default_shape_cfg.mu_rolling,
                verbose=verbose,
            ),
            # Treat non-positive/unauthored material density as "use importer default".
            # Authored collider/body MassAPI mass+inertia is handled later.
            density=desc.density if desc.density > 0.0 else default_shape_density,
            ke=_resolve_contact_attr("ke"),
            kd=_resolve_contact_attr("kd"),
            kf=_resolve_contact_attr("kf"),
            ka=_resolve_contact_attr("ka"),
        )

    if UsdPhysics.ObjectType.RigidBody in ret_dict:
        prim_paths, rigid_body_descs = ret_dict[UsdPhysics.ObjectType.RigidBody]
        for prim_path, rigid_body_desc in zip(prim_paths, rigid_body_descs, strict=False):
            if warn_invalid_desc(prim_path, rigid_body_desc):
                continue
            body_path = str(prim_path)
            if any(re.match(p, body_path) for p in ignore_paths):
                ignored_body_paths.add(body_path)
                continue
            body_specs[body_path] = rigid_body_desc
            prim = stage.GetPrimAtPath(prim_path)

    # Bodies with MassAPI that need ComputeMassProperties fallback (missing mass, inertia, or CoM).
    bodies_requiring_mass_properties_fallback: set[str] = set()
    if UsdPhysics.ObjectType.RigidBody in ret_dict:
        prim_paths, rigid_body_descs = ret_dict[UsdPhysics.ObjectType.RigidBody]
        for prim_path, rigid_body_desc in zip(prim_paths, rigid_body_descs, strict=False):
            if warn_invalid_desc(prim_path, rigid_body_desc):
                continue
            body_path = str(prim_path)
            if body_path in ignored_body_paths:
                continue

            prim = stage.GetPrimAtPath(prim_path)
            if not prim.HasAPI(UsdPhysics.MassAPI):
                continue

            mass_api = UsdPhysics.MassAPI(prim)
            has_authored_mass = mass_api.GetMassAttr().HasAuthoredValue()
            has_authored_inertia = mass_api.GetDiagonalInertiaAttr().HasAuthoredValue()
            has_authored_com = mass_api.GetCenterOfMassAttr().HasAuthoredValue()
            if not (has_authored_mass and has_authored_inertia and has_authored_com):
                bodies_requiring_mass_properties_fallback.add(body_path)

    # Collect joint descriptions regardless of whether articulations are authored.
    for key, value in ret_dict.items():
        if key in {
            UsdPhysics.ObjectType.FixedJoint,
            UsdPhysics.ObjectType.RevoluteJoint,
            UsdPhysics.ObjectType.PrismaticJoint,
            UsdPhysics.ObjectType.SphericalJoint,
            UsdPhysics.ObjectType.D6Joint,
            UsdPhysics.ObjectType.DistanceJoint,
        }:
            paths, joint_specs = value
            for path, joint_spec in zip(paths, joint_specs, strict=False):
                joint_descriptions[str(path)] = joint_spec

    mjc_equality_connect_paths: set[str] = set()
    mjc_equality_weld_paths: set[str] = set()
    for joint_path in joint_descriptions:
        joint_prim = stage.GetPrimAtPath(joint_path)
        if _has_api_schema(joint_prim, "MjcEqualityConnectAPI"):
            mjc_equality_connect_paths.add(joint_path)
        if _has_api_schema(joint_prim, "MjcEqualityWeldAPI"):
            mjc_equality_weld_paths.add(joint_path)
    mjc_equality_connect_or_weld_paths = mjc_equality_connect_paths | mjc_equality_weld_paths

    # Track which joints have been processed during articulation parsing.
    # This allows us to parse orphan joints (joints not included in any articulation)
    # even when articulations are present in the USD.
    processed_joints: set[str] = set()
    authored_articulation_root_paths = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path), Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    authored_articulation_root_paths.sort(key=len, reverse=True)

    # maps from articulation_id to bool indicating if self-collisions are enabled
    articulation_has_self_collision = {}

    if UsdPhysics.ObjectType.Articulation in ret_dict:
        paths, articulation_descs = ret_dict[UsdPhysics.ObjectType.Articulation]

        articulation_id = builder.articulation_count
        parent_prim = None
        body_data = {}
        for path, desc in zip(paths, articulation_descs, strict=False):
            if warn_invalid_desc(path, desc):
                continue
            articulation_path = str(path)
            if any(re.match(p, articulation_path) for p in ignore_paths):
                continue
            articulation_prim = stage.GetPrimAtPath(path)
            articulation_root_xform = usd.get_transform(articulation_prim, local=False, xform_cache=xform_cache)
            root_joint_xform = (
                incoming_world_xform if override_root_xform else incoming_world_xform * articulation_root_xform
            )
            # Collect engine-specific attributes for the articulation root on first encounter
            if collect_schema_attrs:
                R.collect_prim_attrs(articulation_prim)
                # Also collect on the parent prim (e.g. Xform with PhysxArticulationAPI)
                try:
                    parent_prim = articulation_prim.GetParent()
                except Exception:
                    parent_prim = None
                if parent_prim is not None and parent_prim.IsValid():
                    R.collect_prim_attrs(parent_prim)

            # Extract custom attributes for articulation frequency from the articulation root prim
            # (the one with PhysicsArticulationRootAPI, typically the articulation_prim itself or its parent)
            articulation_custom_attrs = {}
            # First check if articulation_prim itself has the PhysicsArticulationRootAPI
            if articulation_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                if verbose:
                    print(f"Extracting articulation custom attributes from {articulation_prim.GetPath()}")
                articulation_custom_attrs = usd.get_custom_attribute_values(
                    articulation_prim, builder_custom_attr_articulation
                )
            # If not, check the parent prim
            elif (
                parent_prim is not None and parent_prim.IsValid() and parent_prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            ):
                if verbose:
                    print(f"Extracting articulation custom attributes from parent {parent_prim.GetPath()}")
                articulation_custom_attrs = usd.get_custom_attribute_values(
                    parent_prim, builder_custom_attr_articulation
                )
            if verbose and articulation_custom_attrs:
                print(f"Extracted articulation custom attributes: {articulation_custom_attrs}")
            body_ids = {}
            body_labels = []
            current_body_id = 0
            art_bodies = []
            if verbose:
                print(f"Bodies under articulation {path!s}:")
            for p in desc.articulatedBodies:
                if verbose:
                    print(f"\t{p!s}")
                if p == Sdf.Path.emptyPath:
                    continue
                key = str(p)
                if key in ignored_body_paths:
                    continue

                usd_prim = stage.GetPrimAtPath(p)
                if collect_schema_attrs:
                    # Collect on each articulated body prim encountered
                    R.collect_prim_attrs(usd_prim)

                if key in body_specs:
                    body_desc = body_specs[key]
                    desc_xform = wp.transform(body_desc.position, usd.value_to_warp(body_desc.rotation))
                    body_world = usd.get_transform(usd_prim, local=False, xform_cache=xform_cache)
                    if override_root_xform:
                        # Strip the articulation root's world-space pose and rebase at the user-specified xform.
                        body_in_root_frame = wp.transform_inverse(articulation_root_xform) * body_world
                        desired_world = incoming_world_xform * body_in_root_frame
                    else:
                        desired_world = incoming_world_xform * body_world
                    body_incoming_xform = desired_world * wp.transform_inverse(desc_xform)
                    art_root_for_visuals = articulation_root_xform if override_root_xform else None
                    if bodies_follow_joint_ordering:
                        # we just parse the body information without yet adding it to the builder
                        body_data[current_body_id] = parse_body(
                            body_desc,
                            stage.GetPrimAtPath(p),
                            incoming_xform=body_incoming_xform,
                            add_body_to_builder=False,
                            articulation_root_xform=art_root_for_visuals,
                        )
                    else:
                        # look up description and add body to builder
                        bid: int = parse_body(  # pyright: ignore[reportAssignmentType]
                            body_desc,
                            stage.GetPrimAtPath(p),
                            incoming_xform=body_incoming_xform,
                            add_body_to_builder=True,
                            articulation_root_xform=art_root_for_visuals,
                        )
                        if bid >= 0:
                            art_bodies.append(bid)
                    # remove body spec once we inserted it
                    del body_specs[key]

                body_ids[key] = current_body_id
                body_labels.append(key)
                current_body_id += 1

            if len(body_ids) == 0:
                # no bodies under the articulation or we ignored all of them
                continue

            # determine the joint graph for this articulation
            joint_names: list[str] = []
            joint_edges: list[tuple[int, int]] = []
            # keys of joints that are excluded from the articulation (loop joints)
            joint_excluded: set[str] = set()
            # Groups of joints that share the same body pair (multi-DOF joints from MuJoCo USD).
            # Maps the representative joint path (first encountered) to all joint paths in the group.
            merged_joint_groups: dict[str, list[str]] = {}
            # Track which body pair maps to which representative joint path
            body_pair_to_representative: dict[tuple[int, int], str] = {}
            for p in desc.articulatedJoints:
                joint_path = str(p)
                joint_desc = joint_descriptions[joint_path]
                if joint_path in mjc_equality_connect_or_weld_paths:
                    if verbose:
                        print(f"Skipping equality connect/weld joint '{joint_path}' from articulation joint graph")
                    continue
                # it may be possible that a joint is filtered out in the middle of
                # a chain of joints, which results in a disconnected graph
                # we should raise an error in this case
                if any(re.match(p, joint_path) for p in ignore_paths):
                    continue
                if str(joint_desc.body0) in ignored_body_paths:
                    continue
                if str(joint_desc.body1) in ignored_body_paths:
                    continue
                parent_id, child_id = resolve_joint_parent_child(joint_desc, body_ids, get_transforms=False)  # pyright: ignore[reportAssignmentType]
                if joint_desc.excludeFromArticulation:
                    joint_excluded.add(joint_path)
                else:
                    body_pair = (parent_id, child_id)
                    if body_pair in body_pair_to_representative:
                        # Another joint between the same bodies — merge into existing group
                        rep = body_pair_to_representative[body_pair]
                        merged_joint_groups[rep].append(joint_path)
                    else:
                        # First joint for this body pair
                        body_pair_to_representative[body_pair] = joint_path
                        merged_joint_groups[joint_path] = [joint_path]
                        joint_edges.append(body_pair)
                        joint_names.append(joint_path)

            articulation_joint_indices = []

            if len(joint_edges) == 0:
                # We have an articulation without joints, i.e. only free rigid bodies
                # Use add_base_joint to honor floating, base_joint, and parent_body parameters
                base_parent = parent_body
                if bodies_follow_joint_ordering:
                    for i in body_ids.values():
                        child_body_id = add_body(**body_data[i])
                        # Compute parent_xform to preserve imported pose when attaching to parent_body
                        parent_xform = None
                        if base_parent != -1:
                            # When parent_body is specified, interpret xform parameter as parent-relative offset
                            # body_data[i]["xform"] = USD_local * incoming_world_xform
                            # We want parent_xform to position the child at this location relative to parent
                            # Use incoming_world_xform as the base parent-relative offset
                            parent_xform = incoming_world_xform
                            # If the USD body has a non-identity local transform, compose it with incoming_xform
                            # Note: incoming_world_xform already includes the child's USD local transform via body_incoming_xform
                            # So we can use body_data[i]["xform"] directly for the intended position
                            # But we need it relative to parent. Since parent's body_q may not reflect joint offsets,
                            # we interpret body_data[i]["xform"] as the intended parent-relative transform directly.
                            # For articulations without joints, incoming_world_xform IS the parent-relative offset.
                            parent_xform = incoming_world_xform
                        joint_id = builder._add_base_joint(
                            child_body_id,
                            floating=floating,
                            base_joint=base_joint,
                            parent=base_parent,
                            parent_xform=parent_xform,
                        )
                        # note the free joint's coordinates will be initialized by the body_q of the
                        # child body
                        builder._finalize_imported_articulation(
                            joint_indices=[joint_id],
                            parent_body=parent_body,
                            articulation_label=body_data[i]["label"],
                            custom_attributes=articulation_custom_attrs,
                        )
                else:
                    for i, child_body_id in enumerate(art_bodies):
                        # Compute parent_xform to preserve imported pose when attaching to parent_body
                        parent_xform = None
                        if base_parent != -1:
                            # When parent_body is specified, interpret xform parameter as parent-relative offset
                            parent_xform = incoming_world_xform
                        joint_id = builder._add_base_joint(
                            child_body_id,
                            floating=floating,
                            base_joint=base_joint,
                            parent=base_parent,
                            parent_xform=parent_xform,
                        )
                        # note the free joint's coordinates will be initialized by the body_q of the
                        # child body
                        builder._finalize_imported_articulation(
                            joint_indices=[joint_id],
                            parent_body=parent_body,
                            articulation_label=body_labels[i],
                            custom_attributes=articulation_custom_attrs,
                        )
                sorted_joints = []
            else:
                # we have an articulation with joints, we need to sort them topologically
                if joint_ordering is not None:
                    if verbose:
                        print(f"Sorting joints using {joint_ordering} ordering...")
                    sorted_joints, reversed_joint_list = topological_sort_undirected(
                        joint_edges, use_dfs=joint_ordering == "dfs", ensure_single_root=True
                    )
                    if reversed_joint_list:
                        reversed_joint_paths = [joint_names[joint_id] for joint_id in reversed_joint_list]
                        reversed_joint_names = ", ".join(reversed_joint_paths)
                        raise ValueError(
                            f"Reversed joints are not supported: {reversed_joint_names}. Ensure that the joint parent body is defined as physics:body0 and the child is defined as physics:body1 in the joint prim."
                        )
                    if verbose:
                        print("Joint ordering:", sorted_joints)
                else:
                    # we keep the original order of the joints
                    sorted_joints = np.arange(len(joint_names))

            if len(sorted_joints) > 0:
                # insert the bodies in the order of the joints
                if bodies_follow_joint_ordering:
                    inserted_bodies = set()
                    for jid in sorted_joints:
                        parent, child = joint_edges[jid]
                        if parent >= 0 and parent not in inserted_bodies:
                            b = add_body(**body_data[parent])
                            inserted_bodies.add(parent)
                            art_bodies.append(b)
                            path_body_map[body_data[parent]["label"]] = b
                        if child >= 0 and child not in inserted_bodies:
                            b = add_body(**body_data[child])
                            inserted_bodies.add(child)
                            art_bodies.append(b)
                            path_body_map[body_data[child]["label"]] = b

                first_joint_parent = joint_edges[sorted_joints[0]][0]
                if first_joint_parent != -1:
                    # the mechanism is floating since there is no joint connecting it to the world
                    # we explicitly add a joint connecting the first body in the articulation to the world
                    # (or to parent_body if specified) to make sure generalized-coordinate solvers can simulate it
                    base_parent = parent_body
                    if bodies_follow_joint_ordering:
                        child_body = body_data[first_joint_parent]
                        child_body_id = path_body_map[child_body["label"]]
                    else:
                        child_body_id = art_bodies[first_joint_parent]
                    # Compute parent_xform to preserve imported pose when attaching to parent_body
                    parent_xform = None
                    if base_parent != -1:
                        # When parent_body is specified, use incoming_world_xform as parent-relative offset
                        parent_xform = incoming_world_xform
                    base_joint_id = builder._add_base_joint(
                        child_body_id,
                        floating=floating,
                        base_joint=base_joint,
                        parent=base_parent,
                        parent_xform=parent_xform,
                    )
                    articulation_joint_indices.append(base_joint_id)

                # insert the remaining joints in topological order
                for joint_id, i in enumerate(sorted_joints):
                    if joint_id == 0 and first_joint_parent == -1:
                        # The root joint connects to the world (parent_id=-1).
                        # If base_joint or floating is specified, override the USD's root joint.
                        if base_joint is not None or floating is not None:
                            # Get the child body of the root joint
                            root_joint_child = joint_edges[sorted_joints[0]][1]
                            if bodies_follow_joint_ordering:
                                child_body = body_data[root_joint_child]
                                child_body_id = path_body_map[child_body["label"]]
                            else:
                                child_body_id = art_bodies[root_joint_child]
                            base_parent = parent_body
                            # Compute parent_xform to preserve imported pose
                            parent_xform = None
                            if base_parent != -1:
                                # When parent_body is specified, use incoming_world_xform as parent-relative offset
                                parent_xform = incoming_world_xform
                            else:
                                # body_q is already in world space, use it directly
                                parent_xform = builder.body_q[child_body_id]
                            base_joint_id = builder._add_base_joint(
                                child_body_id,
                                floating=floating,
                                base_joint=base_joint,
                                parent=base_parent,
                                parent_xform=parent_xform,
                            )
                            articulation_joint_indices.append(base_joint_id)
                            group = merged_joint_groups.get(joint_names[i])
                            if group is not None:
                                processed_joints.update(group)
                            else:
                                processed_joints.add(joint_names[i])
                            continue  # Skip parsing the USD's root joint
                        # When body0 maps to world the physics API may resolve
                        # localPose0 into world space (baking the non-body prim's
                        # transform). JointDesc.body0 returns "" for non-rigid
                        # targets, so we attempt to look up the prim directly.
                        root_joint_desc = joint_descriptions[joint_names[i]]
                        b0 = str(root_joint_desc.body0)
                        b1 = str(root_joint_desc.body1)
                        # Determine the world-facing side from this articulation's body set.
                        # path_body_map includes previously imported articulations, so using
                        # it here can misidentify the world-side path for the current root
                        # joint when b0 references an external rigid body.
                        if b0 not in body_ids:
                            world_body_path = b0
                        elif b1 not in body_ids:
                            world_body_path = b1
                        else:
                            # Defensive fallback; root joints should have exactly one side
                            # outside the articulation.
                            world_body_path = b0
                        world_body_prim = stage.GetPrimAtPath(world_body_path) if world_body_path else None
                        if world_body_prim is not None and world_body_prim.IsValid():
                            world_body_xform = usd.get_transform(world_body_prim, local=False, xform_cache=xform_cache)
                        else:
                            # body0/body1 can resolve to world with an empty path (""),
                            # leaving no world-side prim to query.
                            # If the authored world-side local pose is identity, recover
                            # the missing world-side frame from the resolved child body
                            # pose and local poses so root-joint FK stays consistent with
                            # imported body_q.
                            # If the world-side local pose is non-identity, keep the
                            # previous identity fallback: USD often bakes non-rigid world
                            # anchors directly into localPose0/localPose1 in this case.
                            _, child_local_id, parent_tf, child_tf = resolve_joint_parent_child(  # pyright: ignore[reportAssignmentType]
                                root_joint_desc,
                                body_ids,
                                get_transforms=True,
                            )
                            assert parent_tf is not None and child_tf is not None
                            identity_tf = wp.transform_identity()
                            parent_pos = np.array(parent_tf.p, dtype=float)
                            parent_quat = np.array(parent_tf.q, dtype=float)
                            identity_pos = np.array(identity_tf.p, dtype=float)
                            identity_quat = np.array(identity_tf.q, dtype=float)
                            parent_pos_is_identity = np.allclose(parent_pos, identity_pos, atol=1e-6)
                            # q and -q represent the same rotation
                            parent_rot_is_identity = abs(np.dot(parent_quat, identity_quat)) > 1.0 - 1e-6
                            if (
                                parent_pos_is_identity
                                and parent_rot_is_identity
                                and 0 <= child_local_id < len(body_labels)
                            ):
                                child_path = body_labels[child_local_id]
                                child_prim = stage.GetPrimAtPath(child_path)
                            else:
                                child_prim = None
                            if child_prim is not None and child_prim.IsValid():
                                child_world_xform = usd.get_transform(child_prim, local=False, xform_cache=xform_cache)
                                world_body_xform = child_world_xform * child_tf * wp.transform_inverse(parent_tf)
                            else:
                                world_body_xform = wp.transform_identity()
                        root_frame_xform = (
                            wp.transform_inverse(articulation_root_xform)
                            if override_root_xform
                            else wp.transform_identity()
                        )
                        root_incoming_xform = incoming_world_xform * root_frame_xform * world_body_xform
                        group = merged_joint_groups.get(joint_names[i])
                        if group is not None and len(group) > 1:
                            joint = parse_merged_joints(group, incoming_xform=root_incoming_xform)
                        else:
                            joint = parse_joint(
                                joint_descriptions[joint_names[i]],
                                incoming_xform=root_incoming_xform,
                            )
                    else:
                        group = merged_joint_groups.get(joint_names[i])
                        if group is not None and len(group) > 1:
                            joint = parse_merged_joints(group)
                        else:
                            joint = parse_joint(
                                joint_descriptions[joint_names[i]],
                            )
                    if joint is not None:
                        articulation_joint_indices.append(joint)
                        processed_joints.add(joint_names[i])
                        # Mark all paths in the group as processed
                        group = merged_joint_groups.get(joint_names[i])
                        if group is not None:
                            for gp in group:
                                processed_joints.add(gp)

                # insert loop joints
                for joint_path in joint_excluded:
                    parent_id, _ = resolve_joint_parent_child(
                        joint_descriptions[joint_path], path_body_map, get_transforms=False
                    )
                    if parent_id == -1:
                        joint = parse_joint(
                            joint_descriptions[joint_path],
                            incoming_xform=root_joint_xform,
                        )
                    else:
                        # localPose0 is already in the parent body's local frame;
                        # body positions were correctly set during body parsing above.
                        joint = parse_joint(
                            joint_descriptions[joint_path],
                        )
                    if joint is not None:
                        processed_joints.add(joint_path)

            # Create the articulation from all collected joints
            if articulation_joint_indices:
                builder._finalize_imported_articulation(
                    joint_indices=articulation_joint_indices,
                    parent_body=parent_body,
                    articulation_label=articulation_path,
                    custom_attributes=articulation_custom_attrs,
                )

            articulation_bodies[articulation_id] = art_bodies
            articulation_has_self_collision[articulation_id] = bool(
                R.get_value(
                    articulation_prim,
                    prim_type=PrimType.ARTICULATION,
                    key="self_collision_enabled",
                    default=enable_self_collisions,
                    verbose=verbose,
                )
            )
            articulation_id += 1
    no_articulations = UsdPhysics.ObjectType.Articulation not in ret_dict
    has_joints = any(
        (
            not (only_load_enabled_joints and not joint_desc.jointEnabled)
            and not any(re.match(p, joint_path) for p in ignore_paths)
            and str(joint_desc.body0) not in ignored_body_paths
            and str(joint_desc.body1) not in ignored_body_paths
            and joint_path not in mjc_equality_connect_or_weld_paths
        )
        for joint_path, joint_desc in joint_descriptions.items()
    )

    # insert remaining bodies that were not part of any articulation so far
    # (root joints for these bodies will be added after mass properties are resolved)
    for path, rigid_body_desc in body_specs.items():
        key = str(path)
        body_id: int = parse_body(  # pyright: ignore[reportAssignmentType]
            rigid_body_desc,
            stage.GetPrimAtPath(path),
            incoming_xform=incoming_world_xform,
            add_body_to_builder=True,
        )

    # Parse orphan joints: joints that exist in the USD but were not included in any articulation.
    # This can happen when:
    # 1. No articulations are defined in the USD (no_articulations == True)
    # 2. A joint connects bodies that are not under any PhysicsArticulationRootAPI
    orphan_joints_by_body_pair: dict[tuple[str, str], list[str]] = {}
    for joint_path, joint_desc in joint_descriptions.items():
        # Earlier passes already own articulation and equality joints.
        if joint_path in processed_joints:
            continue
        if joint_path in mjc_equality_connect_or_weld_paths:
            if verbose:
                print(f"Skipping equality connect/weld joint '{joint_path}' from orphan joint parsing")
            continue

        # Apply the importer filters before grouping the remaining candidates.
        if only_load_enabled_joints and not joint_desc.jointEnabled:
            continue
        if any(re.match(p, joint_path) for p in ignore_paths):
            continue
        if str(joint_desc.body0) in ignored_body_paths or str(joint_desc.body1) in ignored_body_paths:
            continue

        # Shared endpoints identify joints that may form one compound joint.
        body_pair = (str(joint_desc.body0), str(joint_desc.body1))
        orphan_joints_by_body_pair.setdefault(body_pair, []).append(joint_path)

    # A multi-axis D6 joint may be authored as stacked 1-DOF joints between the same bodies.
    mergeable_joint_types = {UsdPhysics.ObjectType.RevoluteJoint, UsdPhysics.ObjectType.PrismaticJoint}
    orphan_joint_groups: list[list[str]] = []
    for joint_group in orphan_joints_by_body_pair.values():
        if len(joint_group) > 1 and all(
            joint_descriptions[joint_path].type in mergeable_joint_types for joint_path in joint_group
        ):
            orphan_joint_groups.append(joint_group)
        else:
            # Normalize non-mergeable joints to singleton groups for the parsing pass below.
            orphan_joint_groups.extend([[joint_path] for joint_path in joint_group])

    for joint_group in orphan_joint_groups:
        # All members of a merged group share these endpoints, so the first is representative.
        joint_path = joint_group[0]
        joint_desc = joint_descriptions[joint_path]
        body0_path = str(joint_desc.body0)
        body1_path = str(joint_desc.body1)
        # World-connected joints need a reconstructed parent frame before they can be parsed.
        is_body_to_world = body0_path in ("", "/") or body1_path in ("", "/")
        try:
            # Body-to-world joints (the world side may be body0 or body1) have no
            # world-side prim to inherit a frame from, and authoring tools often
            # write the world-side localPose relative to a USD ancestor Xform
            # instead of in world coords. Recover the missing world-side frame from
            # the child body's world pose and the joint local poses so the joint
            # chain FK reproduces the imported child world pose:
            #   world_body = child_world * child_tf * inv(parent_tf)
            # The world-side localPose cancels, so the joint anchors at the
            # USD-authored child body pose however that pose was authored.
            orphan_incoming_xform = incoming_world_xform
            if is_body_to_world:
                _, _, parent_tf_o, child_tf_o = resolve_joint_parent_child(  # pyright: ignore[reportAssignmentType]
                    joint_desc, path_body_map, get_transforms=True
                )
                child_path_o = body1_path if body0_path in ("", "/") else body0_path
                child_prim_o = stage.GetPrimAtPath(child_path_o) if child_path_o else None
                if (
                    parent_tf_o is not None
                    and child_tf_o is not None
                    and child_prim_o is not None
                    and child_prim_o.IsValid()
                ):
                    child_world_xform_o = usd.get_transform(child_prim_o, local=False, xform_cache=xform_cache)
                    world_body_xform_o = child_world_xform_o * child_tf_o * wp.transform_inverse(parent_tf_o)
                    orphan_incoming_xform = incoming_world_xform * world_body_xform_o
            if len(joint_group) > 1:
                parse_merged_joints(joint_group, incoming_xform=orphan_incoming_xform)
            else:
                parse_joint(joint_desc, incoming_xform=orphan_incoming_xform)
        except ValueError as exc:
            if verbose:
                print(f"Skipping joint group {joint_group}: {exc}")

    def _build_mass_info_from_authored_properties(
        prim: Usd.Prim,
        local_pos,
        local_rot,
        shape_geo_type: int,
        shape_scale: wp.vec3,
        shape_src: Mesh | None,
        shape_axis=None,
    ):
        """Build unit-density collider mass information from authored collider MassAPI properties.

        This helper is used for rigid-body fallback mass aggregation via
        ``UsdPhysics.RigidBodyAPI.ComputeMassProperties``. When a collider prim has authored
        ``MassAPI`` mass and diagonal inertia, we convert those values into a
        ``RigidBodyAPI.MassInformation`` payload that represents unit-density collider properties.
        """
        if not prim.HasAPI(UsdPhysics.MassAPI):
            return None

        mass_api = UsdPhysics.MassAPI(prim)
        mass_attr = mass_api.GetMassAttr()
        diag_attr = mass_api.GetDiagonalInertiaAttr()
        if not (mass_attr.HasAuthoredValue() and diag_attr.HasAuthoredValue()):
            return None

        mass = float(mass_attr.Get())
        if mass <= 0.0:
            warnings.warn(
                f"Skipping collider {prim.GetPath()}: authored MassAPI mass must be > 0 to derive volume and density.",
                stacklevel=2,
            )
            return None

        shape_volume, _, _ = compute_inertia_shape(shape_geo_type, shape_scale, shape_src, density=1.0)
        if shape_volume <= 0.0:
            warnings.warn(
                f"Skipping collider {prim.GetPath()}: unable to derive positive collider volume from authored shape parameters.",
                stacklevel=2,
            )
            return None
        density = mass / shape_volume
        if density <= 0.0:
            warnings.warn(
                f"Skipping collider {prim.GetPath()}: derived density from authored mass is non-positive.",
                stacklevel=2,
            )
            return None

        diag = np.array(diag_attr.Get(), dtype=np.float32)
        if np.any(diag < 0.0):
            warnings.warn(
                f"Skipping collider {prim.GetPath()}: authored diagonal inertia contains negative values.",
                stacklevel=2,
            )
            return None
        inertia_diag_unit = diag / density

        if mass_api.GetPrincipalAxesAttr().HasAuthoredValue():
            principal_axes = mass_api.GetPrincipalAxesAttr().Get()
        else:
            principal_axes = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
        if mass_api.GetCenterOfMassAttr().HasAuthoredValue():
            center_of_mass = mass_api.GetCenterOfMassAttr().Get()
        else:
            center_of_mass = Gf.Vec3f(0.0, 0.0, 0.0)

        i_rot = usd.value_to_warp(principal_axes)
        rot = np.array(wp.quat_to_matrix(i_rot), dtype=np.float32).reshape(3, 3)
        inertia_full_unit = rot @ np.diag(inertia_diag_unit) @ rot.T

        mass_info = UsdPhysics.RigidBodyAPI.MassInformation()
        mass_info.volume = float(shape_volume)
        mass_info.centerOfMass = center_of_mass
        mass_info.localPos = Gf.Vec3f(*local_pos)
        mass_info.localRot = _resolve_mass_info_local_rotation(local_rot, shape_geo_type, shape_axis)
        mass_info.inertia = Gf.Matrix3f(*inertia_full_unit.flatten().tolist())
        return mass_info

    def _resolve_mass_info_local_rotation(local_rot, shape_geo_type: int, shape_axis):
        """Match collider mass frame rotation with shape axis correction used by shape insertion."""
        if shape_geo_type not in {GeoType.CAPSULE, GeoType.CYLINDER, GeoType.CONE} or shape_axis is None:
            return local_rot

        axis = usd_axis_to_axis.get(shape_axis)
        if axis is None:
            axis_int_map = {
                int(UsdPhysics.Axis.X): Axis.X,
                int(UsdPhysics.Axis.Y): Axis.Y,
                int(UsdPhysics.Axis.Z): Axis.Z,
            }
            axis = axis_int_map.get(int(shape_axis))
        if axis is None or axis == Axis.Z:
            return local_rot

        local_rot_wp = usd.value_to_warp(local_rot)
        corrected_rot = wp.mul(local_rot_wp, quat_between_axes(Axis.Z, axis))
        return Gf.Quatf(
            float(corrected_rot[3]),
            float(corrected_rot[0]),
            float(corrected_rot[1]),
            float(corrected_rot[2]),
        )

    def _build_mass_info_from_shape_geometry(
        prim: Usd.Prim,
        local_pos,
        local_rot,
        shape_geo_type: int,
        shape_scale: wp.vec3,
        shape_src: Mesh | None,
        shape_axis=None,
        is_solid: bool = True,
        thickness: float = 0.0,
    ):
        """Build unit-density collider mass information from geometric shape parameters.

        This fallback path derives collider volume, center of mass, and inertia from shape
        geometry (box/sphere/capsule/cylinder/cone/mesh) when collider-authored MassAPI mass
        properties are not available.
        """
        shape_mass, shape_com, shape_inertia = compute_inertia_shape(
            shape_geo_type, shape_scale, shape_src, density=1.0, is_solid=is_solid, thickness=thickness
        )
        if shape_mass <= 0.0:
            warnings.warn(
                f"Skipping collider {prim.GetPath()} in mass aggregation: unable to derive positive unit-density mass.",
                stacklevel=2,
            )
            return None

        shape_inertia_np = np.array(shape_inertia, dtype=np.float32).reshape(3, 3)
        mass_info = UsdPhysics.RigidBodyAPI.MassInformation()
        mass_info.volume = float(shape_mass)
        mass_info.centerOfMass = Gf.Vec3f(*shape_com)
        mass_info.localPos = Gf.Vec3f(*local_pos)
        mass_info.localRot = _resolve_mass_info_local_rotation(local_rot, shape_geo_type, shape_axis)
        mass_info.inertia = Gf.Matrix3f(*shape_inertia_np.flatten().tolist())
        return mass_info

    # parse shapes attached to the rigid bodies
    # Canonicalized (sorted) USD path pairs from physics:filteredPairs. Collected from native
    # colliders and deformable participants, applied only after deformable lowering so every
    # endpoint's Newton shapes exist (a cable maps to several capsule shapes created late).
    authored_filtered_path_pairs: set[tuple[str, str]] = set()

    def _collect_filtered_pairs(prim):
        if not prim.HasRelationship("physics:filteredPairs"):
            return
        src = str(prim.GetPath())
        for target in prim.GetRelationship("physics:filteredPairs").GetTargets():
            dst = str(target)
            # The relationship may be authored on either or both endpoints and Newton's
            # filter pair is symmetric; canonicalizing dedups both. A self-pair is invalid.
            if src != dst:
                authored_filtered_path_pairs.add((src, dst) if src < dst else (dst, src))

    no_collision_shapes = set()
    collision_group_ids = {}
    rigid_body_mass_info_map = {}
    expected_fallback_collider_paths: set[str] = set()
    for key, value in ret_dict.items():
        if key in {
            UsdPhysics.ObjectType.CubeShape,
            UsdPhysics.ObjectType.SphereShape,
            UsdPhysics.ObjectType.CapsuleShape,
            UsdPhysics.ObjectType.CylinderShape,
            UsdPhysics.ObjectType.ConeShape,
            UsdPhysics.ObjectType.MeshShape,
            UsdPhysics.ObjectType.PlaneShape,
        }:
            paths, shape_specs = value
            for xpath, shape_spec in zip(paths, shape_specs, strict=False):
                if warn_invalid_desc(xpath, shape_spec):
                    continue
                path = str(xpath)
                if any(re.match(p, path) for p in ignore_paths):
                    continue
                prim = stage.GetPrimAtPath(xpath)
                # Deformable-owned meshes never reach this loop: the scout excludes them
                # from the native parse. A sim-API mesh seen here was deliberately left
                # rigid (e.g. its body API conflicts with RigidBodyAPI), so import it.
                if path in path_shape_map:
                    if verbose:
                        print(f"Shape at {path} already added, skipping.")
                    continue
                body_path = str(shape_spec.rigidBody)
                if verbose:
                    print(f"collision shape {prim.GetPath()} ({prim.GetTypeName()}), body = {body_path}")
                body_id = path_body_map.get(body_path, -1)
                scale = usd.get_scale(prim, local=False)
                collision_group = builder.default_shape_cfg.collision_group

                if len(shape_spec.collisionGroups) > 0:
                    cgroup_name = str(shape_spec.collisionGroups[0])
                    if cgroup_name not in collision_group_ids:
                        # Start from 1 to avoid collision_group = 0 (which means "no collisions")
                        collision_group_ids[cgroup_name] = len(collision_group_ids) + 1
                    collision_group = collision_group_ids[cgroup_name]
                material = material_specs[""]
                has_shape_material = len(shape_spec.materials) >= 1
                if has_shape_material:
                    if len(shape_spec.materials) > 1 and verbose:
                        print(f"Warning: More than one material found on shape at '{path}'.\nUsing only the first one.")
                    material = material_specs[str(shape_spec.materials[0])]
                    if verbose:
                        print(
                            f"\tMaterial of '{path}':\tfriction: {material.dynamicFriction},\ttorsional friction: {material.torsionalFriction},\trolling friction: {material.rollingFriction},\trestitution: {material.restitution},\tdensity: {material.density}"
                        )
                elif verbose:
                    print(f"No material found for shape at '{path}'.")

                # Non-MassAPI body mass accumulation in ModelBuilder uses shape cfg density.
                # Use per-shape physics material density when present; otherwise use default density.
                if has_shape_material:
                    shape_density = material.density
                else:
                    shape_density = default_shape_density
                local_xform = wp.transform(shape_spec.localPos, usd.value_to_warp(shape_spec.localRot))
                if body_id == -1:
                    shape_xform = incoming_world_xform * local_xform
                else:
                    shape_xform = local_xform
                # Extract custom attributes for this shape
                shape_custom_attrs = usd.get_custom_attribute_values(
                    prim, builder_custom_attr_shape, context={"builder": builder}
                )
                if collect_schema_attrs:
                    R.collect_prim_attrs(prim)

                margin_val, margin_resolver = R.get_value_with_resolver(
                    prim,
                    prim_type=PrimType.SHAPE,
                    key="margin",
                    default=builder.default_shape_cfg.margin,
                    verbose=verbose,
                )
                gap_val = R.get_value(
                    prim,
                    prim_type=PrimType.SHAPE,
                    key="gap",
                    verbose=verbose,
                )
                if gap_val == float("-inf"):
                    gap_val = builder.default_shape_cfg.gap
                if legacy_margin_gap and margin_resolver is not None and margin_resolver.name == "mjc":
                    # Legacy pre-3.9 import: newton_margin = mjc_margin - mjc_gap.
                    mjc_gap = usd.get_attribute(prim, "mjc:gap")
                    mjc_gap = 0.0 if mjc_gap is None else float(mjc_gap)
                    newton_margin = float(margin_val) - mjc_gap
                    if newton_margin < 0.0:
                        warnings.warn(
                            f"Prim '{prim.GetPath()}': legacy translation yields "
                            f"negative margin (mjc_margin={margin_val}, mjc_gap={mjc_gap}).",
                            stacklevel=2,
                        )
                    margin_val = newton_margin

                has_body_visual_shapes = load_visual_shapes and body_id in bodies_with_visual_shapes
                material_props = _get_material_props_cached(prim)
                collider_has_visual_material = (
                    key == UsdPhysics.ObjectType.MeshShape and _has_visual_material_properties(material_props)
                )

                # Explicit hide_collision_shapes overrides material-based visibility:
                # if the body already has visual shapes, hide its colliders unconditionally.
                hide_collider_for_body = hide_collision_shapes and has_body_visual_shapes
                show_collider_by_policy = should_show_collider(
                    force_show_colliders,
                    has_visual_shapes=has_body_visual_shapes,
                )
                collider_is_visible = (
                    show_collider_by_policy or collider_has_visual_material
                ) and not hide_collider_for_body
                collider_is_visible = collider_is_visible and _is_effectively_visible(prim)

                # Contact response precedence:
                #   per-shape mjc:solref (non-legacy) > material > legacy per-shape > default
                _default = builder.default_shape_cfg
                mjc_has_priority = False
                for _r in R.resolvers:
                    if _r.name == "mjc":
                        mjc_has_priority = True
                        break
                    if _r.name == "newton":
                        break
                has_solref = mjc_has_priority and usd.get_attribute(prim, "mjc:solref") is not None
                shape_contact = {}
                for _ck in ("ke", "kd", "kf", "ka"):
                    per_shape_val = R.get_value(prim, prim_type=PrimType.SHAPE, key=_ck, verbose=verbose)
                    has_shape = per_shape_val is not None and math.isfinite(float(per_shape_val))
                    mat_val = getattr(material, _ck)
                    has_mat = mat_val is not None and math.isfinite(mat_val)

                    if has_solref and _ck in ("ke", "kd") and has_shape:
                        shape_contact[_ck] = float(per_shape_val)
                    elif has_mat:
                        shape_contact[_ck] = mat_val
                    elif has_shape:
                        shape_contact[_ck] = float(per_shape_val)
                    else:
                        shape_contact[_ck] = getattr(_default, _ck)
                shape_ke = shape_contact["ke"]
                shape_kd = shape_contact["kd"]
                shape_kf = shape_contact["kf"]
                shape_ka = shape_contact["ka"]

                shape_color = material_props.get("color")

                # SDF parameters. Applying NewtonSDFCollisionAPI is the canonical
                # signal that SDF generation is configured for this shape.
                has_sdf_api = prim.HasAPI("NewtonSDFCollisionAPI")
                # NewtonSDFCollisionAPI and NewtonMeshCollisionAPI are independent
                # collision representations and should not be co-applied. SDF wins
                # when both are present.
                if has_sdf_api and prim.HasAPI("NewtonMeshCollisionAPI"):
                    warnings.warn(
                        f"{prim.GetPath()}: NewtonSDFCollisionAPI and NewtonMeshCollisionAPI are "
                        f"independent collision representations and should not be co-applied; "
                        f"SDF configuration will be used.",
                        stacklevel=2,
                    )

                # Resolve target_voxel_size first because it overrides
                # sdf_max_resolution and the two are mutually exclusive in
                # ShapeConfig.validate().
                sdf_target_voxel_size = R.get_value(
                    prim, prim_type=PrimType.SHAPE, key="sdf_target_voxel_size", verbose=verbose
                )
                if sdf_target_voxel_size == float("-inf"):
                    sdf_target_voxel_size = None
                elif sdf_target_voxel_size is not None and sdf_target_voxel_size <= 0:
                    warnings.warn(
                        f"{prim.GetPath()}: newton:sdfTargetVoxelSize={sdf_target_voxel_size!r} is invalid "
                        f"(must be > 0); falling back to default.",
                        stacklevel=2,
                    )
                    sdf_target_voxel_size = None
                if sdf_target_voxel_size is None:
                    sdf_target_voxel_size = builder.default_shape_cfg.sdf_target_voxel_size

                sdf_max_resolution = R.get_value(
                    prim, prim_type=PrimType.SHAPE, key="sdf_max_resolution", verbose=verbose
                )
                if sdf_max_resolution == float("-inf"):
                    sdf_max_resolution = None
                elif sdf_max_resolution is not None and sdf_max_resolution <= 0:
                    warnings.warn(
                        f"{prim.GetPath()}: newton:sdfMaxResolution={sdf_max_resolution!r} is invalid "
                        f"(must be > 0); falling back to default.",
                        stacklevel=2,
                    )
                    sdf_max_resolution = None
                elif sdf_max_resolution is not None and sdf_max_resolution % 8 != 0:
                    warnings.warn(
                        f"{prim.GetPath()}: newton:sdfMaxResolution={sdf_max_resolution!r} must be "
                        f"divisible by 8 (SDF volumes are allocated in 8x8x8 tiles); falling back to default.",
                        stacklevel=2,
                    )
                    sdf_max_resolution = None
                if sdf_target_voxel_size is not None and sdf_max_resolution is not None:
                    warnings.warn(
                        f"{prim.GetPath()}: both newton:sdfTargetVoxelSize and newton:sdfMaxResolution "
                        f"are set; sdfTargetVoxelSize takes precedence.",
                        stacklevel=2,
                    )
                    sdf_max_resolution = None
                if sdf_max_resolution is None:
                    # When the API is applied but neither attribute is authored,
                    # fall back to the schema default (64). When target voxel
                    # size already drives the resolution, leave max_resolution
                    # unset so the two don't conflict in ShapeConfig.validate().
                    if has_sdf_api and sdf_target_voxel_size is None:
                        sdf_max_resolution = 64
                    else:
                        sdf_max_resolution = builder.default_shape_cfg.sdf_max_resolution

                sdf_narrow_band_inner = R.get_value(
                    prim, prim_type=PrimType.SHAPE, key="sdf_narrow_band_inner", verbose=verbose
                )
                if sdf_narrow_band_inner == float("-inf"):
                    sdf_narrow_band_inner = None
                sdf_narrow_band_outer = R.get_value(
                    prim, prim_type=PrimType.SHAPE, key="sdf_narrow_band_outer", verbose=verbose
                )
                if sdf_narrow_band_outer == float("-inf"):
                    sdf_narrow_band_outer = None
                default_nb = builder.default_shape_cfg.sdf_narrow_band_range
                sdf_narrow_band_range = (
                    sdf_narrow_band_inner if sdf_narrow_band_inner is not None else default_nb[0],
                    sdf_narrow_band_outer if sdf_narrow_band_outer is not None else default_nb[1],
                )

                sdf_texture_format = R.get_value(
                    prim, prim_type=PrimType.SHAPE, key="sdf_texture_format", verbose=verbose
                )
                _valid_sdf_tex_fmts = ("float32", "uint16", "uint8")
                if sdf_texture_format is not None and sdf_texture_format not in _valid_sdf_tex_fmts:
                    warnings.warn(
                        f"{prim.GetPath()}: newton:sdfTextureFormat={sdf_texture_format!r} is invalid "
                        f"(expected one of {list(_valid_sdf_tex_fmts)}); falling back to default.",
                        stacklevel=2,
                    )
                    sdf_texture_format = None
                if sdf_texture_format is None:
                    sdf_texture_format = builder.default_shape_cfg.sdf_texture_format

                sdf_padding = R.get_value(prim, prim_type=PrimType.SHAPE, key="sdf_padding", verbose=verbose)
                if sdf_padding == float("-inf"):
                    sdf_padding = None
                elif sdf_padding is not None and sdf_padding < 0:
                    warnings.warn(
                        f"{prim.GetPath()}: newton:sdfPadding={sdf_padding!r} is invalid "
                        f"(must be >= 0); falling back to default.",
                        stacklevel=2,
                    )
                    sdf_padding = None

                hydroelastic_enabled = R.get_value(
                    prim, prim_type=PrimType.SHAPE, key="hydroelastic_enabled", verbose=verbose
                )
                kh = R.get_value(prim, prim_type=PrimType.SHAPE, key="kh", verbose=verbose)
                if kh == float("-inf"):
                    kh = None
                elif kh is not None and kh <= 0:
                    warnings.warn(
                        f"{prim.GetPath()}: newton:hydroelasticStiffness={kh!r} is invalid "
                        f"(must be > 0); falling back to default.",
                        stacklevel=2,
                    )
                    kh = None
                if hydroelastic_enabled is True:
                    is_hydroelastic = True
                elif hydroelastic_enabled is False:
                    is_hydroelastic = False
                elif has_sdf_api:
                    # API applied but hydroelasticEnabled unauthored -> schema default False, not builder default.
                    is_hydroelastic = False
                else:
                    is_hydroelastic = builder.default_shape_cfg.is_hydroelastic
                if kh is None:
                    kh = builder.default_shape_cfg.kh

                # Hydroelastic meshes need an SDF source. For primitives, a texture
                # SDF is generated from a synthesized watertight mesh at finalize(),
                # but meshes require either an attached mesh.sdf or a
                # resolution/voxel_size so one can be built deferred. Warn and
                # disable hydroelastic on this shape rather than aborting the whole
                # import — typically reached when newton:hydroelasticEnabled=true
                # is authored without applying NewtonSDFCollisionAPI.
                if (
                    is_hydroelastic
                    and key == UsdPhysics.ObjectType.MeshShape
                    and sdf_max_resolution is None
                    and sdf_target_voxel_size is None
                ):
                    warnings.warn(
                        f"{prim.GetPath()}: hydroelastic mesh requires newton:sdfMaxResolution "
                        f"or newton:sdfTargetVoxelSize so an SDF can be generated; "
                        f"disabling hydroelastic for this shape.",
                        stacklevel=2,
                    )
                    is_hydroelastic = False
                # Mass model and shell thickness (resolved across Newton / MuJoCo schemas)
                mass_model = R.get_value(prim, PrimType.SHAPE, "mass_model", default="solid")
                shape_is_solid = mass_model != "shell"
                shell_thickness_val = R.get_value(prim, PrimType.SHAPE, "shell_thickness")
                # When shell thickness is authored, pass it as margin so compute_inertia_shape
                # uses the correct thickness. The real collision margin is restored after add_shape.
                if shell_thickness_val is not None and math.isfinite(float(shell_thickness_val)):
                    if float(shell_thickness_val) >= 0.0:
                        inertia_margin = float(shell_thickness_val)
                    else:
                        warnings.warn(
                            f"Shape {path}: negative shell thickness {shell_thickness_val}; falling back to margin.",
                            stacklevel=2,
                        )
                        inertia_margin = margin_val
                else:
                    inertia_margin = margin_val

                shape_params = {
                    "body": body_id,
                    "xform": shape_xform,
                    "cfg": ModelBuilder.ShapeConfig(
                        ke=shape_ke,
                        kd=shape_kd,
                        kf=shape_kf,
                        ka=shape_ka,
                        margin=inertia_margin,
                        gap=gap_val,
                        mu=material.dynamicFriction,
                        restitution=material.restitution,
                        mu_torsional=material.torsionalFriction,
                        mu_rolling=material.rollingFriction,
                        density=shape_density,
                        collision_group=collision_group,
                        is_visible=collider_is_visible,
                        sdf_max_resolution=sdf_max_resolution,
                        sdf_narrow_band_range=sdf_narrow_band_range,
                        sdf_target_voxel_size=sdf_target_voxel_size,
                        sdf_texture_format=sdf_texture_format,
                        sdf_padding=sdf_padding,
                        is_hydroelastic=is_hydroelastic,
                        kh=kh,
                        is_solid=shape_is_solid,
                    ),
                    "label": path,
                    "custom_attributes": shape_custom_attrs,
                    "color": shape_color,
                }
                # print(path, shape_params)
                if key == UsdPhysics.ObjectType.CubeShape:
                    hx, hy, hz = shape_spec.halfExtents
                    shape_id = builder.add_shape_box(
                        **shape_params,
                        hx=hx,
                        hy=hy,
                        hz=hz,
                    )
                elif key == UsdPhysics.ObjectType.SphereShape:
                    if not (scale[0] == scale[1] == scale[2]):
                        print("Warning: Non-uniform scaling of spheres is not supported.")
                    radius = shape_spec.radius
                    shape_id = builder.add_shape_sphere(
                        **shape_params,
                        radius=radius,
                    )
                elif key == UsdPhysics.ObjectType.CapsuleShape:
                    # Apply axis rotation to transform
                    axis = int(shape_spec.axis)
                    shape_params["xform"] = wp.transform(
                        shape_params["xform"].p, shape_params["xform"].q * quat_between_axes(Axis.Z, axis)
                    )
                    radius = shape_spec.radius
                    half_height = shape_spec.halfHeight
                    shape_id = builder.add_shape_capsule(
                        **shape_params,
                        radius=radius,
                        half_height=half_height,
                    )
                elif key == UsdPhysics.ObjectType.CylinderShape:
                    # Apply axis rotation to transform
                    axis = int(shape_spec.axis)
                    shape_params["xform"] = wp.transform(
                        shape_params["xform"].p, shape_params["xform"].q * quat_between_axes(Axis.Z, axis)
                    )
                    radius = shape_spec.radius
                    half_height = shape_spec.halfHeight
                    shape_id = builder.add_shape_cylinder(
                        **shape_params,
                        radius=radius,
                        half_height=half_height,
                    )
                elif key == UsdPhysics.ObjectType.ConeShape:
                    # Apply axis rotation to transform
                    axis = int(shape_spec.axis)
                    shape_params["xform"] = wp.transform(
                        shape_params["xform"].p, shape_params["xform"].q * quat_between_axes(Axis.Z, axis)
                    )
                    radius = shape_spec.radius
                    half_height = shape_spec.halfHeight
                    shape_id = builder.add_shape_cone(
                        **shape_params,
                        radius=radius,
                        half_height=half_height,
                    )
                elif key == UsdPhysics.ObjectType.MeshShape:
                    # Resolve mesh hull vertex limit from schema with fallback to parameter
                    if collider_is_visible:
                        # Visible colliders should render with the same visual material metadata
                        # as visual-only mesh imports.
                        mesh = _get_mesh_with_visual_material(prim, path_name=path)
                    else:
                        mesh = _get_mesh_cached(prim)
                    mesh.maxhullvert = R.get_value(
                        prim,
                        prim_type=PrimType.SHAPE,
                        key="max_hull_vertices",
                        default=mesh_maxhullvert,
                        verbose=verbose,
                    )
                    # add_shape_mesh() rejects SDF cfg fields on meshes; strip them and
                    # write the SDF intent to the builder lists, deferring the build to finalize().
                    mesh_shape_params = dict(shape_params)
                    mesh_shape_params["cfg"] = replace(
                        shape_params["cfg"],
                        sdf_max_resolution=None,
                        sdf_target_voxel_size=None,
                        sdf_narrow_band_range=(-0.1, 0.1),
                        sdf_texture_format="uint16",
                        sdf_padding=None,
                        is_hydroelastic=False,
                    )
                    shape_id = builder.add_shape_mesh(
                        scale=wp.vec3(*shape_spec.meshScale),
                        mesh=mesh,
                        **mesh_shape_params,
                    )
                    builder.shape_sdf_max_resolution[shape_id] = sdf_max_resolution
                    builder.shape_sdf_target_voxel_size[shape_id] = sdf_target_voxel_size
                    builder.shape_sdf_narrow_band_range[shape_id] = sdf_narrow_band_range
                    builder.shape_sdf_texture_format[shape_id] = sdf_texture_format
                    builder.shape_sdf_padding[shape_id] = sdf_padding
                    # kh is a material param; persist regardless of hydro state.
                    builder.shape_material_kh[shape_id] = kh
                    if is_hydroelastic:
                        builder.shape_flags[shape_id] |= ShapeFlags.HYDROELASTIC
                    if not skip_mesh_approximation:
                        approximation = usd.get_attribute(prim, "physics:approximation", None)
                        if approximation is not None:
                            if has_sdf_api and approximation.lower() != "none":
                                # physics:approximation belongs to PhysicsMeshCollisionAPI;
                                # it has no meaning on a NewtonSDFCollisionAPI prim.
                                warnings.warn(
                                    f"{prim.GetPath()}: physics:approximation={approximation!r} is "
                                    f"ignored on a shape with NewtonSDFCollisionAPI applied.",
                                    stacklevel=2,
                                )
                            else:
                                remeshing_method = approximation_to_remeshing_method.get(approximation.lower(), None)
                                if remeshing_method is None:
                                    if verbose:
                                        print(
                                            f"Warning: Unknown physics:approximation attribute '{approximation}' on shape at '{path}'."
                                        )
                                else:
                                    if remeshing_method not in remeshing_queue:
                                        remeshing_queue[remeshing_method] = []
                                    remeshing_queue[remeshing_method].append(shape_id)

                elif key == UsdPhysics.ObjectType.PlaneShape:
                    # Warp uses +Z convention for planes
                    if shape_spec.axis != UsdPhysics.Axis.Z:
                        xform = shape_params["xform"]
                        axis_q = quat_between_axes(Axis.Z, usd_axis_to_axis[shape_spec.axis])
                        shape_params["xform"] = wp.transform(xform.p, xform.q * axis_q)
                    shape_id = builder.add_shape_plane(
                        **shape_params,
                        width=0.0,
                        length=0.0,
                    )
                else:
                    raise NotImplementedError(f"Shape type {key} not supported yet")

                path_shape_map[path] = shape_id
                path_shape_scale[path] = scale

                # Restore the real collision margin when shell thickness was substituted.
                # TODO: Consider adding a dedicated shell_thickness field to ShapeConfig
                # so inertia thickness and collision margin don't share the same slot.
                if shell_thickness_val is not None and math.isfinite(float(shell_thickness_val)) and shape_id >= 0:
                    builder.shape_margin[shape_id] = margin_val

                if body_path in bodies_requiring_mass_properties_fallback:
                    # Prepare collider mass information for ComputeMassProperties fallback path.
                    # Prefer authored collider MassAPI mass+diagonalInertia; otherwise derive
                    # unit-density mass information from shape geometry.
                    shape_geo_type = None
                    shape_scale = wp.vec3(1.0, 1.0, 1.0)
                    shape_src = None
                    if key == UsdPhysics.ObjectType.CubeShape:
                        shape_geo_type = GeoType.BOX
                        hx, hy, hz = shape_spec.halfExtents
                        shape_scale = wp.vec3(hx, hy, hz)
                    elif key == UsdPhysics.ObjectType.SphereShape:
                        shape_geo_type = GeoType.SPHERE
                        shape_scale = wp.vec3(shape_spec.radius, 0.0, 0.0)
                    elif key == UsdPhysics.ObjectType.CapsuleShape:
                        shape_geo_type = GeoType.CAPSULE
                        shape_scale = wp.vec3(shape_spec.radius, shape_spec.halfHeight, 0.0)
                    elif key == UsdPhysics.ObjectType.CylinderShape:
                        shape_geo_type = GeoType.CYLINDER
                        shape_scale = wp.vec3(shape_spec.radius, shape_spec.halfHeight, 0.0)
                    elif key == UsdPhysics.ObjectType.ConeShape:
                        shape_geo_type = GeoType.CONE
                        shape_scale = wp.vec3(shape_spec.radius, shape_spec.halfHeight, 0.0)
                    elif key == UsdPhysics.ObjectType.MeshShape:
                        shape_geo_type = GeoType.MESH
                        shape_scale = wp.vec3(*shape_spec.meshScale)
                        shape_src = _get_mesh_cached(prim)
                    if shape_geo_type is not None:
                        expected_fallback_collider_paths.add(path)
                        shape_axis = getattr(shape_spec, "axis", None)
                        mass_info = _build_mass_info_from_authored_properties(
                            prim,
                            shape_spec.localPos,
                            shape_spec.localRot,
                            shape_geo_type,
                            shape_scale,
                            shape_src,
                            shape_axis,
                        )
                        if mass_info is None:
                            mass_info = _build_mass_info_from_shape_geometry(
                                prim,
                                shape_spec.localPos,
                                shape_spec.localRot,
                                shape_geo_type,
                                shape_scale,
                                shape_src,
                                shape_axis,
                                is_solid=shape_is_solid,
                                thickness=inertia_margin,
                            )
                        if mass_info is not None:
                            rigid_body_mass_info_map[path] = mass_info

                _collect_filtered_pairs(prim)

                if not _is_enabled_collider(prim):
                    no_collision_shapes.add(shape_id)
                    builder.shape_flags[shape_id] &= ~ShapeFlags.COLLIDE_SHAPES

    # approximate meshes
    for remeshing_method, shape_ids in remeshing_queue.items():
        builder.approximate_meshes(method=remeshing_method, shape_indices=shape_ids)

    # Filtered pairs are applied after the deformable passes below, once every endpoint's
    # Newton shapes exist.

    # apply collision filters to all shapes that have no collision
    for shape_id in no_collision_shapes:
        for other_shape_id in range(builder.shape_count):
            if other_shape_id != shape_id:
                builder.add_shape_collision_filter_pair(shape_id, other_shape_id)

    # apply collision filters from articulations that have self collisions disabled
    for art_id, bodies in articulation_bodies.items():
        if not articulation_has_self_collision[art_id]:
            for body1, body2 in itertools.combinations(bodies, 2):
                for shape1 in builder.body_shapes[body1]:
                    for shape2 in builder.body_shapes[body2]:
                        builder.add_shape_collision_filter_pair(shape1, shape2)

    # Load Gaussian splat prims that weren't already captured as children of rigid bodies.
    if load_visual_shapes:
        prims = iter(Usd.PrimRange(stage.GetPrimAtPath(root_path), Usd.TraverseInstanceProxies()))
        for gaussian_prim in prims:
            if str(gaussian_prim.GetPath()).startswith("/Prototypes/"):
                continue

            if gaussian_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prims.PruneChildren()
                continue

            if str(gaussian_prim.GetTypeName()) != "ParticleField3DGaussianSplat":
                continue

            gaussian_path = str(gaussian_prim.GetPath())
            if gaussian_path in path_shape_map:
                continue
            if any(re.match(p, gaussian_path) for p in ignore_paths):
                continue

            body_id = -1

            prim_world_mat = _get_prim_world_mat(gaussian_prim, None, incoming_world_xform)

            g_pos, g_rot, g_scale = wp.transform_decompose(prim_world_mat)
            gaussian = usd.get_gaussian(gaussian_prim)
            splat_cfg = copy.copy(visual_shape_cfg)
            splat_cfg.is_visible = _is_effectively_visible(gaussian_prim)
            splat_material_props = _get_material_props_cached(gaussian_prim)
            shape_id = builder.add_shape_gaussian(
                body_id,
                gaussian=gaussian,
                xform=wp.transform(g_pos, g_rot),
                scale=g_scale,
                cfg=splat_cfg,
                color=splat_material_props.get("color"),
                label=gaussian_path,
            )
            path_shape_map[gaussian_path] = shape_id
            path_shape_scale[gaussian_path] = g_scale
            if verbose:
                print(f"Added Gaussian splat shape {gaussian_path} with id {shape_id}.")

    def _zero_mass_information():
        """Create a reusable zero-contribution collider mass payload for callback fallback."""
        mass_info = UsdPhysics.RigidBodyAPI.MassInformation()
        mass_info.volume = 0.0
        mass_info.centerOfMass = Gf.Vec3f(0.0)
        mass_info.localPos = Gf.Vec3f(0.0)
        mass_info.localRot = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
        mass_info.inertia = Gf.Matrix3f(0.0)
        return mass_info

    zero_mass_information = _zero_mass_information()
    warned_missing_collider_mass_info: set[str] = set()

    def _get_collision_mass_information(collider_prim: Usd.Prim):
        """MassInformation callback for ``ComputeMassProperties`` with one-time warning on misses."""
        collider_path = str(collider_prim.GetPath())
        is_expected_missing = (
            collider_path in expected_fallback_collider_paths and collider_path not in rigid_body_mass_info_map
        )
        if is_expected_missing and collider_path not in warned_missing_collider_mass_info:
            warnings.warn(
                f"Skipping collider {collider_path} in mass aggregation: missing usable collider mass information.",
                stacklevel=2,
            )
            warned_missing_collider_mass_info.add(collider_path)
        return rigid_body_mass_info_map.get(collider_path, zero_mass_information)

    # overwrite inertial properties of bodies that have PhysicsMassAPI schema applied
    if UsdPhysics.ObjectType.RigidBody in ret_dict:
        paths, rigid_body_descs = ret_dict[UsdPhysics.ObjectType.RigidBody]
        for path, rigid_body_desc in zip(paths, rigid_body_descs, strict=False):
            prim = stage.GetPrimAtPath(path)
            if not prim.HasAPI(UsdPhysics.MassAPI):
                continue
            body_path = str(path)
            body_id = path_body_map.get(body_path, -1)
            if body_id == -1:
                continue
            mass_api = UsdPhysics.MassAPI(prim)
            has_authored_mass = mass_api.GetMassAttr().HasAuthoredValue()
            has_authored_inertia = mass_api.GetDiagonalInertiaAttr().HasAuthoredValue()
            has_authored_com = mass_api.GetCenterOfMassAttr().HasAuthoredValue()

            # newton:inertia (compact 6-element tensor) overrides physics:diagonalInertia + physics:principalAxes.
            inertia_tensor_val = (
                usd.get_attribute(prim, "newton:inertia") if usd.has_applied_api_schema(prim, "NewtonMassAPI") else None
            )
            has_inertia_tensor = inertia_tensor_val is not None
            if has_inertia_tensor:
                if len(inertia_tensor_val) != 6:
                    warnings.warn(
                        f"Body {body_path}: newton:inertia has {len(inertia_tensor_val)} elements, expected 6. Ignoring.",
                        stacklevel=2,
                    )
                    has_inertia_tensor = False
                elif not all(math.isfinite(v) for v in inertia_tensor_val):
                    warnings.warn(
                        f"Body {body_path}: newton:inertia contains non-finite values. Ignoring.",
                        stacklevel=2,
                    )
                    has_inertia_tensor = False
                elif any(v < 0.0 for v in inertia_tensor_val[:3]):
                    warnings.warn(
                        f"Body {body_path}: newton:inertia has negative diagonal elements. Ignoring.",
                        stacklevel=2,
                    )
                    has_inertia_tensor = False
                else:
                    ixx, iyy, izz, ixy, ixz, iyz = inertia_tensor_val
                    inertia_np = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=np.float64)
                    if np.any(np.linalg.eigvalsh(inertia_np) < 0.0):
                        warnings.warn(
                            f"Body {body_path}: newton:inertia is not positive semidefinite. Ignoring.",
                            stacklevel=2,
                        )
                        has_inertia_tensor = False
                    else:
                        has_authored_inertia = True
                        inertia_tensor = wp.mat33(ixx, ixy, ixz, ixy, iyy, iyz, ixz, iyz, izz)

            # Compute baseline mass properties via mass computer when at least one property needs resolving.
            if not (has_authored_mass and has_authored_inertia and has_authored_com):
                rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
                cmp_mass, cmp_i_diag, cmp_com, cmp_principal_axes = rigid_body_api.ComputeMassProperties(
                    _get_collision_mass_information
                )
                if cmp_mass < 0.0:
                    # ComputeMassProperties failed to discover colliders (e.g. shapes
                    # created by schema resolvers are not real USD prims). Fall back to
                    # builder-accumulated mass properties from add_shape_*() calls.
                    cmp_mass = builder.body_mass[body_id]
                    if not has_authored_com:
                        cmp_com = builder.body_com[body_id]
                    # When the body has an authored density, rescale accumulated mass
                    # and inertia from the builder's default shape density to the
                    # body-level density (USD body density overrides per-shape density).
                    body_density_attr = mass_api.GetDensityAttr()
                    if (
                        body_density_attr.HasAuthoredValue()
                        and float(body_density_attr.Get()) > 0.0
                        and default_shape_density > 0.0
                    ):
                        density_scale = float(body_density_attr.Get()) / default_shape_density
                        cmp_mass *= density_scale
                        builder.body_inertia[body_id] = wp.mat33(
                            np.array(builder.body_inertia[body_id]) * density_scale
                        )
                    cmp_i_diag = Gf.Vec3f(0.0, 0.0, 0.0)
                    cmp_principal_axes = Gf.Quatf(1.0, 0.0, 0.0, 0.0)

            if has_authored_com:
                # Match the scale/frame convention used by OpenUSD's collider and joint descriptors.
                cmp_com = Gf.CompMult(mass_api.GetCenterOfMassAttr().Get(), rigid_body_desc.scale)

            # Inertia: newton:inertia > physics:diagonalInertia + physics:principalAxes > mass computer.
            # When mass is authored but inertia is not, keep accumulated inertia
            # (scaled to match authored mass below) instead of using mass computer
            # inertia, which may already reflect the authored mass.
            if has_inertia_tensor:
                i_diag_np = None  # skip diagonal path; full matrix set below
            elif has_authored_inertia:
                i_diag_np = np.array(mass_api.GetDiagonalInertiaAttr().Get(), dtype=np.float32)
                if np.any(i_diag_np < 0.0):
                    warnings.warn(
                        f"Body {body_path}: authored diagonal inertia contains negative values. "
                        "Falling back to mass-computer result.",
                        stacklevel=2,
                    )
                    has_authored_inertia = False
                    i_diag_np = np.array(cmp_i_diag, dtype=np.float32)
                    principal_axes = cmp_principal_axes
                elif mass_api.GetPrincipalAxesAttr().HasAuthoredValue():
                    principal_axes = mass_api.GetPrincipalAxesAttr().Get()
                else:
                    principal_axes = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
            elif not has_authored_mass:
                i_diag_np = np.array(cmp_i_diag, dtype=np.float32)
                principal_axes = cmp_principal_axes
            else:
                # Mass authored, inertia not: keep accumulated inertia and scale
                # to match authored mass in the mass block below.
                i_diag_np = None
            if has_inertia_tensor:
                builder.body_inertia[body_id] = inertia_tensor
                det = np.linalg.det(np.array(inertia_tensor).reshape(3, 3))
                if det > 0.0:
                    builder.body_inv_inertia[body_id] = wp.inverse(inertia_tensor)
                else:
                    builder.body_inv_inertia[body_id] = wp.mat33(0.0)
            elif i_diag_np is not None and np.linalg.norm(i_diag_np) > 0.0:
                i_rot = usd.value_to_warp(principal_axes)
                rot = np.array(wp.quat_to_matrix(i_rot), dtype=np.float32).reshape(3, 3)
                inertia = rot @ np.diag(i_diag_np) @ rot.T
                builder.body_inertia[body_id] = wp.mat33(inertia)
                if inertia.any():
                    builder.body_inv_inertia[body_id] = wp.inverse(wp.mat33(*inertia))
                else:
                    builder.body_inv_inertia[body_id] = wp.mat33(0.0)

            # Mass: authored value takes precedence over mass computer.
            if has_authored_mass:
                mass = float(mass_api.GetMassAttr().Get())
                shape_accumulated_mass = builder.body_mass[body_id]
                if not has_authored_inertia and mass_api.GetDensityAttr().HasAuthoredValue():
                    warnings.warn(
                        f"Body {body_path}: authored mass and density without authored diagonalInertia. "
                        f"Ignoring body-level density.",
                        stacklevel=2,
                    )
                # When mass is authored but inertia is not, scale the accumulated
                # inertia to be consistent with the authored mass.
                if not has_authored_inertia and shape_accumulated_mass > 0.0 and mass > 0.0:
                    scale = mass / shape_accumulated_mass
                    builder.body_inertia[body_id] = wp.mat33(np.array(builder.body_inertia[body_id]) * scale)
                    builder.body_inv_inertia[body_id] = wp.inverse(builder.body_inertia[body_id])
            else:
                mass = cmp_mass
            builder.body_mass[body_id] = mass
            builder.body_inv_mass[body_id] = 1.0 / mass if mass > 0.0 else 0.0

            builder.body_com[body_id] = wp.vec3(*cmp_com)

            # Assign nonzero inertia if mass is nonzero to make sure the body can be simulated.
            I_m = np.array(builder.body_inertia[body_id])
            mass = builder.body_mass[body_id]
            if I_m.max() == 0.0:
                if mass > 0.0:
                    # Heuristic: assume a uniform density sphere with the given mass
                    # For a sphere: I = (2/5) * m * r^2
                    # Estimate radius from mass assuming reasonable density (e.g., water density ~1000 kg/m³)
                    # This gives r = (3*m/(4*π*p))^(1/3)
                    density = default_shape_density  # kg/m^3
                    volume = mass / density
                    radius = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
                    _, _, I_default = compute_inertia_sphere(density, radius)

                    # Apply parallel axis theorem if center of mass is offset
                    com = np.array(builder.body_com[body_id], dtype=np.float32)
                    if np.linalg.norm(com) > 1e-6:
                        # I = I_cm + m * d² where d is distance from COM to body origin
                        d_squared = np.sum(com**2)
                        I_default += wp.mat33(mass * d_squared * np.eye(3, dtype=np.float32))

                    builder.body_inertia[body_id] = I_default
                    builder.body_inv_inertia[body_id] = wp.inverse(I_default)

                    if verbose:
                        print(
                            f"Applied default inertia matrix for body {body_path}: diagonal elements = [{I_default[0, 0]}, {I_default[1, 1]}, {I_default[2, 2]}]"
                        )
                else:
                    warnings.warn(
                        f"Body {body_path} has zero mass and zero inertia despite having the MassAPI USD schema applied.",
                        stacklevel=2,
                    )

    # add joints to floating bodies (bodies not connected as children to any joint)
    new_bodies = list(path_body_map.values())
    if no_articulations and has_joints:
        # Preserve authored orphan-joint graphs while still articulating unrelated bodies (#3002).
        connected_bodies = set(builder.joint_parent) | set(builder.joint_child)
        bodies_to_articulate = [body_id for body_id in new_bodies if body_id not in connected_bodies]
    else:
        bodies_to_articulate = new_bodies

    if bodies_to_articulate:
        if parent_body != -1:
            # When parent_body is specified, manually add joints to floating bodies with correct parent
            joint_children = set(builder.joint_child)
            for body_id in bodies_to_articulate:
                if body_id in joint_children:
                    continue  # Already has a joint
                if builder.body_mass[body_id] <= 0:
                    continue  # Skip static bodies
                # Compute parent_xform to preserve imported pose when attaching to parent_body
                # When parent_body is specified, use incoming_world_xform as parent-relative offset
                parent_xform = incoming_world_xform
                joint_id = builder._add_base_joint(
                    body_id,
                    floating=floating,
                    base_joint=base_joint,
                    parent=parent_body,
                    parent_xform=parent_xform,
                )
                # Attach to parent's articulation
                builder._finalize_imported_articulation(
                    joint_indices=[joint_id],
                    parent_body=parent_body,
                    articulation_label=None,
                )
        else:
            joint_children = set(builder.joint_child)
            for body_id in bodies_to_articulate:
                if body_id in joint_children:
                    continue
                if builder.body_mass[body_id] <= 0:
                    continue

                joint_id = builder._add_base_joint(body_id, floating=floating, base_joint=base_joint)
                body_path = builder.body_label[body_id]
                articulation_root_path = next(
                    (
                        root
                        for root in authored_articulation_root_paths
                        if body_path == root or body_path.startswith("/" if root == "/" else f"{root}/")
                    ),
                    None,
                )
                if articulation_root_path is not None:
                    builder._finalize_imported_articulation(
                        joint_indices=[joint_id],
                        parent_body=parent_body,
                        articulation_label=articulation_root_path,
                    )
                else:
                    builder.add_articulation([joint_id], label=body_path)

    # Build deformables (cables/cloth/volume) after rigid bodies, their collider-mass computation,
    # and the floating-body base-joint pass above. The importer wraps each cable into its own
    # articulation, so building deformables last keeps those articulations after any
    # importer-created ones (e.g. kinematic anchors), preserving ascending articulation order.
    # Volume deformables (TetMesh -> soft body). PhysicsVolumeDeformableSimAPI (or a
    # PhysicsDeformableBodyAPI) opts into the mass precedence; a bare TetMesh stays legacy.
    # Mass precedence (proposal): per-point physics:masses > body mass > body density
    # > material density; per-element weighting is left to the add_* builders.
    if _deformable_prims.has_candidates():
        _deformable_ctx = _DeformableImportContext(
            builder=builder,
            stage=stage,
            root_prim=root_prim,
            resolver=R,
            collect_schema_attrs=collect_schema_attrs,
            deformable_read=deformable_read,
            get_prim_world_mat=_get_prim_world_mat,
            get_rigid_body_ancestor_path=_get_rigid_body_ancestor_path,
            get_first_target=_get_first_target,
            get_tetmesh_cached=_get_tetmesh_cached,
            incoming_world_xform=incoming_world_xform,
            linear_unit=linear_unit,
            ignore_paths=ignore_paths,
            verbose=verbose,
            path_body_map=path_body_map,
            path_shape_map=path_shape_map,
            path_cable_map=path_cable_map,
            path_cable_attrs=path_cable_attrs,
            path_cable_segments=path_cable_segments,
            path_cable_point_anchors=path_cable_point_anchors,
            path_cloth_map=path_cloth_map,
            path_cloth_attrs=path_cloth_attrs,
            path_soft_map=path_soft_map,
            path_soft_attrs=path_soft_attrs,
            path_attachment_map=path_attachment_map,
            path_attachment_attrs=path_attachment_attrs,
            prims=_deformable_prims,
        )

        # Curve-to-curve junctions weld into rod graphs before the per-curve cable pass, which skips
        # the consumed curves; the attachment pass below skips the consumed junctions. Each pass runs
        # only when its bucket has candidates; welding additionally needs attachments to weld with.
        consumed_cable_curve_paths: set[str] = set()
        consumed_junction_attachment_paths: set[str] = set()
        if _deformable_prims.cables and _deformable_prims.attachments:
            consumed_cable_curve_paths, consumed_junction_attachment_paths = _deformable_import_cable_graphs(
                _deformable_ctx
            )
        if _deformable_prims.cables:
            _deformable_import_cable(_deformable_ctx, consumed_cable_curve_paths)
        if _deformable_prims.cloth:
            _deformable_import_cloth(_deformable_ctx)
        if _deformable_prims.tetmeshes:
            _deformable_import_volume(_deformable_ctx)

        # PhysicsAttachment prims from the AOUSD deformables proposal. The current
        # builder can faithfully lower the cable/rod subset because imported cables
        # are rigid capsule bodies. Surface/volume attachments require a separate
        # deformable-site constraint model, so those are preserved as attrs and warned.
        if _deformable_prims.attachments:
            _deformable_import_attachments(_deformable_ctx, consumed_junction_attachment_paths)

        # AOUSD PhysicsElementCollisionFilter prims: suppress collision between authored element
        # groups (cable segments / collider shapes); runs after the cables and colliders exist.
        if _deformable_prims.element_filters:
            _deformable_import_element_collision_filters(_deformable_ctx)

        # physics:filteredPairs may be authored on the deformable side: simulation geometry,
        # a deformable body prim, or a deformable-owned collider. Those prims are excluded
        # from the native collider loop, so collect their relationships here (the set
        # deduplicates prims reachable through more than one route).
        for _filter_prim in (*_deformable_prims.cables, *_deformable_prims.cloth, *_deformable_prims.tetmeshes):
            _collect_filtered_pairs(_filter_prim)
        for _filter_path in (*_deformable_prims.body_owner, *_deformable_prims.native_physics_exclude_paths):
            _filter_prim = stage.GetPrimAtPath(_filter_path)
            if _filter_prim and _filter_prim.IsValid():
                _collect_filtered_pairs(_filter_prim)

    def _resolve_collision_shape_ids(path: str) -> tuple[list[int], str | None]:
        """Resolve a filtered-pair endpoint to Newton shape indices, or an unsupported reason.

        Endpoint ownership comes only from the import maps (never path-prefix matching): a
        native collider is one shape, a rigid body or cable is all of its shapes, and a
        deformable body prim resolves through its simulation geometry. Cloth and volume
        deformables are particles, which Newton's shape filter pairs cannot express.
        """
        if path in path_shape_map:
            return [path_shape_map[path]], None
        if path in path_body_map:
            return sorted(set(builder.body_shapes.get(path_body_map[path], []))), None
        if path in path_cable_map:
            shape_ids: set[int] = set()
            for cable_body in path_cable_map[path][0]:
                shape_ids.update(builder.body_shapes.get(cable_body, []))
            return sorted(shape_ids), None
        owner_path = _deformable_prims.body_owner.get(path)
        if owner_path is not None and owner_path != path:
            return _resolve_collision_shape_ids(owner_path)
        if path in path_cloth_map:
            return [], "it is a cloth particle deformable, and standard particle collision filters are not supported"
        if path in path_soft_map:
            return [], "it is a volume particle deformable, and standard particle collision filters are not supported"
        target_prim = stage.GetPrimAtPath(path)
        if not target_prim or not target_prim.IsValid():
            return [], "the target path does not exist"
        return [], "it produced no collision participant (it may be disabled, ignored, malformed, or non-colliding)"

    # physics:filteredPairs may also be authored on a rigid-body prim (UsdPhysics allows
    # collider, body, or articulation endpoints); the collider loop never visits body prims.
    # path_body_map covers every imported body regardless of which creation path added it.
    for body_prim_path in path_body_map:
        body_prim = stage.GetPrimAtPath(body_prim_path)
        if body_prim and body_prim.IsValid():
            _collect_filtered_pairs(body_prim)

    # Apply the authored filtered pairs: every native shape and cable capsule exists now, and
    # the deformable maps allow precise unsupported diagnostics. Shape indices are stable from
    # here on (collapse_fixed_joints only remaps bodies). Seed the dedup set from the builder
    # so pairs the element-filter pass already added are not appended again.
    if authored_filtered_path_pairs:
        existing_filter_pairs = set(builder.shape_collision_filter_pairs)
        for filter_path1, filter_path2 in sorted(authored_filtered_path_pairs):
            shapes1, reason1 = _resolve_collision_shape_ids(filter_path1)
            shapes2, reason2 = _resolve_collision_shape_ids(filter_path2)
            if not shapes1 or not shapes2:
                bad_path, reason = (filter_path1, reason1) if not shapes1 else (filter_path2, reason2)
                warnings.warn(
                    f"{filter_path1} <-> {filter_path2}: physics:filteredPairs was not imported "
                    f"because {bad_path}: {reason}.",
                    stacklevel=2,
                )
                continue
            for shape1 in shapes1:
                for shape2 in shapes2:
                    if shape1 == shape2:
                        continue
                    pair = (shape1, shape2) if shape1 < shape2 else (shape2, shape1)
                    if pair not in existing_filter_pairs:
                        existing_filter_pairs.add(pair)
                        builder.add_shape_collision_filter_pair(*pair)

    # Parse MjcEquality constraints *before* collapsing fixed joints so that the
    # builder's collapse logic can remap body/joint indices and adjust anchors/relposes
    # for any bodies that get merged.
    def _parse_mjc_equality_constraints():
        local_builder_custom_attr_eq = builder_custom_attr_eq
        # The equality custom attributes are declared by ModelBuilder.__init__; register the
        # remaining MuJoCo custom attributes needed to parse and convert the model.
        # register_custom_attributes is idempotent, so re-registering the equality fields is a no-op.
        if convert_mjc_equality_constraints:
            from ..solvers.mujoco.solver_mujoco import SolverMuJoCo  # noqa: PLC0415

            SolverMuJoCo.register_custom_attributes(builder)
            local_builder_custom_attr_eq = builder.get_custom_attributes_by_frequency(["mujoco:equality_constraint"])

        def add_converted_loop_joint(
            eq_type: EqType,
            body1: int,
            body2: int,
            anchor: wp.vec3,
            relpose: wp.transform | None,
            torquescale: float,
            joint_path: str,
            enabled: bool,
            custom_attrs: dict[str, Any],
        ) -> None:
            try:
                _, joint_idx = mjc_add_equality_loop_joint(
                    builder,
                    eq_type,
                    body1,
                    body2,
                    anchor,
                    relpose,
                    torquescale,
                    joint_path,
                    enabled,
                    custom_attrs,
                )
            except ValueError:
                warnings.warn(
                    f"MuJoCo equality '{joint_path}' has no valid body reference; skipping.",
                    stacklevel=2,
                )
                return

            path_joint_map[joint_path] = joint_idx

        for joint_path, joint_desc in joint_descriptions.items():
            joint_prim = stage.GetPrimAtPath(joint_path)
            if not joint_prim or not joint_prim.IsValid():
                continue
            if any(re.match(p, joint_path) for p in ignore_paths):
                continue

            is_connect = joint_path in mjc_equality_connect_paths
            is_weld = joint_path in mjc_equality_weld_paths
            is_eq_joint = _has_api_schema(joint_prim, "MjcEqualityJointAPI")
            if not (is_connect or is_weld or is_eq_joint):
                continue

            if only_load_enabled_joints and not joint_desc.jointEnabled:
                continue

            if collect_schema_attrs and (is_connect or is_weld):
                R.collect_prim_attrs(joint_prim)

            eq_custom_attrs = usd.get_custom_attribute_values(
                joint_prim, local_builder_custom_attr_eq, context={"builder": builder}
            )
            enabled = bool(joint_desc.jointEnabled)

            if is_connect or is_weld:
                schema_name = "MjcEqualityConnectAPI" if is_connect else "MjcEqualityWeldAPI"
                body0_info, body1_info = _resolve_equality_bodies(joint_prim, joint_path, schema_name)
                if body0_info is None or body1_info is None:
                    continue

                body0_idx, site0_local_pos = body0_info
                body1_idx, site1_local_pos = body1_info
                target0 = _get_first_target(joint_prim, "physics:body0")
                target1 = _get_first_target(joint_prim, "physics:body1")

                if is_connect:
                    # Use the authored localPose0 when target0 is a known body or the world
                    # (empty target means world); fall back to the site-derived local position
                    # only when target0 is a site prim that is not itself a body.
                    anchor = (
                        wp.vec3(*joint_desc.localPose0Position)
                        if (_is_world_target(target0) or target0 in path_body_map)
                        else site0_local_pos
                    )
                    if convert_mjc_equality_constraints:
                        add_converted_loop_joint(
                            EqType.CONNECT,
                            body0_idx,
                            body1_idx,
                            anchor,
                            None,
                            0.0,
                            joint_path,
                            enabled,
                            eq_custom_attrs,
                        )
                    else:
                        _add_equality_constraint(
                            builder,
                            constraint_type=EqType.CONNECT,
                            body1=body0_idx,
                            body2=body1_idx,
                            anchor=anchor,
                            label=joint_path,
                            enabled=enabled,
                            custom_attributes=eq_custom_attrs,
                        )
                else:
                    local_rot0 = usd.value_to_warp(joint_desc.localPose0Orientation)
                    local_rot1 = usd.value_to_warp(joint_desc.localPose1Orientation)
                    local_pos0 = wp.vec3(*joint_desc.localPose0Position)
                    local_pos1 = wp.vec3(*joint_desc.localPose1Position)
                    # MuJoCo weld anchors are authored on the body1 side. Direct
                    # body/world targets use localPose1; site targets use the site position.
                    anchor = (
                        wp.vec3(*joint_desc.localPose1Position)
                        if (_is_world_target(target1) or target1 in path_body_map)
                        else site1_local_pos
                    )
                    relpose_rot = local_rot0 * wp.quat_inverse(local_rot1)
                    relpose_pos = local_pos0 - wp.quat_rotate(relpose_rot, local_pos1)
                    torquescale_attr = joint_prim.GetAttribute("mjc:torqueScale")
                    torquescale = (
                        float(torquescale_attr.Get()) if torquescale_attr and torquescale_attr.HasValue() else 1.0
                    )
                    relpose = wp.transform(relpose_pos, relpose_rot)
                    if convert_mjc_equality_constraints:
                        add_converted_loop_joint(
                            EqType.WELD,
                            body0_idx,
                            body1_idx,
                            anchor,
                            relpose,
                            torquescale,
                            joint_path,
                            enabled,
                            eq_custom_attrs,
                        )
                    else:
                        _add_equality_constraint(
                            builder,
                            constraint_type=EqType.WELD,
                            body1=body0_idx,
                            body2=body1_idx,
                            anchor=anchor,
                            relpose=relpose,
                            torquescale=torquescale,
                            label=joint_path,
                            enabled=enabled,
                            custom_attributes=eq_custom_attrs,
                        )
                continue

            if is_eq_joint:
                joint1_idx = path_joint_map.get(joint_path)
                if joint1_idx is None:
                    warnings.warn(
                        f"MjcEqualityJointAPI on '{joint_path}' was not found in path_joint_map; skipping.",
                        stacklevel=2,
                    )
                    continue

                target_rel = joint_prim.GetRelationship("mjc:target")
                targets = target_rel.GetTargets() if target_rel else []
                if not targets:
                    warnings.warn(
                        f"MjcEqualityJointAPI on '{joint_path}' has no mjc:target relationship; skipping.",
                        stacklevel=2,
                    )
                    continue

                target_path = str(targets[0])
                joint2_idx = path_joint_map.get(target_path)
                if joint2_idx is None:
                    warnings.warn(
                        f"MjcEqualityJointAPI on '{joint_path}' references '{target_path}' which was not found in path_joint_map; skipping.",
                        stacklevel=2,
                    )
                    continue

                polycoef = []
                for attr_name, default in (
                    ("mjc:coef0", 0.0),
                    ("mjc:coef1", 1.0),
                    ("mjc:coef2", 0.0),
                    ("mjc:coef3", 0.0),
                    ("mjc:coef4", 0.0),
                ):
                    attr = joint_prim.GetAttribute(attr_name)
                    polycoef.append(float(attr.Get()) if attr and attr.HasValue() else default)

                if convert_mjc_equality_constraints:
                    if mjc_polycoef_has_higher_order(polycoef):
                        warnings.warn(
                            f"Warning: Joint equality '{joint_path}' uses higher-order polycoef terms. "
                            "They are preserved for SolverMuJoCo, but generic Newton mimic constraints use "
                            "only coef0/coef1.",
                            stacklevel=2,
                        )
                    mjc_add_equality_mimic(
                        builder,
                        joint1_idx,
                        joint2_idx,
                        polycoef,
                        joint_path,
                        enabled,
                        eq_custom_attrs,
                    )
                else:
                    _add_equality_constraint(
                        builder,
                        constraint_type=EqType.JOINT,
                        joint1=joint1_idx,
                        joint2=joint2_idx,
                        polycoef=polycoef,
                        label=joint_path,
                        enabled=enabled,
                        custom_attributes=eq_custom_attrs,
                    )

    _parse_mjc_equality_constraints()

    # collapsing fixed joints to reduce the number of simulated bodies connected by fixed joints.
    collapse_results = None
    path_body_relative_transform = {}
    builder_joint_labels_before_collapse = list(builder.joint_label)
    if scene_attributes.get("newton:collapse_fixed_joints", collapse_fixed_joints):
        collapse_results = builder.collapse_fixed_joints()
        body_merged_parent = collapse_results["body_merged_parent"]
        body_merged_transform = collapse_results["body_merged_transform"]
        body_remap = collapse_results["body_remap"]

        for path, body_id in path_body_map.items():
            if body_id in body_remap:
                new_id = body_remap[body_id]
            elif body_id in body_merged_parent:
                # this body has been merged with another body
                new_id = body_remap[body_merged_parent[body_id]]
                path_body_relative_transform[path] = body_merged_transform[body_id]
            else:
                # this body has not been merged
                new_id = body_id

            path_body_map[path] = new_id

        # Cable bodies/joints and attachment joints are addressed by index (not prim path), so
        # remap them through the collapse maps to keep their path maps valid after collapsing.
        path_cable_map, path_attachment_map = _deformable_remap_collapsed(
            path_cable_map,
            path_attachment_map,
            path_attachment_attrs,
            collapse_results["joint_remap"],
            body_remap,
            body_merged_parent,
        )

        # Joint indices may have shifted after collapsing fixed joints; refresh the joint path map accordingly.
        # First rebuild the canonical label→index map, then re-add merged joint aliases.
        new_label_to_idx = {label: idx for idx, label in enumerate(builder.joint_label)}
        old_path_joint_map = path_joint_map
        path_joint_map = dict(new_label_to_idx)
        for path, old_idx in old_path_joint_map.items():
            if path in path_joint_map:
                continue  # already mapped via joint_label
            # Find the new index for this merged alias via the representative label
            old_label = (
                builder_joint_labels_before_collapse[old_idx]
                if old_idx < len(builder_joint_labels_before_collapse)
                else None
            )
            if old_label is not None and old_label in new_label_to_idx:
                path_joint_map[path] = new_label_to_idx[old_label]

    # Mimic constraints from PhysxMimicJointAPI (run after collapse so joint indices are final).
    # PhysxMimicJointAPI is an instance-applied schema (e.g. PhysxMimicJointAPI:rotZ)
    # that couples a follower joint to a leader (reference) joint with a gearing ratio.
    # PhysX convention: jointPos + gearing * refJointPos + offset = 0
    # Newton/URDF convention: joint0 = coef0 + coef1 * joint1
    # Therefore: coef1 = -gearing, coef0 = -offset
    for joint_path, joint_idx in path_joint_map.items():
        joint_prim = stage.GetPrimAtPath(joint_path)
        if not joint_prim or not joint_prim.IsValid():
            continue

        # Skip if NewtonMimicAPI is present — it takes precedence over PhysxMimicJointAPI.
        if usd.has_applied_api_schema(joint_prim, "NewtonMimicAPI"):
            continue
        # Skip if MjcEqualityJointAPI is present — it creates equality constraints, not mimic.
        if _has_api_schema(joint_prim, "MjcEqualityJointAPI"):
            continue

        schemas_listop = joint_prim.GetMetadata("apiSchemas")
        if not schemas_listop:
            continue

        all_schemas = (
            list(schemas_listop.prependedItems)
            + list(schemas_listop.appendedItems)
            + list(schemas_listop.explicitItems)
        )

        for schema in all_schemas:
            schema_str = str(schema)
            if not schema_str.startswith("PhysxMimicJointAPI"):
                continue

            # Extract the axis instance name (e.g. "rotZ" from "PhysxMimicJointAPI:rotZ")
            parts = schema_str.split(":")
            if len(parts) < 2:
                continue
            axis_instance = parts[1]

            ref_joint_rel = joint_prim.GetRelationship(f"physxMimicJoint:{axis_instance}:referenceJoint")
            if not ref_joint_rel:
                continue
            targets = ref_joint_rel.GetTargets()
            if not targets:
                continue
            leader_path = targets[0]
            if not leader_path.IsAbsolutePath():
                leader_path = joint_prim.GetPath().GetParentPath().AppendPath(leader_path)
            leader_path = str(leader_path)

            leader_idx = path_joint_map.get(leader_path)
            if leader_idx is None:
                warnings.warn(
                    f"PhysxMimicJointAPI on '{joint_path}' references '{leader_path}' "
                    f"but leader joint was not found, skipping mimic constraint",
                    stacklevel=2,
                )
                continue

            gearing_attr = joint_prim.GetAttribute(f"physxMimicJoint:{axis_instance}:gearing")
            gearing = float(gearing_attr.Get()) if gearing_attr and gearing_attr.HasValue() else 1.0

            offset_attr = joint_prim.GetAttribute(f"physxMimicJoint:{axis_instance}:offset")
            offset = float(offset_attr.Get()) if offset_attr and offset_attr.HasValue() else 0.0

            builder.add_constraint_mimic(
                joint0=joint_idx,
                joint1=leader_idx,
                coef0=-offset,
                coef1=-gearing,
                enabled=True,
                label=joint_path,
            )

            if verbose:
                print(
                    f"Added PhysxMimicJointAPI constraint: '{joint_path}' follows '{leader_path}' "
                    f"(gearing={gearing}, offset={offset}, axis={axis_instance})"
                )

    # Mimic constraints from NewtonMimicAPI (run after collapse so joint indices are final).
    for joint_path, joint_idx in path_joint_map.items():
        joint_prim = stage.GetPrimAtPath(joint_path)
        if not joint_prim.IsValid() or not joint_prim.HasAPI("NewtonMimicAPI"):
            continue
        if _has_api_schema(joint_prim, "MjcEqualityJointAPI"):
            continue
        mimic_enabled = usd.get_attribute(joint_prim, "newton:mimicEnabled", default=True)
        if not mimic_enabled:
            continue
        mimic_rel = joint_prim.GetRelationship("newton:mimicJoint")
        if not mimic_rel or not mimic_rel.HasAuthoredTargets():
            if verbose:
                print(f"NewtonMimicAPI on {joint_path} has no newton:mimicJoint target; skipping")
            continue
        targets = mimic_rel.GetTargets()
        if not targets:
            if verbose:
                print(f"NewtonMimicAPI on {joint_path}: newton:mimicJoint has no targets; skipping")
            continue
        leader_path = targets[0]
        if not leader_path.IsAbsolutePath():
            leader_path = joint_prim.GetPath().GetParentPath().AppendPath(leader_path)
        leader_path_str = str(leader_path)
        if leader_path_str not in path_joint_map:
            warnings.warn(
                f"NewtonMimicAPI on {joint_path}: leader {leader_path_str} not in path_joint_map; skipping mimic constraint.",
                stacklevel=2,
            )
            continue
        coef0 = usd.get_attribute(joint_prim, "newton:mimicCoef0", default=0.0)
        coef1 = usd.get_attribute(joint_prim, "newton:mimicCoef1", default=1.0)
        leader_idx = path_joint_map[leader_path_str]
        builder.add_constraint_mimic(
            joint0=joint_idx,
            joint1=leader_idx,
            coef0=coef0,
            coef1=coef1,
            enabled=True,
            label=joint_path,
        )

    # Parse Newton actuator prims from the USD stage.
    from ..actuators.delay import Delay  # noqa: PLC0415
    from ..actuators.usd_parser import parse_actuator_prim  # noqa: PLC0415

    actuator_count = 0
    path_to_dof = {
        path: builder.joint_qd_start[idx] + merged_dof_offset.get(path, 0)
        for path, idx in path_joint_map.items()
        if idx < len(builder.joint_qd_start)
    }
    path_to_coord = {
        path: builder.joint_q_start[idx] + merged_dof_offset.get(path, 0)
        for path, idx in path_joint_map.items()
        if idx < len(builder.joint_q_start)
    }
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        prim_path = str(prim.GetPath())
        if any(re.match(pattern, prim_path) for pattern in ignore_paths):
            continue
        parsed = parse_actuator_prim(prim)
        if parsed is None:
            continue
        target_path = parsed.target_path
        if target_path not in path_to_dof:
            raise ValueError(
                f"Actuator prim {prim.GetPath()} targets '{target_path}' which does not resolve to a known joint DOF"
            )
        joint_idx = path_joint_map[target_path]
        dof_start = builder.joint_qd_start[joint_idx]
        next_start = (
            builder.joint_qd_start[joint_idx + 1]
            if joint_idx + 1 < len(builder.joint_qd_start)
            else builder.joint_dof_count
        )
        if next_start - dof_start != 1:
            raise ValueError(
                f"Actuator prim {prim.GetPath()} targets '{target_path}' which has "
                f"{next_start - dof_start} DOF(s); only 1-DOF joints (Revolute/Prismatic) are supported"
            )
        dof_index = path_to_dof[target_path]
        coord_index = path_to_coord.get(target_path)
        pos_index = coord_index if coord_index is not None and coord_index != dof_index else None

        delay_val = None
        clamping_specs = []
        for comp_class, comp_kwargs in parsed.component_specs:
            if comp_class is Delay:
                delay_val = comp_kwargs.get("delay_steps")
            else:
                clamping_specs.append((comp_class, comp_kwargs))

        builder.add_actuator(
            parsed.controller_class,
            index=dof_index,
            clamping=clamping_specs if clamping_specs else None,
            delay_steps=delay_val,
            pos_index=pos_index,
            **parsed.controller_kwargs,
        )
        actuator_count += 1
    if verbose and actuator_count > 0:
        print(f"Added {actuator_count} actuator(s) from USD")

    result = {
        "fps": stage.GetFramesPerSecond(),
        "duration": stage.GetEndTimeCode() - stage.GetStartTimeCode(),
        "up_axis": stage_up_axis,
        "path_body_map": path_body_map,
        "path_joint_map": path_joint_map,
        "path_shape_map": path_shape_map,
        "path_shape_scale": path_shape_scale,
        "mass_unit": mass_unit,
        "linear_unit": linear_unit,
        "scene_attributes": scene_attributes,
        "physics_dt": physics_dt,
        "collapse_results": collapse_results,
        "schema_attrs": R.schema_attrs,
        # "articulation_roots": articulation_roots,
        # "articulation_bodies": articulation_bodies,
        "path_body_relative_transform": path_body_relative_transform,
        "max_solver_iterations": max_solver_iters,
        "actuator_count": actuator_count,
    }

    # Process custom frequencies with USD prim filters
    # Collect frequencies with filters and their attributes, then traverse the imported subtree once
    frequencies_with_filters = []
    for freq_key, freq_obj in builder.custom_frequencies.items():
        if freq_obj.usd_prim_filter is None:
            continue
        freq_attrs = [attr for attr in builder.custom_attributes.values() if attr.frequency == freq_key]
        if not freq_attrs:
            continue
        frequencies_with_filters.append((freq_key, freq_obj, freq_attrs))

    # Traverse the requested root subtree once and check all filters for each prim
    # Use TraverseInstanceProxies to include prims under instanceable prims
    if frequencies_with_filters:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path), Usd.TraverseInstanceProxies()):
            prim_path = str(prim.GetPath())
            if any(re.match(pattern, prim_path) for pattern in ignore_paths):
                continue
            for freq_key, freq_obj, freq_attrs in frequencies_with_filters:
                # Build per-frequency callback context and pass the same object to
                # usd_prim_filter and usd_entry_expander.
                callback_context = {"prim": prim, "result": result, "builder": builder}

                try:
                    matches_frequency = freq_obj.usd_prim_filter(prim, callback_context)
                except Exception as e:
                    raise RuntimeError(
                        f"usd_prim_filter for frequency '{freq_key}' raised an error on prim '{prim.GetPath()}': {e}"
                    ) from e
                if not matches_frequency:
                    continue

                if freq_obj.usd_entry_expander is not None:
                    try:
                        expanded_rows = list(freq_obj.usd_entry_expander(prim, callback_context))
                    except Exception as e:
                        raise RuntimeError(
                            f"usd_entry_expander for frequency '{freq_key}' raised an error on prim '{prim.GetPath()}': {e}"
                        ) from e
                    values_rows = [{attr.key: row.get(attr.key, None) for attr in freq_attrs} for row in expanded_rows]
                    builder.add_custom_values_batch(values_rows)
                    if verbose and len(expanded_rows) > 0:
                        print(
                            f"Parsed custom frequency '{freq_key}' from prim {prim.GetPath()} with {len(expanded_rows)} rows"
                        )
                    continue

                prim_custom_attrs = usd.get_custom_attribute_values(
                    prim,
                    freq_attrs,
                    context={"result": result, "builder": builder},
                )

                # Build a complete values dict for all attributes in this frequency
                # Use None for missing values so add_custom_values can apply defaults
                values_dict = {}
                for attr in freq_attrs:
                    # Use authored value if present, otherwise None (defaults applied at finalize)
                    values_dict[attr.key] = prim_custom_attrs.get(attr.key, None)

                # Always add values for this prim to increment the frequency count,
                # even if all values are None (defaults will be applied during finalization)
                builder.add_custom_values(**values_dict)
                if verbose:
                    print(f"Parsed custom frequency '{freq_key}' from prim {prim.GetPath()}")

    # USD MjcActuator does not preserve the original MJCF authoring tag:
    # MuJoCo's compiler expands <position>/<velocity> shortcuts into raw
    # gain/bias/dyntype fields before USD export, so a <position kp=K> and a
    # hand-written <general> with the same gains produce bit-identical prims.
    # We can't recover the author's intent, so we fix a contract:
    #
    #   USD MjcActuator rows targeting a joint DOF with the position/velocity
    #   shape and default dyntype/gaintype/gear are imported as JOINT_TARGET
    #   and driven by Control.joint_target_q / joint_target_qd (or the legacy
    #   joint_target_pos / joint_target_vel aliases under the DOF layout).
    #
    # Rows that author non-default dyntype (filter, integrator, ...), gaintype,
    # gear, or carry an unresolved dampratio placeholder (positive biasprm[2])
    # stay CTRL_DIRECT, because JOINT_TARGET would silently drop those features
    # when _init_actuators rebuilds the MuJoCo actuators. Tendon/site/body
    # targets and synthesized per-axis spherical DOF labels also stay
    # CTRL_DIRECT (they don't appear in path_to_dof).
    #
    # Note: per-axis prim paths from joints that were merged into a D6 (the
    # cycle-detection fix from #2557) ARE in path_to_dof and map to single
    # DOFs of the merged joint, so they convert just like single-DOF
    # revolutes -- mirroring how the MJCF importer uses mjcf_joint_name_to_dof
    # to target specific DOFs in combined joints (see import_mjcf.py).
    if "mujoco:actuator_target_label" in builder.custom_attributes:
        mjc_actuator_count = builder._custom_frequency_counts.get("mujoco:actuator", 0)
    else:
        mjc_actuator_count = 0

    if mjc_actuator_count > 0:
        # Lazy imports: only needed when MuJoCo custom attributes are registered
        # (i.e. SolverMuJoCo is in use), and avoids a top-level mujoco dependency
        # for USD parsing in non-MuJoCo configurations.
        import mujoco

        from ..solvers.mujoco.solver_mujoco import SolverMuJoCo  # noqa: PLC0415

        biastype_affine = int(mujoco.mjtBias.mjBIAS_AFFINE)
        dyntype_none = int(mujoco.mjtDyn.mjDYN_NONE)
        gaintype_fixed = int(mujoco.mjtGain.mjGAIN_FIXED)
        ctrl_source_joint_target = int(SolverMuJoCo.CtrlSource.JOINT_TARGET)

        def _row(key: str, row: int) -> Any:
            """Row value from a custom-frequency attribute, falling back to its default."""
            attr = builder.custom_attributes[key]
            value = attr.values[row] if row < len(attr.values) else None
            return attr.default if value is None else value

        converted = 0

        for row in range(mjc_actuator_count):
            target_path = _row("mujoco:actuator_target_label", row)
            dof = path_to_dof.get(target_path) if target_path else None
            if dof is None:
                continue

            # Convert only when JOINT_TARGET would not silently drop semantically
            # important authored features. _init_actuators rebuilds JOINT_TARGET
            # actuators with default dyntype/gaintype/biastype/gear, so non-default
            # values for those force the actuator to stay CTRL_DIRECT.
            #
            # ctrlrange/forcerange don't gate: the rebuild re-attaches them
            # (see joint_target_ranges in _init_actuators). Effort limit
            # (jnt_actfrcrange) comes from the joint, not the actuator.
            if (
                int(_row("mujoco:actuator_biastype", row)) != biastype_affine
                or int(_row("mujoco:actuator_dyntype", row)) != dyntype_none
                or int(_row("mujoco:actuator_gaintype", row)) != gaintype_fixed
            ):
                continue
            gear = list(_row("mujoco:actuator_gear", row))
            if not (np.isclose(gear[0], 1.0) and all(np.isclose(g, 0.0) for g in gear[1:])):
                continue

            gainprm = list(_row("mujoco:actuator_gainprm", row))
            biasprm = list(_row("mujoco:actuator_biasprm", row))
            kp = gainprm[0]
            if kp <= 0.0:
                continue

            # MuJoCo "position" shortcut: gainprm=[kp,0,...], biasprm=[0,-kp,(-kv|0),0,...].
            # A positive biasprm[2] is a dampratio placeholder that MuJoCo's compiler
            # resolves via mj_setConst; leaving such rows CTRL_DIRECT preserves that path.
            # MuJoCo "velocity" shortcut: gainprm=[kv,0,...], biasprm=[0,0,-kv,0,...].
            is_position = np.isclose(biasprm[0], 0.0) and np.isclose(biasprm[1], -kp) and biasprm[2] <= 0.0
            is_velocity = np.isclose(biasprm[0], 0.0) and np.isclose(biasprm[1], 0.0) and np.isclose(biasprm[2], -kp)
            if not (is_position or is_velocity):
                continue

            current_mode = builder.joint_target_mode[dof]
            if is_position:
                builder.joint_target_ke[dof] = kp
                if current_mode == int(JointTargetMode.VELOCITY):
                    builder.joint_target_mode[dof] = int(JointTargetMode.POSITION_VELOCITY)
                elif current_mode == int(JointTargetMode.NONE):
                    builder.joint_target_mode[dof] = int(JointTargetMode.POSITION)
                    builder.joint_target_kd[dof] = -biasprm[2]  # 0 or kv from biasprm=[0,-kp,-kv,...]
            else:  # velocity
                builder.joint_target_kd[dof] = kp
                if current_mode == int(JointTargetMode.POSITION):
                    builder.joint_target_mode[dof] = int(JointTargetMode.POSITION_VELOCITY)
                elif current_mode == int(JointTargetMode.NONE):
                    builder.joint_target_mode[dof] = int(JointTargetMode.VELOCITY)

            # Override the row's CTRL_DIRECT default and write the DOF target index
            # so _init_actuators routes through MuJoCo's joint_target_mode actuators.
            builder.custom_attributes["mujoco:ctrl_source"].values[row] = ctrl_source_joint_target
            builder.custom_attributes["mujoco:actuator_trnid"].values[row] = wp.vec2i(dof, 0)
            # Record the kind classified above so the solver doesn't re-derive it.
            builder.custom_attributes["mujoco:ctrl_type"].values[row] = int(
                SolverMuJoCo.CtrlType.POSITION if is_position else SolverMuJoCo.CtrlType.VELOCITY
            )

            converted += 1

        if verbose and converted > 0:
            print(f"Mapped {converted} MuJoCo USD actuator(s) to joint targets")
    if return_deformable_results:
        # The deformable results are opt-in so the default return shape carries no
        # deformable additions and stays isolated from changes to this experimental contract.
        result.update(
            {
                "path_cable_map": path_cable_map,
                "path_cloth_map": path_cloth_map,
                "path_soft_map": path_soft_map,
                "path_cable_attrs": path_cable_attrs,
                "path_cloth_attrs": path_cloth_attrs,
                "path_soft_attrs": path_soft_attrs,
                "path_attachment_map": path_attachment_map,
                "path_attachment_attrs": path_attachment_attrs,
            }
        )

    return result


def resolve_usd_from_url(url: str, target_folder_name: str | None = None, export_usda: bool = False) -> str:
    """Download a USD file from a URL and resolves all references to other USD files to be downloaded to the given target folder.

    Args:
        url: URL to the USD file.
        target_folder_name: Target folder name. If ``None``, a time-stamped
          folder will be created in the current directory.
        export_usda: If ``True``, converts each downloaded USD file to USDA and
          saves the additional USDA file in the target folder with the same
          base name as the original USD file.

    Returns:
        File path to the downloaded USD file.
    """

    import requests

    try:
        from pxr import Usd
    except ImportError as e:
        raise ImportError("Failed to import pxr. Please install USD (e.g. via `pip install usd-core`).") from e

    def _download_https_url(source_url: str):
        """Download a URL while validating every redirect target is HTTPS."""
        current_url = source_url
        request_timeout_s = 30
        for _ in range(10):
            _validate_https_usd_url(current_url)
            response = requests.get(current_url, allow_redirects=False, timeout=request_timeout_s)
            if int(response.status_code) in {301, 302, 303, 307, 308}:
                redirect_url = response.headers.get("Location")
                if not redirect_url:
                    return response, current_url
                current_url = urljoin(current_url, redirect_url)
                continue
            final_url = getattr(response, "url", current_url)
            if not isinstance(final_url, str):
                final_url = current_url
            _validate_https_usd_url(final_url)
            return response, final_url
        raise RuntimeError(f"Too many redirects while downloading USD file: {source_url}")

    response, resolved_url = _download_https_url(url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download USD file. Status code: {response.status_code}")
    file = response.content
    dot = os.path.extsep
    base = posixpath.basename(urlparse(resolved_url).path)
    url_folder = posixpath.dirname(resolved_url)
    base_name = dot.join(base.split(dot)[:-1])
    if target_folder_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        target_folder_name = os.path.join(".usd_cache", f"{base_name}_{timestamp}")
    os.makedirs(target_folder_name, exist_ok=True)
    target_filename = os.path.join(target_folder_name, base)
    with open(target_filename, "wb") as f:
        f.write(file)

    stage = Usd.Stage.Open(target_filename, Usd.Stage.LoadNone)
    root_layer = stage.GetRootLayer()
    stage_str = root_layer.ExportToString()
    print(f"Downloaded USD file to {target_filename}.")

    # Recursively resolve referenced USD files like `references = @./franka_collisions.usd@`
    # Each entry in the queue is (resolved_url, cache_relative_path).
    downloaded_urls: set[str] = {url, resolved_url}
    pending: collections.deque[tuple[str, str]] = collections.deque()

    def _write_layer_string(filename: str, layer, layer_str: str) -> None:
        """Persist rewritten USDA text to both the layer and cache file."""
        import_from_string = getattr(layer, "ImportFromString", None)
        if callable(import_from_string):
            import_from_string(layer_str)
            save = getattr(layer, "Save", None)
            if callable(save):
                save()
        with open(filename, "w") as f:
            f.write(layer_str)

    def _extract_references(layer_str, parent_url_folder, parent_local_folder):
        """Extract references, queue downloads, and return rewritten layer text."""
        rewritten_layer_str = layer_str
        for match in re.finditer(r"references.=.@(.*?)@", layer_str):
            raw_ref = match.group(1)
            ref_url = urljoin(parent_url_folder + "/", raw_ref)
            raw_ref_scheme = urlparse(raw_ref).scheme
            if raw_ref_scheme in {"http", "https"}:
                _validate_https_usd_url(ref_url)
                local_path = _cache_path_for_absolute_usd_reference(ref_url)
                rewritten_layer_str = rewritten_layer_str.replace(f"@{raw_ref}@", f"@{local_path}@")
            else:
                local_path = posixpath.normpath(posixpath.join(parent_local_folder, raw_ref))
            if posixpath.isabs(local_path) or local_path == ".." or local_path.startswith("../"):
                print(f"Skipping reference that escapes target folder: {raw_ref}")
                continue
            if ref_url not in downloaded_urls:
                pending.append((ref_url, local_path))
        return rewritten_layer_str

    rewritten_stage_str = _extract_references(stage_str, url_folder, "")
    if rewritten_stage_str != stage_str:
        _write_layer_string(target_filename, root_layer, rewritten_stage_str)
        stage_str = rewritten_stage_str

    if export_usda:
        usda_filename = os.path.join(target_folder_name, base_name + ".usda")
        with open(usda_filename, "w") as f:
            f.write(stage_str)
            print(f"Exported USDA file to {usda_filename}.")

    while pending:
        ref_url, local_path = pending.popleft()
        if ref_url in downloaded_urls:
            continue
        downloaded_urls.add(ref_url)
        try:
            response, resolved_ref_url = _download_https_url(ref_url)
            if response.status_code != 200:
                print(f"Failed to download reference {local_path}. Status code: {response.status_code}")
                continue
            downloaded_urls.add(resolved_ref_url)
            file = response.content
            local_dir = posixpath.dirname(local_path)
            if local_dir:
                os.makedirs(os.path.join(target_folder_name, local_dir), exist_ok=True)
            ref_filename = os.path.join(target_folder_name, local_path)
            if not os.path.exists(ref_filename):
                with open(ref_filename, "wb") as f:
                    f.write(file)
            print(f"Downloaded USD reference {local_path} to {ref_filename}.")

            ref_stage = Usd.Stage.Open(ref_filename, Usd.Stage.LoadNone)
            ref_layer = ref_stage.GetRootLayer()
            ref_stage_str = ref_layer.ExportToString()

            rewritten_ref_stage_str = _extract_references(ref_stage_str, posixpath.dirname(resolved_ref_url), local_dir)
            if rewritten_ref_stage_str != ref_stage_str:
                _write_layer_string(ref_filename, ref_layer, rewritten_ref_stage_str)
                ref_stage_str = rewritten_ref_stage_str

            if export_usda:
                ref_base = os.path.basename(ref_filename)
                ref_base_name = dot.join(ref_base.split(dot)[:-1])
                usda_filename = (
                    os.path.join(target_folder_name, local_dir, ref_base_name + ".usda")
                    if local_dir
                    else os.path.join(target_folder_name, ref_base_name + ".usda")
                )
                with open(usda_filename, "w") as f:
                    f.write(ref_stage_str)
                    print(f"Exported USDA file to {usda_filename}.")
        except ValueError:
            raise
        except Exception:
            print(f"Failed to download {local_path}.")
    return target_filename


def _raise_on_stage_errors(usd_stage, stage_source: str):
    get_errors = getattr(usd_stage, "GetCompositionErrors", None)
    if get_errors is None:
        return
    errors = get_errors()
    if not errors:
        return
    messages = []
    for err in errors:
        try:
            messages.append(err.GetMessage())
        except Exception:
            messages.append(str(err))
    formatted = "\n".join(f"- {message}" for message in messages)
    raise RuntimeError(f"USD stage has composition errors while loading {stage_source}:\n{formatted}")
