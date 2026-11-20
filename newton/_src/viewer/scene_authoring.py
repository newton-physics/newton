# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""An opt-in, document-owned rigid scene with transactional runtime replacement."""

from __future__ import annotations

import dataclasses
import enum
import io
import logging
import math
import pickle
import time
import uuid
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from ..geometry import GeoType, ShapeFlags
from ..geometry.bvh import (
    compute_box_bounds,
    compute_capsule_bounds,
    compute_cylinder_bounds,
    compute_shape_bounds,
    compute_sphere_bounds,
)
from ..sim import BodyFlags, CollisionPipeline, JointType, Model, ModelBuilder, ModelFlags, eval_ik
from ..sim.collide import compute_shape_aabbs
from ..solvers.xpbd import SolverXPBD
from .authoring_usd import (
    _GROUND_FRICTION,
    _SPHERE_ROLLING_FRICTION,
    NotAuthoringSceneError,
    _asset_dependencies,
    _validate_definition,
    load_scene,
    prepare_usd,
    save_scene,
    template_bounds,
)

if TYPE_CHECKING:
    from ..sim import Contacts, Control, State
    from ..solvers.solver import SolverBase
    from .viewer import ViewerBase

logger = logging.getLogger(__name__)


@wp.kernel
def _placement_surface(
    live: wp.array[wp.transform],
    reset: wp.array[wp.transform],
    bodies: wp.array[int],
    transforms: wp.array[wp.transform],
    kinds: wp.array[int],
    scales: wp.array[wp.vec3],
    flags: wp.array[int],
    margins: wp.array[float],
    local_lower: wp.array[wp.vec3],
    local_upper: wp.array[wp.vec3],
    lower: wp.vec3,
    upper: wp.vec3,
    axis: int,
    result: wp.array[float],
):
    shape = wp.tid()
    if flags[shape] & int(ShapeFlags.COLLIDE_SHAPES) == 0:
        return
    for snapshot in range(2):
        transform = transforms[shape]
        body = bodies[shape]
        if body >= 0:
            pose = live[body]
            if snapshot == 1:
                pose = reset[body]
            transform = wp.transform_multiply(pose, transform)
        margin = margins[shape]
        if kinds[shape] == int(GeoType.PLANE) and scales[shape][0] == 0.0 and scales[shape][1] == 0.0:
            anchor = wp.transform_get_translation(transform)
            normal = wp.quat_rotate(wp.transform_get_rotation(transform), wp.vec3(0.0, 0.0, 1.0))
            if normal[axis] <= 1.0e-6:
                wp.atomic_max(result, 1, 1.0)
            else:
                rise = float(0.0)
                for lateral in range(3):
                    if lateral != axis:
                        rise += wp.max(
                            normal[lateral] * (anchor[lateral] - lower[lateral]),
                            normal[lateral] * (anchor[lateral] - upper[lateral]),
                        )
                wp.atomic_max(result, 0, anchor[axis] + rise / normal[axis] + margin)
        else:
            lo, hi = compute_shape_bounds(transform, wp.vec3(1.0), local_lower[shape], local_upper[shape])
            if kinds[shape] == int(GeoType.PLANE):
                lo, hi = compute_box_bounds(transform, wp.vec3(scales[shape][0] * 0.5, scales[shape][1] * 0.5, 0.0))
            elif kinds[shape] == int(GeoType.SPHERE):
                lo, hi = compute_sphere_bounds(wp.transform_get_translation(transform), scales[shape][0])
            elif kinds[shape] == int(GeoType.CAPSULE):
                lo, hi = compute_capsule_bounds(transform, scales[shape])
            elif kinds[shape] == int(GeoType.CYLINDER):
                lo, hi = compute_cylinder_bounds(transform, scales[shape])
            overlaps = bool(True)
            for lateral in range(3):
                if lateral != axis:
                    if hi[lateral] + margin < lower[lateral] - 0.001:
                        overlaps = False
                    if lo[lateral] - margin > upper[lateral] + 0.001:
                        overlaps = False
            if overlaps:
                wp.atomic_max(result, 0, hi[axis] + margin)


@dataclasses.dataclass(frozen=True)
class _AssetTemplate:
    builder: ModelBuilder
    bounds: tuple[np.ndarray, np.ndarray] | None
    dependencies: tuple[tuple[str, int | None, int | None], ...]


def _dependency_snapshot(paths):
    result = []
    for path in paths:
        try:
            stat = Path(path).stat()
            result.append((path, stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            result.append((path, None, None))
    return tuple(result)


def _template_is_current(template: _AssetTemplate) -> bool:
    return template.dependencies == _dependency_snapshot(path for path, _, _ in template.dependencies)


def _restore_template_dtype(constructor: str, arguments: dict):
    return getattr(wp.types, constructor)(**arguments)


class _TemplatePickler(pickle.Pickler):
    """Transfer host templates without changing global Warp or pickle behavior."""

    def reducer_override(self, value):
        if isinstance(value, type) and "<locals>" in value.__qualname__:
            constructor = getattr(value, "_wp_constructor_", None)
            arguments = getattr(value, "_wp_type_args_", None)
            if constructor in ("transformation", "vector", "matrix", "quaternion") and arguments is not None:
                return _restore_template_dtype, (constructor, arguments)
        return NotImplemented


def _init_asset_worker() -> None:
    # ScopedDevice/ScopedStream change process-wide Warp state. Keep all native
    # USD preparation in this process, separate from the rendering owner.
    wp.config.quiet = True
    wp.set_device("cpu")


def _prepare_asset_payload(source: str, size: tuple, mujoco: bool = False) -> bytes:
    try:
        from pxr import Sdf

        # Native layer caches belong to this CPU child, never the UI process.
        for path in _asset_dependencies(source):
            layer = Sdf.Layer.Find(path)
            if layer is not None:
                layer.Reload()
        template = SceneEditor._prepare_template(source, size, mujoco=mujoco)
        # Parsing is complete. Local importer callbacks are not pickleable;
        # retain values and reference remappers needed by native add_builder.
        template.builder.custom_attributes = {
            key: dataclasses.replace(
                attribute, usd_value_transformer=None, mjcf_value_transformer=None, urdf_value_transformer=None
            )
            for key, attribute in template.builder.custom_attributes.items()
        }
        buffer = io.BytesIO()
        _TemplatePickler(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump(template)
        return buffer.getvalue()
    except Exception as error:
        # Tf.ErrorException contains unpickleable native diagnostics.
        raise ValueError(str(error)) from None


def _pose_delta(target: wp.transform, current: wp.transform) -> wp.transform:
    if tuple(target.q) == tuple(current.q):
        return wp.transform(target.p - current.p, wp.quat_identity())
    rotation = wp.normalize(wp.mul(target.q, wp.quat_inverse(current.q)))
    # Use the kernel's same float32 rotation when cancelling the current
    # position; two successive rotations otherwise accumulate translation bias.
    translation = target.p - wp.quat_rotate(rotation, current.p)
    return wp.transform(translation, rotation)


@wp.func
def _compose_pose(parent: wp.transform, local: wp.transform) -> wp.transform:
    pose = wp.transform_multiply(parent, local)
    # Subsequent inverse transforms assume unit quaternions; repeated edits
    # otherwise amplify float32 roundoff into invalid orientations.
    return wp.transform(wp.transform_get_translation(pose), wp.normalize(wp.transform_get_rotation(pose)))


@wp.kernel
def _move_bodies(
    start: int,
    delta: wp.transform,
    initial_delta: wp.transform,
    initial: wp.array[wp.transform],
    poses: wp.array[wp.transform],
    other: wp.array[wp.transform],
    velocity: wp.array[wp.spatial_vector],
    other_velocity: wp.array[wp.spatial_vector],
    clear_velocity: bool,
):
    index = start + wp.tid()
    pose = _compose_pose(delta, poses[index])
    poses[index] = pose
    other[index] = pose
    if clear_velocity:
        velocity[index] = wp.spatial_vector()
        other_velocity[index] = wp.spatial_vector()
    initial[index] = _compose_pose(initial_delta, initial[index])


@wp.kernel
def _move_world_shapes(
    start: int,
    delta: wp.transform,
    bodies: wp.array[int],
    transforms: wp.array[wp.transform],
):
    index = start + wp.tid()
    if bodies[index] == -1:
        transforms[index] = _compose_pose(delta, transforms[index])


@wp.kernel
def _move_world_anchors(
    start: int,
    delta: wp.transform,
    parents: wp.array[int],
    kinds: wp.array[int],
    anchors: wp.array[wp.transform],
):
    index = start + wp.tid()
    if parents[index] == -1 and kinds[index] != int(JointType.FREE):
        anchors[index] = _compose_pose(delta, anchors[index])


@wp.kernel
def _appearance(
    start: int,
    tint: wp.vec3,
    visible: bool,
    friction: float,
    base_colors: wp.array[wp.vec3],
    base_opacity: wp.array[float],
    colors: wp.array[wp.vec3],
    opacity: wp.array[float],
    coefficients: wp.array[float],
):
    index = start + wp.tid()
    colors[index] = wp.cw_mul(base_colors[index], tint)
    opacity[index] = wp.where(visible, base_opacity[index], 0.0)
    coefficients[index] = friction


@wp.kernel
def _copy_bodies(
    old_indices: wp.array[int],
    new_indices: wp.array[int],
    deltas: wp.array[wp.transform],
    clear_velocity: wp.array[bool],
    old_q: wp.array[wp.transform],
    old_qd: wp.array[wp.spatial_vector],
    q: wp.array[wp.transform],
    qd: wp.array[wp.spatial_vector],
    other_q: wp.array[wp.transform],
    other_qd: wp.array[wp.spatial_vector],
):
    thread = wp.tid()
    old = old_indices[thread]
    new = new_indices[thread]
    pose = wp.transform_multiply(deltas[thread], old_q[old])
    velocity = old_qd[old]
    if clear_velocity[thread]:
        velocity = wp.spatial_vector()
    q[new] = pose
    qd[new] = velocity
    other_q[new] = pose
    other_qd[new] = velocity


@dataclasses.dataclass(frozen=True)
class _Binding:
    bodies: tuple[int, int]
    shapes: tuple[int, int]
    joints: tuple[int, int]
    reference: tuple[float, ...] | None
    follows_body: bool


@dataclasses.dataclass
class _Runtime:
    model: Model
    state_0: State
    state_1: State
    control: Control
    solver: SolverBase | None
    pipeline: CollisionPipeline
    contacts: Contacts
    bindings: dict[str, _Binding]
    objects: dict[str, Any]
    base_colors: wp.array[wp.vec3]
    base_opacity: wp.array[float]
    base_friction: wp.array[float]
    observables: SolverBase.Observables | None = None
    solver_graphs: dict[tuple, tuple[Any, tuple]] = dataclasses.field(default_factory=dict)
    solver_warm_binding: tuple | None = None
    contacts_dirty: bool = True


class SceneEditor:
    """Author a rigid scene while owning its simulation and compiled resources.

    This entire class is **experimental**. It owns an XPBD or MuJoCo scene,
    rather than mutating another application's solver or graphs. MuJoCo uses
    its native CPU backend on CPU and MuJoCo Warp on CUDA.
    Positions and dimensions use metres, rotations use XYZW quaternions, and
    masses use kilograms. Pose, color, visibility, and friction edits retain
    Model/State storage. Add/remove, geometry, mass, and motion-type changes
    create one candidate runtime when committed, never on every drag sample.

    Use :meth:`before_frame` before stepping or rendering, including when
    paused. The example runner does this automatically. Exported documents use
    standard OpenUSD geometry and physics schemas. USD conversion is optional
    and loaded only when importing/exporting an asset.
    """

    class Solver(enum.Enum):
        """Physics backends supported by the experimental authoring scene."""

        XPBD = "xpbd"
        MUJOCO = "mujoco"

    class Shape(enum.Enum):
        """Geometry categories supported by the experimental authoring scene."""

        BOX = "box"
        SPHERE = "sphere"
        CAPSULE = "capsule"
        CYLINDER = "cylinder"
        USD = "usd"

    @dataclasses.dataclass(frozen=True)
    class Object:
        """Immutable authored values; live simulation poses remain in State."""

        id: str
        """Stable document identity, retained across runtime replacements."""
        name: str
        """Display name, independent of physical body labels."""
        kind: SceneEditor.Shape
        """Primitive type or a referenced USD asset."""
        position: tuple[float, float, float] = (0.0, 0.0, 1.0)
        """Authored reset placement in the Z-up scene [m]."""
        rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
        """Authored reset orientation as an XYZW quaternion."""
        size: tuple[float, float, float] = (1.0, 1.0, 1.0)
        """Full primitive XYZ dimensions [m], or multiplicative USD scale."""
        color: tuple[float, float, float] = (0.55, 0.72, 0.85)
        """RGB tint in [0, 1]; white preserves a USD asset's original colors."""
        mass: float = 1.0
        """Primitive mass or total USD rigid-body mass [kg]."""
        friction: float = 0.6
        """Nonnegative surface friction coefficient."""
        dynamic: bool = True
        """Whether physics advances the object; false makes it kinematic."""
        source: str | None = None
        """Absolute root USD path, present only for USD assets."""
        visible: bool = True
        """Rendering visibility, independent of collision participation."""
        overrides: tuple[str, ...] = ()
        """Explicit USD mass, friction, or dynamic overrides; otherwise retain native values."""

    def __init__(
        self,
        viewer: ViewerBase,
        *,
        device: str | wp.Device | None = None,
        solver: Solver | str = Solver.XPBD,
    ) -> None:
        """Create an empty document without compiling a model or starting network work.

        Args:
            viewer: Viewer receiving the owned model, state, and interaction forces.
            device: Simulation device; defaults to the current Warp device.
            solver: Fixed physics backend. MuJoCo uses native contacts on CPU
                and Newton contacts with MuJoCo Warp on CUDA.
        """
        self.viewer = viewer
        self.device = wp.get_device(device)
        self._solver_type = self.Solver(solver)
        self._mujoco_cpu = self._solver_type == self.Solver.MUJOCO and self.device.is_cpu
        self.scene_editor = self
        self.frame_dt = 1.0 / 60.0
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.model = self.state_0 = self.state_1 = self.control = self.solver = self.contacts = None
        self.collision_pipeline = None
        self.selected_id: str | None = None
        self.auto_apply = True
        self.revision = 0
        self._document_epoch = 0
        self.rebuild_count = 0
        self.last_build_ms = 0.0
        self.error: str | None = None
        self.status = "Add a primitive or import a USD asset."
        self._objects: tuple[SceneEditor.Object, ...] = ()
        self._runtime: _Runtime | None = None
        self._topology_dirty = True
        self._fast_pending: dict[str, SceneEditor.Object] = {}
        self._placement_requests: set[str] = set()
        self._undo: list[tuple[SceneEditor.Object, ...]] = []
        self._redo: list[tuple[SceneEditor.Object, ...]] = []
        self._edit_before: tuple[SceneEditor.Object, ...] | None = None
        self._requested_commit = False
        self._failed_attempt: tuple[tuple, str, int] | None = None
        self._templates: dict[tuple[Any, ...], _AssetTemplate] = {}
        self._preparing: dict[tuple[Any, ...], Future] = {}
        self._executor: ProcessPoolExecutor | None = None
        self._closed = False
        self._preview: tuple | None = None
        self._preview_last: tuple | None = None
        self._preview_buffers = None
        self._preview_source: SceneEditor.Object | None = None
        self._preview_error: tuple | None = None
        self._placement_result = None
        self._placement_cache = None
        attach = getattr(getattr(viewer, "gui", None), "set_scene_editor", None)
        if attach is not None:
            attach(self)

    @property
    def solver_type(self) -> Solver:
        """Return the fixed physics backend selected when the editor was created."""
        return self._solver_type

    @property
    def objects(self) -> tuple[Object, ...]:
        """Return authored objects in stable document order."""
        return self._objects

    @property
    def pending(self) -> bool:
        """Report document changes awaiting publication to the physical scene."""
        return self._topology_dirty or bool(self._fast_pending)

    @property
    def busy(self) -> bool:
        """Report background USD preparation required by the authored document."""
        active = {(obj.source, obj.size) for obj in self._objects if obj.kind == self.Shape.USD}
        return any(key[:2] in active and not future.done() for key, future in self._preparing.items())

    @property
    def can_undo(self) -> bool:
        """Report whether an authored edit can be undone."""
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        """Report whether an authored edit can be redone."""
        return bool(self._redo)

    def get_object(self, object_id: str) -> Object:
        """Resolve a stable document ID, rejecting stale compiled indices."""
        return next((obj for obj in self._objects if obj.id == object_id), None) or self._missing(object_id)

    @staticmethod
    def _missing(object_id: str):
        raise KeyError(f"Unknown scene object {object_id!r}")

    def select(self, object_id: str | None) -> None:
        """Select an authored object without changing simulation state."""
        if object_id is not None:
            self.get_object(object_id)
        self.selected_id = object_id

    @staticmethod
    def _validate(obj: Object) -> Object:
        definition = vars(obj).copy()
        definition["kind"] = obj.kind.value
        _validate_definition(definition, normalize_rotation=True)
        return dataclasses.replace(
            obj, **{name: definition[name] for name in ("position", "rotation", "size", "color", "overrides")}
        )

    @staticmethod
    def _vector(values, count: int, name: str) -> tuple:
        result = tuple(float(value) for value in values)
        if len(result) != count or not all(math.isfinite(value) for value in result):
            raise ValueError(f"{name} requires {count} finite components")
        return result

    def _remember(self) -> None:
        if self._edit_before is None:
            self._undo.append(self._objects)
            del self._undo[:-100]
        self._redo.clear()

    def _changed(self) -> None:
        self.revision += 1
        self.error = None

    def add_primitive(
        self,
        kind: Shape | str,
        *,
        position: tuple[float, float, float] = (0.0, 0.0, 1.0),
        size: tuple[float, float, float] = (1.0, 1.0, 1.0),
        name: str | None = None,
        color: tuple[float, float, float] = (0.55, 0.72, 0.85),
        dynamic: bool = True,
        mass: float = 1.0,
    ) -> str:
        """Stage a primitive without finalizing a model.

        Args:
            kind: Box, sphere, capsule, or cylinder category.
            position: Initial center in the Z-up scene [m].
            size: Full XYZ extent [m]. Capsule height includes both hemispheres.
            name: Display name, defaulting to the shape category.
            color: RGB color in [0, 1].
            dynamic: Whether physics advances the object; false makes it kinematic.
            mass: Total mass [kg].

        Returns:
            Stable object ID, valid across structural commits.
        """
        kind = self.Shape(kind)
        if kind == self.Shape.USD:
            raise ValueError("Use add_usd() for USD assets")
        obj = self._validate(
            self.Object(
                uuid.uuid4().hex,
                name or kind.value.title(),
                kind,
                position=position,
                size=size,
                color=color,
                dynamic=dynamic,
                mass=mass,
            )
        )
        self._remember()
        self._objects += (obj,)
        self.selected_id = obj.id
        self._topology_dirty = True
        self._changed()
        return obj.id

    def add_usd(
        self,
        source: str | Path,
        *,
        position: tuple[float, float, float] = (0.0, 0.0, 1.0),
        name: str | None = None,
    ) -> str:
        """Stage a local USD package; preparation runs independently of UI drawing.

        Args:
            source: Root USD path; referenced dependencies must be available locally.
            position: Initial root placement in the Z-up scene [m].
            name: Display name, defaulting to the source filename stem.

        Returns:
            Stable object ID. Import failures are reported by :attr:`error`.
        """
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        obj = self._validate(
            self.Object(
                uuid.uuid4().hex,
                name or path.stem,
                self.Shape.USD,
                position=position,
                color=(1.0, 1.0, 1.0),
                source=str(path),
            )
        )
        self._remember()
        self._objects += (obj,)
        self.selected_id = obj.id
        self._topology_dirty = True
        self._changed()
        self._schedule_template(obj)
        return obj.id

    def update_object(self, object_id: str, **fields) -> None:
        """Queue supported edits; transforms and appearance retain compiled storage.

        Args:
            object_id: Stable authored identity.
            **fields: Editable :class:`Object` values. ID, kind, source, and the
                internal override record cannot be changed. Size, mass, and
                dynamic require a structural commit. Placement edits target
                the live pose and also update the authored reset pose.
        """
        before = self.get_object(object_id)
        allowed = {field.name for field in dataclasses.fields(before)} - {"id", "kind", "source", "overrides"}
        if unknown := set(fields) - allowed:
            raise ValueError(f"Unsupported authored fields: {', '.join(sorted(unknown))}")
        after = self._validate(dataclasses.replace(before, **fields))
        if after.kind == self.Shape.USD:
            overrides = set(before.overrides) | (set(fields) & {"mass", "friction", "dynamic"})
            after = dataclasses.replace(after, overrides=tuple(sorted(overrides)))
        if after == before:
            return
        self._remember()
        self._objects = tuple(after if item.id == object_id else item for item in self._objects)
        if self._structure(before) != self._structure(after):
            self._topology_dirty = True
        else:
            self._fast_pending.setdefault(object_id, before)
        self._changed()

    def place_object(self, object_id: str) -> None:
        """Queue conservative, collision-free insertion of a newly staged object.

        Raise its complete native shape bounds along the scene's up axis above
        occupied footprints when committed. Explicit adds and USD loads retain
        their exact positions unless this experimental operation is requested.
        Infinite planes require explicit placement instead.

        Args:
            object_id: Stable ID of an added object that has not yet been committed.
        """
        self.get_object(object_id)
        if self._runtime is not None and object_id in self._runtime.bindings:
            raise ValueError("Automatic placement requires a newly added object")
        if object_id not in self._placement_requests:
            self._placement_requests.add(object_id)
            self._topology_dirty = True
            self._changed()

    def remove_object(self, object_id: str) -> None:
        """Stage object removal, retaining the current physical scene until commit."""
        self.get_object(object_id)
        self._remember()
        self._objects = tuple(obj for obj in self._objects if obj.id != object_id)
        self._fast_pending.pop(object_id, None)
        self._placement_requests.discard(object_id)
        if self.selected_id == object_id:
            self.selected_id = None
        self._topology_dirty = True
        self._changed()

    def duplicate_object(self, object_id: str) -> str:
        """Duplicate an object's authored data while sharing its immutable asset template."""
        obj = self.get_object(object_id)
        position, rotation = self.get_pose(object_id)
        duplicate = dataclasses.replace(
            obj,
            id=uuid.uuid4().hex,
            name=obj.name + " copy",
            position=(position[0] + 0.4, position[1] + 0.4, position[2]),
            rotation=rotation,
        )
        self._remember()
        self._objects += (duplicate,)
        self.selected_id = duplicate.id
        self._topology_dirty = True
        self._changed()
        return duplicate.id

    def begin_edit(self) -> None:
        """Start one undoable gesture; delay structural compilation until release."""
        if self._edit_before is None:
            self._edit_before = self._objects

    def end_edit(self) -> None:
        """Finish a gesture without compiling on the UI callback stack."""
        if self._edit_before is not None and self._edit_before != self._objects:
            self._undo.append(self._edit_before)
            del self._undo[:-100]
        self._edit_before = None

    def _restore_document(self, objects: tuple[Object, ...]) -> None:
        objects = self._resolved_defaults(objects)
        old = {obj.id: obj for obj in self._objects}
        new = {obj.id: obj for obj in objects}
        structural = old.keys() != new.keys() or any(
            self._structure(old[key]) != self._structure(new[key]) for key in old.keys() & new.keys()
        )
        self._objects = objects
        self._placement_requests.intersection_update(new)
        self._topology_dirty |= structural
        for key in old.keys() & new.keys():
            if old[key] != new[key]:
                self._fast_pending.setdefault(key, old[key])
        if self.selected_id not in new:
            self.selected_id = None
        self._changed()

    def undo(self) -> None:
        """Restore authored values, including reset poses, without rewinding unrelated physics."""
        self.end_edit()
        if self._undo:
            self._redo.append(self._objects)
            self._restore_document(self._undo.pop())

    def redo(self) -> None:
        """Restore an undone authored edit."""
        self.end_edit()
        if self._redo:
            self._undo.append(self._objects)
            self._restore_document(self._redo.pop())

    @staticmethod
    def _structure(obj: Object) -> tuple:
        overrides = (
            tuple(name in obj.overrides for name in ("mass", "dynamic")) if obj.kind == SceneEditor.Shape.USD else ()
        )
        return obj.kind, obj.size, obj.source, obj.mass, obj.dynamic, overrides

    def _commit_signature(self) -> tuple:
        records = []
        for obj in self._objects:
            source = None
            if obj.source is not None:
                try:
                    stat = Path(obj.source).stat()
                    source = stat.st_mtime_ns, stat.st_size
                except OSError:
                    pass
            # A staged object's placement can itself prevent compilation;
            # moving a committed prop must not retry an unchanged failed asset.
            staged_pose = (
                (obj.position, obj.rotation) if self._runtime is None or obj.id not in self._runtime.bindings else None
            )
            records.append((obj.id, self._structure(obj), source, staged_pose))
        return tuple(records), frozenset(self._placement_requests)

    def _template_key(self, obj: Object) -> tuple:
        path = Path(obj.source)
        stat = path.stat()
        key = str(path), obj.size, stat.st_mtime_ns, stat.st_size
        if key in self._templates and not _template_is_current(self._templates[key]):
            del self._templates[key]
        return key

    @staticmethod
    def _prepare_template(source: str, size: tuple, *, mujoco: bool = False) -> _AssetTemplate:
        logger.info("Preparing USD asset %s", source)
        with wp.ScopedDevice("cpu"):
            dependencies = _dependency_snapshot(_asset_dependencies(source))
            builder = prepare_usd(source, scale=size, mujoco=mujoco)
            template = _AssetTemplate(builder, template_bounds(builder), dependencies)
            if not _template_is_current(template):
                raise ValueError("USD dependencies changed during preparation; retry the import")
            return template

    def _adopt_template(self, key: tuple, template: _AssetTemplate) -> tuple:
        if not _template_is_current(template):
            raise ValueError("USD dependencies changed after preparation; retry the import")
        stat = Path(key[0]).stat()
        key = (*key[:2], stat.st_mtime_ns, stat.st_size)
        for previous in list(self._templates):
            if previous[:2] == key[:2] and previous != key:
                del self._templates[previous]
        self._templates[key] = template
        return key

    def _schedule_template(self, obj: Object) -> tuple:
        key = self._template_key(obj)
        for previous in list(self._preparing):
            if previous[:2] == key[:2] and previous != key:
                self._preparing.pop(previous).cancel()
        if key in self._templates or key in self._preparing:
            return key
        if self._executor is None:
            self._executor = self._create_preparation_executor()
        try:
            self._preparing[key] = self._executor.submit(
                _prepare_asset_payload, obj.source, obj.size, self.solver_type == self.Solver.MUJOCO
            )
        except BrokenProcessPool:
            self._close_preparation_executor()
            self._executor = self._create_preparation_executor()
            self._preparing[key] = self._executor.submit(
                _prepare_asset_payload, obj.source, obj.size, self.solver_type == self.Solver.MUJOCO
            )
        return key

    @staticmethod
    def _create_preparation_executor() -> ProcessPoolExecutor:
        return ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn"), initializer=_init_asset_worker)

    def _close_preparation_executor(self) -> None:
        executor, self._executor = self._executor, None
        if executor is None:
            return
        processes = list((getattr(executor, "_processes", None) or {}).values())
        manager = getattr(executor, "_executor_manager_thread", None)
        result_queue = getattr(executor, "_result_queue", None)
        terminate = getattr(executor, "terminate_workers", None)
        if terminate is not None:
            terminate()
        else:
            # Python <3.14 has no public process-pool termination API.
            executor.shutdown(wait=False, cancel_futures=True)
            for process in processes:
                if process.is_alive():
                    process.terminate()
        for process in processes:
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
        # Release partial result reads, then settle concurrent worker exit-code publication.
        if result_queue is not None:
            result_queue._writer.close()
        if manager is not None:
            manager.join(timeout=1)

    def _trim_templates(self) -> None:
        # Retain every active source, but bound old scale variants after gestures.
        active = {(obj.source, obj.size) for obj in self._objects if obj.kind == self.Shape.USD}
        if self._preview_source is not None:
            active.add((self._preview_source.source, self._preview_source.size))
        for key in list(self._templates):
            if len(self._templates) <= 32:
                break
            if key[:2] not in active:
                del self._templates[key]

    def request_commit(self) -> None:
        """Request a structural commit at the next safe frame boundary, including retry."""
        self._requested_commit = True
        self._failed_attempt = None

    def before_frame(self) -> None:
        """Publish edits before workspace/history binding, even while physics is paused."""
        if self._closed:
            return
        self._apply_pending()
        self._refresh_contacts_for_inspection()

    def _apply_pending(self) -> None:
        """Publish prepared assets and authored changes at the frame boundary."""
        document_sources = (
            {(obj.source, obj.size) for obj in self._objects if obj.kind == self.Shape.USD} if self._preparing else ()
        )
        requested = document_sources
        if self._preview_source is not None:
            requested = {*requested, (self._preview_source.source, self._preview_source.size)}
        for key, future in list(self._preparing.items()):
            if key[:2] not in requested:
                future.cancel()
                del self._preparing[key]
                continue
            if not future.done():
                continue
            try:
                # Only our trusted CPU child produces this private host payload.
                self._adopt_template(key, pickle.loads(future.result()))
                self._trim_templates()
            except Exception as exc:
                message = f"Could not prepare USD asset {Path(key[0]).name}: {exc}"
                if key[:2] in document_sources:
                    self.error = message
                    self._failed_attempt = self._commit_signature(), message, self.revision
                if self._preview_source is not None and key[:2] == (
                    self._preview_source.source,
                    self._preview_source.size,
                ):
                    self._preview_error = key, message
                logger.error("%s", message)
            del self._preparing[key]
        if self._fast_pending:
            self._apply_fast_edits()
        if self._topology_dirty and self._failed_attempt is not None:
            signature, message, examined_revision = self._failed_attempt
            if examined_revision == self.revision or signature == self._commit_signature():
                self._failed_attempt = signature, message, self.revision
                self.error = message
                return
            self._failed_attempt = None
        if self._topology_dirty and self._edit_before is None and (self.auto_apply or self._requested_commit):
            try:
                for obj in self._objects:
                    if obj.kind == self.Shape.USD:
                        self._schedule_template(obj)
                if self.busy:
                    return
                self.commit()
            except Exception as exc:
                self.error = str(exc)
                self._failed_attempt = self._commit_signature(), self.error, self.revision
                self.status = "Could not apply scene changes. The previous simulation is retained."
                logger.error("Could not compile authored scene: %s", exc)

    def _refresh_contacts_for_inspection(self) -> None:
        runtime = self._runtime
        if runtime is None or not runtime.contacts_dirty:
            return
        workspace = getattr(getattr(self.viewer, "gui", None), "workspace", None)
        if workspace is not None and workspace.example is not self:
            workspace = None
        collision = workspace.collision if workspace is not None else None
        if workspace is not None and not workspace._at_live_frame():
            return
        if not (self.viewer.show_contacts or (collision is not None and collision._interested)):
            return
        self._collide(runtime)
        if workspace is not None:
            # Snapshot the freshly detected contacts on the next bind. Render
            # must not change live physics or a historical display snapshot.
            workspace.reset_history()
            collision.token = None

    def _template(self, obj: Object) -> ModelBuilder:
        key = self._schedule_template(obj)
        if key not in self._templates:
            future = self._preparing.pop(key)
            key = self._adopt_template(key, pickle.loads(future.result()))
        return self._templates[key].builder

    def _resolved_defaults(self, objects: tuple[Object, ...]) -> tuple[Object, ...]:
        result = []
        for original in objects:
            obj = original
            template = next(
                (
                    template.builder
                    for key, template in reversed(self._templates.items())
                    if obj.kind == self.Shape.USD and key[:2] == (obj.source, obj.size)
                ),
                None,
            )
            if template is not None:
                mass = sum(template.body_mass)
                defaults = {}
                if mass > 0 and "mass" not in obj.overrides:
                    defaults["mass"] = mass
                if template.body_count and "dynamic" not in obj.overrides:
                    defaults["dynamic"] = any(flag & int(BodyFlags.DYNAMIC) for flag in template.body_flags)
                elif not template.body_count and "dynamic" not in obj.overrides:
                    defaults["dynamic"] = not any(
                        kind in (GeoType.PLANE, GeoType.HFIELD) for kind in template.shape_type
                    )
                if template.shape_count and "friction" not in obj.overrides:
                    defaults["friction"] = float(template.shape_material_mu[0])
                obj = dataclasses.replace(obj, **defaults)
            result.append(obj)
        return tuple(result)

    def _add_primitive(self, builder: ModelBuilder, obj: Object) -> None:
        x, y, z = obj.size
        radius = x * 0.5
        if obj.kind == self.Shape.BOX:
            volume = x * y * z
        elif obj.kind == self.Shape.SPHERE:
            volume = 4.0 * math.pi * radius**3 / 3.0
        elif obj.kind == self.Shape.CYLINDER:
            volume = math.pi * radius**2 * z
        else:
            volume = math.pi * radius**2 * (z - x) + 4.0 * math.pi * radius**3 / 3.0
        body = builder.add_body(
            xform=wp.transform(obj.position, obj.rotation), label=f"objects/{obj.id}/body", is_kinematic=not obj.dynamic
        )
        cfg = ModelBuilder.ShapeConfig(density=obj.mass / volume, mu=obj.friction)
        common = {"cfg": cfg, "color": (1.0, 1.0, 1.0), "label": f"objects/{obj.id}/shape"}
        if obj.kind == self.Shape.BOX:
            builder.add_shape_box(body, hx=x * 0.5, hy=y * 0.5, hz=z * 0.5, **common)
        elif obj.kind == self.Shape.SPHERE:
            cfg.mu_rolling = _SPHERE_ROLLING_FRICTION
            if self.solver_type == self.Solver.MUJOCO:
                common["custom_attributes"] = {"mujoco:condim": 6}
            builder.add_shape_sphere(body, radius=radius, **common)
        elif obj.kind == self.Shape.CYLINDER:
            builder.add_shape_cylinder(body, radius=radius, half_height=z * 0.5, **common)
        else:
            builder.add_shape_capsule(body, radius=radius, half_height=(z - x) * 0.5, **common)

    def _add_asset(self, builder: ModelBuilder, obj: Object) -> Object:
        template = self._template(obj)
        obj = self._resolved_defaults((obj,))[0]
        if template.particle_count:
            raise ValueError("Scene authoring currently supports rigid USD assets, not particles or deformables")
        start = builder.body_count
        if not template.body_count:
            if not template.shape_count:
                raise ValueError("The USD asset contains no supported rigid geometry")
            if self.solver_type == self.Solver.MUJOCO and not obj.dynamic:
                start_shape = builder.shape_count
                builder.add_builder(
                    template, xform=wp.transform(obj.position, obj.rotation), label_prefix=f"objects/{obj.id}/"
                )
                for index in range(start_shape, builder.shape_count):
                    builder.shape_flags[index] |= int(ShapeFlags.COLLIDE_SHAPES)
                return obj
            body = builder.add_body(
                xform=wp.transform(obj.position, obj.rotation),
                label=f"objects/{obj.id}/body",
                is_kinematic=not obj.dynamic,
            )
            for index in range(template.shape_count):
                if template.shape_type[index] in (GeoType.PLANE, GeoType.HFIELD) and obj.dynamic:
                    raise ValueError("Planes and heightfields must be imported as static scene geometry")
                flags = template.shape_flags[index]
                cfg = ModelBuilder.ShapeConfig(
                    mu=obj.friction,
                    has_shape_collision=True,
                    has_particle_collision=bool(flags & int(ShapeFlags.COLLIDE_PARTICLES)),
                    is_visible=bool(flags & int(ShapeFlags.VISIBLE)),
                )
                builder.add_shape(
                    body=body,
                    type=template.shape_type[index],
                    xform=template.shape_transform[index],
                    scale=template.shape_scale[index],
                    src=template.shape_source[index],
                    cfg=cfg,
                    color=template.shape_color[index],
                    opacity=template.shape_opacity[index],
                    label=f"objects/{obj.id}/shape_{index}",
                    custom_attributes={
                        key: attribute.values[index]
                        for key, attribute in template.custom_attributes.items()
                        if attribute.frequency == Model.AttributeFrequency.SHAPE and index in attribute.values
                    },
                )
            mass = builder.body_mass[body]
            if mass <= 0:
                if not obj.dynamic:
                    return obj
                raise ValueError("The USD asset has no finite-volume collision geometry for a rigid prop")
            ratio = obj.mass / mass
            builder.body_mass[body] *= ratio
            builder.body_inertia[body] = np.asarray(builder.body_inertia[body]) * ratio
            return obj
        builder.add_builder(template, xform=wp.transform(obj.position, obj.rotation), label_prefix=f"objects/{obj.id}/")
        for index in range(start, builder.body_count):
            if "dynamic" in obj.overrides:
                builder.body_flags[index] = int(BodyFlags.DYNAMIC if obj.dynamic else BodyFlags.KINEMATIC)
        if "mass" in obj.overrides:
            previous = sum(builder.body_mass[start:])
            if previous <= 0:
                raise ValueError("This USD has no positive rigid-body mass to override")
            ratio = obj.mass / previous
            for index in range(start, builder.body_count):
                builder.body_mass[index] *= ratio
                builder.body_inertia[index] = np.asarray(builder.body_inertia[index]) * ratio
        return obj

    def _build(self) -> _Runtime:
        builder = ModelBuilder()
        if self.solver_type == self.Solver.MUJOCO:
            from ..solvers.mujoco.solver_mujoco import SolverMuJoCo  # noqa: PLC0415

            SolverMuJoCo.register_custom_attributes(builder)
        builder.add_ground_plane(cfg=ModelBuilder.ShapeConfig(mu=_GROUND_FRICTION), color=(0.28, 0.31, 0.34))
        bindings = {}
        objects = {}
        for authored in self._objects:
            obj = authored
            b, s, j = builder.body_count, builder.shape_count, builder.joint_count
            if obj.kind == self.Shape.USD:
                obj = self._add_asset(builder, obj)
            else:
                self._add_primitive(builder, obj)
            objects[obj.id] = obj
            reference = None
            follows = False
            if builder.body_count > b:
                placement = wp.transform(obj.position, obj.rotation)
                reference = tuple(wp.transform_multiply(wp.transform_inverse(placement), builder.body_q[b]))
                roots = [k for k in range(j, builder.joint_count) if builder.joint_parent[k] == -1]
                follows = len(roots) == 1 and builder.joint_type[roots[0]] == JointType.FREE
                follows &= all(body >= b for body in builder.shape_body[s:])
            bindings[obj.id] = _Binding(
                (b, builder.body_count), (s, builder.shape_count), (j, builder.joint_count), reference, follows
            )
        model = builder.finalize(device=self.device)
        runtime = _Runtime(
            model,
            model.state(),
            model.state(),
            model.control(),
            None,
            None,
            None,
            bindings,
            objects,
            wp.clone(model.shape_color),
            wp.clone(model.shape_opacity),
            wp.clone(model.shape_material_mu),
        )
        runtime.pipeline = CollisionPipeline(model)
        runtime.contacts = runtime.pipeline.contacts()
        for obj in objects.values():
            self._apply_appearance(runtime, obj)
        return runtime

    def _create_solver(self, model: Model, contact_max: int) -> tuple[SolverBase | None, SolverBase.Observables | None]:
        if self.solver_type == self.Solver.XPBD:
            return SolverXPBD(model, iterations=8), None
        # MuJoCo requires a joint; static-only scenes need no solver.
        if not model.joint_count:
            return None, None
        from ..solvers.mujoco.solver_mujoco import SolverMuJoCo  # noqa: PLC0415

        solver = SolverMuJoCo(
            model,
            use_mujoco_cpu=self._mujoco_cpu,
            use_mujoco_contacts=self._mujoco_cpu,
            integrator="implicitfast",
            nconmax=contact_max,
            strict_capacity=not self._mujoco_cpu,
            kinematic_switchable_roots=False,
        )
        return solver, None if self._mujoco_cpu else solver.observables({solver.ObservableKind.OVERFLOW})

    def _preserve_state(self, old: _Runtime, new: _Runtime) -> None:
        labels = {label: index for index, label in enumerate(old.model.body_label)}
        old_indices, new_indices, deltas, clear_velocity = [], [], [], []
        for object_id, obj in new.objects.items():
            previous_obj = old.objects.get(object_id)
            if previous_obj is None or (obj.kind, obj.source) != (previous_obj.kind, previous_obj.source):
                continue
            moved = (obj.position, obj.rotation) != (previous_obj.position, previous_obj.rotation)
            delta = wp.transform_identity()
            old_binding, new_binding = old.bindings[object_id], new.bindings[object_id]
            changed_pivot = (
                old_binding.follows_body and new_binding.follows_body and old_binding.reference != new_binding.reference
            )
            if moved or changed_pivot:
                position, rotation = self.get_pose(object_id)
                current = wp.transform(position, rotation)
                target = wp.transform(obj.position, obj.rotation) if moved else current
                if changed_pivot:
                    target = wp.transform_multiply(
                        target, wp.transform(new_binding.reference[:3], new_binding.reference[3:])
                    )
                    current = wp.transform_multiply(
                        current, wp.transform(old_binding.reference[:3], old_binding.reference[3:])
                    )
                delta = _pose_delta(target, current)
            start, end = new.bindings[object_id].bodies
            for index in range(start, end):
                previous = labels.get(new.model.body_label[index])
                if previous is not None:
                    old_indices.append(previous)
                    new_indices.append(index)
                    deltas.append(delta)
                    clear_velocity.append(moved or not obj.dynamic)
        if old_indices:
            wp.launch(
                _copy_bodies,
                len(old_indices),
                inputs=[
                    wp.array(old_indices, dtype=int, device=self.device),
                    wp.array(new_indices, dtype=int, device=self.device),
                    wp.array(deltas, dtype=wp.transform, device=self.device),
                    wp.array(clear_velocity, dtype=bool, device=self.device),
                    old.state_0.body_q,
                    old.state_0.body_qd,
                    new.state_0.body_q,
                    new.state_0.body_qd,
                    new.state_1.body_q,
                    new.state_1.body_qd,
                ],
                device=self.device,
            )
            self._sync_coordinates(new)
        previous_joints = {name: index for index, name in enumerate(old.model.joint_label)}
        old_types, new_types = old.model.joint_type.numpy(), new.model.joint_type.numpy()
        old_q, old_d = old.model.joint_q_start.numpy(), old.model.joint_qd_start.numpy()
        new_q, new_d = new.model.joint_q_start.numpy(), new.model.joint_qd_start.numpy()
        for object_id, obj in new.objects.items():
            previous_obj = old.objects.get(object_id)
            if previous_obj is None or (obj.kind, obj.source) != (previous_obj.kind, previous_obj.source):
                continue
            for joint in range(*new.bindings[object_id].joints):
                previous = previous_joints.get(new.model.joint_label[joint])
                if previous is None:
                    continue
                if (
                    old_types[previous] != new_types[joint]
                    or old_q[previous + 1] - old_q[previous] != new_q[joint + 1] - new_q[joint]
                    or old_d[previous + 1] - old_d[previous] != new_d[joint + 1] - new_d[joint]
                ):
                    continue
                for field in ("joint_f", "joint_act", "joint_target_q", "joint_target_qd"):
                    source, target = getattr(old.control, field, None), getattr(new.control, field, None)
                    if source is None or target is None:
                        continue
                    coordinate = field == "joint_target_q"
                    starts_old, starts_new = (old_q, new_q) if coordinate else (old_d, new_d)
                    count = int(starts_new[joint + 1] - starts_new[joint])
                    if count == 0:
                        continue
                    wp.copy(
                        target,
                        source,
                        dest_offset=int(starts_new[joint]),
                        src_offset=int(starts_old[previous]),
                        count=count,
                    )

    @staticmethod
    def _sync_coordinates(runtime: _Runtime) -> None:
        if runtime.model.joint_count:
            eval_ik(runtime.model, runtime.state_0, runtime.state_0.joint_q, runtime.state_0.joint_qd)
            eval_ik(runtime.model, runtime.state_1, runtime.state_1.joint_q, runtime.state_1.joint_qd)

    def _publish(self, runtime: _Runtime) -> None:
        self._runtime = runtime
        self.model, self.state_0, self.state_1 = runtime.model, runtime.state_0, runtime.state_1
        self.control, self.solver = runtime.control, runtime.solver
        self.collision_pipeline, self.contacts = runtime.pipeline, runtime.contacts

    def _place_candidate(self, runtime: _Runtime) -> None:
        requested = [obj.id for obj in self._objects if obj.id in self._placement_requests]
        if not requested:
            return
        model, pipeline, contacts = runtime.model, runtime.pipeline, runtime.contacts
        snapshots = []
        for poses in (runtime.state_0.body_q, model.body_q):
            wp.launch(
                compute_shape_aabbs,
                model.shape_count,
                inputs=[
                    poses,
                    model.shape_transform,
                    model.shape_body,
                    model.shape_type,
                    model.shape_scale,
                    model.shape_collision_radius,
                    model.shape_source_ptr,
                    model.shape_margin,
                    model.shape_gap,
                    model.shape_collision_aabb_lower,
                    model.shape_collision_aabb_upper,
                    model.shape_flags,
                    pipeline._shape_variant_row,
                    contacts.contact_counters,
                    contacts.contact_generation,
                    pipeline.broad_phase_pair_count,
                    contacts.contact_counters.shape[0],
                ],
                outputs=[
                    pipeline.narrow_phase.shape_aabb_lower,
                    pipeline.narrow_phase.shape_aabb_upper,
                    pipeline.geom_data,
                    pipeline.geom_transform,
                ],
                device=self.device,
                record_tape=False,
            )
            # CPU numpy() shares Warp storage; preserve LIVE before RESET writes
            # the same buffers. Keep two intervals, not a swept union of poses.
            snapshots.append(
                tuple(
                    array.numpy().copy()
                    for array in (
                        pipeline.narrow_phase.shape_aabb_lower,
                        pipeline.narrow_phase.shape_aabb_upper,
                        pipeline.geom_transform,
                    )
                )
            )
        lower, upper, transforms = (np.concatenate([snapshot[index] for snapshot in snapshots]) for index in range(3))
        types, scales = np.tile(model.shape_type.numpy(), 2), np.tile(model.shape_scale.numpy(), (2, 1))
        margins, search_gap = np.tile(model.shape_margin.numpy(), 2), np.tile(model.shape_gap.numpy(), 2)
        lower += search_gap[:, None]
        upper -= search_gap[:, None]
        infinite = (types == GeoType.PLANE) & (scales[:, 0] == 0) & (scales[:, 1] == 0)
        for shape in np.flatnonzero((types == GeoType.PLANE) & ~infinite):
            transform = wp.transform(wp.vec3(*transforms[shape, :3]), wp.quat(*transforms[shape, 3:]))
            lo, hi = compute_box_bounds(transform, wp.vec3(*(scales[shape, :2] * 0.5), 0.0))
            lower[shape], upper[shape] = np.asarray(lo) - margins[shape], np.asarray(hi) + margins[shape]
        colliding = np.tile((model.shape_flags.numpy() & int(ShapeFlags.COLLIDE_SHAPES)) != 0, 2)
        occupied = colliding.copy()
        for object_id in requested:
            start, end = runtime.bindings[object_id].shapes
            occupied[start:end] = occupied[start + model.shape_count : end + model.shape_count] = False
        axis, clearance = int(model.up_axis), 0.001
        lateral = [index for index in range(3) if index != axis]
        for object_id in requested:
            binding, obj = runtime.bindings[object_id], runtime.objects[object_id]
            shapes = slice(*binding.shapes)
            rows = np.r_[np.arange(shapes.start, shapes.stop), np.arange(shapes.start, shapes.stop) + model.shape_count]
            if shapes.start == shapes.stop or infinite[shapes].any():
                raise ValueError("Automatic placement requires bounded geometry; place infinite planes explicitly")
            physical = np.arange(shapes.start, shapes.stop)
            if colliding[physical].any():
                physical = physical[colliding[physical]]
            lo, hi = lower[physical].min(axis=0), upper[physical].max(axis=0)
            if not np.isfinite([lo, hi]).all():
                raise ValueError("Automatic placement requires finite shape bounds")
            footprint = occupied & np.all(upper[:, lateral] >= lo[lateral] - clearance, axis=1)
            footprint &= np.all(lower[:, lateral] <= hi[lateral] + clearance, axis=1)
            supports = [0.0]
            for shape in np.flatnonzero(footprint & infinite):
                anchor = transforms[shape, :3]
                normal = wp.quat_rotate(wp.quat(*transforms[shape, 3:]), wp.vec3(0, 0, 1))
                if normal[axis] <= 1.0e-6:
                    raise ValueError("Automatic placement against a vertical or downward plane is unsupported")
                # Evaluate the plane over this footprint, not its million-metre AABB.
                rise = sum(max(normal[j] * (anchor[j] - lo[j]), normal[j] * (anchor[j] - hi[j])) for j in lateral)
                supports.append(anchor[axis] + rise / normal[axis] + margins[shape])
            finite = upper[footprint & ~infinite, axis]
            height = max(*supports, float(finite.max(initial=0)))
            lift = max(0.0, height + clearance - float(lo[axis]))
            if lift:
                translation = [0.0, 0.0, 0.0]
                translation[axis] = lift
                delta = wp.transform(translation, wp.quat_identity())
                start, end = binding.bodies
                if end > start:
                    wp.launch(
                        _move_bodies,
                        end - start,
                        inputs=[
                            start,
                            delta,
                            delta,
                            model.body_q,
                            runtime.state_0.body_q,
                            runtime.state_1.body_q,
                            runtime.state_0.body_qd,
                            runtime.state_1.body_qd,
                            False,
                        ],
                        device=self.device,
                    )
                wp.launch(
                    _move_world_shapes,
                    shapes.stop - shapes.start,
                    inputs=[shapes.start, delta, model.shape_body, model.shape_transform],
                    device=self.device,
                )
                start, end = binding.joints
                if end > start:
                    wp.launch(
                        _move_world_anchors,
                        end - start,
                        inputs=[start, delta, model.joint_parent, model.joint_type, model.joint_X_p],
                        device=self.device,
                    )
                position = tuple(value + translation[index] for index, value in enumerate(obj.position))
                runtime.objects[object_id] = dataclasses.replace(obj, position=position)
                lower[rows, axis] += lift
                upper[rows, axis] += lift
            occupied[rows] = colliding[rows]
        self._sync_coordinates(runtime)
        eval_ik(model, model, model.joint_q, model.joint_qd)

    def _resolve_placement_history(self, runtime: _Runtime) -> None:
        # Pending additions can appear in later undo snapshots before a batch
        # commits. Keep their insertion baseline aligned with the resolved pose.
        offsets = {
            obj.id: tuple(
                after - before for after, before in zip(runtime.objects[obj.id].position, obj.position, strict=True)
            )
            for obj in self._objects
            if obj.id in self._placement_requests and runtime.objects[obj.id].position != obj.position
        }
        if not offsets:
            return

        def translated(objects):
            return tuple(
                dataclasses.replace(
                    obj,
                    position=tuple(value + shift for value, shift in zip(obj.position, offsets[obj.id], strict=True)),
                )
                if obj.id in offsets
                else obj
                for obj in objects
            )

        self._undo = [translated(objects) for objects in self._undo]
        self._redo = [translated(objects) for objects in self._redo]
        if self._edit_before is not None:
            self._edit_before = translated(self._edit_before)

    def commit(self) -> bool:
        """Build and adopt one candidate, keeping the old runtime on failure.

        This explicit operation may block for first-use kernel compilation.
        Interactive callers should use request_commit()/before_frame() so it
        never happens within an ImGui draw or a captured physics operation.
        """
        if self._closed:
            raise RuntimeError("Scene editor is closed")
        if not self._topology_dirty:
            self._apply_fast_edits()
            return False
        with self._loading("Building model…") as ready:
            return self._commit_impl() if ready else False

    @contextmanager
    def _loading(self, text: str):
        """Present native feedback before blocking work, retaining outer splash ownership."""
        viewer = self.viewer
        gui = getattr(viewer, "gui", None)
        renderer = getattr(viewer, "renderer", None)
        show = getattr(viewer, "show_loading_splash", None)
        hide = getattr(viewer, "hide_loading_splash", None)
        has_exit = getattr(renderer, "has_exit", None)
        if (
            gui is None
            or getattr(viewer, "_headless", True)
            or not getattr(gui, "is_available", False)
            or not all(callable(method) for method in (show, hide, has_exit))
        ):
            yield True
            return
        if has_exit():
            yield False
            return
        if getattr(gui, "_loading_splash_active", False):
            yield True
            return
        paused = viewer.is_paused()
        try:
            show(text)
            try:
                for _ in range(2):
                    viewer.begin_frame(self.sim_time)
                    viewer.end_frame()
                    if has_exit():
                        break
            finally:
                viewer._paused = paused
            yield not has_exit()
        finally:
            hide()

    def _commit_impl(self) -> bool:
        started = time.perf_counter()
        old = self._runtime
        candidate = self._build()
        if old is not None:
            self._preserve_state(old, candidate)
        self._place_candidate(candidate)
        if self.solver_type == self.Solver.XPBD:
            self._collide(candidate)
            candidate.solver, candidate.observables = self._create_solver(
                candidate.model, candidate.contacts.rigid_contact_max
            )
        # Surface asynchronous build errors before replacing the displayed scene.
        wp.synchronize_device(self.device)
        camera = getattr(self.viewer, "camera", None) if old is not None else None
        camera_pose = (tuple(camera.pos), camera.pitch, camera.yaw) if camera is not None else None
        try:
            self.viewer.set_model(candidate.model)
            self.viewer.set_control(candidate.control)
            self.viewer.set_solver(candidate.solver)
            self.viewer.set_collision_pipeline(candidate.pipeline)
        except Exception:
            if old is not None:
                self.viewer.set_model(old.model)
                self.viewer.set_control(old.control)
                self.viewer.set_solver(old.solver)
                self.viewer.set_collision_pipeline(old.pipeline)
            raise
        finally:
            # An edit replaces the Model inside the same scene. An explicit
            # camera pose takes precedence over the viewer's new-scene framing.
            if camera_pose is not None:
                self.viewer.set_camera(*camera_pose)
        self._publish(candidate)
        self._resolve_placement_history(candidate)
        self._objects = tuple(candidate.objects[obj.id] for obj in self._objects)
        self._placement_requests.difference_update(candidate.bindings)
        self._topology_dirty = self._requested_commit = False
        self._failed_attempt = None
        self._fast_pending.clear()
        self._trim_templates()
        self.rebuild_count += 1
        self.last_build_ms = (time.perf_counter() - started) * 1000.0
        self.error = None
        self.status = f"{len(self._objects)} objects · applied in {self.last_build_ms:.1f} ms"
        self._preview_last = None
        return True

    def body_indices(self, object_id: str) -> tuple[int, ...]:
        """Resolve current body indices; callers must not retain them across commits."""
        binding = self._runtime.bindings.get(object_id) if self._runtime else None
        return tuple(range(*binding.bodies)) if binding else ()

    def get_pose(self, object_id: str) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Read a selected free prop's live placement; scene assemblies use their authored root."""
        obj = self.get_object(object_id)
        runtime = self._runtime
        binding = runtime.bindings.get(object_id) if runtime else None
        if binding and binding.follows_body:
            index = binding.bodies[0]
            values = runtime.state_0.body_q[index : index + 1].numpy()[0]
            placement = wp.transform_multiply(
                wp.transform(values[:3], values[3:]),
                wp.transform_inverse(wp.transform(binding.reference[:3], binding.reference[3:])),
            )
            return tuple(wp.transform_get_translation(placement)), tuple(wp.transform_get_rotation(placement))
        return obj.position, obj.rotation

    @staticmethod
    def _apply_appearance(runtime: _Runtime, obj: Object) -> None:
        start, end = runtime.bindings[obj.id].shapes
        if end > start:
            wp.launch(
                _appearance,
                end - start,
                inputs=[
                    start,
                    wp.vec3(obj.color),
                    obj.visible,
                    obj.friction,
                    runtime.base_colors,
                    runtime.base_opacity,
                    runtime.model.shape_color,
                    runtime.model.shape_opacity,
                    runtime.model.shape_material_mu,
                ],
                device=runtime.model.device,
            )
            if obj.kind == SceneEditor.Shape.USD and "friction" not in obj.overrides:
                wp.copy(
                    runtime.model.shape_material_mu,
                    runtime.base_friction,
                    dest_offset=start,
                    src_offset=start,
                    count=end - start,
                )

    def _apply_fast_edits(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        current = {obj.id: obj for obj in self._objects}
        moved = False
        changed = 0
        world_shapes = runtime.model.body_shapes.get(-1, ())
        for object_id, previous in list(self._fast_pending.items()):
            if object_id not in current or object_id not in runtime.bindings:
                continue
            obj = current[object_id]
            if self._structure(obj) != self._structure(runtime.objects[object_id]):
                continue
            if (obj.position, obj.rotation) != (previous.position, previous.rotation):
                moved = True
                binding = runtime.bindings[object_id]
                live_position, live_rotation = self.get_pose(object_id)
                if not binding.follows_body:
                    live_position, live_rotation = previous.position, previous.rotation
                target = wp.transform(obj.position, obj.rotation)
                delta = _pose_delta(target, wp.transform(live_position, live_rotation))
                initial_delta = _pose_delta(target, wp.transform(previous.position, previous.rotation))
                start, end = binding.bodies
                if end > start:
                    wp.launch(
                        _move_bodies,
                        end - start,
                        inputs=[
                            start,
                            delta,
                            initial_delta,
                            runtime.model.body_q,
                            runtime.state_0.body_q,
                            runtime.state_1.body_q,
                            runtime.state_0.body_qd,
                            runtime.state_1.body_qd,
                            True,
                        ],
                        device=self.device,
                    )
                start, end = binding.shapes
                if any(start <= shape < end for shape in world_shapes):
                    changed |= ModelFlags.SHAPE_PROPERTIES
                if end > start:
                    wp.launch(
                        _move_world_shapes,
                        end - start,
                        inputs=[start, initial_delta, runtime.model.shape_body, runtime.model.shape_transform],
                        device=self.device,
                    )
                start, end = binding.joints
                if end > start:
                    if not binding.follows_body:
                        changed |= ModelFlags.JOINT_PROPERTIES
                    wp.launch(
                        _move_world_anchors,
                        end - start,
                        inputs=[
                            start,
                            initial_delta,
                            runtime.model.joint_parent,
                            runtime.model.joint_type,
                            runtime.model.joint_X_p,
                        ],
                        device=self.device,
                    )
            if (obj.color, obj.visible, obj.friction, obj.overrides) != (
                previous.color,
                previous.visible,
                previous.friction,
                previous.overrides,
            ):
                self._apply_appearance(runtime, obj)
                if (obj.friction, "friction" in obj.overrides) != (
                    previous.friction,
                    "friction" in previous.overrides,
                ):
                    changed |= ModelFlags.SHAPE_PROPERTIES
            runtime.objects[object_id] = obj
        if moved:
            self._sync_coordinates(runtime)
            eval_ik(runtime.model, runtime.model, runtime.model.joint_q, runtime.model.joint_qd)
        if self.solver_type == self.Solver.MUJOCO and runtime.solver is not None and changed:
            runtime.solver.notify_model_changed(changed)
        if moved:
            if self.solver_type == self.Solver.MUJOCO:
                runtime.contacts_dirty = True
                runtime.contacts.contact_counters.zero_()
            else:
                self._collide(runtime)
            if runtime.solver is not None:
                runtime.solver.reset(runtime.state_0, flags=0)
        self._fast_pending.clear()
        # Previous snapshots use the old authored poses and materials. Drop
        # them without clearing the selection or interrupting a held gizmo.
        gui = getattr(self.viewer, "gui", None)
        workspace = getattr(gui, "workspace", None)
        if workspace is not None and workspace.example is self:
            workspace.reset_history()

    def _collide(self, runtime: _Runtime) -> bool:
        """Detect complete contacts before solving, growing owned buffers if needed."""
        runtime.contacts_dirty = False
        pipeline, contacts = runtime.pipeline, runtime.contacts
        try:
            for attempt in range(4):
                pipeline.collide(runtime.state_0, contacts)
                counters = contacts.contact_counters.numpy()
                count = int(counters[0])
                if count < 0:
                    raise RuntimeError("Rigid contact count exceeded supported indexing")
                if count <= contacts.rigid_contact_max and not counters[2:].any():
                    if pipeline is not runtime.pipeline:
                        if runtime is self._runtime:
                            self.viewer.set_collision_pipeline(pipeline)
                            if self.solver_type == self.Solver.MUJOCO and not self._mujoco_cpu:
                                self.viewer.set_solver(None)
                        previous = runtime.contacts.rigid_contact_max
                        runtime.pipeline, runtime.contacts = pipeline, contacts
                        if self.solver_type == self.Solver.MUJOCO and not self._mujoco_cpu:
                            runtime.solver = runtime.observables = None
                            runtime.solver_graphs.clear()
                            runtime.solver_warm_binding = None
                        runtime.model.rigid_contact_max = contacts.rigid_contact_max
                        runtime.model.soft_contact_max = contacts.soft_contact_max
                        runtime.model._contact_capacity_initialized = True
                        if runtime is self._runtime:
                            self.collision_pipeline, self.contacts = pipeline, contacts
                            self.solver = runtime.solver
                        logger.warning(
                            "Expanded rigid contact capacity from %d to %d; retained all %d contacts",
                            previous,
                            contacts.rigid_contact_max,
                            count,
                        )
                    return True
                if attempt == 3:
                    raise RuntimeError(
                        f"Rigid contacts remain incomplete after three growth attempts (count={count}, capacity={contacts.rigid_contact_max}, overflow={counters[2:].tolist()})"
                    )
                pipeline = CollisionPipeline(
                    runtime.model,
                    rigid_contact_max=max(2 * contacts.rigid_contact_max, 2 * count),
                    soft_contact_max=contacts.soft_contact_max,
                    publish_capacity=False,
                )
                contacts = pipeline.contacts()
        except (MemoryError, RuntimeError, ValueError) as error:
            message = f"Contact detection could not produce a complete buffer: {error}"
            if runtime is not self._runtime:
                raise RuntimeError(message) from error
            self._pause_physics(runtime, message)
        return False

    def _pause_physics(self, runtime: _Runtime, message: str) -> None:
        self.error = message
        self.status = "Physics paused; the current scene is retained."
        # An overflowing count must not reach display kernels either.
        runtime.contacts.contact_counters.zero_()
        logger.error("%s", message)
        if not hasattr(self.viewer, "_paused"):
            raise RuntimeError(message)
        self.viewer._paused = True

    def _step_solver(self, runtime: _Runtime, dt: float) -> None:
        """Capture only the solver; validate collision and overflow outside graphs."""
        solver = runtime.solver

        def solve():
            solver.step(
                runtime.state_0,
                runtime.state_1,
                runtime.control,
                runtime.contacts,
                dt,
                observables=runtime.observables,
            )

        if self.solver_type != self.Solver.MUJOCO or self._mujoco_cpu:
            solve()
            return
        arrays = tuple(
            getattr(state, field)
            for state in sorted((runtime.state_0, runtime.state_1), key=id)
            for field in ("body_q", "body_qd", "body_f", "joint_q", "joint_qd")
        ) + tuple(
            getattr(runtime.control, field, None)
            for field in ("joint_f", "joint_act", "joint_target_q", "joint_target_qd")
        )
        arrays += (
            getattr(getattr(runtime.control, "mujoco", None), "ctrl", None),
            getattr(runtime.observables, "overflow", None),
        )
        signature = (
            id(solver),
            id(runtime.contacts),
            id(runtime.control),
            id(runtime.observables),
            dt,
            *((id(array), array.ptr, array.shape) if array is not None else None for array in arrays),
        )
        references = (solver, runtime.contacts, runtime.control, runtime.observables, *arrays)
        if runtime.solver_warm_binding is None or runtime.solver_warm_binding[0] != signature:
            runtime.solver_graphs.clear()
            # Warm changed bindings on one real substep before capture. Keep
            # buffers alive and ignore the normal A/B State alternation.
            solve()
            runtime.solver_warm_binding = signature, references
            return
        key = id(runtime.state_0), id(runtime.state_1)
        entry = runtime.solver_graphs.get(key)
        if entry is None:
            with wp.ScopedCapture(device=self.device, force_module_load=False) as capture:
                solve()
            entry = capture.graph, references
            if len(runtime.solver_graphs) == 2:
                del runtime.solver_graphs[next(iter(runtime.solver_graphs))]
            runtime.solver_graphs[key] = entry
        wp.capture_launch(entry[0])

    def reset_scene(self) -> None:
        """Restore the authored reset poses without rebuilding model or solver."""
        self._placement_cache = None
        if self._runtime is not None:
            runtime = self._runtime
            if runtime.model.body_count:
                for state in (runtime.state_0, runtime.state_1):
                    wp.copy(state.body_q, runtime.model.body_q)
                    state.body_qd.zero_()
                    state.body_f.zero_()
            self._sync_coordinates(runtime)
            if runtime.solver is not None:
                runtime.solver.reset(runtime.state_0, flags=0)
            if self.solver_type == self.Solver.MUJOCO:
                runtime.contacts_dirty = True
                runtime.contacts.contact_counters.zero_()
            else:
                self._collide(runtime)
            self.sim_time = 0.0
            workspace = getattr(getattr(self.viewer, "gui", None), "workspace", None)
            if workspace is not None and workspace.example is self:
                workspace.reset_history()

    def step(self) -> None:
        """Step owned physics; topology never changes during these launches."""
        runtime = self._runtime
        if runtime is None:
            self.before_frame()
            runtime = self._runtime
        if runtime is None:
            return
        if runtime.solver is None and not runtime.model.joint_count:
            self.sim_time += self.frame_dt
            return
        if self.solver_type == self.Solver.MUJOCO and runtime.solver is None:
            with self._loading("Starting physics…") as ready:
                if ready:
                    self._step_impl(runtime)
        else:
            self._step_impl(runtime)

    def _step_impl(self, runtime: _Runtime) -> None:
        dt = self.frame_dt / self.sim_substeps
        for substep in range(self.sim_substeps):
            runtime.state_0.clear_forces()
            self.viewer.apply_forces(runtime.state_0)
            if not self._mujoco_cpu and not self._collide(runtime):
                self._sync_coordinates(runtime)
                self._publish(runtime)
                self.sim_time += substep * dt
                return
            if runtime.solver is None:
                try:
                    solver, observables = self._create_solver(runtime.model, runtime.contacts.rigid_contact_max)
                    self.viewer.set_solver(solver)
                except (ImportError, MemoryError, OSError, RuntimeError, TypeError, ValueError) as error:
                    self.sim_time += substep * dt
                    self._pause_physics(runtime, f"MuJoCo could not start: {error}")
                    return
                runtime.solver, runtime.observables = solver, observables
                self.solver = solver
            self._step_solver(runtime, dt)
            if runtime.observables is not None:
                overflow = runtime.observables.overflow.numpy()
                if overflow.any():
                    runtime.solver.reset(runtime.state_0, flags=0)
                    self._sync_coordinates(runtime)
                    self._publish(runtime)
                    self.sim_time += substep * dt
                    self._pause_physics(runtime, f"MuJoCo Warp capacity overflow: {overflow.tolist()}")
                    return
            runtime.state_0, runtime.state_1 = runtime.state_1, runtime.state_0
        if self.solver_type == self.Solver.XPBD:
            self._sync_coordinates(runtime)
        if self._mujoco_cpu:
            self._update_cpu_contacts(runtime)
        self._publish(runtime)
        self.sim_time += self.frame_dt

    def _update_cpu_contacts(self, runtime: _Runtime) -> None:
        """Export native contact geometry once per frame, without affecting CPU physics."""
        capacity = runtime.solver.get_max_contact_count()
        if capacity > runtime.contacts.rigid_contact_max:
            pipeline = CollisionPipeline(
                runtime.model,
                rigid_contact_max=capacity,
                soft_contact_max=runtime.contacts.soft_contact_max,
                publish_capacity=False,
            )
            contacts = pipeline.contacts()
            self.viewer.set_collision_pipeline(pipeline)
            runtime.pipeline, runtime.contacts = pipeline, contacts
            runtime.model.rigid_contact_max = contacts.rigid_contact_max
            runtime.model.soft_contact_max = contacts.soft_contact_max
            runtime.model._contact_capacity_initialized = True
        # Native MuJoCo solves every contact. Its configured nconmax only
        # limits this display export, with a warning when it is truncated.
        runtime.solver.update_contacts(runtime.contacts)
        runtime.contacts_dirty = False

    def render(self) -> None:
        """Render live or history-selected state with optional placement preview."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self._draw_preview()
        self.viewer.end_frame()

    def preview_primitive(
        self,
        kind: Shape | str,
        *,
        position,
        size=(1.0, 1.0, 1.0),
        color=(0.3, 0.85, 1.0),
        placement: bool = False,
    ) -> tuple[float, float, float]:
        """Outline primitive dimensions [m] without modifying the document/model.

        When ``placement`` is true, raise the bounds above every collider under
        their footprint and return the resolved object position [m].
        """
        kind = self.Shape(kind)
        if kind == self.Shape.USD:
            raise ValueError("Use preview_usd() for referenced asset geometry")
        self._preview_source = None
        self._preview = None
        half = np.asarray(self._vector(size, 3, "size")) * 0.5
        return self._preview_bounds(position, (-half, half), color, placement=placement)

    def preview_usd(
        self,
        source: str | Path,
        *,
        position,
        scale=(1.0, 1.0, 1.0),
        color=(0.3, 0.85, 1.0),
        placement: bool = False,
    ) -> tuple[float, float, float] | None:
        """Outline cached USD collision bounds without building a simulation model.

        Prepare the native template and its bounds once on a CPU worker. Return
        ``None`` while preparation is pending. Otherwise return the object pivot
        position [m], raised above colliders when ``placement`` is true. The
        outline retains offsets between the pivot and the collision geometry.
        Unbounded geometry cannot be previewed.
        """
        obj = self._validate(
            self.Object(
                id="preview",
                name="Preview",
                kind=self.Shape.USD,
                source=str(Path(source).expanduser().resolve()),
                size=scale,
            )
        )
        key = self._template_key(obj)
        self._preview = None
        if self._edit_before is not None and key not in self._templates:
            # Scaling a cached envelope keeps a held field responsive. Import
            # the new scale only after release, alongside its structural edit.
            for previous, template in reversed(self._templates.items()):
                if previous[0] == key[0] and previous[2:] == key[2:] and template.bounds is not None:
                    ratio = np.asarray(obj.size) / previous[1]
                    bounds = tuple(bound * ratio for bound in template.bounds)
                    self._preview_source = None
                    return self._preview_bounds(position, bounds, color, placement=placement)
        self._preview_source = obj
        if self._preview_error is not None and self._preview_error[0] == key:
            raise ValueError(self._preview_error[1])
        self._schedule_template(obj)
        template = self._templates.get(key)
        if template is None:
            return None
        if template.bounds is None:
            raise ValueError("The asset has no finite collision bounds for a placement preview")
        return self._preview_bounds(position, template.bounds, color, placement=placement)

    def _preview_bounds(self, position, bounds, color, *, placement: bool) -> tuple[float, float, float]:
        point = np.asarray(self._vector(position, 3, "position"))
        lo, hi = bounds
        if placement:
            axis = int(self.model.up_axis) if self.model is not None else 2
            height = 0.0
            if self._runtime is not None:
                key = (id(self.model), self.sim_time, self.revision, tuple(point), tuple(lo), tuple(hi))
                if self._placement_cache is None or self._placement_cache[0] != key:
                    if self._placement_result is None:
                        self._placement_result = wp.zeros(2, dtype=float, device=self.device)
                    else:
                        self._placement_result.zero_()
                    model = self.model
                    wp.launch(
                        _placement_surface,
                        model.shape_count,
                        inputs=[
                            self.state_0.body_q,
                            model.body_q,
                            model.shape_body,
                            model.shape_transform,
                            model.shape_type,
                            model.shape_scale,
                            model.shape_flags,
                            model.shape_margin,
                            model.shape_collision_aabb_lower,
                            model.shape_collision_aabb_upper,
                            wp.vec3(*(point + lo)),
                            wp.vec3(*(point + hi)),
                            axis,
                        ],
                        outputs=[self._placement_result],
                        device=self.device,
                        record_tape=False,
                    )
                    height, unsupported = self._placement_result.numpy()
                    if unsupported:
                        raise ValueError("Automatic placement against a vertical or downward plane is unsupported")
                    self._placement_cache = key, float(height)
                height = self._placement_cache[1]
            point[axis] = max(point[axis], height + 0.001 - lo[axis])
        self._preview = "box", tuple(point + (lo + hi) * 0.5), tuple(hi - lo), tuple(color)
        return tuple(float(value) for value in point)

    def clear_preview(self) -> None:
        """Hide the transient placement outline on the next render."""
        self._preview = None
        self._preview_source = None
        self._preview_error = None

    def _draw_preview(self) -> None:
        name = "__scene_authoring/placement"
        if self._preview is None:
            if self._preview_last is not None:
                self.viewer.log_lines(name, None, None, None, hidden=True)
            self._preview_last = None
            return
        if self._preview != self._preview_last:
            _, position, size, _color = self._preview
            corners = np.asarray(
                [(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)], dtype=np.float32
            )
            corners = corners * np.asarray(size) + np.asarray(position)
            edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if (i ^ j) in (1, 2, 4)]
            starts = np.asarray([corners[a] for a, _ in edges], dtype=np.float32)
            ends = np.asarray([corners[b] for _, b in edges], dtype=np.float32)
            if self._preview_buffers is None:
                self._preview_buffers = (
                    wp.empty(12, dtype=wp.vec3, device=self.device),
                    wp.empty(12, dtype=wp.vec3, device=self.device),
                )
            self._preview_buffers[0].assign(starts)
            self._preview_buffers[1].assign(ends)
            self._preview_last = self._preview
        self.viewer.log_lines(name, *self._preview_buffers, self._preview[3])

    def save_usd(self, path: str | Path) -> None:
        """Export the authored document using standard USD geometry/physics schemas."""
        save_scene(path, self._objects)
        self.status = f"Saved {Path(path).name}"

    def load_usd(self, path: str | Path) -> None:
        """Open an editor document, or import an ordinary USD as a scene asset."""
        try:
            records = load_scene(path)
        except NotAuthoringSceneError:
            self.add_usd(path)
            return
        objects = tuple(
            self._validate(self.Object(**{**record, "kind": self.Shape(record["kind"])})) for record in records
        )
        if len({obj.id for obj in objects}) != len(objects):
            raise ValueError("Scene document contains duplicate object IDs")
        self._remember()
        self._restore_document(objects)
        # Explicit Open refreshes dependencies even when authored records match.
        self._topology_dirty = True
        self._failed_attempt = None
        self._document_epoch += 1
        for obj in objects:
            if obj.kind == self.Shape.USD:
                self._schedule_template(obj)

    def close(self) -> None:
        """Cancel pending preparation; retain already compiled state for caller inspection."""
        if self._closed:
            return
        self._closed = True
        workspace = getattr(getattr(self.viewer, "gui", None), "workspace", None)
        if workspace is not None and workspace.example is self:
            workspace.set_scene_editor(None)
        for future in self._preparing.values():
            future.cancel()
        self._close_preparation_executor()

    def test_final(self) -> None:
        """Validate finite physical state and one compiled scene for example test mode."""
        if self._runtime is None:
            raise AssertionError("The authoring scene was not initialized")
        if self.model.body_count and not np.isfinite(self._runtime.state_0.body_q.numpy()).all():
            raise AssertionError("The authored scene contains non-finite body poses")
