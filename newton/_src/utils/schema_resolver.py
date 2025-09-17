from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import warp as wp
from pxr import Usd


@dataclass
class Attribute:
    """
    Specifies a USD attribute and its transformation function.

    Args:
        usd_name (str): The name of the USD attribute.
        default (Any | None): Default USD-authored value from schema, if any.
        transform (Callable[[Any], Any] | None): A function to transform the attribute value.
    """

    usd_name: str
    default: Any | None = None
    transform: Callable[[Any], Any] | None = None


class schemaPlugin:
    name: str
    # mapping is a dictionary for known variables in Newton. Its purpose is to map usd attributes to exisiting Newton data.
    # prim_type -> variable -> list[Attribute]
    mapping: dict[str, dict[str, list[Attribute]]]

    # extra_attr_namespaces is a list of namespaces for extra attributes that are not in the mapping.
    extra_attr_namespaces: ClassVar[list[str]] = []

    def __init__(self) -> None:
        # Precompute the full set of USD attribute names referenced by this plugin's mapping.
        names: set[str] = set()
        try:
            mapping_items = self.mapping.items()
        except AttributeError:
            mapping_items = []
        for _prim_type, var_map in mapping_items:
            try:
                var_items = var_map.items()
            except AttributeError:
                continue
            for _var, specs in var_items:
                for spec in specs:
                    names.add(spec.usd_name)
        self._engine_attributes: list[str] = list(names)

    def get_value(self, prim, prim_type: str, key: str) -> tuple[Any, str] | None:
        """
        Get attribute value for a given prim type and key.

        Args:
            prim: USD prim to query
            prim_type: Prim type ("scene", "joint", "shape", "body", "material", "actuator")
            key: Attribute key within the prim type

        Returns:
            Tuple of (value, usd_attr_name) if found, None otherwise
        """
        if prim is None:
            return None
        for spec in self.mapping.get(prim_type, {}).get(key, []):
            v = _get_attr(prim, spec.usd_name)
            if v is not None:
                return (spec.transform(v) if spec.transform is not None else v), spec.usd_name
        return None

    def collect_prim_engine_attrs(self, prim) -> dict[str, Any]:
        """
        Collect engine-specific attributes for a single prim.
        Returns dictionary of engine-specific attributes for this prim.
        """
        if prim is None:
            return {}

        # Collect explicit attribute names defined in the plugin mapping (precomputed)
        prim_engine_attrs = (
            _collect_engine_mapped_attrs(prim, self._engine_attributes) if self._engine_attributes else {}
        )

        # Collect attributes by known engine-specific prefixes
        # PhysX uses multiple prefixes, others use {name}:
        main_prefix = self.name if self.name == "physx" else f"{self.name}:"
        all_prefixes = [main_prefix]
        if self.extra_attr_namespaces:
            all_prefixes.extend(self.extra_attr_namespaces)
        prefixed_attrs = _collect_engine_specific_attrs(prim, all_prefixes)

        # Merge and return (explicit names take precedence)
        merged: dict[str, Any] = {}
        merged.update(prefixed_attrs)
        merged.update(prim_engine_attrs)
        return merged


def _get_attr(prim, name: str):
    if prim is None:
        return None
    attr = prim.GetAttribute(name)
    if not (attr and attr.IsValid() and attr.HasAuthoredValue()):
        return None
    return attr.Get()


def _collect_engine_mapped_attrs(prim, names: list[str]) -> dict[str, Any]:
    """Collect engine-specific attributes authored on the prim that have direct mappings in the plugin mapping"""
    out = {}
    for n in names:
        v = _get_attr(prim, n)
        if v is not None:
            out[n] = v
    return out


def _collect_engine_specific_attrs(prim, namespaces: list[str]) -> dict[str, Any]:
    """Collect engine-specific authored attributes using USD namespace queries."""

    out: dict[str, Any] = {}
    if prim is None:
        return out

    for ns in namespaces:
        for prop in prim.GetAuthoredPropertiesInNamespace(ns):
            if isinstance(prop, Usd.Attribute) and prop.IsValid() and prop.HasAuthoredValue():
                out[prop.GetName()] = prop.Get()

    return out


class NewtonPlugin(schemaPlugin):
    name: ClassVar[str] = "newton"
    mapping: ClassVar[dict[str, dict[str, list[Attribute]]]] = {
        "scene": {
            "time_step": [Attribute("newton:timeStep", 0.002)],
            "max_solver_iterations": [Attribute("newton:maxSolverIterations", 5)],
            "enable_gravity": [Attribute("newton:enableGravity", True)],
            "contact_margin": [Attribute("newton:contactMargin", 0.0)],
        },
        "joint": {
            "armature": [Attribute("newton:armature", 1.0e-2)],
            "friction": [Attribute("newton:friction", 0.0)],
            "limit_linear_ke": [Attribute("newton:linear:limitStiffness", 1.0e4)],
            "limit_angular_ke": [Attribute("newton:angular:limitStiffness", 1.0e4)],
            "limit_rotX_ke": [Attribute("newton:rotX:limitStiffness", 1.0e4)],
            "limit_rotY_ke": [Attribute("newton:rotY:limitStiffness", 1.0e4)],
            "limit_rotZ_ke": [Attribute("newton:rotZ:limitStiffness", 1.0e4)],
            "limit_linear_kd": [Attribute("newton:linear:limitDamping", 1.0e1)],
            "limit_angular_kd": [Attribute("newton:angular:limitDamping", 1.0e1)],
            "limit_rotX_kd": [Attribute("newton:rotX:limitDamping", 1.0e1)],
            "limit_rotY_kd": [Attribute("newton:rotY:limitDamping", 1.0e1)],
            "limit_rotZ_kd": [Attribute("newton:rotZ:limitDamping", 1.0e1)],
            "angular_position": [Attribute("newton:angular:position", 0.0)],
            "linear_position": [Attribute("newton:linear:position", 0.0)],
            "rotX_position": [Attribute("newton:rotX:position", 0.0)],
            "rotY_position": [Attribute("newton:rotY:position", 0.0)],
            "rotZ_position": [Attribute("newton:rotZ:position", 0.0)],
            "angular_velocity": [Attribute("newton:angular:velocity", 0.0)],
            "linear_velocity": [Attribute("newton:linear:velocity", 0.0)],
            "rotX_velocity": [Attribute("newton:rotX:velocity", 0.0)],
            "rotY_velocity": [Attribute("newton:rotY:velocity", 0.0)],
            "rotZ_velocity": [Attribute("newton:rotZ:velocity", 0.0)],
        },
        "shape": {
            "mesh_hull_vertex_limit": [Attribute("newton:hullVertexLimit", -1)],
            # Use ShapeConfig.thickness default for contact margin
            "contact_margin": [Attribute("newton:contactMargin", 1.0e-5)],
        },
        "body": {
            "rigid_body_linear_damping": [Attribute("newton:damping", 0.0)],
        },
        "material": {
            "priority": [Attribute("newton:priority", 0)],
            "weight": [Attribute("newton:weight", 1.0)],
            "stiffness": [Attribute("newton:stiffness", 1.0e5)],
            "damping": [Attribute("newton:damping", 1000.0)],
        },
        "actuator": {
            # Mirror MuJoCo actuator defaults when applicable
            "ctrl_low": [Attribute("newton:ctrlRange:low", 0.0)],
            "ctrl_high": [Attribute("newton:ctrlRange:high", 0.0)],
            "force_low": [Attribute("newton:forceRange:low", 0.0)],
            "force_high": [Attribute("newton:forceRange:high", 0.0)],
            "act_low": [Attribute("newton:actRange:low", 0.0)],
            "act_high": [Attribute("newton:actRange:high", 0.0)],
            "length_low": [Attribute("newton:lengthRange:low", 0.0)],
            "length_high": [Attribute("newton:lengthRange:high", 0.0)],
            "gainPrm": [Attribute("newton:gainPrm", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])],
            "gainType": [Attribute("newton:gainType", "fixed")],
            "biasPrm": [Attribute("newton:biasPrm", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])],
            "biasType": [Attribute("newton:biasType", "none")],
            "dynPrm": [Attribute("newton:dynPrm", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])],
            "dynType": [Attribute("newton:dynType", "none")],
            # The following have no MuJoCo counterpart; keep unspecified defaults
            "speedTorqueGradient": [Attribute("newton:speedTorqueGradient", None)],
            "torqueSpeedGradient": [Attribute("newton:torqueSpeedGradient", None)],
            "maxVelocity": [Attribute("newton:maxVelocity", None)],
            "gear": [Attribute("newton:gear", [1, 0, 0, 0, 0, 0])],
        },
    }


class PhysxPlugin(schemaPlugin):
    name: ClassVar[str] = "physx"
    extra_attr_namespaces: ClassVar[list[str]] = [
        # Scene and rigid body
        "physxScene",
        "physxRigidBody",
        # Collisions and meshes
        "physxCollision",
        "physxConvexHullCollision",
        "physxConvexDecompositionCollision",
        "physxTriangleMeshCollision",
        "physxTriangleMeshSimplificationCollision",
        "physxSDFMeshCollision",
        # Materials
        "physxMaterial",
        # Joints and limits
        "physxJoint",
        "physxLimit",
        # Articulations
        "physxArticulation",
    ]

    mapping: ClassVar[dict[str, dict[str, list[Attribute]]]] = {
        "scene": {
            "time_step": [
                Attribute("physxScene:timeStepsPerSecond", 60, lambda hz: (1.0 / hz) if (hz and hz > 0) else None)
            ],
            "max_solver_iterations": [Attribute("physxScene:maxVelocityIterationCount", 255)],
            "enable_gravity": [Attribute("physxRigidBody:disableGravity", False, lambda value: not value)],
            "contact_margin": [Attribute("physxScene:contactOffset", 0.0)],
        },
        "joint": {
            "armature": [Attribute("physxJoint:armature", 0.0)],
            # Per-axis linear limit aliases
            "limit_transX_ke": [Attribute("physxLimit:linear:stiffness", 0.0)],
            "limit_transY_ke": [Attribute("physxLimit:linear:stiffness", 0.0)],
            "limit_transZ_ke": [Attribute("physxLimit:linear:stiffness", 0.0)],
            "limit_transX_kd": [Attribute("physxLimit:linear:damping", 0.0)],
            "limit_transY_kd": [Attribute("physxLimit:linear:damping", 0.0)],
            "limit_transZ_kd": [Attribute("physxLimit:linear:damping", 0.0)],
            "limit_linear_ke": [Attribute("physxLimit:linear:stiffness", 0.0)],
            "limit_angular_ke": [Attribute("physxLimit:angular:stiffness", 0.0)],
            "limit_rotX_ke": [Attribute("physxLimit:rotX:stiffness", 0.0)],
            "limit_rotY_ke": [Attribute("physxLimit:rotY:stiffness", 0.0)],
            "limit_rotZ_ke": [Attribute("physxLimit:rotZ:stiffness", 0.0)],
            "limit_linear_kd": [Attribute("physxLimit:linear:damping", 0.0)],
            "limit_angular_kd": [Attribute("physxLimit:angular:damping", 0.0)],
            "limit_rotX_kd": [Attribute("physxLimit:rotX:damping", 0.0)],
            "limit_rotY_kd": [Attribute("physxLimit:rotY:damping", 0.0)],
            "limit_rotZ_kd": [Attribute("physxLimit:rotZ:damping", 0.0)],
            "angular_position": [Attribute("state:angular:physics:position", 0.0)],
            "linear_position": [Attribute("state:linear:physics:position", 0.0)],
            "rotX_position": [Attribute("state:rotX:physics:position", 0.0)],
            "rotY_position": [Attribute("state:rotY:physics:position", 0.0)],
            "rotZ_position": [Attribute("state:rotZ:physics:position", 0.0)],
            "angular_velocity": [Attribute("state:angular:physics:velocity", 0.0)],
            "linear_velocity": [Attribute("state:linear:physics:velocity", 0.0)],
            "rotX_velocity": [Attribute("state:rotX:physics:velocity", 0.0)],
            "rotY_velocity": [Attribute("state:rotY:physics:velocity", 0.0)],
            "rotZ_velocity": [Attribute("state:rotZ:physics:velocity", 0.0)],
        },
        "shape": {
            # Mesh hull vertex limit
            "mesh_hull_vertex_limit": [Attribute("physxConvexHullCollision:hullVertexLimit", 64)],
            # Collision contact offset
            "contact_margin": [Attribute("physxCollision:contactOffset", float("-inf"))],
        },
        "material": {
            "stiffness": [Attribute("physxMaterial:compliantContactStiffness", 0.0)],
            "damping": [Attribute("physxMaterial:compliantContactDamping", 0.0)],
        },
        "body": {
            # Rigid body damping
            "rigid_body_linear_damping": [Attribute("physxRigidBody:linearDamping", 0.0)],
            "rigid_body_angular_damping": [Attribute("physxRigidBody:angularDamping", 0.05)],
        },
    }


def _solref_to_stiffness(solref):
    """Convert MuJoCo solref (timeconst, dampratio) to internal stiffness.

    k = 1 / (timeconst^2)
    """
    try:
        timeconst = float(solref[0])
    except Exception:
        return None
    # Direct mode: both negative → interpret as (damping, stiffness)
    if timeconst <= 0.0:
        return -timeconst
    return 1.0 / (timeconst * timeconst)


def _solref_to_damping(solref):
    """Convert MuJoCo solref (timeconst, dampratio) to internal damping.

    b = 2 * dampratio / timeconst
    """
    try:
        timeconst = float(solref[0])
        dampratio = float(solref[1])
    except Exception:
        return None
    # Direct mode: both negative → interpret as (damping, stiffness)
    if timeconst <= 0.0 or dampratio <= 0.0:
        return -dampratio
    return (2.0 * dampratio) / timeconst


class MjcPlugin(schemaPlugin):
    name: ClassVar[str] = "mjc"

    mapping: ClassVar[dict[str, dict[str, list[Attribute]]]] = {
        "scene": {
            "time_step": [Attribute("mjc:option:timestep", 0.002)],
            "max_solver_iterations": [Attribute("mjc:option:iterations", 100)],
            "enable_gravity": [Attribute("mjc:flag:gravity", True)],
            "contact_margin": [Attribute("mjc:option:o_margin", 0.0)],
        },
        "joint": {
            "armature": [Attribute("mjc:armature", 0.0)],
            "friction": [Attribute("mjc:frictionloss", 0.0)],
            # Per-axis linear aliases mapped to solref
            "limit_transX_ke": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_stiffness)],
            "limit_transY_ke": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_stiffness)],
            "limit_transZ_ke": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_stiffness)],
            "limit_transX_kd": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_damping)],
            "limit_transY_kd": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_damping)],
            "limit_transZ_kd": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_damping)],
            "limit_linear_ke": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_stiffness)],
            "limit_angular_ke": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_stiffness)],
            "limit_rotX_ke": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_stiffness)],
            "limit_rotY_ke": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_stiffness)],
            "limit_rotZ_ke": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_stiffness)],
            "limit_linear_kd": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_damping)],
            "limit_angular_kd": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_damping)],
            "limit_rotX_kd": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_damping)],
            "limit_rotY_kd": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_damping)],
            "limit_rotZ_kd": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_damping)],
        },
        "shape": {
            # Mesh
            "mesh_hull_vertex_limit": [Attribute("mjc:maxhullvert", -1)],
            # Collisions
            "rigid_contact_margin": [Attribute("mjc:margin", 0.0)],
        },
        "material": {
            # Materials and contact models
            "priority": [Attribute("mjc:priority", 0)],
            "weight": [Attribute("mjc:solmix", 1.0)],
            "stiffness": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_stiffness)],
            "damping": [Attribute("mjc:solref", [0.02, 1.0], _solref_to_damping)],
        },
        "body": {
            # Rigid body / joint domain
            "rigid_body_linear_damping": [Attribute("mjc:damping", 0.0)],
        },
        "actuator": {
            # Actuators
            "ctrl_low": [Attribute("mjc:ctrlRange:min", 0.0)],
            "ctrl_high": [Attribute("mjc:ctrlRange:max", 0.0)],
            "force_low": [Attribute("mjc:forceRange:min", 0.0)],
            "force_high": [Attribute("mjc:forceRange:max", 0.0)],
            "act_low": [Attribute("mjc:actRange:min", 0.0)],
            "act_high": [Attribute("mjc:actRange:max", 0.0)],
            "length_low": [Attribute("mjc:lengthRange:min", 0.0)],
            "length_high": [Attribute("mjc:lengthRange:max", 0.0)],
            "gainPrm": [Attribute("mjc:gainPrm", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])],
            "gainType": [Attribute("mjc:gainType", "fixed")],
            "biasPrm": [Attribute("mjc:biasPrm", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])],
            "biasType": [Attribute("mjc:biasType", "none")],
            "dynPrm": [Attribute("mjc:dynPrm", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])],
            "dynType": [Attribute("mjc:dynType", "none")],
            "gear": [Attribute("mjc:gear", [1, 0, 0, 0, 0, 0])],
        },
    }


class Resolver:
    def __init__(self, plugins: list[schemaPlugin], collect_engine_attrs: bool = True):
        """
        Initialize resolver with plugin instances in priority order.

        Args:
            plugins: List of instantiated plugins in priority order
        """
        # Use provided plugin instances directly
        self.plugins = list(plugins) if plugins is not None else []
        self._collect_engine_attrs = bool(collect_engine_attrs)

        # Dictionary to accumulate engine-specific attributes as prims are encountered
        # Pre-initialize maps for each configured plugin
        self.engine_specific_attrs: dict[str, dict[str, dict[str, Any]]] = {p.name: {} for p in self.plugins}

        # accumulator for special custom assignment attributes following the pattern:
        #   newton:assignment:frequency:variable_name
        # where assignment in {model, state, control, contact}
        # and frequency in {joint, joint_dof, joint_coord, body, shape}
        # we store per-variable specs and occurrences by prim path.
        self._custom_attributes: dict[tuple[str, str, str], dict[str, Any]] = {}

    def _collect_on_first_use(self, plugin: schemaPlugin, prim) -> None:
        """Collect and store engine-specific attributes for this plugin/prim on first use."""
        if prim is None:
            return
        if not self._collect_engine_attrs:
            return
        prim_path = str(prim.GetPath())
        if prim_path in self.engine_specific_attrs[plugin.name]:
            return
        attrs = plugin.collect_prim_engine_attrs(prim)
        if attrs:
            self.engine_specific_attrs[plugin.name][prim_path] = attrs

        # also scan and accumulate custom assignment attributes from the
        # "newton" engine-specific attributes we just collected
        newton_attrs = self.engine_specific_attrs.get("newton", {}).get(prim_path)
        if newton_attrs:
            self._accumulate_custom_attributes(prim_path, newton_attrs)

    def _parse_custom_attr_name(self, name: str) -> tuple[str, str, str] | None:
        """Parse names like 'newton:assignment:frequency:variable_name'."""
        try:
            head, assignment, frequency, variable = name.split(":", 3)
        except ValueError:
            return None
        if head != "newton":
            return None
        if assignment not in {"model", "state", "control", "contact"}:
            return None
        if frequency not in {"joint", "joint_dof", "joint_coord", "body", "shape"}:
            return None
        if not variable:
            return None
        return assignment, frequency, variable

    def _accumulate_custom_attributes(self, prim_path: str, attrs: dict[str, Any]) -> None:
        """collect custom attributes from a pre-fetched attribute map (name->value)."""

        def _usd_to_wp(v: Any):
            # Convert USD types to Warp-friendly representations
            try:
                # Handle Gf.Quat[f/d] → wp.quat(x, y, z, w) normalized
                if hasattr(v, "real") and hasattr(v, "imaginary"):
                    try:
                        return wp.normalize(wp.quat(*v.imaginary, v.real))
                    except Exception:
                        pass
            except Exception:
                pass
            return v

        def _infer_wp_dtype(v: Any):
            # Heuristic mapping from USD value to Warp dtype
            try:
                # Check for quat first (before generic length checks)
                if hasattr(v, "real") and hasattr(v, "imaginary"):
                    return wp.quat
                # wp.quat-like (object with x,y,z,w after conversion)
                if all(hasattr(v, c) for c in ("x", "y", "z", "w")):
                    return wp.quat
                # Vector3-like
                if hasattr(v, "__len__") and len(v) == 3:
                    return wp.vec3
                # Vector2-like
                if hasattr(v, "__len__") and len(v) == 2:
                    return wp.vec2
                # Vector4-like (but not quat)
                if hasattr(v, "__len__") and len(v) == 4:
                    return wp.vec4
            except Exception:
                pass
            if isinstance(v, bool):
                return wp.bool
            if isinstance(v, int):
                return wp.int32
            # default to float32 for scalars
            return wp.float32

        for name, value in attrs.items():
            parsed = self._parse_custom_attr_name(name)
            if not parsed:
                continue
            # Convert USD typed values (e.g., quatf) to Warp-friendly values
            converted_value = _usd_to_wp(value)
            assignment, frequency, variable = parsed
            key = (assignment, frequency, variable)
            spec = self._custom_attributes.get(key)
            if spec is None:
                dtype = _infer_wp_dtype(converted_value)
                spec = {
                    "assignment": assignment,
                    "frequency": frequency,
                    "name": variable,
                    "dtype": dtype,
                    "occurrences": {},
                }
                self._custom_attributes[key] = spec
            spec["occurrences"][prim_path] = converted_value

    def get_value(self, prim, prim_type: str, key: str, default: Any = None) -> Any:
        """
        Resolve value using engine priority, with layered fallbacks:

        1) First authored value found in plugin order (highest priority first)
        2) If none authored, use the provided 'default' argument if not None
        3) If still None, use the first non-None mapping default from plugins in priority order

        Args:
            prim: USD prim to query (for scene prim_type, this should be scene_prim)
            prim_type: Prim type ("scene", "joint", "shape", "body", "material", "actuator")
            key: Attribute key within the prim type
            default: Default value if not found

        Returns:
            Resolved value according to the precedence above.
        """
        # 1) Authored value by engine priority
        for p in self.plugins:
            got = p.get_value(prim, prim_type, key)
            if got is not None:
                val, _usd_attr = got
                if val is not None:
                    if self._collect_engine_attrs:
                        self._collect_on_first_use(p, prim)
                    return val

        # 2) Caller-provided default, if any
        if default is not None:
            return default

        # 3) Engine mapping defaults in priority order
        for plugin in self.plugins:
            specs = plugin.mapping.get(prim_type, {}).get(key, []) if hasattr(plugin, "mapping") else []
            for spec in specs:
                d = getattr(spec, "default", None)
                if d is not None:
                    return d

        # Nothing found
        try:
            prim_path = str(prim.GetPath()) if prim is not None else "<None>"
        except Exception:
            prim_path = "<invalid>"
        print(
            f"Error: Cannot resolve value for '{prim_type}:{key}' on prim '{prim_path}'; "
            f"no authored value, no explicit default, and no engine mapping default."
        )
        return None

    def collect_prim_engine_attrs(self, prim) -> None:
        """
        Collect and accumulate engine-specific attributes for a single prim.

        Args:
            prim: USD prim to collect engine attributes from
        """
        if prim is None:
            return

        prim_path = str(prim.GetPath())

        if not self._collect_engine_attrs:
            return
        for plugin in self.plugins:
            # only collect if we haven't seen this prim for this plugin
            if prim_path not in self.engine_specific_attrs[plugin.name]:
                attrs = plugin.collect_prim_engine_attrs(prim)
                if attrs:
                    self.engine_specific_attrs[plugin.name][prim_path] = attrs
                    # accumulate custom attributes from newton attrs if available
                    if plugin.name == "newton":
                        self._accumulate_custom_attributes(prim_path, attrs)

    def get_engine_specific_attrs(self) -> dict[str, dict[str, dict[str, Any]]]:
        """
        Get the accumulated engine-specific attributes.

        Returns:
            Dictionary with structure: engine_name -> prim_path -> {attr_name: attr_value}
            e.g., {"mjc": {"/World/Cube": {"mjc:option:timestep": 0.01}}}
        """
        return self.engine_specific_attrs.copy()

    def get_custom_attributes(self) -> dict[tuple[str, str, str], dict[str, Any]]:
        """
        get accumulated custom property specifications and occurrences.

        returns:
            dict keyed by (assignment, frequency, variable_name) with entries:
                {"assignment","frequency","name","dtype","default","occurrences"}
        """
        # return a shallow copy; nested dicts are fine to share for our usage
        return self._custom_attributes.copy()
