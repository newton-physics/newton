from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass
class AttrSpec:
    """
    Specifies a USD attribute and its transformation function.

    Args:
        usd_name (str): The name of the USD attribute.
        transform (Callable[[Any], Any] | None): A function to transform the attribute value.
    """

    usd_name: str
    transform: Callable[[Any], Any] | None = None


class EngineSchemaPlugin:
    name: str
    # mapping is a dictionary for known variables in Newton. Its purpose is to map usd attributes to exisiting Newton data.
    # prim_type -> variable -> list[AttrSpec]
    mapping: dict[str, dict[str, list[AttrSpec]]]

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

    @property
    def engine_attr_prefix(self) -> str:
        """Return the main engine attribute prefix (e.g., 'newton:', 'physx', 'mjc:')"""
        return f"{self.name}:"

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
        all_prefixes = [self.engine_attr_prefix]
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


def _collect_engine_specific_attrs(prim, prefixes: list[str]) -> dict[str, Any]:
    """Collect engine-specific attributes authored on the prim with the given prefixes.
    These attributes don't have direct mappings in the plugin mapping, but are still important to collect for later use"""

    out = {}
    try:
        attrs = prim.GetAttributes()
    except Exception:
        return out
    for a in attrs:
        try:
            name = a.GetName()
        except Exception:
            continue
        if any(name.startswith(p) for p in prefixes):
            if a.IsValid() and a.HasAuthoredValue():
                try:
                    out[name] = a.Get()
                except Exception:
                    pass
    return out


class NewtonPlugin(EngineSchemaPlugin):
    name: ClassVar[str] = "newton"
    mapping: ClassVar[dict[str, dict[str, list[AttrSpec]]]] = {
        "scene": {
            "time_step": [AttrSpec("newton:timeStep")],
            "max_solver_iterations": [AttrSpec("newton:maxSolverIterations")],
            "enable_gravity": [AttrSpec("newton:enableGravity")],
            "contact_offset": [AttrSpec("newton:contactOffset")],
        },
        "joint": {
            "armature": [AttrSpec("newton:armature")],
            "limit_linear_ke": [AttrSpec("newton:linear:limitStiffness")],
            "limit_angular_ke": [AttrSpec("newton:angular:limitStiffness")],
            "limit_rotX_ke": [AttrSpec("newton:rotX:limitStiffness")],
            "limit_rotY_ke": [AttrSpec("newton:rotY:limitStiffness")],
            "limit_rotZ_ke": [AttrSpec("newton:rotZ:limitStiffness")],
            "limit_linear_kd": [AttrSpec("newton:linear:limitDamping")],
            "limit_angular_kd": [AttrSpec("newton:angular:limitDamping")],
            "limit_rotX_kd": [AttrSpec("newton:rotX:limitDamping")],
            "limit_rotY_kd": [AttrSpec("newton:rotY:limitDamping")],
            "limit_rotZ_kd": [AttrSpec("newton:rotZ:limitDamping")],
            "friction": [AttrSpec("newton:friction")],
            "angular_position": [AttrSpec("newton:angular:position")],
            "linear_position": [AttrSpec("newton:linear:position")],
            "rotX_position": [AttrSpec("newton:rotX:position")],
            "rotY_position": [AttrSpec("newton:rotY:position")],
            "rotZ_position": [AttrSpec("newton:rotZ:position")],
            "angular_velocity": [AttrSpec("newton:angular:velocity")],
            "linear_velocity": [AttrSpec("newton:linear:velocity")],
            "rotX_velocity": [AttrSpec("newton:rotX:velocity")],
            "rotY_velocity": [AttrSpec("newton:rotY:velocity")],
            "rotZ_velocity": [AttrSpec("newton:rotZ:velocity")],
        },
        "shape": {
            "mesh_hull_vertex_limit": [AttrSpec("newton:hullVertexLimit")],
            "collision_contact_offset": [AttrSpec("newton:collision:contactOffset")],
        },
        "body": {
            "rigid_body_damping": [AttrSpec("newton:damping")],
        },
        "material": {
            "priority": [AttrSpec("newton:priority")],
            "weight": [AttrSpec("newton:weight")],
            "stiffness": [AttrSpec("newton:stiffness")],
            "damping": [AttrSpec("newton:damping")],
        },
        "actuator": {
            "ctrl_low": [AttrSpec("newton:ctrlRange:low")],
            "ctrl_high": [AttrSpec("newton:ctrlRange:high")],
            "force_low": [AttrSpec("newton:forceRange:low")],
            "force_high": [AttrSpec("newton:forceRange:high")],
            "act_low": [AttrSpec("newton:actRange:low")],
            "act_high": [AttrSpec("newton:actRange:high")],
            "length_low": [AttrSpec("newton:lengthRange:low")],
            "length_high": [AttrSpec("newton:lengthRange:high")],
            "gainPrm": [AttrSpec("newton:gainPrm")],
            "gainType": [AttrSpec("newton:gainType")],
            "biasPrm": [AttrSpec("newton:biasPrm")],
            "biasType": [AttrSpec("newton:biasType")],
            "dynPrm": [AttrSpec("newton:dynPrm")],
            "dynType": [AttrSpec("newton:dynType")],
            "speedTorqueGradient": [AttrSpec("newton:speedTorqueGradient")],
            "torqueSpeedGradient": [AttrSpec("newton:torqueSpeedGradient")],
            "maxVelocity": [AttrSpec("newton:maxVelocity")],
            "gear": [AttrSpec("newton:gear")],
        },
    }


class PhysxPlugin(EngineSchemaPlugin):
    name: ClassVar[str] = "physx"
    extra_attr_namespaces: ClassVar[list[str]] = [
        # Scene and rigid body
        "physxScene:",
        "physxRigidBody:",
        # Collisions and meshes
        "physxCollision:",
        "physxConvexHullCollision:",
        "physxConvexDecompositionCollision:",
        "physxTriangleMeshCollision:",
        "physxTriangleMeshSimplificationCollision:",
        "physxSDFMeshCollision:",
        # Materials
        "physxMaterial:",
        # Joints and limits
        "physxJoint:",
        "physxLimit:",
        # Articulations
        "physxArticulation:",
    ]

    @property
    def engine_attr_prefix(self) -> str:
        """PhysX uses multiple prefixes, so we return the main one"""
        return "physx"

    mapping: ClassVar[dict[str, dict[str, list[AttrSpec]]]] = {
        "scene": {
            "time_step": [
                AttrSpec("physxScene:timeStepsPerSecond", lambda hz: (1.0 / hz) if (hz and hz > 0) else None)
            ],
            "max_solver_iterations": [AttrSpec("physxScene:maxVelocityIterationCount")],
            "enable_gravity": [AttrSpec("physxRigidBody:disableGravity", lambda value: not value)],
        },
        "joint": {
            "armature": [AttrSpec("physxJoint:armature")],
            # Per-axis linear limit aliases
            "limit_transX_ke": [AttrSpec("physxLimit:linear:stiffness")],
            "limit_transY_ke": [AttrSpec("physxLimit:linear:stiffness")],
            "limit_transZ_ke": [AttrSpec("physxLimit:linear:stiffness")],
            "limit_transX_kd": [AttrSpec("physxLimit:linear:damping")],
            "limit_transY_kd": [AttrSpec("physxLimit:linear:damping")],
            "limit_transZ_kd": [AttrSpec("physxLimit:linear:damping")],
            "limit_linear_ke": [AttrSpec("physxLimit:linear:stiffness")],
            "limit_angular_ke": [AttrSpec("physxLimit:angular:stiffness")],
            "limit_rotX_ke": [AttrSpec("physxLimit:rotX:stiffness")],
            "limit_rotY_ke": [AttrSpec("physxLimit:rotY:stiffness")],
            "limit_rotZ_ke": [AttrSpec("physxLimit:rotZ:stiffness")],
            "limit_linear_kd": [AttrSpec("physxLimit:linear:damping")],
            "limit_angular_kd": [AttrSpec("physxLimit:angular:damping")],
            "limit_rotX_kd": [AttrSpec("physxLimit:rotX:damping")],
            "limit_rotY_kd": [AttrSpec("physxLimit:rotY:damping")],
            "limit_rotZ_kd": [AttrSpec("physxLimit:rotZ:damping")],
            "angular_position": [AttrSpec("state:angular:physics:position")],
            "linear_position": [AttrSpec("state:linear:physics:position")],
            "rotX_position": [AttrSpec("state:rotX:physics:position")],
            "rotY_position": [AttrSpec("state:rotY:physics:position")],
            "rotZ_position": [AttrSpec("state:rotZ:physics:position")],
            "angular_velocity": [AttrSpec("state:angular:physics:velocity")],
            "linear_velocity": [AttrSpec("state:linear:physics:velocity")],
            "rotX_velocity": [AttrSpec("state:rotX:physics:velocity")],
            "rotY_velocity": [AttrSpec("state:rotY:physics:velocity")],
            "rotZ_velocity": [AttrSpec("state:rotZ:physics:velocity")],
        },
        "shape": {
            # Mesh hull vertex limit
            "mesh_hull_vertex_limit": [AttrSpec("physxConvexHullCollision:hullVertexLimit")],
            # Collision contact offset
            "collision_contact_offset": [AttrSpec("physxCollision:contactOffset")],
        },
        "material": {
            "stiffness": [AttrSpec("physxMaterial:compliantContactStiffness")],
            "damping": [AttrSpec("physxMaterial:compliantContactDamping")],
        },
        "body": {
            # Rigid body damping
            "rigid_body_damping": [AttrSpec("physxRigidBody:linearDamping"), AttrSpec("physxRigidBody:angularDamping")],
        },
    }


def _solref_to_stiffness(solref):
    """Convert MuJoCo solref (timeconst, dampratio) to internal stiffness.

    k = 1 / (timeconst^2)
    """
    try:
        timeconst = float(solref[0])
        dampratio = float(solref[1]) if len(solref) > 1 else None
    except Exception:
        return None
    # Direct mode: both negative → interpret as (damping, stiffness)
    if timeconst is not None and timeconst < 0.0 and dampratio < 0.0:
        return -timeconst
    if not (timeconst and timeconst > 0.0):
        return None
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
    if timeconst < 0.0 and dampratio < 0.0:
        return -dampratio
    if not (dampratio and dampratio > 0.0):
        return None
    return (2.0 * dampratio) / timeconst


class MjcPlugin(EngineSchemaPlugin):
    name: ClassVar[str] = "mjc"

    mapping: ClassVar[dict[str, dict[str, list[AttrSpec]]]] = {
        "scene": {
            "time_step": [AttrSpec("mjc:option:timestep")],
            "max_solver_iterations": [AttrSpec("mjc:option:iterations")],
            "enable_gravity": [AttrSpec("mjc:flag:gravity")],
            "contact_offset": [AttrSpec("mjc:option:o_margin")],
        },
        "joint": {
            "armature": [AttrSpec("mjc:armature")],
            "friction": [AttrSpec("mjc:frictionloss")],
            # Per-axis linear aliases mapped to solref
            "limit_transX_ke": [AttrSpec("mjc:solref", _solref_to_stiffness)],
            "limit_transY_ke": [AttrSpec("mjc:solref", _solref_to_stiffness)],
            "limit_transZ_ke": [AttrSpec("mjc:solref", _solref_to_stiffness)],
            "limit_transX_kd": [AttrSpec("mjc:solref", _solref_to_damping)],
            "limit_transY_kd": [AttrSpec("mjc:solref", _solref_to_damping)],
            "limit_transZ_kd": [AttrSpec("mjc:solref", _solref_to_damping)],
            "limit_linear_ke": [AttrSpec("mjc:solref", _solref_to_stiffness)],
            "limit_angular_ke": [AttrSpec("mjc:solref", _solref_to_stiffness)],
            "limit_rotX_ke": [AttrSpec("mjc:solref", _solref_to_stiffness)],
            "limit_rotY_ke": [AttrSpec("mjc:solref", _solref_to_stiffness)],
            "limit_rotZ_ke": [AttrSpec("mjc:solref", _solref_to_stiffness)],
            "limit_linear_kd": [AttrSpec("mjc:solref", _solref_to_damping)],
            "limit_angular_kd": [AttrSpec("mjc:solref", _solref_to_damping)],
            "limit_rotX_kd": [AttrSpec("mjc:solref", _solref_to_damping)],
            "limit_rotY_kd": [AttrSpec("mjc:solref", _solref_to_damping)],
            "limit_rotZ_kd": [AttrSpec("mjc:solref", _solref_to_damping)],
        },
        "shape": {
            # Mesh
            "mesh_hull_vertex_limit": [AttrSpec("mjc:maxhullvert")],
            # Collisions
            "collision_contact_offset": [AttrSpec("mjc:margin")],
        },
        "material": {
            # Materials and contact models
            "priority": [AttrSpec("mjc:priority")],
            "weight": [AttrSpec("mjc:solmix")],
            "stiffness": [AttrSpec("mjc:solref", _solref_to_stiffness)],
            "damping": [AttrSpec("mjc:solref", _solref_to_damping)],
        },
        "body": {
            # Rigid body / joint domain
            "rigid_body_damping": [AttrSpec("mjc:damping")],
        },
        "actuator": {
            # Actuators
            "ctrl_low": [AttrSpec("mjc:ctrlRange:min")],
            "ctrl_high": [AttrSpec("mjc:ctrlRange:max")],
            "force_low": [AttrSpec("mjc:forceRange:min")],
            "force_high": [AttrSpec("mjc:forceRange:max")],
            "act_low": [AttrSpec("mjc:actRange:min")],
            "act_high": [AttrSpec("mjc:actRange:max")],
            "length_low": [AttrSpec("mjc:lengthRange:min")],
            "length_high": [AttrSpec("mjc:lengthRange:max")],
            "gainPrm": [AttrSpec("mjc:gainPrm")],
            "gainType": [AttrSpec("mjc:gainType")],
            "biasPrm": [AttrSpec("mjc:biasPrm")],
            "biasType": [AttrSpec("mjc:biasType")],
            "dynPrm": [AttrSpec("mjc:dynPrm")],
            "dynType": [AttrSpec("mjc:dynType")],
            "gear": [AttrSpec("mjc:gear")],
        },
    }


class Resolver:
    def __init__(self, engine_priority: list[str]):
        """
        Initialize resolver with engine priority list.

        Args:
            engine_priority: List of engine names in priority order (e.g., ["newton", "physx", "mjc"])
        """
        # Available plugin classes
        available_plugins = {
            "newton": NewtonPlugin,
            "physx": PhysxPlugin,
            "mjc": MjcPlugin,
        }

        # Construct plugins based on priority order
        self.plugins = []
        for name in engine_priority:
            if name in available_plugins:
                self.plugins.append(available_plugins[name]())

        # Dictionary to accumulate engine-specific attributes as prims are encountered
        # Pre-initialize maps for each configured plugin
        self.engine_specific_attrs: dict[str, dict[str, dict[str, Any]]] = {p.name: {} for p in self.plugins}

        # accumulator for special custom assignment attributes following the pattern:
        #   newton:assignment:frequency:variable_name
        # where assignment in {model, state, control, contact}
        # and frequency in {joint, joint_dof, joint_coord, body, shape}
        # we store per-variable specs and occurrences by prim path.
        self._custom_properties: dict[tuple[str, str, str], dict[str, Any]] = {}

    def _collect_on_first_use(self, plugin: EngineSchemaPlugin, prim) -> None:
        """Collect and store engine-specific attributes for this plugin/prim on first use."""
        if prim is None:
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
            self._accumulate_custom_properties(prim_path, newton_attrs)

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

    def _accumulate_custom_properties(self, prim_path: str, attrs: dict[str, Any]) -> None:
        """collect custom properties from a pre-fetched attribute map (name->value)."""
        for name, value in attrs.items():
            parsed = self._parse_custom_attr_name(name)
            if not parsed:
                continue
            assignment, frequency, variable = parsed
            key = (assignment, frequency, variable)
            spec = self._custom_properties.get(key)
            if spec is None:
                data_type = type(value).__name__
                spec = {
                    "assignment": assignment,
                    "frequency": frequency,
                    "name": variable,
                    "data_type": data_type,
                    "occurrences": {},
                }
                self._custom_properties[key] = spec
            spec["occurrences"][prim_path] = value

    def get_value(self, prim, prim_type: str, key: str, default: Any = None) -> Any:
        """
        Get attribute value for a given prim type and key with plugin precedence.

        Args:
            prim: USD prim to query (for scene prim_type, this should be scene_prim)
            prim_type: Prim type ("scene", "joint", "shape", "body", "material", "actuator")
            key: Attribute key within the prim type
            default: Default value if not found

        Returns:
            Attribute value if found, default otherwise
        """
        for p in self.plugins:
            got = p.get_value(prim, prim_type, key)
            if got is not None:
                val, _usd_attr = got
                if val is not None:
                    self._collect_on_first_use(p, prim)
                    return val
        return default

    def collect_prim_engine_attrs(self, prim) -> None:
        """
        Collect and accumulate engine-specific attributes for a single prim.

        Args:
            prim: USD prim to collect engine attributes from
        """
        if prim is None:
            return

        prim_path = str(prim.GetPath())

        for plugin in self.plugins:
            # only collect if we haven't seen this prim for this plugin
            if prim_path not in self.engine_specific_attrs[plugin.name]:
                attrs = plugin.collect_prim_engine_attrs(prim)
                if attrs:
                    self.engine_specific_attrs[plugin.name][prim_path] = attrs
                    # accumulate custom properties from newton attrs if available
                    if plugin.name == "newton":
                        self._accumulate_custom_properties(prim_path, attrs)

    def get_engine_specific_attrs(self) -> dict[str, dict[str, dict[str, Any]]]:
        """
        Get the accumulated engine-specific attributes.

        Returns:
            Dictionary with structure: engine_name -> prim_path -> {attr_name: attr_value}
            e.g., {"mjc": {"/World/Cube": {"mjc:option:timestep": 0.01}}}
        """
        return self.engine_specific_attrs.copy()

    def get_custom_properties(self) -> dict[tuple[str, str, str], dict[str, Any]]:
        """
        get accumulated custom property specifications and occurrences.

        returns:
            dict keyed by (assignment, frequency, variable_name) with entries:
                {"assignment","frequency","name","data_type","default","occurrences"}
        """
        # return a shallow copy; nested dicts are fine to share for our usage
        return self._custom_properties.copy()
