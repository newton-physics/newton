import warp as wp

from .collide import CollisionPipeline, Contacts
from .inertia import compute_shape_inertia, transform_inertia
from .kernels import compute_shape_radius
from .types import (
    GEO_BOX,
    GEO_CAPSULE,
    GEO_CONE,
    GEO_CYLINDER,
    GEO_MESH,
    GEO_NONE,
    GEO_PLANE,
    GEO_SDF,
    GEO_SPHERE,
    SDF,
    Mesh,
)


@wp.func
def create_sphere(radius: float):
    return (GEO_SPHERE, wp.vec3(radius, 0.0, 0.0))


@wp.func
def create_box(width: float, height: float, depth: float):
    return (GEO_BOX, wp.vec3(width, height, depth))


@wp.func
def create_capsule(radius: float, height: float):
    return (GEO_CAPSULE, wp.vec3(radius, height, 0.0))


@wp.func
def create_cylinder(radius: float, height: float):
    return (GEO_CYLINDER, wp.vec3(radius, height, 0.0))


@wp.func
def create_cone(radius: float, height: float):
    return (GEO_CONE, wp.vec3(radius, height, 0.0))


@wp.func
def create_plane(width: float = 0.0, height: float = 0.0):
    """Create a plane. If width and height are 0.0, creates an infinite
    plane."""
    return (GEO_PLANE, wp.vec3(width, height, 0.0))


@wp.func
def create_none():
    """Create an empty/null geometry."""
    return (GEO_NONE, wp.vec3(0.0, 0.0, 0.0))


# Only expose what's commonly needed for independent use
__all__ = [
    "GEO_BOX",
    "GEO_CAPSULE",
    "GEO_CONE",
    "GEO_CYLINDER",
    "GEO_MESH",
    "GEO_NONE",
    "GEO_PLANE",
    "GEO_SDF",
    # Geometry type constants (most commonly used)
    "GEO_SPHERE",
    "SDF",
    "CollisionPipeline",
    "Contacts",
    # Geometry classes
    "Mesh",
    "compute_shape_inertia",
    "compute_shape_radius",
    "create_box",
    "create_capsule",
    "create_cone",
    "create_cylinder",
    "create_none",
    "create_plane",
    "create_sphere",
    "transform_inertia",
]
