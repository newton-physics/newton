import enum


class GeomType(enum.IntEnum):
    PLANE = 0
    # HFIELD = 1
    SPHERE = 2
    CAPSULE = 3
    ELLIPSOID = 4
    CYLINDER = 5
    BOX = 6
    MESH = 7
    # SDF = 8
    CONE = 9
    # CONVEX_MESH = 10
    NONE = 11


class LightType(enum.IntEnum):
    SPOTLIGHT = 0
    DIRECTIONAL = 1
