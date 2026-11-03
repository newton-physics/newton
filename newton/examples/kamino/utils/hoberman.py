# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate six coupled Hoberman rings without an authored scene asset.

Each pair of ring planes meets at two antipodal rays. Each ray carries an
inner and an outer ball joint. Subdivide the ten 36-degree arcs per ring to
change resolution while retaining all sixty inter-ring joints. Resolution two
recreates the original 240-strut, 421-joint sphere, within USD float precision.

Dimensions below use the original asset's meter scale; the example scales the
result to a 1 m resting diameter and 0.18 kg total mass.
"""

import math
from collections import deque
from dataclasses import dataclass
from itertools import product

import numpy as np
import warp as wp


@dataclass
class HobermanSphere:
    """Describe a mechanism in the original asset's meter scale."""

    bodies: list[tuple]
    """Records of (label, world pose, local center of mass [m], inertia [kg m²], mass [kg]).

    Poses contain (x, y, z, qx, qy, qz, qw), with positions in meters.
    Inertias are 3-by-3 matrices about the center of mass in body axes.
    """
    joints: list[tuple]
    """Records of (kind, label, parent, child, parent pose, child pose, axes, coordinates).

    Body indices use -1 for the world. Joint poses are relative to each body;
    axes are in the joint frame. Coordinates are a free pose, ball quaternion,
    or revolute angle [rad], according to kind.
    """
    tiles: list[tuple]
    """Visual box records of (body index, body-local pose, half-extents [m])."""
    tree_joint_count: int
    """Number of leading joints forming the articulation tree; remaining joints close loops."""
    opening_label: str
    """Label of the outer crossing hinge that drives the mechanism."""
    mount_body: int
    """Index of the body supported at the outer crossing."""
    mount_frame: tuple
    """Body-local mount pose, with position in meters."""
    pole: np.ndarray
    """World-space outer crossing position in the neutral mechanism [m]."""
    radius_rest: float
    """Mean radius of the body centers in the rest pose [m]."""
    opening_slope: float
    """Dimensionless sine coefficient in the radial gain cos(phi) + slope*sin(phi)."""
    reference_opening_slope: float
    """Radial gain coefficient for the original resolution-two mechanism."""
    diameter_rest: float
    """Diameter of the origin-centered bounding sphere enclosing the resting strut boxes [m]."""


def _rotation_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def _pose(position, rotation=None):
    quaternion = (0.0, 0.0, 0.0, 1.0) if rotation is None else tuple(wp.quat_from_matrix(wp.mat33(rotation)))
    return (*position, *quaternion)


def generate_sphere(resolution: int = 2) -> HobermanSphere:
    """Build a ring mechanism with an integer number of cells per 36-degree arc.

    Args:
        resolution: Positive integer; two gives twenty cells per ring. Every
            cell contains two bent struts, three revolute joints and four boxes.

    Returns:
        Neutral body, joint and visual geometry, with tree joints before loops.
        Body and joint ordering is generated rather than inherited from USD.
    """
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 1:
        raise ValueError("Hoberman resolution must be a positive integer")
    count = 10 * resolution
    step = 2.0 * math.pi / count
    reference_half_step = math.pi / 20
    outer, inner = 2.2, 1.35896
    width = 0.025 * min(1.0, 2.0 / resolution)
    bodies, joints, tiles = [], [], []
    centers, rotations, starts = [], [], []
    corner_signs = np.array(list(product((-1, 1), repeat=3)))
    radius_bounds = 0.0

    # The six unoriented icosahedral axes: one polar axis and five tilted axes.
    tilt = math.acos(1.0 / math.sqrt(5.0))
    for azimuth, sign in ((0, 0), (18, 1), (-18, -1), (54, -1), (-54, 1), (90, 1)):
        c, s = math.cos(sign * tilt), math.sin(sign * tilt)
        rotations.append(_rotation_z(math.radians(azimuth)) @ np.array(((c, 0, s), (0, 1, 0), (-s, 0, c))))

    def cell_anchors(half_step):
        c, s = math.cos(half_step), math.sin(half_step)
        return np.array(((inner * c, -inner * s), ((inner + outer) / (2 * c), 0), (outer * c, outer * s)))

    # One strut's neutral mesh centroid and inertia, expressed in its cell's
    # radial/tangential/normal basis. All other struts follow by symmetry.
    centroid = np.array((1.7787009786036, 0.0342589168447, 0.025))
    inertia = np.array(
        ((0.0775690539384, -0.111760600551, 0), (-0.111760600551, 0.169415565709, 0), (0, 0, 0.245869636536))
    )
    mass = 2.6759607791900635
    reference = np.column_stack((cell_anchors(reference_half_step), np.ones(3)))
    affine = np.linalg.solve(reference, cell_anchors(step / 2)).T
    deformation = np.zeros((3, 3))
    deformation[:2, :2] = affine[:, :2]
    deformation[2, 2] = width / 0.025
    centroid = deformation @ centroid + np.append(affine[:, 2], 0)
    covariance = 0.5 * np.trace(inertia) * np.eye(3) - inertia
    covariance = deformation @ covariance @ deformation.T
    inertia = np.trace(covariance) * np.eye(3) - covariance

    for ring, rotation in enumerate(rotations):
        # The tilted rings' intersections are shifted by 18 degrees. Odd
        # subdivisions need that phase shift; even ones retain the USD labels.
        start = math.radians(72 + (18 if ring and resolution % 2 else 0))
        starts.append(start)
        for cell in range(count):
            basis = rotation @ _rotation_z(start - cell * step + step / 2)
            anchors = np.column_stack((cell_anchors(step / 2), np.zeros(3)))
            for side in range(2):
                mirror = np.diag((1, 1 - 2 * side, 1 - 2 * side))
                frame = basis @ mirror
                center = frame @ centroid
                body = len(bodies)
                centers.append(center)
                bodies.append(
                    (
                        f"/hoberman_sphere/RigidBodies/circle{ring}_rb{2 * cell + side}",
                        _pose(center),
                        (0.0, 0.0, 0.0),
                        frame @ inertia @ frame.T,
                        mass,
                    )
                )
                for end in (anchors[0], anchors[2]):
                    direction = end - anchors[1]
                    distance = np.linalg.norm(direction)
                    direction /= distance
                    normal = frame[:, 2]
                    length_axis = frame @ direction
                    axes = np.column_stack((length_axis, np.cross(normal, length_axis), normal))
                    half_length = 0.5 * (distance + width)
                    # The hinge is on the slab face; offset the visual to its
                    # midplane, matching the original two-box strut rendering.
                    position = frame @ (anchors[1] + half_length * direction + np.array((0, 0, width)))
                    quantum = min(0.001, width)
                    extents = tuple(np.round(np.array((half_length, width, width)) / quantum) * quantum)
                    tiles.append((body, _pose(position - center, axes), extents))
                    corners = position + (corner_signs * extents) @ axes.T
                    radius_bounds = max(radius_bounds, float(np.linalg.norm(corners, axis=1).max()))

    def body_index(ring, cell, side):
        return ring * 2 * count + 2 * (cell % count) + side

    def anchor(ring, cell, radius):
        angle = starts[ring] - cell * step
        return rotations[ring] @ np.array((radius * math.cos(angle), radius * math.sin(angle), 0))

    def add_joint(kind, label, parent, child, position, normal=None):
        # Records remain in world coordinates until tree traversal assigns a
        # direction. Reversing both bodies and hinge axis preserves drive signs.
        joints.append((kind, label, parent, child, position, normal))

    for ring, rotation in enumerate(rotations):
        normal = rotation[:, 2]
        for cell in range(count):
            prefix = f"/hoberman_sphere/Joints/circle{ring}_"
            add_joint(
                "REVOLUTE",
                f"{prefix}inner{cell}",
                body_index(ring, cell, 0),
                body_index(ring, cell + 1, 1),
                anchor(ring, cell, inner),
                normal,
            )
            add_joint(
                "REVOLUTE",
                f"{prefix}outer{cell}",
                body_index(ring, cell + 1, 0),
                body_index(ring, cell, 1),
                anchor(ring, cell, outer),
                -normal,
            )
            midpoint = (
                rotations[ring]
                @ _rotation_z(starts[ring] - cell * step + step / 2)
                @ np.array(((inner + outer) / (2 * math.cos(step / 2)), 0, 0))
            )
            add_joint(
                "REVOLUTE", f"{prefix}mid{cell}", body_index(ring, cell, 0), body_index(ring, cell, 1), midpoint, normal
            )

    for ring_a in range(6):
        for ring_b in range(ring_a + 1, 6):
            direction = np.cross(rotations[ring_a][:, 2], rotations[ring_b][:, 2])
            direction /= np.linalg.norm(direction)
            for sign in (-1, 1):
                ray = sign * direction
                cells = []
                for ring in (ring_a, ring_b):
                    local = rotations[ring].T @ ray
                    angle = math.atan2(local[1], local[0])
                    cell = round((starts[ring] - angle) / step) % count
                    cells.append(cell)
                for side, radius in enumerate((inner, outer)):
                    add_joint(
                        "BALL",
                        f"/hoberman_sphere/Joints/cross_{ring_a}_{ring_b}_{sign}_{side}",
                        body_index(ring_a, cells[0], side),
                        body_index(ring_b, cells[1], side),
                        radius * ray,
                    )

    mount_cell = 7 * resolution
    mount_body = body_index(0, mount_cell, 1)
    pole = np.array((-outer, 0.0, 0.0))
    mount_rotation = np.array(((0, 1, 0), (0, 0, -1), (-1, 0, 0)))
    opening_label = f"/hoberman_sphere/Joints/circle0_outer{mount_cell}"

    # Traverse the actual joint graph to construct a valid parent-before-child
    # articulation. Remaining edges close its loops, independent of resolution.
    adjacency = [[] for _ in bodies]
    for index, joint in enumerate(joints):
        adjacency[joint[2]].append(index)
        adjacency[joint[3]].append(index)
    visited = {mount_body}
    queue = deque((mount_body,))
    tree, selected = [], set()
    while queue:
        parent = queue.popleft()
        for index in adjacency[parent]:
            kind, label, a, b, position, normal = joints[index]
            child = b if a == parent else a
            if child in visited:
                continue
            visited.add(child)
            queue.append(child)
            selected.add(index)
            tree.append((kind, label, parent, child, position, normal if a == parent or normal is None else -normal))
    if len(visited) != len(bodies):
        raise RuntimeError("Hoberman joint graph is disconnected")
    ordered = [
        (
            "FREE",
            "flower_root",
            -1,
            mount_body,
            _pose(np.zeros(3)),
            _pose(np.zeros(3)),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)) * 2,
            bodies[mount_body][1],
        )
    ]
    for kind, label, parent, child, position, normal in tree + [j for i, j in enumerate(joints) if i not in selected]:
        frame = None
        axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        coordinates = (0.0, 0.0, 0.0, 1.0)
        if normal is not None:
            tangent = np.cross(normal, position)
            tangent /= np.linalg.norm(tangent)
            frame = np.column_stack((normal, tangent, np.cross(normal, tangent)))
            axes, coordinates = ((1.0, 0.0, 0.0),), (0.0,)
        ordered.append(
            (
                kind,
                label,
                parent,
                child,
                _pose(position - centers[parent], frame),
                _pose(position - centers[child], frame),
                axes,
                coordinates,
            )
        )

    # The mean pin radius changes by cos(phi) + opening_slope*sin(phi),
    # where phi is half the driven outer hinge's rotation from rest.
    opening_slope = (outer - inner) / (outer + inner) / math.tan(step / 2)
    return HobermanSphere(
        bodies,
        ordered,
        tiles,
        len(bodies),
        opening_label,
        mount_body,
        _pose(pole - centers[mount_body], mount_rotation),
        pole,
        float(np.linalg.norm(centers[0])),
        opening_slope,
        (outer - inner) / (outer + inner) / math.tan(reference_half_step),
        2.0 * radius_bounds,
    )
