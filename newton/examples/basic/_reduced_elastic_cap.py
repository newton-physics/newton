# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Procedural slit cap and a single, setup-time POD reduced model."""

import itertools
import math

import numpy as np

import newton


def cap_mesh(length=0.010, radius=0.046, thickness=0.00012, subdivisions=16):
    """One connected surface; slit edges are separated, flap roots share the ring."""
    points, triangles, sector, depth = [], [], [], []
    known = {}

    def vertex(x, y, flap=-1, s=0.0):
        key = (round(x, 10), round(y, 10))
        if key in known:
            return known[key]
        k = len(points)
        known[key] = k
        points.append((x, y, 0.0))
        sector.append(flap)
        depth.append(s)
        return k

    for flap in range(4):
        a = flap * math.pi / 2
        normal = np.array([math.cos(a), math.sin(a)])
        tangent = np.array([-math.sin(a), math.cos(a)])
        indices = {}
        for i in range(subdivisions + 1):
            s = i / subdivisions
            for j in range(subdivisions - i + 1):
                eta = 0.0 if i == subdivisions else 2 * j / (subdivisions - i) - 1
                x = length * (1 - s) + 0.00008 * s
                y = length * (1 - s) * eta * (1 - 0.008 * s)
                xy = normal * x + tangent * y
                indices[i, j] = vertex(*xy, flap=flap, s=1 - x / length)
        for i in range(subdivisions):
            for j in range(subdivisions - i):
                # Viewed from +Z, these triangles have positive winding.
                triangles.append((indices[i, j], indices[i, j + 1], indices[i + 1, j]))
                if i + j < subdivisions - 1:
                    triangles.append((indices[i + 1, j], indices[i, j + 1], indices[i + 1, j + 1]))
        bands = [[indices[0, j] for j in range(subdivisions + 1)]]
        for layer in range(1, 5):
            band = []
            for j in range(subdivisions + 1):
                root = normal * length + tangent * length * (2 * j / subdivisions - 1)
                outside = root / np.linalg.norm(root) * radius
                xy = root + (outside - root) * (layer / 4)
                band.append(vertex(*xy))
            bands.append(band)
        for inner, outer in itertools.pairwise(bands):
            for j in range(subdivisions):
                triangles.extend([(inner[j], outer[j], outer[j + 1]), (inner[j], outer[j + 1], inner[j + 1])])
    mid = np.asarray(points, dtype=np.float64)
    tri = np.asarray(triangles, dtype=np.int32)
    area = np.linalg.norm(np.cross(mid[tri[:, 1]] - mid[tri[:, 0]], mid[tri[:, 2]] - mid[tri[:, 0]]), axis=1) / 2
    assert np.min(area) > 0
    n = len(mid)
    top, bottom = mid.copy(), mid.copy()
    top[:, 2] = thickness / 2
    bottom[:, 2] = -thickness / 2
    faces = tri.tolist() + [(int(c + n), int(b + n), int(a + n)) for a, b, c in tri]
    edges = {}
    for a, b, c in tri:
        for u, v in [(a, b), (b, c), (c, a)]:
            edges.setdefault(tuple(sorted((int(u), int(v)))), []).append((int(u), int(v)))
    for edge in edges.values():
        assert len(edge) <= 2
        if len(edge) == 1:
            a, b = edge[0]
            faces.extend([(a, a + n, b), (b, a + n, b + n)])
    return (
        np.concatenate([top, bottom]),
        np.asarray(faces, dtype=np.int32),
        mid,
        tri,
        np.asarray(sector),
        np.asarray(depth),
        area,
    )


def build_cap_rom(cfg):
    v, f, mid, tri, sector, s, area = cap_mesh(cfg.flap_length, cfg.cap_radius, cfg.thickness, cfg.subdivisions)
    # Large-rotation poses from arclength-preserving cylindrical bending of
    # each tongue. These are geometric exemplars, not FEM simulation output.
    nodes, weights = np.polynomial.legendre.leggauss(48)
    samples = []
    energies = []
    angles = (25, 40, 55, 70, 85)
    decay = 0.20
    denom = 1 - math.exp(-1 / decay)
    D = cfg.young_modulus * cfg.thickness**3 / (12 * (1 - cfg.poisson_ratio**2))
    xi = (nodes + 1) / 2
    dtheta = np.exp(-xi / decay) / (decay * denom)
    curvature_integral = float(np.sum(weights / 2 * 2 * (1 - xi) * dtheta**2))
    for flap in range(4):
        ids = np.flatnonzero((sector == flap) & (s > 0))
        a = flap * math.pi / 2
        normal = np.array([math.cos(a), math.sin(a), 0.0])
        for deg in angles:
            angle = math.radians(deg)
            u = s[ids, None] * (nodes[None, :] + 1) / 2
            theta = angle * (1 - np.exp(-u / decay)) / denom
            arc_x = cfg.flap_length * s[ids] * np.sum(np.cos(theta) * weights[None, :] / 2, axis=1)
            arc_z = -cfg.flap_length * s[ids] * np.sum(np.sin(theta) * weights[None, :] / 2, axis=1)
            disp = np.zeros_like(mid)
            disp[ids] = (cfg.flap_length * s[ids] - arc_x)[:, None] * normal
            disp[ids, 2] = arc_z
            samples.append(disp)
            energies.append(0.5 * D * angle**2 * curvature_integral)
    mass = np.zeros(len(mid), dtype=float)
    np.add.at(mass, tri.flatten(), np.repeat(area * cfg.thickness * cfg.density / 3, 3))
    samples = np.asarray(samples)
    pod = newton.ModalGeneratorPOD(
        sample_points=mid,
        displacements=samples,
        mode_count=4,
        total_mass=float(mass.sum()),
        stiffness_scale=1.0,
        damping_ratio=0.0,
        subtract_mean=False,
        label="whole_cap_slit_fold_POD",
    ).build()
    P = np.asarray(pod.sample_phi, dtype=float).transpose(0, 2, 1).reshape(len(mid) * 3, 4)
    W = np.repeat(mass, 3)
    M = P.T @ (W[:, None] * P)
    eigenvalues, U = np.linalg.eigh(M)
    assert eigenvalues.min() > 0
    P = P @ (U / np.sqrt(eigenvalues)[None, :])
    # Mass whiten at setup; all four equivalent flap directions have the same
    # frequency. Fit that scalar to bending energy across the snapshot set.
    Q = samples.reshape(len(samples), -1) @ (W[:, None] * P)
    predictors = 0.5 * np.sum(Q * Q, axis=1)
    omega2 = float(np.dot(predictors, energies) / np.dot(predictors, predictors))
    scales = np.linalg.norm(P.reshape(len(mid), 3, 4), axis=1).max(axis=0)
    P /= scales[None, :]
    masses = 1 / scales**2
    stiffness = omega2 * masses
    damping = 2 * cfg.damping_ratio * np.sqrt(masses * stiffness)
    phi = P.reshape(len(mid), 3, 4).transpose(0, 2, 1)
    # Top and bottom follow the same thin midsurface field. The rendering mesh
    # is a thin closed skin, not an independent hidden collision mesh.
    phi = np.concatenate([phi, phi], axis=0).astype(np.float32)
    sample_mass = np.r_[mass / 2, mass / 2]
    basis = newton.ModalBasis(
        sample_points=v,
        sample_phi=phi,
        sample_mass=sample_mass,
        mode_mass=masses,
        mode_stiffness=stiffness,
        mode_damping=damping,
        label="whole_cap_four_mode_POD",
    )
    projected = (Q @ (P * scales[None, :]).T).reshape(samples.shape)
    rms = float(np.sqrt(np.sum(mass[None, :, None] * (samples - projected) ** 2) / (len(samples) * mass.sum())))
    max_error = float(np.linalg.norm(samples - projected, axis=2).max())
    gram = P.T @ (W[:, None] * P)
    np.testing.assert_allclose(gram, np.diag(masses), rtol=1e-5, atol=1e-12)
    fit = omega2 * predictors
    audit = {
        "mode_count": 4,
        "snapshot_count": len(samples),
        "snapshot_angles_deg": list(angles),
        "geometry": "one connected cap mesh with clamped outer annulus and pre-cut cross slit",
        "method": "POD of geometric, arclength-preserving bent-flap poses; not FEM snapshots",
        "material_energy": "isotropic cylindrical plate bending; one fitted modal frequency",
        "snapshot_mass_weighted_rms_error_m": rms,
        "snapshot_max_vertex_error_m": max_error,
        "bending_energy_relative_rms_fit_error": float(np.linalg.norm(fit - energies) / np.linalg.norm(energies)),
        "mode_frequency_hz": math.sqrt(omega2) / (2 * math.pi),
        "cap_mass_kg": float(mass.sum()),
        "mode_mass_kg": masses.tolist(),
        "mode_stiffness_N_per_m": stiffness.tolist(),
        "limitations": [
            "Linear subspace is not an inextensibility constraint.",
            "Four retained modes favor downward insertion; not arbitrary buckling or upward folding.",
            "Material values are assumed, not identified from the reference.",
            "No lid self-contact, tearing, plasticity, or fluid.",
        ],
    }
    return v.astype(np.float32), f, basis, audit, mid, tri, samples, sector, s


def ring_mesh(inner, outer, z, thickness, segments=128):
    p = []
    for dz in (thickness / 2, -thickness / 2):
        for r in (inner, outer):
            for i in range(segments):
                a = 2 * math.pi * i / segments
                p.append((r * math.cos(a), r * math.sin(a), z + dz))
    f = []
    for i in range(segments):
        j = (i + 1) % segments
        for a, b, c, d in [
            (i, segments + i, segments + j, j),
            (2 * segments + i, 2 * segments + j, 3 * segments + j, 3 * segments + i),
            (i, j, 2 * segments + j, 2 * segments + i),
            (segments + i, 3 * segments + i, 3 * segments + j, segments + j),
        ]:
            f.extend([(a, b, c), (a, c, d)])
    return np.asarray(p, dtype=np.float32), np.asarray(f, dtype=np.int32)


def cup_mesh(height=0.125, bottom_radius=0.032, top_radius=0.046, segments=128):
    p = []
    for r, z in [
        (bottom_radius, 0.0),
        (top_radius, height),
        (bottom_radius - 0.0006, 0.0006),
        (top_radius - 0.0006, height),
    ]:
        for i in range(segments):
            a = 2 * math.pi * i / segments
            p.append((r * math.cos(a), r * math.sin(a), z))
    f = []
    for i in range(segments):
        j = (i + 1) % segments
        for a, b, c, d in [
            (i, j, segments + j, segments + i),
            (2 * segments + i, 3 * segments + i, 3 * segments + j, 2 * segments + j),
            (segments + i, segments + j, 3 * segments + j, 3 * segments + i),
            (i, 2 * segments + i, 2 * segments + j, j),
        ]:
            f.extend([(a, b, c), (a, c, d)])
    return np.asarray(p, dtype=np.float32), np.asarray(f, dtype=np.int32)
