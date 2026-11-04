# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Apply a recorded, down-only contact cap before any preparation or capture.

Cloth uses membrane coefficients and rods use their stretch/shear slots.
Tet coefficients require an explicit numeric comparison or an E*length proxy.
Neither policy establishes that a grasp succeeds; that needs a simulation.
"""

from __future__ import annotations

import hashlib
import itertools
import math

import numpy as np


def describe_policy(policy):
    """Return a stable policy identity; None preserves the authored setup."""
    if policy is None:
        return {"name": "authored", "version": 1}
    ratio = float(policy["max_ratio"])
    tet_policy = policy.get("tet_policy", "numeric")
    if not math.isfinite(ratio) or not 0 < ratio <= 10:
        raise ValueError("Contact max ratio must be finite, positive, and at most 10")
    if tet_policy not in ("numeric", "mesh"):
        raise ValueError("Tet policy must be numeric or mesh")
    return {"name": "elasticity-cap", "version": 1, "max_ratio": ratio, "tet_policy": tet_policy}


def _values(model, field):
    value = getattr(model, field, None)
    return np.empty(0) if value is None else np.asarray(value.numpy())


def _positive_min(values):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Elastic coefficients must be finite and nonnegative")
    positive = values[values > 0]
    if not positive.size:
        raise ValueError("No positive elastic coefficient for this material")
    return float(positive.min())


def _references(model, tet_policy):
    terms = []
    tri = _values(model, "tri_materials")
    if tri.size and (not np.all(np.isfinite(tri[:, :2])) or np.any(tri[:, :2] < 0)):
        raise ValueError("Membrane coefficients must be finite and nonnegative")
    # Tet boundary triangles carry zero membrane coefficients, not cloth energy.
    if tri.size and np.any(tri[:, :2] > 0):
        terms.append(
            {
                "material": "cloth",
                "source": "tri_materials[:, :2]",
                "units": "N/m",
                "value": _positive_min(tri[:, :2]),
                "physical_units_match": True,
            }
        )
    tet = _values(model, "tet_materials")
    if tet.size:
        mu, lam = tet[:, 0].astype(float), tet[:, 1].astype(float)
        _positive_min(np.column_stack((mu, lam)))
        if tet_policy == "numeric":
            terms.append(
                {
                    "material": "tet",
                    "source": "tet_materials[:, :2]",
                    "units": "Pa",
                    "value": _positive_min(tet[:, :2]),
                    "physical_units_match": False,
                    "meaning": "Numerical coefficient comparison only; Pa and N/m are different units.",
                }
            )
        else:
            if np.any(mu <= 0) or np.any(mu + lam <= 0):
                raise ValueError("Mesh stiffness proxy needs positive shear modulus and lambda+mu")
            young = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
            indices = _values(model, "tet_indices").reshape(-1, 4)
            edges = np.concatenate([indices[:, [i, j]] for i, j in itertools.combinations(range(4), 2)])
            edges = np.unique(np.sort(edges, axis=1), axis=0)
            q = _values(model, "particle_q")
            lengths = np.linalg.norm(q[edges[:, 0]] - q[edges[:, 1]], axis=1)
            if not lengths.size or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
                raise ValueError("Mesh stiffness proxy needs finite, positive tet edge lengths")
            length = float(np.median(lengths))
            terms.append(
                {
                    "material": "tet",
                    "source": "min(E) * median(unique rest tet edge length)",
                    "units": "N/m",
                    "value": float(young.min()) * length,
                    "young_modulus_min_Pa": float(young.min()),
                    "mesh_length_m": length,
                    "physical_units_match": True,
                    "meaning": "A declared dimensional proxy, not an assembled nodal stiffness or grip bound.",
                }
            )
    joint_type = _values(model, "joint_type")
    if joint_type.size:
        import newton  # noqa: PLC0415 -- policy inspection must not initialize Newton

        rods = np.flatnonzero(joint_type == int(newton.JointType.ROD))
        if rods.size:
            starts = _values(model, "joint_qd_start").astype(int)
            ke = _values(model, "joint_target_ke")
            linear = np.concatenate([ke[starts[j] : starts[j] + 2] for j in rods])
            terms.append(
                {
                    "material": "rod",
                    "source": "ROD joint_target_ke stretch/shear slots",
                    "units": "N/m",
                    "value": _positive_min(linear),
                    "physical_units_match": True,
                }
            )
    if not terms:
        raise ValueError("Contact cap requires a cloth, tet, or rod elastic reference")
    return terms


def _material_summary(model):
    shapes = _values(model, "shape_material_ke")
    return {
        "soft_contact_ke": float(model.soft_contact_ke),
        "shape_count": int(shapes.size),
        "shape_material_ke_min": float(shapes.min()) if shapes.size else None,
        "shape_material_ke_max": float(shapes.max()) if shapes.size else None,
        "shape_material_ke_values": np.unique(shapes).tolist(),
        "shape_material_ke_sha256": hashlib.sha256(shapes.tobytes()).hexdigest(),
    }


def apply_contact_policy(model, policy):
    """Cap both contact materials once, before building a solver on this model."""
    identity = describe_policy(policy)
    previous = getattr(model, "collision_benchmark_contact_policy", None)
    if previous is not None:
        if previous["policy"] != identity:
            raise ValueError("Do not switch contact policy on a prepared model; build a fresh scene")
        if _material_summary(model) != previous["effective"]:
            raise ValueError("Contact materials changed after the policy was applied")
        return previous
    authored = _material_summary(model)
    report = {
        "policy": identity,
        "authored": authored,
        "scope": "This material policy changes only soft_contact_ke and shape_material_ke. Scene-specific pose corrections are outside this policy and are recorded by the scene adapter.",
        "mixing": "rigid-soft: arithmetic mean of soft and shape ke; rigid-rigid: mean of the two shape ke values",
        "grip_validation": "not established by coefficient bounds",
    }
    if policy is not None:
        terms = _references(model, identity["tet_policy"])
        reference = min(term["value"] for term in terms)
        cap = identity["max_ratio"] * reference
        shapes = _values(model, "shape_material_ke")
        if (
            not math.isfinite(model.soft_contact_ke)
            or model.soft_contact_ke < 0
            or not np.all(np.isfinite(shapes))
            or np.any(shapes < 0)
        ):
            raise ValueError("Contact coefficients must be finite and nonnegative")
        # Use a float32 value no greater than the double precision cap.
        array_cap = np.float32(cap)
        if float(array_cap) > cap:
            array_cap = np.nextafter(array_cap, np.float32(0))
        model.soft_contact_ke = min(float(model.soft_contact_ke), float(array_cap))
        if shapes.size:
            model.shape_material_ke.assign(np.minimum(shapes, array_cap))
        report.update(
            reference_terms=terms,
            reference_coefficient=reference,
            cap_N_per_m=cap,
            dimensionally_comparable=all(term["physical_units_match"] for term in terms),
            mixed_material_rule="Use the smallest declared reference; one global soft-contact coefficient.",
        )
    report["effective"] = _material_summary(model)
    report["changed"] = report["authored"] != report["effective"]
    model.collision_benchmark_contact_policy = report
    return report
