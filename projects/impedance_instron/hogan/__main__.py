# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Track one measured stance with phase-scheduled impedance across shoe variants.

Example::

    uv run python -m projects.impedance_instron.hogan \\
        --reference reference.npz --profile profile.json \\
        --shoe-artifact shoe.json --mount 0.0 0.0 0.1 \\
        --contact shoe_replay --modulus-scales 0.8 1.0 1.2 \\
        --adaptation fixed series --output runs/hogan
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..cartesian import data as reference_data
from ..cartesian import profile as profile_data
from ..cartesian.gpu.runner_rollout import material_variant
from ..cartesian.shoe import Shoe
from .adaptation import ADAPTATION_MODES, ShoeAdaptation, characterize_shoe, leg_stiffness
from .control import Impedance
from .mechanics import COORDINATE_NAMES, RestOfBody, chain_from_profile
from .plan import CONTACT_SOURCES, build_plan
from .registration import register_static_height
from .rollout import PHASE_MODES, Config, simulate

# Zero hip x/z stiffness: the body is carried only by the shoe ground force.
DEFAULT_STIFFNESS = (0.0, 0.0, 500.0, 500.0, 300.0, 300.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--shoe-artifact", type=Path, required=True)
    parser.add_argument("--mount", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pitch", type=float, help="Static shoe pitch [rad]; defaults to the reference value")
    parser.add_argument("--contact", choices=CONTACT_SOURCES, default="measured")
    parser.add_argument("--stiffness", type=float, nargs=6, default=DEFAULT_STIFFNESS, metavar="K")
    parser.add_argument("--damping-ratio", type=float, default=1.0)
    parser.add_argument("--phase", choices=PHASE_MODES, default="touchdown")
    parser.add_argument("--dt", type=float, default=1.25e-4)
    parser.add_argument(
        "--friction-model",
        choices=("elastic_coulomb", "column_maxwell", "maxwell", "legacy"),
        default="elastic_coulomb",
    )
    parser.add_argument("--rest-com", type=float, nargs=2, default=RestOfBody.com_local_m, metavar=("X", "Y"))
    parser.add_argument("--rest-gyration", type=float, default=RestOfBody.radius_of_gyration_m)
    parser.add_argument("--no-residual-feedforward", action="store_true")
    parser.add_argument("--modulus-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--adaptation", choices=ADAPTATION_MODES[:2], nargs="+", default=["fixed"])
    parser.add_argument(
        "--static-load-fraction",
        type=float,
        default=0.5,
        help="Fraction of body weight on this foot in the static trial, used for registration",
    )
    parser.add_argument("--height-offset", type=float, help="Override the static registration offset [m]")
    parser.add_argument("--cutoff", type=float, default=12.0, help="Kinematic low-pass cutoff [Hz]; 0 disables")
    parser.add_argument(
        "--hip-source", choices=("markers", "grf_com"), help="Hip reference; defaults to grf_com for measured contact"
    )
    parser.add_argument(
        "--pelvis-cutoff", type=float, default=4.0, help="Pelvis tilt low-pass cutoff [Hz]; 0 keeps the raw tilt"
    )
    parser.add_argument(
        "--no-leg-ik", action="store_true", help="Use forward kinematics instead of pinning the measured ankle"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    reference = reference_data.load(args.reference)
    profile = profile_data.load(args.profile)
    if args.pitch is not None:
        pitch = args.pitch
    else:
        pitch = float(reference.get("shoe_static_pitch_rad", reference["static_pitch_rad"]))
    chain = chain_from_profile(reference, profile, RestOfBody(tuple(args.rest_com), args.rest_gyration))
    args.output.mkdir(parents=True, exist_ok=True)

    def make_shoe(path: Path) -> Shoe:
        return Shoe(path, args.mount, pitch, friction_model=args.friction_model)

    reference_shoe = make_shoe(args.shoe_artifact)
    static_load = args.static_load_fraction * float(reference["subject_mass_kg"]) * 9.81
    registration = register_static_height(reference, reference_shoe, static_load_n=static_load).to_dict()
    if args.height_offset is not None:
        registration["override_m"] = args.height_offset
    offset = registration.get("override_m", registration["height_offset_m"])
    plan = build_plan(
        reference,
        chain,
        dt_s=args.dt,
        contact=args.contact,
        shoe=reference_shoe,
        height_offset_m=offset,
        cutoff_hz=args.cutoff or None,
        hip_source=args.hip_source,
        pelvis_cutoff_hz=args.pelvis_cutoff or None,
        leg_ik=not args.no_leg_ik,
    )
    mass, _ = chain.dynamics(plan.q[0], plan.v[0])
    impedance = Impedance.critically_damped(
        args.stiffness, np.diag(mass), plan.duration_s, damping_ratio=args.damping_ratio
    )
    impedance.save(args.output / "impedance.npz", {"coordinates": list(COORDINATE_NAMES)})
    reference_descriptor, _ = characterize_shoe(reference_shoe)
    leg_k = leg_stiffness(plan, chain)
    config = Config(phase=args.phase, residual_feedforward=not args.no_residual_feedforward)

    variants = args.output / "shoes"
    variants.mkdir(exist_ok=True)
    runs = []
    for scale in args.modulus_scales:
        if scale == 1.0:
            shoe = reference_shoe
        else:
            path = material_variant(args.shoe_artifact, variants / f"modulus_{scale:g}.json", scale)
            shoe = make_shoe(path)
        descriptor, curve = characterize_shoe(shoe)
        np.save(variants / f"compression_{scale:g}.npy", curve)
        for mode in args.adaptation:
            adaptation = ShoeAdaptation(mode, reference_descriptor, leg_stiffness_n_m=leg_k)
            stiffness_factor, damping_factor = adaptation.factors(descriptor)
            trace, summary = simulate(
                plan, chain, impedance.scaled(stiffness_factor, damping_factor), shoe, config=config
            )
            name = f"modulus_{scale:g}_{mode}"
            np.savez(args.output / f"{name}.npz", **trace)
            runs.append(
                {
                    "name": name,
                    "modulus_scale": scale,
                    "adaptation": mode,
                    "shoe": descriptor.__dict__,
                    "stiffness_factor": stiffness_factor.tolist(),
                    "damping_factor": damping_factor.tolist(),
                    **summary,
                }
            )

    report = {
        "reference": str(args.reference),
        "profile": str(args.profile),
        "shoe_artifact": str(args.shoe_artifact),
        "shoe_mount_m": list(args.mount),
        "shoe_static_pitch_rad": pitch,
        "friction_model": args.friction_model,
        "rest_of_body": {
            "mass_kg": float(chain.masses_kg[0]),
            "com_local_m": list(args.rest_com),
            "radius_of_gyration_m": args.rest_gyration,
            "provenance": "Provisional estimates unless supplied from subject data",
        },
        "plan": plan.diagnostics,
        "registration": registration,
        "reference_shoe": reference_descriptor.__dict__,
        "leg_stiffness_n_m": leg_k,
        "impedance": {"stiffness": impedance.stiffness[0].tolist(), "damping": impedance.damping[0].tolist()},
        "runs": runs,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    print(
        f"static registration: ankle {registration['static_ankle_height_m']:.4f} m, "
        f"shoe seat {registration['unloaded_ankle_height_m'] - registration['static_compression_m']:.4f} m, "
        f"offset {offset * 1000:.1f} mm"
    )
    print(f"plan residual force RMS [BW]: {plan.diagnostics['residual_force_rms_body_weight']}")
    print(
        f"plan residual moment RMS {plan.diagnostics['residual_moment_rms_nm']:.1f} N m, "
        f"hip adjustment RMS {np.round(np.multiply(plan.diagnostics['hip_adjustment_rms_m'], 1e3), 1)} mm"
    )
    print(f"leg stiffness {leg_k:.0f} N/m, reference shoe {reference_descriptor.stiffness_n_m:.0f} N/m")
    print(f"{'run':<24} {'status':<10} {'peak Fz [N]':>12} {'contact [ms]':>13} {'k_shoe [N/m]':>13} {'K factor':>9}")
    for run in runs:
        peak = run.get("peak_grf_n", [np.nan, np.nan])[1]
        contact = 1000.0 * run.get("contact_duration_s", np.nan)
        print(
            f"{run['name']:<24} {run['status']:<10} {peak:>12.0f} {contact:>13.1f} "
            f"{run['shoe']['stiffness_n_m']:>13.0f} {run['stiffness_factor'][3]:>9.3f}"
        )


if __name__ == "__main__":
    main()
