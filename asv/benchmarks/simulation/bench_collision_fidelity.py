# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Full collision-fidelity cases; each cache runs exactly one scene.

Select a case by its class name. Full holds last 60 simulated seconds and can
take hours of wall time. No physics backend is imported during discovery.
Unfiltered ASV runs include these cases; selecting a class limits the workload.
ASV caches are local to its benchmark execution; no persistent prepared-state
cache is read. The JSON result records the installed source and environment.
"""

import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from collision_fidelity.catalog import load_cases
from collision_fidelity.version import semantic_version as _version

_CASES = load_cases()


def _run_case(case_id):
    """Check support and CUDA inside the unparameterized case cache itself."""
    support = _CASES[case_id].get("support", {})
    if support.get("status") == "unsupported":
        return {"execution": {"status": "unsupported", "reason": support.get("reason", "Unsupported scene")}}
    import warp as wp  # noqa: PLC0415

    if not wp.is_cuda_available():
        return {"execution": {"status": "unsupported", "reason": "Full collision-fidelity ASV cases require CUDA"}}
    from collision_fidelity.runner import run_case  # noqa: PLC0415

    return run_case(case_id, profile="full", device="cuda:0")


def _value(result, section, key, multiplier=1.0):
    value = result.get("measurements", {}).get(section, {}).get(key)
    return float(value) * multiplier if isinstance(value, (int, float)) and math.isfinite(value) else math.nan


def _covered_windows(result):
    stability = result.get("measurements", {}).get("stability", {})
    declared = stability.get("declared_complete_window_count")
    return isinstance(declared, int) and declared > 0 and stability.get("complete_window_count") == declared


def _finite_complete(result):
    return (
        result.get("execution", {}).get("full_timeline_completed") is True
        and result.get("measurements", {}).get("finite", {}).get("all_samples_finite") is True
    )


class _FidelityTracks:
    """Share one cached scene result across measurements and coverage tracks."""

    def setup(self, result):
        """Surface execution errors while retaining measured numerical failures."""
        execution = result.get("execution", {})
        status = execution.get("status")
        if status == "unsupported":
            raise NotImplementedError(execution.get("reason", "Unsupported scene"))
        if status in ("error", "invalid"):
            message = f"Collision-fidelity execution {status}: {execution.get('reason', 'No reason recorded')}"
            if execution.get("traceback"):
                message += f"\n{execution['traceback']}"
            raise RuntimeError(message)

    def track_jitter(self, result):
        """Report worst-point, worst-window Jitter only with full declared coverage."""
        return _value(result, "stability", "jitter_m", 1000.0) if _covered_windows(result) else math.nan

    track_jitter.unit = "mm"

    def track_rms_speed(self, result):
        """Report worst-point, worst-window RMS speed with mean velocity retained."""
        return _value(result, "stability", "rms_speed_m_s") if _covered_windows(result) else math.nan

    track_rms_speed.unit = "m/s"

    def track_rigid_soft_penetration_fixed_supports(self, result):
        """Report sampled particle-shell penetration over a complete finite run."""
        return (
            _value(result, "penetration", "rigid_soft_penetration_m", 1000.0) if _finite_complete(result) else math.nan
        )

    track_rigid_soft_penetration_fixed_supports.unit = "mm"

    def track_complete_stationary_windows(self, result):
        """Expose the count of valid complete windows, including failed-run prefixes."""
        return _value(result, "stability", "complete_window_count")

    track_complete_stationary_windows.unit = "windows"

    def track_declared_stationary_windows(self, result):
        """Expose the required window count rather than hiding unobserved intervals."""
        return _value(result, "stability", "declared_complete_window_count")

    track_declared_stationary_windows.unit = "windows"

    def track_finite_completion(self, result):
        """Report one only for a complete finite scheduled timeline, not task success."""
        return int(_finite_complete(result))

    track_finite_completion.unit = "completed=1"

    def track_overall(self, result):
        """Encode confirmed Fail as zero, Pass as one, and missing evidence as NaN."""
        return {"fail": 0.0, "pass": 1.0}.get(result.get("criteria", {}).get("overall", {}).get("status"), math.nan)

    track_overall.unit = "pass=1, fail=0"


# Each explicit setup_cache definition has a different source line. ASV uses
# that source line as its cache key; inherited or generated closures would share
# a cache and could run unrelated scenes when only one case was selected.


class CollisionFidelityR5FrankaPatchContactV2(_FidelityTracks):
    """Full protocol for ``r5_franka_patch_contact_v2``."""

    case_id = "r5_franka_patch_contact_v2"
    version = _version(case_id)

    def setup_cache(self):
        """Run only r5_franka_patch_contact_v2 and share its result across tracks."""
        return _run_case("r5_franka_patch_contact_v2")

    setup_cache.timeout = 3600


class CollisionFidelityR7H1Towel(_FidelityTracks):
    """Full protocol for ``r7_h1_towel``."""

    case_id = "r7_h1_towel"
    version = _version(case_id)

    def setup_cache(self):
        """Run only r7_h1_towel and share its result across tracks."""
        return _run_case("r7_h1_towel")

    setup_cache.timeout = 3600


class CollisionFidelityR7H1Sheet(_FidelityTracks):
    """Full protocol for ``r7_h1_sheet``."""

    case_id = "r7_h1_sheet"
    version = _version(case_id)

    def setup_cache(self):
        """Run only r7_h1_sheet and share its result across tracks."""
        return _run_case("r7_h1_sheet")

    setup_cache.timeout = 3600


class CollisionFidelityR5FrankaTrousers(_FidelityTracks):
    """Full protocol for ``r5_franka_trousers``."""

    case_id = "r5_franka_trousers"
    version = _version(case_id)

    def setup_cache(self):
        """Run only r5_franka_trousers and share its result across tracks."""
        return _run_case("r5_franka_trousers")

    setup_cache.timeout = 3600


class CollisionFidelityR5FrankaDress(_FidelityTracks):
    """Full protocol for ``r5_franka_dress``."""

    case_id = "r5_franka_dress"
    version = _version(case_id)

    def setup_cache(self):
        """Run only r5_franka_dress and share its result across tracks."""
        return _run_case("r5_franka_dress")

    setup_cache.timeout = 3600


class CollisionFidelityR5FrankaTshirtContactV2(_FidelityTracks):
    """Full protocol for ``r5_franka_tshirt_contact_v2``."""

    case_id = "r5_franka_tshirt_contact_v2"
    version = _version(case_id)

    def setup_cache(self):
        """Run only r5_franka_tshirt_contact_v2 and share its result across tracks."""
        return _run_case("r5_franka_tshirt_contact_v2")

    setup_cache.timeout = 3600


class CollisionFidelityR5FrankaSkirtContactV2(_FidelityTracks):
    """Full protocol for ``r5_franka_skirt_contact_v2``."""

    case_id = "r5_franka_skirt_contact_v2"
    version = _version(case_id)

    def setup_cache(self):
        """Run only r5_franka_skirt_contact_v2 and share its result across tracks."""
        return _run_case("r5_franka_skirt_contact_v2")

    setup_cache.timeout = 3600


class CollisionFidelityR5FrankaToteContactV2(_FidelityTracks):
    """Full protocol for ``r5_franka_tote_contact_v2``."""

    case_id = "r5_franka_tote_contact_v2"
    version = _version(case_id)

    def setup_cache(self):
        """Run only r5_franka_tote_contact_v2 and share its result across tracks."""
        return _run_case("r5_franka_tote_contact_v2")

    setup_cache.timeout = 3600


class CollisionFidelityR5FrankaPouchContactV2(_FidelityTracks):
    """Full protocol for ``r5_franka_pouch_contact_v2``."""

    case_id = "r5_franka_pouch_contact_v2"
    version = _version(case_id)

    def setup_cache(self):
        """Run only r5_franka_pouch_contact_v2 and share its result across tracks."""
        return _run_case("r5_franka_pouch_contact_v2")

    setup_cache.timeout = 3600


class CollisionFidelityR7H1GussetBag(_FidelityTracks):
    """Full protocol for ``r7_h1_gusset_bag``."""

    case_id = "r7_h1_gusset_bag"
    version = _version(case_id)

    def setup_cache(self):
        """Run only r7_h1_gusset_bag and share its result across tracks."""
        return _run_case("r7_h1_gusset_bag")

    setup_cache.timeout = 3600


class CollisionFidelityJ3H1Jacket(_FidelityTracks):
    """Full protocol for ``j3_h1_jacket``."""

    case_id = "j3_h1_jacket"
    version = _version(case_id)

    def setup_cache(self):
        """Run only j3_h1_jacket and share its result across tracks."""
        return _run_case("j3_h1_jacket")

    setup_cache.timeout = 21600


class CollisionFidelityR6FrankaPad(_FidelityTracks):
    """Full protocol for ``r6_franka_pad``."""

    case_id = "r6_franka_pad"
    version = _version(case_id)

    def setup_cache(self):
        """Run only r6_franka_pad and share its result across tracks."""
        return _run_case("r6_franka_pad")

    setup_cache.timeout = 3600


class CollisionFidelityR6AllegroBlock(_FidelityTracks):
    """Full protocol for ``r6_allegro_block``."""

    case_id = "r6_allegro_block"
    version = _version(case_id)

    def setup_cache(self):
        """Run only r6_allegro_block and share its result across tracks."""
        return _run_case("r6_allegro_block")

    setup_cache.timeout = 3600


class CollisionFidelityR6AllegroSphere(_FidelityTracks):
    """Full protocol for ``r6_allegro_sphere``."""

    case_id = "r6_allegro_sphere"
    version = _version(case_id)

    def setup_cache(self):
        """Run only r6_allegro_sphere and share its result across tracks."""
        return _run_case("r6_allegro_sphere")

    setup_cache.timeout = 3600


class CollisionFidelityR6FrankaCapsule(_FidelityTracks):
    """Full protocol for ``r6_franka_capsule``."""

    case_id = "r6_franka_capsule"
    version = _version(case_id)

    def setup_cache(self):
        """Run only r6_franka_capsule and share its result across tracks."""
        return _run_case("r6_franka_capsule")

    setup_cache.timeout = 3600


class CollisionFidelityR6FrankaTorus(_FidelityTracks):
    """Full protocol for ``r6_franka_torus``."""

    case_id = "r6_franka_torus"
    version = _version(case_id)

    def setup_cache(self):
        """Run only r6_franka_torus and share its result across tracks."""
        return _run_case("r6_franka_torus")

    setup_cache.timeout = 3600


class CollisionFidelityR6FrankaBranched(_FidelityTracks):
    """Full protocol for ``r6_franka_branched``."""

    case_id = "r6_franka_branched"
    version = _version(case_id)

    def setup_cache(self):
        """Run only r6_franka_branched and share its result across tracks."""
        return _run_case("r6_franka_branched")

    setup_cache.timeout = 3600


class CollisionFidelityM3FrankaToteEmpty(_FidelityTracks):
    """Full protocol for ``m3_franka_tote_empty``."""

    case_id = "m3_franka_tote_empty"
    version = _version(case_id)

    def setup_cache(self):
        """Run only m3_franka_tote_empty and share its result across tracks."""
        return _run_case("m3_franka_tote_empty")

    setup_cache.timeout = 3600


class CollisionFidelityM3FrankaTote2(_FidelityTracks):
    """Full protocol for ``m3_franka_tote_2``."""

    case_id = "m3_franka_tote_2"
    version = _version(case_id)

    def setup_cache(self):
        """Run only m3_franka_tote_2 and share its result across tracks."""
        return _run_case("m3_franka_tote_2")

    setup_cache.timeout = 3600


class CollisionFidelityM3FrankaTote2Spheres(_FidelityTracks):
    """Full protocol for ``m3_franka_tote_2_spheres``."""

    case_id = "m3_franka_tote_2_spheres"
    version = _version(case_id)

    def setup_cache(self):
        """Run only m3_franka_tote_2_spheres and share its result across tracks."""
        return _run_case("m3_franka_tote_2_spheres")

    setup_cache.timeout = 3600


class CollisionFidelityM3FrankaGusset4(_FidelityTracks):
    """Full protocol for ``m3_franka_gusset_4``."""

    case_id = "m3_franka_gusset_4"
    version = _version(case_id)

    def setup_cache(self):
        """Run only m3_franka_gusset_4 and share its result across tracks."""
        return _run_case("m3_franka_gusset_4")

    setup_cache.timeout = 3600


class CollisionFidelityM2FrankaPouch(_FidelityTracks):
    """Full protocol for ``m2_franka_pouch``."""

    case_id = "m2_franka_pouch"
    version = _version(case_id)

    def setup_cache(self):
        """Run only m2_franka_pouch and share its result across tracks."""
        return _run_case("m2_franka_pouch")

    setup_cache.timeout = 3600


class CollisionFidelityC6CrossedCables10substeps(_FidelityTracks):
    """Full protocol for ``c6_crossed_cables_10substeps``."""

    case_id = "c6_crossed_cables_10substeps"
    version = _version(case_id)

    def setup_cache(self):
        """Run only c6_crossed_cables_10substeps and share its result across tracks."""
        return _run_case("c6_crossed_cables_10substeps")

    setup_cache.timeout = 3600


class CollisionFidelityC7CylinderWrap10substeps(_FidelityTracks):
    """Full protocol for ``c7_cylinder_wrap_10substeps``."""

    case_id = "c7_cylinder_wrap_10substeps"
    version = _version(case_id)

    def setup_cache(self):
        """Run only c7_cylinder_wrap_10substeps and share its result across tracks."""
        return _run_case("c7_cylinder_wrap_10substeps")

    setup_cache.timeout = 3600


class CollisionFidelityC9BundleSqueezeWithdraw(_FidelityTracks):
    """Full protocol for ``c9_bundle_squeeze_withdraw``."""

    case_id = "c9_bundle_squeeze_withdraw"
    version = _version(case_id)

    def setup_cache(self):
        """Run only c9_bundle_squeeze_withdraw and share its result across tracks."""
        return _run_case("c9_bundle_squeeze_withdraw")

    setup_cache.timeout = 3600


class CollisionFidelityC9PilePickup(_FidelityTracks):
    """Full protocol for ``c9_pile_pickup``."""

    case_id = "c9_pile_pickup"
    version = _version(case_id)

    def setup_cache(self):
        """Run only c9_pile_pickup and share its result across tracks."""
        return _run_case("c9_pile_pickup")

    setup_cache.timeout = 3600


class CollisionFidelityS1Cloth210substeps(_FidelityTracks):
    """Full protocol for ``s1_cloth_2_10substeps``."""

    case_id = "s1_cloth_2_10substeps"
    version = _version(case_id)

    def setup_cache(self):
        """Run only s1_cloth_2_10substeps and share its result across tracks."""
        return _run_case("s1_cloth_2_10substeps")

    setup_cache.timeout = 3600


class CollisionFidelityS1Cloth510substeps(_FidelityTracks):
    """Full protocol for ``s1_cloth_5_10substeps``."""

    case_id = "s1_cloth_5_10substeps"
    version = _version(case_id)

    def setup_cache(self):
        """Run only s1_cloth_5_10substeps and share its result across tracks."""
        return _run_case("s1_cloth_5_10substeps")

    setup_cache.timeout = 3600


class CollisionFidelityS1Cloth1010substeps(_FidelityTracks):
    """Full protocol for ``s1_cloth_10_10substeps``."""

    case_id = "s1_cloth_10_10substeps"
    version = _version(case_id)

    def setup_cache(self):
        """Run only s1_cloth_10_10substeps and share its result across tracks."""
        return _run_case("s1_cloth_10_10substeps")

    setup_cache.timeout = 3600


class CollisionFidelityS1Pads210substeps(_FidelityTracks):
    """Full protocol for ``s1_pads_2_10substeps``."""

    case_id = "s1_pads_2_10substeps"
    version = _version(case_id)

    def setup_cache(self):
        """Run only s1_pads_2_10substeps and share its result across tracks."""
        return _run_case("s1_pads_2_10substeps")

    setup_cache.timeout = 3600


class CollisionFidelityS1Pads510substeps(_FidelityTracks):
    """Full protocol for ``s1_pads_5_10substeps``."""

    case_id = "s1_pads_5_10substeps"
    version = _version(case_id)

    def setup_cache(self):
        """Run only s1_pads_5_10substeps and share its result across tracks."""
        return _run_case("s1_pads_5_10substeps")

    setup_cache.timeout = 3600


class CollisionFidelityS2Impact(_FidelityTracks):
    """Full protocol for ``s2_impact``."""

    case_id = "s2_impact"
    version = _version(case_id)

    def setup_cache(self):
        """Run only s2_impact and share its result across tracks."""
        return _run_case("s2_impact")

    setup_cache.timeout = 3600


class CollisionFidelityS2Lower(_FidelityTracks):
    """Full protocol for ``s2_lower``."""

    case_id = "s2_lower"
    version = _version(case_id)

    def setup_cache(self):
        """Run only s2_lower and share its result across tracks."""
        return _run_case("s2_lower")

    setup_cache.timeout = 3600


class CollisionFidelityS3Cloth10substeps(_FidelityTracks):
    """Full protocol for ``s3_cloth_10substeps``."""

    case_id = "s3_cloth_10substeps"
    version = _version(case_id)

    def setup_cache(self):
        """Run only s3_cloth_10substeps and share its result across tracks."""
        return _run_case("s3_cloth_10substeps")

    setup_cache.timeout = 3600


class CollisionFidelityS3Pads10substeps(_FidelityTracks):
    """Full protocol for ``s3_pads_10substeps``."""

    case_id = "s3_pads_10substeps"
    version = _version(case_id)

    def setup_cache(self):
        """Run only s3_pads_10substeps and share its result across tracks."""
        return _run_case("s3_pads_10substeps")

    setup_cache.timeout = 3600


class CollisionFidelityS4Pads(_FidelityTracks):
    """Full protocol for ``s4_pads``."""

    case_id = "s4_pads"
    version = _version(case_id)

    def setup_cache(self):
        """Run only s4_pads and share its result across tracks."""
        return _run_case("s4_pads")

    setup_cache.timeout = 3600


class CollisionFidelityS5Pack4(_FidelityTracks):
    """Full protocol for ``s5_pack_4``."""

    case_id = "s5_pack_4"
    version = _version(case_id)

    def setup_cache(self):
        """Run only s5_pack_4 and share its result across tracks."""
        return _run_case("s5_pack_4")

    setup_cache.timeout = 3600


class CollisionFidelityS5Pack8(_FidelityTracks):
    """Full protocol for ``s5_pack_8``."""

    case_id = "s5_pack_8"
    version = _version(case_id)

    def setup_cache(self):
        """Run only s5_pack_8 and share its result across tracks."""
        return _run_case("s5_pack_8")

    setup_cache.timeout = 3600


class CollisionFidelityR5FrankaTowelContactV2(_FidelityTracks):
    """Full protocol for ``r5_franka_towel_contact_v2``."""

    case_id = "r5_franka_towel_contact_v2"
    version = _version(case_id)

    def setup_cache(self):
        """Run only r5_franka_towel_contact_v2 and share its result across tracks."""
        return _run_case("r5_franka_towel_contact_v2")

    setup_cache.timeout = 3600


class CollisionFidelityR2FrankaDuck(_FidelityTracks):
    """Full protocol for ``r2_franka_duck``."""

    case_id = "r2_franka_duck"
    version = _version(case_id)

    def setup_cache(self):
        """Run only r2_franka_duck and share its result across tracks."""
        return _run_case("r2_franka_duck")

    setup_cache.timeout = 3600


class CollisionFidelityM2FrankaCable(_FidelityTracks):
    """Full protocol for ``m2_franka_cable``."""

    case_id = "m2_franka_cable"
    version = _version(case_id)

    def setup_cache(self):
        """Run only m2_franka_cable and share its result across tracks."""
        return _run_case("m2_franka_cable")

    setup_cache.timeout = 3600


class CollisionFidelityR5AllegroTowel(_FidelityTracks):
    """Full protocol for ``r5_allegro_towel``."""

    case_id = "r5_allegro_towel"
    version = _version(case_id)

    def setup_cache(self):
        """Run only r5_allegro_towel and share its result across tracks."""
        return _run_case("r5_allegro_towel")

    setup_cache.timeout = 3600


class CollisionFidelityR3AllegroGroundSphere(_FidelityTracks):
    """Full protocol for ``r3_allegro_ground_sphere``."""

    case_id = "r3_allegro_ground_sphere"
    version = _version(case_id)

    def setup_cache(self):
        """Run only r3_allegro_ground_sphere and share its result across tracks."""
        return _run_case("r3_allegro_ground_sphere")

    setup_cache.timeout = 3600


class CollisionFidelityM2AllegroCable(_FidelityTracks):
    """Full protocol for ``m2_allegro_cable``."""

    case_id = "m2_allegro_cable"
    version = _version(case_id)

    def setup_cache(self):
        """Run only m2_allegro_cable and share its result across tracks."""
        return _run_case("m2_allegro_cable")

    setup_cache.timeout = 3600


class CollisionFidelityS4PokerCardsNoBall(_FidelityTracks):
    """Full protocol for ``s4_poker_cards_no_ball``."""

    case_id = "s4_poker_cards_no_ball"
    version = _version(case_id)

    def setup_cache(self):
        """Run only s4_poker_cards_no_ball and share its result across tracks."""
        return _run_case("s4_poker_cards_no_ball")

    setup_cache.timeout = 3600
