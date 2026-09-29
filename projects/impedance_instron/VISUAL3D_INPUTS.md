# Visual3D inputs

Visual3D exports are a separate source format from raw C3D. Keep your filtering,
gap filling, model construction and force calibration in Visual3D, then export
full recordings without gait-cycle normalization. The Newton adapter must retain
source clocks and processing provenance rather than applying the raw-C3D
preparation assumptions to already processed data.

## Inspect and import

Run from the repository root:

```bash
uv run --no-sync -m projects.impedance_instron visual3d inspect data/F01
uv run --no-sync -m projects.impedance_instron visual3d inspect data/F01/FR3_1 \
  --output /tmp/fr3_1-audit.json
uv run --no-sync -m projects.impedance_instron visual3d manifest-template \
  data/F01/FR3_1/visual3d_manifest.json
```

Fill in the manifest using verified lab conventions. For the updated script,
length units are `m`, force units `N`, moment units `N*m`, and time units `s`.
Set `up_axis` and `forward_axis` explicitly (signed `X`, `Y`, or `Z`). Leave
rates `null` to infer them from the exported clocks, or supply rates to check
against those clocks. The importer requires explicit, uniform TIME and
ANALOGTIME exports in seconds and preserves their relative origin.

Once the manifest and dynamic exports are complete:

```bash
uv run --no-sync -m projects.impedance_instron visual3d normalize data/F01/FR3_1 \
  --output /tmp/fr3_1-measurements.npz
```

This writes SI measurements, marker names/validity masks, separate point/force
clocks and SHA-256 provenance. It does not perform gait-cycle normalization,
filtering, or controller fitting. The normalized measurements are an ingestion
artifact, not the Cartesian solver's `reference.npz`. Existing output files and
manifest templates are not overwritten. The audit reports missing files and
sample counts without granting simulation readiness.

## F01 exports

The supplied `data/F01/FR3_1`, `FR3_2` and `FR3_3` directories now contain
dynamic and static targets, explicit point and analog clocks, model joint
centers and angles, FP1 force, COP, and free moment. All three normalize with
the current importer. The exported model mass is `70.0` kg in each trial; this
is the model metric and still needs confirmation against the subject record.
The static heel-to-metatarsal and heel-to-toe vectors point along laboratory
`+Y`, so the F01 manifests declare `forward_axis: "+Y"`.

For a provisional controller run, record transferred inertial assumptions
separately from the measured exports. Carry over the fixed shoe artifact and
foot-to-shoe registration from the selected baseline; do not tune the shoe
placement against a trial. Select a single stance and anchor treadmill
translation at that window's first point sample; a translation from the start
of the full recording changes the controller position bounds with the stride's
recording time.

Use the updated `data/F01/FullBuild.v3s`, verify the model in Visual3D, then run
`data/F01/Export_FR3_ImpedanceInstron.v3s` into a **fresh export directory**.
The Windows `CAL_FOLDER` and `EXPORT_FOLDER` parameters must match your machine.
The export script opens each standing recording as motion data as well as using
its calibration model, and exports:

- `static_all_targets.txt` and `motion_all_targets.txt`: original XYZ targets.
- `motion_processed_targets.txt`: your final `TARGET/PROCESSED` signals.
- `static_time.txt` and `motion_time.txt`: explicit point clocks in seconds.
- `static_analog_time.txt` and `motion_analog_time.txt`: native analog clocks.
- `static_joint_centers.txt` and `motion_joint_centers.txt`: `LHIP`, `LKNEE`,
  `LANKLE`, `RHIP`, `RKNEE`, `RANKLE` in laboratory coordinates.
- `static_joint_angles.txt` and `motion_joint_angles.txt`: 3D Cardan joint angles
  (`RHipAngle`, `RKneeAngle`, `RAnkleAngle`, `RVirtualFootAngle`, `L...`) in degrees.
- `FORCE`, `COFP`, `FREEMOMENT`: separate XYZ signals at their native rate.
- `model_mass.txt`: the model mass metric, which still needs verification.

### Foot and Ankle Handling with Visual3D

Following the [HAS-Motion Foot and Ankle Angles Tutorial](https://has-motion.com/wiki/doku.php?id=visual3d:tutorials:kinematics_and_kinetics:foot_and_ankle_angles&s[]=ankle),
standard kinetic foot segments (`RFT`/`LFT`) define the ankle joint from the malleoli to the
distal toe marker. Because the ankle joint center is elevated above the floor compared to the
toe marker, the anatomical foot axis slopes downward, causing raw anatomical ankle angles
(`RAnkleAngle`/`LAnkleAngle`) to have a nonzero offset in standing; its size depends on the segment definition.

The name “Virtual Foot” does not identify a ground-referenced angle. The
[HAS-Motion tutorial](https://has-motion.com/wiki/doku.php?id=visual3d:tutorials:kinematics_and_kinetics:foot_and_ankle_angles)
describes several definitions of neutral. In the supplied `FullBuild.v3s`,
`RVirualFootAngle` is a `JOINT_ANGLE` between `Right_Virtual_Foot` and `RSK`;
the left counterpart uses `LSK`. That source therefore describes a relative
angle. The corrected spelling is also accepted in the exported ASCII headers.

`prepare-visual3d` supports `--virtual-foot-reference reconstructed_ground` (or `ground`),
which reconstructs foot ground pitch by combining the exported 3D Cardan joint rotation
$R_{\text{rel}} = R_\alpha R_\beta R_\gamma$ with the reconstructed shank frame $R_{\text{shank}}$:
$R_{\text{foot}} = R_{\text{shank}} @ R_{\text{rel}}$. The forward axis $[1, 0, 0]$ is transported
by $R_{\text{foot}}$ and projected onto the sagittal simulation plane:
$\theta_{\text{ground}} = \text{arctan2}(v_{\text{fwd}, z}, v_{\text{fwd}, x})$.
This properly models forward shank inclination during mid-to-late stance and avoids the erroneous
toe-up artifact (+24.3° at 0.370 s) caused by treating raw relative ankle angle $\alpha$ directly as
ground pitch. Reconstructed ground pitch agrees with physical marker pitch (heel-to-MTH) with
a mean absolute error of 1.67° across the entire stance window.

For reconstructed ground pitch $g$, the solver's internal ankle coordinate is:
`q_ankle = g + fixed_shoe_pitch - (q_thigh + q_knee) - pi/2`.
The unchanged shoe adapter subtracts the fixed shoe pitch after combining the three limb angles,
so the actual contact carrier and rendered mesh both rotate by $g$. The saved reference retains
the reconstructed ground target separately from the raw Visual3D relative angle and the derived
solver joint coordinate. CPU, resident CUDA, and adjoint objectives score knee angle and ground pitch
directly; a changing shank cannot hide a ground-angle error. The report and plots display all three
angles separately in degrees (shank inclination, relative virtual-foot angle, and shoe pitch to ground),
with positive pitch indicating toe-up and negative indicating toe-down.

When only anatomical ankle angles are present, the preparer subtracts the
matched static baseline. If angle signals are absent, it derives pitch from
the heel cluster and joint centers. Neither fallback is accepted as a declared
exported virtual-foot ground angle.

Force-platform bouts alternate feet in these exports. A trial-level
`force_side` label is not proof of the side of every bout: select the contact
using synchronized COP and bilateral foot markers. Preserve the point and
native analog time origins together when cropping and applying treadmill
translation.

The segment endpoint exports assume `LTH/LSK/LFT` and `RTH/RSK/RFT`, matching the
segment names used in FullBuild. Confirm that their proximal endpoints represent
your intended anatomical joint centers; especially check the foot's proximal
endpoint against the ankle. Do not substitute tracking-cluster markers for
anatomical centers. The scripts cannot be executed in this Linux workspace;
inspect the Visual3D execution log for missing signals and export errors.


FullBuild now explicitly enables processed targets and selects all loaded motion
trials before force processing/filtering. Its analog filter uses **6 Hz**, while
the target filter uses **20 Hz**. The updated comment reflects the configured
analog cutoff; the numerical cutoff and session-specific plate calibration are
unchanged. Verify the platform calibration for this session before re-exporting.

## Physical metadata

ASCII sample indices are not seconds. Supply the explicit clock exports.
Declared sample rates, when provided, are checked against those clocks. Do not separately zero the point and
force clocks: their relative offset must survive ingestion. Retain missing-value
samples for quality screening; never replace missing positions with zero.

Record the subject's measured mass, the laboratory forward/up axes, treadmill
speed and synchronization, segment masses/COM/inertias, and static shoe
registration. These quantities cannot be recovered uniquely from the supplied
dynamic foot markers. A complete export does not itself certify a fitted
controller or calibrate the shoe. Missing physical inputs must prevent a fit
rather than silently reuse another subject's model.

## Export conventions and sources

HAS-Motion documents the [five-line ASCII header and MKS units](https://wiki.has-motion.com/doku.php?id=visual3d:documentation:definitions:file_formats:visual3d_ascii_format),
and [explicit FRAME_NUMBERS time signals](https://www.wiki.has-motion.com/doku.php?id=visual3d:documentation:definitions:file_formats:ascii_format).
The script keeps native sampling and disables normalization using the documented
[ASCII export command](https://wiki.has-motion.com/doku.php?id=visual3d:documentation:pipeline:file_commands:export_data_to_ascii_file).

`ORIGINAL` force, COP, and free-moment folders can contain results calculated
from processed analog data; that choice follows
[Visual3D's processed-signal settings](https://wiki.has-motion.com/doku.php?id=visual3d:documentation:visual3d_signal_types:used_process).
Consequently, the script exports these derived plate signals from `ORIGINAL`
without applying body-weight scaling. The joint positions use the documented
[segment proximal endpoint operation](https://wiki.has-motion.com/doku.php?id=visual3d:documentation:visual3d_signal_types:link_model_based_data_type:seg_proximal_joint)
with both reference and resolution coordinate systems set to the laboratory.
