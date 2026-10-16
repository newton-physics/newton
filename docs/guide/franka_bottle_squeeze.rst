.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Franka bottle squeeze
=====================

This worked example squeezes an open PET bottle from three angles, spills
water through its neck, and retains the dents between grips. It combines GPU VBD
shell mechanics, self-contact, optional VBD bending plasticity, and 32,000
XPBD fluid particles. Newton's :class:`newton.geometry.ParticleSurface`
reconstructs an anisotropic water surface, as in the MPM water examples.

.. image:: /images/examples/example_franka_bottle_squeeze.jpg
   :width: 320
   :alt: A Franka squeezing an open bottle with spilled water on the table.

Run it on a CUDA GPU after installing the example dependencies:

.. code-block:: console

   uv sync --extra examples
   uv run --extra examples -m newton.examples franka_bottle_squeeze

The default run covers 24 simulated seconds with grip yaw angles of 60°, 120°,
and 0°, at heights of 112, 130, and 94 mm above the bottle base.
Each eight-second cycle settles, squeezes, holds, releases and
withdraws. The open gripper rises above the neck before rotating and
approaching the next grip. Plastic rest angles and water state carry across
all three cycles. The high shell stiffness and fluid constraint
iterations make this an offline demonstration rather than a real-time example.
Use ``--squeeze-angles`` to select another sequence. For two grips:

.. code-block:: console

   uv run --extra examples -m newton.examples franka_bottle_squeeze \
       --squeeze-angles 0 60 --num-frames 960

Use ``--squeeze-heights`` to supply one height per angle, in meters above the
bottle base. A single value applies to every grip; for example,
``--squeeze-heights 0.112`` keeps all squeezes at the same height. Without this
option, the three default heights repeat for longer sequences.

The standard ``--num-frames`` option controls the run duration at 60 fps;
use 480 frames per squeeze. After the last cycle, the arm stays withdrawn.
The Cartesian transitions are sampled before simulation to keep a continuous
IK branch. Pose errors, joint limits and maximum interpolation speeds are
checked during initialization; an unreachable sequence raises a clear error.
The 2.4-second turn above the neck keeps the default 120° to 0° transition
within the Franka's joint velocity limits.
Use ``--no-plasticity`` to compare with an elastic bottle under the same motion.
Use ``--show-particles`` to inspect the underlying fluid samples.

CUDA graphs capture the robot and VBD shell substeps by default, including
self-contact and plasticity. The XPBD water loop and surface reconstruction
execute outside capture. Use ``--no-cuda-graph`` to run the shell eagerly as
well. This option does not claim capture of the full simulation or rendering.

The default solve uses 320 VBD iterations per substep. Under-solving this stiff
shell changes the buckling mode and the plastic loading path. Increasing the
iteration count improves the solve without changing the supplied 3 GPa modulus.
Because the fingers are kinematic, their prescribed closing motion can crush
even a stiff shell; this example does not enforce a gripper force limit.

Each finger carries an aluminum squeeze plate with a 60 mm vertical extent,
48 mm horizontal extent, and 3 mm thickness. The same body-local box geometry
is used for collision and rendering. The contact faces are flush with the
finger pads. The initial 40 mm opening per finger provides an 80 mm gap,
leaving room to approach a bottle that is already dented and has bulged sideways.
``--minimum-opening`` controls the joint displacement and half the plate gap.
The default 17 mm opening per finger gives a 34 mm closed plate gap. Use
``--minimum-opening 0.018`` for the previous 36 mm gap.

Material and permanent creases
-----------------------------

The body material inputs are thickness :math:`t = 0.3` mm, Young's modulus
:math:`E = 3` GPa, and yield stress :math:`\sigma_y = 55` MPa. Poisson ratio
:math:`\nu = 0.38` and PET density 1380 kg/m³ are additional assumptions.
The membrane uses thickness-integrated plane-stress Lamé coefficients.
Flexural rigidity is :math:`D = E t^3 / (12 (1 - \nu^2))`.

The neck finish uses a 1.6 mm wall, blended smoothly from the 0.3 mm body
between 175 and 190 mm above the base. This retains the same PET modulus
and yield stress while increasing neck membrane stiffness by 5.33 times and
flexural rigidity by about 152 times. Triangle masses, contact radii and
hinge yield curvature use the local thickness as well. The neck remains free
to move with the bottle; it is not pinned or made rigid.
The thickness is a modeling assumption informed by the 24.94 mm outside and
21.74 mm bore diameters in a
`PCO 1881 finish drawing <https://www.wiegand-glas.de/products/pet/5239514.pdf>`_,
not a calibration of this procedural bottle.

For a hinge with rest dual width :math:`d`, VBD's bending coefficient is
:math:`D/d`, and the elastic yield angle is
:math:`\theta_y = (2 \sigma_y / (E t)) d`. After each VBD substep, a return map
projects the elastic angle onto :math:`[-\theta_y, \theta_y]`; any excess is
stored in ``state.vbd.edge_plastic_angle``. The effective rest angle is the
model's undeformed angle plus this state-local offset. Unloading retains this
new resting shape without changing the model's shared rest geometry.
The elastic portion can spring back, so the final dent is smaller than the
maximum indentation during the squeeze. Yielding does not continually move
the rest shape when the load is held fixed.

The example registers :meth:`newton.solvers.SolverVBD.register_custom_attributes`
before finalizing its model, authors ``model.vbd.edge_bending_yield_angle``, and
passes ``particle_enable_bending_plasticity=True`` to
:class:`newton.solvers.SolverVBD`. This solver option defaults to ``False``;
``--no-plasticity`` selects the original elastic solve. Material history follows
the normal input/output State buffers and supports CUDA graph replay.
``solver.reset(state, flags=newton.StateFlags.PARTICLE_Q)`` clears the selected
worlds' plastic offsets with their positions; velocity-only resets retain them.
The mode is an ideal bending-only law with no hardening or automatic
differentiation. Yield angles are procedural runtime values indexed by generated
hinges; no USD mesh material schema is introduced.

Updating resting curvature after yield follows the simple shell plasticity
approach in `TRACKS: Toward Directable Thin Shells, section 4.4
<https://www.cs.columbia.edu/cg/pdfs/124-tracks.pdf>`_. The bottle uses a rest dual
width to convert the PET yield curvature to each discrete hinge threshold.

Lengths in the physics model use centimeters, masses use kilograms, and time
uses seconds. Rigidity, yield curvature, shell density, robot geometry, and
collision distances are converted consistently. Rendering uses meters.
The 3 GPa modulus is retained without an artificial stiffness reduction.

RTX appearance
--------------

The PET shell defaults to 0.55 display opacity. Use ``--bottle-opacity`` to
adjust it. A printed Newton label follows the deformed wall, offset by 0.4 mm
to avoid overlapping the PET surface. Its texture reuses the project's logo
and repeats around the bottle; separate triangle corners keep the UV seam
continuous during deformation. An open rolled lip, closure beads and support
flange follow the neck's local frame. These details are render geometry and
add no further collision thickness or stiffness. The underlying physical
neck uses the thicker PET wall described above, and the mouth remains open.

The RTX viewer uses separate PET and water materials. The closed water surface
uses an MDL OmniGlass material with refractive index 1.333, roughness 0.02,
and a light blue tint for transmission and reflections (RGB 0.90, 0.97, 1.0).
This traces transmission through the liquid volume rather than approximating
water with opacity alone. PET uses the viewer's standard material and opacity
handling, including its RTX correction for multiple translucent surfaces.
Opaque water shadows are disabled because the path tracer
does not resolve focused droplet caustics in this scene. This is a visual
approximation, not a photometric validation. See the NVIDIA
`OmniGlass documentation <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/templates/OmniGlass.html>`_.
Install the RTX dependencies and run with:

.. code-block:: console

   uv run --extra examples --extra rtx -m newton.examples franka_bottle_squeeze --viewer rtx

Fluid boundaries and reconstruction
------------------------------------

The example-local XPBD solver in ``_bottle.py`` uses PBF density kernels
with a hash-grid neighbor search. Pressure multipliers accumulate within each
substep, position corrections use multiplier increments, and history resets
before the next substep. Clamping the density deficit and projecting the total
multiplier prevents tensile pressure at the free surface. The default infinite bulk modulus gives zero
compliance; the finite iteration count controls the remaining density error.
This density solver is an example helper, separate from Newton's
:class:`newton.solvers.SolverXPBD` for solids and rigid bodies.
An integrated Poly6 half-space completes
missing kernel support near the bottle wall. Without this boundary term, a
fluid particle would need extra neighbors and excessive compression to attain
its rest density near a solid surface.

Water contacts both sides of the finite shell triangles. Continuous segment
queries prevent fast particles tunneling through the thin wall, and interior
particles preserve their inward contact side at concave ribs. The neck has
no closing triangles, so water can leave through its opening. A reconstructed
surface is extracted every rendered frame with anisotropic kernels and
marching cubes. The kernel support can cross a thin shell even when all
particle centers are contained; reducing particle collision radii alone does
not solve this rendering artifact.

``BottleSurface`` in the same helper intersects the public sparse liquid field
with the deformed PET boundary before marching cubes, then checks the final
vertices at folds that are smaller than a voxel. It recomputes normals after
this correction. The nearest actual particle determines whether to keep liquid
inside or outside the bottle, so escaped droplets remain visible. A temporary
cap is used only for winding-number classification; distances come from the
open PET mesh, and liquid above the neck is unconstrained. The reconstructed
surface also respects the table plane.

``--water-wall-margin`` sets clearance from the PET mid-surface, defaulting to
0.5 mm. It must include at least the 0.15 mm half-thickness of the wall. The
effective clearance is at least half a reconstruction voxel: 0.6 mm in the
1.2 mm export, or 1 mm with the default 2 mm voxels. This changes surface
geometry without changing particle positions, water mass, or shell mechanics.
For finer creases, decrease ``--surface-voxel-size`` while retaining a small
positive margin; a large gap is unnecessary.

The default 2 mm voxels and four-million-cell capacity include
room for spilled droplets; ``--surface-voxel-size`` and
``--surface-max-grid-cells`` control this accuracy and memory tradeoff.

Scope and validation
--------------------

This demonstrates ideal bending plasticity, not a calibrated PET material
model. It omits membrane plasticity, rate dependence, fracture, enclosed air,
and cap mechanics. The bottle bottom is fixed, PET self-weight is neglected,
and the Franka follows joint trajectories obtained from inverse kinematics.
The label and neck details are visual overlays and add no stiffness.

The coupling is one-way: the deforming shell supplies the water boundary, and
water pressure does not feed back onto the shell. A quantitative bottle
squeeze with pressure and
force prediction requires a stable two-way solve; explicit feedback into this
lightweight shell can suffer added-mass instability. XSPH velocity smoothing
and wet-table drag are numerical choices, not calibrated water viscosity or
surface chemistry.

``test_post_step()`` checks finite water positions, table penetration, initial
containment, and absence of yielding before the squeeze. ``test_final()`` checks
the fixed base, reconstructed surface, and retained deformation after release.
Solver tests verify yielding, springback, state ownership, masked resets, and
CUDA graph replay. GPU example tests protect wall contact, surface containment,
open-neck spills, and robot motion; geometry checks protect material scaling
and the plates' physical and visible openings.

Save rendered frames for a recording:

.. code-block:: console

   uv run --extra examples -m newton.examples franka_bottle_squeeze \
       --viewer gl --headless --test --save-frames /tmp/bottle-frames

The frame sequence is 60 fps. An external encoder can turn the PNG sequence
into an MP4. ``--viewer null --test`` runs the same physics checks without an
OpenGL window, while still reconstructing the fluid surface.

The fluid method combines Macklin and Müller's
`Position Based Fluids <https://mmacklin.com/pbf_sig_preprint.pdf>`_ with
`XPBD <https://mmacklin.com/xpbd.pdf>`_, equation 18. Nonzero rest
dihedral angles follow the shell formulation illustrated in
`Discrete Shells <https://www.multires.caltech.edu/pubs/Shells.pdf>`_.
