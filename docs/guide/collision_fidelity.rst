.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Collision-fidelity benchmarks
=============================

This suite measures deformable contact and manipulation with standard VBD.
It includes robot grippers and hands, cloth and garments, bags and contents,
soft solids, cables, and a passive stack of playing cards.

The benchmark calls the installed Newton solver through supported APIs.
It does not patch the solver or add contact behavior. Unsupported cases stay
in the catalog with a reason. The duck recipe requires a second dynamics
solver and is excluded from VBD execution.

The benchmark enables rigid-soft DAT for particle scenes and disables fast
math. It keeps VBD's default rigid formulation. The shared contact policy
caps contact coefficients after the scene supplies its authored values.
These are public solver and model settings. Results record their actual values.

Run a case
----------

Use an environment with Newton's examples and development dependencies.
List cases without building a scene or downloading assets:

.. code-block:: console

   uv run --extra examples python asv/run_collision_fidelity.py --list

Run a short construction and stepping check, then the full task:

.. code-block:: console

   uv run --extra examples python asv/run_collision_fidelity.py \
       --case m3_franka_tote_empty --profile smoke --out results/tote-smoke
   uv run --extra examples python asv/run_collision_fidelity.py \
       --case m3_franka_tote_empty --profile full --out results/tote-full

Use ``--all`` for the complete catalog. Repeat ``--case`` to select several
cases. Use ``--device cuda:0`` to select a device. Add ``--usd`` to record a USD
scene. Rendering is optional and is not imported by numerical runs.
Each output directory contains one JSON file per case and ``summary.json``.
Existing results are not overwritten. A runtime error gives a nonzero exit
status. An unsupported case produces an explicit result without a physics run.
An empty output directory is allowed. The first run claims it with
``.collision-fidelity-run``. Use a new directory after a completed or interrupted
run. Concurrent commands cannot share an output directory.

The smoke profile runs at most one physical second. It does not report a
completed long hold. ``--duration`` can stop a run early; the missing coverage
remains visible. The full profile extends declared manipulation holds to
60 seconds. Other motion keeps its authored speed. Passive tests keep their
declared schedules. Every case uses at most ten substeps per frame.

Run through ASV
---------------

.. code-block:: console

   uvx --with virtualenv asv run --launch-method spawn HEAD^! \
       --bench CollisionFidelity

Use an ASV class name such as ``--bench CollisionFidelityM3FrankaToteEmpty``
to select one scene. The standalone command uses the snake-case case ID.
Each case has its
own cache setup. Its scalar tracks share one simulation result. Selecting
one case does not run the full catalog. CUDA support is checked inside cache
setup. Full tasks are outside the ``Fast*`` PR timing selection.
An unfiltered ASV run still includes these full tasks. Their complete runtime
has not yet been measured.

ASV's ``--quick`` flag does not shorten the physics schedule. Use the standalone
smoke profile for a short check. The Newton commit is the revision under test;
case, asset, protocol and measurement changes define benchmark semantics.

Measurements
------------

The core measurements use exposed particle and rigid-body state. They do not
use a reference simulation. Two free-fall cases also use the analytic gravity
solution. Values in JSON use SI units. ASV reports distances in millimetres.

.. list-table:: Core measurements
   :header-rows: 1
   :widths: 28 72

   * - Name
     - Meaning
   * - Jitter (mm)
     - Maximum position residual RMS after removing a fitted linear trend.
   * - RMS speed (m/s)
     - Maximum speed RMS. Mean velocity is retained.
   * - Rigid-soft penetration: fixed supports (mm)
     - Maximum sampled particle-shell overlap with declared fixed planes and boxes.

For each complete 0.5-second interval, fit a separate line
:math:`a_i + b_i t` to each sampled point's position :math:`x_i(t)`.
For step duration :math:`\Delta t_k` and interval length :math:`T`, compute:

.. math::

   \mathrm{jitter}_{i,w}
   = \sqrt{\frac{1}{T}\sum_k \Delta t_k
       \left\|x_i(t_k)-a_i-b_i t_k\right\|^2},
   \qquad
   \mathrm{speed}_{i,w}
   = \sqrt{\frac{1}{T}\sum_k \Delta t_k\|v_i(t_k)\|^2}.

Report the maximum of each quantity across sampled points and complete
intervals. These maxima can occur at different points and times. Jitter can
include swinging, deformation and falling. It does not isolate solver error.
Samples cover soft particles and declared cable surface points, not only the
grip region. The measurement frame is declared; the default is the world frame.

For a sampled particle-shell gap :math:`g`, penetration is
:math:`\max(0,-\min g)`. The current measurement covers fixed supports only.
It excludes robot contact, soft-soft penetration, self-contact penetration,
triangle interiors and swept crossings. These missing measurements remain
explicit. A separate signed gap includes the declared contact margin.

Keep substeps and iterations separate. A substep advances physical time.
Solver iterations refine the result within that step. Results record both.

Outcomes and limits
-------------------

Grasp, Hold, Transport, and Place and release apply to declared manipulation
tasks. They do not apply to passive stacking or free fall. A prepared hold
does not test grasp acquisition. Case metadata defines each category's scope.

Robot hands follow prescribed poses and joint commands. The suite does not
measure actuator force capacity, full-arm reach or continuous hand-sweep
safety. VBD supplies all contact response and deformable motion.

Simulation completion is separate from task success. Finite state passes only
the numerical check. A sampled separation bound can prove grip loss; overlap
cannot prove retention. Final hand-object separation alone does not prove
successful placement and release. Measurements and acceptance limits that
are missing produce Incomplete, never Pass or a fabricated zero.

An overall Pass requires every applicable required criterion to pass. An
established required violation gives Fail. Missing evidence otherwise gives
Incomplete. No weighted score is used. Acceptance limits remain unset where
they have not been established. These cases still produce raw measurements.
No current case can receive an overall Pass. Required measurements or
acceptance rules are still missing. Incomplete is not a measured failure.

The summary reports pass counts, evaluated counts and applicable counts.
Pass rate is ``pass / (pass + fail)``. Evaluation coverage is
``(pass + fail) / applicable``. Unsupported cases are counted separately.
Mean motion values average per-case maxima with complete required interval
coverage. Penetration means require complete trajectory coverage. Each mean
lists its contributing cases. A valid hold measurement remains useful even
if a later task stage fails.

Contact-capacity observations use exposed counts after each solver step.
They do not cover peaks in every internal detection pass. Self-contact
capacity checks are unavailable through this observer. The benchmark does
not add solver hooks to obtain them.

The contact-coefficient policy caps contact values at ten times the declared
elastic reference. Newton declares cloth and tetrahedral material coefficients
in Pa, while contact coefficients use N/m. Their cap compares numerical
values with different units. Rod stretch and shear coefficients use N/m.
This is a parameter policy, not a general physical stiffness ratio. Asset
notices distinguish synthetic material values from measured properties.

Procedural assets need to preserve their size, connected structure, openings,
seams and grasp regions. They need not have identical vertex bytes or triangle
order on every platform. Generated mesh hashes identify the actual inputs;
they are provenance, not an acceptance test.

Historical evidence
-------------------

Earlier runs document useful failures. Their code and settings differ from
the current benchmark and do not validate the current PR head:

* `Cloth manipulation recordings <https://kanna.ankachen.com/artifacts/newton-dat-ogc-synced-main/index.html>`__.
* `Shopping-bag trials <https://kanna.ankachen.com/artifacts/newton-shopping-bag-grip/index.html>`__.
* `Poker cards without a moving ball <https://kanna.ankachen.com/artifacts/newton-poker-cards-no-ball/index.html>`__.

This addition is a runtime benchmark. It adds no public solver API, USD schema
or importer behavior. Optional USD output uses the existing viewer.
