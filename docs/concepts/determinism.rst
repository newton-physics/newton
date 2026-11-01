.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

.. _determinism:

Deterministic Simulation
========================

By default, a Newton simulation on the GPU is not bit-for-bit repeatable: two
runs from the same initial state with the same inputs usually agree to a few
units in the last place at first and then drift apart, because many GPU threads
accumulate into the same values and GPU scheduling decides the order in which
they arrive. Newton can trade speed and memory for *run-to-run determinism*,
where repeated runs on the same GPU model, driver, and Warp build produce
bitwise identical results. Use it for regression tests, debugging,
reproducible training and data generation, and replays that must match a
recording.

Determinism has to be requested from every stage that produces a step: the
collision pipeline, which decides which contacts exist and in what order, and
the solver, which consumes them.

.. code-block:: python

   import warp as wp

   import newton

   pipeline = newton.CollisionPipeline(model, deterministic=True)
   solver = newton.solvers.SolverXPBD(model, deterministic=wp.DeterministicMode.RUN_TO_RUN)

   contacts = pipeline.contacts()
   for _ in range(num_steps):
       state_0.clear_forces()
       pipeline.collide(state_0, contacts)
       solver.step(state_0, state_1, control, contacts, dt)
       state_0, state_1 = state_1, state_0

A solver's ``deterministic`` argument defaults to ``None``, which inherits
``wp.config.deterministic`` at construction, so setting
``wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN`` before
constructing solvers enables the mode process-wide. The collision pipeline only
follows its own ``deterministic`` flag.

Sources of nondeterminism
-------------------------

Three kinds of GPU work make results depend on thread scheduling:

- **Floating-point accumulation.** Many threads add into one value with atomic
  operations, for example contact forces into a body or element forces into a
  particle. Floating-point addition is not associative, so the sum depends on
  the order in which threads arrive.
- **Slot allocation.** Threads reserve output slots with an atomic counter,
  for example when compacting contacts or reserving constraint rows. Every run
  writes the same set of entries but in a different order, which changes the
  order of later sums and of iterative solves such as Gauss-Seidel.
- **Races that pick a winner.** For example, a contact reduction that keeps the
  deepest contact per bin must break ties the same way every time.

Newton addresses these with Warp's
`deterministic execution mode <https://nvidia.github.io/warp/stable/user_guide/deterministic_execution.html>`__,
which records atomic updates and reduces them in a fixed order and assigns
counter slots in thread order, and, where that is not enough or too slow, with
solver-specific ordering such as fixed constraint-row reservation or sorting.
Integer accumulation and code paths that write each output from exactly one
thread are already deterministic and cost nothing extra.

Collision pipeline
------------------

``CollisionPipeline(deterministic=True)`` sorts contacts into a canonical order
after the narrow phase, breaks contact-reduction ties with geometry-derived
fingerprints, and accumulates hydroelastic aggregates in fixed point; see
:ref:`Deterministic Contacts`. Contact matching (``contact_matching`` other than
``"disabled"``) implies ``deterministic=True``. Deterministic mode disables
hydroelastic pre-pruning, so its contact set differs from the default.

Solver support
--------------

Solvers that support the mode apply it to the Warp modules whose kernels use
atomic operations. Solvers that share kernel modules re-apply their own mode
before each step, so solvers with different modes can coexist in one process.
:class:`warp.DeterministicMode` also defines ``GPU_TO_GPU`` (reproducible across
GPU models); Newton only tests ``RUN_TO_RUN``.

.. list-table::
   :header-rows: 1
   :widths: 24 12 64

   * - Solver
     - Support
     - Notes
   * - :class:`~solvers.SolverXPBD`
     - Yes
     - Warp deterministic mode on the solver's atomic kernels.
   * - :class:`~solvers.SolverSemiImplicit`
     - Yes
     - Warp deterministic mode. With triangle contact enabled, large meshes
       cannot be graph-captured in this mode (see `Performance`_).
   * - :class:`~solvers.SolverFeatherstone`
     - Yes
     - Warp deterministic mode on the solver's atomic kernels.
   * - :class:`~solvers.SolverVBD`
     - Yes
     - Warp deterministic mode; body-particle contacts are gathered per
       particle instead of with atomics. With particle self-contact, record
       buffers grow with the contact buffer sizes and need a lot of memory
       (see `Performance`_).
   * - :class:`~solvers.SolverSteel`
     - Yes
     - Reserves constraint rows in a fixed per-world order instead of with
       atomic counters, so it costs about as much as the default.
       ``RUN_TO_RUN`` only; ``GPU_TO_GPU`` raises. See
       ``python -m newton.examples steel_determinism_arms``.
   * - :class:`~solvers.SolverMuJoCo`
     - CPU only
     - The CPU backend (``use_mujoco_cpu=True``) is always deterministic. On
       the MuJoCo Warp backend, ``deterministic`` currently fails to compile
       MuJoCo Warp's kernels with Warp 1.19, and MuJoCo Warp does not yet
       order its contacts or constraint rows.
   * - :class:`~solvers.SolverStyle3D`
     - No
     - Accumulates cloth and contact forces with floating-point atomics.
   * - :class:`~solvers.SolverImplicitMPM`
     - No
     - Has no ``deterministic`` option. In our measurements the default PIC
       transfer was repeatable; the GIMP transfer is not.
   * - Coupled solvers (:mod:`newton.solvers.experimental.coupled`)
     - No
     - Entry solvers can be deterministic, but the couplers' own kernels
       accumulate with atomics.
   * - :class:`~solvers.SolverKamino`
     - No
     - Converts contacts and reserves constraint rows with atomic counters.

Performance
-----------

Determinism costs speed and memory, and the cost depends strongly on the solver
and on whether the step runs eagerly or from a captured CUDA graph:

- **Recorded atomics.** Warp's deterministic mode writes every floating-point
  atomic update to a record buffer, sorts the records, and reduces them in a
  fixed order. Solvers that accumulate heavily with atomics pay for that sort
  in every launch.
- **Graph replay sorts worst-case buffers.** When stepping eagerly, each
  deterministic launch reads back how many records it produced and sorts only
  those, at the cost of one host synchronization per launch. A captured graph
  cannot read anything back, so every replay sorts the full buffer capacity:
  the launch size times the per-thread record bound. Contact kernels launch over
  the whole contact buffer, so under graph replay their cost follows the buffer
  size rather than the number of contacts.
- **Memory.** Record buffers are sized for the worst case and stay allocated for
  the lifetime of a captured graph.
- **Solver-specific ordering.** :class:`~solvers.SolverSteel` orders its work
  itself instead of relying on Warp's mode and costs little or nothing.

The table below shows steady-state step times, including collision detection,
with determinism off and on, measured on an NVIDIA L4 with Warp 1.19.0rc1 over
200 steps after 120 warm-up steps. "Graph" times replay a captured two-step
graph; "Eager" is the on/off ratio without capture, which also includes Python
launch overhead and therefore understates the GPU cost. "Peak memory" is the
CUDA memory-pool high-water mark, including graph capture. Every run with
determinism on repeated bitwise. Timings come from a shared GPU, so treat them
as indicative.

.. list-table::
   :header-rows: 1
   :widths: 34 16 18 10 22

   * - Scene and solver
     - Graph off [ms]
     - Graph on [ms]
     - Eager
     - Peak memory [MB]
   * - Rigid pile: XPBD
     - 0.33
     - 19.6 (59x)
     - 5.3x
     - 26 → 830
   * - Rigid pile: SemiImplicit [1]_
     - 0.10
     - 3.78 (39x)
     - 2.3x
     - 26 → 172
   * - Rigid pile: Featherstone [1]_
     - 0.24
     - 3.85 (16x)
     - 2.3x
     - 30 → 174
   * - Rigid pile: Steel
     - 5.27
     - 5.02 (1.0x)
     - 1.1x
     - 435 → 484
   * - Humanoids: Featherstone [1]_
     - 44.7
     - 49.3 (1.1x)
     - 1.0x
     - 42 → 218
   * - Humanoids: Steel
     - 3.38
     - 3.47 (1.0x)
     - 1.1x
     - 187 → 247
   * - Humanoids: XPBD
     - 0.35
     - 23.0 (66x)
     - 6.8x
     - 29 → 1008
   * - Cloth: XPBD
     - 0.54
     - 5.34 (10x)
     - 7.9x
     - 24 → 172
   * - Cloth: VBD
     - 47.6
     - fails [2]_
     -
     -
   * - Soft bodies: SemiImplicit
     - 3.72
     - fails [3]_
     - 4.0x
     - 26 → 7841
   * - Soft bodies: XPBD
     - 0.71
     - 10.5 (15x)
     - 12.7x
     - 26 → 425
   * - Soft bodies: VBD
     - 0.86
     - 0.89 (1.0x)
     - 1.1x
     - 27 → 27

Scenes: *rigid pile* is 64 worlds of 48 falling boxes, spheres, and capsules
(3,072 free bodies); *humanoids* is 256 MJCF humanoids on a ground plane
(3,328 bodies); *cloth* is a 97 x 97-particle cloth draped over a sphere;
*soft bodies* is eight 11 x 11 x 11 tetrahedral grids (10,648 particles)
dropped on the ground. Each solver has enough constraint capacity that no rows
are dropped.

.. [1] Simulated at 1 kHz (rigid pile) or 600 Hz (humanoids), which these
   explicit integrators need with the default contact stiffness.
.. [2] Runs out of memory. With particle self-contact, VBD bounds the per-thread
   records of every kernel in its particle module by the self-contact buffer
   sizes.
.. [3] SemiImplicit's triangle-particle contact kernel launches one thread per
   triangle-particle pair, so its worst-case record buffer exceeds Warp's
   32-bit array limit during graph capture. Step eagerly or pass
   ``enable_tri_contact=False``.

In practice:

- Use :class:`~solvers.SolverSteel` when determinism matters for articulated or
  rigid-body scenes.
- For other solvers, measure both eager stepping and graph replay with
  determinism on; eager stepping can be many times faster.
- Size the :class:`CollisionPipeline` contact buffers (``rigid_contact_max``,
  ``soft_contact_max``) close to the contacts a scene produces. Contact kernels
  launch over the whole buffer, so under graph replay their record buffers grow
  with it.
- Keep deterministic mode for the runs that need it, such as tests, debugging,
  and data generation.

Caveats
-------

- **Scope of the guarantee.** ``RUN_TO_RUN`` results are repeatable on the same
  GPU model with the same driver, CUDA toolkit, Warp build, and Newton version,
  and with the same model, inputs, solver settings, and step sequence. They
  usually differ across GPU models and between the CPU and CUDA backends.
- **Everything in the loop counts.** Controllers, sensors, resets, and user
  kernels that accumulate with atomics must be deterministic as well. Reset
  solver state such as warm starts the same way in every run.
- **Graph capture.** Deterministic kernels can be captured in CUDA graphs. Warp
  checks its record buffers for overflow only outside of capture, so run a few
  steps eagerly before capturing.
- **Mixing modes.** Solvers that share kernel modules but use different modes
  switch the shared modules' options on every step, which reloads those modules.
  Prefer one mode per process.
- **Testing.** Compare repeated runs bitwise, for example with
  :func:`numpy.testing.assert_array_equal`, rather than within a tolerance; a
  tolerance can hide the first divergence until it grows.
