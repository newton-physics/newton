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
``wp.config.deterministic`` at construction. Prefer the per-object arguments
over setting ``wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN``
process-wide: the global setting is also Warp's default for every module that
does not set its own option, so it additionally compiles the collision
pipeline's kernels, other Newton utility kernels, and your own kernels with
recorded atomics. That costs speed and memory and can overflow the record
buffers of kernels with many atomics per thread. It does not enable the
collision pipeline's contact sorting, which follows only the pipeline's own
``deterministic`` flag.

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
       ``RUN_TO_RUN`` only; ``GPU_TO_GPU`` raises. The ``BODY_PARENT_F``
       observable sums contact wrenches with atomics and is outside the
       guarantee. See ``python -m newton.examples steel_determinism_arms``.
   * - :class:`~solvers.SolverMuJoCo`
     - CPU only
     - The CPU backend (``use_mujoco_cpu=True``) is always deterministic. On
       the MuJoCo Warp backend, ``deterministic`` currently fails to compile
       MuJoCo Warp's kernels with Warp 1.19, and MuJoCo Warp does not yet
       order its contacts or constraint rows. Tracked in
       `#5042 <https://github.com/newton-physics/newton/issues/5042>`__.
   * - :class:`~solvers.SolverStyle3D`
     - No
     - Accumulates cloth and contact forces with floating-point atomics.
   * - :class:`~solvers.SolverImplicitMPM`
     - No
     - Has no ``deterministic`` option. In our measurements the default PIC
       transfer was repeatable; the GIMP transfer is not.
   * - Coupled solvers (:mod:`newton.solvers.experimental.coupled`)
     - No
     - The couplers route contacts and sum coupling forces with atomics, so
       contact order is not fixed inside a coupler even when an entry solver
       is deterministic. A Steel entry with ``RUN_TO_RUN`` raises
       :class:`NotImplementedError`; see :doc:`coupling`.
   * - :class:`~solvers.SolverKamino`
     - No
     - Converts contacts and reserves constraint rows with atomic counters.

Planned work on missing solver support and on the cost of the mode is tracked
in `#5049 <https://github.com/newton-physics/newton/issues/5049>`__.

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

Some combinations currently hit record-buffer limits:

- :class:`~solvers.SolverVBD` with particle self-contact bounds the per-thread
  records of every kernel in its particle module by the self-contact buffer
  sizes, so large cloth meshes can run out of memory.
- :class:`~solvers.SolverSemiImplicit`'s triangle-particle contact kernel
  launches one thread per triangle-particle pair, so for large meshes its
  worst-case record buffer exceeds Warp's 32-bit array limit during graph
  capture. Step eagerly or pass ``enable_tri_contact=False``.

In practice:

- Use :class:`~solvers.SolverSteel` when determinism matters for articulated or
  rigid-body scenes.
- For other solvers, measure both eager stepping and graph replay with
  determinism on; depending on the scene, eager stepping can be faster.
- Size the :class:`CollisionPipeline` contact buffers (``rigid_contact_max``,
  ``soft_contact_max``) just above the peak contact count of the densest
  expected frame, not far beyond it. Contact kernels launch over the whole
  buffer, so under graph replay their record buffers grow with it, but a frame
  that exceeds the capacity drops contacts; watch for the
  ``Contact buffer overflowed`` and ``Mesh soft-contact capacity exceeded``
  warnings.
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
