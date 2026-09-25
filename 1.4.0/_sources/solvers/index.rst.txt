.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Solvers
========

Solvers are used to integrate the dynamics of a Newton model.
The typical workflow is to construct a :class:`~newton.Model` and a :class:`~newton.State` object, then use a solver to advance the state forward in time
via the :meth:`~newton.solvers.SolverBase.step` method:

.. mermaid::
  :config: {"theme": "forest", "themeVariables": {"lineColor": "#76b900"}}

  flowchart LR
      subgraph Input["Input Data"]
          M[newton.Model]
          S[newton.State]
          C[newton.Control]
          K[newton.Contacts]
          DT[Time step dt]
      end

      STEP["solver.step()"]

      subgraph Output["Output Data"]
          SO["newton.State (updated)"]
      end

      %% Connections
      M --> STEP
      S --> STEP
      C --> STEP
      K --> STEP
      DT --> STEP
      STEP --> SO

See the :doc:`solver API reference </api/newton_solvers>` for constructor and
method details. For symptom-driven diagnosis and parameter tuning, start with
:ref:`Simulation Tuning`.

Choosing a Solver
-----------------

MuJoCo and Kamino currently have dedicated backend guides:

- :doc:`MuJoCo <mujoco>` — generalized-coordinate rigid-body simulation and
  MuJoCo or MJCF workflows.
- :doc:`Kamino <kamino>` — constrained rigid mechanisms with kinematic loops
  and hard frictional contacts; experimental.

The other solver backends are documented through their linked API references
and the comparison tables below.

For a rigid articulated robot, first decide which coordinate representation
fits the application. :class:`~newton.solvers.SolverMuJoCo` and
:class:`~newton.solvers.SolverFeatherstone` use generalized coordinates, while
:class:`~newton.solvers.SolverXPBD`,
:class:`~newton.solvers.SolverSemiImplicit`, and
:class:`~newton.solvers.SolverKamino` use maximal coordinates. For deformable
bodies, particles, or differentiable simulation, use the feature matrix below
to narrow the choice, then follow the linked API documentation.

.. _Supported Features:

Supported Features
------------------

.. list-table::
   :header-rows: 1
   :widths: auto
   :stub-columns: 0

   * - Solver
     - :abbr:`Integration (Available methods for integrating the dynamics)`
     - Rigid bodies
     - :ref:`Articulations <Articulations>`
     - Particles
     - Cloth
     - Soft bodies
     - Differentiable
   * - :class:`~newton.solvers.SolverFeatherstone`
     - Semi-implicit
     - ✅
     - ✅ generalized coordinates
     - ✅
     - 🟨 no self-collision
     - ✅
     - 🟨 basic :sup:`2`
   * - :class:`~newton.solvers.SolverImplicitMPM`
     - Implicit
     - ❌
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
   * - :class:`~newton.solvers.SolverKamino`
     - Semi-implicit: Euler, Moreau-Jean
     - ✅ maximal coordinates
     - ✅ maximal coordinates
     - ❌
     - ❌
     - ❌
     - ❌
   * - :class:`~newton.solvers.SolverMuJoCo`
     - Explicit, Semi-implicit, Implicit-in-velocity
     - ✅ :sup:`1`
     - ✅ generalized coordinates
     - ❌
     - ❌
     - ❌
     - ❌
   * - :class:`~newton.solvers.SolverSemiImplicit`
     - Semi-implicit
     - ✅
     - ✅ maximal coordinates
     - ✅
     - 🟨 no self-collision
     - ✅
     - 🟨 basic :sup:`2`
   * - :class:`~newton.solvers.SolverStyle3D`
     - Implicit
     - ❌
     - ❌
     - ✅
     - ✅
     - ❌
     - ❌
   * - :class:`~newton.solvers.SolverVBD`
     - Implicit
     - ✅
     - 🟨 :ref:`limited joint support <Joint feature support>`
     - ✅
     - ✅
     - ✅
     - ❌
   * - :class:`~newton.solvers.SolverXPBD`
     - Implicit
     - ✅
     - ✅ maximal coordinates
     - ✅
     - 🟨 no self-collision
     - 🟨 experimental
     - ❌

| :sup:`1` Uses MuJoCo's built-in collision pipeline by default.
  Set ``use_mujoco_contacts=False`` to use Newton's collision pipeline.
| :sup:`2` ``basic`` means Newton includes several examples that use these solvers in diffsim workflows,
  see :ref:`Differentiability` for further details.

.. experimental::
    :class:`~newton.solvers.SolverKamino`'s public API and behavior may change without prior notice.

.. experimental::
    :class:`~newton.solvers.SolverVBD`'s public API and behavior may change without prior notice.

.. _Contact material support:

Contact Material Support
------------------------

:class:`~newton.ModelBuilder.ShapeConfig` and the matching :class:`~newton.Model`
shape material arrays store solver-neutral contact data. This section documents
which fields are currently used by Newton's built-in solvers. External solvers
may use different subsets or interpret these fields according to their own
formulation.

- ``mu``: :class:`~newton.solvers.SolverFeatherstone`,
  :class:`~newton.solvers.SolverSemiImplicit`,
  :class:`~newton.solvers.SolverXPBD`, :class:`~newton.solvers.SolverMuJoCo`,
  :class:`~newton.solvers.SolverVBD`, :class:`~newton.solvers.SolverKamino`,
  :class:`~newton.solvers.SolverStyle3D`, and
  :class:`~newton.solvers.SolverImplicitMPM`.
- ``ke`` / ``kd``: :class:`~newton.solvers.SolverFeatherstone`,
  :class:`~newton.solvers.SolverSemiImplicit`,
  :class:`~newton.solvers.SolverMuJoCo`, and
  :class:`~newton.solvers.SolverVBD`.
- ``kf`` / ``ka``: :class:`~newton.solvers.SolverFeatherstone` and
  :class:`~newton.solvers.SolverSemiImplicit`.
- ``restitution``: :class:`~newton.solvers.SolverXPBD` when
  ``enable_restitution=True``, and :class:`~newton.solvers.SolverKamino`.
- ``mu_torsional`` / ``mu_rolling``: :class:`~newton.solvers.SolverXPBD` and
  :class:`~newton.solvers.SolverMuJoCo`.
- ``kh``: consumed by hydroelastic contact generation for Newton-generated
  contacts used by :class:`~newton.solvers.SolverFeatherstone`,
  :class:`~newton.solvers.SolverSemiImplicit`, and
  :class:`~newton.solvers.SolverMuJoCo` when ``use_mujoco_contacts=False``. See
  :ref:`Hydroelastic Contacts`.

.. _Joint feature support:

Joint Feature Support
---------------------

Not every solver supports every joint type or joint property.
The tables below document which joint features each solver handles.

Only :class:`~newton.solvers.SolverFeatherstone` and :class:`~newton.solvers.SolverMuJoCo`
operate on :ref:`articulations <Articulations>` (generalized/reduced coordinates).
The maximal-coordinate solvers (:class:`~newton.solvers.SolverSemiImplicit`,
:class:`~newton.solvers.SolverXPBD`, and :class:`~newton.solvers.SolverKamino`)
enforce joints as pairwise body constraints but do not use the articulation kinematic-tree structure.
:class:`~newton.solvers.SolverVBD` supports a subset of joint types via soft constraints (AVBD).
:class:`~newton.solvers.SolverStyle3D` and :class:`~newton.solvers.SolverImplicitMPM` do not support joints.

**Joint types**

.. list-table::
   :header-rows: 1
   :widths: auto
   :stub-columns: 1

   * - Joint type
     - :class:`~newton.solvers.SolverFeatherstone`
     - :class:`~newton.solvers.SolverSemiImplicit`
     - :class:`~newton.solvers.SolverXPBD`
     - :class:`~newton.solvers.SolverMuJoCo`
     - :class:`~newton.solvers.SolverVBD`
     - :class:`~newton.solvers.SolverKamino`
   * - PRISMATIC
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
   * - REVOLUTE
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
   * - BALL
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
   * - FIXED
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
   * - FREE
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
   * - DISTANCE
     - 🟨 :sup:`1`
     - 🟨 :sup:`1`
     - |yes|
     - |no|
     - |no|
     - |no|
   * - D6
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |no|
   * - CABLE
     - |no|
     - |no|
     - |no|
     - |no|
     - |yes|
     - |no|

| :sup:`1` DISTANCE joints are treated as FREE (no distance constraint enforcement).

**Joint properties**

.. list-table::
   :header-rows: 1
   :widths: auto
   :stub-columns: 1

   * - Property
     - :class:`~newton.solvers.SolverFeatherstone`
     - :class:`~newton.solvers.SolverSemiImplicit`
     - :class:`~newton.solvers.SolverXPBD`
     - :class:`~newton.solvers.SolverMuJoCo`
     - :class:`~newton.solvers.SolverVBD`
     - :class:`~newton.solvers.SolverKamino`
   * - :attr:`~newton.Model.joint_enabled`
     - |no|
     - |yes|
     - |yes|
     - |no|
     - |yes|
     - |no|
   * - :attr:`~newton.Model.joint_armature`
     - |yes|
     - |no|
     - |no|
     - |yes|
     - |no|
     - |yes|
   * - :attr:`~newton.Model.joint_friction`
     - |no|
     - |no|
     - |no|
     - |yes|
     - |no|
     - |no|
   * - :attr:`~newton.Model.joint_limit_lower` / :attr:`~newton.Model.joint_limit_upper`
     - |yes|
     - |yes| :sup:`2`
     - |yes|
     - |yes|
     - |yes|
     - |yes|
   * - :attr:`~newton.Model.joint_limit_ke` / :attr:`~newton.Model.joint_limit_kd`
     - |yes|
     - |yes| :sup:`2`
     - |no|
     - |yes|
     - |yes| :sup:`4`
     - |no|
   * - :attr:`~newton.Model.joint_effort_limit`
     - |no|
     - |no|
     - |no|
     - |yes|
     - |no|
     - |no|
   * - :attr:`~newton.Model.joint_velocity_limit`
     - |no|
     - |no|
     - |no|
     - |no|
     - |no|
     - |no|

| :sup:`2` Not enforced for BALL joints in SemiImplicit.

**Actuation and control**

.. list-table::
   :header-rows: 1
   :widths: auto
   :stub-columns: 1

   * - Feature
     - :class:`~newton.solvers.SolverFeatherstone`
     - :class:`~newton.solvers.SolverSemiImplicit`
     - :class:`~newton.solvers.SolverXPBD`
     - :class:`~newton.solvers.SolverMuJoCo`
     - :class:`~newton.solvers.SolverVBD`
     - :class:`~newton.solvers.SolverKamino`
   * - :attr:`~newton.Model.joint_target_ke` / :attr:`~newton.Model.joint_target_kd`
     - |yes|
     - |yes| :sup:`2`
     - |yes|
     - |yes|
     - |yes| :sup:`4`
     - |yes|
   * - :attr:`~newton.Model.joint_target_mode`
     - |no|
     - |no|
     - |no|
     - |yes|
     - |no|
     - |yes|
   * - :attr:`~newton.Control.joint_f` (feedforward forces)
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|

**Constraints**

.. list-table::
   :header-rows: 1
   :widths: auto
   :stub-columns: 1

   * - Feature
     - :class:`~newton.solvers.SolverFeatherstone`
     - :class:`~newton.solvers.SolverSemiImplicit`
     - :class:`~newton.solvers.SolverXPBD`
     - :class:`~newton.solvers.SolverMuJoCo`
     - :class:`~newton.solvers.SolverVBD`
     - :class:`~newton.solvers.SolverKamino`
   * - Equality constraints (CONNECT, WELD, JOINT)
     - |no|
     - |no|
     - |no|
     - |yes|
     - |no|
     - |no|
   * - Mimic constraints
     - |no|
     - |no|
     - |no|
     - |yes| :sup:`3`
     - |no|
     - |no|

| :sup:`3` Mimic constraints in MuJoCo are supported for REVOLUTE and PRISMATIC joints only.
| :sup:`4` VBD interprets ``joint_target_kd`` and ``joint_limit_kd`` as absolute damping coefficients in physical units.



.. _Differentiability:

Differentiability
-----------------

Differentiable simulation in Newton typically runs a forward rollout inside
``wp.Tape()``, computes a scalar loss from the simulated state, and then calls
``tape.backward(loss)`` to populate gradients on differentiable state,
control, or model arrays. In practice, this starts by calling
:meth:`~newton.ModelBuilder.finalize` with ``requires_grad=True``.

.. testcode::

    import warp as wp
    import newton

    @wp.kernel
    def loss_kernel(particle_q: wp.array[wp.vec3], target: wp.vec3, loss: wp.array[float]):
        delta = particle_q[0] - target
        loss[0] = wp.dot(delta, delta)

    builder = newton.ModelBuilder()
    builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.0), vel=wp.vec3(1.0, 0.0, 0.0), mass=1.0)

    model = builder.finalize(requires_grad=True)
    solver = newton.solvers.SolverSemiImplicit(model)

    state_in = model.state(requires_grad=True)
    state_out = model.state(requires_grad=True)
    control = model.control()
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    target = wp.vec3(0.25, 0.0, 0.0)

    tape = wp.Tape()
    with tape:
        state_in.clear_forces()
        solver.step(state_in, state_out, control, None, 1.0 / 60.0)
        wp.launch(
            loss_kernel,
            dim=1,
            inputs=[state_out.particle_q, target],
            outputs=[loss],
        )

    tape.backward(loss)
    initial_velocity_grad = state_in.particle_qd.grad.numpy()
    assert float(initial_velocity_grad[0, 0]) < 0.0

See the `DiffSim examples on GitHub`_ for the current reference workflows.

.. _DiffSim examples on GitHub: https://github.com/newton-physics/newton/tree/main/newton/examples/diffsim

.. |yes| unicode:: U+2705
.. |no| unicode:: U+274C
