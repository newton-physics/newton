.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. _Worlds:

Worlds
======

Newton enables multiple independent simulations, referred to as *worlds*, within a single :class:`~newton.Model` object.
Each *world* provides an index-based grouping of all primary simulation entities such as particles, bodies, shapes, joints, articulations and equality constraints.


Overview
--------

GPU-accelerated operations in Newton often involve parallelizing over an entire set of model entities, e.g. bodies, shapes or joints, without needing to consider which specific world they belong to.
However, some operations, such as those part of Collision Detection (CD), can exploit world-based grouping to effectively filter out potential collisions between shapes that belong to different worlds.
Moreover, world-based grouping can also facilitate partitioning of thread grids according to both world indices and the number of entities per world.
Such operations facilitate support for simulating multiple, and potentially heterogeneous, worlds defined within a :class:`~newton.Model` instance.
Lastly, world-based grouping also enables selectively operating on only the entities that belong to a specific world, i.e. masking, as well as partitioning of the :class:`~newton.Model` and :class:`~newton.State` data.

.. experimental::

   Support for fully heterogeneous simulations is still under active development
   and may change without prior notice.
   At present time, although the :class:`~newton.ModelBuilder` and :class:`~newton.Model` objects support instantiating worlds with different disparate entities, not all solvers are able to simulate them.
   Moreover, the selection API still operates under the assumption of model homogeneity, but this is expected to also support heterogeneous simulations in the near future.

.. _World assignment:

World Assignment
----------------

World assignment is managed by :class:`~newton.ModelBuilder` when entities are added through methods such as :meth:`~newton.ModelBuilder.add_body`,
:meth:`~newton.ModelBuilder.add_joint`, and :meth:`~newton.ModelBuilder.add_shape`. Assignment can either be global (world index ``-1``) or specific to a particular world (indices ``0, 1, 2, ...``).
The supported workflows are:

* Add entities before the first call to :meth:`~newton.ModelBuilder.begin_world` or after the last call to :meth:`~newton.ModelBuilder.end_world` to place them in the global world (index ``-1``), or between those calls to place them in a specific world.
* Create worlds from a sub-builder with :meth:`~newton.ModelBuilder.add_world` or :meth:`~newton.ModelBuilder.replicate`.

Within a world scope, each entity is assigned the current world index. The :attr:`~newton.ModelBuilder.current_world` attribute is a read-only property that reflects the active builder context and should not be set directly.

The following example creates two different worlds within a single model:

.. testcode::

   import warp as wp
   import newton

   builder = newton.ModelBuilder()

   # Global entity at front (world -1)
   builder.add_ground_plane()

   # World 0: two free-floating spheres
   builder.begin_world()
   b0 = builder.add_body(mass=1.0)
   b1 = builder.add_body(mass=1.0)
   builder.add_shape_sphere(body=b0, radius=0.1)
   builder.add_shape_sphere(body=b1, radius=0.1)
   builder.end_world()

   # World 1: fixed-base revolute articulation with boxes
   builder.begin_world()
   link0 = builder.add_link(mass=1.0)
   j0 = builder.add_joint_fixed(parent=-1, child=link0)
   link1 = builder.add_link(mass=2.0)
   j1 = builder.add_joint_revolute(parent=link0, child=link1)
   builder.add_articulation(joints=[j0, j1])
   builder.add_shape_box(body=link0, hx=0.1, hy=0.1, hz=0.1)
   builder.add_shape_box(body=link1, hx=0.1, hy=0.1, hz=0.1)
   builder.end_world()

   # Global entity at back (world -1)
   builder.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.05)

   model = builder.finalize()

In this example, we create a model with two worlds (world ``0`` and world ``1``) containing different bodies, shapes and joints, as well as two global entities (the ground plane at the front and a static box at the back, both with world index ``-1``).

For homogeneous multi-world scenes, prefer :meth:`~newton.ModelBuilder.add_world` or :meth:`~newton.ModelBuilder.replicate` instead of manually repeating world scopes for each copy.


.. _World grouping:

World Grouping
--------------

The :class:`~newton.ModelBuilder` maintains internal lists that track the world assignment of each entity added to it.
When :meth:`~newton.ModelBuilder.finalize` is called, the :class:`~newton.Model` object generated will contain arrays that store the world indices for each entity type.

Specifically, the entity types that currently support world grouping include:

- Particles: :attr:`~newton.Model.particle_world`
- Bodies: :attr:`~newton.Model.body_world`
- Shapes: :attr:`~newton.Model.shape_world`
- Joints: :attr:`~newton.Model.joint_world`
- Articulations: :attr:`~newton.Model.articulation_world`

The corresponding world grouping arrays for the example above are:

.. testcode::

   print("Body worlds:", model.body_world.numpy().tolist())
   print("Shape worlds:", model.shape_world.numpy().tolist())
   print("Joint worlds:", model.joint_world.numpy().tolist())

.. testoutput::

   Body worlds: [0, 0, 1, 1]
   Shape worlds: [-1, 0, 0, 1, 1, -1]
   Joint worlds: [0, 0, 1, 1]


.. _World starts:

World Start Indices & Dimensions
--------------------------------

In addition to the world grouping arrays, the :class:`~newton.Model` object will also contain Warp arrays that store the per-world starting indices for each entity type.

These arrays include:

- Particles: :attr:`~newton.Model.particle_world_start`
- Bodies: :attr:`~newton.Model.body_world_start`
- Shapes: :attr:`~newton.Model.shape_world_start`
- Joints: :attr:`~newton.Model.joint_world_start`
- Articulations: :attr:`~newton.Model.articulation_world_start`

To handle the special case of joint entities, that vary in the number of DOFs, coordinates and constraints, the model also provides arrays that store the per-world starting indices in these specific dimensions:

- Joint DOFs: :attr:`~newton.Model.joint_dof_world_start`
- Joint Coordinates: :attr:`~newton.Model.joint_coord_world_start`
- Joint Constraints: :attr:`~newton.Model.joint_constraint_world_start`

All ``*_world_start`` arrays adopt a special format that facilitates accounting of the total number of entities in each world as well as the global world (index ``-1``) at the front and back of each per-entity array such as :attr:`~newton.Model.body_world`.
Specifically, each ``*_world_start`` array contains ``world_count + 2`` entries, with the first ``world_count`` entries corresponding to starting indices of each ``world >= 0`` world,
the second-to-last entry corresponding to the starting index of the global entities at the back (world index ``-1``), and the last entry corresponding to total number of entities or dimensions in the model.

With this format, we can easily compute the number of entities per world by computing the difference between consecutive entries in these arrays (since they are essentially cumulative sums),
as well as the total number of global entities by summing the first entry with the difference of the last two.

Continuing the same example, we can compute the per-world shape counts as follows:

.. testcode::

   print("world_count:", model.world_count)

   # Shape start indices per world
   # Entries: [start_world_0, start_world_1, start_global_back, total_shapes]
   shape_start = model.shape_world_start.numpy()
   print("Shape starts:", shape_start.tolist())

   # Compute per-world shape counts
   world_shape_counts = [
       int(shape_start[i + 1] - shape_start[i])
       for i in range(model.world_count)
   ]
   # Global shapes: those at the front (before start_world_0) plus at the back
   global_shape_count = int(shape_start[0]) + int(shape_start[-1] - shape_start[-2])

   print("Shape counts per world:", world_shape_counts)
   print("Global shape count:", global_shape_count)

.. testoutput::

   world_count: 2
   Shape starts: [1, 3, 5, 6]
   Shape counts per world: [2, 2]
   Global shape count: 2


.. _Convenience methods:

Convenience Methods: ``add_world`` and ``replicate``
----------------------------------------------------

While :meth:`~newton.ModelBuilder.begin_world` and :meth:`~newton.ModelBuilder.end_world` give full control, Newton provides two convenience methods for the most common multi-world patterns:

**add_world**: adds a pre-built :class:`~newton.ModelBuilder` as a new world in a single call (combines ``begin_world`` / :meth:`~newton.ModelBuilder.add_builder` / ``end_world``):

.. testcode::

   import newton

   # Build a simple two-link arm
   arm = newton.ModelBuilder()
   link0 = arm.add_link(mass=1.0)
   j0 = arm.add_joint_fixed(parent=-1, child=link0)
   link1 = arm.add_link(mass=1.0)
   j1 = arm.add_joint_revolute(parent=link0, child=link1)
   arm.add_articulation(joints=[j0, j1])
   arm.add_shape_box(body=link0, hx=0.1, hy=0.1, hz=0.1)
   arm.add_shape_box(body=link1, hx=0.1, hy=0.1, hz=0.1)

   # Create a scene with two instances of the same arm
   scene = newton.ModelBuilder()
   scene.add_ground_plane()
   scene.add_world(arm)
   scene.add_world(arm)

   multi_arm_model = scene.finalize()
   print("world_count:", multi_arm_model.world_count)

.. testoutput::

   world_count: 2

**replicate**: creates ``N`` copies of a builder, each as its own world, with optional spatial offsets.

.. tip::
   Using physical ``spacing`` to separate replicated worlds moves bodies away from the origin,
   which can reduce numerical stability. For visual separation, prefer using viewer-level world
   offsets (e.g. ``viewer.set_world_offsets()``) while keeping ``spacing=(0, 0, 0)`` so that all
   worlds remain at the origin in the physics simulation.

.. testcode::

   import newton

   arm = newton.ModelBuilder()
   link0 = arm.add_link(mass=1.0)
   j0 = arm.add_joint_fixed(parent=-1, child=link0)
   link1 = arm.add_link(mass=1.0)
   j1 = arm.add_joint_revolute(parent=link0, child=link1)
   arm.add_articulation(joints=[j0, j1])

   scene = newton.ModelBuilder()
   scene.add_ground_plane()
   scene.replicate(arm, world_count=4, spacing=(2.0, 0.0, 0.0))

   replicated_model = scene.finalize()
   print("world_count:", replicated_model.world_count)
   print("body_count:", replicated_model.body_count)

.. testoutput::

   world_count: 4
   body_count: 8

.. important::
   Call :meth:`~newton.ModelBuilder.approximate_meshes` on the sub-builder
   **before** passing it to :meth:`~newton.ModelBuilder.replicate`.
   Replication copies mesh references across worlds, so approximating first
   produces a single simplified copy shared by all worlds; approximating
   afterwards allocates one copy per replicated shape.

.. testcode::

   import newton

   arm = newton.ModelBuilder()
   link = arm.add_link(mass=1.0)
   mesh = newton.Mesh.create_box(0.5, 0.5, 0.5, compute_inertia=False)
   arm.add_shape_mesh(body=link, mesh=mesh)
   arm.approximate_meshes(method="convex_hull")

   scene = newton.ModelBuilder()
   scene.replicate(arm, world_count=4)

   replicated_model = scene.finalize()
   print("world_count:", replicated_model.world_count)

.. testoutput::

   world_count: 4


.. _implicit-mpm-worlds:

Implicit MPM worlds
~~~~~~~~~~~~~~~~~~~

.. experimental::

    Independent per-world Implicit MPM simulation, collider filtering, reset,
    and graph-capture behavior may change without prior notice.

For models with more than one world, :class:`~newton.solvers.SolverImplicitMPM`
retains the legacy shared Warp FEM topology by default. Set
``SolverImplicitMPM.Config.separate_worlds = True`` to opt into independent FEM
environments. In isolated mode, worlds can occupy identical physics-space
coordinates without sharing grid mass, momentum, stress, or collider response.

Every MPM particle in an isolated multi-world model must belong to a local
world; global MPM particles are rejected. In this mode, world-local colliders
affect only their assigned world, and body-backed colliders inherit their
body's world. Global static colliders and global colliders backed by bodies
marked :attr:`~newton.BodyFlags.KINEMATIC` affect every world. A global
collider backed by a dynamic body is rejected and must instead use replicated
or otherwise world-local bodies. Custom meshes passed to
:meth:`~newton.solvers.SolverImplicitMPM.setup_collider` are global by default;
use ``collider_world_ids`` to assign them to specific worlds.

Dense and fixed grids use common physical bounds for all environments. Sparse
grids instead store each world's active local voxels, so they are preferable
when particle bounds differ substantially between worlds or worlds are far
apart in physical coordinates. Dense-grid bounds are still recomputed through
a host read and dense steps remain eager. Leaving ``separate_worlds=False``
selects the default shared FEM topology; the isolation and capacity rules below
apply only after opting into ``separate_worlds=True``.

Sparse capacity and environment packing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Implicit MPM has two sparse allocation modes:

- An **allocating sparse grid** creates NanoVDB topology as needed. This is the
  behavior when ``max_active_cell_count <= 0`` or when padding or the selected
  basis topology cannot support an in-place rebuild. It
  remains available for eager stepping but is not outer-capture safe.
- A **capacity-bounded rebuildable sparse grid** reserves persistent topology
  storage and rebuilds its active contents in place. It requires
  ``max_active_cell_count > 0``, ``grid_padding == 0``, a ``"Q1"`` velocity
  basis, compatible strain and collider bases, and rebuildable NanoVDB support
  from the installed Warp artifact.

For a multi-world solver, ``max_active_cell_count`` is one total capacity
shared by the batched FEM geometry partition, not a separate quota for every
world. Estimate the worst useful active-cell count per world, multiply it by
the number of worlds, and pass that total. Worlds may use the shared reserve
unevenly. The sparse topology contains active local voxels and guard regions;
it does not allocate all voxels spanning the physical distance between worlds.

Rebuildable NanoVDB topology has four separately configurable capacity levels:
active voxels, leaf nodes, lower internal nodes, and upper internal nodes.
Configure the latter three with ``max_leaf_node_count``,
``max_lower_node_count``, and ``max_upper_node_count`` when application-level
spatial bounds provide known limits. These are also total capacities shared by
all worlds. Resolved capacities must satisfy
``upper <= lower <= leaf <= active``. The default ``-1`` keeps automatic
sizing: leaf capacity equals ``max_active_cell_count``; automatic lower
capacity is capped by the resolved leaf capacity; and automatic upper capacity
is capped by the resolved lower capacity. Lower and upper estimates reserve up
to 16 times the initial packed topology at their respective hierarchy levels,
with minimum estimates of eight lower and four upper nodes when the finer-level
capacity permits. Explicit values are not adjusted, so an explicitly
inconsistent hierarchy raises during construction.

Node capacity depends on spatial distribution, not just active-cell count. A
single escaped particle can enter a new internal node without materially
changing the active-voxel count; an upper node spans 4096 voxels along each
axis. Increasing ``max_active_cell_count`` alone therefore does not resolve an
internal-node overflow. Prefer application bounds or outflow handling, then
set explicit node capacities from that bounded domain. Arbitrarily scattered
particles require worst-case node capacities as large as the active-cell
capacity, which is usually impractical because upper nodes are comparatively
expensive.

When Newton generates packed-environment offsets, a rebuild recomputes their
values on the device from current per-world bounds. This lets independently
moving worlds remain disjoint in packed NanoVDB coordinates. Offset values and
active counts may change between steps, but environment count, offset-array
shape, guard width, alignment, storage addresses, and declared capacities are
structural invariants. Explicit caller-provided offsets do not move
automatically, so the caller must keep their packed bounds disjoint.

Outer graph-capture lifecycle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An isolated multi-world step can be captured in an outer graph when all of
the following conditions hold:

- Although Warp supports CPU graphs generally, Implicit MPM graph replay is
  currently supported only on CUDA, with conditional graph-node support and an
  enabled memory pool. Rebuildable sparse-grid capture specifically requires
  CUDA-safe topology rebuilds.
- Construct :class:`~newton.solvers.SolverImplicitMPM` with
  ``enable_timers=False``. Section timers synchronize the device and therefore
  cannot run during outer capture.
- ``SolverImplicitMPM.Config.separate_worlds`` is ``True``.
- ``SolverImplicitMPM.Config.grid_type`` is either ``"fixed"`` with bounds
  created before capture or ``"sparse"`` in the capacity-bounded rebuildable
  mode described above.
- ``SolverImplicitMPM.Config.max_active_cell_count`` is positive and large
  enough for every replay. Any explicit leaf, lower, and upper node capacities
  are also large enough for the bounded particle distribution. Active subsets
  may change within those fixed total capacities.
- Every selected sparse basis provides rebuildable topology.

Check :attr:`~newton.solvers.SolverImplicitMPM.supports_graph_capture`
before recording, then call
:meth:`~newton.solvers.SolverImplicitMPM.prepare_graph_capture` to
materialize persistent buffers without advancing the simulation. Model
topology, world and particle ordering, array shapes, fixed-grid bounds,
capacities, solver configuration, captured state-buffer sequence, and captured
step arguments must remain unchanged across replays. Particle state, active
cells, and automatically generated sparse packing offsets may change within
those fixed structures.

If the application alternates two state buffers, record both directions in one
graph so every replay finishes with the buffers in the same roles:

.. code-block:: python

   solver.prepare_graph_capture()
   with wp.ScopedCapture(device=model.device) as capture:
       solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
       solver.step(state_1, state_0, control=None, contacts=None, dt=dt)

   wp.capture_launch(capture.graph)
   solver.check_status()

Call :meth:`~newton.solvers.SolverImplicitMPM.check_status` at an explicit
safe host boundary after replay. Coupled solvers forward this check recursively
to their entries. For rebuildable sparse grids, capacity failures accumulate in
a sticky device status so a later successful rebuild cannot hide an earlier overflow.
The check synchronizes and raises a descriptive capacity error. A valid solver
reset establishes a new status boundary.

Keep resets outside the captured graph. Calling
:meth:`~newton.solvers.SolverImplicitMPM.reset` with ``world_mask`` restores MPM
history and grid-backed warm starts only for selected isolated worlds and
preserves unselected worlds; a valid reset also clears sparse rebuild status.
The mask has ``model.world_count + 1`` entries: the first entries select local
worlds by index, while the final entry selects global objects assigned to world
``-1``.
A masked reset on a shared multi-world grid is rejected because one warm-start
node may combine contributions from several worlds. Use ``separate_worlds=True``
for selective reset, or omit ``world_mask`` to reset the shared grid in full.
Recapture after changing any structural input, capacity, buffer sequence, or
time step.

Allocating sparse and dense grids remain eager. Fixed and capacity-bounded
rebuildable sparse grids support outer capture on CUDA with valid rheology
solver configurations, including the default Q1 Gauss-Seidel solver.

.. _Per-world gravity:

Per-World Gravity
-----------------

Each world can have its own gravity vector, which is useful for simulating different environments
(e.g., Earth gravity in one world, lunar gravity in another).
Per-world gravity can be configured at build time via the ``gravity`` argument of :meth:`~newton.ModelBuilder.begin_world`,
or modified at runtime via :meth:`~newton.Model.set_gravity`:

.. note::
   Global entities (world index ``-1``) use the gravity of world ``0``.
   Keep this in mind when mixing global and world-specific entities with different gravity vectors.

.. testcode::

   import warp as wp
   import newton

   robot_builder = newton.ModelBuilder()
   robot_builder.add_body(mass=1.0)

   scene = newton.ModelBuilder()
   scene.add_world(robot_builder)
   scene.add_world(robot_builder)
   model = scene.finalize()

   # Set different gravity for each world
   model.set_gravity((0.0, 0.0, -9.81), world=0)  # Earth
   model.set_gravity((0.0, 0.0, -1.62), world=1)  # Moon

   print("Gravity shape:", model.gravity.numpy().shape)

.. testoutput::

   Gravity shape: (2, 3)


.. _World-entity partitioning:

World-Entity GPU Thread Partitioning
------------------------------------

Another important use of world grouping is to facilitate partitioning of GPU thread grids according to both world indices and the number of entities per world, i.e. into 2D world-entity grids.

For example:

.. code-block:: python

   import warp as wp
   import newton

   @wp.kernel
   def world_body_2d_kernel(
       body_world_start: wp.array[wp.int32],
       body_qd: wp.array[wp.spatial_vectorf],
   ):
       world_id, body_world_id = wp.tid()
       world_start = body_world_start[world_id]
       num_bodies_in_world = body_world_start[world_id + 1] - world_start
       if body_world_id < num_bodies_in_world:
           global_body_id = world_start + body_world_id
           twist = body_qd[global_body_id]
           # ... perform computations on twist ...

   # Create model with multiple worlds
   builder = newton.ModelBuilder()
   # ... add entities to multiple worlds ...
   model = builder.finalize()

   # Define number of entities per world (e.g., bodies)
   body_world_start = model.body_world_start.numpy()
   num_bodies_per_world = [
       body_world_start[i + 1] - body_world_start[i]
       for i in range(model.world_count)
   ]

   # Launch kernel with 2D grid: (world_count, max_entities_per_world)
   state = model.state()
   wp.launch(
       world_body_2d_kernel,
       dim=(model.world_count, max(num_bodies_per_world)),
       inputs=[model.body_world_start, state.body_qd],
   )

This kernel thread partitioning allows each thread to uniquely identify both the world it is operating on (via ``world_id``) and the relative entity index w.r.t that world (via ``body_world_id``).
The world-relative index is useful in certain operations such as accessing the body-specific column of constraint Jacobian matrices in maximal-coordinate formulations, which are stored in contiguous blocks per world.
This relative index can then be mapped to the global entity index within the model by adding the corresponding starting index from the ``*_world_start`` arrays.

Note that in the simpler case of a homogeneous model consisting of identical worlds, the ``max(num_bodies_per_world)`` reduces to a constant value, and this effectively becomes a *batched* operation.
For the more general heterogeneous case, the kernel needs to account for the varying number of entities per world, and an important pattern arises w.r.t 2D thread indexing and memory allocations that applies to all per-entity and per-world arrays.

Essentially, ``sum(num_bodies_per_world)`` equals the total number of *world-local* bodies (i.e. ``body_world_start[-2] - body_world_start[0]``), which excludes any global entities (world index ``-1``).
Note that ``model.body_count`` may be larger than this sum when global bodies are present, since it includes both world-local and global entities (see :attr:`~newton.Model.body_world_start` for the explicit distinction).
The maximum ``max(num_bodies_per_world)`` determines the second dimension of the 2D thread grid used to launch the kernel.
However, since different worlds may have different numbers of bodies, some threads in the 2D grid will be inactive for worlds with fewer bodies than the maximum.
Therefore, kernels need to check whether the relative entity index is within bounds for the current world before performing any operations, as shown in the example above.

This pattern of computing ``sum`` and ``max`` of per-world entity counts provides a consistent way to handle memory allocations and thread grid dimensions for heterogeneous multi-world simulations in Newton.


See Also
--------

* :class:`~newton.ModelBuilder`
* :class:`~newton.Model`
* :meth:`~newton.ModelBuilder.begin_world`
* :meth:`~newton.ModelBuilder.end_world`
* :meth:`~newton.ModelBuilder.add_world`
* :meth:`~newton.ModelBuilder.replicate`
* :meth:`~newton.Model.set_gravity`
