.. _Worlds:

Worlds
======

Newton enables multiple independent simulations, referred to *worlds*, within a single :class:`~newton.Model` object.
Each *world*, thus provides an index-based grouping of all primary simulation entities such as particles, bodies, shapes, joints, articulations and equality constraints.


Overview
--------

GPU-accelerated operations in Newton often involve parallelizing over an entire set of model entities, e.g. bodies, shapes or joints, without needing to consider which specific world they belong to.
However, some operations such as those part of Collision Detection (CD), can exploit world-based grouping to effectively filter-out potential collisions between shapes that belong to different worlds.
Moreover, world-based grouping can also facilitate partitioning of thread grids according to both world indices and the number of entities per world.
Such operations facilitate support for simulating multiple, and potentially heterogeneous, worlds defined within a :class:`~newton.Model` instance.
Lastly, world-based grouping also enables selectively operating on only the entities that belong to a specific world, i.e. masking, as well as partitioning of the :class:`~newton.Model` and :class:`~newton.State` data.

.. note::
   Support for fully heterogeneous simulations is still under active development and quite experimental.
   At present time, although the :class:`~newton.ModelBuilder` and :class:`~newton.Model` objects support instantiating worlds with different disparate entities, not all solvers are able to simulate them.
   Moreover, the selection API still operates under the assumption of model homogeneity, but this is expected to also support heterogeneous simulations in the near future.

.. _World assignment:

World Assignment
----------------

World assignment occurs when entities are added to an instance of :class:`~newton.ModelBuilder`, using one of the entity-specific methods such as :meth:`~newton.ModelBuilder.add_body`,
:meth:`~newton.ModelBuilder.add_joint`, :meth:`~newton.ModelBuilder.add_shape` etc, and this can either be global (world index ``-1``) or specific to a particular world (world index ``0, 1, 2, ...``).
When entities are added before the first call to :meth:`~newton.ModelBuilder.begin_world`, or the last call to :meth:`~newton.ModelBuilder.end_world`, they are assigned to the global world (index ``-1``).
Conversely, entities can be assigned to specific worlds when added between calls to :meth:`~newton.ModelBuilder.begin_world` and :meth:`~newton.ModelBuilder.end_world`.
Each entity added between these calls is assigned the current world index. The following example illustrates how to create two different worlds within a single model:

.. code-block::

   import newton

   builder = newton.ModelBuilder()

   # Global entity at front (world -1)
   ground = builder.add_ground_plane()

   # World 0
   builder.begin_world()
   body00 = builder.add_body(mass=1.1, ...)
   body01 = builder.add_body(mass=1.2, ...)
   shape00 = builder.add_shape_sphere(body=body00, ...)
   shape02 = builder.add_shape_sphere(body=body01, ...)
   builder.end_world()

   # World 1
   builder.begin_world()
   body10 = builder.add_body(mass=1.0, ...)
   body11 = builder.add_link(mass=2.0, ...)
   joint11 = builder.add_joint_revolute(parent=body10, child=body11, ...)
   shape10 = builder.add_shape_box(body=body10, ...)
   shape11 = builder.add_shape_box(body=body11, ...)
   builder.end_world()

   # Global entity at back (world -1)
   static_box = builder.add_shape_box(...)

   # Finalize model
   model = builder.finalize()

In this example, we are creating a model with two worlds (world ``0`` and world ``1``) containing different bodies, shapes and joints, as well as the global ground plane entity (with world index ``-1``).


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
- Equality Constraints: :attr:`~newton.Model.equality_constraint_world`

For the example above, the corresponding world grouping arrays would be as follows:

.. code-block::

   print("Body worlds:", model.body_world.numpy())  # Example: Body worlds: [0  0  1  1]
   print("Shape worlds:", model.shape_world.numpy())  # Example: Shape worlds: [-1  0  0  1  1  -1]
   print("Joint worlds:", model.joint_world.numpy())  # Example: Joint worlds: [0  0  1  1]


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
- Equality Constraints: :attr:`~newton.Model.equality_constraint_world_start`

To handle the special case of joint entities, that vary in the number of DOFs, coordinates and constraints, the model also provides arrays that store the per-world starting indices in these specific dimensions:
- Joint DOFs: :attr:`~newton.Model.joint_dof_world_start`
- Joint Coordinates: :attr:`~newton.Model.joint_coord_world_start`
- Joint Constraints: :attr:`~newton.Model.joint_constraint_world_start`

All :attr:`~newton.Model.world_*_start` arrays adopt a special format that facilitates accounting of the total number of entities in each world as well as the global world (index ``-1``) at the front and back of each per-entity array such as :attr:`~newton.Model.body_world`.
Specifically, each :attr:`~newton.Model.world_*_start` array contains ``world_count + 2`` entries, with the first ``world_count`` entries corresponding to starting indices of each ``world >= 0`` world,
the second last entry corresponds to the starting index of the global entities at the back (world index ``-1``), and the last entry corresponding to total number of entities or dimensions in the model.

With this format, we can easily compute the number of entities per world by computing the difference between consecutive entries in these arrays (since they are essentially cumulative sums),
as well as the total number of global entities by summing the first entry with the difference of the last two.

For the previous example, we can compute the per-world shape counts as follows:

.. code-block::

   # Total number of worlds
   print("model.world_count :", model.world_count)  # In this example, we have worlds 0 and 1, and world_count = 2

   # Shape start indices per world
   # Entries: [start_world_0, start_world_1, start_global_back, total_shapes]
   shape_start = model.shape_world_start.numpy()
   print("Shape starts: ", shape_start)
   # Output: Shape starts: [1  3  5  6]  # 1 global shape at front, 2 shapes in world 0, 2 shapes in world 1, 1 global shape at back, total 6 shapes

   # Compute per-world body counts
   world_shape_counts = [shape_start[i+1] - shape_start[i] for i in range(model.world_count)]
   global_shape_count = shape_start[-1] - shape_start[-2] + shape_start[0]  # Global shapes at front and back

   # Print shape counts
   print("Shape counts per world: ", world_shape_counts)  # Output: Shape counts per world: [2, 2]
   print("Global shape count: ", global_shape_count)      # Output: Global shape count: 2


.. _World-entity partitioning:

World-Entity GPU Thread Partitioning
------------------------------------

Another important use of world grouping is to facilitate partitioning of GPU thread grids according to both world indices and the number of entities per world, i.e. into 2D world-entity grids.

For example:

.. code-block::

   import warp as wp
   import newton

   @wp.kernel
   def 2d_world_body_example_kernel(
       body_world_start: wp.array(dtype=wp.int32),
       body_world: wp.array(dtype=wp.int32),
       body_twist: wp.array(dtype=wp.spatial_vectorf),
   ):
       world_id, body_world_id = wp.tid()  # 2D world-entity grid
       # Perform operations specific to the world and entity here
       world_start = body_world_start[world_id]
       num_bodies_in_world = body_world_start[world_id + 1] - world_start
       if body_world_id < num_bodies_in_world:
          global_body_id = world_start + body_world_id
          # Access body-specific data using global_body_id
          twist = body_twist[global_body_id]
          # ... perform computations on twist ...

   # Create model with multiple worlds
   builder = newton.ModelBuilder()
   # ... add entities to multiple worlds ...
   model = builder.finalize()

   # Define number of entities per world (e.g., bodies)
   body_world_start = model.body_world_start.numpy()
   num_bodies_per_world = [body_world_start[i+1] - body_world_start[i] for i in range(model.world_count)]

   # Launch kernel with 2D grid: (world_count, max_num_entities)
   wp.launch(2d_world_body_example_kernel, dim=(model.world_count, max(num_bodies_per_world)), ...)

This kernel thread partitioning allows each thread to uniquely identify both the world it is operating on (via ``world_id``) and the relative entity index w.r.t that world (via ``entity_id``).
The world-relative ``entity_id`` index is useful in certain operations such as accessing the body-specific column of constraint Jacobian matrices in maximal-coordinate formulations, which are stored in contiguous blocks per world.
This relative index can then be mapped to the global entity index within the model by adding the corresponding starting index from the :attr:`~newton.Model.world_*_start` arrays.

Note that in the simpler case of a homogeneous model consisting of identical worlds, the ``max(num_bodies_per_world)`` reduces to a constant value, and this effectively becomes a *batched* operation.
For the more general heterogeneous case, the kernel needs to account for the varying number of entities per world, and an important pattern arises w.r.t 2D thread indexing and memory allocations that applies to all per-entity and per-world arrays.

Essentially, the sum ``sum(num_bodies_per_world)`` will always equal the total number of bodies in the model ``model.num_bodies`` corresponding memory allocated for per-body arrays (i.e. when multiplied by the size of the relevant ``dtype``),
and the maximum ``max(num_bodies_per_world)`` will determine the second dimension of the 2D thread grid used to launch the kernel.
However, since different worlds may have different number of bodies, some threads in the 2D grid will be inactive for worlds with fewer bodies than the maximum.
Therefore, kernels need to check whether the relative entity index is within bounds for the current world before performing any operations, as shown in the example above.

This pattern of computing ``sum_of_num_*`` and ``max_of_num_*`` thus provides a consistent way to handle memory allocations and thread grid dimensions for heterogeneous multi-world simulations in Newton.


See Also
--------

* :class:`~newton.ModelBuilder`
* :class:`~newton.Model`