ModelView Metadata Follow-up Plan
=================================

This is an internal implementation plan for the experimental coupled-solver
framework. It is intentionally not included in the public documentation
toctree.

``ModelView`` remains public experimental API, but its projection machinery
must stop inferring attribute semantics from names, array lengths, or concrete
solver namespaces. This is an implementation plan, not a proposal to change
the public ``newton.solvers.experimental.coupled`` namespace.

Current debt
~~~~~~~~~~~~

The builder already knows most of the information needed for generic
projection. In particular, :class:`~newton.ModelBuilder.CustomAttribute`
records ``frequency``, ``assignment``, and ``references``. Finalization only
copies the first two fields to ``Model.attribute_frequency`` and
``Model.attribute_assignment``; the reference target is lost.

The coupled solver currently compensates for that missing information in five
ways:

- ``ModelView._frequency_count_for_attribute()`` has a table of unregistered
  core arrays and ``_count_limited_attribute()`` has separate tables for
  offset, world-start, and color-group data.
- ``SolverCoupled._select_compact_prefix_attributes()`` scans ``dir(model)``
  and treats names beginning with ``body_``, ``joint_``, ``shape_``, and other
  prefixes as indexed data when their lengths happen to match.
- ``SolverCoupled._compact_equality_constraint_order()`` discovers MuJoCo
  equality rows by matching the ``equality_constraint`` suffix and then looks
  for four exact attribute names.
- ``SolverCoupled._compact_custom_attribute_namespaces()`` clones namespaces
  by scanning their attributes and remaps only the four known MuJoCo equality
  reference arrays.
- Topology and derived data such as ``joint_q_start``, ``articulation_start``,
  ``body_shapes``, color groups, collision pairs, and per-world start arrays
  are handled by unrelated one-off code paths.

Target metadata
~~~~~~~~~~~~~~~

Keep the existing public builder declarations. During
:meth:`~newton.ModelBuilder.finalize`, normalize them into a private,
read-only descriptor registry on the finalized model. Each descriptor should
contain:

``name``
  The full key, including a namespace when present, for example
  ``"mujoco:equality_constraint_body1"``.
``frequency``
  The row domain, using :class:`~newton.Model.AttributeFrequency` or a full
  custom-frequency key such as ``"mujoco:equality_constraint"``.
``assignment``
  Whether the value belongs to ``Model``, ``State``, ``Control``, or
  ``Contacts``. Finalized backing values still live on the model, so indexed
  backing values for all assignments must be projected. This ensures that
  ``ModelView.state()`` and control/contact allocation clone view-local data
  instead of the first ``N`` parent rows.
``references``
  The domain indexed by each non-negative integer value, such as ``BODY``,
  ``JOINT``, or ``"mujoco:tendon"``. Negative values remain sentinels and are
  never passed through an index map.
``projection``
  One of ``select``, ``reference``, or a private derived-handler key. Most
  arrays use ``select``; an indexed integer array with ``references`` also uses
  the generic reference-remapping pass. Only structural data that cannot be
  represented as row selection uses a handler.

Do not replace or reinterpret ``Model.attribute_frequency`` and
``Model.attribute_assignment`` in this change; existing callers may rely on
them. The private registry is a normalized companion assembled from the same
declarations. Core fields need descriptors too, including currently
unregistered arrays such as ``particle_q``, ``body_world``, and
``shape_world``. Their declarations should live in one table near ``Model``
construction instead of being repeated in ``ModelView`` and
``SolverCoupled``.

The builder must carry ``CustomAttribute.references`` into this registry.
Reference strings already supported by the builder -- body, particle, shape,
joint, joint DOF, joint coordinate, articulation, world, deformable element,
and custom-frequency keys -- should be normalized once during finalization.
An unknown reference domain must fail finalization, rather than leaving a
partially remappable model. A reference-bearing attribute must also use a
scalar integral dtype; compound or floating-point values need a purpose-built
derived handler instead of being interpreted as ids.

Expected code ownership is:

- ``newton/_src/sim/builder.py`` keeps the public declaration API and passes
  all projection metadata through finalization.
- ``newton/_src/sim/model.py`` defines the private descriptor type, the core
  descriptor table, and the finalized read-only registry.
- ``newton/_src/solvers/coupled/model_view.py`` consumes descriptors for count
  limiting and read-through namespace overlays; its caches and projection
  helpers stay private.
- ``newton/_src/solvers/coupled/solver_coupled.py`` owns generic frequency-map
  construction, gathers, reference remapping, and derived-handler dispatch.
- ``newton/tests/test_coupled_solver.py`` contains end-to-end view projection
  coverage; builder/model metadata retention belongs with the existing custom
  attribute tests.

Generic compaction algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Replace the fixed ``_CompactIndexLists`` fields with a private map from
frequency to an index projection. A projection contains ordered
``local_to_global`` indices and a dense ``global_to_local`` inverse with ``-1``
for hidden entries. Body, joint, joint-coordinate, joint-DOF, shape,
articulation, mimic-constraint, and custom-frequency projections then use the
same representation. ``SolverEntry`` may still expose the maps needed during
state distribution, but construction should have a complete map for every
projected frequency.

Compaction should run in this order:

1. Compute visible body, particle, joint, and shape sets from entry ownership
   and proxy requirements. Preserve the existing homogeneous multi-world
   ordering check; failure to form the same local layout in every world falls
   back to the existing prefix/global view behavior.
2. Derive joint-coordinate and joint-DOF projections from the selected joints'
   ragged ranges. Derive articulation and mimic-constraint rows from their
   declared references.
3. For each custom frequency, inspect all attributes on that frequency that
   declare ``references``. Keep a row when every non-negative reference in the
   row is visible in its target projection. A negative reference is optional
   and does not remove the row. Resolve custom-to-custom dependencies to a
   fixed point: removing a row from one frequency may remove rows that
   reference it from another frequency. This also handles cycles without
   depending on registration order. If a custom frequency has no
   reference-bearing attribute, retain its complete identity projection
   because there is no metadata-supported basis for dropping rows.
4. Order retained custom rows by their declared world-reference attribute, if
   present, and rebuild ``<frequency>_world_start``. Do not infer either the
   namespace or entity type from a frequency suffix.
5. Select every model-assigned attribute using its frequency projection. Then
   remap attributes with a reference target through that target's inverse map.
   A non-negative reference missing from the target projection is a
   construction error naming the attribute, row, and referenced id; it must
   not become a ``KeyError`` or silently remain global.
6. Create private read-through namespace overlays containing only registered
   projected attributes and synthesized count/start metadata. This replaces
   the current ``dir(namespace)`` copy and guarantees that view-local arrays do
   not mutate the parent namespace.

``preserve_shape_ids=True`` remains an explicit exception to dense shape
compaction. Its shape projection is the global identity map, while hidden
shapes have their body reference replaced by the existing ``-1`` sentinel and
their collision flags disabled. ``preserve_shape_ids=False`` uses the normal
dense shape map and remaps shape pairs and ``body_shapes`` into local ids.

Derived handlers
~~~~~~~~~~~~~~~~

Metadata cannot turn every structure into a simple gather. Keep a small
private handler registry, keyed by descriptor rather than detected from a
name suffix, for these cases:

- Ragged offsets: rebuild ``joint_q_start`` and ``joint_qd_start`` from the
  selected joint widths.
- Articulation extents: rebuild ``articulation_start``, ``articulation_end``,
  ``max_joints_per_articulation``, and ``max_dofs_per_articulation``.
- World partitions: rebuild the core and custom ``*_world_start`` arrays from
  projected world ids, including the leading global ``-1`` bucket.
- Adjacency and grouped indices: remap ``body_shapes``, particle/body color
  groups, shape collision-filter pairs, and explicit contact pairs.
- Derived topology: recompute ``joint_ancestor`` after remapping joint parents
  and children instead of treating it as an independent source array.

The handler registry is intentionally private. Adding a normal indexed or
reference-valued custom attribute must not require a new handler or any edit in
the coupled solver.

Migration stages
~~~~~~~~~~~~~~~~

1. **Retain metadata without changing behavior.** Add the finalized private
   descriptors, populate them for core and custom attributes, and test that
   ``references`` survives builder finalization. Keep the existing compaction
   code active in this stage.
2. **Introduce the generic projector.** Build frequency maps and project plain
   core arrays through descriptors. For one transition stage, compare the new
   projected arrays and counts with the existing path in tests.
3. **Move reference remapping.** Register core references such as
   ``joint_parent -> BODY``, ``joint_child -> BODY``,
   ``joint_articulation -> ARTICULATION``, ``shape_body -> BODY``, and mimic
   joint references. Use the same path for custom body, particle, joint, and
   custom-frequency references.
4. **Move structural data to handlers.** Migrate starts, adjacency, color
   groups, and pair collections. Remove the old special cases only after each
   handler has direct regression coverage.
5. **Delete inference.** Remove ``_select_compact_prefix_attributes()``, the
   unregistered-core frequency table in ``model_view.py``, equality-specific
   discovery/remapping, and all ``dir(model)``/``dir(namespace)`` projection
   scans. An indexed model attribute lacking a descriptor should now produce a
   precise construction error.

Tests and completion criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extend ``newton/tests/test_coupled_solver.py`` with parameterized cases that
exercise:

- a non-prefix body/joint selection for each built-in reference target;
- custom rows containing body, particle, joint, world, and another custom-row
  reference, including ``-1`` sentinels and missing-positive-reference errors;
- custom-frequency names whose suffix does not match a built-in entity name,
  proving that no name-based dispatch remains;
- multiple worlds, a leading global bucket, rebuilt counts/start arrays, and
  rejection of heterogeneous per-world selections;
- both shape-id policies, ragged joint coordinate/DOF layouts, articulations,
  mimic constraints, VBD color groups, and collision/contact pair remapping;
- model, state, and namespace immutability after creating and mutating multiple
  views from the same parent model; and
- a newly registered model attribute projecting correctly with no coupled
  solver changes, plus an intentionally undeclared indexed field failing with
  its full attribute name.

The migration is complete when builder metadata is the single declaration of
frequency/assignment/reference semantics, adding an ordinary custom reference
array requires no change under ``newton/_src/solvers/coupled/``, and no
compaction path guesses semantics from names or matching array lengths.
