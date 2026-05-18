.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

Coupled Solvers
===============

.. warning::

   Coupled solvers are experimental. The public API currently lives under
   :mod:`newton.solvers.coupled_experimental`, and both the API shape and the
   numerical defaults may change while the feature is refined.

Newton's coupled-solver framework lets one simulation step be split across
multiple solver backends while those backends still exchange forces, poses, and
constraint information through a shared :class:`Model`. This is useful when a
scene combines material models or algorithms that are best handled by different
solvers: for example a MuJoCo or Kamino rigid mechanism coupled to VBD cloth,
XPBD particles coupled to MPM material, or rigid bodies connected to particles
through an ADMM constraint.

The framework is exposed as an experimental namespace rather than as flat
symbols on :mod:`newton.solvers`. Import the coupled solver types directly from
that namespace:

.. code-block:: python

   from newton.solvers import SolverMuJoCo, SolverVBD
   from newton.solvers.coupled_experimental import (
       SolverAdmmCoupled,
       SolverCoupled,
       SolverProxyCoupled,
   )

The main public types are:

- :class:`newton.solvers.coupled_experimental.ModelView`: a view-local overlay
  on a shared :class:`Model`.
- :class:`newton.solvers.coupled_experimental.CouplingInterface`: the hook
  protocol implemented by solvers that need custom coupled behavior.
- :class:`newton.solvers.coupled_experimental.SolverCoupled`: the shared base
  for partitioning models, distributing state, stepping entries, and
  reconciling results.
- :class:`newton.solvers.coupled_experimental.SolverProxyCoupled`: a lagged or
  staggered proxy coupling wrapper.
- :class:`newton.solvers.coupled_experimental.SolverAdmmCoupled`: a fixed
  iteration ADMM coupling wrapper for model-derived joints, attachments, and
  contacts.

Shared Model, Entry Views, and Ownership
----------------------------------------

Coupled simulations start from a single :class:`Model`. Each sub-solver receives
a :class:`~newton.solvers.coupled_experimental.ModelView` rather than the raw
model. A view delegates reads to the parent model until the coupler or user
applies a view-local override. The important idea is that sub-solvers can see
the same model topology while owning only the bodies, particles, joints, or
shapes assigned to their entry.

A :class:`~newton.solvers.coupled_experimental.SolverCoupled.Entry` describes
one sub-solver:

.. code-block:: python

   entry = SolverCoupled.Entry(
       name="soft",
       solver=SolverVBD,
       bodies=soft_body_ids,
       particles=cloth_particle_ids,
       shapes=cloth_shape_ids,
       solver_kwargs={"iterations": 20},
       substeps=2,
   )

The entry lists the objects the sub-solver owns. During construction,
``SolverCoupled`` creates a model view for every entry, deactivates non-owned
dynamic endpoints where appropriate, constructs the sub-solver with
``solver(model=view, **solver_kwargs)``, and keeps per-entry input and output
states. After a top-level step, only owned outputs are reconciled back into the
caller's shared ``state_out``. This prevents two sub-solvers from overwriting
the same body or particle unless an explicit coupling algorithm is responsible
for arbitration.

The shared base also manages:

- per-entry substeps, so one solver can take smaller time intervals than
  another;
- copying public force input from ``State`` and ``Control`` into entry-local
  state;
- entry-local collision visibility and shape ownership;
- input-state notifications for solvers with private history buffers;
- fallback effective-mass estimates from public model mass and inertia arrays.

``ModelView`` mutator methods use copy-on-write semantics. Methods such as
``zero_body_mass()``, ``scale_body_mass()``, ``mark_proxy_bodies()``,
``mark_proxy_particles()``, ``zero_particle_mass()``, and
``scale_particle_mass()`` clone and override arrays in the view without changing
the parent model. Direct writes through a returned Warp array are not
intercepted, so view-local edits should use the explicit mutators.

Coupling Hooks
--------------

Some solvers keep important state outside the public :class:`State` arrays or
can report interface forces more accurately than a generic momentum fallback.
Those solvers implement
:class:`newton.solvers.coupled_experimental.CouplingInterface` hooks. The
coupler detects hook methods by name. Missing hooks use generic fallbacks, while
hooks listed in ``coupling_unsupported`` make the coupler raise
:class:`NotImplementedError` instead of silently using an invalid path.

The protocol currently covers these concepts:

- ``NOTIFY_INPUT_STATE_UPDATE`` tells a solver that public state arrays were
  changed by the coupler. VBD uses this to realign private previous-pose state
  after proxy synchronization or ADMM iteration restarts. MPM uses it to keep
  collider caches consistent.
- ``BODY_PROXY_REWIND_VELOCITY`` and ``PARTICLE_PROXY_REWIND_VELOCITY`` let a
  destination solver prepare proxy velocities before a lagged proxy pass.
- ``BODY_PROXY_HARVEST`` and ``PARTICLE_PROXY_HARVEST`` let a destination solver
  report feedback forces from solver-native contact or transfer data.
- ``PROXY_CONTACT_PREPARE`` lets a destination solver filter or prepare
  proxy-local contacts before its step.
- ``EFFECTIVE_MASS_DIAGONAL`` and ``EFFECTIVE_MASS_BLOCK`` let a solver provide
  endpoint effective mass instead of using raw model mass and inertia.

Force injection itself is not a hook. Couplers write into public
``state.body_f``, ``state.particle_f``, and ``control.joint_f`` buffers, then
call the normal solver step. Likewise, virtual and proximal mass changes are
applied to a ``ModelView`` and refreshed through the usual
``notify_model_changed()`` path when a solver must rebuild private caches.

Proxy Coupling
--------------

Proxy coupling represents an endpoint owned by one solver as a proxy endpoint in
another solver. The source solver owns the real object. The destination solver
receives a proxy body or proxy particle with scaled virtual inertia, solves its
own local problem against that proxy, then returns feedback to the source on a
later pass or iteration.

This is a good match for coupling algorithms that are naturally one-way within a
substep but can converge through repeated lagged iterations. Examples include a
rigid body acting as a proxy collider inside a soft-body solve, XPBD particles
driving MPM transfer particles, or VBD reporting contact forces back to a rigid
source body.

A proxy pair is declared with
:class:`newton.solvers.coupled_experimental.SolverProxyCoupled.Proxy`:

.. code-block:: python

   solver = SolverProxyCoupled(
       model,
       entries=[rigid_entry, soft_entry],
       coupling=SolverProxyCoupled.Config(
           proxies=[
               SolverProxyCoupled.Proxy(
                   source="rigid",
                   destination="soft",
                   bodies=robot_body_ids,
                   proxy_bodies=robot_proxy_body_ids,
                   particles=(),
                   proxy_particles=(),
                   mass_scale=0.25,
                   mode="lagged",
               )
           ],
           iterations=4,
       ),
   )

``source`` and ``destination`` name entries. ``bodies`` and ``particles`` are
source endpoints. ``proxy_bodies`` and ``proxy_particles`` name the
corresponding destination endpoints. If a proxy list is ``None``, the source
indices are reused in the destination view. ``mass_scale`` scales proxy body
mass/inertia and proxy particle mass in that destination view.

Two proxy modes are available through the ``mode`` string:

- ``LAGGED`` synchronizes the source begin pose and end velocity into the
  destination proxy, rewinds destination proxy velocity by previously applied
  feedback, public force input, and gravity, then steps the destination. This is
  the most common mode for relaxed fixed-point coupling.
- ``STAGGERED`` synchronizes the source end pose and velocity into the
  destination and skips the generic lagged rewind. This is useful when the
  scheduling already gives the destination a current source state.

After the destination step, the coupler harvests feedback. If the destination
solver implements a body or particle harvest hook, that hook can report
contact-native forces or transfer impulses. Otherwise the shared fallback
estimates feedback from proxy momentum change. The fallback is convenient for
simple particle proxy cases, but contact-rich or solver-private interactions are
usually better served by a custom harvest hook.

Proxy-local collision detection is optional. A proxy can provide a
``collision_pipeline`` factory that receives the destination ``ModelView``. If
the factory returns a pipeline, the coupler owns a persistent contact buffer and
refreshes it at ``collide_interval``. If the factory returns ``None`` or no
factory is supplied, the destination solve uses contacts passed to the outer
``step()`` call.

The generic proxy loop currently supports at most two solver entries. Within
that limit, body and particle mappings are grouped by ``(source,
destination)``. One source step and one destination step are performed for each
solver pair and proxy iteration, so a single proxy declaration can carry both
body and particle mappings around the same destination solve.

ADMM Coupling
-------------

ADMM coupling is the symmetric coupling path. Instead of placing a virtual proxy
inside another solver, it constructs interface rows between endpoints owned by
different entries. Each iteration restores entry states, applies a proximal
velocity target when configured, lets sub-solvers advance, solves local
interface rows, updates dual variables, and splats equal and opposite coupling
forces back to endpoint force buffers.

The implemented ADMM wrapper discovers constraint rows from the shared model and
enables contact rows through explicit
:class:`newton.solvers.coupled_experimental.SolverAdmmCoupled.ContactPair`
objects. It does not currently accept arbitrary user-authored endpoint records
as public API. Supported row sources are:

- cross-solver model joints;
- custom body-particle attachment attributes;
- internally detected rigid-rigid, rigid-particle, and particle-particle
  contacts.

Cross-solver model joints are owned by the coupler only when the two connected
bodies belong to different entries and the joint itself is not owned by either
sub-solver. This avoids solving the same constraint twice. The current generic
ADMM path supports ``BALL``, ``FIXED``, and ``REVOLUTE`` joints. Ball joints
create translational anchor-coincidence rows. Fixed joints add angular rows.
Revolute joints preserve the hinge axis and can add a dry-friction row from
model joint friction. Prismatic, distance, and D6 joint rows are not yet part of
the experimental API.

Body-particle attachments cover interfaces that cannot be represented by a
model joint because one endpoint is a particle. The helper
``SolverAdmmCoupled.add_body_particle_attachment()`` registers and fills custom
attributes under ``coupling:body_particle_attachment`` with body id, particle id,
body-local point, stiffness, damping, and enabled state. Importers can author the
same custom attributes directly. Rows whose endpoints are unowned or owned by
the same entry are ignored; only cross-solver attachments are coupled by ADMM.

Contact coupling is enabled by adding one or more ``ContactPair`` values to
``SolverAdmmCoupled.Config.contact_pairs``. A contact pair names two entries and
optionally overrides ``contact_distance`` and ``detection_margin`` for that
interface. ``SolverAdmmCoupled.auto_detect_contact_pairs(entries, ...)`` can
build the complete pair list for every distinct entry combination.

For enabled contact pairs, the coupler owns private detection data and builds
rows from solver ownership: particle-shape rows between particle entries and
shapes on bodies owned by other entries, rigid-rigid rows from cross-entry shape
pairs, and particle-particle rows from cross-entry particle sets through a
private hash-grid stream. Friction is read from model material properties such
as ``shape_material_mu`` and ``Model.particle_mu`` at row-fill time; it is not a
``ContactPair`` field. Contact rows use an isotropic Coulomb
maximum-dissipation projection. They do not solve cone complementarity directly.

ADMM contact buffers are fixed-capacity device arrays. Persistent contacts
warm-start local variables and dual variables by stable contact keys across
steps. The particle-particle stream is contacts-like and hash-grid based, but it
is internal to the ADMM coupler and should not be treated as a public contact
stream.

The main ADMM parameters are:

- ``iterations``: fixed iteration count, chosen to be graph-capture friendly;
- ``rho``: penalty weight for interface rows;
- ``gamma``: proximal inertia and velocity weight;
- ``baumgarte``: positional error stabilization for attachment/contact rows;
- stiffness and damping values for model-joint and body-particle attachment
  rows;
- per-contact-pair distance and detection margins.

When ``gamma`` is positive, the coupler scales owned body and particle masses in
each entry ``ModelView``, asks sub-solvers to refresh model-derived caches, and
shifts entry input velocities toward the previous ADMM iterate. Endpoint
effective mass uses solver hooks when available and model fallbacks otherwise.
This keeps the implementation compatible with solvers that can provide an
articulated mass estimate, such as MuJoCo Warp, while still allowing simpler
solvers to participate.

Choosing Proxy or ADMM Coupling
-------------------------------

Use proxy coupling when one solver can reasonably treat the other solver's
endpoint as an obstacle, transfer participant, or virtual body over a substep.
Proxy coupling is often easier to tune for collider-style interactions and can
reuse destination solver contact machinery. It is also the path that currently
supports MPM transfer-active proxy particles and deformable collider particles.

Use ADMM coupling when the interface should be represented as a symmetric
constraint or frictional contact between entries. ADMM is better suited for
cross-solver joints, body-particle attachments, and contact rows that need equal
and opposite forces. It is more structured, but it also has more tuning
parameters and a narrower set of supported row types.

The two approaches share the same base concepts: model views, ownership,
entry-local state, force-buffer injection, input-state notifications, and
effective-mass hooks. A scene can often be formulated either way, but the
numerical behavior will differ. Proxy coupling behaves like a relaxed
fixed-point iteration over solver-specific dynamics. ADMM behaves like a fixed
iteration constrained optimization split over the entry solvers and interface
rows.

Solver-Specific Behavior
------------------------

Coupled solvers rely on solver-specific hooks only where generic public
model/state behavior is insufficient.

VBD uses proxy contact preparation, body-proxy harvesting, and input-state
notifications. The notification hook keeps private previous-body state aligned
when proxy poses are synchronized or ADMM iterations restart. The harvest path
reduces final rigid-rigid and body-particle contact forces onto proxy bodies
instead of relying on aggregate momentum differences.

Implicit MPM supports proxy body and proxy particle rewind/harvest hooks.
Transfer-active proxy particles can participate in P2G/G2P momentum transfer
while being excluded from material volume, stress, strain, and constitutive
updates. Deformable collider particles registered through collider-particle ids
use collider impulse collection rather than material transfer.

XPBD understands proxy particles and proxy bodies in particle contact kernels.
Owned particles may collide with destination proxy particles, but proxy-proxy,
proxy-static, and proxy-particle versus proxy-body contacts are filtered so the
destination solve does not create feedback between two proxy endpoints or
against immovable particles.

MuJoCo provides GPU effective-mass hooks from MuJoCo Warp data so proxy virtual
inertia and ADMM endpoint weights can use articulated mass estimates rather than
raw body mass. MuJoCo currently marks proxy impulse harvesting as unsupported
because the hook set does not expose a MuJoCo contact-only feedback path.

SemiImplicit preserves externally injected and coupled particle forces by
accumulating particle contact forces into the existing ``particle_f`` buffer.
Kamino accepts ``ModelView`` instances in solver construction and marks proxy
impulse harvesting as unsupported until it exposes a contact-only feedback path.

Current Limitations
-------------------

The coupled-solver framework is useful today, but it is still experimental:

- Proxy stability is tuning-sensitive. Virtual inertia scale, contact
  stiffness, solver iterations, and lagged versus staggered scheduling strongly
  affect damping and convergence.
- Generic momentum harvesting is only a fallback. Solver-private contact modes
  should expose custom harvest hooks where possible.
- ADMM contact detection is internal and does not consume arbitrary caller
  :class:`Contacts` rows as a public interface stream.
- ADMM joint support is limited to ball, fixed, and revolute rows.
- Particle-particle ADMM contacts use a private stream, not a public contact API.
- Effective-mass weighting falls back to simple model mass/inertia where no
  custom hook is available.
- USD ownership, automatic coupled-solver construction, and high-level tuning
  guidance are not part of the experimental public API yet.

Treat coupled solvers as an advanced feature for controlled experiments and
solver integration work. Prefer focused regression tests and explicit scene
tuning when using them in new examples.
