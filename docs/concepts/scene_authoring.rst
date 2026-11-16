.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. _scene-authoring:

Scene Authoring
===============

.. experimental::

   :class:`newton.viewer.SceneEditor` and the ``scene_editor`` application are
   experimental. Their authoring API and saved editor metadata may change.

The scene editor combines the Newton workspace with an explicitly owned
simulation. The example uses MuJoCo Warp on CUDA by default; XPBD remains
available on CPU and CUDA and remains the CLI default on CPU. Native MuJoCo
is also available explicitly on CPU. Added objects
participate in physics. It supports rigid primitives
and local USD assets, including packages browsed from NVIDIA's SimReady catalog.

Launch and edit
---------------

Choose **New scene** in the header of a native Newton example viewer to open an
empty, paused editor without launching another process. The dialog offers
**MuJoCo (CPU)**, **MuJoCo Warp (GPU)** when CUDA is available, and **XPBD (CPU)**.
Choose Cancel to retain the current scene. Creating a scene replaces the
current example; save an existing authored document as USD first to keep it.
The action uses the example runner's scene lifecycle. Embedded viewers without
that runner retain their application's simulation ownership.

The ``examples`` extra includes the native viewer and USD import dependencies:

.. code-block:: console

   uv run --extra examples -m newton.examples scene_editor
   uv run --extra examples -m newton.examples scene_editor --empty
   uv run --extra examples -m newton.examples scene_editor --scene scene.usda
   uv run --extra examples -m newton.examples scene_editor --solver mujoco --device cpu
   uv run --extra examples -m newton.examples scene_editor --solver xpbd --device cpu

``--solver mujoco`` selects native MuJoCo on a CPU device and MuJoCo Warp on
a CUDA device. The solver and device are fixed for the editor session. The
window title and authoring header identify the active solver. When constructing :class:`~newton.viewer.SceneEditor`
directly, pass ``solver="mujoco", device="cpu"`` for native MuJoCo or select a
CUDA device for MuJoCo Warp; the class retains XPBD as its default.
An empty or static world with no joints needs
no solver. The example opens paused. MuJoCo Warp is constructed on the first
Play or single-step after a topology commit; browsing and placement do not
compile it. Native CPU MuJoCo follows the same lazy initialization. Use
``--no-paused`` to start playback at launch.

Native CPU MuJoCo suits small rigid scenes and does not capture CUDA graphs.
Its collision detection treats mesh shapes as convex hulls, so concave assets
can contact differently from Newton's mesh collision pipeline. Choose MuJoCo
Warp to retain Newton mesh contacts. XPBD on CPU also retains Newton mesh
contacts, but detailed triangle meshes can make CPU collision detection slow.
Assets requiring CUDA-only texture SDF data still need a CUDA device.
First use of either device can compile Warp kernels; CPU selection does not
guarantee an immediate first Play.

MuJoCo Warp uses Newton's collision pipeline, retaining the imported mesh
geometry instead of replacing concave meshes with MuJoCo convex hulls. Imported
``mjc:`` properties are registered before USD parsing. A topology commit replaces
the owned model and invalidates its solver. It preserves Newton body state and
compatible controls; the next simulation step creates a solver with fresh
warm-start and actuator activation history. Large assets can still pause
placement while their model and new render geometry are prepared. Unchanged
mesh boundary topology is cached with its source geometry, and matching GL
geometry is retained across model replacement. Geometry edits invalidate those
caches; removing an object releases render resources no longer used. Unchanged
mesh bounds and packed collision edge geometry are also reused across model
builds. The host cache retains one edge set per mesh, at 32 bytes per edge,
and is released with that mesh. Structural edits still finalize a replacement
model while paused.

In the visible native GL viewer, structural changes (including Delete and Undo)
show **Building model…** before the blocking rebuild begins. First playback
shows **Starting physics…** while collision kernels and the solver initialize.
These indicators preserve the camera and playback setting. The indicator clears
on success or failure: failed rebuilds retain the previous scene, while failed
physics initialization pauses the current scene. Heavy compilation still runs
on the viewer's owning thread.

Dragging a SimReady asset displays a nonblocking activity card in the viewport,
with the asset name and an animated download or preparation indicator. It stays
visible after dropping until that object is published. Ready states use a static
icon; package progress is indeterminate rather than an estimated percentage.
The card does not capture input or cancel the held drag.

The window title identifies the source commit at launch, with ``-dirty`` for
tracked local edits. Installed wheels show the package version. Restart the
application to load subsequent Python changes.

The left panel has three tabs:

* **Create** adds boxes, spheres, capsules, and cylinders. Click a card to place
  an object near the viewport center, or drag it onto the scene. A
  collision-bounds outline follows the pointer. Scale, color, and Dynamic
  configure new primitives. **Import USD** and local file drops add USD assets.
* **Assets** searches the SimReady catalog. Select a card to inspect its source
  and available license information. Drag a card or its large inspector preview
  into the viewport, double-click it, or choose **Place in scene** to download
  and import it.
* **Scene** lists authored objects and filters them by name. Select an object
  here or in the viewport. Its context menu offers Duplicate and Delete.

The right panel edits the selected object's name, position, rotation, dimensions,
color, visibility, friction, mass, and Dynamic setting. Move and Rotate handles
edit its live pose directly in the viewport. A handle gesture pauses simulation
temporarily, then restores its previous playback state. **Visible** controls
rendering; collision remains enabled. A fixed object remains a physical collider.

The authoring toolbar provides Open, Save, Undo, Redo, and Reset. Undo and Redo
operate on authored edits; they do not rewind unrelated simulation motion.
Undoing a pose edit restores its previous authored reset pose, which can differ
from the object's live position immediately before the edit.
Reset restores the authored poses. Save writes the authored scene, rather than
a recording of its current simulated state. Open restores an editor document;
opening an ordinary USD file imports it as an asset. Press **H** to hide or show
the workspace, and use the timeline controls to pause or advance physics.
Return from history replay to the live frame before authoring.
The workspace's **Restart example** action also resets physics while retaining
the authored objects, rather than recreating the initial demonstration.

SimReady packages and offline use
---------------------------------

The browser reads the public
`SimReady submission registry <https://github.com/NVIDIA-Omniverse/simready-central/blob/main/submissions/nvidia/simready-assets>`_.
Each registered asset names a pinned Hugging Face revision. Only visible
thumbnails are requested; dragging an asset into the viewport also starts its
package download so its geometry can be previewed. Downloads keep relative paths, referenced files,
metadata, and license files, and verify the package before importing it.
Review the source package's license for its permitted uses.

Catalog loading, cache verification, thumbnails, and package transfers run in
background workers. Native texture uploads happen on the UI thread, with a
bounded thumbnail cache. Downloaded packages use Newton's asset cache. Set
``HF_HUB_OFFLINE=1`` to prevent network requests; previously cached catalog
entries and complete packages remain available. A missing package then reports
an error rather than silently importing an incomplete asset. Local USD import
does not require the catalog or a Hugging Face account.
Completed placement requests wait during history replay and active editing
gestures. Opening another authored document discards its predecessor's queued
placements while retaining downloaded packages in the cache.

Fast edits and structural changes
---------------------------------

Pose, color, visibility, and friction edits update existing Model and State
buffers. Moving several objects in one frame synchronizes joint coordinates
once after all poses have been updated. Composed rotations stay normalized
during paused drags. Primitive dimensions, USD scale, mass,
motion type, and object additions or removals require a replacement runtime.
A drag groups these structural edits into one commit on release. Its outline is
transient viewer geometry and does not rerun ``ModelBuilder`` for each pointer movement.
During a USD scale gesture, the outline scales its cached envelope; the updated
USD template is prepared once after release.

Direct edits, MCP edits, and saved USD metadata share the same geometry and
physics validation. Sphere diameters must agree on all axes; capsules and
cylinders require equal X and Y diameters. Direct edits normalize nonzero
quaternions; saved metadata must already contain unit quaternions.
Values written to native arrays must be representable in Newton's float32
storage; invalid edits are rejected before changing the document or runtime.

USD preparation runs in one reusable, spawned CPU process and caches read-only
builder templates. Its Warp device and stream state are isolated from the
rendering process. Only host geometry, physics metadata, and bounds are returned;
no simulation Model is built for a preview. A native preparation failure is
reported without terminating the viewer, and a later request can start a fresh
worker. Closing the editor stops its owned preparation process. At a safe frame
boundary, the editor constructs a candidate Model, States, Control,
and collision pipeline before adopting them. Failed model preparation keeps the
previous simulation available and displays the error. MuJoCo compilation is
deferred until playback; a solver failure pauses the newly authored scene and
displays the error without replacing its visible state. Stable authored
IDs identify objects across model replacement; compiled body indices must be
resolved again after a commit. Matching bodies retain live state. Joint controls
are preserved only when the source, joint type, and coordinate layout remain
compatible; changed joints retain their imported defaults.
The current simulation continues during downloads and background preparation.
Cached templates track local referenced layers and asset files, including
missing optional assets. Reimporting, previewing, or reopening an authored
document refreshes templates when those files change. Changes during preparation
reject the result and retain the previous runtime; retry after the source files
are stable. Dependency checks do not add a file scan to normal physics stepping.

Interactive additions and duplicates aim at the surface under the cursor.
The bottom of the placement outline stays above the highest collider under its
whole footprint, including separated overhead surfaces. Tall colliders elsewhere
in the scene do not raise the outline. USD outlines use cached native collision
bounds, including their offset from the asset pivot, instead of a fixed-size box.
Downloading or preparing an asset shows a status message until its bounds are
available. Preview preparation reuses the import template and does not disable
dragging or add an object to the document. An unchanged pointer, camera and
published scene reuse the previous placement query. Changed surface-height
queries read two floats from the device; they do not rebuild the model.
Before publication, collision bounds are checked again, including other additions in the same batch.
This conservative placement can stack crowded objects instead of inserting
overlapping volumes. It checks both live and authored reset poses and retains
the resolved insertion coordinates in undo history. Existing bodies keep their
live poses and velocities.
Programmatic ``add_primitive`` and ``add_usd`` calls retain exact coordinates;
call :meth:`~newton.viewer.SceneEditor.place_object` before the commit to opt
into the same placement. Infinite planes require explicit placement.
The transient :meth:`~newton.viewer.SceneEditor.preview_primitive` and
:meth:`~newton.viewer.SceneEditor.preview_usd` methods accept ``placement=True``
to return a safe pivot position for a subsequent add. USD preview returns
``None`` while its template is being prepared.

When using Newton collision detection, the owned runtime checks contact
capacity before solving and expands collision buffers when dense mesh contacts
exceed the initial estimate. This does not
rebuild the model or reset Newton state. With MuJoCo Warp, contact growth also
invalidates the solver, which is recreated at the larger capacity on the next
physics substep. Solver overflow pauses physics
before publishing the failed step. These checks read collision counters and,
for MuJoCo Warp, its overflow mask from the device on each substep. Pose edits
and Reset clear solver warm-start history; friction edits notify the solver
without rebuilding the model.
Native CPU MuJoCo solves its own contacts without a Newton collision pass or
CUDA overflow readback. It exports native contact geometry after each simulated
frame. Its contact export capacity limits inspection, rather than CPU physics;
the solver warns if that display export is incomplete. Paused contact refresh
after authored edits uses Newton collision detection until the next CPU step.
MuJoCo collision detection also waits for a simulation step while browsing.
Paused pose edits invalidate the previous contacts. Opening contact inspection
or enabling contact display refreshes them once, without compiling the solver.
Every physics substep still checks complete collision data before solving.

MuJoCo Warp captures solver work into two CUDA graphs during playback,
after one real warm-up substep. Collision detection and overflow checks
remain outside those graphs. Subsequent frames reuse the graphs; pose,
appearance, friction and Reset edits retain them. Replacing a model, solver,
contact buffer, control or state storage, or changing the timestep invalidates
the captures. Graph capture never advances an extra simulation step.
The editor ground uses sliding friction 0.8. Authored spheres also have a small
rolling-friction length of 0.005 m; MuJoCo's six-dimensional sphere contacts
apply this resistance without adding rolling constraints to every mesh contact.
XPBD continues to use its native contact model.

This ownership boundary is deliberate. The editor does not mutate arbitrary
examples' private solvers or captured CUDA graphs. Solvers beyond XPBD and
MuJoCo (native CPU or Warp), replicated
training environments, deformable authoring, and editing an existing Isaac Lab
simulation are outside this application's supported scope.

The design follows the distinction between inexpensive live edits and explicit
recompilation found in
`RoboSimStudio's parameter tiers <https://github.com/robosimstudio/robosimstudio/blob/d474a18b0b70c8f8fae3bbd8ce43a8456e57969b/robosimstudio/specs/params.py#L26-L33>`_.
`mjlab's model-expansion path <https://github.com/mujocolab/mjlab/blob/dcdf6ce519abfd2ea45e9503ae6fb650b75bb869/src/mjlab/sim/sim.py#L441-L459>`_
also shows why replacing simulation arrays requires an owner that can refresh
dependent bridges, sensors, and graphs. Newton's editor owns its full runtime
and makes no assumptions about another application's graph lifecycle.
The authored records remain independent of OpenUSD so primitive-only editing
does not require a persistent USD stage. USD provides interchange, while live
poses remain in Newton State. Preallocated body slots would constrain arbitrary
USD topology without eliminating these ownership requirements.

USD interchange and limits
--------------------------

Primitive dimensions and positions are in metres, mass is in kilograms, and
rotations are XYZW quaternions in the Python API. Capsule height includes both
hemispheres. USD scale is multiplicative in the editor's Z-up frame. Native USD
import preserves supported rigid bodies, joints, colliders, and visuals.
Jointed props without articulation-root metadata receive standard articulation
roots in the anonymous session layer for native runtime import and USD export.
Geometry without an authored body is promoted to a rigid prop with collision
geometry. SimReady backend variants select Newton when available and disable
alternative backend variants in an anonymous session layer; source files stay
unchanged.

Assets containing authored rigid-body, joint, or mass data must declare
``metersPerUnit=1`` and ``kilogramsPerUnit=1``. Geometry without authored physics
can be converted from other stage length units. Particle and deformable
simulation, legacy ``TetMesh`` simulation, and ``resetXformStack`` are rejected.
Harmless grasp-marker curves and point annotations are permitted. Dynamic
planes and heightfields are unsupported.

Export uses standard OpenUSD geometry and physics schemas, with editor metadata
for stable IDs and authored values. USD assets remain references to their
source packages, so retain those packages when moving a saved scene.
Save rejects destinations that alias an imported source or any of its local
dependencies, preventing self-references
and accidental source-file replacement. White asset tint preserves source
materials. Exporting a nonwhite tint supports
``displayColor`` and scalar ``UsdPreviewSurface`` color; connected color
textures and other shader graphs require white tint and otherwise raise an
error. The native viewer remains subject to its existing material and texture
support.

Embedding the experimental API
------------------------------

Import the editor through its public module:

.. testcode:: scene-authoring

   from newton.viewer import SceneEditor, ViewerNull

   viewer = ViewerNull()
   editor = SceneEditor(viewer, device="cpu")
   box = editor.add_primitive("box", position=(0, 0, 1), size=(0.5, 0.5, 0.5))
   editor.place_object(box)
   editor.before_frame()
   editor.begin_edit()
   editor.update_object(box, size=(0.7, 0.5, 0.5))
   editor.end_edit()
   editor.before_frame()
   editor.step()
   editor.render()
   editor.close()
   viewer.close()

Call :meth:`~newton.viewer.SceneEditor.before_frame` before workspace binding,
stepping, or rendering, including while paused. Newton's example runner calls
it automatically for an example exposing ``scene_editor``. Use
:meth:`~newton.viewer.SceneEditor.begin_edit` and
:meth:`~newton.viewer.SceneEditor.end_edit` to group a gesture, or disable
``auto_apply`` and call :meth:`~newton.viewer.SceneEditor.request_commit` for an
explicit structural commit. Call :meth:`~newton.viewer.SceneEditor.close` when
the application is disposed to stop queued preparation.
Scripts importing USD assets must put application startup under
``if __name__ == "__main__":`` for Python's spawned-worker lifecycle. The normal
``scene_editor`` launcher already does this.

Agent access over MCP
---------------------

.. experimental::

   :class:`newton.viewer.SceneEditorMCP` exposes this owned scene editor to local
   agents. It does not attach to arbitrary examples or control physical robots.

Install the optional MCP dependencies and enable the server explicitly:

.. code-block:: console

   uv run --extra examples --extra mcp -m newton.examples scene_editor \
       --mcp-port 0 --mcp-info /tmp/newton-scene-mcp.json

Port zero selects an available localhost port. The connection-settings file
contains ``url``, ``token``, ``session_id``, and ``pid``. It is created with mode
0600 and removed when the editor closes. Existing files are never overwritten.
Without ``--mcp-info``, the application logs the generated temporary file's path.
Configure an MCP client with that URL and an ``Authorization: Bearer <token>``
header. The token changes on each launch. Only loopback connections and local
Host/Origin headers are accepted. The optional SDK uses the MIT license.

Tools use stable object IDs, world Z-up metres, kilograms, seconds, and XYZW
quaternions. Read the observed scene revision before submitting mutations:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Tool
     - Behavior
   * - ``get_scene``
     - Active ``solver`` and ``device``, paginated authored objects, root poses and their ``pose_kind``, revision,
       model generation, camera, playback state, and pending/busy/error status.
       ``physics_ready`` indicates whether physics has been initialized for the
       current model; a deferred solver becomes ready on Play or single-step.
   * - ``get_viewport``
     - A PNG cropped to the scene viewport, with frame, time, and scene metadata.
       ``include_ui`` returns the whole workspace instead; ``max_size`` bounds
       the image's longest edge. During replay,
       ``display_time`` identifies the selected frame and ``sim_time`` remains live.
   * - ``apply_edits``
     - A typed batch of ``add_primitive``, ``add_usd``, ``update``, and ``remove``
       operations, with the observed ``expected_revision``. Adds use safe
       placement by default; ``place=False`` retains exact coordinates.
   * - ``undo_redo``
     - Undo or redo one authored batch, including recovery from failed imports.
   * - ``set_camera``
     - Set world camera position and pitch/yaw in degrees.
   * - ``set_playing`` / ``step_simulation``
     - Pause/resume, or advance a paused scene by at most 120 frames.
   * - ``save_scene``
     - Save the authored USD document. Replacing an existing file requires
       ``overwrite=True``.

``pose_kind=live_root`` means ``live_position`` and ``live_rotation`` follow one
free physical root. Assemblies with independent or world-fixed roots return
``pose_kind=authored_root`` and their authored placement instead. Their component
motion remains visible in the viewport; the fallback does not describe each
body's current pose.

For example, the following SDK client inspects the current scene and stages one
box. It runs in another process while the viewer remains open:

.. code-block:: python

   import asyncio
   import json
   from pathlib import Path

   import httpx2
   from mcp.client import ClientSession
   from mcp.client.streamable_http import streamable_http_client

   async def edit_scene():
       settings = json.loads(Path("/tmp/newton-scene-mcp.json").read_text())
       headers = {"Authorization": "Bearer " + settings["token"]}
       async with httpx2.AsyncClient(headers=headers, timeout=90) as http:
           async with streamable_http_client(settings["url"], http_client=http) as (read, write):
               async with ClientSession(read, write) as client:
                   await client.initialize()
                   observed = await client.call_tool("get_scene", {})
                   result = await client.call_tool("apply_edits", {
                       "expected_revision": observed.structured_content["revision"],
                       "edits": [{
                           "operation": "add_primitive",
                           "kind": "box",
                           "properties": {"name": "Agent box", "position": [0, 0, 1]},
                       }],
                   })
                   assert not result.is_error, result.content

   asyncio.run(edit_scene())

The full batch is validated before changing the document. Successful batches
create one undo entry and at most one structural commit. Pose and material edits
retain the existing Model. USD preparation remains asynchronous: ``pending``
means the authored document is not yet fully published. A later import or
compilation error retains the previous runtime and is reported by ``get_scene``;
it does not roll back the authored document. Inspect the error and use
``undo_redo`` to recover. Stale revisions, active human gestures, and history
replay reject mutations instead of overwriting another interaction.

Commands use a bounded queue and execute on the viewer thread before the editor
update. Pixels are copied after rendering on that same thread; resizing and PNG
encoding run in a worker. There is no continuous screenshot stream. Cancellation
withdraws requests that have not started. A running native operation finishes
safely; after a timeout, inspect the scene before retrying a mutation.

The design uses the official SDK's
`Streamable HTTP transport <https://py.sdk.modelcontextprotocol.io/run/>`_
and `image results <https://py.sdk.modelcontextprotocol.io/servers/media/>`_.
Its owner-thread queue follows the same simulation boundary as
`RoboSimStudio's SimBus <https://github.com/robosimstudio/robosimstudio/blob/013b7181d3f5810cd29f954e00131bf6d7b97b61/robosimstudio/core/bus.py>`_.

For embedding, construct ``SceneEditorMCP(editor)`` on the viewer thread. Call
``bridge.before_frame()`` before ``editor.before_frame()`` and
``bridge.after_frame()`` after rendering, then close the bridge before disposing
the editor. The normal scene-editor launcher already supplies these hooks.
