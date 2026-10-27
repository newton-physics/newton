# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example: Cloth VR Teleoperation (WebXR)
#
# Interact with a cloth sheet draped over a table using your hands, tracked
# from a WebXR-capable headset (e.g. a Meta Quest browser). The example
# starts a small HTTP server that serves a WebXR client page; head and hand
# poses are streamed back as JSON and applied to the simulation:
#
#   - hand poses drive two sphere proxies whose velocity is set every
#     substep, so they push and drag the cloth,
#   - the head pose drives the viewer camera.
#
# Without a headset the example runs with deterministic procedural hand
# motion (also used in test mode), and the client page offers a mouse-driven
# fallback. Sessions can be recorded to and replayed from ``.npz`` files.
#
# Headset note: WebXR requires a secure context, which a plain http:// LAN
# address is not. From a standalone headset use adb (e.g.
# ``adb reverse tcp:8091 tcp:8091``) and open ``http://localhost:8091/``.
#
# Command: python -m newton.examples cloth_vr_teleop [--vr-port 8091]
#
###########################################################################

import argparse
import json
import math
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import warp as wp

import newton
import newton.examples

# Rotation about +X by 90 degrees: converts poses from the Y-up WebXR
# reference space into Newton's default Z-up model space.
_YUP_TO_ZUP_QUAT = np.array([0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float32)

# Reject tracked positions further than this from the origin (network input).
_MAX_TRACKED_POS = 100.0

_WEBXR_CLIENT_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Newton WebXR Client</title>
<style>
  body { background: #14161a; color: #d8dce2; font-family: system-ui, sans-serif; margin: 0;
         display: flex; flex-direction: column; align-items: center; justify-content: center;
         height: 100vh; gap: 14px; }
  h2 { margin: 0; font-weight: 600; }
  #status { color: #7fbf7f; min-height: 1.2em; }
  button { font-size: 17px; padding: 10px 26px; border-radius: 8px; border: 1px solid #3a4048;
           background: #232830; color: inherit; }
  button:disabled { opacity: 0.45; }
  canvas { border: 1px solid #3a4048; border-radius: 8px; touch-action: none; background: #0c0e11; }
  .hint { color: #8a919c; font-size: 13px; max-width: 380px; text-align: center; }
</style>
</head>
<body>
<h2>Newton WebXR Client</h2>
<div id="status">checking WebXR support&#8230;</div>
<button id="enter" disabled>Enter VR</button>
<canvas id="pad" width="360" height="240"></canvas>
<div class="hint">Desktop fallback: drag on the pad to move the right hand over the table.
Press the headset button to enter VR.</div>
<script>
"use strict";
const statusEl = document.getElementById("status");
const enterBtn = document.getElementById("enter");
const pad = document.getElementById("pad");
const ctx = pad.getContext("2d");

let sendQueued = false;  // keep at most one POST in flight

function post(msg) {
  if (sendQueued) return;
  sendQueued = true;
  fetch("/pose", { method: "POST", headers: { "Content-Type": "application/json" },
                   body: JSON.stringify(msg) })
    .catch(() => {})
    .finally(() => { sendQueued = false; });
}

function status(text) { statusEl.textContent = text; }

const msg = { t: 0, head: null, hands: {} };

function quatOf(xf) { const q = xf.orientation; return [q.x, q.y, q.z, q.w]; }
function posOf(xf) { const p = xf.position; return [p.x, p.y, p.z]; }

// ---- WebXR session ----
let session = null;

async function startXR() {
  if (!navigator.xr) return;
  try {
    session = await navigator.xr.requestSession("immersive-vr", {
      optionalFeatures: ["local-floor", "hand-tracking"],
    });
  } catch (err) {
    status("WebXR session failed: " + err.message);
    return;
  }
  enterBtn.disabled = true;
  let refSpaceName = "local-floor";
  let ref;
  try {
    ref = await session.requestReferenceSpace(refSpaceName);
  } catch (err) {
    refSpaceName = "local";
    ref = await session.requestReferenceSpace(refSpaceName);
  }
  status("VR session running (" + refSpaceName + ")");
  session.addEventListener("end", () => {
    session = null;
    enterBtn.disabled = false;
    status("VR session ended");
  });

  function onFrame(t, frame) {
    if (!session) return;
    const viewer = frame.getViewerPose(ref);
    if (viewer) msg.head = { p: posOf(viewer.transform), q: quatOf(viewer.transform) };
    msg.hands = {};
    for (const src of session.inputSources) {
      const pose = handPose(frame, src, ref);
      if (pose) msg.hands[src.handedness] = pose;
    }
    msg.t = t / 1000;
    post(msg);
    session.requestAnimationFrame(onFrame);
  }
  session.requestAnimationFrame(onFrame);
}

function handPose(frame, src, ref) {
  if (src.hand) {
    const wrist = frame.getJointPose(src.hand.get("wrist"), ref);
    if (!wrist) return null;
    return { p: posOf(wrist.transform), q: quatOf(wrist.transform) };
  }
  if (src.gripSpace) {
    const grip = frame.getPose(src.gripSpace, ref);
    if (grip) return { p: posOf(grip.transform), q: quatOf(grip.transform) };
  }
  return null;
}

// ---- Desktop fallback: drag on the pad to move the right hand ----
// The scene center sits 1.5 m in front of the user in the Y-up client frame.
let handX = 0.4;    // sideways, meters
let handF = 0.6;    // forward from the user, meters
let handH = 1.0;    // height, meters
let dragging = false;

const HEAD_FALLBACK = { p: [0, 1.6, 0], q: [0, 0, 0, 1] };

pad.addEventListener("pointerdown", (ev) => {
  dragging = true;
  pad.setPointerCapture(ev.pointerId);
  onPad(ev);
});
pad.addEventListener("pointerup", () => { dragging = false; });
pad.addEventListener("pointermove", (ev) => { if (dragging) onPad(ev); });

function onPad(ev) {
  const r = pad.getBoundingClientRect();
  const u = (ev.clientX - r.left) / r.width;
  const v = (ev.clientY - r.top) / r.height;
  handX = (u - 0.5) * 1.4;
  handF = 0.25 + v * 1.0;
  handH = 1.15 - v * 0.35;
}

function drawPad() {
  ctx.clearRect(0, 0, pad.width, pad.height);
  ctx.strokeStyle = "#3a4048";
  ctx.strokeRect(0.5, 0.5, pad.width - 1, pad.height - 1);
  ctx.fillStyle = "#7fbf7f";
  ctx.beginPath();
  ctx.arc(((handX / 1.4) + 0.5) * pad.width, ((handF - 0.25) / 1.0) * pad.height,
          9, 0, 2 * Math.PI);
  ctx.fill();
}

setInterval(() => {
  if (session) return;  // in VR, the session loop drives the poses
  msg.head = HEAD_FALLBACK;
  msg.hands = { right: { p: [handX, handH, -1.5 - handF], q: [0, 0, 0, 1] } };
  msg.t = performance.now() / 1000;
  post(msg);
  drawPad();
}, 33);

(async () => {
  drawPad();
  if (!navigator.xr) {
    status("WebXR unavailable - desktop fallback active");
    return;
  }
  const ok = await navigator.xr.isSessionSupported("immersive-vr").catch(() => false);
  if (!ok) {
    status("immersive-vr unsupported - desktop fallback active");
    return;
  }
  status("ready");
  enterBtn.disabled = false;
  enterBtn.addEventListener("click", startXR);
})();
</script>
</body>
</html>
"""


class _VRRequestHandler(BaseHTTPRequestHandler):
    # HTTP/1.1 keep-alive: the client posts poses at ~30-60 Hz and reuses the connection.
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return
        body = _WEBXR_CLIENT_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/pose":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        if length <= 0 or length > 16384:
            self.send_error(400)
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError):
            self.send_error(400)
            return
        self.server.record_pose(payload)
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format, *args):
        pass  # keep the simulation console quiet


class VRInputServer(ThreadingHTTPServer):
    """Serves the WebXR client page and receives head/hand pose messages."""

    daemon_threads = True

    def __init__(self, address):
        self._lock = threading.Lock()
        self._pose = None
        self._packets = 0
        super().__init__(address, _VRRequestHandler)

    def record_pose(self, payload):
        if not isinstance(payload, dict):
            return
        with self._lock:
            self._pose = payload
            self._packets += 1

    def latest_pose(self):
        with self._lock:
            return self._pose

    @property
    def packets(self):
        with self._lock:
            return self._packets

    def serve_background(self):
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        self.shutdown()
        self._thread.join(timeout=2.0)
        self.server_close()


def _as_pos(entry):
    if not isinstance(entry, dict):
        return None
    p = entry.get("p")
    if not isinstance(p, (list, tuple)) or len(p) != 3:
        return None
    try:
        p = np.asarray(p, dtype=np.float32)
    except (ValueError, TypeError, OverflowError):
        return None
    if not np.all(np.isfinite(p)) or np.linalg.norm(p) > _MAX_TRACKED_POS:
        return None
    return p


def _as_quat(entry):
    identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if not isinstance(entry, dict):
        return identity
    q = entry.get("q")
    if not isinstance(q, (list, tuple)) or len(q) != 4:
        return identity
    try:
        q = np.asarray(q, dtype=np.float32)
    except (ValueError, TypeError, OverflowError):
        return identity
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm < 1e-6:
        return identity
    return q / norm


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def _quat_rotate(q, v):
    r = wp.quat_rotate(wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3])), wp.vec3(*v))
    return np.array([r[0], r[1], r[2]], dtype=np.float32)


def _quat_between(a, b):
    w = 1.0 + float(np.dot(a, b))
    if w < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    q = np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0], w], dtype=np.float32)
    return q / np.linalg.norm(q)


def _frame_from_message(message, origin_yup):
    """Convert a client pose message to model-space frame data, or None if unusable."""

    def convert(entry):
        p = _as_pos(entry)
        if p is None:
            return None
        p = _quat_rotate(_YUP_TO_ZUP_QUAT, p + origin_yup)
        q = _quat_mul(_YUP_TO_ZUP_QUAT, _as_quat(entry))
        return p, q

    frame = {}
    head = _as_pos(message.get("head"))
    if head is not None:
        head = _quat_rotate(_YUP_TO_ZUP_QUAT, head + origin_yup)
        frame["head"] = (head, _quat_mul(_YUP_TO_ZUP_QUAT, _as_quat(message.get("head"))))
    hands = message.get("hands")
    if isinstance(hands, dict):
        for side in ("left", "right"):
            converted = convert(hands.get(side))
            if converted is not None:
                frame[side] = converted
    return frame or None


class XRPoseSource:
    """Latest head/hand poses from the WebXR client, converted to model space."""

    def __init__(self, server, origin_yup):
        self.server = server
        self.origin_yup = origin_yup

    def next(self, frame_idx):
        message = self.server.latest_pose()
        if message is None:
            return None
        return _frame_from_message(message, self.origin_yup)


class ProceduralPoseSource:
    """Deterministic head/hand choreography used when no client is connected."""

    def __init__(self, half_extent_y):
        self.half_extent_y = half_extent_y

    def next(self, frame_idx):
        t = frame_idx * (1.0 / 60.0)
        head_p = np.array(
            [0.10 * math.sin(0.7 * t), -2.1 + 0.08 * math.sin(0.45 * t), 1.85 + 0.04 * math.sin(0.6 * t)],
            dtype=np.float32,
        )
        look_at = np.array([0.0, 0.0, 0.55], dtype=np.float32)
        direction = look_at - head_p
        direction /= np.linalg.norm(direction)
        # Stored head quats map the head-local frame to model space, with the
        # local forward direction at (0, 0, -1) (same convention as WebXR).
        head_q = _quat_between(np.array([0.0, 0.0, -1.0], dtype=np.float32), direction.astype(np.float32))

        # Hand waypoints in model space; x is mirrored for the left hand.
        home = np.array([0.55, -0.10, 1.30], dtype=np.float32)
        touch = np.array([0.45, 0.75 * self.half_extent_y, 0.86], dtype=np.float32)
        drag = np.array([0.70, 1.15 * self.half_extent_y, 0.45], dtype=np.float32)

        def sweep(phase):
            # 0.0-0.4s approach, 0.4-1.4s drag, 1.4-1.8s hold, 1.8-2.4s return
            if phase < 0.4:
                s = _smoothstep(phase / 0.4)
                return home + (touch - home) * s
            if phase < 1.4:
                s = _smoothstep((phase - 0.4) / 1.0)
                return touch + (drag - touch) * s
            if phase < 1.8:
                return drag
            s = _smoothstep((phase - 1.8) / 0.6)
            return drag + (home - drag) * s

        period = 2.4
        right_p = sweep((t + 0.6) % period)
        left_p = sweep(t % period)
        left_p[0] = -left_p[0]

        return {
            "head": (head_p, head_q),
            "left": (left_p, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)),
            "right": (right_p, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)),
        }


def _smoothstep(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


class ReplayPoseSource:
    """Head/hand poses recorded to .npz by --record, played back frame by frame."""

    def __init__(self, path):
        data = np.load(path)
        self.data = {key: np.asarray(data[key]) for key in data.files}
        required = ("head_p", "left_p", "right_p")
        missing = [key for key in required if key not in self.data]
        if missing:
            raise ValueError(f"Replay file {path} is missing arrays: {missing}")
        self.count = len(self.data["head_p"])

    def next(self, frame_idx):
        i = min(frame_idx, self.count - 1)

        def quat(key):
            q = self.data.get(key)
            if q is None:
                return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            q = np.asarray(q[i], dtype=np.float32)
            norm = float(np.linalg.norm(q))
            return q / norm if norm > 1e-6 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        frame = {"head": (np.asarray(self.data["head_p"][i], dtype=np.float32), quat("head_q"))}
        for side in ("left", "right"):
            frame[side] = (
                np.asarray(self.data[f"{side}_p"][i], dtype=np.float32),
                quat(f"{side}_q"),
            )
        return frame


@wp.kernel
def drive_hand_proxies(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    targets: wp.array[wp.vec3],
    params: wp.array[wp.float32],
    dt: float,
    left_body: int,
    right_body: int,
):
    tid = wp.tid()
    body = left_body
    if tid == 1:
        body = right_body
    x = wp.transform_get_translation(body_q[body])
    v = (targets[tid] - x) * (1.0 / dt)
    speed = wp.length(v)
    limit = params[0]
    if speed > limit:
        v = v * (limit / speed)
    # spatial vectors store the linear part on top and the angular part on the bottom
    body_qd[body] = wp.spatial_vector(v, wp.vec3())


# Set by __main__ so that Example(viewer, args) can pick the server up without
# carrying a non-serializable object through the example browser's deep-copied args.
_ACTIVE_SERVER = None


class Example:
    def __init__(self, viewer, args):
        self.args = args
        self.viewer = viewer

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.cell = 0.08
        self.table_top = 0.72
        half_y = args.height * self.cell * 0.5

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        # Table: a heavy box the cloth drapes over.
        table_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, self.table_top * 0.5), wp.quat_identity()),
            label="table",
        )
        builder.add_shape_box(table_body, hx=0.8, hy=0.5, hz=self.table_top * 0.5)

        # Cloth sheet, horizontal above the table.
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, self.table_top + 0.1),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=args.width,
            dim_y=args.height,
            cell_x=self.cell,
            cell_y=self.cell,
            mass=0.1,
            particle_radius=0.02,
            add_springs=True,
            spring_ke=1.0e3,
            spring_kd=1.0,
        )

        # Hand proxies: free spheres whose velocity is set from tracked poses.
        self.left_body = builder.add_body(
            xform=wp.transform(wp.vec3(-0.55, -0.10, 1.30), wp.quat_identity()), label="hand_left"
        )
        builder.add_shape_sphere(self.left_body, radius=0.05)
        self.right_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.55, -0.10, 1.30), wp.quat_identity()), label="hand_right"
        )
        builder.add_shape_sphere(self.right_body, radius=0.05)

        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 1.0e2
        self.model.soft_contact_mu = 1.0

        self.solver = newton.solvers.SolverXPBD(model=self.model, iterations=self.iterations)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        # Initial particle positions, kept for the interaction assertions in test mode.
        self.initial_particle_q = self.state_0.particle_q.numpy().copy()

        self.home_left = np.array([-0.55, -0.10, 1.30], dtype=np.float32)
        self.home_right = np.array([0.55, -0.10, 1.30], dtype=np.float32)
        self._targets_np = np.stack([self.home_left, self.home_right]).astype(np.float32)
        self._targets = wp.array(self._targets_np, dtype=wp.vec3)
        self._params_np = np.array([args.max_hand_speed], dtype=np.float32)
        self._params = wp.array(self._params_np, dtype=wp.float32)

        self._hands_enabled = True
        self._frame = None
        self._frame_idx = 0
        self._mode_name = "procedural"
        self._source = None
        self._fallback_source = None
        self._server = None
        self._cloth_touched = False
        self._camera_framed = False

        origin_yup = np.asarray([float(v) for v in args.xr_origin.split(",")], dtype=np.float32)
        if args.replay:
            self._source = ReplayPoseSource(args.replay)
            self._mode_name = f"replay ({args.replay})"
        else:
            server = _ACTIVE_SERVER
            if server is not None:
                self._source = XRPoseSource(server, origin_yup)
                self._fallback_source = ProceduralPoseSource(half_y)
                self._server = server
                self._mode_name = "vr (procedural until client connects)"
            else:
                self._source = ProceduralPoseSource(half_y)
                self._mode_name = "procedural"

        self._record_path = args.record
        self._rec = {"t": []}
        for key in ("head", "left", "right"):
            self._rec[f"{key}_p"] = []
            self._rec[f"{key}_q"] = []

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            wp.launch(
                drive_hand_proxies,
                dim=2,
                inputs=[
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self._targets,
                    self._params,
                    self.sim_dt,
                    self.left_body,
                    self.right_body,
                ],
            )

            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        frame = self._source.next(self._frame_idx)
        if frame is not None:
            self._frame = frame
            if self._server is not None:
                self._mode_name = "vr"
        elif self._fallback_source is not None:
            # No client connected yet: run the procedural choreography so the
            # example matches its documented behavior from the first frame.
            frame = self._fallback_source.next(self._frame_idx)
            if frame is not None:
                self._frame = frame

        targets = [self.home_left, self.home_right]
        if self._hands_enabled and self._frame is not None:
            for i, side in enumerate(("left", "right")):
                if side in self._frame:
                    targets[i] = np.asarray(self._frame[side][0], dtype=np.float32)
        self._targets_np[0] = targets[0]
        self._targets_np[1] = targets[1]
        self._targets.assign(self._targets_np)

        self._record_frame()
        self._frame_idx += 1

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

    def _record_frame(self):
        if not self._record_path:
            return
        frame = self._frame or {}
        self._rec["t"].append(self._frame_idx * self.frame_dt)

        head = frame.get("head")
        if head is not None:
            head_p, head_q = np.asarray(head[0], dtype=np.float32), np.asarray(head[1], dtype=np.float32)
        elif self._rec["head_p"]:
            head_p, head_q = self._rec["head_p"][-1], self._rec["head_q"][-1]
        else:
            head_p = np.array([0.0, -2.0, 1.8], dtype=np.float32)
            head_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self._rec["head_p"].append(head_p)
        self._rec["head_q"].append(head_q)

        # Record the targets the simulation actually used (after the hands
        # toggle and home fallback), so replay reproduces the run exactly.
        identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        for i, side in enumerate(("left", "right")):
            self._rec[f"{side}_p"].append(self._targets_np[i].copy())
            entry = frame.get(side)
            self._rec[f"{side}_q"].append(np.asarray(entry[1], dtype=np.float32) if entry is not None else identity)

        if len(self._rec["t"]) % 100 == 0:
            self._flush_record()

    def _flush_record(self):
        if not self._record_path or not self._rec["t"]:
            return
        data = {key: np.asarray(value, dtype=np.float32) for key, value in self._rec.items()}
        tmp_path = self._record_path + ".tmp"
        with open(tmp_path, "wb") as f:
            np.savez(f, **data)
        os.replace(tmp_path, self._record_path)

    def _apply_head_camera(self):
        if self._frame is None or "head" not in self._frame:
            return
        follow = self._mode_name == "vr" or isinstance(self._source, ReplayPoseSource)
        if not follow:
            # Procedural and not-yet-connected runs frame the scene once and
            # then leave the camera to the user (mouse/keyboard orbit).
            if self._camera_framed:
                return
            self._camera_framed = True
        p, q = self._frame["head"]
        # Stored quats map the head-local frame to model space with local
        # forward (0, 0, -1); model space is Z-up and the viewer camera up
        # axis follows the model, so yaw/pitch follow the Z-up conventions.
        f = _quat_rotate(q, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        norm = float(np.linalg.norm(f))
        if norm < 1e-6:
            return
        f = f / norm
        yaw = math.degrees(math.atan2(float(f[1]), float(f[0])))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, float(f[2])))))
        self.viewer.set_camera(wp.vec3(float(p[0]), float(p[1]), float(p[2])), pitch, yaw)

    def test_post_step(self):
        # Witness that a hand proxy actually reached the cloth; combined with
        # the lateral-displacement check in test_final this proves the drag.
        if self._cloth_touched:
            return
        particle_q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        for body in (self.left_body, self.right_body):
            center = body_q[body][:3]
            if float(np.linalg.norm(particle_q - center, axis=1).min()) < 0.15:
                self._cloth_touched = True
                return

    def test_final(self):
        self._flush_record()

        newton.examples.test_particle_state(
            self.state_0,
            "particles are above the ground",
            lambda q, qd: q[2] > -0.05,
        )
        lower = wp.vec3(-3.0, -3.0, -0.05)
        upper = wp.vec3(3.0, 3.0, 2.5)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, lower, upper),
        )

        # The hand proxies must have interacted with the cloth. test_post_step
        # witnesses a hand proxy actually reaching the cloth (the overhang
        # flaps sway on their own, so motion alone is not proof of a drag);
        # the lateral displacement check confirms the sheet actually moved.
        if not self._cloth_touched:
            raise ValueError("Hand proxies never reached the cloth during the test run")
        particle_q = self.state_0.particle_q.numpy()
        lateral = np.linalg.norm(particle_q[:, 0:2] - self.initial_particle_q[:, 0:2], axis=1)
        moved = float(np.mean(lateral > 0.03))
        if moved < 0.02:
            raise ValueError(
                f"Hand proxies did not drag the cloth: only {moved:.1%} of particles moved more than 3 cm laterally"
            )

    def render(self):
        self.viewer.begin_frame(self._frame_idx * self.frame_dt)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self._apply_head_camera()
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text(f"Input: {self._mode_name}")
        if self._server is not None:
            host, port = self._server.server_address[:2]
            ui.text(f"Client page: http://{host}:{port}/")
            ui.text(f"Packets received: {self._server.packets}")
        changed, enabled = ui.checkbox("Hands enabled", self._hands_enabled)
        if changed:
            self._hands_enabled = enabled
        changed, speed = ui.slider_float("Max hand speed [m/s]", float(self._params_np[0]), 0.5, 5.0, "%.1f")
        if changed:
            self._params_np[0] = speed
            self._params.assign(self._params_np)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--width", type=int, default=40, help="Cloth resolution in x.")
        parser.add_argument("--height", type=int, default=40, help="Cloth resolution in y.")
        parser.add_argument("--max-hand-speed", type=float, default=2.5, help="Maximum tracked hand proxy speed [m/s].")
        parser.add_argument(
            "--vr-host",
            type=str,
            default="127.0.0.1",
            help="Host address for the WebXR client server. For standalone headsets use 0.0.0.0 together "
            "with 'adb reverse tcp:<port> tcp:<port>' and open http://localhost:<port>/ on the headset.",
        )
        parser.add_argument("--vr-port", type=int, default=8091, help="Port for the WebXR client server.")
        parser.add_argument(
            "--vr-server",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Whether to start the WebXR client server (off in test mode).",
        )
        parser.add_argument(
            "--xr-origin",
            type=str,
            default="0.0,0.0,1.5",
            help="User origin offset in WebXR Y-up space applied to tracked poses [m].",
        )
        parser.add_argument("--record", type=str, default=None, help="Record head/hand poses to a .npz file.")
        parser.add_argument("--replay", type=str, default=None, help="Replay head/hand poses from a .npz file.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    server = None
    if not args.test and args.vr_server:
        try:
            server = VRInputServer((args.vr_host, args.vr_port))
            server.serve_background()
            host, port = server.server_address[:2]
            print(f"WebXR client page: http://{host}:{port}/")
            if host in ("0.0.0.0", "::"):
                print(
                    f"Standalone headsets: run 'adb reverse tcp:{port} tcp:{port}' on this machine, then open "
                    f"http://localhost:{port}/ in the headset browser (WebXR requires a secure context; "
                    "plain http:// LAN addresses do not provide one)."
                )
            _ACTIVE_SERVER = server
        except OSError as err:
            print(f"Warning: could not start the WebXR client server ({err}); using procedural input")

    example = Example(viewer, args)
    newton.examples.run(example, args)
    example._flush_record()

    if server is not None:
        server.stop()
