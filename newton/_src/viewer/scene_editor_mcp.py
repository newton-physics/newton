# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optional MCP transport for the application-owned scene editor."""

import asyncio
import base64
import dataclasses
import json
import logging
import math
import os
import queue
import secrets
import socket
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

if TYPE_CHECKING:
    from .scene_authoring import SceneEditor

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _Request:
    operation: str
    arguments: dict
    future: Future = dataclasses.field(default_factory=Future)
    cancelled: threading.Event = dataclasses.field(default_factory=threading.Event)
    id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)


class _BearerAuth:
    def __init__(self, app, *, token: str):
        self.app, self.token = app, token

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            supplied = headers.get(b"authorization", b"")
            if not secrets.compare_digest(supplied, f"Bearer {self.token}".encode()):
                await send(
                    {"type": "http.response.start", "status": 401, "headers": [(b"www-authenticate", b"Bearer")]}
                )
                await send({"type": "http.response.body", "body": b"MCP connection token required"})
                return
        await self.app(scope, receive, send)


class SceneEditorMCP:
    """Expose an owned scene editor to authenticated local MCP clients.

    This class is **experimental** and requires the optional ``mcp`` extra.
    Construction starts a localhost Streamable HTTP server. Only the viewer
    thread may call :meth:`before_frame` before the editor's update, and
    :meth:`after_frame` after rendering. Network handlers never access Newton,
    Warp, or OpenGL state directly. Images are captured only on request.
    Viewport observations require the native OpenGL viewer.

    Args:
        editor: Application-owned scene and runtime.
        port: Local TCP port; zero chooses an available port.
        info_file: Connection-settings file, created with mode 0600. A temporary
            file is used by default. It contains the URL and per-launch token,
            and is removed on close. Existing files are never overwritten.
    """

    def __init__(self, editor: "SceneEditor", *, port: int = 0, info_file: str | Path | None = None):
        import uvicorn  # noqa: PLC0415
        from mcp.server import MCPServer  # noqa: PLC0415
        from mcp.server.transport_security import TransportSecuritySettings  # noqa: PLC0415

        self.editor = editor
        self.session_id = uuid.uuid4().hex
        self._owner = threading.get_ident()
        self._queue = queue.Queue(maxsize=16)
        self._pending = []
        self._frame = 0
        self._closed = False
        self._failure = None
        self._last_edit_request = None
        token = secrets.token_urlsafe(32)
        server = MCPServer(
            "Newton Scene Editor",
            log_level="WARNING",
            instructions=(
                "Inspect get_scene and get_viewport before editing. Object IDs are stable; body indices are not. "
                "get_scene identifies the solver and device: mujoco uses native MuJoCo on CPU and MuJoCo Warp on CUDA; "
                "xpbd is XPBD. Native MuJoCo convex-hulls collision meshes. "
                "physics_ready distinguishes a published scene from physics initialized by Play or an explicit step. "
                "Positions and sizes are world Z-up metres, rotations are XYZW quaternions, mass is kilograms. "
                "Use the observed revision for apply_edits and undo_redo. Batch edits into one request. "
                "A pending document is not yet a compiled scene: inspect pending, busy and error until ready. "
                "Object pose_kind distinguishes live_root poses from authored_root fallbacks for independent assemblies. "
                "A failed import retains the previous runtime; undo_redo can discard the failed document. "
                "These tools control this simulation editor, not physical robot hardware."
            ),
        )
        self._register_tools(server)
        app = server.streamable_http_app(
            max_request_body_size=65536,
            max_sessions=8,
            session_idle_timeout=60,
            transport_security=TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=["127.0.0.1:*", "localhost:*"],
                allowed_origins=["http://127.0.0.1:*", "http://localhost:*"],
            ),
        )
        app.add_middleware(_BearerAuth, token=token)
        self._socket = socket.socket()
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("127.0.0.1", port))
        self._socket.listen(16)
        self.url = f"http://127.0.0.1:{self._socket.getsockname()[1]}/mcp"
        self._server = uvicorn.Server(
            uvicorn.Config(app, log_config=None, log_level=None, access_log=False, timeout_graceful_shutdown=1)
        )
        self._thread = threading.Thread(target=self._serve, name="newton-scene-mcp", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 5
        while not self._server.started and self._failure is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if not self._server.started:
            self.close()
            raise RuntimeError("Could not start the local scene MCP server") from self._failure
        try:
            if info_file is None:
                fd, path = tempfile.mkstemp(prefix="newton-scene-mcp-", suffix=".json")
                self.info_file = Path(path)
            else:
                self.info_file = Path(info_file).expanduser().resolve()
                fd = os.open(self.info_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                json.dump({"url": self.url, "token": token, "session_id": self.session_id, "pid": os.getpid()}, stream)
            self._info_inode = self.info_file.stat().st_ino
        except Exception:
            self.close()
            raise
        logger.info("Scene MCP listening at %s; connection settings: %s", self.url, self.info_file)

    def _serve(self) -> None:
        try:
            self._server.run(sockets=[self._socket])
        except BaseException as error:
            self._failure = error

    async def _submit(self, operation: str, **arguments):
        from mcp.server.mcpserver.exceptions import ToolError  # noqa: PLC0415

        request = _Request(operation, arguments)
        if self._closed:
            raise ToolError("The scene MCP server is closed")
        try:
            self._queue.put_nowait(request)
        except queue.Full as error:
            raise ToolError("The viewer command queue is full; wait for pending requests") from error
        try:
            return await asyncio.wait_for(asyncio.wrap_future(request.future), timeout=60)
        except asyncio.CancelledError:
            request.cancelled.set()
            request.future.cancel()
            raise
        except TimeoutError as error:
            request.cancelled.set()
            withdrawn = request.future.cancel()
            state = "withdrawn before execution" if withdrawn else "already started; inspect get_scene before retrying"
            raise ToolError(f"Request {request.id} timed out: {state}") from error
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
            raise ToolError(str(error)) from error

    def _register_tools(self, server) -> None:
        from mcp.types import CallToolResult, ImageContent, TextContent, ToolAnnotations  # noqa: PLC0415
        from pydantic import BaseModel, ConfigDict, Field  # noqa: PLC0415

        class Properties(BaseModel):
            model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
            name: str | None = Field(default=None, max_length=256)
            position: tuple[float, float, float] | None = None
            rotation: tuple[float, float, float, float] | None = None
            size: tuple[float, float, float] | None = None
            color: tuple[float, float, float] | None = None
            mass: float | None = Field(default=None, gt=0)
            friction: float | None = Field(default=None, ge=0)
            dynamic: bool | None = None
            visible: bool | None = None

        class AddPrimitive(BaseModel):
            model_config = ConfigDict(extra="forbid")
            operation: Literal["add_primitive"]
            kind: Literal["box", "sphere", "capsule", "cylinder"]
            properties: Properties = Field(default_factory=Properties)
            place: bool = True

        class AddUSD(BaseModel):
            model_config = ConfigDict(extra="forbid")
            operation: Literal["add_usd"]
            source: str = Field(max_length=4096)
            properties: Properties = Field(default_factory=Properties)
            place: bool = True

        class Update(BaseModel):
            model_config = ConfigDict(extra="forbid")
            operation: Literal["update"]
            object_id: str
            properties: Properties

        class Remove(BaseModel):
            model_config = ConfigDict(extra="forbid")
            operation: Literal["remove"]
            object_id: str

        edit_type = Annotated[AddPrimitive | AddUSD | Update | Remove, Field(discriminator="operation")]
        read = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
        write = ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=False)

        @server.tool(annotations=read)
        async def get_scene(offset: int = 0, limit: Annotated[int, Field(ge=1, le=256)] = 100) -> dict[str, Any]:
            """Inspect objects, pose_kind, units and revision; paginate with offset/limit.

            live_position/live_rotation follow a single free root when pose_kind
            is live_root. Assemblies without one free root use authored_root and
            return the authored placement, not their independent bodies' motion.
            """
            return await self._submit("scene", offset=offset, limit=limit)

        @server.tool(annotations=read)
        async def get_viewport(
            max_size: Annotated[int, Field(ge=128, le=2048)] = 1024, include_ui: bool = False
        ) -> CallToolResult:
            """Observe a rendered PNG plus matching frame metadata; max_size bounds the longest image edge."""
            pixels, stamp = await self._submit("viewport", include_ui=include_ui)

            def encode():
                from PIL import Image

                image = Image.fromarray(pixels)
                image.thumbnail((max_size, max_size))
                stream = BytesIO()
                image.save(stream, format="PNG")
                return stream.getvalue(), image.size

            png, size = await asyncio.to_thread(encode)
            stamp.update(width=size[0], height=size[1])
            return CallToolResult(
                content=[
                    ImageContent(type="image", data=base64.b64encode(png).decode(), mime_type="image/png"),
                    TextContent(type="text", text=json.dumps(stamp)),
                ],
                structured_content=stamp,
            )

        @server.tool(annotations=write)
        async def apply_edits(
            edits: Annotated[list[edit_type], Field(min_length=1, max_length=64)], expected_revision: int
        ) -> dict[str, Any]:
            """Prevalidate and stage one undoable edit batch using the observed revision. USD preparation may be pending."""
            return await self._submit("edits", edits=edits, expected_revision=expected_revision)

        @server.tool(annotations=write)
        async def undo_redo(direction: Literal["undo", "redo"], expected_revision: int) -> dict[str, Any]:
            """Undo or redo one authored batch, including recovery from a failed pending import."""
            return await self._submit("undo_redo", direction=direction, expected_revision=expected_revision)

        @server.tool(annotations=write)
        async def save_scene(path: str, overwrite: bool = False) -> dict[str, Any]:
            """Save the compiled authored document as USD; existing files require overwrite=True."""
            return await self._submit("save", path=path, overwrite=overwrite)

        @server.tool(annotations=write)
        async def set_camera(position: tuple[float, float, float], pitch: float, yaw: float) -> dict[str, Any]:
            """Move the viewer camera to a world position [m], with pitch/yaw in degrees."""
            return await self._submit("camera", position=position, pitch=pitch, yaw=yaw)

        @server.tool(annotations=write)
        async def set_playing(playing: bool) -> dict[str, Any]:
            """Pause or resume the owned simulation while leaving its viewer responsive."""
            return await self._submit("playing", playing=playing)

        @server.tool(annotations=write)
        async def step_simulation(frames: Annotated[int, Field(ge=1, le=120)] = 1) -> dict[str, Any]:
            """Advance a paused, compiled scene by bounded simulation frames."""
            return await self._submit("step", frames=frames)

    def _guard(self, *, expected_revision=None, allow_pending=False) -> None:
        editor = self.editor
        workspace = getattr(getattr(editor.viewer, "gui", None), "workspace", None)
        panel = getattr(workspace, "authoring", None)
        history = getattr(workspace, "history", None)
        if editor._edit_before is not None or (panel is not None and panel.interacting):
            raise ValueError("A human editing gesture is active; wait for release")
        if history is not None and history.replaying:
            raise ValueError("Return the timeline to its live frame before editing")
        if expected_revision is not None and expected_revision != editor.revision:
            raise ValueError(f"Stale scene revision {expected_revision}; current revision is {editor.revision}")
        if not allow_pending and (editor.pending or editor.busy):
            raise ValueError("Scene changes are pending; inspect get_scene until ready, or undo the failed edit")

    def _edit(self, edits, expected_revision: int) -> dict:
        self._guard(expected_revision=expected_revision)
        editor = self.editor
        objects = {obj.id: obj for obj in editor.objects}
        placed, changed = set(), []
        for edit in edits:
            operation = edit.operation
            if operation == "remove":
                del objects[edit.object_id]
                placed.discard(edit.object_id)
                changed.append(edit.object_id)
                continue
            fields = edit.properties.model_dump(exclude_unset=True)
            if operation == "update":
                obj = objects[edit.object_id]
                if obj.kind == editor.Shape.USD:
                    fields["overrides"] = tuple(
                        sorted(set(obj.overrides) | (fields.keys() & {"mass", "friction", "dynamic"}))
                    )
                obj = editor._validate(dataclasses.replace(obj, **fields))
            else:
                source = None
                if operation == "add_usd":
                    source = str(Path(edit.source).expanduser().resolve())
                    if not Path(source).is_file():
                        raise ValueError(f"USD source is not a local file: {source}")
                    kind = editor.Shape.USD
                    fields["overrides"] = tuple(sorted(fields.keys() & {"mass", "friction", "dynamic"}))
                    fields.setdefault("color", (1.0, 1.0, 1.0))
                else:
                    kind = editor.Shape(edit.kind)
                name = fields.pop("name", Path(source).stem if source else kind.value.title())
                obj = editor._validate(
                    editor.Object(id=uuid.uuid4().hex, name=name, kind=kind, source=source, **fields)
                )
                if edit.place:
                    placed.add(obj.id)
            objects[obj.id] = obj
            changed.append(obj.id)
        document = tuple(objects.values())
        if document != editor.objects:
            editor.begin_edit()
            try:
                editor._remember()
                editor._restore_document(document)
                editor._placement_requests.update(placed)
            finally:
                editor.end_edit()
        return {"changed_object_ids": changed}

    def _execute(self, request: _Request):
        operation, arguments = request.operation, request.arguments
        editor, viewer = self.editor, self.editor.viewer
        if operation in ("scene", "viewport"):
            if operation == "viewport" and not callable(
                getattr(getattr(viewer, "renderer", None), "_get_scene_viewport", None)
            ):
                raise ValueError("Viewport observations require the native OpenGL viewer")
            return {}
        if operation == "edits":
            result = self._edit(**arguments)
            self._last_edit_request = request.id
            return result
        if operation == "undo_redo":
            self._guard(expected_revision=arguments["expected_revision"], allow_pending=True)
            getattr(editor, arguments["direction"])()
            return {}
        self._guard()
        if operation == "save":
            path = Path(arguments["path"]).expanduser().resolve()
            if path.suffix.lower() not in (".usd", ".usda", ".usdc"):
                raise ValueError("Save paths must have a USD extension")
            if path.exists() and not arguments["overwrite"]:
                raise ValueError("The save file already exists; set overwrite=True to replace it")
            editor.save_usd(path)
            return {"path": str(path)}
        if operation == "camera":
            if getattr(viewer, "camera", None) is None:
                raise ValueError("The current viewer has no camera")
            editor._vector(arguments["position"], 3, "camera position")
            if not math.isfinite(arguments["pitch"]) or not math.isfinite(arguments["yaw"]):
                raise ValueError("Camera angles must be finite")
            viewer.set_camera(pos=arguments["position"], pitch=arguments["pitch"], yaw=arguments["yaw"])
        elif operation == "playing":
            if not hasattr(viewer, "_paused"):
                raise ValueError("The current viewer has no interactive playback controller")
            viewer._paused = not arguments["playing"]
        elif operation == "step":
            if not viewer.is_paused():
                raise ValueError("Pause simulation before requesting bounded steps")
            for _ in range(arguments["frames"]):
                editor.step()
                if editor.error:
                    raise RuntimeError(editor.error)
        return {}

    def _stamp(self) -> dict:
        editor, viewer = self.editor, self.editor.viewer
        workspace = getattr(getattr(viewer, "gui", None), "workspace", None)
        history = getattr(workspace, "history", None)
        camera = getattr(viewer, "camera", None)
        replaying = bool(history is not None and history.replaying)
        runtime = editor._runtime
        return {
            "session_id": self.session_id,
            "revision": editor.revision,
            "model_generation": editor.rebuild_count,
            "solver": editor.solver_type.value,
            "device": str(editor.device),
            "physics_ready": bool(
                runtime is not None and (runtime.solver is not None or not runtime.model.joint_count)
            ),
            "frame": self._frame,
            "sim_time": editor.sim_time,
            "display_time": history.frames[history.index].time if replaying else editor.sim_time,
            "pending": editor.pending,
            "busy": editor.busy,
            "error": editor.error,
            "paused": viewer.is_paused(),
            "history_replaying": replaying,
            "selected_id": editor.selected_id,
            "last_edit_request": self._last_edit_request,
            "camera": None
            if camera is None
            else {"position": list(camera.pos), "pitch": camera.pitch, "yaw": camera.yaw},
            "units": {"length": "m", "mass": "kg", "time": "s", "rotation": "xyzw", "up_axis": "Z"},
        }

    def before_frame(self) -> None:
        """Execute at most four queued operations on the owning viewer thread."""
        if threading.get_ident() != self._owner:
            raise RuntimeError("MCP commands must be pumped on the viewer thread")
        if self._closed:
            return
        for _ in range(4):
            try:
                request = self._queue.get_nowait()
            except queue.Empty:
                break
            if request.cancelled.is_set() or not request.future.set_running_or_notify_cancel():
                continue
            try:
                result = self._execute(request)
                self._pending.append((request, result))
            except Exception as error:
                request.future.set_exception(error)

    def after_frame(self) -> None:
        """Complete results and requested pixel captures after rendering on the owner thread."""
        if threading.get_ident() != self._owner:
            raise RuntimeError("MCP observations must be captured on the viewer thread")
        self._frame += 1
        pending, self._pending = self._pending, []
        captures = {}
        for request, payload in pending:
            if request.cancelled.is_set():
                request.future.set_exception(
                    RuntimeError("Request cancelled after execution started; inspect the scene")
                )
                continue
            try:
                stamp = self._stamp()
                stamp["request_id"] = request.id
                if request.operation == "scene":
                    arguments = request.arguments
                    offset = arguments["offset"]
                    if offset < 0:
                        raise ValueError("Object offset must be nonnegative")
                    rows = []
                    for obj in self.editor.objects[offset : offset + arguments["limit"]]:
                        row = dataclasses.asdict(obj)
                        row["kind"] = obj.kind.value
                        row["live_position"], row["live_rotation"] = self.editor.get_pose(obj.id)
                        runtime = self.editor._runtime
                        binding = runtime.bindings.get(obj.id) if runtime is not None else None
                        row["pose_kind"] = "live_root" if binding and binding.follows_body else "authored_root"
                        rows.append(row)
                    result = {**stamp, "objects": rows, "total_objects": len(self.editor.objects)}
                elif request.operation == "viewport":
                    include_ui = request.arguments["include_ui"]
                    if include_ui not in captures:
                        viewer = self.editor.viewer
                        pixels = viewer.get_frame(render_ui=include_ui).numpy()
                        if not include_ui:
                            # The renderer already owns DPI, docking and hidden-panel
                            # geometry. Its rectangle uses OpenGL's bottom-left origin.
                            x, y, width, height = viewer.renderer._get_scene_viewport()
                            top = pixels.shape[0] - y - height
                            pixels = pixels[top : top + height, x : x + width]
                        captures[include_ui] = pixels.copy()
                    result = captures[include_ui], stamp
                else:
                    result = {**stamp, **payload}
                request.future.set_result(result)
            except Exception as error:
                request.future.set_exception(error)

    def close(self) -> None:
        """Stop the local transport, release pending callers, and remove owned connection settings."""
        if self._closed:
            return
        self._closed = True
        requests = [request for request, _ in self._pending]
        self._pending.clear()
        while not self._queue.empty():
            requests.append(self._queue.get_nowait())
        for request in requests:
            if not request.future.done():
                request.future.set_exception(RuntimeError("The scene MCP server closed before completing the request"))
        self._server.should_exit = True
        self._thread.join(timeout=3)
        self._socket.close()
        path = getattr(self, "info_file", None)
        if path is not None and path.exists() and path.stat().st_ino == getattr(self, "_info_inode", None):
            path.unlink()
