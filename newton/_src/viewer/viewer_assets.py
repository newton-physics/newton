# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Asynchronous browsing and verified caching of public SimReady packages.

The submission registry supplies immutable Hugging Face package revisions. Only
visible thumbnails and explicitly requested packages are downloaded. No optional
Hub client, USD runtime, or network connection is needed to import this module.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from queue import Empty, SimpleQueue
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urljoin, urlsplit
from urllib.request import Request, urlopen

from ... import config

_REGISTRY_URL = (
    "https://raw.githubusercontent.com/NVIDIA-Omniverse/simready-central/main/submissions/nvidia/simready-assets"
)
_HUB_URL = "https://huggingface.co"
_MANIFEST = "com.nvidia.simready.packaging.json"
_ROOTS = ".metadata/com.nvidia.simready.root_usds.json"
_BOM = ".metadata/com.nvidia.simready.packaging.bom.json"
_TIMEOUT = 10
_JSON_LIMIT = 16 * 1024 * 1024


@dataclass(frozen=True)
class AssetRecord:
    """A package at an immutable source revision, independent of browser state."""

    id: str
    """Stable identity containing repository, revision, and package directory."""
    name: str
    """Display name derived from the registered package directory."""
    repo_id: str
    """Hugging Face dataset repository, including its owner."""
    revision: str
    """Full immutable Git commit SHA registered by the asset provider."""
    directory: str
    """Repository-relative directory containing the complete package."""
    url: str
    """Canonical pinned source URL of the package manifest."""
    thumbnail_url: str | None = None
    """Pinned thumbnail URL, populated when its metadata is requested."""
    license: str | None = None
    """License identifier supplied by the package manifest, when available."""


def _relative(value: str) -> str:
    """Validate a package-relative path before accessing the cache."""
    path = PurePosixPath(value)
    if not value or "\\" in value or path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError(f"Unsafe package path: {value!r}")
    return path.as_posix()


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AssetLibrary:
    """A two-worker, polling asset service for the viewer's main thread.

    Call :meth:`poll` once per frame. ``search`` filters an existing catalog
    immediately; its first call loads the registry in the background. ``refresh``
    explicitly refreshes the registry. Errors remain visible until a new search,
    refresh, or download, and a failed download can be retried with ``download``.
    ``HF_HUB_OFFLINE`` disables every network request, including registry access.
    """

    def __init__(self, cache_dir: str | os.PathLike | None = None):
        selected = config.cache_dir_assets if cache_dir is None else cache_dir
        if selected is None:
            selected = tempfile.mkdtemp(prefix="newton-simready-")
            atexit.register(shutil.rmtree, selected, ignore_errors=True)
        self._cache = Path(selected).expanduser() / "simready"
        self.records: list[AssetRecord] = []
        self.error: str | None = None
        self.progress = ""
        self._catalog: list[AssetRecord] | None = None
        self._query = ""
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="newton-assets")
        self._jobs: dict[tuple[str, str], Future] = {}
        self._messages: SimpleQueue[str] = SimpleQueue()
        self._stop = threading.Event()
        self._verified: dict[str, Path | None] = {}
        self._thumbnails: dict[str, Path | None] = {}
        self._download_errors: dict[str, str] = {}

    @property
    def loading(self) -> bool:
        """Whether a catalog, thumbnail, download, or cache check is pending."""
        return not self._stop.is_set() and bool(self._jobs)

    def _submit(self, kind: str, identity: str, function, *args) -> None:
        key = (kind, identity)
        if not self._stop.is_set() and key not in self._jobs:
            self._jobs[key] = self._executor.submit(function, *args)

    def search(self, query: str = "") -> None:
        """Filter the catalog, loading it asynchronously on first use."""
        self._query = query.casefold().strip()
        self.error = None
        if self._catalog is None:
            self._submit("catalog", "", self._load_catalog, False)
        else:
            self._filter()

    def refresh(self) -> None:
        """Refresh the public registry without discarding a usable catalog."""
        self.error = None
        self._submit("catalog", "", self._load_catalog, True)

    def _filter(self) -> None:
        self.records = [
            record
            for record in self._catalog or []
            if self._query in (record.name + " " + record.directory + " " + record.repo_id).casefold()
        ]

    def poll(self) -> None:
        """Publish completed background work; never wait for a worker."""
        if self._stop.is_set():
            return
        while True:
            try:
                self.progress = self._messages.get_nowait()
            except Empty:
                break
        for key, future in list(self._jobs.items()):
            if not future.done():
                continue
            del self._jobs[key]
            kind, identity = key
            try:
                result = future.result()
                if kind == "catalog":
                    self._catalog = result
                    self._filter()
                    self.progress = f"{len(result):,} assets"
                elif kind == "thumbnail":
                    path, enriched = result
                    self._thumbnails[identity] = path
                    if self._catalog is not None:
                        self._catalog = [enriched if record.id == identity else record for record in self._catalog]
                        self._filter()
                else:
                    self._verified[identity] = result
                    if result is not None:
                        self.progress = "Asset ready"
            except Exception as error:
                self.error = str(error)
                self.progress = "Asset operation failed"
                if kind == "thumbnail":
                    self._thumbnails[identity] = None
                elif kind != "catalog":
                    self._verified[identity] = None
                    if kind == "download":
                        self._download_errors[identity] = str(error)

    def thumbnail(self, record: AssetRecord) -> Path | None:
        """Return a cached thumbnail, scheduling at most two visible requests."""
        if record.id not in self._thumbnails:
            if sum(kind == "thumbnail" for kind, _ in self._jobs) < 2:
                self._submit("thumbnail", record.id, self._thumbnail, record)
        return self._thumbnails.get(record.id)

    def download(self, record: AssetRecord) -> None:
        """Download and verify the entire pinned package, or retry a failure."""
        self.error = None
        self._download_errors.pop(record.id, None)
        self._submit("download", record.id, self._download, record)

    def download_error(self, record: AssetRecord) -> str | None:
        """Return this package's download failure, cleared when it is retried."""
        return self._download_errors.get(record.id)

    def ready(self, record: AssetRecord) -> Path | None:
        """Return the complete local root USD, verifying existing caches off-thread."""
        if record.id not in self._verified and ("download", record.id) not in self._jobs:
            self._submit("verify", record.id, self._check_cache, record)
        return self._verified.get(record.id)

    def close(self) -> None:
        """Cancel queued work and signal active transfers without waiting."""
        self._stop.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._jobs.clear()

    def _check_cancelled(self) -> None:
        if self._stop.is_set():
            raise RuntimeError("Asset operation cancelled")

    @staticmethod
    def _offline() -> bool:
        return os.environ.get("HF_HUB_OFFLINE", "").upper() in ("1", "ON", "YES", "TRUE")

    def _request(self, url: str, destination: Path | None = None) -> tuple[bytes, str]:
        self._check_cancelled()
        if self._offline():
            raise RuntimeError("Offline mode: the requested asset is not complete in the cache")
        for attempt in range(3):
            temporary = None
            try:
                request = Request(url, headers={"User-Agent": "Newton-SimReady-browser", "Accept": "application/json"})
                with urlopen(request, timeout=_TIMEOUT) as response:
                    link = response.headers.get("Link", "")
                    total = response.headers.get("Content-Length", "?")
                    if destination is not None:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".part")
                        with temporary.open("wb") as stream:
                            size = 0
                            while chunk := response.read(256 * 1024):
                                self._check_cancelled()
                                stream.write(chunk)
                                size += len(chunk)
                                self._messages.put(f"Downloading {destination.name}: {size:,} / {total} bytes")
                        if total != "?" and size != int(total):
                            raise ValueError(f"Incomplete download: {destination.name}")
                        self._check_cancelled()
                        temporary.replace(destination)
                        return b"", link
                    data = response.read(_JSON_LIMIT + 1)
                    if len(data) > _JSON_LIMIT:
                        raise ValueError("Asset catalog or metadata exceeds the supported size")
                    return data, link
            except HTTPError as error:
                error.close()
                if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise RuntimeError(f"HTTP {error.code} fetching {url}") from error
                retry = error.headers.get("Retry-After", "1")
                delay = float(retry) if retry.isdigit() else 1.0
                if delay > 10:
                    raise RuntimeError(f"HTTP {error.code}: server requests a later retry") from error
                if self._stop.wait(delay):
                    self._check_cancelled()
            except URLError as error:
                raise RuntimeError(f"Cannot fetch {url}: {error.reason}") from error
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Cannot fetch {url}")

    def _load_catalog(self, refresh: bool) -> list[AssetRecord]:
        path = self._cache / "registry.txt"
        if path.is_file() and (not refresh or self._offline()):
            data = path.read_bytes()
        else:
            data, _ = self._request(_REGISTRY_URL)
        records = {}
        for line in data.decode().splitlines():
            url = line.strip()
            if not url or url.startswith("#"):
                continue
            parsed = urlsplit(url)
            if parsed.netloc != urlsplit(_HUB_URL).netloc:
                continue
            match = re.fullmatch(
                r"/datasets/([^/]+/[^/]+)/resolve/([0-9a-f]{40})/(.+)/" + re.escape(_MANIFEST), unquote(parsed.path)
            )
            if match is None:
                continue
            repo, revision, directory = match.groups()
            _relative(repo)
            _relative(directory)
            identity = f"{repo}@{revision}:{directory}"
            records[identity] = AssetRecord(
                identity, PurePosixPath(directory).name.replace("_", " "), repo, revision, directory, url
            )
        if not records:
            raise ValueError("The SimReady registry contains no pinned asset packages")
        self._check_cancelled()
        _atomic(path, data)
        return sorted(records.values(), key=lambda record: record.name.casefold())

    def _package(self, record: AssetRecord) -> Path:
        return self._cache / "packages" / hashlib.sha256(record.id.encode()).hexdigest()

    def _file_url(self, record: AssetRecord, name: str) -> str:
        _relative(name)
        return (
            f"{_HUB_URL}/datasets/{quote(record.repo_id, safe='/')}/resolve/{record.revision}/"
            f"{quote(record.directory + '/' + name, safe='/')}"
        )

    def _metadata(self, record: AssetRecord, name: str) -> dict:
        path = self._package(record) / "content" / _relative(name)
        if not path.is_file():
            self._request(self._file_url(record, name), path)
        return json.loads(path.read_bytes())

    def _root(self, record: AssetRecord) -> str:
        entries = self._metadata(record, _ROOTS).get("entries", [])
        if not entries:
            raise ValueError(f"No root USD listed for {record.name}")
        return _relative(entries[0])

    def _thumbnail(self, record: AssetRecord) -> tuple[Path | None, AssetRecord]:
        manifest = self._metadata(record, _MANIFEST)
        root = PurePosixPath(self._root(record))
        name = str(root.parent / ".thumbs" / "256x256" / (root.name + ".png"))
        url = self._file_url(record, name)
        enriched = replace(record, thumbnail_url=url, license=manifest.get("license"))
        path = self._package(record) / "content" / name
        if not path.is_file():
            try:
                self._request(url, path)
            except RuntimeError as error:
                if "HTTP 404" in str(error):
                    return None, enriched
                raise
        return path, enriched

    def _tree(self, record: AssetRecord) -> list[dict]:
        url = (
            f"{_HUB_URL}/api/datasets/{quote(record.repo_id, safe='/')}/tree/{record.revision}/"
            f"{quote(record.directory, safe='/')}?recursive=true&limit=100"
        )
        entries = {}
        seen = set()
        while url:
            self._check_cancelled()
            if url in seen or urlsplit(url).netloc != urlsplit(_HUB_URL).netloc:
                raise ValueError("Invalid asset tree pagination")
            seen.add(url)
            data, link = self._request(url)
            for entry in json.loads(data):
                if entry["type"] != "file":
                    continue
                prefix = record.directory + "/"
                if not entry["path"].startswith(prefix):
                    raise ValueError("Asset tree returned a file outside its package")
                name = _relative(entry["path"][len(prefix) :])
                entries[name] = entry
            match = re.search(r'<([^>]+)>;\s*rel="next"', link)
            url = urljoin(url, match.group(1)) if match else ""
        return [{"name": name, **entry} for name, entry in entries.items()]

    def _check_cache(self, record: AssetRecord) -> Path | None:
        package = self._package(record)
        marker = package / "complete.json"
        if not marker.is_file():
            return None
        try:
            saved = json.loads(marker.read_bytes())
            if saved["id"] != record.id or not saved["files"]:
                return None
            for entry in saved["files"]:
                self._check_cancelled()
                path = package / "content" / _relative(entry["name"])
                if not path.is_file() or path.stat().st_size != entry["size"] or _sha256(path) != entry["sha256"]:
                    return None
            root = package / "content" / _relative(saved["root"])
            return root if root.is_file() else None
        except (KeyError, ValueError, OSError):
            return None

    def _download(self, record: AssetRecord) -> Path:
        existing = self._check_cache(record)
        if existing is not None:
            return existing
        if self._offline():
            raise RuntimeError("Offline mode: this package is missing or incomplete in the cache")
        package = self._package(record)
        marker = package / "complete.json"
        marker.unlink(missing_ok=True)
        entries = self._tree(record)
        if not entries:
            raise ValueError(f"Empty package: {record.name}")
        names = {entry["name"] for entry in entries}
        if not {_MANIFEST, _ROOTS, _BOM}.issubset(names):
            raise ValueError("Package tree is missing required SimReady metadata")
        content = package / "content"
        for index, entry in enumerate(entries):
            self._messages.put(f"{record.name}: file {index + 1} / {len(entries)}")
            path = content / entry["name"]
            self._request(self._file_url(record, entry["name"]), path)
            if path.stat().st_size != entry["size"]:
                raise ValueError(f"Size mismatch: {entry['name']}")
            expected = entry.get("lfs", {}).get("oid")
            if expected and _sha256(path) != expected:
                raise ValueError(f"SHA256 mismatch: {entry['name']}")
        manifest = self._metadata(record, _MANIFEST)
        expected_bom = next(
            (item for item in manifest.get("metadata", []) if item.get("name") == PurePosixPath(_BOM).name), None
        )
        if expected_bom is None or _sha256(content / _BOM) != expected_bom.get("hash", {}).get("sha256"):
            raise ValueError("Package bill of materials failed SHA256 verification")
        items = self._metadata(record, _BOM).get("items", [])
        if not items:
            raise ValueError("Package bill of materials is empty")
        for item in items:
            name = _relative(item["relative_path"])
            path = content / name
            if (
                name not in names
                or path.stat().st_size != item["size"]
                or _sha256(path) != item.get("hash", {}).get("sha256")
            ):
                raise ValueError(f"Package content failed SHA256 verification: {name}")
        root = self._root(record)
        if root not in names:
            raise ValueError("Root USD is missing from the package")
        files = [
            {"name": entry["name"], "size": entry["size"], "sha256": _sha256(content / entry["name"])}
            for entry in entries
        ]
        self._check_cancelled()
        _atomic(marker, json.dumps({"id": record.id, "root": root, "files": files}).encode())
        return content / root
