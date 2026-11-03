# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check installed-wheel documentation offline, outside a source checkout."""

import json
import socket
from importlib.metadata import distribution
from pathlib import Path


def check() -> None:
    def offline(*args, **kwargs):
        raise AssertionError("Installed documentation attempted network access")

    socket.create_connection = offline
    socket.getaddrinfo = offline
    socket.socket.connect = offline
    installed = distribution("newton")
    root = Path(installed.locate_file("newton/docs"))
    declared_licenses = installed.metadata.get_all("License-File", [])
    for notice in (root.parent / "licenses").glob("*.txt"):
        relative = notice.relative_to(root.parent.parent).as_posix()
        assert relative in declared_licenses, f"Missing license metadata: {relative}"
        copies = [path for path in installed.files if str(path).endswith(f".dist-info/licenses/{relative}")]
        assert len(copies) == 1, f"Missing license copy: {relative}"
        assert Path(installed.locate_file(copies[0])).read_bytes() == notice.read_bytes(), relative
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == installed.version, manifest
    assert len(manifest["source_digest"]) == 64, manifest
    for page, expected in (
        ("guide/installation.md", "Installation"),
        ("concepts/conventions.md", "Conventions"),
        ("solvers/vbd.md", "experimental"),
        ("faq.md", "FAQ"),
        ("tutorials/00_introduction.md", "import newton"),
        ("api/_generated/newton.ModelBuilder.md", "finalize"),
    ):
        assert expected in (root / page).read_text(encoding="utf-8"), page
    for line in (root / "llms.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("- ["):
            target = line.rsplit("](", 1)[1].removesuffix(")")
            if "://" not in target:
                assert (root / target).is_file(), target
    print(f"Verified offline Newton {installed.version} documentation in {root}")


if __name__ == "__main__":
    check()
