# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Adapt existing MJCF fixtures to the optional converter without repairing their contents."""

import inspect
import os
from pathlib import Path

import newton
from newton.solvers import SolverMuJoCo
from newton.usd import SchemaResolverMjc, SchemaResolverNewton

_MJCF_SIGNATURE = inspect.signature(newton.ModelBuilder.add_mjcf)
_USD_OPTIONS = set(inspect.signature(newton.ModelBuilder.add_usd).parameters)


class UsdComparisonImporter:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.records = []

    def __call__(self, builder, source, *args, **kwargs):
        from mujoco_usd_converter import Converter  # noqa: PLC0415

        record = {"call": len(self.records), "phase": "adapter", "status": "running"}
        self.records.append(record)
        directory = self.directory / f"import-{record['call']:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        record["directory"] = str(directory)
        try:
            arguments = _MJCF_SIGNATURE.bind(builder, source, *args, **kwargs).arguments
            options = {}
            aliases = {"parse_sites": "load_sites", "parse_visuals": "load_visual_shapes"}
            unsupported = []
            for name, value in arguments.items():
                if name in ("self", "source"):
                    continue
                if name in aliases:
                    options[aliases[name]] = value
                elif name in _USD_OPTIONS:
                    options[name] = value
                elif value != _MJCF_SIGNATURE.parameters[name].default:
                    unsupported.append(name)
            if unsupported:
                raise NotImplementedError(f"USD comparison adapter has no mapping for: {', '.join(unsupported)}")

            source = os.fspath(source)
            if source.lstrip().startswith("<"):
                source_path = directory / "source.xml"
                source_path.write_text(source, encoding="utf-8")
            else:
                source_path = Path(source).absolute()
            record["source"] = str(source_path)
            if source_path.is_file():
                record["source_xml"] = source_path.read_text(encoding="utf-8")
            record["phase"] = "conversion"
            asset = Converter().convert(str(source_path), str(directory / "usd"))
            record["asset"] = asset.path
            record["phase"] = "usd_import"
            SolverMuJoCo.register_custom_attributes(builder)
            result = builder.add_usd(
                asset.path,
                schema_resolvers=[SchemaResolverMjc(), SchemaResolverNewton()],
                **options,
            )
            record["status"] = "passed"
            return result
        except Exception as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
