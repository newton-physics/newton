---
orphan: true
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# USD schema fallback resolution

## Status

This document describes the design implemented by PR #3888. The PR continues and supersedes PR #3572, addresses the fallback-precedence part of issue #3307, and prepares the internal seam used by PR #3984, which continues and supersedes PR #3568. Complete public provenance remains future work.

## Goal

USD import can obtain a property from several places that were previously all called defaults:

- an authored USD value;
- a fallback owned by a registered USD schema;
- an importer default from `ModelBuilder` or `add_usd()`;
- a compatibility default retained by a resolver; or
- an explicit importer override.

These sources have different authority. The resolution policy makes that authority explicit while preserving the old behavior during migration.

## Resolution policies

The {ref}`schema resolver guide <schema_resolvers>` is authoritative for user-visible behavior. The two policies are summarized here to explain the implementation:

| Policy | Resolution order |
| --- | --- |
| Legacy: omitted or `False` | Authored value → importer default → compatibility default → unresolved. |
| Registered schema: `True` | Explicit override → each resolver's authored value or registered fallback → importer default → eligible compatibility default → unresolved. |

Legacy behavior is deprecated and audited against registered-schema precedence. One quirk is intentionally preserved: after selecting a compatibility default, a property transformer may turn it into `None`, but resolution does not continue to a later compatibility default.

Under registered-schema precedence, resolver priority applies to each resolver's complete candidate. An earlier resolver's registered fallback therefore wins over a later resolver's authored value.

### Omission and override

Selected `add_usd()` arguments distinguish omission from an explicit value. When omitted, their normal Python default participates as an importer default. When explicitly passed under registered-schema precedence, they override USD resolution.

The omission-tracking wrapper preserves ordinary typed defaults for introspection and static analysis. The currently supported options are `enable_self_collisions`, `mesh_maxhullvert`, `joint_drive_gains_scaling`, and `collapse_fixed_joints`. Every keyword-only `add_usd()` argument must be classified exactly once so new controls cannot enter property resolution accidentally.

## Module design

The implementation has three roles:

| Module | Responsibility |
| --- | --- |
| `schema_resolver.py` | Select candidates, adapt PXR schema data, cache reads, and collect migration diagnostics. |
| `_usd_resolution_policy.py` | Interpret importer properties and assemble values that depend on several USD inputs. |
| `import_usd.py` | Traverse the stage, parse geometry, and mutate `ModelBuilder`. |

`_SchemaResolutionPolicy` contains the source-neutral selection algorithm. It receives callbacks for reading values, checking schema applicability, and reading registered fallbacks. `SchemaResolverManager` supplies the PXR-backed callbacks and preserves custom-resolver compatibility.

`_UsdResolutionPolicy` is private to the importer. It groups property logic by consumer—for example scene, joint, material, contact, and collision-shape resolution—without traversing the stage or mutating the builder. This keeps property interpretation out of the already large `import_usd.py` function without creating a second importer. Specialized consumers may customize legacy behavior, but must reuse the shared registered-schema selection policy.

## Correctness rules

### Schema ownership

Registered fallbacks come from the composed prim definition in `Usd.SchemaRegistry`. Newton does not copy vendor fallback catalogs into Python.

The `newton[importers]` extra requires `newton-usd-schemas>=0.5.0`; the repository lock currently tests version 0.5.0. PhysX and MuJoCo schema plugins are optional. Their resolvers still read authored vendor attributes when a plugin is missing, but only a registered plugin can provide authoritative schema fallbacks. The schema packages, rather than Newton, own plugin registration and fallback metadata.

Registration and fallback presence are distinct:

| Schema state | Result |
| --- | --- |
| Registered property with fallback | The fallback is authoritative. |
| Registered property without fallback | The schema supplies no fallback for that property. |
| Unregistered schema | A compatibility default may be considered after the importer default. |

Newton's built-in `SchemaResolver` subclasses declare ownership internally with `_schema_ownership`. A schema name owns every mapped key for a prim type, while a nested mapping declares ownership per key. `_use_compatibility_defaults` controls whether unregistered or unowned mappings retain their compatibility defaults. Custom resolvers without an internal ownership declaration keep their existing compatibility behavior.

### Usable candidates

Registered-schema resolution continues until it finds a usable resolution candidate. A candidate is unusable when its resolver getter or transformer returns `None`. This rule applies while selecting authored values, registered fallbacks, and compatibility defaults; it does not restart resolution after the importer has selected and interpreted a candidate.

A property can mark a registered fallback as unset. Sentinel rules remain property-specific: an unlimited velocity fallback is absent, while an authored hull value of `-1` is an exact unbounded choice.

A blocked attribute is unauthored, but its applicable registered fallback remains eligible under Newton's accepted contract. If a PXR-only custom getter cannot evaluate one fallback candidate, only that candidate is skipped.

Reader-backed compound properties resolve each constituent independently. Authored constituents are retained and missing or blocked constituents can come from the registered schema definition. The mixed result is classified as a registered fallback; per-constituent provenance is deferred to PR #3984.

### Property interpretation

Migration auditing compares the value observed by the importer, not the raw USD value. After candidate selection, property-specific interpreters handle unit conversion, sentinels, validation, and source-dependent behavior. Consumer interpretation does not resume candidate selection. The importer consumes the same interpreted result used by the audit.

Properties assembled from several inputs are compared after assembly. Examples include joint-limit gains, contact response, and SDF configuration. Warnings name only the inputs that contribute to the interpreted change.

Omitted SDF padding is compared with the gap selected by the same policy. For a colliding hydroelastic shape, the comparison also includes that policy's margin.

The audit stops at the property seam rather than shadow-running all model construction. An unrelated downstream rule can therefore mask a reported property change without making the property resolution itself unchanged.

Equal numbers are normally equivalent. Source provenance is compared only when the source changes consumer behavior, such as MuJoCo joint-limit modes.

## Migration diagnostics

Legacy imports resolve the registered-schema result using cached reads and emit at most one aggregated `DeprecationWarning` per import. The warning is emitted only for a proven interpreted value or relevant source change; audit failures alone do not warn. It identifies the old and new suppliers, bounds the list of prim paths, and explains how to adopt or preserve the behavior.

An explicit importer override suppresses migration auditing for that property because the caller has already selected its future value.

## Performance

The policy extraction adds no stage traversal and does not build a second model. Registered-schema mode evaluates only the selected policy. Legacy mode also evaluates the future result for migration diagnostics, but authored reads and composed schema definitions are cached for the import.

Local before-and-after timings should cover a large authored-value stage and a fallback-heavy stage. This is an engineering check rather than a committed benchmark or public performance guarantee.

## Extending the policy

When adding a resolved property:

1. Add its mapping and, for a built-in resolver, its internal schema ownership.
2. Separate its importer default from any explicit override.
3. Define property-specific unset and interpretation rules.
4. Use ordinary resolution unless the consumer observes source semantics.
5. Compare the final interpreted or assembled property.
6. Test both policies and the terminal sources relevant to that property.

## Future work

This PR does not:

- package PhysX or MuJoCo codeless schema plugins;
- copy vendor fallback catalogs;
- migrate the remaining deformable body and material proposal attributes;
- expose complete public resolver/source provenance or dynamic applicability;
- turn every `add_usd()` argument into a resolved option; or
- remove the legacy policy.

PR #3984 builds the public, source-neutral provenance API on this seam. Codeless vendor schemas and deformable schema resolution remain separate integration tasks.
