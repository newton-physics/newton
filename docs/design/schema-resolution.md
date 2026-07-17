<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Source-neutral USD schema resolution

Status: design proposal

## Summary

Newton's schema resolvers currently read `Usd.Prim` objects, transform raw
attributes, choose between configured schemas, and provide mapping defaults.
Some source-sensitive precedence remains in `import_usd.py`. This couples the
resolution policy to the PXR importer and prevents other populated scene
representations from reusing the same behavior.

This proposal separates attribute access from scalar resolution while keeping
the current `ModelBuilder.add_usd()` construction path and resolver behavior by
default. The first implementation adds an opt-in source-neutral engine shared
by PXR and non-PXR consumers. Applied
schema identity is part of the input: an applied schema owns the properties it
defines, including its USD fallbacks. Resolver definitions identify ownership
and conversion. A private catalog supplies built-in Newton, PhysX, and MuJoCo
fallbacks when their schema plugins are unavailable. Batched resolution, Warp
functions, and direct ModelBuilder buffer population are deliberately deferred.

## Goals

- Keep core Newton as the single owner of schema precedence and conversion.
- Preserve the behavior of `add_usd()` and `schema_resolvers=` by default.
- Resolve values and applied schemas supplied without a `Usd.Prim`.
- Read fallbacks from registered USD schemas on the PXR path.
- Share built-in fallbacks with schema-neutral sources.
- Let callers override or extend the fallback catalog for other schema versions.
- Keep the external mechanism small and hide resolver definitions and storage.
- Make the scalar contract suitable for later columnar and Warp execution.
- Allow ovnewton to adopt the resolver without changing its current build loop.

## Non-goals

- Changing canonical USDPhysics discovery, topology, or geometry parsing.
- Replacing `ModelBuilder.add_*()` with direct array construction.
- Defining an ovstage column layout or public batch API.
- Making arbitrary downstream resolver definitions device-capable.
- Introducing a scene intermediate representation.

## Current behavior

Before this change, `SchemaResolverManager.get_value()` applied this order:

1. The first authored value in resolver priority order.
2. The caller-provided importer default, when non-`None`.
3. The first non-`None` mapping default in resolver priority order.
4. `None`.

That makes an unapplied schema contribute defaults and lets an importer default
override the fallback of an applied schema. It also disagrees with specialized
paths such as MuJoCo joint-limit resolution.

The opt-in order is evaluated per resolver, in priority order:

1. Use a source value when present.
2. If the schema owning the property is applied, use its USD fallback and stop.
3. Otherwise continue to the next resolver.
4. Use the caller default only when no resolver owns the property.

Typed schemas participate like applied API schemas. Legacy mapping defaults are
retained for existing `schema_resolvers=` calls but ignored by the opt-in path
when a mapping declares schema ownership. Unowned entries in the built-in
resolvers remain authored-only in the opt-in path. A schema fallback may itself
be an engine-default sentinel; using the builder in response to that sentinel
is still schema-defined behavior.

## Proposed boundary

The resolver engine consumes source values and schema identity:

```text
read(attribute_name) -> value or missing
schemas -> applied API and typed schema names
fallback(schema_name, attribute_name) -> raw USD value or missing
```

The PXR adapter reads authored values and applied/type metadata, then asks
`Usd.SchemaRegistry` for the composed prim definition and its attribute
fallbacks. A populated source provides already-composed values and recorded
schema identity; the resolver uses Newton's private fallback catalog. Resolver
priority, transformations, and selected-source provenance are shared. If an
applied schema owns a requested property but neither the registry nor catalog
can provide its fallback, resolution fails instead of silently using a builder
default.

Canonical USDPhysics descriptors remain importer inputs. Resolvers only supply
extension-schema properties such as armature, friction, contact parameters,
initial joint state, and Newton-specific collision configuration.

## Compatibility strategy

The existing public classes and methods remain valid:

- `SchemaResolver` and its built-in subclasses.
- `SchemaResolver.get_value(prim, ...)`.
- `ModelBuilder.add_usd(..., schema_resolvers=...)`.

Existing `schema_resolvers=` calls retain their authored value, caller default,
and mapping default precedence. The new behavior is selected by passing the
opaque facade instead:

```python
resolution = newton.usd.create_schema_resolution(
    resolvers,
)
builder.add_usd(stage, schema_resolution=resolution)
```

`schema_resolvers` and `schema_resolution` are mutually exclusive.

Built-in composite getters are migrated to the source-neutral reader. The
existing `usd_value_getter` callback remains supported for downstream resolver
subclasses on the legacy PXR path. Attempting to use such a resolver through
the source-neutral facade raises an actionable error rather than silently
changing behavior.

The opt-in facade is created through one entry point:

```python
resolution = newton.usd.create_schema_resolution(
    resolvers,
    # Only needed for custom schemas or version-specific overrides.
    schema_fallbacks={
        "ExampleJointAPI": {"example:armature": 0.0},
    },
)
requirements = resolution.requirements(PrimType.JOINT)
schemas = resolution.schemas(PrimType.JOINT)
properties = resolution.resolve(
    PrimType.JOINT,
    values,
    schemas=applied_schemas,
    defaults={"armature": builder.default_joint_cfg.armature},
)
```

Fallback tables contain raw USD values; the common resolver applies the same
composite getter and value transformer as it does to authored values. These
methods accept an optional `keys=` selection so an integration can request only
the logical properties it can preserve.

The returned object is opaque. Candidate types, provenance, mapping
definitions, transforms, diagnostics, and future Warp storage remain internal.
`requirements()` returns source attribute names, `schemas()` returns the schema
identities needed for ownership, and `resolve()` returns canonical logical keys
and values. This scalar mapping interface is intended for the existing
`ModelBuilder.add_*()` path, not the future hot path.

## `add_usd()` migration

When `schema_resolution` is provided, `SchemaResolverManager` delegates priority
and fallback resolution to the source-neutral engine through a PXR attribute
source. It caches composed schema fallbacks by prim type and schema. An
explicitly supplied fallback table is used only when the registered schema
cannot provide the property. Without the opt-in object, the existing resolver
algorithm remains unchanged. Attribute collection for the returned
`schema_attrs` dictionary remains separate and unchanged.

The first implementation does not move every higher-level rule out of
`import_usd.py`. Source-specific branches such as MuJoCo raw-limit provenance,
material-versus-shape contact precedence, and legacy margin handling migrate in
small behavior-preserving changes after the common scalar engine is established.

## Ovnewton migration

Ovnewton supplies populated columns plus `usd-schemas` and `usd-prim-type`
identity to the facade inside its existing body, shape, and joint loops. It uses
Newton's built-in fallback catalog, so it owns no schema-default table. Its
topology discovery, ordering, and `ModelBuilder.add_*()` calls remain unchanged.
Equivalent local precedence and conversion code is removed as each entity
family moves to the shared engine.

A populated source must preserve the USD fallback semantics expected by the
resolver. A source that substitutes an engine descriptor default for a USD
fallback cannot provide exact parity for that property without provenance or a
source-contract change. Any temporary value normalization belongs in the source
adapter and must document the ambiguous explicitly-authored value.

This is intentionally scalar. It establishes semantic parity before optimizing
data movement.

## Future batch execution

A later design may bind `requirements()` to aligned source columns and execute
one kernel per entity family. Built-in transforms can then be represented by
internal operation identifiers with Python and `wp.func` implementations.
ModelBuilder must first provide reserved numeric destination ranges; otherwise
device results would be converted back into Python lists.

No batch types or Warp implementation details are part of the initial public
contract.

## Diagnostics

The scalar path preserves existing Python warnings. A future device path cannot
emit warnings from a `wp.func`; it will return internal diagnostic bits for the
host to report after execution. Diagnostic representation is not public.

## Testing

- Existing schema resolver and `add_usd()` APIs remain valid.
- Tests cover absent schemas, applied-but-unauthored fallbacks, authored zero,
  authored nonzero, resolver ordering, transformations, and missing values.
- The same values and schema identity resolve equally through PXR and mapping
  sources.
- Every built-in owned mapping has an entry in the private fallback catalog.
- Registered Newton schema fallbacks match the catalog exactly.
- Legacy PXR-only callbacks continue to work and fail explicitly through the
  source-neutral facade.
- Later ovnewton differential tests compare final builder/model fields against
  `add_usd()` for the same asset.

## Alternatives

### Duplicate the mappings in ovnewton

This keeps the implementations independent but guarantees semantic drift and
duplicates every future schema fix.

### Make ovnewton construct temporary USD prims

This retains the current resolver but restores the PXR dependency and defeats
the populated-stage architecture.

### Standardize a scene IR first

An IR would combine schema resolution, topology, and construction concerns.
The scalar source boundary is sufficient for reuse and does not constrain a
future bulk ModelBuilder design.

### Expose candidate and Warp structures publicly

This would freeze implementation details before the column and builder storage
contracts exist. An opaque facade leaves those choices open.

## Rollout

1. Introduce the source abstraction and shared scalar engine.
2. Declare property ownership for built-in resolver mappings while retaining
   their legacy compatibility defaults.
3. Add an explicit `add_usd()` opt-in to the engine.
4. Expose the opt-in opaque facade and source-neutral tests.
5. Move higher-level `add_usd()` resolver policy into shared entity helpers.
6. Adopt those helpers incrementally in ovnewton.
7. Design batch column binding and ModelBuilder reservation separately.

## Open questions

- Whether downstream custom resolver authoring should remain supported publicly.
- Which higher-level importer rules belong in entity resolvers versus canonical
  USDPhysics lowering.
- Whether the eventual batch entry point belongs to `newton.usd` or
  `ModelBuilder`.
