<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Source-neutral USD schema resolution

Status: design proposal

## Summary

Newton's schema resolvers currently read `Usd.Prim` objects, transform raw
attributes, choose between configured schemas, and provide mapping defaults.
Some source-sensitive precedence remains in `import_usd.py`. This couples the
resolution policy to the PXR importer and prevents scene representations that
do not expose PXR prims from reusing the same behavior.

This proposal separates attribute access from scalar resolution while keeping
the current `ModelBuilder.add_usd()` construction path and behavior. One typed
`SchemaResolution` object configures both PXR and non-PXR consumers; the
existing `schema_resolvers=` argument constructs that object internally.
Applied schema identity is part of the input: an applied schema owns the
properties it defines, including its USD fallbacks. Resolver definitions
identify ownership and conversion. A private catalog supplies built-in Newton,
PhysX, and MuJoCo fallbacks when their schema plugins are unavailable. Batched
resolution, Warp functions, and direct ModelBuilder buffer population are
deliberately deferred.

## Goals

- Keep core Newton as the single owner of schema precedence and conversion.
- Preserve the behavior of `add_usd()` and `schema_resolvers=` by default.
- Resolve values and applied schemas supplied without a `Usd.Prim`.
- Read fallbacks from registered USD schemas on the PXR path.
- Share built-in fallbacks with schema-neutral sources.
- Let callers override or extend the fallback catalog for other schema versions.
- Keep the external mechanism small and hide candidates, provenance, and storage.
- Make the scalar contract suitable for later columnar and Warp execution.
- Let non-PXR scene consumers reuse the resolver without changing their
  construction loops.

## Non-goals

- Changing canonical USDPhysics discovery, topology, or geometry parsing.
- Replacing `ModelBuilder.add_*()` with direct array construction.
- Defining an external scene representation or public batch API.
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

The composed order is evaluated per resolver, in priority order:

1. Use a source value when present.
2. If the schema owning the property is applied, use its USD fallback and stop.
3. Otherwise continue to the next resolver.
4. Use the caller default only when no resolver owns the property.

Typed schemas participate like applied API schemas. During migration, a private
policy computes this result but returns the legacy value. If the results differ,
the public entry points emit a `DeprecationWarning`. The policy is shared by PXR
and mapping sources and is not user-selectable. A later release changes one
private default to return the composed result; the legacy branch can then be
removed after its deprecation window. Unowned entries retain their mapping
defaults. A schema fallback may itself be an engine-default sentinel; using the
builder in response to that sentinel is still schema-defined behavior.

## Proposed boundary

The resolver engine consumes source values and schema identity:

```text
read(attribute_name) -> value or missing
schemas -> applied API and typed schema names
fallback(schema_name, attribute_name) -> raw USD value or missing
```

The PXR adapter reads authored values and applied/type metadata, then asks
`Usd.SchemaRegistry` for the composed prim definition and its attribute
fallbacks. A non-PXR source provides already-composed values and recorded
schema identity; the resolver uses Newton's private fallback catalog. Resolver
priority, transformations, and selected-source provenance are shared. Once the
composed policy is active, an applied schema that owns a requested property but
has no fallback in either the registry or catalog fails instead of silently
using a builder default. During the audit period, that future error does not
break the legacy result.

Canonical USDPhysics descriptors remain importer inputs. Resolvers only supply
extension-schema properties such as armature, friction, contact parameters,
initial joint state, and Newton-specific collision configuration.

## Compatibility strategy

The existing public classes and methods remain valid:

- `SchemaResolver` and its built-in subclasses.
- `SchemaResolver.get_value(prim, ...)`.
- `ModelBuilder.add_usd(..., schema_resolvers=...)`.

`SchemaResolution` is the common setup object:

```python
resolution = newton.usd.SchemaResolution(
    resolvers,
)
builder.add_usd(stage, schema_resolution=resolution)
```

`schema_resolvers` and `schema_resolution` are mutually exclusive. The former
is a compatibility shorthand for constructing `SchemaResolution(resolvers)`;
it does not select another engine or fallback policy. After consumers have had
time to migrate to a reusable object, the shorthand can be deprecated and then
removed without changing resolution semantics.

Built-in composite getters are migrated to the source-neutral reader. The
existing `usd_value_getter` callback remains supported for downstream resolver
subclasses on the PXR adapter. Attempting to use such a resolver with mapping
inputs raises an actionable error rather than silently changing behavior.

The same object resolves mapping inputs directly:

```python
resolution = newton.usd.SchemaResolution(
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

`SchemaResolution` is public and typed; its internals are not. Candidate types,
provenance, mapping definitions, transforms, diagnostics, compatibility policy,
and future Warp storage remain private. `requirements()` returns source
attribute names, `schemas()` returns the schema identities needed for
ownership, and `resolve()` returns canonical logical keys and values. This
scalar mapping interface is intended for the existing `ModelBuilder.add_*()`
path, not the future hot path.

## `add_usd()` migration

`ModelBuilder.add_usd()` always delegates priority and fallback resolution to a
`SchemaResolution` through the PXR adapter. It constructs a default Newton
resolution when neither argument is supplied and wraps `schema_resolvers=` when
that shorthand is used. The adapter caches composed schema fallbacks by prim
type and schema. An explicitly supplied fallback table is used only when the
registered schema cannot provide the property. Attribute collection for the
returned `schema_attrs` dictionary remains separate and unchanged.

The first implementation does not move every higher-level rule out of
`import_usd.py`. Source-specific branches such as MuJoCo raw-limit provenance,
material-versus-shape contact precedence, and legacy margin handling migrate in
small behavior-preserving changes after the common scalar engine is established.

## Non-PXR source integration

A schema-aware scene source supplies attribute values plus typed and applied
schema identities to the same `SchemaResolution` inside its existing body,
shape, and joint loops. It uses Newton's built-in fallback catalog, so it owns
no schema-default table. Topology discovery, ordering, and
`ModelBuilder.add_*()` calls remain source concerns. Equivalent local
precedence and conversion code can be removed as each entity family moves to
the shared engine.

A non-PXR source must preserve the USD fallback semantics expected by the
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
- Later cross-source tests compare final builder/model fields against
  `add_usd()` for the same asset.

## Alternatives

### Duplicate mappings in each scene integration

This keeps the implementations independent but guarantees semantic drift and
duplicates every future schema fix.

### Construct temporary USD prims in non-PXR integrations

This retains the current resolver but restores the PXR dependency and defeats
the source-neutral boundary.

### Standardize a scene IR first

An IR would combine schema resolution, topology, and construction concerns.
The scalar source boundary is sufficient for reuse and does not constrain a
future bulk ModelBuilder design.

### Expose candidate and Warp structures publicly

This would freeze implementation details before the column and builder storage
contracts exist. The narrow typed facade leaves those choices open.

## Rollout

1. Introduce the typed setup object and shared scalar engine.
2. Route both `add_usd()` arguments through that object while retaining legacy
   outputs behind one private compatibility policy.
3. Declare property ownership and audit applied-schema fallback differences for
   both PXR and mapping sources.
4. Flip the private policy after the deprecation window, then remove the legacy
   branch and eventually the `schema_resolvers=` shorthand.
5. Move higher-level `add_usd()` resolver policy into shared entity helpers.
6. Adopt those helpers incrementally in non-PXR scene integrations.
7. Design batch column binding and ModelBuilder reservation separately.

## Open questions

- Whether downstream custom resolver authoring should remain supported publicly.
- Which higher-level importer rules belong in entity resolvers versus canonical
  USDPhysics lowering.
- Whether the eventual batch entry point belongs to `newton.usd` or
  `ModelBuilder`.
