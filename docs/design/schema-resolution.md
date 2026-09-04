<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Source-neutral USD schema resolution

Status: implemented scalar design

## Summary

Newton's schema resolution separates attribute access from scalar resolution
while keeping the current `ModelBuilder.add_usd()` construction path. The
typed `SchemaResolution` object configures PXR and non-PXR consumers, while the
existing `schema_resolvers=` argument remains a compatibility shorthand for
the same internal policy.

Applied schema identity is part of the input: an applied schema owns the
properties it defines, including its USD fallbacks. Resolver definitions
identify ownership and conversion. The source adapter supplies registered
schema fallbacks: the PXR adapter reads them from `Usd.SchemaRegistry`, while a
PXR-free adapter provides the equivalent versioned schema metadata explicitly.
Property interpretation and multi-input assembly live in the private
`_usd_resolution_policy.py` module. Each source-neutral result reports its
value, source category, resolver, schema ownership, and source attributes.
Batched resolution, Warp functions, dynamic schema discovery, and direct
ModelBuilder buffer population are deliberately deferred.

## Resolution flow

This change shares the rules for choosing property values. Scene traversal, geometry, topology, and `ModelBuilder` construction remain the responsibility of each importer.

```{mermaid}
:config: {"theme": "forest", "themeVariables": {"lineColor": "#76b900"}}

flowchart LR
    Setup["SchemaResolution<br/>resolver order + fallback mode<br/>requirements() + schemas()"]

    subgraph EntryPoints["Resolution entry points"]
        PXR["SchemaResolverManager<br/>PXR prim + SchemaRegistry"]
        Mapping["SchemaResolution.resolve()<br/>values + schemas + fallbacks"]
    end

    Shared["_SchemaResolutionPolicy<br/>candidate selection + conversions"]

    subgraph Consumers["Importer-owned work"]
        Newton["_UsdResolutionPolicy<br/>interpret and assemble properties"]
        Result["SchemaResolution.Result<br/>value + source + resolver<br/>schema + attributes"]
        Builder["ModelBuilder"]
        Other["Non-PXR importer<br/>for example, OVNewton"]
    end

    Setup -. "configures" .-> PXR
    Setup -. "configures" .-> Mapping
    PXR -- "PXR-backed callbacks" --> Shared
    Mapping -- "mapping-backed callbacks" --> Shared
    Shared -- "PXR path" --> Newton --> Builder
    Shared -- "mapping path" --> Result --> Other
```

The two entry points differ only in how they provide attribute values, schema applicability, and registered fallbacks. The shared policy selects and converts a value. On the PXR path, the existing importer continues interpreting that value and updating `ModelBuilder`. On the mapping path, `SchemaResolution.Result` also reports where the value came from so a non-PXR importer can reuse the same schema-resolution rules.

## Goals

- Keep core Newton as the single owner of schema precedence and conversion.
- Preserve the behavior of `add_usd()` and `schema_resolvers=` by default.
- Resolve values and applied schemas supplied without a `Usd.Prim`.
- Read fallbacks from registered USD schemas on the PXR path.
- Accept registered schema fallbacks from schema-neutral source adapters.
- Keep schema metadata versioning in the adapter that owns the scene source.
- Keep the public interface small and hide candidates and storage.
- Report the winning source without exposing internal candidate state.
- Make the scalar contract suitable for later columnar and Warp execution.
- Let non-PXR scene consumers reuse the resolver without changing their
  construction loops.

## Non-goals

- Changing canonical USDPhysics discovery, topology, or geometry parsing.
- Replacing `ModelBuilder.add_*()` with direct array construction.
- Defining an external scene representation or public batch interface.
- Making arbitrary downstream resolver definitions device-capable.
- Introducing a scene intermediate representation.
- Applying importer-specific SDF, contact, joint-axis, or builder rules to
  source-neutral results.

## Current behavior

Before this change, `SchemaResolverManager.get_value()` applied this order:

1. The first authored value in resolver priority order.
2. The caller-provided importer default, when non-`None`.
3. The first non-`None` mapping default in resolver priority order.
4. `None`.

That makes an unapplied schema contribute defaults and lets an importer default
override the fallback of an applied schema. It also disagrees with specialized
paths such as MuJoCo joint-limit resolution.

Registered-schema precedence is evaluated per resolver, in priority order:

1. Use a usable authored value.
2. If a registered schema owns the property, use its usable USD fallback.
3. Otherwise continue to the next resolver.
4. Use the importer default when no resolver supplies a candidate.
5. Use an eligible compatibility default from an unregistered or unowned
   resolver mapping.

Typed schemas participate like applied API schemas. By default, the importer
returns the legacy value without evaluating registered-schema precedence.
Callers can explicitly audit the registered-schema result with
`audit_registered_schema_fallbacks=True`; the audit emits a
`DeprecationWarning` when the interpreted property would change. Callers can
instead select registered-schema precedence with
`use_registered_schema_fallbacks=True`, either on `SchemaResolution` or directly
on `add_usd()` when no shared object is supplied. Unowned and unregistered
entries may retain compatibility defaults after importer defaults. A schema
fallback may itself be an engine-default sentinel; property-specific handling
decides whether that candidate is usable.

## Implemented boundary

The resolver engine consumes source values and schema identity:

```text
read(attribute_name) -> authored value, blocked value, or missing
schemas -> every applicable API and typed schema identity
fallback(schema_name, attribute_name) -> raw USD value or missing
```

The PXR adapter reads authored values and applied/type metadata, then asks
`Usd.SchemaRegistry` for the composed prim definition and its attribute
fallbacks. A non-PXR source provides already-composed values, applicable schema
identity, and the fallbacks for schemas it treats as registered. The adapter
expands typed-schema ancestry when its source supports it. Resolver priority,
transformations, candidate selection, and selected-source provenance are
shared.

Schemas without authoritative metadata remain a second-class compatibility
path. Their authored values still participate in resolver priority, while
importer defaults precede resolver compatibility defaults. They do not claim
registered-schema fallback ownership.

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
    use_registered_schema_fallbacks=True,
)
builder.add_usd(stage, schema_resolution=resolution)
```

`schema_resolvers` and `schema_resolution` are mutually exclusive. The former
is a compatibility shorthand for configuring the same internal policy. The
shared object owns the fallback-policy and audit choices, so explicitly passing
either setting at the same time is rejected. The importer tracks omission
separately while preserving the public `False` defaults for signature
inspection.

After consumers have had time to migrate to a reusable object, the shorthand
can be deprecated and then removed without changing resolution semantics.

Built-in composite getters are migrated to the source-neutral reader. The
existing `usd_value_getter` callback remains supported for downstream resolver
subclasses on the PXR adapter. Attempting to use such a resolver with mapping
inputs raises an actionable error rather than silently changing behavior.

The same object resolves mapping inputs directly:

```python
resolution = newton.usd.SchemaResolution(
    resolvers,
    use_registered_schema_fallbacks=True,
)
requirements = resolution.requirements(PrimType.JOINT)
schemas = resolution.schemas(PrimType.JOINT)
properties = resolution.resolve(
    PrimType.JOINT,
    values,
    schemas=applied_schemas,
    # Registered schema metadata supplied by this source adapter.
    schema_fallbacks={
        "ExampleJointAPI": {"example:armature": 0.0},
    },
    defaults={"armature": builder.default_joint_cfg.armature},
)

armature = properties["armature"]
print(armature.value, armature.source, armature.resolver)
```

Default-map membership distinguishes omission from an explicit `None` value.
The PXR adapter keeps the same distinction: omitting its default supplies no
candidate, while passing `None` selects an explicit null importer default.

Fallback tables contain raw USD values; the common resolver applies the same
composite getter and value transformer as it does to authored values. These
methods accept an optional `keys=` selection so an integration can request only
the logical properties it can preserve.

`SchemaResolution` is public and typed; its internals are not. Candidate types,
mapping definitions, transforms, diagnostics, compatibility policy, and future
Warp storage remain private. `requirements()` returns source attribute names,
`schemas()` returns the schema identities needed for ownership, and `resolve()`
returns one `SchemaResolution.Result` per logical key. A result contains the
canonical value, its `SchemaResolution.Source`, the winning resolver, declared
schema ownership, and source attribute names. This scalar mapping interface is
intended for the existing `ModelBuilder.add_*()` path, not the future hot path.

## `add_usd()` migration

`ModelBuilder.add_usd()` delegates priority and fallback resolution to
`SchemaResolverManager`, which is the PXR adapter. The manager accepts either a
shared `SchemaResolution` or the `schema_resolvers=` shorthand and owns their
mutual-exclusion and policy validation. The importer supplies the Newton
resolver when neither configuration is supplied.

The adapter caches composed schema fallbacks by prim type and schema.
`Usd.SchemaRegistry` remains authoritative on this path; `schema_fallbacks` is
input for source-neutral `resolve()` calls. Attribute collection for the
returned `schema_attrs` dictionary remains separate and unchanged.

The private `_UsdResolutionPolicy` handles importer-facing interpretation and
assembly, including MuJoCo joint-limit meaning, material-versus-shape contact
precedence, margin and gap compatibility, joint properties, and SDF settings.
It does not traverse the stage or mutate the builder.

## Non-PXR source integration

A schema-aware scene source supplies attribute values plus every typed and
applied schema identity that is applicable to the same `SchemaResolution`
inside its existing body, shape, and joint loops. It also supplies the
registered fallbacks for the schema versions its transport exposes. Topology
discovery, ordering, and `ModelBuilder.add_*()` calls remain source concerns.
Equivalent local precedence code can be removed as each entity family moves to
the shared engine. Source-specific construction and importer interpretation do
not move into schema resolution.

A non-PXR source must preserve the USD fallback semantics expected by the
resolver. A source that substitutes an engine descriptor default for a USD
fallback cannot provide exact parity for that property without provenance or a
source-contract change. Any temporary value normalization belongs in the source
adapter and must document whether the value was authored or substituted.

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
- Adapter-supplied fallbacks establish registered ownership without importing
  PXR.
- Every terminal source reports its value and winning provenance.
- Registered Newton schema fallbacks are read directly from the PXR registry.
- Legacy PXR-only callbacks continue to work and fail explicitly through the
  source-neutral facade.
- Cross-source tests compare mapping results with PXR-backed resolution for the
  same values, schemas, and registered fallbacks.

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

## Implementation status

The typed setup object, shared scalar engine, PXR adapter, ownership metadata,
public scalar provenance, opt-in migration audit, and importer property policy
are implemented. The default returns legacy values without evaluating
registered-schema precedence.

Remaining work is to adopt the scalar facade in non-PXR scene integrations,
expand adapter-specific schema applicability where needed, finish the
deprecation window, and design batch column binding separately.

## Open questions

- Whether downstream custom resolver authoring should remain supported publicly.
- Which higher-level importer rules belong in entity resolvers versus canonical
  USDPhysics lowering.
- Whether the eventual batch entry point belongs to `newton.usd` or
  `ModelBuilder`.
