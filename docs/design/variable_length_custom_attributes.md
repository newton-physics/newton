# Variable-Length Custom Attributes

## Problem Statement

Newton's custom attribute system previously only supported attributes tied to built-in entity counts (bodies, shapes, joints, etc.). This limitation meant that solver-specific data structures with independent cardinality—such as MuJoCo contact pairs—could not be represented as custom attributes.

**Specific use case:** MuJoCo's `<contact><pair>` elements define explicit contact pairs between geometries with custom solver parameters (friction, margin, condim, etc.). These pairs:
- Have their own count independent of bodies/shapes/joints
- Reference shapes by index (which must be remapped during `add_world()` merging)
- Need world assignment for multi-world simulations

The existing system couldn't handle this because:
1. `CustomAttribute.frequency` was required and had to match a `ModelAttributeFrequency` enum value
2. No mechanism existed to transform attribute values (e.g., offset shape indices) during builder merging
3. No API existed to append values to variable-length attributes

## Solution Overview

### 1. Variable-Length Frequency (`frequency=None`)

`CustomAttribute.frequency` is now optional. When `None`, the attribute's array length is determined by the number of values added, not by an entity count.

```python
# Before: frequency was required
CustomAttribute(
    name="my_attr",
    frequency=ModelAttributeFrequency.BODY,  # Required
    dtype=wp.float32,
)

# After: frequency can be None for variable-length
CustomAttribute(
    name="pair_geom1",
    frequency=None,  # Variable length
    dtype=wp.int32,
)
```

### 2. Reference Transformation (`references` field)

New `references` field specifies how attribute values should be transformed during `add_world()`/`add_builder()` merging.

```python
CustomAttribute(
    name="pair_geom1",
    frequency=None,
    dtype=wp.int32,
    references="shape",  # Values are shape indices, offset during merge
)
```

Supported reference types:
- **Built-in entities**: `"body"`, `"shape"`, `"joint"`, `"joint_dof"`, `"joint_coord"`, `"articulation"` — values are offset by entity count
- **Special**: `"world"` — values are replaced with `current_world` (not offset)
- **Custom attributes**: Any attribute key (e.g., `"mujoco:pair_geom1"`) — values are offset by that attribute's value count

### 3. Value Appending API (`add_custom_values()`)

New method to append values to variable-length attributes:

```python
builder.add_custom_values(**{
    "mujoco:pair_world": builder.current_world,
    "mujoco:pair_geom1": geom1_idx,
    "mujoco:pair_geom2": geom2_idx,
    "mujoco:pair_condim": 3,
})
```

Returns a dict mapping attribute keys to the indices where values were added.

### 4. Finalization Changes

During `finalize()`, variable-length attributes create arrays sized by `len(attr.values)`:

```python
if frequency is None:
    count = len(custom_attr.values) if custom_attr.values else 0
```

## Changes Made

### `newton/_src/sim/builder.py`

1. **`CustomAttribute` dataclass**:
   - `frequency` is now `ModelAttributeFrequency | None` (default `None`)
   - Added `references: str | None` field for value transformation

2. **`add_custom_values()` method**: New API for appending to variable-length attributes

3. **`add_builder()` merging logic**:
   - Handles `frequency=None` by offsetting indices by pre-merge value count
   - Applies `references` transformation to values (offset or replace)
   - Supports custom attribute cross-references

4. **`finalize()`**: Handles `frequency=None` by using `len(values)` as array size

### `newton/_src/sim/model.py`

1. **`ModelAttributeFrequency`**: Renamed `EQUALITY_CONSTRAINT` to `WORLD` (index 7)

### `newton/_src/solvers/mujoco/solver_mujoco.py`

1. **`register_custom_attributes()`**: Added 10 pair attributes with `frequency=None`:
   - `pair_world`, `pair_geom1`, `pair_geom2` (with appropriate `references`)
   - `pair_condim`, `pair_solref`, `pair_solreffriction`, `pair_solimp`
   - `pair_margin`, `pair_gap`, `pair_friction`

2. **`_validate_pair_attributes()`**: Validates all pair arrays have consistent lengths

3. **`_init_pairs()`**: Converts Newton pair attributes to MuJoCo spec pairs

### `newton/_src/utils/import_mjcf.py`

1. **Contact pair parsing**: Parses `<contact><pair>` elements using `add_custom_values()`

---

## Example: MuJoCo Contact Pairs

### Attribute Registration

```python
# In SolverMuJoCo.register_custom_attributes()
builder.add_custom_attribute(
    ModelBuilder.CustomAttribute(
        name="pair_world",
        frequency=None,
        dtype=wp.int32,
        default=0,
        namespace="mujoco",
        references="world",  # Replace with current_world during merge
    )
)
builder.add_custom_attribute(
    ModelBuilder.CustomAttribute(
        name="pair_geom1",
        frequency=None,
        dtype=wp.int32,
        default=-1,
        namespace="mujoco",
        references="shape",  # Offset by shape count during merge
    )
)
builder.add_custom_attribute(
    ModelBuilder.CustomAttribute(
        name="pair_geom2",
        frequency=None,
        dtype=wp.int32,
        default=-1,
        namespace="mujoco",
        references="shape",
    )
)
# ... additional pair attributes (condim, margin, friction, etc.)
```

### MJCF Parsing

```xml
<mujoco>
  <worldbody>
    <body name="body1"><geom name="geom1" type="sphere" size="0.1"/></body>
    <body name="body2"><geom name="geom2" type="sphere" size="0.1"/></body>
  </worldbody>
  <contact>
    <pair geom1="geom1" geom2="geom2" margin="0.02" condim="4"/>
  </contact>
</mujoco>
```

```python
# In import_mjcf.py
for pair in contact.findall("pair"):
    geom1_idx = builder.shape_key.index(pair.attrib["geom1"])
    geom2_idx = builder.shape_key.index(pair.attrib["geom2"])
    
    builder.add_custom_values(**{
        "mujoco:pair_world": builder.current_world,
        "mujoco:pair_geom1": geom1_idx,
        "mujoco:pair_geom2": geom2_idx,
        "mujoco:pair_condim": int(pair.attrib.get("condim", 3)),
        "mujoco:pair_margin": float(pair.attrib.get("margin", 0.0)),
        # ... etc
    })
```

### Multi-World Merging

```python
template = ModelBuilder()
SolverMuJoCo.register_custom_attributes(template)
template.add_mjcf("robot.xml")  # Has 2 shapes (0, 1) and 1 pair (geom1=0, geom2=1)

main = ModelBuilder()
SolverMuJoCo.register_custom_attributes(main)
main.add_world(template)  # world 0: shapes 0,1; pair(0,1)
main.add_world(template)  # world 1: shapes 2,3; pair(2,3)

model = main.finalize()
print(model.mujoco.pair_world.numpy())   # [0, 1]
print(model.mujoco.pair_geom1.numpy())   # [0, 2]  <- offset by shape count
print(model.mujoco.pair_geom2.numpy())   # [1, 3]  <- offset by shape count
```

### Solver Consumption

```python
# In SolverMuJoCo._init_pairs()
pair_count = self._validate_pair_attributes(model)
pair_world = model.mujoco.pair_world.numpy()
pair_geom1 = model.mujoco.pair_geom1.numpy()

for i in range(pair_count):
    if pair_world[i] != template_world:
        continue  # Only use pairs from template world
    
    geom_name1 = shape_mapping[pair_geom1[i]]
    geom_name2 = shape_mapping[pair_geom2[i]]
    spec.add_pair(geomname1=geom_name1, geomname2=geom_name2, ...)
```

---

## Design Decisions

### Why `frequency=None` instead of a new enum value?

A new enum like `VARIABLE` would imply a specific indexing scheme. `None` clearly indicates "no predefined frequency—length determined by usage."

### Why `references` as a string instead of an enum?

The `references` field needs to support both built-in entity types AND custom attribute keys (for cross-referencing). A string provides this flexibility without a complex union type.

### Why return indices from `add_custom_values()`?

While not strictly necessary, returning indices allows callers to verify synchronization across attributes and enables advanced use cases like building cross-references.

