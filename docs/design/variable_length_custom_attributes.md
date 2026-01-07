# Custom Frequency Attributes

## Problem Statement

Newton's custom attribute system previously only supported attributes tied to built-in entity counts (bodies, shapes, joints, etc.). This limitation meant that solver-specific data structures with independent cardinality—such as MuJoCo contact pairs—could not be represented as custom attributes.

**Specific use case:** MuJoCo's `<contact><pair>` elements define explicit contact pairs between geometries with custom solver parameters (friction, margin, condim, etc.). These pairs:
- Have their own count independent of bodies/shapes/joints
- Reference shapes by index (which must be remapped during `add_world()` merging)
- Need world assignment for multi-world simulations

The existing system couldn't handle this because:
1. `CustomAttribute.frequency` was required and had to match a `ModelAttributeFrequency` enum value
2. No mechanism existed to transform attribute values (e.g., offset shape indices) during builder merging
3. No API existed to append values to custom entity types

## Solution Overview

### 1. String Frequencies for Custom Entity Types

`CustomAttribute.frequency` now accepts either a `ModelAttributeFrequency` enum value OR a string for custom entity types. String frequencies (e.g., `"mujoco:pair"`) enable:
- Custom entity types with independent counts
- Validation that all attributes with the same string frequency have consistent counts
- Proper index offsetting during multi-world merging

```python
# Built-in frequency (enum)
CustomAttribute(
    name="my_attr",
    frequency=ModelAttributeFrequency.BODY,
    dtype=wp.float32,
)

# Custom string frequency for solver-specific entity types
CustomAttribute(
    name="pair_geom1",
    frequency="mujoco:pair",  # Custom entity type
    dtype=wp.int32,
)
```

**Key benefit:** All attributes sharing the same string frequency are validated at `finalize()` time to have the same count, preventing subtle bugs from mismatched array sizes.

### 2. Reference Transformation (`references` field)

The `references` field specifies how attribute values should be transformed during `add_world()`/`add_builder()` merging.

```python
CustomAttribute(
    name="pair_geom1",
    frequency="mujoco:pair",
    dtype=wp.int32,
    references="shape",  # Values are shape indices, offset during merge
)
```

Supported reference types:
- **Built-in entities**: `"body"`, `"shape"`, `"joint"`, `"joint_dof"`, `"joint_coord"`, `"articulation"` — values are offset by entity count
- **Special**: `"world"` — values are replaced with `current_world` (not offset)
- **Custom string frequencies**: Any string frequency (e.g., `"mujoco:pair"`) — values are offset by that frequency's count
- **Custom attributes**: Any attribute key (e.g., `"mujoco:pair_geom1"`) — values are offset by that attribute's value count

### 3. Value Appending API (`add_custom_values()`)

Method to append values to attributes with string frequencies:

```python
builder.add_custom_values(**{
    "mujoco:pair_world": builder.current_world,
    "mujoco:pair_geom1": geom1_idx,
    "mujoco:pair_geom2": geom2_idx,
    "mujoco:pair_condim": 3,
    # ... all other pair attributes
})
```

Returns a dict mapping attribute keys to the indices where values were added.

### 4. Finalization Changes

During `finalize()`:
1. **Validation**: All attributes with the same string frequency are validated to have consistent counts
2. **Array sizing**: String frequency attributes create arrays sized by `len(attr.values)`

```python
# Validation ensures all "mujoco:pair" attributes have same count
if isinstance(frequency, str):
    count = string_frequency_counts.get(frequency, 0)
```

## Changes Made

### `newton/_src/sim/builder.py`

1. **`CustomAttribute` dataclass**:
   - `frequency` is now `ModelAttributeFrequency | str` (string for custom entity types)
   - `references` field supports string frequencies for offsetting

2. **`add_custom_values()` method**: Appends values to attributes with string frequencies

3. **`add_builder()` merging logic**:
   - Handles string frequencies by offsetting indices by pre-merge frequency count
   - Applies `references` transformation (supports string frequencies)
   - Supports custom attribute and frequency cross-references

4. **`finalize()`**: 
   - Validates all attributes with same string frequency have consistent counts
   - Stores `custom_frequency_counts` on the Model

### `newton/_src/sim/model.py`

1. **`attribute_frequency`**: Now `dict[str, ModelAttributeFrequency | str]`
2. **`custom_frequency_counts`**: New dict storing counts per string frequency
3. **`get_custom_frequency_count()`**: New method to query custom frequency counts
4. **`get_attribute_frequency()`**: Returns `ModelAttributeFrequency | str`

### `newton/_src/solvers/mujoco/solver_mujoco.py`

1. **`register_custom_attributes()`**: 10 pair attributes with `frequency="mujoco:pair"`:
   - `pair_world`, `pair_geom1`, `pair_geom2` (with appropriate `references`)
   - `pair_condim`, `pair_solref`, `pair_solreffriction`, `pair_solimp`
   - `pair_margin`, `pair_gap`, `pair_friction`

2. **`_validate_pair_attributes()`**: Simplified to use `model.get_custom_frequency_count()`

3. **`_init_pairs()`**: Converts Newton pair attributes to MuJoCo spec pairs

### `newton/_src/utils/import_mjcf.py`

1. **Contact pair parsing**: Parses `<contact><pair>` elements using `add_custom_values()`

### `newton/_src/utils/selection.py`

1. **`ArticulationView`**: Raises clear error for string frequency attributes (not per-articulation)

---

## Example: MuJoCo Contact Pairs

### Attribute Registration

```python
# In SolverMuJoCo.register_custom_attributes()
# All pair attributes share "mujoco:pair" frequency - validated at finalize time
builder.add_custom_attribute(
    ModelBuilder.CustomAttribute(
        name="pair_world",
        frequency="mujoco:pair",  # Custom string frequency
        dtype=wp.int32,
        default=0,
        namespace="mujoco",
        references="world",  # Replace with current_world during merge
    )
)
builder.add_custom_attribute(
    ModelBuilder.CustomAttribute(
        name="pair_geom1",
        frequency="mujoco:pair",
        dtype=wp.int32,
        default=-1,
        namespace="mujoco",
        references="shape",  # Offset by shape count during merge
    )
)
builder.add_custom_attribute(
    ModelBuilder.CustomAttribute(
        name="pair_geom2",
        frequency="mujoco:pair",
        dtype=wp.int32,
        default=-1,
        namespace="mujoco",
        references="shape",
    )
)
# ... additional pair attributes (condim, margin, friction, etc.)
# All must use frequency="mujoco:pair" for validation
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

### Why string frequencies instead of `frequency=None`?

String frequencies provide:
1. **Validation**: All attributes with the same string frequency are validated at finalize time to have consistent counts
2. **Semantics**: The string clearly identifies the entity type (e.g., `"mujoco:pair"`)
3. **Selection support**: Future compatibility with the ArticulationView selection mechanism
4. **Namespacing**: Following the `namespace:name` pattern avoids conflicts between solvers

### Why `references` as a string instead of an enum?

The `references` field needs to support both built-in entity types, custom string frequencies, AND custom attribute keys (for cross-referencing). A string provides this flexibility without a complex union type.

### Why return indices from `add_custom_values()`?

While not strictly necessary, returning indices allows callers to verify synchronization across attributes and enables advanced use cases like building cross-references.

---

## Parsing: Automated vs Manual

### Automated Parsing (Fixed-Frequency Attributes)

For attributes with a known frequency (e.g., `BODY`, `SHAPE`, `JOINT`), parsing is largely automated via the `parse_custom_attributes()` utility and the `mjcf_attribute_name` field.

**How it works:**

1. Register attribute with `mjcf_attribute_name`:
   ```python
   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(
           name="joint_stiffness",
           frequency=ModelAttributeFrequency.JOINT,
           dtype=wp.float32,
           default=0.0,
           namespace="mujoco",
           mjcf_attribute_name="stiffness",  # Maps to XML attribute
       )
   )
   ```

2. During MJCF parsing, when a `<joint>` element is encountered, `parse_custom_attributes()` automatically:
   - Looks up "stiffness" in the XML attributes
   - Converts the string value to `wp.float32`
   - Returns a dict with `{"mujoco:joint_stiffness": parsed_value}`

3. The parsed value is passed to `builder.add_joint(..., custom_attributes=...)` and stored at the correct index.

**Key points:**
- One XML element → one entity → one attribute value
- Index is implicit (determined by entity creation order)
- String-to-dtype conversion is automatic
- `mjcf_value_transformer` can customize parsing (e.g., "true" → 1)

### Manual Parsing (Variable-Length Attributes)

Variable-length attributes (`frequency=None`) typically require **manual parsing** because:

1. **No 1:1 entity mapping**: A `<pair>` element isn't tied to a body/shape/joint—it creates its own entry
2. **Cross-references**: Values often reference other entities by name (e.g., geom names → shape indices)
3. **Nested structure**: Some elements have sub-elements (e.g., `<fixed><joint coef="..."/></fixed>`)
4. **Grouping**: Multiple attributes must be added together for one logical entity

**Example: Contact pairs require manual parsing because:**
- `geom1="name"` must be resolved to a shape index
- `pair_world` must be set to `builder.current_world`
- All pair attributes must be added atomically for one pair

```python
# Manual parsing in import_mjcf.py
for pair in contact.findall("pair"):
    # 1. Resolve references manually
    geom1_idx = builder.shape_key.index(pair.attrib["geom1"])
    geom2_idx = builder.shape_key.index(pair.attrib["geom2"])
    
    # 2. Use parse_custom_attributes for the "simple" attributes
    pair_attrs = parse_custom_attributes(
        pair.attrib, 
        builder_custom_attr_pair,  # Excludes geom1/geom2/world
        parsing_mode="mjcf"
    )
    
    # 3. Build complete values dict
    pair_values = {
        "mujoco:pair_world": builder.current_world,
        "mujoco:pair_geom1": geom1_idx,
        "mujoco:pair_geom2": geom2_idx,
    }
    pair_values.update(pair_attrs)
    
    # 4. Add atomically
    builder.add_custom_values(**pair_values)
```

### Hybrid Approach

You can combine automated and manual parsing:
- Use `parse_custom_attributes()` for simple scalar/vector attributes
- Handle references (name→index) and special fields manually
- Exclude manually-handled attributes from the automated list

---

## Implementing New Custom Entity Types with String Frequencies

### Checklist

When adding a new custom entity type (like contact pairs), you need:

#### 1. Attribute Registration (solver or builder)

```python
def register_custom_attributes(builder: ModelBuilder):
    # Define a unique string frequency for your entity type
    ENTITY_FREQ = "myns:entity"  # Recommended format: "namespace:entity_type"
    
    # World tracking (almost always needed for multi-world support)
    builder.add_custom_attribute(
        ModelBuilder.CustomAttribute(
            name="myentity_world",
            frequency=ENTITY_FREQ,  # All entity attrs share this frequency
            dtype=wp.int32,
            default=0,
            namespace="myns",
            references="world",  # Replaced with current_world during merge
        )
    )
    
    # Index references (if your entity references bodies/shapes/joints)
    builder.add_custom_attribute(
        ModelBuilder.CustomAttribute(
            name="myentity_body_idx",
            frequency=ENTITY_FREQ,
            dtype=wp.int32,
            default=-1,
            namespace="myns",
            references="body",  # Offset by body count during merge
        )
    )
    
    # Simple data attributes
    builder.add_custom_attribute(
        ModelBuilder.CustomAttribute(
            name="myentity_stiffness",
            frequency=ENTITY_FREQ,
            dtype=wp.float32,
            default=0.0,
            namespace="myns",
            mjcf_attribute_name="stiffness",  # For parse_custom_attributes()
        )
    )
```

**Key decisions:**
- `frequency`: Use a unique string like `"namespace:entity_type"` for all related attributes
- `references`: What entity type do index values refer to? (for multi-world offset)
- `mjcf_attribute_name`: What's the XML attribute name? (for automated parsing)
- `mjcf_value_transformer`: Need special string parsing? (e.g., "true"→1, "auto"→2)

**Important:** All attributes with the same string frequency MUST have the same count at finalize time.

#### 2. MJCF Parsing (import_mjcf.py)

```python
# Filter attributes for automated parsing (exclude manually-handled ones)
builder_custom_attr_myentity = [
    attr for attr in builder.custom_attributes.values()
    if isinstance(attr.frequency, str)  # String frequencies
    and attr.name.startswith("myentity_")
    and attr.name not in ("myentity_world", "myentity_body_idx")  # Manual
]

# Check if attributes are registered (solver may not be used)
has_myentity_attrs = "myns:myentity_world" in builder.custom_attributes
myentity_section = root.find("myentity")

if myentity_section is not None and has_myentity_attrs:
    for elem in myentity_section.findall("item"):
        # Resolve name references to indices
        body_name = elem.attrib.get("body")
        body_idx = builder.body_key.index(body_name)
        
        # Use automated parsing for simple attributes
        attrs = parse_custom_attributes(
            elem.attrib, 
            builder_custom_attr_myentity,
            parsing_mode="mjcf"
        )
        
        # Build complete values
        values = {
            "myns:myentity_world": builder.current_world,
            "myns:myentity_body_idx": body_idx,
        }
        for attr in builder_custom_attr_myentity:
            values[attr.key] = attrs.get(attr.key, attr.default)
        
        builder.add_custom_values(**values)
```

**Key patterns:**
- Filter to get only your attributes
- Check `has_*_attrs` before parsing (attributes may not be registered)
- Resolve names → indices manually
- Set `*_world` to `builder.current_world`
- Use `parse_custom_attributes()` for simple fields
- Call `add_custom_values()` with all attributes together

#### 3. Validation (solver)

Validation is now handled automatically at `finalize()` time via the string frequency mechanism.
Your solver can simply query the count:

```python
@staticmethod
def _get_myentity_count(model: Model) -> int:
    """Get the number of myentity elements."""
    # Validation was done at finalize time via string frequency mechanism
    return model.get_custom_frequency_count("myns:entity")
```

#### 4. Solver Consumption

```python
def _init_myentities(self, model: Model, spec, body_mapping, template_world):
    count = model.get_custom_frequency_count("myns:entity")
    if count == 0:
        return
    
    attrs = model.myns
    world_arr = attrs.myentity_world.numpy()
    body_idx_arr = attrs.myentity_body_idx.numpy()
    stiffness_arr = attrs.myentity_stiffness.numpy()
    
    for i in range(count):
        # Filter to template world only (MuJoCo replicates across worlds)
        if int(world_arr[i]) != template_world:
            continue
        
        # Map Newton indices to MuJoCo names/indices
        body_name = body_mapping[int(body_idx_arr[i])]
        
        # Create in MuJoCo spec
        spec.add_myentity(body=body_name, stiffness=float(stiffness_arr[i]))
```

**Key patterns:**
- Use `model.get_custom_frequency_count()` to get count (validation already done)
- Filter by `template_world` (MuJoCo replicates the template)
- Map Newton indices → MuJoCo names using the mapping dicts
- Handle missing/invalid indices gracefully

---

## Summary: Enum vs String Frequencies

| Aspect | Enum Frequency (`BODY`, `SHAPE`, etc.) | String Frequency (`"mujoco:pair"`, etc.) |
|--------|----------------------------------------|------------------------------------------|
| **Array size** | Entity count (e.g., `body_count`) | Number of `add_custom_values()` calls |
| **Index** | Implicit (entity creation order) | Explicit (append order) |
| **Parsing** | Automatic via `mjcf_attribute_name` | Manual + `parse_custom_attributes()` |
| **Multi-world** | Automatic (entity offsets) | Manual `references` field |
| **Validation** | N/A | All same-frequency attrs must have same count |
| **Use case** | Per-entity properties | Independent/custom entity types |

