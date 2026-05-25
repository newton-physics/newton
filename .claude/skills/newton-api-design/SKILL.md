---
name: newton-api-design
description: Use when designing, adding, or reviewing public API for the Newton physics engine — class names, method signatures, type hints, docstrings, or parameter conventions. Also use when unsure if new API conforms to project conventions.
---

# Newton API Design Conventions

Detailed patterns that supplement AGENTS.md. Read AGENTS.md first for the basics (prefix-first naming, PEP 604, Google-style docstrings, SI units, Sphinx cross-refs).

## Builder Method Signature Template

All `ModelBuilder.add_shape_*` methods follow this parameter order:

```python
def add_shape_cone(
    self,
    body: int,
    xform: Transform | None = None,
    # shape-specific params here (radius, half_height, etc.)
    radius: float = 1.0,
    half_height: float = 0.5,
    cfg: ShapeConfig | None = None,
    as_site: bool = False,
    color: Vec3 | None = None,
    label: str | None = None,
    custom_attributes: dict[str, Any] | None = None,
) -> int:
    """Adds a cone collision shape to a body.

    Args:
        body: Index of the parent body. Use -1 for static shapes.
        xform: Transform in parent body's local frame. If ``None``,
            identity transform is used.
        radius: Cone base radius [m].
        half_height: Half the cone height [m].
        cfg: Shape configuration. If ``None``, uses
            :attr:`default_shape_cfg`.
        as_site: If ``True``, creates a site instead of a collision shape.
        color: Optional display RGB color in [0, 1]. If ``None``, uses
            the per-shape palette color.
        label: Optional label for identifying the shape.
        custom_attributes: Dictionary of custom attribute names to values.

    Returns:
        Index of the newly added shape.
    """
```

**Key conventions:**
- `xform` (not `tf`, `transform`, or `pose`) — always `Transform | None = None`
- `cfg` (not `config`, `shape_config`) — always `ShapeConfig | None = None`
- `body`, `color`, `label`, `custom_attributes` — standard params on all builder methods
- Defaults are `None`, not constructed objects like `wp.transform()`

## Nested Classes

Use `IntEnum` (not `Enum` with strings) for enumerations:

```python
class Model:
    class AttributeAssignment(IntEnum):
        MODEL = 0
        STATE = 1
```

When an `IntEnum` includes a `NONE` member, define it first at `0`:

```python
class GeoType(IntEnum):
    NONE = 0
    PLANE = 1
    HFIELD = 2
```

This keeps the sentinel value stable and leaves room to append future real
members at the end instead of inserting them before a trailing `NONE`.

Dataclass field docstrings go on the line immediately below the field:

```python
@dataclass
class ShapeConfig:
    density: float = 1000.0
    """The density of the shape material."""
    ke: float = 2.5e3
    """The contact elastic stiffness."""
```

## Array Documentation Format

Annotate Warp arrays with the dtype, e.g. `wp.array[wp.vec3]`, `wp.array2d[float]`, `wp.array[wp.spatial_vector] | None`.
Document units and shape in the docstring.

```python
"""Rigid body velocities [m/s, rad/s], shape [body_count]."""
"""Joint forces [N or N·m], shape [joint_dof_count]."""
"""Contact points [m], shape [count, 3]."""
```

For compound arrays, list per-component units:
```python
"""[0] k_mu [Pa], [1] k_lambda [Pa], ..."""
```

Use `wp.array[X]` for 1-D, `wp.array2d[X]` for 2-D, and `wp.array[Any]` for polymorphic dtypes.

## Quick Checklist

When reviewing new API, verify:

- [ ] Parameters use project vocabulary (`xform`, `cfg`, `body`, `label`)
- [ ] Defaults are `None`, not constructed objects
- [ ] Nested enumerations use `IntEnum` with int values
- [ ] Enumerations with `NONE` define `NONE = 0` first
- [ ] Dataclass fields have docstrings on the line below
- [ ] Warp array annotations include the dtype (e.g. `wp.array[wp.vec3]`); docstrings give units and shape
- [ ] Builder methods include `as_site`, `color`, `label`, `custom_attributes`
