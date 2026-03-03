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
    # shape-specific params here (radius, height, etc.)
    radius: float = 0.5,
    height: float = 1.0,
    cfg: ShapeConfig | None = None,
    as_site: bool = False,
    label: str | None = None,
    custom_attributes: dict[str, Any] | None = None,
) -> int:
    """Adds a cone collision shape to a body.

    Args:
        body: Index of the parent body. Use -1 for static shapes.
        xform: Transform in parent body's local frame. If ``None``,
            identity transform is used.
        radius: Cone base radius [m].
        height: Cone height [m].
        cfg: Shape configuration. If ``None``, uses
            :attr:`default_shape_cfg`.
        as_site: If ``True``, creates a site instead of a collision shape.
        label: Optional label for identifying the shape.
        custom_attributes: Dictionary of custom attribute names to values.

    Returns:
        Index of the newly added shape.
    """
```

**Key conventions:**
- `xform` (not `tf`, `transform`, or `pose`) — always `Transform | None = None`
- `cfg` (not `config`, `shape_config`) — always `ShapeConfig | None = None`
- `body`, `label`, `custom_attributes` — standard params on all builder methods
- Defaults are `None`, not constructed objects like `wp.transform()`

## Nested Classes

Use `IntEnum` (not `Enum` with strings) for enumerations:

```python
class Model:
    class AttributeAssignment(IntEnum):
        MODEL = 0
        STATE = 1
```

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

Document shape, dtype, and units in attribute docstrings:

```python
"""Rigid body velocities [m/s, rad/s], shape (body_count,), dtype :class:`spatial_vector`."""
"""Joint forces [N or N·m], shape (joint_dof_count,), dtype float."""
"""Contact points [m], shape [count, 3], float."""
```

For compound arrays, list per-component units:
```python
"""[0] k_mu [Pa], [1] k_lambda [Pa], ..."""
```

For **public API** attributes and method signatures, use bare `wp.array | None` and document the concrete dtype in the docstring (e.g., `dtype :class:\`vec3\``). Warp kernel parameters require concrete dtypes inline (`wp.array(dtype=wp.vec3)`) per AGENTS.md.

## Quick Checklist

When reviewing new API, verify:

- [ ] Parameters use project vocabulary (`xform`, `cfg`, `body`, `label`)
- [ ] Defaults are `None`, not constructed objects
- [ ] Nested enums use `IntEnum` with int values
- [ ] Dataclass fields have docstrings on the line below
- [ ] Array docs include shape, dtype, and units
- [ ] Builder methods include `as_site`, `label`, `custom_attributes`
