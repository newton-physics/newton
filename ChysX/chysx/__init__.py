"""ChysX: minimal CUDA cloth physics simulator.

The simulator does not own particle data: each step takes raw CUDA
device pointers to externally-allocated position / velocity buffers
(one ``float3`` per particle).  This mirrors libuipc's ``BufferView``
design and lets callers (e.g. Newton + Warp) share GPU memory without
copies.

Public API
----------

* :class:`ClothMaterial` — plain-old-data material parameters
  (Lamé mu/lambda, bending, density, damping, gravity).

* :class:`ClothSimulator` — combines a ClothMaterial with externally-
  owned device pointers and runs the cloth step in ChysX.  Material
  parameters are *copied* into the simulator; pointers are
  *referenced* (not copied).

Typical use::

    import chysx

    mat = chysx.ClothMaterial()
    mat.gx, mat.gy, mat.gz = 0.0, 0.0, -9.81
    mat.damping = 0.1

    sim = chysx.ClothSimulator()
    sim.set_material(mat)
    sim.set_external_buffers(pos_ptr=pos.ptr,
                             vel_ptr=vel.ptr,
                             particle_count=n)
    sim.step(dt=1.0 / 60.0)
"""

from __future__ import annotations

from . import _chysx_native

ClothMaterial = _chysx_native.ClothMaterial
ClothSimulator = _chysx_native.ClothSimulator

__version__ = "0.1.0"


__all__ = [
    "ClothMaterial",
    "ClothSimulator",
    "__version__",
]
