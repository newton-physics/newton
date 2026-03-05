"""Definitive diagnostic: check if pressure field is spatially flipped."""
import sys
sys.path.insert(0, "d:/newton")

import numpy as np
import warp as wp
wp.init()

import newton
import newton.examples
from pxr import Usd, UsdGeom
from newton._src.geometry.sdf_hydroelastic import HydroelasticSDF

# Load foot1 with the same scaling as the example
usd_path = newton.examples.get_asset("foot1.usd")
stage = Usd.Stage.Open(usd_path)
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Mesh):
        mesh_prim = prim
        break

src_mesh = newton.usd.get_mesh(mesh_prim)
vertices = np.asarray(src_mesh.vertices, dtype=np.float32)
center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
vertices = (vertices - center) * 5.0
indices = np.asarray(src_mesh.indices, dtype=np.int32)
mesh = newton.Mesh(vertices, indices, compute_inertia=False)

print("=== Foot mesh orientation ===")
print(f"Vertex count: {len(vertices)}")
# Find extremes to identify heel vs toe direction
x_min_idx = np.argmin(vertices[:, 0])
x_max_idx = np.argmax(vertices[:, 0])
z_min_idx = np.argmin(vertices[:, 2])
z_max_idx = np.argmax(vertices[:, 2])

print(f"Most negative X vertex: {vertices[x_min_idx]} (index {x_min_idx})")
print(f"Most positive X vertex: {vertices[x_max_idx]} (index {x_max_idx})")
print(f"Most negative Z vertex: {vertices[z_min_idx]} (index {z_min_idx})")
print(f"Most positive Z vertex: {vertices[z_max_idx]} (index {z_max_idx})")

# Build SDF and hydro
mesh.build_sdf(max_resolution=96, narrow_band_range=(-0.1, 0.1), margin=0.1)
device = "cuda:0"

v_min = np.min(vertices, axis=0)
v_max = np.max(vertices, axis=0)
extents = np.maximum(v_max - v_min, 1e-4)
support_hx = max(0.20, 0.75 * float(extents[0]))
support_hy = max(0.20, 0.75 * float(extents[1]))
support_hz = max(0.05, 0.20 * float(extents[2]))
overlap = max(0.003, 0.02 * float(extents[2]))
support_center_xy = 0.5 * (v_min[:2] + v_max[:2])
support_top = float(v_min[2]) + overlap
support_center = wp.vec3(float(support_center_xy[0]), float(support_center_xy[1]), support_top - support_hz)

cfg_main = newton.ModelBuilder.ShapeConfig(
    hydroelastic_type=newton.HydroelasticType.COMPLIANT,
    hydroelastic_contact_workflow=newton.HydroelasticContactWorkflow.PRESSURE,
    gap=0.02, kh=2.0e8, margin=1e-5,
)
cfg_support = newton.ModelBuilder.ShapeConfig(
    hydroelastic_type=newton.HydroelasticType.RIGID,
    hydroelastic_contact_workflow=newton.HydroelasticContactWorkflow.CLASSIC,
    sdf_max_resolution=96, sdf_narrow_band_range=(-0.1, 0.1),
    gap=0.02, kh=2.0e8, margin=1e-5,
)

builder = newton.ModelBuilder(gravity=0.0)
body_main = builder.add_body(xform=wp.transform_identity(), label="main_shape")
main_shape_idx = builder.add_shape_mesh(body=body_main, mesh=mesh, cfg=cfg_main)
body_support = builder.add_body(xform=wp.transform(support_center, wp.quat_identity()), label="support_shape")
builder.add_shape_box(body=body_support, hx=support_hx, hy=support_hy, hz=support_hz, cfg=cfg_support)

model = builder.finalize(device=device)
sdf_config = HydroelasticSDF.Config(output_contact_surface=True, buffer_fraction=1.0)
pipeline = newton.CollisionPipeline(model, broad_phase="explicit", sdf_hydroelastic_config=sdf_config)
hydro = pipeline.hydroelastic_sdf

main_sdf_idx = int(model.shape_sdf_index.numpy()[main_shape_idx])
main_pressure_idx = int(hydro.shape_pressure_index.numpy()[main_shape_idx])
pressure_table = hydro.compact_pressure_field_data.numpy()
pressure_volume_id = wp.uint64(int(pressure_table[main_pressure_idx]["pressure_ptr"]))
pressure_max = float(pressure_table[main_pressure_idx]["pressure_max"])

sdf_data = model.sdf_data.numpy()[main_sdf_idx]
sdf_center = wp.vec3(sdf_data["center"])
sdf_half_extents = wp.vec3(sdf_data["half_extents"])

print(f"\nSDF center: ({float(sdf_center[0]):.4f}, {float(sdf_center[1]):.4f}, {float(sdf_center[2]):.4f})")
print(f"SDF half: ({float(sdf_half_extents[0]):.4f}, {float(sdf_half_extents[1]):.4f}, {float(sdf_half_extents[2]):.4f})")
print(f"P_max: {pressure_max:.6f}")

# Now sample the pressure at specific diagnostic points along X axis (at Y=0, Z=0)
@wp.kernel
def sample_pressure_at_points(
    pressure_volume_id: wp.uint64,
    test_points: wp.array(dtype=wp.vec3),
    out_pressure: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    sample = test_points[tid]
    idx = wp.volume_world_to_index(pressure_volume_id, sample)
    p = wp.volume_sample_f(pressure_volume_id, idx, wp.Volume.LINEAR)
    if wp.isnan(p):
        p = 0.0
    p = wp.max(p, 0.0)
    out_pressure[tid] = p

# Sample along X axis at Y=0, Z=0
x_range = np.linspace(-0.6, 0.6, 25)
test_pts = np.array([[x, 0.0, 0.0] for x in x_range], dtype=np.float32)
test_points_wp = wp.array(test_pts, dtype=wp.vec3, device=device)
out_pressure_wp = wp.zeros(len(test_pts), dtype=wp.float32, device=device)

wp.launch(sample_pressure_at_points, dim=len(test_pts),
    inputs=[pressure_volume_id, test_points_wp],
    outputs=[out_pressure_wp], device=device)

pressures = out_pressure_wp.numpy()
print(f"\n=== Pressure along X axis (Y=0, Z=0) ===")
for i, (x, p) in enumerate(zip(x_range, pressures)):
    marker = "[+]" if p > 1e-6 else "[ ]"
    print(f"  {marker} x={x:+.3f}  pressure={p:.6f}")

# Now check: where are the wireframe line endpoints along X?
mesh.finalize(device=device)
verts_final = np.asarray(mesh.vertices, dtype=np.float32)

# Centroid of all vertices near Z=0 (within 0.05)
near_z0 = np.abs(verts_final[:, 2]) < 0.05
if np.any(near_z0):
    vz0 = verts_final[near_z0]
    print(f"\n=== Wireframe vertices near Z=0 ===")
    print(f"  Count: {np.sum(near_z0)}")
    print(f"  X range: [{vz0[:,0].min():.4f}, {vz0[:,0].max():.4f}]")
    print(f"  Y range: [{vz0[:,1].min():.4f}, {vz0[:,1].max():.4f}]")
    print(f"  Mean X: {vz0[:,0].mean():.4f}")

# Check: sample pressure at actual vertex positions near Z=0
# Pick 10 vertices spread across X
if np.any(near_z0):
    sorted_by_x = vz0[np.argsort(vz0[:, 0])]
    sample_idxs = np.linspace(0, len(sorted_by_x) - 1, 10, dtype=int)
    vert_samples = sorted_by_x[sample_idxs]

    test_vert_wp = wp.array(vert_samples, dtype=wp.vec3, device=device)
    out_vert_wp = wp.zeros(len(vert_samples), dtype=wp.float32, device=device)

    wp.launch(sample_pressure_at_points, dim=len(vert_samples),
        inputs=[pressure_volume_id, test_vert_wp],
        outputs=[out_vert_wp], device=device)

    vert_pressures = out_vert_wp.numpy()
    print(f"\n=== Pressure at actual wireframe vertices (near Z=0, sorted by X) ===")
    for v, p in zip(vert_samples, vert_pressures):
        marker = "[+]" if p > 1e-6 else "[ ]"
        print(f"  {marker} ({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f})  pressure={p:.6f}")

print("\nDone.")
