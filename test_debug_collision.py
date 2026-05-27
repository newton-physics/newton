import sys
import numpy as np
import warp as wp
import newton
from newton.geometry import HydroelasticSDF
from pathlib import Path
import json

# Setup Newton initialization
wp.init()

# We need the args and parsing
from projects.digital_instron_v2.foundation import FoundationMaterial
from projects.digital_instron_v2.geometry import _load_obj_mesh, compute_grid_neighbors, condition_midsole_mesh
from projects.digital_instron_v2.manifest import load_manifest
from projects.digital_instron_v2.workflow import _load_spring_grid

manifest = load_manifest("DigitalInstron/manifest_v2.json")
output_dir = manifest.cache_dir
spring_grid, midsole_vertices = _load_spring_grid(manifest, output_dir)

# Load material
with open("DigitalInstron/processed/v2_cache/digital_instron_v2_foundation_material.json") as f:
    mat_data = json.load(f)["material"]
material = FoundationMaterial(**mat_data)

# Load foot mesh
foot_v, foot_f = _load_obj_mesh(Path("FeetFinder/0002-B.obj"))

# Align and transform foot mesh as in example_hydro_shoe.py
sign = 1.0 # mirror_foot = True
foot_v_transformed = np.zeros_like(foot_v)
foot_v_transformed[:, 0] = sign * foot_v[:, 1]
foot_v_transformed[:, 1] = foot_v[:, 0]
foot_v_transformed[:, 2] = foot_v[:, 2]

foot_f_transformed = foot_f.copy()
foot_f_transformed[:, [1, 2]] = foot_f_transformed[:, [2, 1]]

# Center horizontally
foot_center = 0.5 * (np.min(foot_v_transformed, axis=0) + np.max(foot_v_transformed, axis=0))
midsole_center = 0.5 * (np.min(spring_grid.grid_uv_m, axis=0) + np.max(spring_grid.grid_uv_m, axis=0))
foot_v_transformed[:, 0] += midsole_center[0] - foot_center[0]
foot_v_transformed[:, 1] += midsole_center[1] - foot_center[1]

# Foot yaw (0.0)
# Vertical alignment
spacing = spring_grid.spacing_m
z_foot_sole = np.full(len(spring_grid.grid_uv_m), np.nan)
for i, (x_g, y_g) in enumerate(spring_grid.grid_uv_m):
    in_cell = (np.abs(foot_v_transformed[:, 0] - x_g) <= spacing * 0.5) & (
        np.abs(foot_v_transformed[:, 1] - y_g) <= spacing * 0.5
    )
    if np.any(in_cell):
        z_foot_sole[i] = np.min(foot_v_transformed[in_cell, 2])

valid = np.isfinite(z_foot_sole)
if np.any(valid):
    z_offsets = spring_grid.top_m[valid] - z_foot_sole[valid]
    Z_offset = np.max(z_offsets)
    foot_v_transformed[:, 2] += Z_offset

print(f"Foot bounds after transform: min={np.min(foot_v_transformed, axis=0)}, max={np.max(foot_v_transformed, axis=0)}")

# Load midsole
report = condition_midsole_mesh(
    manifest.midsole_mesh,
    output_dir,
    source_units="mm",
    min_thickness_m=0.005,
    max_thickness_m=0.08,
)
midsole_v, midsole_f = _load_obj_mesh(Path(str(report["repaired_mesh"])))

midsole_v_transformed = np.zeros_like(midsole_v)
midsole_v_transformed[:, 0] = midsole_v[:, 0]
midsole_v_transformed[:, 1] = midsole_v[:, 2]
midsole_v_transformed[:, 2] = midsole_v[:, 1]

midsole_f_transformed = midsole_f.copy()
midsole_f_transformed[:, [1, 2]] = midsole_f_transformed[:, [2, 1]]

print(f"Midsole bounds after transform: min={np.min(midsole_v_transformed, axis=0)}, max={np.max(midsole_v_transformed, axis=0)}")

device = "cuda:0"

# Build SDFs
foot_mesh = newton.Mesh(foot_v_transformed, foot_f_transformed.reshape(-1))
foot_mesh.build_sdf(max_resolution=64, narrow_band_range=(-0.010, 0.010), margin=0.005, device=device)

midsole_mesh = newton.Mesh(midsole_v_transformed, midsole_f_transformed.reshape(-1))
midsole_mesh.build_sdf(max_resolution=64, narrow_band_range=(-0.010, 0.010), margin=0.005, device=device)

# Builder
builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
builder.add_ground_plane()

mass = 80.0
min_bottom_m = np.min(spring_grid.bottom_m)
start_z = -min_bottom_m + 0.005
print(f"min_bottom_m={min_bottom_m}, start_z={start_z}")

foot_body_id = builder.add_body(
    xform=wp.transform(p=wp.vec3(0.0, 0.0, start_z), q=wp.quat_identity()),
    mass=mass,
    inertia=wp.mat33(10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0),
    lock_inertia=True,
)

foot_cfg = newton.ModelBuilder.ShapeConfig(is_hydroelastic=True, gap=0.005)
foot_shape_id = builder.add_shape_mesh(
    foot_body_id,
    mesh=foot_mesh,
    cfg=foot_cfg,
)

midsole_cfg = newton.ModelBuilder.ShapeConfig(is_hydroelastic=True, gap=0.005)
midsole_shape_id = builder.add_shape_mesh(
    body=-1,
    mesh=midsole_mesh,
    cfg=midsole_cfg,
)

model = builder.finalize(device=device)
state = model.state()
newton.eval_fk(model, model.joint_q, model.joint_qd, state)

sdf_config = HydroelasticSDF.Config(
    reduce_contacts=True,
    output_contact_surface=True,
    anchor_contact=True,
    buffer_fraction=1.0,
)

collision_pipeline = newton.CollisionPipeline(
    model,
    rigid_contact_max=2000,
    sdf_hydroelastic_config=sdf_config,
)
contacts = collision_pipeline.contacts()

# Test collision at start_z
collision_pipeline.collide(state, contacts)
print(f"shape_gap: {model.shape_gap.numpy()}")
print(f"shape_aabb_lower: {collision_pipeline.narrow_phase.shape_aabb_lower.numpy()}")
print(f"shape_aabb_upper: {collision_pipeline.narrow_phase.shape_aabb_upper.numpy()}")
print(f"shape_pairs_sdf_sdf_count: {collision_pipeline.narrow_phase.shape_pairs_sdf_sdf_count.numpy()[0]}")
for i, count in enumerate(collision_pipeline.narrow_phase.hydroelastic_sdf.iso_buffer_counts):
    print(f"  iso_buffer_counts[{i}]: {count.numpy()[0]}")
if collision_pipeline.narrow_phase.shape_pairs_sdf_sdf_count.numpy()[0] > 0:
    print(f"shape_pairs_sdf_sdf: {collision_pipeline.narrow_phase.shape_pairs_sdf_sdf.numpy()[:collision_pipeline.narrow_phase.shape_pairs_sdf_sdf_count.numpy()[0]]}")
print(f"Collision at start_z={start_z:.6f}:")
print(f"  Rigid contact count: {contacts.rigid_contact_count.numpy()[0]}")
if contacts.rigid_contact_count.numpy()[0] > 0:
    s0 = contacts.rigid_contact_shape0.numpy()[:contacts.rigid_contact_count.numpy()[0]]
    s1 = contacts.rigid_contact_shape1.numpy()[:contacts.rigid_contact_count.numpy()[0]]
    print(f"  Shape pairs in contact: {list(zip(s0, s1))}")
    # Print first few contact points
    p0 = contacts.rigid_contact_point0.numpy()[:contacts.rigid_contact_count.numpy()[0]]
    p1 = contacts.rigid_contact_point1.numpy()[:contacts.rigid_contact_count.numpy()[0]]
    print(f"  Contact point0: {p0}")
    print(f"  Contact point1: {p1}")

hydro = collision_pipeline.narrow_phase.hydroelastic_sdf
contact_surface = hydro.get_contact_surface()
if contact_surface is not None:
    face_count = contact_surface.face_contact_count.numpy()[0]
    print(f"  Hydroelastic face count: {face_count}")

# Test collision with penetration
print("\nMoving foot down by 25mm (Z penetration)...")
body_q = state.body_q.numpy()
body_q[foot_body_id, :3] = [0.0, 0.0, start_z - 0.025]
state.body_q.assign(body_q)
newton.eval_fk(model, model.joint_q, model.joint_qd, state)

collision_pipeline.collide(state, contacts)
print(f"Collision at start_z-0.025={start_z-0.025:.6f}:")
print(f"  Rigid contact count: {contacts.rigid_contact_count.numpy()[0]}")
if contacts.rigid_contact_count.numpy()[0] > 0:
    s0 = contacts.rigid_contact_shape0.numpy()[:contacts.rigid_contact_count.numpy()[0]]
    s1 = contacts.rigid_contact_shape1.numpy()[:contacts.rigid_contact_count.numpy()[0]]
    print(f"  Shape pairs in contact: {list(zip(s0, s1))}")
    p0 = contacts.rigid_contact_point0.numpy()[:contacts.rigid_contact_count.numpy()[0]]
    print(f"  Contact point0 (first 5): {p0[:5]}")

contact_surface = hydro.get_contact_surface()
if contact_surface is not None:
    face_count = contact_surface.face_contact_count.numpy()[0]
    print(f"  Hydroelastic face count: {face_count}")
