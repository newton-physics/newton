import numpy as np
import warp as wp
import newton
from pathlib import Path
import json

# Setup Newton
wp.init()

from projects.digital_instron_v2.foundation import FoundationMaterial
from projects.digital_instron_v2.geometry import _load_obj_mesh, condition_midsole_mesh
from projects.digital_instron_v2.manifest import load_manifest
from projects.digital_instron_v2.workflow import _load_spring_grid

manifest = load_manifest("DigitalInstron/manifest_v2.json")
output_dir = manifest.cache_dir
spring_grid, midsole_vertices = _load_spring_grid(manifest, output_dir)

# Load material
with open("DigitalInstron/processed/v2_cache/digital_instron_v2_foundation_material.json") as f:
    mat_data = json.load(f)["material"]
material = FoundationMaterial(**mat_data)

# Load meshes
foot_v, foot_f = _load_obj_mesh(Path("FeetFinder/0002-B.obj"))
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

device = "cuda:0"

# Build meshes and SDFs
foot_mesh = newton.Mesh(foot_v_transformed, foot_f_transformed.reshape(-1))
foot_mesh.build_sdf(max_resolution=64, narrow_band_range=(-0.010, 0.010), margin=0.005, device=device)

midsole_mesh = newton.Mesh(midsole_v_transformed, midsole_f_transformed.reshape(-1))
midsole_mesh.build_sdf(max_resolution=64, narrow_band_range=(-0.010, 0.010), margin=0.005, device=device)

# Builder
builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
builder.add_ground_plane()

min_bottom_m = np.min(spring_grid.bottom_m)
start_z = -min_bottom_m + 0.005

foot_body_id = builder.add_body(
    xform=wp.transform(p=wp.vec3(0.0, 0.0, start_z - 0.025), q=wp.quat_identity()), # Penetrating by 25mm
    mass=80.0,
    inertia=wp.mat33(10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0),
    lock_inertia=True,
)

foot_cfg = newton.ModelBuilder.ShapeConfig(is_hydroelastic=True, gap=0.005)
foot_shape_id = builder.add_shape_mesh(foot_body_id, mesh=foot_mesh, cfg=foot_cfg)

midsole_cfg = newton.ModelBuilder.ShapeConfig(is_hydroelastic=True, gap=0.005)
midsole_shape_id = builder.add_shape_mesh(body=-1, mesh=midsole_mesh, cfg=midsole_cfg)

model = builder.finalize(device=device)
state = model.state()
newton.eval_fk(model, model.joint_q, model.joint_qd, state)

from newton.geometry import HydroelasticSDF
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

collision_pipeline.collide(state, contacts)

hydro = collision_pipeline.narrow_phase.hydroelastic_sdf
voxel_count = hydro.iso_voxel_count.numpy()[0]
print(f"Number of iso-voxels: {voxel_count}")

if voxel_count > 0:
    coords = hydro.iso_voxel_coords.numpy()[:voxel_count]
    shape_pairs = hydro.iso_voxel_shape_pair.numpy()[:voxel_count]
    
    z_coords = coords[:, 2]
    unique_z, counts = np.unique(z_coords, return_counts=True)
    print("\nZ-coordinate distribution of iso-voxels:")
    for z_val, cnt in zip(unique_z, counts):
        # Calculate approximate world Z for this index
        world_z = -0.01506 + 0.0399 + z_val * 0.0037
        print(f"  Z index {z_val} (~{world_z*1000:.1f} mm): {cnt} voxels")
        
    # Find the spring index with the maximum Z offset (where the foot first touches the midsole)
    z_offsets = spring_grid.top_m[valid] - z_foot_sole[valid]
    first_touch_idx = np.where(valid)[0][np.argmax(z_offsets)]
    first_touch_uv = spring_grid.grid_uv_m[first_touch_idx]
    print(f"\nFoot first touches midsole at spring index {first_touch_idx}: {first_touch_uv} mm")
    print(f"  Touch Top Z: {spring_grid.top_m[first_touch_idx]*1000:.2f} mm")
    
    # Let's find iso-voxels near this touch location
    # Convert first_touch_uv to foot voxel coordinates
    target_x_idx = int(np.round((first_touch_uv[0] - (-0.0797)) / 0.0037))
    target_y_idx = int(np.round((first_touch_uv[1] - 0.0151) / 0.0037))
    print(f"Target voxel indices: X={target_x_idx}, Y={target_y_idx}")
    
    touch_indices = np.where(
        (np.abs(coords[:, 0] - target_x_idx) <= 2) &
        (np.abs(coords[:, 1] - target_y_idx) <= 2)
    )[0]
    print(f"Found {len(touch_indices)} voxels near first touch location.")
    if len(touch_indices) > 0:
        # Choose the one with Z index = 4
        z4_touch_indices = touch_indices[coords[touch_indices, 2] == 4]
        if len(z4_touch_indices) > 0:
            target_idx = z4_touch_indices[0]
        else:
            target_idx = touch_indices[np.argmin(np.abs(coords[touch_indices, 2] - 4))]
        target_coord = coords[target_idx]
    else:
        target_idx = 0
        target_coord = coords[0]
    
    @wp.func
    def get_rel_stiffness(k_a: float, k_b: float) -> tuple[float, float]:
        k_m_inv = 1.0 / wp.sqrt(k_a * k_b)
        return k_a * k_m_inv, k_b * k_m_inv

    from newton._src.geometry.sdf_texture import TextureSDFData, texture_sample_sdf, texture_sample_sdf_at_voxel

    @wp.kernel
    def inspect_single_voxel_kernel(
        coords: wp.vec3us,
        pair: wp.vec2i,
        corner_offsets_table: wp.array[wp.vec3ub],
        sdf_data: wp.array[TextureSDFData],
        shape_transform: wp.array[wp.transform],
        shape_material_kh: wp.array[float],
        # outputs
        out_valA: wp.array[float],
        out_valB: wp.array[float],
        out_vdiff: wp.array[float],
        out_local_pos_b: wp.array[wp.vec3],
        out_point_a: wp.array[wp.vec3],
    ):
        shape_a = pair[0]
        shape_b = pair[1]

        sdf_data_a = sdf_data[shape_a]
        sdf_data_b = sdf_data[shape_b]

        transform_a = shape_transform[shape_a]
        transform_b = shape_transform[shape_b]

        k_a = shape_material_kh[shape_a]
        k_b = shape_material_kh[shape_b]
        k_eff_a, k_eff_b = get_rel_stiffness(k_a, k_b)

        x_id = wp.int32(coords.x)
        y_id = wp.int32(coords.y)
        z_id = wp.int32(coords.z)

        X_b_to_a = wp.transform_multiply(wp.transform_inverse(transform_a), transform_b)

        for i in range(8):
            corner_offset = wp.vec3i(corner_offsets_table[i])
            x = x_id + corner_offset.x
            y = y_id + corner_offset.y
            z = z_id + corner_offset.z

            local_pos_b = sdf_data_b.sdf_box_lower + wp.cw_mul(wp.vec3(float(x), float(y), float(z)), sdf_data_b.voxel_size)
            point_a = wp.transform_point(X_b_to_a, local_pos_b)

            valB = texture_sample_sdf_at_voxel(sdf_data_b, x, y, z)
            valA = texture_sample_sdf(sdf_data_a, point_a)

            if valB < 0.0 and valA < 0.0:
                v_diff = k_eff_b * valB - k_eff_a * valA
            else:
                v_diff = valB - valA

            out_valB[i] = valB
            out_valA[i] = valA
            out_vdiff[i] = v_diff
            out_local_pos_b[i] = local_pos_b
            out_point_a[i] = point_a

    out_valA_gpu = wp.zeros(8, dtype=float, device=device)
    out_valB_gpu = wp.zeros(8, dtype=float, device=device)
    out_vdiff_gpu = wp.zeros(8, dtype=float, device=device)
    out_local_pos_b_gpu = wp.zeros(8, dtype=wp.vec3, device=device)
    out_point_a_gpu = wp.zeros(8, dtype=wp.vec3, device=device)

    wp.launch(
        inspect_single_voxel_kernel,
        dim=1,
        inputs=[
            wp.vec3us(int(target_coord[0]), int(target_coord[1]), int(target_coord[2])),
            wp.vec2i(int(shape_pairs[target_idx][0]), int(shape_pairs[target_idx][1])),
            hydro.mc_tables[3],
            hydro._shape_sdf_data,
            collision_pipeline.geom_transform,
            model.shape_material_kh,
        ],
        outputs=[
            out_valA_gpu,
            out_valB_gpu,
            out_vdiff_gpu,
            out_local_pos_b_gpu,
            out_point_a_gpu,
        ],
        device=device
    )
    wp.synchronize()

    valA_np = out_valA_gpu.numpy()
    valB_np = out_valB_gpu.numpy()
    vdiff_np = out_vdiff_gpu.numpy()
    local_pos_b_np = out_local_pos_b_gpu.numpy()
    point_a_np = out_point_a_gpu.numpy()

    print(f"\n--- Target Touch Voxel (Coords: {target_coord}, Pair: {shape_pairs[target_idx]}) ---")
    for i in range(8):
        lp = local_pos_b_np[i]
        pa = point_a_np[i]
        print(f"  Corner {i}:")
        print(f"    local_pos_b (foot local) = [{lp[0]:.4f}, {lp[1]:.4f}, {lp[2]:.4f}]")
        print(f"    point_a (midsole local)  = [{pa[0]:.4f}, {pa[1]:.4f}, {pa[2]:.4f}]")
        print(f"    valA (midsole)={valA_np[i]:.6f}, valB (foot)={valB_np[i]:.6f}, vdiff={vdiff_np[i]:.6f}")





    # Query nearest spring grid top/bottom
    target_x = -0.0797 + target_coord[0] * 0.0037
    target_y = 0.0151 + target_coord[1] * 0.0037
    print(f"\nTarget world XY: [{target_x:.4f}, {target_y:.4f}]")
    
    dists = np.sum((spring_grid.grid_uv_m - np.array([target_x, target_y]))**2, axis=1)
    nearest_idx = np.argmin(dists)
    nearest_uv = spring_grid.grid_uv_m[nearest_idx]
    print(f"Nearest spring grid point {nearest_idx} at: {nearest_uv}")
    print(f"  Slack length (thickness): {spring_grid.slack_length_m[nearest_idx]*1000:.2f} mm")
    print(f"  Top Z: {spring_grid.top_m[nearest_idx]*1000:.2f} mm")
    print(f"  Bottom Z: {spring_grid.bottom_m[nearest_idx]*1000:.2f} mm")
