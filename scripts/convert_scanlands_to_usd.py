#!/usr/bin/env python3
"""
Pathways Landscape Ingestion: Scanlands.blend -> OpenUSD (Scanlands.usdc)
Extracts base geometry without flattening, preserving 187,490 foliage elements
as native UsdGeomPointInstancer primitives with 8 deduplicated BLAS prototypes.
"""

import os
import sys
import time
import argparse

# If executed directly outside Blender, invoke headless Blender subprocess
try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

if not IN_BLENDER:
    import subprocess
    def run_via_blender():
        blend_default = "scenes/Scanlands.blend"
        if not os.path.exists(blend_default) and os.path.exists(os.path.join("..", blend_default)):
            blend_default = os.path.join("..", blend_default)
        cmd = ["blender", "-b", blend_default, "-P", os.path.abspath(__file__), "--"] + sys.argv[1:]
        print(f"[Launcher] Executing in Blender: {' '.join(cmd)}")
        res = subprocess.run(cmd)
        sys.exit(res.returncode)

    if __name__ == "__main__":
        run_via_blender()

# -----------------------------------------------------------------------------
# Inside Blender Runtime Execution
# -----------------------------------------------------------------------------
# Ensure system pxr package is available within Blender's python runtime
sys.path.insert(0, "/home/naoki/.local/lib/python3.14/site-packages")
from pxr import Usd, UsdGeom, Gf, Vt, Sdf

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Scanlands.blend to OpenUSD")
    parser.add_argument("--blend", default="scenes/Scanlands.blend", help="Path to input .blend file")
    parser.add_argument("--out-dir", default="scenes/Scanlands", help="Directory for output assets")
    parser.add_argument("--out-file", default="Scanlands.usdc", help="Output USDC filename")

    # Arguments passed after '--' in blender CLI
    args_to_parse = []
    if "--" in sys.argv:
        args_to_parse = sys.argv[sys.argv.index("--") + 1:]
    return parser.parse_args(args_to_parse)

def convert_scanlands():
    args = parse_args()
    blend_path = args.blend
    out_dir = args.out_dir
    out_usdc = os.path.join(out_dir, args.out_file)

    if not os.path.exists(blend_path) and os.path.exists(os.path.join("..", blend_path)):
        blend_path = os.path.join("..", blend_path)

    # Open .blend file if not already loaded
    current_filepath = bpy.data.filepath
    if not current_filepath or not os.path.samefile(current_filepath, blend_path) if (current_filepath and os.path.exists(current_filepath) and os.path.exists(blend_path)) else True:
        if os.path.exists(blend_path):
            print(f"Loading '{blend_path}' into Blender session...")
            bpy.ops.wm.open_mainfile(filepath=blend_path)

    os.makedirs(out_dir, exist_ok=True)

    print("==========================================================")
    print("  Pathways: Scanlands OpenUSD Point Instancing Pipeline   ")
    print("==========================================================")
    print(f"Input Scene:  {blend_path}")
    print(f"Output USDC:  {out_usdc}")

    proto_names = [
        'tree_Lp.007', 'tree_Lp.008', 'tree_Lp.009',
        'tree_Lp.001', 'tree_Lp.002', 'tree_Lp.003',
        'branch_1.703', 'branch_1'
    ]
    proto_map = {name: idx for idx, name in enumerate(proto_names)}

    # Step 1: Harvest particle scatter instances from evaluated dependency graph
    print(f"[1/4] Extracting particle scatter instances from evaluated depsgraph...")
    t0 = time.time()
    depsgraph = bpy.context.evaluated_depsgraph_get()

    positions = []
    orientations = []
    scales = []
    proto_indices = []

    proto_counts = {name: 0 for name in proto_names}

    for inst in depsgraph.object_instances:
        if not inst.is_instance:
            continue
        obj_name = inst.instance_object.name
        if obj_name not in proto_map:
            continue

        p_idx = proto_map[obj_name]
        mat = inst.matrix_world
        loc, rot, sca = mat.decompose()

        positions.append(Gf.Vec3f(loc.x, loc.y, loc.z))
        orientations.append(Gf.Quath(rot.w, rot.x, rot.y, rot.z))
        scales.append(Gf.Vec3f(sca.x, sca.y, sca.z))
        proto_indices.append(p_idx)
        proto_counts[obj_name] += 1

    harvest_time = time.time() - t0
    total_instances = len(positions)
    print(f"  Harvested {total_instances:,} foliage instances in {harvest_time:.2f}s:")
    for name in proto_names:
        print(f"    - {name:14s} (BLAS {proto_map[name]}): {proto_counts[name]:,} instances")

    if total_instances == 0:
        raise RuntimeError(f"Error: 0 instances harvested from {blend_path}. Check prototype names.")

    # Step 2: Mute emitter particle systems to prevent raw geometry / individual Xform explosion
    print(f"[2/4] Muting emitter particle systems for non-instanced base mesh export...")
    muted_systems = 0
    for o in bpy.data.objects:
        if o.particle_systems:
            for ps in o.particle_systems:
                ps.settings.count = 0
                muted_systems += 1
    print(f"  Muted {muted_systems} particle systems across emitter meshes.")

    # Step 2b: Remove atmospheric cloud/separator meshes to prevent ray occlusion
    print(f"[2b/4] Removing atmospheric occlusion meshes ('separator' / 'clouds+alpha')...")
    removed_objects = 0
    for obj in list(bpy.data.objects):
        in_clouds = any("clouds" in c.name.lower() or "alpha" in c.name.lower() for c in obj.users_collection)
        if obj.name == "separator" or in_clouds or "separator" in obj.name.lower():
            print(f"  Removing atmospheric occlusion object: {obj.name}")
            bpy.data.objects.remove(obj, do_unlink=True)
            removed_objects += 1
    print(f"  Successfully removed {removed_objects} atmospheric occlusion objects.")

    # Step 3: Export base meshes and prototypes to binary USDC via Blender C++ exporter
    print(f"[3/4] Exporting base geometry & prototype roots to USDC...")
    t_exp = time.time()
    bpy.ops.wm.usd_export(
        filepath=out_usdc,
        export_hair=False,
        use_instancing=False,
        evaluation_mode='RENDER',
        export_materials=True,
        export_textures_mode='NEW',
        overwrite_textures=True,
        relative_paths=True
    )
    exp_time = time.time() - t_exp
    file_size_mb = os.path.getsize(out_usdc) / (1024 * 1024)
    print(f"  Base USDC export completed in {exp_time:.2f}s ({file_size_mb:.2f} MB)")

    # Step 4: Author UsdGeomPointInstancer and reset prototype roots in USD stage
    print(f"[4/4] Authoring UsdGeomPointInstancer in USD stage...")
    t_inj = time.time()
    stage = Usd.Stage.Open(out_usdc)
    if not stage:
        raise RuntimeError(f"Failed to open exported USDC stage: {out_usdc}")

    # Ensure separator prim is removed from stage if it somehow persisted
    sep_prim = stage.GetPrimAtPath("/root/separator")
    if sep_prim.IsValid():
        stage.RemovePrim(Sdf.Path("/root/separator"))
        print("  Pruned /root/separator prim from USD stage.")

    # Reset prototype root transforms so prototype meshes center at local origin
    # and instance world matrices remain unperturbed by prototype workspace offsets
    for name in proto_names:
        sanitized = name.replace('.', '_')
        proto_prim = stage.GetPrimAtPath(f"/root/{sanitized}")
        if proto_prim:
            xformable = UsdGeom.Xformable(proto_prim)
            xformable.ClearXformOpOrder()

    usd_proto_paths = [
        Sdf.Path(f"/root/{name.replace('.', '_')}") for name in proto_names
    ]

    instancer = UsdGeom.PointInstancer.Define(stage, "/root/FoliageInstancer")
    instancer.GetPrototypesRel().SetTargets(usd_proto_paths)
    instancer.GetProtoIndicesAttr().Set(Vt.IntArray(proto_indices))
    instancer.GetPositionsAttr().Set(Vt.Vec3fArray(positions))
    instancer.GetOrientationsAttr().Set(Vt.QuathArray(orientations))
    instancer.GetScalesAttr().Set(Vt.Vec3fArray(scales))

    stage.GetRootLayer().Save()
    inj_time = time.time() - t_inj
    total_time = time.time() - t0
    final_size_mb = os.path.getsize(out_usdc) / (1024 * 1024)

    print(f"  PointInstancer authored in {inj_time:.2f}s")
    print(f"Total conversion completed in {total_time:.2f}s (Final USDC size: {final_size_mb:.2f} MB)")
    print("==========================================================")
    print("  Scanlands OpenUSD Conversion SUCCESSFUL                 ")
    print("==========================================================")

if __name__ == "__main__" and IN_BLENDER:
    convert_scanlands()
