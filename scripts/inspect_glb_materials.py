#!/usr/bin/env python3
"""
Inspect glTF/GLB Materials and Meshes
Parses the JSON chunk of a GLB (or a .gltf file) and prints a structured summary
of materials, their PBR parameters, extensions, extras, and primitive assignments.
"""

import sys
import os
import json
import struct

def inspect_glb(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    with open(filepath, "rb") as f:
        magic = f.read(4)
        if magic == b"glTF":
            version, length = struct.unpack("<II", f.read(8))
            chunk_length, chunk_type = struct.unpack("<II", f.read(8))
            if chunk_type != 0x4E4F534A: # 'JSON'
                print(f"Error: First chunk is not JSON (type 0x{chunk_type:08X})", file=sys.stderr)
                sys.exit(1)
            json_bytes = f.read(chunk_length)
            data = json.loads(json_bytes.decode("utf-8"))
        else:
            f.seek(0)
            data = json.load(f)

    print(f"=== GLB Inspection: {filepath} ===")
    materials = data.get("materials", [])
    print(f"Total Materials: {len(materials)}")
    for i, mat in enumerate(materials):
        name = mat.get("name", f"mat_{i}")
        pbr = mat.get("pbrMetallicRoughness", {})
        base_color = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
        metallic = pbr.get("metallicFactor", 1.0)
        roughness = pbr.get("roughnessFactor", 1.0)
        extensions = mat.get("extensions", {})
        extras = mat.get("extras", {})

        print(f"\nMaterial [{i}] '{name}':")
        print(f"  Base Color Factor: {base_color}")
        print(f"  Metallic: {metallic}, Roughness: {roughness}")
        if "KHR_materials_transmission" in extensions:
            trans = extensions["KHR_materials_transmission"]
            print(f"  KHR_materials_transmission: factor={trans.get('transmissionFactor', 0.0)}")
        if "KHR_materials_ior" in extensions:
            ior = extensions["KHR_materials_ior"]
            print(f"  KHR_materials_ior: ior={ior.get('ior', 1.5)}")
        if "KHR_materials_volume" in extensions:
            vol = extensions["KHR_materials_volume"]
            print(f"  KHR_materials_volume: thickness={vol.get('thicknessFactor', 0.0)}, attenColor={vol.get('attenuationColor')}, attenDist={vol.get('attenuationDistance')}")
        if "KHR_materials_clearcoat" in extensions:
            print(f"  KHR_materials_clearcoat: {extensions['KHR_materials_clearcoat']}")
        if "KHR_materials_specular" in extensions:
            print(f"  KHR_materials_specular: {extensions['KHR_materials_specular']}")
        if extras:
            print(f"  Extras: {extras}")

    meshes = data.get("meshes", [])
    print(f"\nTotal Meshes: {len(meshes)}")
    for i, mesh in enumerate(meshes):
        m_name = mesh.get("name", f"mesh_{i}")
        prims = mesh.get("primitives", [])
        mat_indices = [p.get("material", "None") for p in prims]
        print(f"  Mesh [{i}] '{m_name}': {len(prims)} primitives, materials={mat_indices}")

    nodes = data.get("nodes", [])
    print(f"\nTotal Nodes: {len(nodes)}")
    for i, node in enumerate(nodes):
        n_name = node.get("name", f"node_{i}")
        mesh_idx = node.get("mesh", None)
        if mesh_idx is not None:
            m_name = meshes[mesh_idx].get("name", f"mesh_{mesh_idx}") if mesh_idx < len(meshes) else "?"
            print(f"  Node [{i}] '{n_name}': mesh={mesh_idx} ('{m_name}')")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_glb_materials.py <path_to_glb_or_gltf>")
        sys.exit(1)
    inspect_glb(sys.argv[1])
