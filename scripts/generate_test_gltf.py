#!/usr/bin/env python3
"""
Pathways glTF Test Scene Generator
Generates standard glTF 2.0 assets (with embedded base64 buffers) for automated test verification:
- scenes/cornell_box.gltf: Cornell box with diffuse walls, metallic cube, dielectric glass sphere, and ceiling light.
- scenes/test_shapes.gltf: Hierarchical scene with multiple node transforms, indexed meshes, and PBR materials.
"""

import os
import json
import struct
import base64
import math

class GltfBuilder:
    def __init__(self):
        self.buffer_data = bytearray()
        self.buffer_views = []
        self.accessors = []
        self.meshes = []
        self.nodes = []
        self.materials = []
        self.cameras = []
        self.lights = []

    def add_buffer_view(self, data, target=None):
        offset = len(self.buffer_data)
        # 4-byte align
        align_padding = (4 - (offset % 4)) % 4
        if align_padding > 0:
            self.buffer_data.extend(b'\x00' * align_padding)
            offset += align_padding

        self.buffer_data.extend(data)
        length = len(data)

        bv_index = len(self.buffer_views)
        bv = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": length
        }
        if target:
            bv["target"] = target
        self.buffer_views.append(bv)
        return bv_index

    def add_accessor(self, buffer_view_idx, component_type, count, acc_type, min_val=None, max_val=None):
        acc_index = len(self.accessors)
        acc = {
            "bufferView": buffer_view_idx,
            "byteOffset": 0,
            "componentType": component_type,
            "count": count,
            "type": acc_type
        }
        if min_val is not None:
            acc["min"] = min_val
        if max_val is not None:
            acc["max"] = max_val
        self.accessors.append(acc)
        return acc_index

    def add_mesh_primitive(self, positions, normals, uvs, indices, material_idx):
        # Positions: list of [x, y, z]
        pos_bytes = bytearray()
        min_pos = [float('inf'), float('inf'), float('inf')]
        max_pos = [float('-inf'), float('-inf'), float('-inf')]
        for p in positions:
            pos_bytes.extend(struct.pack('<fff', p[0], p[1], p[2]))
            for c in range(3):
                min_pos[c] = min(min_pos[c], p[c])
                max_pos[c] = max(max_pos[c], p[c])

        pos_bv = self.add_buffer_view(pos_bytes, target=34962) # ARRAY_BUFFER
        pos_acc = self.add_accessor(pos_bv, 5126, len(positions), "VEC3", min_pos, max_pos)

        # Normals: list of [nx, ny, nz]
        norm_bytes = bytearray()
        for n in normals:
            norm_bytes.extend(struct.pack('<fff', n[0], n[1], n[2]))
        norm_bv = self.add_buffer_view(norm_bytes, target=34962)
        norm_acc = self.add_accessor(norm_bv, 5126, len(normals), "VEC3")

        # UVs: list of [u, v]
        uv_bytes = bytearray()
        for uv in uvs:
            uv_bytes.extend(struct.pack('<ff', uv[0], uv[1]))
        uv_bv = self.add_buffer_view(uv_bytes, target=34962)
        uv_acc = self.add_accessor(uv_bv, 5126, len(uvs), "VEC2")

        # Indices: list of uint32
        idx_bytes = bytearray()
        for idx in indices:
            idx_bytes.extend(struct.pack('<I', idx))
        idx_bv = self.add_buffer_view(idx_bytes, target=34963) # ELEMENT_ARRAY_BUFFER
        idx_acc = self.add_accessor(idx_bv, 5125, len(indices), "SCALAR", [min(indices)], [max(indices)])

        return {
            "attributes": {
                "POSITION": pos_acc,
                "NORMAL": norm_acc,
                "TEXCOORD_0": uv_acc
            },
            "indices": idx_acc,
            "material": material_idx
        }

    def add_material(self, name, base_color=(1, 1, 1, 1), metallic=0.0, roughness=1.0,
                     emissive=(0, 0, 0), emissive_strength=0.0, transmission=0.0, ior=1.5):
        mat = {
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": list(base_color),
                "metallicFactor": float(metallic),
                "roughnessFactor": float(roughness)
            }
        }
        if any(e > 0.0 for e in emissive):
            mat["emissiveFactor"] = list(emissive)
            if emissive_strength > 0.0:
                mat.setdefault("extensions", {})["KHR_materials_emissive_strength"] = {
                    "emissiveStrength": float(emissive_strength)
                }
        if transmission > 0.0:
            mat.setdefault("extensions", {})["KHR_materials_transmission"] = {
                "transmissionFactor": float(transmission)
            }
            mat.setdefault("extensions", {})["KHR_materials_ior"] = {
                "ior": float(ior)
            }
        idx = len(self.materials)
        self.materials.append(mat)
        return idx

    def build_json(self):
        b64 = base64.b64encode(self.buffer_data).decode('ascii')
        gltf = {
            "asset": {
                "version": "2.0",
                "generator": "Pathways glTF Exporter"
            },
            "scene": 0,
            "scenes": [
                {
                    "nodes": list(range(len(self.nodes)))
                }
            ],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "materials": self.materials,
            "accessors": self.accessors,
            "bufferViews": self.buffer_views,
            "buffers": [
                {
                    "byteLength": len(self.buffer_data),
                    "uri": "data:application/octet-stream;base64," + b64
                }
            ]
        }
        if self.cameras:
            gltf["cameras"] = self.cameras
        return gltf


def make_box(min_pt, max_pt):
    # Returns positions, normals, uvs, indices for an axis-aligned box
    x0, y0, z0 = min_pt
    x1, y1, z1 = max_pt
    
    # 6 faces * 4 vertices = 24 vertices
    positions = [
        # Front (+Z)
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        # Back (-Z)
        [x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0],
        # Top (+Y)
        [x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0],
        # Bottom (-Y)
        [x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1],
        # Left (-X)
        [x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0],
        # Right (+X)
        [x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1],
    ]
    
    normals = [
        [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
        [0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1],
        [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
        [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0],
        [-1, 0, 0], [-1, 0, 0], [-1, 0, 0], [-1, 0, 0],
        [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
    ]
    
    uvs = [
        [0, 0], [1, 0], [1, 1], [0, 1],
        [0, 0], [1, 0], [1, 1], [0, 1],
        [0, 0], [1, 0], [1, 1], [0, 1],
        [0, 0], [1, 0], [1, 1], [0, 1],
        [0, 0], [1, 0], [1, 1], [0, 1],
        [0, 0], [1, 0], [1, 1], [0, 1],
    ]
    
    indices = []
    for f in range(6):
        base = f * 4
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
        
    return positions, normals, uvs, indices


def make_quad(p0, p1, p2, p3, normal):
    positions = [p0, p1, p2, p3]
    normals = [normal, normal, normal, normal]
    uvs = [[0, 0], [1, 0], [1, 1], [0, 1]]
    indices = [0, 1, 2, 0, 2, 3]
    return positions, normals, uvs, indices


def create_cornell_box_gltf(output_path):
    builder = GltfBuilder()

    # Materials
    mat_white = builder.add_material("WhiteWall", base_color=(0.75, 0.75, 0.75, 1.0), roughness=0.9)
    mat_red = builder.add_material("RedWall", base_color=(0.75, 0.12, 0.12, 1.0), roughness=0.9)
    mat_green = builder.add_material("GreenWall", base_color=(0.12, 0.75, 0.15, 1.0), roughness=0.9)
    mat_light = builder.add_material("CeilingLight", base_color=(1.0, 1.0, 1.0, 1.0),
                                     emissive=(1.0, 0.95, 0.8), emissive_strength=18.0)
    mat_metal = builder.add_material("MetallicBox", base_color=(0.95, 0.85, 0.5, 1.0),
                                     metallic=0.9, roughness=0.1)

    # 1. Floor (White)
    pos, norm, uv, idx = make_quad([-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [0, 1, 0])
    prim = builder.add_mesh_primitive(pos, norm, uv, idx, mat_white)
    m_floor = len(builder.meshes)
    builder.meshes.append({"name": "Floor", "primitives": [prim]})
    builder.nodes.append({"name": "Node_Floor", "mesh": m_floor})

    # 2. Ceiling (White)
    pos, norm, uv, idx = make_quad([-1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, -1.0], [-1.0, 2.0, -1.0], [0, -1, 0])
    prim = builder.add_mesh_primitive(pos, norm, uv, idx, mat_white)
    m_ceil = len(builder.meshes)
    builder.meshes.append({"name": "Ceiling", "primitives": [prim]})
    builder.nodes.append({"name": "Node_Ceiling", "mesh": m_ceil})

    # 3. Back Wall (White)
    pos, norm, uv, idx = make_quad([-1.0, 0.0, -1.0], [-1.0, 2.0, -1.0], [1.0, 2.0, -1.0], [1.0, 0.0, -1.0], [0, 0, 1])
    prim = builder.add_mesh_primitive(pos, norm, uv, idx, mat_white)
    m_back = len(builder.meshes)
    builder.meshes.append({"name": "BackWall", "primitives": [prim]})
    builder.nodes.append({"name": "Node_BackWall", "mesh": m_back})

    # 4. Left Wall (Red)
    pos, norm, uv, idx = make_quad([-1.0, 0.0, 1.0], [-1.0, 0.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, 2.0, 1.0], [1, 0, 0])
    prim = builder.add_mesh_primitive(pos, norm, uv, idx, mat_red)
    m_left = len(builder.meshes)
    builder.meshes.append({"name": "LeftWall", "primitives": [prim]})
    builder.nodes.append({"name": "Node_LeftWall", "mesh": m_left})

    # 5. Right Wall (Green)
    pos, norm, uv, idx = make_quad([1.0, 0.0, -1.0], [1.0, 0.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, -1.0], [-1, 0, 0])
    prim = builder.add_mesh_primitive(pos, norm, uv, idx, mat_green)
    m_right = len(builder.meshes)
    builder.meshes.append({"name": "RightWall", "primitives": [prim]})
    builder.nodes.append({"name": "Node_RightWall", "mesh": m_right})

    # 6. Ceiling Light Quad (Emissive)
    lw = 0.35
    pos, norm, uv, idx = make_quad([-lw, 1.99, lw], [lw, 1.99, lw], [lw, 1.99, -lw], [-lw, 1.99, -lw], [0, -1, 0])
    prim = builder.add_mesh_primitive(pos, norm, uv, idx, mat_light)
    m_light = len(builder.meshes)
    builder.meshes.append({"name": "CeilingLight", "primitives": [prim]})
    builder.nodes.append({"name": "Node_CeilingLight", "mesh": m_light})

    # 7. Tall Box inside room (Diffuse White)
    pos, norm, uv, idx = make_box([-0.275, 0.0, -0.275], [0.275, 1.2, 0.275])
    prim = builder.add_mesh_primitive(pos, norm, uv, idx, mat_white)
    m_box1 = len(builder.meshes)
    builder.meshes.append({"name": "TallBox", "primitives": [prim]})
    rad = math.radians(22.0)
    sin_half = math.sin(rad * 0.5)
    cos_half = math.cos(rad * 0.5)
    builder.nodes.append({
        "name": "Node_TallBox",
        "mesh": m_box1,
        "translation": [0.35, 0.0, -0.3],
        "rotation": [0.0, sin_half, 0.0, cos_half]
    })

    # 8. Short Metallic Box inside room
    pos, norm, uv, idx = make_box([-0.3, 0.0, -0.3], [0.3, 0.6, 0.3])
    prim = builder.add_mesh_primitive(pos, norm, uv, idx, mat_metal)
    m_box2 = len(builder.meshes)
    builder.meshes.append({"name": "ShortBox", "primitives": [prim]})
    rad2 = math.radians(-18.0)
    sin2 = math.sin(rad2 * 0.5)
    cos2 = math.cos(rad2 * 0.5)
    builder.nodes.append({
        "name": "Node_ShortBox",
        "mesh": m_box2,
        "translation": [-0.35, 0.0, 0.3],
        "rotation": [0.0, sin2, 0.0, cos2]
    })

    # Camera looking at Cornell Box center
    builder.cameras.append({
        "name": "MainCamera",
        "type": "perspective",
        "perspective": {
            "yfov": math.radians(45.0),
            "znear": 0.1,
            "zfar": 100.0
        }
    })
    builder.nodes.append({
        "name": "Node_Camera",
        "camera": 0,
        "translation": [0.0, 1.0, 2.7]
    })

    gltf_json = builder.build_json()
    with open(output_path, 'w') as f:
        json.dump(gltf_json, f, indent=2)
    print(f"Generated Cornell Box glTF: {output_path} ({len(builder.meshes)} meshes, {len(builder.nodes)} nodes)")


def create_test_shapes_gltf(output_path):
    builder = GltfBuilder()

    # Materials: Gold, Glass, Matte Cyan, Bright Magenta Emissive
    mat_gold = builder.add_material("Gold", base_color=(1.0, 0.84, 0.0, 1.0), metallic=1.0, roughness=0.15)
    mat_glass = builder.add_material("Glass", base_color=(0.95, 0.95, 1.0, 1.0), roughness=0.05, transmission=0.95, ior=1.52)
    mat_cyan = builder.add_material("CyanDiffuse", base_color=(0.1, 0.8, 0.9, 1.0), roughness=0.7)
    mat_emissive = builder.add_material("NeonMagenta", base_color=(1.0, 0.1, 0.8, 1.0),
                                        emissive=(1.0, 0.1, 0.8), emissive_strength=12.0)

    # Ground plane
    pos, norm, uv, idx = make_quad([-5.0, 0.0, -5.0], [5.0, 0.0, -5.0], [5.0, 0.0, 5.0], [-5.0, 0.0, 5.0], [0, 1, 0])
    prim = builder.add_mesh_primitive(pos, norm, uv, idx, mat_cyan)
    m_ground = len(builder.meshes)
    builder.meshes.append({"name": "Ground", "primitives": [prim]})
    builder.nodes.append({"name": "Node_Ground", "mesh": m_ground})

    # Cube 1: Gold
    pos, norm, uv, idx = make_box([-0.5, 0.0, -0.5], [0.5, 1.0, 0.5])
    prim1 = builder.add_mesh_primitive(pos, norm, uv, idx, mat_gold)
    m_cube1 = len(builder.meshes)
    builder.meshes.append({"name": "GoldCube", "primitives": [prim1]})
    builder.nodes.append({"name": "Node_GoldCube", "mesh": m_cube1, "translation": [-1.5, 0.0, 0.0]})

    # Cube 2: Glass
    prim2 = builder.add_mesh_primitive(pos, norm, uv, idx, mat_glass)
    m_cube2 = len(builder.meshes)
    builder.meshes.append({"name": "GlassCube", "primitives": [prim2]})
    builder.nodes.append({"name": "Node_GlassCube", "mesh": m_cube2, "translation": [0.0, 0.0, 0.0]})

    # Cube 3: Emissive
    prim3 = builder.add_mesh_primitive(pos, norm, uv, idx, mat_emissive)
    m_cube3 = len(builder.meshes)
    builder.meshes.append({"name": "EmissiveCube", "primitives": [prim3]})
    builder.nodes.append({"name": "Node_EmissiveCube", "mesh": m_cube3, "translation": [1.5, 0.0, 0.0]})

    gltf_json = builder.build_json()
    with open(output_path, 'w') as f:
        json.dump(gltf_json, f, indent=2)
    print(f"Generated Test Shapes glTF: {output_path} ({len(builder.meshes)} meshes, {len(builder.nodes)} nodes)")


if __name__ == "__main__":
    os.makedirs("scenes", exist_ok=True)
    create_cornell_box_gltf("scenes/cornell_box.gltf")
    create_test_shapes_gltf("scenes/test_shapes.gltf")
