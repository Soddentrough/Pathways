#!/usr/bin/env python3
"""
Pathways glTF Test Scene Generator
Generates standard glTF 2.0 assets (with embedded base64 buffers) for automated test verification:
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
    create_test_shapes_gltf("scenes/test_shapes.gltf")
