#!/usr/bin/env python3
"""
Generate glTF 2.0 PBR Showcase Scene for Pathways
Tests:
- Base color texture (sRGB)
- Metallic-Roughness texture (Linear: G=roughness, B=metallic)
- Normal map (Linear: tangent-space normal perturbations)
- Ambient occlusion texture (Linear: R=occlusion)
- Emissive texture (sRGB) + KHR_materials_emissive_strength
- Alpha Mask Cutout (alphaMode: "MASK", alphaCutoff: 0.5)
- Glass Transmission (KHR_materials_transmission + KHR_materials_ior)
- Cook-Torrance GGX microfacet specular and diffuse shading
"""

import os
import math
import json
import struct
import base64
import numpy as np
from PIL import Image

def generate_textures(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    res = 256

    # 1. Albedo Map (sRGB): Tiles with metallic border and brushed center
    albedo = np.zeros((res, res, 4), dtype=np.uint8)
    for y in range(res):
        for x in range(res):
            border = (x < 16 or x >= res - 16 or y < 16 or y >= res - 16)
            if border:
                albedo[y, x] = [180, 160, 120, 255] # Brass/bronze frame
            else:
                # Brushed blue steel pattern
                stripes = int(128 + 40 * math.sin(x * 0.2))
                albedo[y, x] = [50, 90, stripes, 255]
    Image.fromarray(albedo).save(os.path.join(out_dir, "pbr_albedo.png"))

    # 2. Metallic-Roughness Map (Linear):
    # glTF standard: Green = Roughness, Blue = Metallic
    mr = np.zeros((res, res, 4), dtype=np.uint8)
    for y in range(res):
        for x in range(res):
            border = (x < 16 or x >= res - 16 or y < 16 or y >= res - 16)
            if border:
                # Highly metallic, smooth border
                mr[y, x] = [0, 40, 240, 255] # roughness ~0.15, metallic ~0.94
            else:
                # Center: rough non-metal in stripes, metallic in grooves
                is_metal = 180 if ((x // 32) % 2 == 0) else 20
                roughness = 60 if ((y // 32) % 2 == 0) else 200
                mr[y, x] = [0, roughness, is_metal, 255]
    Image.fromarray(mr).save(os.path.join(out_dir, "pbr_mr.png"))

    # 3. Normal Map (Linear): Tangent-space bevels and pyramidal studs
    normal = np.zeros((res, res, 4), dtype=np.uint8)
    for y in range(res):
        for x in range(res):
            # Center concentric ripple normals
            dx = (x - 128) / 128.0
            dy = (y - 128) / 128.0
            r = math.sqrt(dx * dx + dy * dy)
            if r > 0.001 and r < 0.8:
                nx = -math.sin(r * 25.0) * (dx / r) * 0.4
                ny = -math.sin(r * 25.0) * (dy / r) * 0.4
            else:
                nx = 0.0
                ny = 0.0
            nz = math.sqrt(max(0.0, 1.0 - nx * nx - ny * ny))
            r_byte = int((nx * 0.5 + 0.5) * 255.0)
            g_byte = int((ny * 0.5 + 0.5) * 255.0)
            b_byte = int((nz * 0.5 + 0.5) * 255.0)
            normal[y, x] = [r_byte, g_byte, b_byte, 255]
    Image.fromarray(normal).save(os.path.join(out_dir, "pbr_normal.png"))

    # 4. Occlusion Map (Linear): R = ambient occlusion factor (0 = occluded, 255 = unoccluded)
    ao = np.zeros((res, res, 4), dtype=np.uint8)
    for y in range(res):
        for x in range(res):
            dist_edge = min(x, res - 1 - x, y, res - 1 - y)
            factor = min(255, int(dist_edge * 8))
            ao[y, x] = [factor, factor, factor, 255]
    Image.fromarray(ao).save(os.path.join(out_dir, "pbr_ao.png"))

    # 5. Emissive Map (sRGB): Glowing central logo/ring
    emissive = np.zeros((res, res, 4), dtype=np.uint8)
    for y in range(res):
        for x in range(res):
            dx = (x - 128) / 128.0
            dy = (y - 128) / 128.0
            r = math.sqrt(dx * dx + dy * dy)
            if 0.35 <= r <= 0.45:
                # Glowing neon cyan ring
                emissive[y, x] = [0, 255, 220, 255]
            elif 0.0 <= r <= 0.15:
                # Glowing core
                emissive[y, x] = [255, 120, 0, 255]
            else:
                emissive[y, x] = [0, 0, 0, 255]
    Image.fromarray(emissive).save(os.path.join(out_dir, "pbr_emissive.png"))

    # 6. Alpha Mask Texture (sRGB): Cutout grille with circular holes
    alpha_mask = np.zeros((res, res, 4), dtype=np.uint8)
    for y in range(res):
        for x in range(res):
            # Lattice of circular holes
            cx = (x % 64) - 32
            cy = (y % 64) - 32
            dist = math.sqrt(cx * cx + cy * cy)
            if dist < 22:
                # Cutout hole: alpha = 0
                alpha_mask[y, x] = [0, 0, 0, 0]
            else:
                # Opaque bronze lattice: alpha = 255
                alpha_mask[y, x] = [210, 150, 60, 255]
    Image.fromarray(alpha_mask).save(os.path.join(out_dir, "pbr_alpha_mask.png"))
    print(f"Generated 6 PBR textures in: {out_dir}")

def generate_gltf(out_path):
    # Builds glTF JSON with references to generated textures
    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "Pathways PBR Showcase Generator"
        },
        "scene": 0,
        "scenes": [{"nodes": [0, 1, 2, 3, 4, 5]}],
        "images": [
            {"uri": "pbr_albedo.png"},
            {"uri": "pbr_mr.png"},
            {"uri": "pbr_normal.png"},
            {"uri": "pbr_ao.png"},
            {"uri": "pbr_emissive.png"},
            {"uri": "pbr_alpha_mask.png"}
        ],
        "textures": [
            {"source": 0}, # 0: albedo
            {"source": 1}, # 1: mr
            {"source": 2}, # 2: normal
            {"source": 3}, # 3: ao
            {"source": 4}, # 4: emissive
            {"source": 5}  # 5: alpha mask
        ],
        "materials": [
            # 0: Full PBR Material (albedo, mr, normal, ao, emissive)
            {
                "name": "FullPBR_Panel",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicRoughnessTexture": {"index": 1}
                },
                "normalTexture": {"index": 2, "scale": 1.0},
                "occlusionTexture": {"index": 3, "strength": 1.0},
                "emissiveTexture": {"index": 4},
                "emissiveFactor": [1.0, 1.0, 1.0],
                "extensions": {
                    "KHR_materials_emissive_strength": {"emissiveStrength": 5.0}
                }
            },
            # 1: Alpha Cutout Grille
            {
                "name": "AlphaCutout_Grille",
                "alphaMode": "MASK",
                "alphaCutoff": 0.5,
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 5},
                    "metallicFactor": 0.8,
                    "roughnessFactor": 0.2
                }
            },
            # 2: Dielectric Glass Transmission
            {
                "name": "DielectricGlass",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.95, 0.95, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.02
                },
                "extensions": {
                    "KHR_materials_transmission": {"transmissionFactor": 0.98},
                    "KHR_materials_ior": {"ior": 1.52}
                }
            },
            # 3: Backing Red Wall (behind the alpha mask to prove cutout visibility)
            {
                "name": "BackingWall",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.85, 0.08, 0.08, 1.0],
                    "metallicFactor": 0.1,
                    "roughnessFactor": 0.5
                }
            },
            # 4: Floor (Checkerboard-style diffuse)
            {
                "name": "FloorGrey",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.35, 0.35, 0.38, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.8
                }
            }
        ],
        "nodes": [
            {"name": "Node_PBRPanel", "mesh": 0, "translation": [-1.2, 0.8, 0.0]},
            {"name": "Node_AlphaGrille", "mesh": 1, "translation": [1.2, 0.8, 0.4]},
            {"name": "Node_BackingWall", "mesh": 2, "translation": [1.2, 0.8, -0.4]},
            {"name": "Node_GlassCube", "mesh": 3, "translation": [0.0, 0.6, 0.2], "scale": [0.6, 0.6, 0.6]},
            {"name": "Node_Floor", "mesh": 4, "translation": [0.0, 0.0, 0.0]},
            {"name": "Node_Camera", "camera": 0, "translation": [0.0, 1.0, 3.8]}
        ],
        "cameras": [
            {
                "type": "perspective",
                "perspective": {
                    "yfov": math.radians(45.0),
                    "znear": 0.1,
                    "zfar": 100.0
                }
            }
        ]
    }

    # Binary buffer builder
    buf = bytearray()
    buffer_views = []
    accessors = []

    def add_bv(data, target=None):
        pad = (4 - (len(buf) % 4)) % 4
        if pad: buf.extend(b'\x00' * pad)
        off = len(buf)
        buf.extend(data)
        idx = len(buffer_views)
        bv = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target: bv["target"] = target
        buffer_views.append(bv)
        return idx

    def add_acc(bv_idx, comp_type, count, acc_type, min_v=None, max_v=None):
        idx = len(accessors)
        acc = {"bufferView": bv_idx, "byteOffset": 0, "componentType": comp_type, "count": count, "type": acc_type}
        if min_v is not None: acc["min"] = min_v
        if max_v is not None: acc["max"] = max_v
        accessors.append(acc)
        return idx

    def make_quad(p0, p1, p2, p3, n, t, mat_id):
        # 4 vertices
        pos_b = bytearray()
        norm_b = bytearray()
        tang_b = bytearray()
        uv_b = bytearray()
        pts = [p0, p1, p2, p3]
        uvs = [[0, 0], [1, 0], [1, 1], [0, 1]]
        for p in pts: pos_b.extend(struct.pack('<fff', *p))
        for _ in pts: norm_b.extend(struct.pack('<fff', *n))
        for _ in pts: tang_b.extend(struct.pack('<ffff', t[0], t[1], t[2], 1.0))
        for uv in uvs: uv_b.extend(struct.pack('<ff', *uv))
        
        idx_b = bytearray()
        for i in [0, 1, 2, 0, 2, 3]: idx_b.extend(struct.pack('<I', i))

        pos_bv = add_bv(pos_b, 34962)
        norm_bv = add_bv(norm_b, 34962)
        tang_bv = add_bv(tang_b, 34962)
        uv_bv = add_bv(uv_b, 34962)
        idx_bv = add_bv(idx_b, 34963)

        min_p = [min(p[c] for p in pts) for c in range(3)]
        max_p = [max(p[c] for p in pts) for c in range(3)]

        pos_acc = add_acc(pos_bv, 5126, 4, "VEC3", min_p, max_p)
        norm_acc = add_acc(norm_bv, 5126, 4, "VEC3")
        tang_acc = add_acc(tang_bv, 5126, 4, "VEC4")
        uv_acc = add_acc(uv_bv, 5126, 4, "VEC2")
        idx_acc = add_acc(idx_bv, 5125, 6, "SCALAR", [0], [3])

        return {
            "primitives": [{
                "attributes": {
                    "POSITION": pos_acc,
                    "NORMAL": norm_acc,
                    "TANGENT": tang_acc,
                    "TEXCOORD_0": uv_acc
                },
                "indices": idx_acc,
                "material": mat_id
            }]
        }

    def make_box(half_size, mat_id):
        # 6 faces of a cube
        h = half_size
        faces = [
            # Front (+Z)
            ([-h, -h,  h], [ h, -h,  h], [ h,  h,  h], [-h,  h,  h], [0, 0, 1], [1, 0, 0]),
            # Back (-Z)
            ([ h, -h, -h], [-h, -h, -h], [-h,  h, -h], [ h,  h, -h], [0, 0, -1], [-1, 0, 0]),
            # Top (+Y)
            ([-h,  h,  h], [ h,  h,  h], [ h,  h, -h], [-h,  h, -h], [0, 1, 0], [1, 0, 0]),
            # Bottom (-Y)
            ([-h, -h, -h], [ h, -h, -h], [ h, -h,  h], [-h, -h,  h], [0, -1, 0], [1, 0, 0]),
            # Left (-X)
            ([-h, -h, -h], [-h, -h,  h], [-h,  h,  h], [-h,  h, -h], [-1, 0, 0], [0, 0, 1]),
            # Right (+X)
            ([ h, -h,  h], [ h, -h, -h], [ h,  h, -h], [ h,  h,  h], [1, 0, 0], [0, 0, -1]),
        ]
        pos_b = bytearray()
        norm_b = bytearray()
        tang_b = bytearray()
        uv_b = bytearray()
        idx_b = bytearray()

        min_p = [-h, -h, -h]
        max_p = [h, h, h]

        v_base = 0
        for p0, p1, p2, p3, n, t in faces:
            pts = [p0, p1, p2, p3]
            uvs = [[0, 0], [1, 0], [1, 1], [0, 1]]
            for p in pts: pos_b.extend(struct.pack('<fff', *p))
            for _ in pts: norm_b.extend(struct.pack('<fff', *n))
            for _ in pts: tang_b.extend(struct.pack('<ffff', t[0], t[1], t[2], 1.0))
            for uv in uvs: uv_b.extend(struct.pack('<ff', *uv))
            for i in [0, 1, 2, 0, 2, 3]: idx_b.extend(struct.pack('<I', v_base + i))
            v_base += 4

        pos_bv = add_bv(pos_b, 34962)
        norm_bv = add_bv(norm_b, 34962)
        tang_bv = add_bv(tang_b, 34962)
        uv_bv = add_bv(uv_b, 34962)
        idx_bv = add_bv(idx_b, 34963)

        pos_acc = add_acc(pos_bv, 5126, 24, "VEC3", min_p, max_p)
        norm_acc = add_acc(norm_bv, 5126, 24, "VEC3")
        tang_acc = add_acc(tang_bv, 5126, 24, "VEC4")
        uv_acc = add_acc(uv_bv, 5126, 24, "VEC2")
        idx_acc = add_acc(idx_bv, 5125, 36, "SCALAR", [0], [23])

        return {
            "primitives": [{
                "attributes": {
                    "POSITION": pos_acc,
                    "NORMAL": norm_acc,
                    "TANGENT": tang_acc,
                    "TEXCOORD_0": uv_acc
                },
                "indices": idx_acc,
                "material": mat_id
            }]
        }

    # Mesh 0: PBR Panel (Facing camera)
    mesh0 = make_quad([-0.8, -0.8, 0.0], [0.8, -0.8, 0.0], [0.8, 0.8, 0.0], [-0.8, 0.8, 0.0],
                      [0, 0, 1], [1, 0, 0], 0)
    # Mesh 1: Alpha Cutout Grille (Facing camera)
    mesh1 = make_quad([-0.7, -0.7, 0.0], [0.7, -0.7, 0.0], [0.7, 0.7, 0.0], [-0.7, 0.7, 0.0],
                      [0, 0, 1], [1, 0, 0], 1)
    # Mesh 2: Red Backing Wall (Behind grille)
    mesh2 = make_quad([-0.9, -0.9, 0.0], [0.9, -0.9, 0.0], [0.9, 0.9, 0.0], [-0.9, 0.9, 0.0],
                      [0, 0, 1], [1, 0, 0], 3)
    # Mesh 3: Glass Cube (in center foreground)
    mesh3 = make_box(0.5, 2)
    # Mesh 4: Floor Plane
    mesh4 = make_quad([-5.0, 0.0, -5.0], [5.0, 0.0, -5.0], [5.0, 0.0, 5.0], [-5.0, 0.0, 5.0],
                      [0, 1, 0], [1, 0, 0], 4)

    gltf["meshes"] = [mesh0, mesh1, mesh2, mesh3, mesh4]
    b64_buf = base64.b64encode(buf).decode('ascii')
    gltf["buffers"] = [{
        "byteLength": len(buf),
        "uri": "data:application/octet-stream;base64," + b64_buf
    }]
    gltf["bufferViews"] = buffer_views
    gltf["accessors"] = accessors

    with open(out_path, 'w') as f:
        json.dump(gltf, f, indent=2)
    print(f"Saved PBR Showcase glTF to: {out_path}")

if __name__ == "__main__":
    scenes_dir = os.path.join(os.path.dirname(__file__), "..", "scenes")
    generate_textures(scenes_dir)
    generate_gltf(os.path.join(scenes_dir, "pbr_showcase.gltf"))
