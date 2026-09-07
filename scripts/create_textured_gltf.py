#!/usr/bin/env python3
import struct
import base64
import json

def generate_textured_gltf():
    # Quad centered in X-Y plane facing +Z, or X-Z plane
    # V0: (-2, -1.5, 0) UV (0, 0)
    # V1: ( 2, -1.5, 0) UV (2, 0)
    # V2: ( 2,  1.5, 0) UV (2, 2)
    # V3: (-2,  1.5, 0) UV (0, 2)
    # Normal: (0, 0, 1)

    positions = [
        -2.0, -1.5, 0.0,
         2.0, -1.5, 0.0,
         2.0,  1.5, 0.0,
        -2.0,  1.5, 0.0
    ]
    normals = [
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0
    ]
    texcoords = [
        0.0, 0.0,
        2.0, 0.0,
        2.0, 2.0,
        0.0, 2.0
    ]
    indices = [
        0, 1, 2,
        0, 2, 3
    ]

    pos_bytes = struct.pack(f"<{len(positions)}f", *positions)
    norm_bytes = struct.pack(f"<{len(normals)}f", *normals)
    uv_bytes = struct.pack(f"<{len(texcoords)}f", *texcoords)
    idx_bytes = struct.pack(f"<{len(indices)}H", *indices)

    def pad(b):
        return b + b"\x00" * ((4 - (len(b) % 4)) % 4)

    b_all = pad(pos_bytes) + pad(norm_bytes) + pad(uv_bytes) + pad(idx_bytes)
    b64_data = base64.b64encode(b_all).decode("ascii")

    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "Pathways Textured Scene Generator"
        },
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [
            {"mesh": 0, "name": "TexturedPlane"},
            {"camera": 0, "name": "MainCamera", "translation": [0.0, 0.0, 3.5]}
        ],
        "cameras": [
            {
                "type": "perspective",
                "perspective": {
                    "yfov": 0.8,
                    "znear": 0.1,
                    "zfar": 100.0
                }
            }
        ],
        "meshes": [{
            "name": "PlaneMesh",
            "primitives": [{
                "attributes": {
                    "POSITION": 0,
                    "NORMAL": 1,
                    "TEXCOORD_0": 2
                },
                "indices": 3,
                "material": 0
            }]
        }],
        "materials": [{
            "name": "CheckeredMaterial",
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.1,
                "roughnessFactor": 0.5
            },
            "normalTexture": {"index": 1}
        }],
        "textures": [
            {"source": 0},
            {"source": 1}
        ],
        "images": [
            {"uri": "checker.png"},
            {"uri": "ripples_normal.png"}
        ],
        "accessors": [
            {"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 4, "type": "VEC3",
             "min": [-2.0, -1.5, 0.0], "max": [2.0, 1.5, 0.0]},
            {"bufferView": 1, "byteOffset": 0, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 2, "byteOffset": 0, "componentType": 5126, "count": 4, "type": "VEC2"},
            {"bufferView": 3, "byteOffset": 0, "componentType": 5123, "count": 6, "type": "SCALAR"}
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_bytes)},
            {"buffer": 0, "byteOffset": len(pad(pos_bytes)), "byteLength": len(norm_bytes)},
            {"buffer": 0, "byteOffset": len(pad(pos_bytes)) + len(pad(norm_bytes)), "byteLength": len(uv_bytes)},
            {"buffer": 0, "byteOffset": len(pad(pos_bytes)) + len(pad(norm_bytes)) + len(pad(uv_bytes)), "byteLength": len(idx_bytes)}
        ],
        "buffers": [{
            "byteLength": len(b_all),
            "uri": f"data:application/octet-stream;base64,{b64_data}"
        }]
    }

    with open("scenes/textured_quad.gltf", "w") as f:
        json.dump(gltf, f, indent=2)
    print("Created scenes/textured_quad.gltf")

if __name__ == "__main__":
    generate_textured_gltf()
