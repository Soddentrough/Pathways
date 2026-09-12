#!/usr/bin/env python3
"""
bin_to_image.py - Inspection tool for Pathways raw binary training tensors (.bin).
Unpacks any frame tensor into individual PNG/EXR channel maps for debugging.
"""

import sys
import os
import argparse
import struct
import numpy as np

HEADER_FORMAT = "<4sIIIIIIIQ24s"
HEADER_SIZE = 64

def parse_header(filepath):
    with open(filepath, "rb") as f:
        data = f.read(HEADER_SIZE)
        if len(data) < HEADER_SIZE:
            raise ValueError(f"File {filepath} is smaller than 64-byte header.")
        magic, version, width, height, channels, data_type, frame_index, spp, payload_size, _ = struct.unpack(HEADER_FORMAT, data)
        magic_str = magic.decode("ascii", errors="replace")
        if magic_str != "PTTD":
            raise ValueError(f"Invalid magic '{magic_str}'. Expected 'PTTD'.")
        dtype = np.float16 if data_type == 0 else np.float32
        return {
            "magic": magic_str,
            "version": version,
            "width": width,
            "height": height,
            "channels": channels,
            "data_type": dtype,
            "frame_index": frame_index,
            "spp": spp,
            "payload_size": payload_size
        }

def load_tensor(filepath):
    hdr = parse_header(filepath)
    tensor = np.memmap(filepath, dtype=hdr["data_type"], mode="r", offset=HEADER_SIZE,
                       shape=(hdr["height"], hdr["width"], hdr["channels"]))
    return hdr, np.array(tensor, copy=True)

def save_image_channels(hdr, tensor, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    from PIL import Image

    def to_uint8(arr):
        arr_clipped = np.clip(arr, 0.0, 1.0)
        return (arr_clipped * 255.0).astype(np.uint8)

    prefix = f"frame_{hdr['frame_index']:05d}"
    channels = hdr["channels"]

    if channels == 4:
        # Reference Ground Truth (RGB Radiance + Alpha/Samples)
        rgb = tensor[..., :3].astype(np.float32)
        samples = tensor[..., 3:4].astype(np.float32)
        if np.any(samples > 1.0):
            rgb = rgb / np.maximum(samples, 1.0)
        # Reinhard tonemap for preview
        ldr = to_uint8(rgb / (1.0 + rgb))
        Image.fromarray(ldr).save(os.path.join(out_dir, f"{prefix}_reference_preview.png"))
        print(f"Saved {prefix}_reference_preview.png (shape: {tensor.shape})")
    elif channels >= 16:
        # 16-channel training tensor:
        # 0..2: Demod Diffuse (RGB)
        # 3..5: Demod Specular (RGB)
        # 6..8: Albedo (RGB)
        # 9: Roughness
        # 10: Depth
        # 11: Hit Distance
        # 12..13: V_surface (RG)
        # 14..15: V_specular (RG)

        diff = tensor[..., 0:3]
        spec = tensor[..., 3:6]
        albedo = tensor[..., 6:9]
        roughness = tensor[..., 9]
        depth = tensor[..., 10]
        hit_dist = tensor[..., 11]
        v_surf = tensor[..., 12:14]
        v_spec = tensor[..., 14:16]

        # Save Albedo
        Image.fromarray(to_uint8(albedo)).save(os.path.join(out_dir, f"{prefix}_albedo.png"))

        # Save Roughness
        Image.fromarray(to_uint8(roughness)).save(os.path.join(out_dir, f"{prefix}_roughness.png"))

        # Save Depth (normalized for display)
        max_d = np.percentile(depth[depth < 9000], 99) if np.any(depth < 9000) else 10.0
        depth_norm = np.clip(depth / max(max_d, 1e-3), 0.0, 1.0)
        Image.fromarray(to_uint8(depth_norm)).save(os.path.join(out_dir, f"{prefix}_depth.png"))

        # Save Demod Diffuse & Specular (tonemapped preview)
        Image.fromarray(to_uint8(diff / (1.0 + diff))).save(os.path.join(out_dir, f"{prefix}_demod_diffuse.png"))
        Image.fromarray(to_uint8(spec / (1.0 + spec))).save(os.path.join(out_dir, f"{prefix}_demod_specular.png"))

        # Recombined Radiance Preview: (diff * albedo + spec)
        comb = diff * albedo + spec
        Image.fromarray(to_uint8(comb / (1.0 + comb))).save(os.path.join(out_dir, f"{prefix}_recombined_noisy.png"))

        # Save Motion Vector Flow visualizations
        def flow_to_rgb(flow):
            flow = flow.astype(np.float32)
            mag = np.linalg.norm(flow, axis=-1)
            max_mag = np.percentile(mag, 99) if np.max(mag) > 0 else 1.0
            norm_flow = flow / max(max_mag, 1e-4) * 0.5 + 0.5
            vis = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
            vis[..., 0] = to_uint8(norm_flow[..., 0])
            vis[..., 1] = to_uint8(norm_flow[..., 1])
            vis[..., 2] = 128
            return vis

        Image.fromarray(flow_to_rgb(v_surf)).save(os.path.join(out_dir, f"{prefix}_motion_surface.png"))
        Image.fromarray(flow_to_rgb(v_spec)).save(os.path.join(out_dir, f"{prefix}_motion_specular.png"))

        print(f"Extracted all 16 channels to {out_dir} with prefix {prefix}")

def main():
    parser = argparse.ArgumentParser(description="Convert Pathways raw binary tensors to image maps.")
    parser.add_argument("input", help="Path to .bin tensor file")
    parser.add_argument("--out-dir", "-o", default="output/debug_view", help="Output directory")
    args = parser.parse_args()

    hdr, tensor = load_tensor(args.input)
    print(f"Loaded {args.input}: {hdr['width']}x{hdr['height']}x{hdr['channels']} (dtype={hdr['data_type'].__name__}, SPP={hdr['spp']})")
    save_image_channels(hdr, tensor, args.out_dir)

if __name__ == "__main__":
    main()
