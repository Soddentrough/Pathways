#!/usr/bin/env python3
import struct
import base64
import json
import os
import math
import zlib

def create_textures():
    os.makedirs("scenes", exist_ok=True)
    width, height = 64, 64

    # 1. Checkerboard RGBA
    raw_data = bytearray()
    for y in range(height):
        raw_data.append(0) # Filter type 0 (None)
        for x in range(width):
            check = ((x // 8) + (y // 8)) % 2
            if check == 0:
                raw_data.extend([220, 50, 50, 255]) # Red check
            else:
                raw_data.extend([240, 240, 240, 255]) # White check

    def make_png(w, h, raw):
        def chunk(tag, data):
            c = tag + data
            crc = zlib.crc32(c) & 0xffffffff
            return struct.pack(">I", len(data)) + c + struct.pack(">I", crc)

        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
        compressed = zlib.compress(raw, 9)
        png += chunk(b"IDAT", compressed)
        png += chunk(b"IEND", b"")
        return png

    checker_png = make_png(width, height, raw_data)
    with open("scenes/checker.png", "wb") as f:
        f.write(checker_png)
    print("Created scenes/checker.png")

    # 2. Normal map (perturbed ripples)
    normal_data = bytearray()
    for y in range(height):
        normal_data.append(0)
        for x in range(width):
            fx = (x / width) * 4.0 * math.pi
            fy = (y / height) * 4.0 * math.pi
            dx = math.sin(fx) * 0.4
            dy = math.cos(fy) * 0.4
            # Normal vector: normalize(-dx, -dy, 1.0)
            len_n = math.sqrt(dx*dx + dy*dy + 1.0)
            nx = -dx / len_n
            ny = -dy / len_n
            nz = 1.0 / len_n
            r = int((nx * 0.5 + 0.5) * 255)
            g = int((ny * 0.5 + 0.5) * 255)
            b = int((nz * 0.5 + 0.5) * 255)
            normal_data.extend([r, g, b, 255])

    normal_png = make_png(width, height, normal_data)
    with open("scenes/ripples_normal.png", "wb") as f:
        f.write(normal_png)
    print("Created scenes/ripples_normal.png")

if __name__ == "__main__":
    create_textures()
