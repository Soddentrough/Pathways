#!/usr/bin/env python3
"""
Test LDS bank access for s_albedo[10][12] (vec3).
In GLSL shared memory:
- If vec3 is aligned to 16 bytes (4 dwords) like std140/std430: element size = 4 dwords.
- If vec3 is packed as 12 bytes (3 dwords): element size = 3 dwords.
"""

from collections import defaultdict

def test_albedo(words_per_element):
    inner_stride = 12
    base_bank_counts = defaultdict(list)
    for tid in range(32):
        tx = tid % 8
        ty = tid // 8
        word_offset = (ty * inner_stride + tx) * words_per_element
        bank = word_offset % 32
        base_bank_counts[bank].append((tx, ty, word_offset))
    max_conflict = max(len(v) for v in base_bank_counts.values())
    print(f"words_per_element = {words_per_element}: Max base conflict = {max_conflict}-way, Active banks = {len(base_bank_counts)}/32")

if __name__ == "__main__":
    print("Alignment 16B (4 words):")
    test_albedo(4)
    print("Packed 12B (3 words):")
    test_albedo(3)
