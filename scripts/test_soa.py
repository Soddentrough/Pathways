#!/usr/bin/env python3
"""
Test SoA (Structure of Arrays) vs AoS for LDS access:
If scalar float arrays are used: s_norm[10][N]
Thread (tx, ty) accesses s_norm[ty + dy][tx + dx]
Word offset = (ty + dy) * N + (tx + dx)
Bank = word_offset % 32
"""

from collections import defaultdict

def test_soa():
    print("Testing SoA scalar float array `shared float s[10][N]`:")
    for N in [10, 11, 12, 13, 16]:
        base_bank_counts = defaultdict(list)
        for tid in range(32):
            tx = tid % 8
            ty = tid // 8
            word_offset = ty * N + tx
            bank = word_offset % 32
            base_bank_counts[bank].append((tx, ty))
        max_conflict = max(len(v) for v in base_bank_counts.values())
        num_banks = len(base_bank_counts)
        print(f"Stride N = {N}: Max conflict = {max_conflict}-way, Active banks = {num_banks}/32")

if __name__ == "__main__":
    test_soa()
