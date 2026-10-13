#!/usr/bin/env python3
"""
Find true conflict-free LDS layouts for 8x8 workgroup Wave32:
Tests:
1. SoA scalar floats: s[10][N]
2. Transposed layouts
3. Stride with different thread mapping
"""

from collections import defaultdict

def find_zero_conflict_soa():
    print("Searching for zero-conflict stride in SoA float s[10][N]:")
    for N in range(10, 65):
        max_c = 0
        for dy in range(3):
            for dx in range(3):
                bank_counts = defaultdict(int)
                for tid in range(32):
                    tx = tid % 8
                    ty = tid // 8
                    offset = (ty + dy) * N + (tx + dx)
                    bank = offset % 32
                    bank_counts[bank] += 1
                c = max(bank_counts.values())
                if c > max_c:
                    max_c = c
        if max_c == 1:
            print(f"Zero conflict found! Stride N = {N}")
            return N
    print(f"Minimum across all strides was: {min(max_c for N in range(10, 65))}")

if __name__ == "__main__":
    find_zero_conflict_soa()
