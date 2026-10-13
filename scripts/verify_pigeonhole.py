#!/usr/bin/env python3
"""
Exhaustive mathematical verification:
Can ANY 2D array `shared vec4 s[10][N]` have < 4-way bank conflict on Wave32?
"""

def test_all_strides():
    print("Testing strides N from 10 to 64 for `shared vec4 s[10][N]`:")
    min_conflict = 999
    best_strides = []
    
    for N in range(10, 65):
        # 32 threads: tx in 0..7, ty in 0..3
        # element = ty * N + tx
        # dwords = element * 4 = 4 * (ty * N + tx)
        # bank = (4 * (ty * N + tx)) % 32
        banks = {}
        for tid in range(32):
            tx = tid % 8
            ty = tid // 8
            word_offset = (ty * N + tx) * 4
            bank = word_offset % 32
            banks.setdefault(bank, []).append((tx, ty, word_offset))
        max_conflict = max(len(threads) for threads in banks.values())
        if max_conflict < min_conflict:
            min_conflict = max_conflict
            best_strides = [N]
        elif max_conflict == min_conflict:
            best_strides.append(N)
            
    print(f"Minimum conflict across ALL strides: {min_conflict}-way")
    print(f"Strides achieving this minimum: {best_strides[:10]}... (all of them!)")

if __name__ == "__main__":
    test_all_strides()
