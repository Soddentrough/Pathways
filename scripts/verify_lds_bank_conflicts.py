#!/usr/bin/env python3
"""
Verification harness for LDS bank conflicts in Pass 0.
Tests 8x8 workgroups on 32-bank GPU shared memory (Wave32 / Warp32).
Compares stride 10 (unpadded), stride 12 (padded), stride 11, and stride 16.
"""

from collections import defaultdict

def analyze_lds_access(inner_stride_vec4, num_banks=32):
    """
    Simulates Wave 0 (32 threads, tx in 0..7, ty in 0..3) accessing
    s_normDepth[ty + dy][tx + dx] for dy in [0..2], dx in [0..2].
    Each element is vec4 (4 dwords).
    Returns max conflict degree per bank for:
      - 32-bit dword accesses (if scalarized)
      - 128-bit vector accesses (base dword bank)
    """
    words_per_element = 4
    results = {}
    
    # Test each 3x3 tap offset
    for dy in range(3):
        for dx in range(3):
            # For each lane in Wave 0 (tid in 0..31)
            # Lane tid maps to tx = tid % 8, ty = tid // 8
            # Base word offset:
            # ( (ty + dy) * inner_stride_vec4 + (tx + dx) ) * words_per_element
            
            # 1. Base bank for 128-bit vector access
            base_bank_counts = defaultdict(list)
            for tid in range(32):
                tx = tid % 8
                ty = tid // 8
                y = ty + dy
                x = tx + dx
                word_offset = (y * inner_stride_vec4 + x) * words_per_element
                bank = word_offset % num_banks
                base_bank_counts[bank].append((tid, tx, ty, word_offset))
            
            # 2. Dword-by-dword access across the 4 words of vec4
            # On GPU, a 128-bit load across 32 threads accesses 32 * 4 = 128 dwords.
            # In hardware, this is serviced in 4 phases (sub-instructions) or 4 cycles.
            # Phase p in [0..3] loads component p of vec4:
            phase_conflicts = []
            for p in range(words_per_element):
                bank_counts = defaultdict(list)
                for tid in range(32):
                    tx = tid % 8
                    ty = tid // 8
                    y = ty + dy
                    x = tx + dx
                    word_offset = (y * inner_stride_vec4 + x) * words_per_element + p
                    bank = word_offset % num_banks
                    bank_counts[bank].append((tid, tx, ty, word_offset))
                max_in_phase = max(len(v) for v in bank_counts.values())
                phase_conflicts.append(max_in_phase)
                
            results[(dy, dx)] = {
                "base_bank_max": max(len(v) for v in base_bank_counts.values()),
                "phase_conflicts": phase_conflicts,
                "base_bank_distribution": {k: len(v) for k, v in sorted(base_bank_counts.items())}
            }
            
    return results

def main():
    strides = [10, 11, 12, 13, 14, 15, 16]
    print("=" * 80)
    print("LDS BANK CONFLICT ANALYSIS (8x8 Workgroup, Wave32, 32 Banks, vec4 elements)")
    print("=" * 80)
    
    for stride in strides:
        res = analyze_lds_access(stride)
        # Check center tap (dy=1, dx=1)
        center = res[(1, 1)]
        print(f"\n--- Stride {stride} elements ({stride * 4} dwords per row) ---")
        print(f"Base bank max threads per bank (center tap): {center['base_bank_max']}-way")
        print(f"Phase conflicts per 32-bit component: {center['phase_conflicts']}")
        print(f"Active banks utilized for base bank: {len(center['base_bank_distribution'])} of 32")
        print(f"Bank distribution: {dict(list(center['base_bank_distribution'].items())[:8])}...")
        
        # Check all 9 taps
        all_max_base = max(r["base_bank_max"] for r in res.values())
        all_max_phase = max(max(r["phase_conflicts"]) for r in res.values())
        print(f"Across all 9 Sobel taps: Max Base = {all_max_base}-way, Max Phase = {all_max_phase}-way")

if __name__ == "__main__":
    main()
