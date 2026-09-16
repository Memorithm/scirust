#!/usr/bin/env python3
"""Thor-only compatibility patch for Dream's rank-1 RoPE outer product.

Dream expresses the RoPE frequency outer product as a batched matrix multiply
whose contracted dimension is exactly one. On the pinned NVIDIA Thor / CUDA
13.0 / PyTorch 2.10.0+cu130 qualification stack, that call currently fails in
cuBLAS with CUBLAS_STATUS_INVALID_VALUE before any cache-policy measurement is
produced. For tensors shaped [B, D, 1] and [B, 1, S], broadcast multiplication
computes the same outer product with no reduction dimension and avoids routing
this operation through GEMM.

This patch is deliberately separate from patch_elastic_cache.py so historical
SciRust cache-policy overlays remain byte-for-byte reproducible. It is applied
only by the Thor qualification workflow and fails closed if the pinned upstream
source changes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

OLD = """            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)\n"""
NEW = """            # Thor compatibility: K=1 batched matmul is exactly this broadcast outer product.\n            freqs = (inv_freq_expanded.float() * position_ids_expanded.float()).transpose(1, 2)\n"""


def patch(root: Path) -> Path:
    target = root / "dream" / "model" / "modeling_dream.py"
    text = target.read_text(encoding="utf-8")
    count = text.count(OLD)
    if count != 1:
        raise RuntimeError(f"Thor RoPE compatibility marker: expected 1 exact match, found {count}")
    if "Thor compatibility: K=1 batched matmul" in text:
        raise RuntimeError("Thor RoPE compatibility patch is already present")
    target.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("elastic_cache_root", type=Path)
    args = parser.parse_args()
    target = patch(args.elastic_cache_root)
    print(f"patched {target}")


if __name__ == "__main__":
    main()
