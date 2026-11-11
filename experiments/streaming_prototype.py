#!/usr/bin/env python3
"""Phase E prototype: MoE expert streaming from disk with LRU cache."""

import argparse
import json
import os
import random
import time
from pathlib import Path

from streaming.offload import ExpertLRUCache


def ensure_experts(expert_dir: Path, num_experts: int, expert_size_kb: int):
    expert_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_experts):
        p = expert_dir / f"expert_{i}.bin"
        if not p.exists():
            p.write_bytes(os.urandom(expert_size_kb * 1024))


def simulate(expert_dir: Path, num_experts: int, top_k: int, num_tokens: int, cache_bytes: int):
    cache = ExpertLRUCache(cache_bytes)
    io_time = 0.0
    compute_time = 0.0
    hits = 0
    misses = 0

    for _ in range(num_tokens):
        selected = random.sample(range(num_experts), top_k)
        for eid in selected:
            t0 = time.perf_counter()
            data = cache.get(eid)
            if data is None:
                # stream from disk
                p = expert_dir / f"expert_{eid}.bin"
                data = p.read_bytes()
                size = len(data)
                cache.put(eid, data, size)
                misses += 1
            else:
                hits += 1
            io_time += time.perf_counter() - t0

            # simulate compute
            tc = time.perf_counter()
            _ = sum(data[:64])  # tiny compute
            compute_time += time.perf_counter() - tc

    total = io_time + compute_time
    return {
        "num_tokens": num_tokens,
        "top_k": top_k,
        "num_experts": num_experts,
        "cache_capacity_bytes": cache_bytes,
        "cache_entries": cache.stats()["entries"],
        "cache_hits": hits,
        "cache_misses": misses,
        "hit_rate": round(hits / (hits + misses), 4) if (hits + misses) else 0.0,
        "io_time_s": round(io_time, 4),
        "compute_time_s": round(compute_time, 4),
        "io_share": round(io_time / total, 4) if total else 0.0,
        "simulated_tok_s": round(num_tokens / total, 2) if total else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert-dir", default="cache/experts")
    ap.add_argument("--num-experts", type=int, default=64)
    ap.add_argument("--expert-size-kb", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--num-tokens", type=int, default=100)
    ap.add_argument("--cache-mb", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    expert_dir = Path(args.expert_dir)
    ensure_experts(expert_dir, args.num_experts, args.expert_size_kb)

    res = simulate(
        expert_dir,
        args.num_experts,
        args.top_k,
        args.num_tokens,
        args.cache_mb * 1024 * 1024,
    )
    print(json.dumps(res, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
