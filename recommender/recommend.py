"""Recommend a model/config given a VRAM budget."""

import os

import yaml


def load_catalog(path="models/catalog.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["models"]


def recommend(vram_gb: float, objective: str = "quality"):
    catalog = load_catalog()
    candidates = [m for m in catalog if m["min_vram_gb"] <= vram_gb]
    if not candidates:
        return None
    if objective == "speed":
        # prefer fewer active params
        candidates.sort(key=lambda m: (m["active_b"], -m["quality_score"]))
    else:
        candidates.sort(key=lambda m: (-m["quality_score"], m["active_b"]))
    best = candidates[0]
    speculative = "dflash" if best["min_vram_gb"] >= 4 else "none"
    return {
        "model": best["id"],
        "quant": best["quant"],
        "min_vram_gb": best["min_vram_gb"],
        "active_b": best["active_b"],
        "quality_score": best["quality_score"],
        "speculative": speculative,
        "objective": objective,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--vram", type=float, required=True)
    ap.add_argument("--objective", default="quality", choices=["quality", "speed"])
    args = ap.parse_args()
    print(recommend(args.vram, args.objective))
