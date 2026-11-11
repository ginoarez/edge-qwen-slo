#!/usr/bin/env python3
"""Phase C experiment: quality recovery via evolution strategies (Agentic ESOpt-inspired)."""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

from adaptation.es_adapter import EvolutionStrategyAdapter


def synthetic_eval(quality: float):
    """Return a scalar reward around current quality."""
    return quality + random.uniform(-0.05, 0.05)


def run_recovery(initial_quality: float, steps: int, population: int):
    adapter = EvolutionStrategyAdapter(population_size=population)
    quality = initial_quality
    history = [quality]
    for step in range(steps):
        delta, rewards = adapter.step(lambda: synthetic_eval(quality))
        quality = min(1.0, quality + delta)
        history.append(round(quality, 4))
    return {"initial": initial_quality, "final": round(quality, 4), "gain": round(quality - initial_quality, 4), "history": history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--initial-quality", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = run_recovery(args.initial_quality, args.steps, args.population)
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
