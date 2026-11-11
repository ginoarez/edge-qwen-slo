#!/usr/bin/env python3
"""Auto-tuner: pick best config from benchmark matrix results."""

import argparse
import json
from pathlib import Path


def pick_best(results):
    passing = [r for r in results if r.get("slo_pass")]
    pool = passing if passing else results
    # maximize toks_median, then quality if available
    return max(pool, key=lambda r: (r.get("toks_median", 0), r.get("quality_avg", 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    best = pick_best(data)
    print(json.dumps(best, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
