#!/usr/bin/env python3
"""Phase A experiment: ternary model baseline (tok/s + quality)."""

import argparse
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import yaml

from quality_eval import EVAL_SET, score


def run_once(host, model, prompt, num_predict):
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"num_predict": num_predict, "temperature": 0.0}}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        obj = json.loads(resp.read())
    wall = time.perf_counter() - t0
    eval_count = obj.get("eval_count", 0)
    eval_duration_ns = obj.get("eval_duration", 0)
    toks = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns else 0.0
    ttft = (obj.get("prompt_eval_duration", 0) or 0) / 1e9
    return {"toks": round(toks, 2), "wall_s": round(wall, 3), "ttft_s": round(ttft, 3), "response": obj.get("response", "")}


def run_model(host, model, runs, num_predict):
    prompt = "Explica en detalle qué es speculative decoding y cuándo acelera la inferencia local."
    perf_runs = [run_once(host, model, prompt, num_predict) for _ in range(runs)]
    toks = [r["toks"] for r in perf_runs]
    ttfts = [r["ttft_s"] for r in perf_runs]

    # quality
    q_scores = []
    for item in EVAL_SET:
        r = run_once(host, model, item["prompt"], 128)
        q_scores.append(score(r["response"], item["expected"]))

    import statistics

    return {
        "model": model,
        "runs": runs,
        "toks_median": round(statistics.median(toks), 2),
        "toks_min": round(min(toks), 2),
        "toks_max": round(max(toks), 2),
        "ttft_median_s": round(statistics.median(ttfts), 3),
        "quality_avg": round(sum(q_scores) / len(q_scores), 3),
        "quality_items": q_scores,
        "slo_pass": statistics.median(toks) >= 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--num-predict", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = [run_model(args.host, m, args.runs, args.num_predict) for m in args.models]
    for r in results:
        print(f"{r['model']}: {r['toks_median']} tok/s | quality={r['quality_avg']} | slo_pass={r['slo_pass']}")

    out = args.out or f"results/ternary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
