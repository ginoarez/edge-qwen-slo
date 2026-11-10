#!/usr/bin/env python3
"""Systematic benchmark matrix for 8GB GPUs."""

import argparse
import json
import statistics
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import yaml


def run_once_ollama(host, model, prompt, num_predict):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": num_predict, "temperature": 0.0},
    }
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
    return {"toks": round(toks, 2), "wall_s": round(wall, 3), "ttft_s": round(ttft, 3), "eval_count": eval_count}


def summarize_runs(runs):
    toks = [r["toks"] for r in runs]
    ttfts = [r["ttft_s"] for r in runs]
    return {
        "toks_median": round(statistics.median(toks), 2),
        "toks_min": round(min(toks), 2),
        "toks_max": round(max(toks), 2),
        "ttft_median_s": round(statistics.median(ttfts), 3),
        "slo_pass": statistics.median(toks) >= 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="matrix.yaml")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    runs_per = cfg.get("runs", 5)
    num_predict = cfg.get("num_predict", 256)
    prompts = cfg.get("prompts", [])
    results = []

    for model_cfg in cfg.get("models", []):
        name = model_cfg["name"]
        for prompt in prompts:
            runs = [run_once_ollama(args.host, name, prompt, num_predict) for _ in range(runs_per)]
            agg = summarize_runs(runs)
            results.append(
                {
                    "model": name,
                    "quant": model_cfg.get("quant"),
                    "backend": model_cfg.get("backend"),
                    "speculative": model_cfg.get("speculative"),
                    "prompt": prompt[:80],
                    "runs": runs_per,
                    **agg,
                    "details": runs,
                }
            )
            print(f"{name} | {agg['toks_median']} tok/s | slo_pass={agg['slo_pass']}")

    out_path = args.out or f"results/matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
