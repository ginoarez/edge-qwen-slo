#!/usr/bin/env python3
"""Benchmark reproducible de tok/s para Ollama (RTX 5060 Ti 8GB)."""

import argparse
import json
import statistics
import time
import urllib.request


def run_once(model: str, prompt: str, num_predict: int, host: str) -> dict:
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
    prompt_eval_duration_ns = obj.get("prompt_eval_duration", 0)
    ttft = prompt_eval_duration_ns / 1e9 if prompt_eval_duration_ns else None

    return {
        "model": model,
        "eval_count": eval_count,
        "toks": round(toks, 2),
        "wall_s": round(wall, 3),
        "ttft_s": round(ttft, 3) if ttft is not None else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--num-predict", type=int, default=256)
    ap.add_argument("--prompt", default="Explica en detalle qué es speculative decoding y cuándo acelera la inferencia local.")
    args = ap.parse_args()

    results = [run_once(args.model, args.prompt, args.num_predict, args.host) for _ in range(args.runs)]
    summary = summarize(args.model, args.runs, results)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def summarize(model: str, runs: int, results: list) -> dict:
    toks = [r["toks"] for r in results]
    ttfts = [r["ttft_s"] for r in results if r["ttft_s"] is not None]
    return {
        "model": model,
        "runs": runs,
        "toks_median": round(statistics.median(toks), 2),
        "toks_min": round(min(toks), 2),
        "toks_max": round(max(toks), 2),
        "ttft_median_s": round(statistics.median(ttfts), 3) if ttfts else None,
        "slo_pass": statistics.median(toks) >= 100,
        "details": results,
    }


if __name__ == "__main__":
    main()
