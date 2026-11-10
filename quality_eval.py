#!/usr/bin/env python3
"""Small quality gate for 8GB LLM configs."""

import argparse
import json
import urllib.request


EVAL_SET = [
    {
        "prompt": "Responde solo con un número: ¿Cuánto es 17*23?",
        "expected": "391",
    },
    {
        "prompt": "Escribe una función Python llamada is_even que devuelva True si un número es par.",
        "expected": "def is_even",
    },
    {
        "prompt": "Explica en una frase qué es speculative decoding.",
        "expected": "draft",
    },
]


def call_ollama(host, model, prompt):
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.0, "num_predict": 128}}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        obj = json.loads(resp.read())
    return obj.get("response", "")


def score(response, expected):
    return 1.0 if expected.lower() in response.lower() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--model", required=True)
    ap.add_argument("--baseline-score", type=float, default=None)
    args = ap.parse_args()

    scores = []
    for item in EVAL_SET:
        resp = call_ollama(args.host, args.model, item["prompt"])
        s = score(resp, item["expected"])
        scores.append({"prompt": item["prompt"], "expected": item["expected"], "score": s, "response": resp[:200]})
        print(f"score={s} expected={item['expected']}")

    avg = sum(x["score"] for x in scores) / len(scores)
    result = {"model": args.model, "avg_score": round(avg, 3), "max_drop_allowed": 3.0, "items": scores}
    if args.baseline_score is not None:
        drop_points = (args.baseline_score - avg) * 100
        result["drop_points"] = round(drop_points, 2)
        result["quality_pass"] = drop_points <= 3.0
        print(f"avg={avg:.3f} baseline={args.baseline_score:.3f} drop_points={drop_points:.2f} pass={result['quality_pass']}")
    else:
        print(f"avg={avg:.3f}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
