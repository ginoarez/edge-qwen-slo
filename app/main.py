import asyncio
import json
import os
import time
from collections import deque
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from .metrics import (
    REQUESTS,
    TOKENS,
    LATENCY,
    TTFT,
    TOKS_PER_SEC,
    SLO_PASS,
)
from .vram import vram_sampler

CONFIG_PATH = os.getenv("EDGE_QWEN_CONFIG", "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8080")
SLO_MIN_TOKS = CONFIG.get("slo", {}).get("min_decode_toks", 100)

RECENT = deque(maxlen=200)

app = FastAPI(title="edge-qwen-slo", version="0.2.0")

_stop = None
_sampler_thread = None


@app.on_event("startup")
def startup():
    global _stop, _sampler_thread
    import threading

    _stop = threading.Event()
    _sampler_thread = threading.Thread(target=vram_sampler, args=(_stop,), daemon=True)
    _sampler_thread.start()


@app.on_event("shutdown")
def shutdown():
    if _stop is not None:
        _stop.set()


@app.get("/health")
def health():
    return {"status": "ok", "backend": BACKEND_URL, "slo_min_toks": SLO_MIN_TOKS}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/stats")
def stats():
    if not RECENT:
        return {"count": 0, "median_toks": None, "slo_pass_rate": None, "last": None}
    toks = [r["toks"] for r in RECENT]
    import statistics

    pass_rate = sum(1 for r in RECENT if r["slo_pass"]) / len(RECENT)
    return {
        "count": len(RECENT),
        "median_toks": round(statistics.median(toks), 2),
        "min_toks": round(min(toks), 2),
        "max_toks": round(max(toks), 2),
        "slo_pass_rate": round(pass_rate, 3),
        "last": RECENT[-1],
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    html_path = Path(__file__).resolve().parent.parent / "static" / "dashboard.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


def _record(model, stream, status, elapsed, ttft, toks, completion_tokens):
    slo_pass = toks >= SLO_MIN_TOKS if toks else False
    REQUESTS.labels(model=model, stream=str(stream).lower(), status=status).inc()
    LATENCY.labels(model=model, stream=str(stream).lower()).observe(elapsed)
    if ttft is not None:
        TTFT.labels(model=model).observe(ttft)
    if toks is not None:
        TOKS_PER_SEC.labels(model=model).observe(toks)
        TOKENS.labels(model=model).inc(completion_tokens or 0)
        SLO_PASS.set(1 if slo_pass else 0)
    RECENT.append(
        {
            "ts": time.time(),
            "model": model,
            "stream": stream,
            "status": status,
            "elapsed_s": round(elapsed, 3),
            "ttft_s": round(ttft, 3) if ttft is not None else None,
            "toks": round(toks, 2) if toks is not None else None,
            "completion_tokens": completion_tokens,
            "slo_pass": slo_pass,
        }
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "qwen3:4b")
    stream = bool(body.get("stream", False))

    t0 = time.perf_counter()
    ttft = None

    if stream:
        async def gen():
            nonlocal ttft
            first = True
            completion_tokens = 0
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream(
                    "POST", f"{BACKEND_URL}/v1/chat/completions", json=body
                ) as resp:
                    async for line in resp.aiter_lines():
                        if first and line.strip():
                            ttft = time.perf_counter() - t0
                            first = False
                        if line.startswith("data: ") and line.strip() != "data: [DONE]":
                            try:
                                obj = json.loads(line[6:])
                                delta = obj.get("choices", [{}])[0].get("delta", {})
                                if delta.get("content"):
                                    completion_tokens += 1
                            except Exception:
                                pass
                        yield line + "\n"
            elapsed = time.perf_counter() - t0
            toks = completion_tokens / elapsed if elapsed > 0 else 0.0
            _record(model, True, "ok", elapsed, ttft, toks, completion_tokens)

        return StreamingResponse(gen(), media_type="text/event-stream")

    # non-stream
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(f"{BACKEND_URL}/v1/chat/completions", json=body)
        elapsed = time.perf_counter() - t0
        data = resp.json()
        usage = data.get("usage", {}) or {}
        completion_tokens = usage.get("completion_tokens", 0)
        toks = completion_tokens / elapsed if elapsed > 0 and completion_tokens else None
        if toks is None:
            # fallback: try backend-specific fields
            toks = 0.0
        _record(model, False, str(resp.status_code), elapsed, None, toks, completion_tokens)
        return JSONResponse(content=data, status_code=resp.status_code)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        _record(model, False, "error", elapsed, None, 0.0, 0)
        return JSONResponse({"error": str(e)}, status_code=502)
