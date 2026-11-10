# edge-qwen-slo

![ci](https://img.shields.io/github/actions/workflow/status/ginoarez/edge-qwen-slo/ci.yml?branch=main)
![license](https://img.shields.io/github/license/ginoarez/edge-qwen-slo)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![gpu](https://img.shields.io/badge/GPU-RTX%205060%20Ti%208GB-76B900)

**Hardware-agnostic LLM edge lab.**  
Goal: find the best (model + quantization + runtime + speculative decoding + adaptation) for **any** VRAM budget, from 2 GB to 24 GB+, with reproducible benchmarks and quality gates.

Inspired by **Agentic ESOpt** (gradient-free, inference-level-memory adaptation), this project also explores recovering quality after aggressive quantization.

> This project is intentionally different from training-focused toolkits (e.g. Soup).  
> Focus here is **low-latency edge inference**: quantization, speculative decoding, KV-cache discipline, and observability.

---

## 1. Hardware & SLO target

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 5060 Ti 8 GB (GB206, 448 GB/s) |
| Usable VRAM | ~8 GB (model + KV cache + runtime overhead) |
| SLO (decode) | **≥ 100 tok/s** (median of 5 runs) |
| SLO (TTFT) | **≤ 500 ms** |
| SLO (VRAM peak) | **≤ 7.5 GB** |
| SLO (quality) | ≤ 3-point drop vs Q4 baseline on a small eval set |

---

## 2. Model matrix (8 GB)

Throughput numbers are measured with Ollama Q4_K_M on RTX 5060-class 8 GB GPUs (see `results/README.md`).

| Model | Quant | Size | tok/s | Role |
|---|---|---:|---:|---|
| `qwen3:0.6b` | Q4_K_M | 0.5 GB | ~211 | Draft / ultra-low-latency |
| `qwen3:1.7b` | Q4_K_M | 1.4 GB | ~156 | Simple path, meets SLO directly |
| `qwen3:4b` | Q4_K_M | 2.5 GB | ~90 | Quality baseline |
| `qwen3:4b` + DFlash | Q3_K_M target | ~2.0 GB | 100–180 (target) | **Recommended** quality + speed |
| `qwen3:4b` + EAGLE-3 | Q3_K_M target | ~2.0 GB | 100–180 (target) | Alternative speculative path |
| `LFM2.5-8B-A1B` | Q4_K_M | 5.8 GB | 84–175 | MoE, tool-calling / agents |

**Recommendation:** use **Qwen3 4B + speculative decoding (DFlash or EAGLE-3)** for the best quality/speed trade-off.  
If you want zero tuning, **Qwen3 1.7B** already exceeds 100 tok/s.

---

## 3. Architecture

```text
Client (OpenAI-compatible)
        |
        v
  FastAPI gateway (app/main.py)
        |  - /v1/chat/completions (stream + non-stream)
        |  - /metrics (Prometheus)
        |  - /dashboard (Chart.js)
        |  - /api/stats (JSON aggregates)
        v
  llama.cpp server (CUDA)
        |  - target: Qwen3 4B (Q3_K_M / Q4_K_M)
        |  - draft:  DFlash or EAGLE-3
        |  - -fa on, --jinja, speculative decoding
        v
  nvidia-smi VRAM sampler (background thread)
```

Key techniques:

- **Speculative decoding** (`draft-dflash`, `draft-eagle3`) to push 4B models past 100 tok/s.
- **KV-cache discipline**: cap context (`-c 8192`) to keep VRAM under 7.5 GB.
- **Flash attention** (`-fa on`) for prompt processing.
- **Observability**: per-request tok/s, TTFT, latency, VRAM, SLO pass/fail.
- **Quality gate**: compare candidate quantization against Q4 baseline.

---

## 4. Quick start

### 4.1 Simple path (Ollama, no tuning)

```powershell
ollama pull qwen3:1.7b
ollama run qwen3:1.7b
python benchmark.py --model qwen3:1.7b --runs 5
```

### 4.2 Advanced path (llama.cpp + speculative decoding)

Download the target and draft GGUFs, then:

```powershell
# DFlash
.\serve.ps1
```

or manually:

```powershell
llama-server `
  -m Qwen3-4B-Q4_K_M.gguf `
  -md Qwen3-4B-DFlash.gguf `
  --spec-type draft-dflash `
  --spec-draft-n-max 15 `
  -fa on --jinja `
  -c 8192 --port 8080
```

EAGLE-3 variant:

```powershell
llama-server `
  -m Qwen3-4B-Q4_K_M.gguf `
  -md Qwen3-4B-EAGLE3.gguf `
  --spec-type draft-eagle3 `
  -fa on --jinja `
  -c 8192 --port 8080
```

### 4.3 Gateway (OpenAI-compatible)

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- Chat completions: `POST /v1/chat/completions`
- Prometheus metrics: `GET /metrics`
- Dashboard: `GET /dashboard`
- Aggregated stats: `GET /api/stats`
- Health: `GET /health`

Environment variables:

| Var | Default | Description |
|---|---|---|
| `BACKEND_URL` | `http://127.0.0.1:8080` | llama.cpp / Ollama base URL |
| `EDGE_QWEN_CONFIG` | `config.yaml` | Path to SLO config |

---

## 5. Benchmarking

Single config:

```powershell
python benchmark.py --model qwen3:1.7b --runs 5 --num-predict 256
```

Systematic matrix:

```powershell
python benchmark_matrix.py --config matrix.yaml --runs 5
```

Quality gate:

```powershell
python quality_eval.py --model qwen3:4b --baseline-score 0.85
```

Output (JSON):

```json
{
  "model": "qwen3:1.7b",
  "runs": 5,
  "toks_median": 156.39,
  "toks_min": 150.12,
  "toks_max": 162.44,
  "ttft_median_s": 0.18,
  "slo_pass": true
}
```

Results are appended to `results/README.md` for reproducibility.

---

## 6. Metrics & SLO gating

Prometheus metrics exposed at `/metrics`:

| Metric | Type | Description |
|---|---|---|
| `edge_qwen_requests_total` | Counter | Requests by model/stream/status |
| `edge_qwen_tokens_total` | Counter | Generated tokens |
| `edge_qwen_request_duration_seconds` | Histogram | End-to-end latency |
| `edge_qwen_ttft_seconds` | Histogram | Time to first token |
| `edge_qwen_decode_toks_per_sec` | Histogram | Per-request decode throughput |
| `edge_qwen_vram_used_gb` | Gauge | GPU VRAM used |
| `edge_qwen_slo_pass` | Gauge | 1 if last request met ≥100 tok/s |

The dashboard (`/dashboard`) plots live tok/s and shows SLO pass rate.

---

## 7. Quality gate

Before shipping a new quantization or speculative config:

1. Run `benchmark.py` (5 runs) → median tok/s must be ≥ 100.
2. Record TTFT ≤ 500 ms and VRAM peak ≤ 7.5 GB.
3. Compare quality against Q4 baseline on a small task set (MMLU subset, HumanEval subset, or custom). Max allowed drop: 3 points.
4. Update `results/README.md`.

---

## 8. Project structure

```
edge-qwen-slo/
├── .github/workflows/ci.yml
├── app/                          # FastAPI gateway + metrics
├── hardware/                     # Hardware profiler
├── models/                       # Model catalog
├── adaptation/                   # ES adaptation (Agentic ESOpt inspired)
├── recommender/                  # Config recommender by VRAM
├── static/                       # Dashboard
├── tests/
├── docs/
├── results/
├── benchmark.py
├── benchmark_matrix.py
├── quality_eval.py
├── matrix.yaml
├── config.yaml
├── conftest.py
├── pyproject.toml
├── serve.ps1
├── Dockerfile
├── requirements.txt
├── LICENSE
├── CONTRIBUTING.md
└── README.md
```

---

## 9. Testing

```powershell
pip install -r requirements.txt
pytest -q
```

CI runs on every push/PR to `main`.

---

## 10. Roadmap

- [ ] vLLM backend adapter
- [ ] Automated quality eval harness (MMLU/HumanEval subsets)
- [ ] Grafana dashboard JSON export
- [ ] Multi-GPU / tensor-parallel support
- [ ] Speculative decoding auto-tuning (draft length, acceptance rate)

---

## License

MIT — see [LICENSE](LICENSE).
