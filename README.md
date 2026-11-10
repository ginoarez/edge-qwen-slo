# edge-qwen-slo

![ci](https://img.shields.io/github/actions/workflow/status/your-user/edge-qwen-slo/ci.yml?branch=main)
![license](https://img.shields.io/github/license/your-user/edge-qwen-slo)
![python](https://img.shields.io/badge/python-3.11%2B-blue)

Servicio local de LLM optimizado para una **RTX 5060 Ti de 8 GB VRAM** con un SLO explícito:

- **>= 100 tokens/s** en decode de un solo usuario
- **Modelo "bueno"**, no solo el más pequeño posible
- **Medición reproducible** (tok/s, TTFT, VRAM) y puerta de calidad

Este proyecto es intencionalmente distinto a *Soup* (que se centra en fine-tuning con streaming de capas). Aquí el foco es **inferencia de baja latencia en edge**: cuantización, speculative decoding, cache de contexto y observabilidad.

## Recomendación de modelo (8 GB)

| Objetivo | Modelo | Cuantización | Velocidad esperada | Notas |
|---|---|---:|---:|---|
| Simple, sin tuning | `qwen3:1.7b` | Q4_K_M | ~156 tok/s | Apache 2.0, thinking mode, 32K ctx |
| Mejor calidad, SLO ajustado | `qwen3:4b` | Q4_K_M | ~90 tok/s | Rivaliza con Qwen2.5-72B-Instruct |
| Mejor calidad **y** >=100 tok/s | `qwen3:4b` + draft DFlash/EAGLE-3 | Q3_K_M target + draft | 100-180 tok/s | Requiere llama.cpp con speculative decoding |
| MoE para agentes/herramientas | `LFM2.5-8B-A1B` | Q4_K_M | 84-175 tok/s | 8B total / 1.5B activos, tool calling |

**Conclusión práctica:** si quieres "un Qwen bueno" y mínimo 100 tok/s en 8 GB, la ruta recomendada es **Qwen3 4B con speculative decoding** (DFlash o EAGLE-3). Si no quieres complicarte, **Qwen3 1.7B** ya supera 100 tok/s directamente.

## Setup rápido (Ollama)

```powershell
ollama pull qwen3:1.7b
ollama run qwen3:1.7b
```

Benchmark:

```powershell
python benchmark.py --model qwen3:1.7b --runs 5
```

## Setup avanzado (llama.cpp + speculative decoding)

Descarga los GGUF (target + draft) y ejecuta:

```powershell
llama-server `
  -m Qwen3-4B-Q4_K_M.gguf `
  -md Qwen3-4B-DFlash.gguf `
  --spec-type draft-dflash `
  --spec-draft-n-max 15 `
  -fa on --jinja `
  -c 8192 --port 8080
```

Para EAGLE-3, cambia `--spec-type draft-eagle3` y usa el draft `AngelSlim/Qwen3-4B_eagle3`.

## SLO y puerta de calidad

- Decode >= 100 tok/s (mediana de 5 corridas, prompt de 512 tokens, generación de 256 tokens).
- TTFT <= 500 ms.
- VRAM pico <= 7.5 GB (dejar margen al sistema).
- Calidad: no caer más de 3 puntos frente a la variante Q4 en un set pequeño de tareas (MMLU subset, HumanEval subset, o tu propio eval).

## Gateway OpenAI-compatible + dashboard

Levanta el gateway (proxy a llama.cpp/Ollama con métricas Prometheus):

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- OpenAI-compatible: `POST /v1/chat/completions`
- Métricas Prometheus: `GET /metrics`
- Dashboard: `GET /dashboard`
- Stats JSON: `GET /api/stats`

Variables de entorno:

- `BACKEND_URL` (default `http://127.0.0.1:8080`)
- `EDGE_QWEN_CONFIG` (default `config.yaml`)

## Estructura

```
edge-qwen-slo/
  app/            # FastAPI gateway + métricas + VRAM sampler
  static/         # dashboard HTML
  tests/          # pytest
  docs/           # arquitectura
  results/        # benchmarks reproducibles
  benchmark.py
  config.yaml
  serve.ps1
  Dockerfile
```
