# Architecture

```text
Client (OpenAI-compatible)
        |
        v
  llama-server (target Qwen3 4B + draft DFlash/EAGLE-3)
        |
        v
  benchmark.py / metrics exporter
        |
        v
  SLO gate (tok/s, TTFT, VRAM, quality)
```

## Components

- **Target model**: Qwen3 4B Q4_K_M (o Q3_K_M para SLO).
- **Draft model**: DFlash/EAGLE-3 para speculative decoding.
- **Runtime**: llama.cpp con CUDA.
- **Metrics**: tok/s, TTFT, VRAM pico.
- **Quality gate**: comparación contra baseline Q4 en tareas pequeñas.
```

Create results/README.md with benchmark table template.<|end|><|start|>Assistant to=bash<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name="command">
mkdir -p "C:\Users\user\Documents\Default Project\edge-qwen-slo\results" && cat > "C:\Users\user\Documents\Default Project\edge-qwen-slo\results\README.md" <<'EOF'
# Benchmark results

Llena esta tabla con mediciones reproducibles (mediana de 5 corridas).

| GPU | Modelo | Cuantización | tok/s | TTFT (ms) | VRAM pico (GB) | Notas |
|---|---|---|---:|---:|---:|---|
| RTX 5060 Ti 8GB | Qwen3 1.7B | Q4_K_M | 156 | - | - | Directo |
| RTX 5060 Ti 8GB | Qwen3 4B | Q4_K_M | 90 | - | - | Baseline |
| RTX 5060 Ti 8GB | Qwen3 4B + DFlash | Q3_K_M | TBD | - | - | Objetivo ≥100 |
EOF