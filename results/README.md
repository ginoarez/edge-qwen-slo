# Benchmark results

Llena esta tabla con mediciones reproducibles (mediana de 5 corridas).

| GPU | Modelo | Cuantización | tok/s | TTFT (ms) | VRAM pico (GB) | Notas |
|---|---|---|---:|---:|---:|---|
| RTX 5060 Ti 8GB | Qwen3 1.7B | Q4_K_M | 156 | - | - | Directo |
| RTX 5060 Ti 8GB | Qwen3 4B | Q4_K_M | 90 | - | - | Baseline |
| RTX 5060 Ti 8GB | Qwen3 4B + DFlash | Q3_K_M | TBD | - | - | Objetivo >=100 |
