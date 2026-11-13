# Benchmark results

Llena esta tabla con mediciones reproducibles (mediana de 5 corridas).

| GPU | Modelo | Cuantización | tok/s | TTFT (ms) | VRAM pico (GB) | Notas |
|---|---|---|---:|---:|---:|---|
| Example 8GB GPU | Qwen3 1.7B | Q4_K_M | 156 | - | - | Directo |
| Example 8GB GPU | Qwen3 4B | Q4_K_M | 90 | - | - | Baseline |
| Example 8GB GPU | Qwen3 4B + DFlash | Q3_K_M | TBD | - | - | Objetivo >=100 |
| Example 8GB GPU | DeepSeek-V4-Pro-Qwen3.5-4B | Q4_K_M | 105.28 | 0.047 | - | SLO pass (num_ctx=4096) |
| Example 8GB GPU | DeepSeek-V4-Pro-Qwen3.5-9B | Q4_K_M | 67.46 | 0.063 | - | SLO fail |
| Example 8GB GPU | DeepSeek-V4-Pro-Qwen3.5-9B | Q3_K_M | 77.93 | 0.049 | - | SLO fail (best 9B config; thinking level does not affect tok/s) |
