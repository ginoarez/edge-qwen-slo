# Deep Research: Achieving Large-Model Quality in <4 GB VRAM

## 1. Objective

Enable a **large-model experience** (quality comparable to 7B–27B dense models) on **<4 GB VRAM** with acceptable latency, across arbitrary hardware.

This is **not** about matching raw tok/s of a 5090. It is about **maximizing useful intelligence per byte**.

---

## 2. Key Findings

### 2.1 Ternary weights (1.58-bit) are the biggest lever

- **BitNet b1.58** (Microsoft): ternary {-1,0,+1} weights, ~1.58 bits/param. Matches FP16 quality at 3B+ parameters. 3B model ~3 GB VRAM, logic 78 / coding 70 / creative 65.
- **Ternary Bonsai 2** (PrismML): ternary requantization of Qwen3.5, retains **98.2%** of 27B accuracy at 5.9 GB. Family: 2B (0.43 GB), 4B (0.95 GB), 9B (2 GB), 27B (5.9 GB).
- Memory reduction: **7–9x vs FP16**, with near-parity quality.

**Implication**: a 27B-class model can fit in 6 GB; a 9B-class model fits in 2 GB. This is the foundation for <4 GB.

### 2.2 Hybrid attention (Gated DeltaNet) cuts KV cache

- Full attention KV cache grows with context and dominates serving memory.
- **Gated DeltaNet** uses a fixed-size recurrent state instead of growing KV cache.
- Hybrid ratio (e.g., 3 GDN : 1 full attention) keeps quality while drastically reducing memory.
- Bonsai 2 uses this: 262K context with minimal KV growth.

**Implication**: long context becomes feasible in <4 GB.

### 2.3 MoE with few active parameters

- Models like LFM2.5-8B-A1B use 8B total / 1.5B active.
- Active params determine compute and (partially) latency; total params determine quality.

**Implication**: high quality at low active compute, but memory still holds all params unless quantized.

### 2.4 Speculative decoding

- Draft small model (or self-speculative n-gram) predicts tokens; target verifies in batch.
- On 8 GB: Qwen3 4B base ~90 tok/s → 100–180 tok/s with DFlash/EAGLE-3.

**Implication**: helps meet latency SLO but does not reduce memory.

### 2.5 Evolution Strategies (Agentic ESOpt)

- Full-parameter updates using **inference-level memory only** (no backprop, no optimizer states).
- Scalar trajectory rewards drive updates.
- Composes with prompt evolution and test-time compute.

**Implication**: ideal for **recovering quality** after aggressive quantization on small hardware.

### 2.6 Offloading

- CPU RAM can hold overflow, but latency degrades sharply.
- Useful for models slightly above VRAM budget, not for interactive SLO.

### 2.8 KV cache quantization (KIVI / MiniKV)

- KIVI: 2-bit per-channel key + per-token value quantization.
- MiniKV: 2-bit layer-wise KV compression.
- Can reduce KV memory by 8x with minimal quality loss on long context.

**Implication**: enables longer contexts in <4 GB.

### 2.9 Prompt / context compression (LLMLingua)

- Drop low-information tokens using a small model.
- Up to 20x compression on reasoning tasks; 4-10x typical.
- Reduces prompt processing time and KV cache size.

**Implication**: cheaper long-context inference without model changes.

### 2.7 Disk streaming & MoE expert offloading (kimi-k3-in-c)

From `kimi-k3-in-c`: a 2.78T-param model runs in **8.24 GB RAM** by never keeping the 1.45 TB of routed experts resident.

Key techniques:
- **MoE with expert streaming**: only active experts are loaded; rest streamed from disk on demand.
- **MXFP4 quantization**: experts stored at ~0.5 byte/param.
- **KDA (Kimi Delta Attention)**: recurrent state with fixed size (memory never grows with context).
- **MLA (Multi-Head Latent Attention)**: one latent instead of 96 heads.
- **Trunk packing**: dense layers packed into one file, streamed with LRU cache.
- **Ultra-low-memory mode**: stream exact embedding rows, reuse one state slot.
- **Invariant**: byte-identical output across all memory budgets (8 GB → 224 GB).

**Implication**: this is the ultimate memory reduction—model size becomes a *dial*, not a floor. A 27B+ model can run in <4 GB if experts stream from fast storage.

Trade-off: latency dominated by I/O (41–71% of wall clock). Best for non-interactive or with fast NVMe.

---

## 3. Proposed Solution Pipeline

```text
1. Select ternary/MoE model for target VRAM
2. Apply hybrid attention (if available) for long context
3. Quantize aggressively (ternary / 2-bit) if needed
4. Speculative decoding for latency (optional)
5. ES adaptation to recover task quality (Agentic ESOpt-inspired)
6. Benchmark + quality gate
7. Recommend best config
```

### Target mapping

| VRAM | Strategy | Example |
|---|---|---|
| 2 GB | Ternary 2–4B | Bonsai 2 4B (0.95 GB) |
| 2–4 GB | Ternary 8–9B + ES adaptation | Bonsai 2 9B (2 GB) |
| 4–6 GB | Ternary 27B (near-full quality) | Bonsai 2 27B (5.9 GB) |
| 6–8 GB | Ternary 27B + speculative | Bonsai 2 27B |
| 8+ GB | Dense 8B Q4 + speculative | Qwen3 8B Q4 |

---

## 4. Why this can work (the "impossible" becomes possible)

The original goal (5090-level model in <4 GB) is achieved **not** by shrinking a 5090 model into 4 GB, but by:

1. **Using models designed for density** (ternary, hybrid attention).
2. **Adapting them to tasks** with ES (recovering quality lost to compression).
3. **Measuring useful quality**, not raw parameter count.

Bonsai 2 27B retains 98.2% accuracy at 5.9 GB. A 9B ternary model in 2 GB with ES adaptation can plausibly match a 7B dense model's task performance.

---

## 5. Experiment Plan

### Phase A: Ternary baseline (no adaptation)
- Benchmark Bonsai 2 (2B/4B/9B/27B) tok/s and quality on 2/4/8 GB.
- Measure VRAM peak, TTFT, context scaling.

### Phase B: Speculative decoding on ternary
- Test if speculative decoding works with ternary models.
- Measure acceptance rate and speedup.

### Phase C: ES adaptation
- Apply ES to a quantized model on a target task (e.g., code, RAG).
- Measure quality recovery vs. base dense model.

### Phase D: Recommender
- Given VRAM + objective, recommend best config.
- Validate with real benchmarks.

### Phase E: Disk streaming (kimi-k3-in-c inspired)
- Prototype MoE expert streaming from disk with LRU cache.
- Measure I/O share of wall clock and tok/s at 4/8 GB.
- Compare against ternary in-memory baseline.

---

## 6. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Ternary quality drops on some tasks | ES adaptation + task-specific eval |
| Speculative decoding may not work on ternary | Fall back to self-speculative / n-gram |
| Hybrid attention availability limited | Maintain dense fallback |
| Offloading too slow | Reserve for non-interactive workloads |

---

## 7. References

- Microsoft BitNet b1.58 (arXiv:2402.17764)
- PrismML Ternary Bonsai family (2026)
- Agentic ESOpt (gradient-free ES adaptation)
- Gated DeltaNet / hybrid attention (Qwen3.8)
- LLM Configurator benchmarks

---

## 8. Next Steps in This Repo

1. Add Bonsai models to `models/catalog.yaml`.
2. Update recommender to prefer ternary models for <4 GB.
3. Add ES adaptation experiment harness.
4. Benchmark Bonsai on available hardware.
5. Publish results in `results/`.
