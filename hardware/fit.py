"""Hardware-fit preflight (inspired by Soup's `soup profile`).

Answers 'will it fit?' for a given model + config without running training.
"""

MODEL_SIZES_GB = {
    "qwen3-4b": 2.5,
    "qwen3-8b": 5.2,
    "bonsai-2-27b": 5.9,
}


def estimate_peak_vram(model_id: str, quant: str = "Q4", seq_len: int = 2048, batch: int = 1):
    base = MODEL_SIZES_GB.get(model_id, 4.0)
    # rough: model + kv cache + overhead
    kv = (seq_len * 8 * 2048 * 2) / (1024 ** 3)  # layers*hidden*2 bytes
    overhead = 0.5
    return round(base + kv * batch + overhead, 2)


def will_it_fit(model_id: str, vram_gb: float, **kwargs):
    peak = estimate_peak_vram(model_id, **kwargs)
    return {"model": model_id, "peak_vram_gb": peak, "fits": peak <= vram_gb * 0.95}
