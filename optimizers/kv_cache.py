"""KV cache quantization helpers (KIVI-style, MiniKV-style)."""

KV_MODES = {
    "fp16": {"bits": 16, "memory_factor": 1.0},
    "q8": {"bits": 8, "memory_factor": 0.5},
    "q4": {"bits": 4, "memory_factor": 0.25},
    "kivi_int2": {"bits": 2, "memory_factor": 0.125},
}


def estimate_kv_bytes(seq_len: int, num_layers: int, hidden: int, mode: str = "fp16"):
    base = seq_len * num_layers * hidden * 2  # K+V, fp16
    factor = KV_MODES.get(mode, KV_MODES["fp16"])["memory_factor"]
    return int(base * factor)


def recommend_kv_mode(vram_gb: float, seq_len: int):
    if vram_gb <= 4:
        return "kivi_int2"
    if vram_gb < 8:
        return "q4"
    if vram_gb < 16:
        return "q8"
    return "fp16"
