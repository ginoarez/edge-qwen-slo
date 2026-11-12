"""Select model architecture family given constraints."""


def select_architecture(vram_gb: float, context_len: int = 4096, objective: str = "quality"):
    if vram_gb < 4 or context_len > 32768:
        return "ssm"  # constant memory, long context
    if objective == "speed" and vram_gb < 8:
        return "ternary-transformer"
    return "transformer"
