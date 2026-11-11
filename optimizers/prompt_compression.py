"""Prompt/context compression helpers (LLMLingua-style)."""


def compress_text(text: str, target_ratio: float = 0.5) -> str:
    """Placeholder: return text unchanged.

    Real implementation would score tokens with a small model and drop
    low-information tokens. Kept as interface for integration.
    """
    if target_ratio >= 1.0:
        return text
    # naive fallback: keep first N chars
    keep = int(len(text) * target_ratio)
    return text[:keep]
