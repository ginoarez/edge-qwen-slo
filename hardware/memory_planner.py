"""Memory planner: split RAM budget between trunk and expert cache (kimi-k3-in-c style)."""


def plan_memory(total_gb: float, trunk_ratio: float = 0.7):
    """Return (trunk_gb, cache_gb) respecting a 95% safety margin."""
    usable = total_gb * 0.95
    trunk = round(usable * trunk_ratio, 2)
    cache = round(usable - trunk, 2)
    return {"total_gb": total_gb, "usable_gb": round(usable, 2), "trunk_gb": trunk, "cache_gb": cache}


def recommend_mode(total_gb: float):
    if total_gb < 8:
        return "ultra"
    if total_gb < 32:
        return "laptop"
    if total_gb < 96:
        return "desktop"
    if total_gb < 128:
        return "workstation"
    return "server"
