"""Detect hardware capabilities for LLM inference planning."""

import os
import platform
import subprocess


def _nvidia_info():
    info = {"gpu_name": None, "vram_gb": None}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        line = out.strip().splitlines()[0]
        name, mem_mib = [x.strip() for x in line.split(",")]
        info["gpu_name"] = name
        info["vram_gb"] = round(float(mem_mib) / 1024.0, 2)
    except Exception:
        pass
    return info


def _ram_gb():
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        return None


def get_profile():
    """Return a dict describing the local machine."""
    nv = _nvidia_info()
    return {
        "os": platform.system(),
        "arch": platform.machine(),
        "cpu_cores": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "gpu_name": nv["gpu_name"],
        "vram_gb": nv["vram_gb"],
    }
