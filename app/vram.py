import subprocess
import threading
import time


def sample_vram_once() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        # memory.used is in MiB
        mib = float(out.strip().splitlines()[0])
        return round(mib / 1024.0, 3)
    except Exception:
        return None


def vram_sampler(stop_event: threading.Event, interval: float = 2.0):
    from .metrics import VRAM_USED

    while not stop_event.is_set():
        val = sample_vram_once()
        if val is not None:
            VRAM_USED.set(val)
        stop_event.wait(interval)
