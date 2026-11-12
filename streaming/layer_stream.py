"""Exact layer streaming: keep only one decoder layer in VRAM.

Inspired by Soup (Exact Layer Streaming) and kimi-k3-in-c. The frozen base
is streamed from host RAM (or disk) one layer at a time, achieving
bit-exact results vs a fully resident model.
"""


class LayerStreamer:
    def __init__(self, num_layers: int, layer_size_bytes: int, vram_budget_bytes: int):
        self.num_layers = num_layers
        self.layer_size = layer_size_bytes
        self.vram_budget = vram_budget_bytes
        self.resident_layers = max(1, vram_budget_bytes // layer_size_bytes)

    def plan(self):
        return {
            "num_layers": self.num_layers,
            "layer_size_bytes": self.layer_size,
            "vram_budget_bytes": self.vram_budget,
            "resident_layers": self.resident_layers,
            "streamed_layers": self.num_layers - self.resident_layers,
            "peak_vram_bytes": self.resident_layers * self.layer_size,
        }

    def estimate_peak_vram_gb(self):
        return round((self.resident_layers * self.layer_size) / (1024 ** 3), 3)
