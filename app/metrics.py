from prometheus_client import Counter, Histogram, Gauge

REQUESTS = Counter(
    "edge_qwen_requests_total",
    "Total chat completion requests",
    ["model", "stream", "status"],
)
TOKENS = Counter(
    "edge_qwen_tokens_total",
    "Total generated tokens",
    ["model"],
)
LATENCY = Histogram(
    "edge_qwen_request_duration_seconds",
    "End-to-end request latency",
    ["model", "stream"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)
TTFT = Histogram(
    "edge_qwen_ttft_seconds",
    "Time to first token",
    ["model"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
TOKS_PER_SEC = Histogram(
    "edge_qwen_decode_toks_per_sec",
    "Decode throughput per request",
    ["model"],
    buckets=(10, 25, 50, 75, 100, 125, 150, 200, 300),
)
VRAM_USED = Gauge(
    "edge_qwen_vram_used_gb",
    "GPU VRAM used (GB)",
)
SLO_PASS = Gauge(
    "edge_qwen_slo_pass",
    "1 if last request met >=100 tok/s SLO",
)
