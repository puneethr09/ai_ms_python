import json
import statistics
import time
from pathlib import Path
from typing import Callable

HARDWARE_JSON = Path(__file__).resolve().parent.parent / "results" / "hardware.json"


def time_fn(fn: Callable[[], object], warmup: int = 3, iters: int = 20,
            sync: Callable[[], None] | None = None) -> dict:
    """Run fn repeatedly and return timings in milliseconds. Pass sync for async devices (MPS/CUDA)."""
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        samples.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": statistics.median(samples), "min_ms": min(samples), "iters": iters}


def load_peaks() -> dict:
    if not HARDWARE_JSON.exists():
        raise FileNotFoundError("No measured peaks yet. Run: python -m bench.membw")
    return json.loads(HARDWARE_JSON.read_text())


def pct_of_bandwidth(bytes_moved: float, seconds: float, device: str = "cpu") -> float:
    """Achieved bandwidth as a percentage of this machine's measured peak for device ('cpu' or 'gpu')."""
    achieved_gbps = bytes_moved / seconds / 1e9
    return 100.0 * achieved_gbps / load_peaks()[device]["peak_gbps"]


def predicted_decode_tok_s(model_bytes: float, device: str = "cpu") -> float:
    """Upper bound on decode speed: every weight byte crosses the memory bus once per token."""
    return load_peaks()[device]["peak_gbps"] * 1e9 / model_bytes
