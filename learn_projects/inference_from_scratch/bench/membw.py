"""Measure this machine's real memory bandwidth (CPU via bench/membw.c, GPU via PyTorch MPS)."""
import json
import platform
import subprocess
from pathlib import Path

from bench.harness import HARDWARE_JSON, time_fn

BENCH_DIR = Path(__file__).resolve().parent


def cpu_bandwidth() -> dict:
    binary = BENCH_DIR / "membw"
    subprocess.run(["clang", "-O3", "-march=native", "-o", str(binary), str(BENCH_DIR / "membw.c"), "-lpthread"],
                   check=True)
    out = subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout
    rows = []
    for line in out.strip().splitlines():
        kernel, threads, gbps = line.split()
        rows.append({"kernel": kernel, "threads": int(threads), "gbps": float(gbps)})
        print(f"  cpu  {kernel:<6} {threads:>2} threads  {float(gbps):7.2f} GB/s")
    return {"peak_gbps": max(r["gbps"] for r in rows), "runs": rows}


def gpu_bandwidth() -> dict | None:
    import torch
    if not torch.backends.mps.is_available():
        return None
    n = 128 * 1024 * 1024  # 512 MiB of float32
    src = torch.ones(n, dtype=torch.float32, device="mps")
    dst = torch.empty_like(src)
    nbytes = n * 4
    sync = torch.mps.synchronize
    rows = []
    for kernel, fn, moved in [
        ("read", lambda: src.sum(), nbytes),
        ("copy", lambda: dst.copy_(src), 2 * nbytes),
    ]:
        t = time_fn(fn, warmup=5, iters=30, sync=sync)
        gbps = moved / (t["min_ms"] / 1e3) / 1e9
        rows.append({"kernel": kernel, "gbps": round(gbps, 2)})
        print(f"  gpu  {kernel:<6}             {gbps:7.2f} GB/s")
    return {"peak_gbps": max(r["gbps"] for r in rows), "runs": rows}


def main() -> None:
    print("Measuring CPU bandwidth (takes ~30s)...")
    cpu = cpu_bandwidth()
    print("Measuring GPU bandwidth (MPS)...")
    gpu = gpu_bandwidth()
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    result = {"chip": chip or platform.processor(), "cpu": cpu, "gpu": gpu}
    HARDWARE_JSON.parent.mkdir(exist_ok=True)
    HARDWARE_JSON.write_text(json.dumps(result, indent=2))
    print(f"\n{result['chip']}: CPU peak {cpu['peak_gbps']:.1f} GB/s"
          + (f", GPU peak {gpu['peak_gbps']:.1f} GB/s" if gpu else "")
          + f"\nSaved -> {HARDWARE_JSON}")


if __name__ == "__main__":
    main()
