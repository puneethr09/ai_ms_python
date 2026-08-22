# 🛠️ The Hardware Lab Blueprint: Mac, Raspberry Pi 5 & RTX 3090 Workstation
> **Comprehensive Setup, Deployment & Edge AI Architecture Manual**

---

## 🗺️ Visual Lab Topology

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               TAILSCALE SECURE MESH VPN                                │
│                                                                                        │
│  [ 💻 Local Mac (16 GB) ] <═════════════════════════════════> [ 🍓 Raspberry Pi 5 (8 GB)]
│  • Primary Dev & High-Speed 7B AI                             • 24/7 Autonomous Edge AI  │
│  • Metal GPU Shaders (38 t/s)                                 • Pironman 5 + 1 TB NVMe   │
│  • FastAPI + C++ pybind11                                     • AI Overwatcher & Sentinel│
│                                                                                        │
│                                           ▲                                            │
│                                           ║ Tailscale Mesh                             │
│                                           ▼                                            │
│                       [ 🏢 Office Workstation (24 GB VRAM) ]                          │
│                       • Heavy Multi-User vLLM / TensorRT-LLM                          │
│                       • 32B Coder / 70B Heavy Reasoning Models                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

# Part 1: Raspberry Pi 5 (The 24/7 Edge Autonomous Node)

### ⚙️ Physical Hardware Profile
* **SoC:** Broadcom BCM2712 Quad-core ARM Cortex-A76 @ 2.4 GHz.
* **RAM:** 8 GB LPDDR4X (17 GB/s Memory Bandwidth).
* **Storage:** 1 TB M.2 NVMe SSD over PCIe Gen 2/3 (Boot drive & model storage).
* **Chassis:** Pironman 5 with Tower Cooler, 0.96" OLED screen & Addressable RGB LEDs.
* **Networking:** Gigabit Ethernet + Tailscale Mesh IP.

---

### 🚀 1. The Autonomous AI Overwatcher (Log Sentinel & Auto-Fixer)

An always-on background daemon that monitors container health and runs local AI diagnostics when errors occur.

```
[ Docker / OS Logs ] ──(Trigger: ERROR/OOM/PANIC)──> [ Local 1B/3B AI ] ──> [ Telegram Alert + Action Buttons ]
```

#### Blueprint:
* **Listener:** Python daemon using `docker` and `systemd` journal tailing.
* **Diagnostician:** Calls local `llama-server` on `http://127.0.0.1:8080/v1/chat/completions` with **Llama 3.2 1B / Qwen 2.5 3B**.
* **Dispatcher:** Sends an encrypted alert to your private **Telegram Bot** with actionable inline callback buttons (`[🔄 Restart Container]`, `[📜 View Full Logs]`).

---

### 📟 2. Pironman 5 OLED Screen & RGB Hardware Telemetry

A dedicated Python script reading hardware stats and displaying live AI telemetry on the physical case.

* **OLED Display Cycling (Every 5 seconds):**
  * **Screen 1 (Sentinel Status):** `🟢 AI Overwatcher: HEALTHY | Containers: 12/12`
  * **Screen 2 (Stock Screener):** Top 3 ticker price updates & daily sentiment tags.
  * **Screen 3 (Hardware Telemetry):** CPU Temp, NVMe I/O read/write speeds, RAM usage.
* **RGB LED Alert System:**
  * **Normal State:** Breathing blue / green.
  * **Critical Alert (Crash / OOM detected by Overwatcher):** Solid red flashing until acknowledged.

---

### 💻 3. Private VS Code Copilot via Tailscale (Continue.dev)

Turn the Pi 5 into a private, zero-cost GitHub Copilot server accessible from your Mac anywhere in the world.

* **Backend:** Run `llama-server` on Pi 5:
  ```bash
  ./llama-server -m models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --port 8080 --host 0.0.0.0
  ```
* **Frontend (On your Mac):** Install the **Continue.dev** extension in VS Code and set `apiBase` to `http://<PI_TAILSCALE_IP>:8080/v1`.
* **Benefit:** Fast, private, offline code completions directly in your Mac's editor!

---

### 📄 4. Automated Document & Receipt Intelligence (Paperless-ngx)

* **Directory:** `/storage/documents/inbox/` on your 1 TB SSD.
* **Workflow:** Drop utility bills, medical scans, or PDF receipts into the folder.
* **AI Action:** The local 3B model extracts Vendor, Date, Total Amount, and Tax Categories, automatically archiving the file as `2026-08-ElectricityBill-$45.pdf` with full-text search.

---

### 🛡️ 5. Network Ad-Blocking & Subnet Routing (Pi-hole + Tailscale)

* **Ad-Blocking:** Runs **Pi-hole / AdGuard Home** in Docker. Blocks 95%+ of ads for all devices on your home Wi-Fi with zero browser extensions.
* **Subnet Router:** Enable Tailscale subnet routing (`tailscale up --advertise-routes=192.168.1.0/24`) to access your entire home local network securely from remote locations.

---

### 📊 Pi 5 RAM Allocation Budget (8 GB Total)

| Workload / Service | RAM Allocated | CPU State (Idle) |
| :--- | :--- | :--- |
| **Linux OS & Systemd** | ~400 MB | 0.2% |
| **Plex Media Server** | ~350 MB | 0.5% |
| **Immich (Photos + Vector DB)** | ~1,200 MB | 1.0% |
| **`*arr` Stack (Sonarr, Radarr, Prowlarr)** | ~600 MB | 0.5% |
| **Calorie Tracker & Stock App** | ~200 MB | 0.1% |
| **Pi-hole & Tailscale VPN** | ~150 MB | 0.1% |
| **AI Overwatcher & OLED Daemon** | ~50 MB | 0.1% |
| **Local LLM (`llama-server` 1.5B/3B Q4)** | **~1,800 MB** | **0% (Wakes on demand)** |
| **FREE BUFFER & DISK CACHE** | **~3,250 MB FREE!** | **~97% CPU Free!** |

---

# Part 2: Local Mac (The High-Speed Dev & 7B/8B AI Workstation)

### ⚙️ Physical Hardware Profile
* **Chip:** Apple Silicon M-Series (Unified Memory Architecture).
* **RAM:** 16 GB Unified Memory (80–100+ GB/s Memory Bandwidth).
* **Primary Role:** High-speed development, local dual-language engineering, and primary 7B/8B coding model workstation.

---

### 🚀 1. Local Daily Coding Copilot (Qwen 2.5 Coder 7B)

* **Engine:** `llama.cpp` built with `-DGGML_METAL=ON`.
* **Model:** `qwen2.5-coder-7b-instruct-q4_k_m.gguf` (4.68 GB in RAM).
* **Speed:** **`38 tokens/second generation`** (~86 GB/s memory bandwidth).
* **Workflow:** Run as a local background server:
  ```bash
  cd learn_projects/llama.cpp
  ./build/bin/llama-server -m models/qwen2.5-coder-7b-instruct-q4_k_m.gguf --port 8080 -ngl 99
  ```
* **UI:** Access ChatGPT-style UI at `http://localhost:8080` or connect to VS Code.

---

### ⚡ 2. The High-Performance Hybrid C++/Python Development Node

* **Workspace:** `learn_projects/hybrid_api/`
* **Architecture:** FastAPI async networking front-end + C++20 `pybind11` compute engine with scoped GIL release and Zero-Copy Buffer protocol.
* **Benchmark:** $26,000\times$ speedup on heavy math, sub-5ms zero-copy array operations.

---

# Part 3: Office Workstation (The Heavy RTX 3090 AI Workhorse)

### ⚙️ Physical Hardware Profile
* **GPU:** NVIDIA GeForce RTX 3090 (24 GB GDDR6X VRAM - Ampere Architecture).
* **Host RAM:** 251 GB DDR4 RAM.
* **CPU:** Intel Xeon W-2275 @ 3.30 GHz (28 threads).
* **Primary Role:** Enterprise serving (vLLM / TensorRT-LLM), 32B Coder / 70B quantized models, custom CUDA kernel development.

---

### 🚀 1. Enterprise Model Serving (vLLM Continuous Batching)
* **Models:** **Qwen 2.5 Coder 32B (Q4)** or **Llama 3.1 70B (AWQ/GPTQ)**.
* **Features:** PagedAttention for KV-cache, FlashAttention-2, multi-client continuous batching.
* **Inference Speed:** **`50 to 90 tokens/sec`** with massive context windows (32k+ tokens).

---

# 📅 Master Execution Roadmap

| Milestone | Target Device | Focus & Deliverables |
| :--- | :--- | :--- |
| **Stage 1 (Done ✅)** | **Local Mac** | FastAPI + C++ `pybind11` Zero-Copy Hybrid Server |
| **Stage 2 (Done ✅)** | **Local Mac** | `llama.cpp` Metal GPU Build + Qwen 7B/3B Benchmarking |
| **Stage 3 (Next 🔜)** | **Raspberry Pi 5** | Cross-Compiling `llama.cpp` + Deploying the AI Overwatcher & OLED Daemon |
| **Stage 4** | **Google Colab** | Writing Custom CUDA C++ Kernels from Scratch (`.cu`) |
| **Stage 5** | **Office RTX 3090** | Enterprise vLLM + FlashAttention-2 Profiling & 70B Model Deployment |
