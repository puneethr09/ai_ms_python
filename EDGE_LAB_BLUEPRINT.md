# 🗺️ Edge & Local AI Systems Blueprint
> **Comprehensive Deployment & Roadmap Guide: Apple Mac & Raspberry Pi 5 (Pironman 5 + 1TB NVMe)**

---

## 📍 Current Checkpoint Status

```
[ Active Stage ]: Stage 2 (Local Mac) — llama.cpp Initial Setup Complete!
[ Mac Status   ]: llama.cpp compiled with Metal GPU; Qwen 2.5 Coder 7B (4.68 GB) benchmarked at 18.4 t/s.
[ Next Step    ]: Deep-dive into C++ inference internals (KV-Cache, Prefill vs Decode, Quantization math).
[ Pi 5 Status  ]: Planned for Stage 3 execution.
```

---

# 🖥️ Part 1: The Apple Mac Workstation (Development & Heavy Inference)

### ⚙️ Hardware Profile
* **Architecture:** Apple Silicon (Unified Memory Architecture - UMA)
* **RAM:** 16 GB High-Bandwidth LPDDR5 (100–400 GB/s)
* **Role:** Primary development machine, C++ systems compilation, and local heavy coding model execution (7B–8B parameter tier).

---

### 📋 Mac Milestone Checklist

#### ✅ Phase 1: Systems & Concurrency Foundations (COMPLETED)
- [x] **Project 1:** CLI Smart File Organizer (`pathlib`, generators, POSIX `rename`).
- [x] **Project 2:** HackerNews Data Cruncher (`requests`, safe map reads, list comprehensions).
- [x] **Project 3:** High-Performance Hybrid API (`pybind11`, `dlopen` mechanics, Scoped GIL Release, Zero-Copy C-Buffer Protocol).
- [x] **Knowledge Base:** Modularized into 6 textbook-grade chapters with the *8 Core Laws of Systems Engineering*.

#### 🔄 Phase 2: `llama.cpp` Deep Dive (IN PROGRESS - CURRENT STAGE)
- [x] Clone and compile `llama.cpp` from source with Apple Metal GPU support (`-DGGML_METAL=ON`).
- [x] Download and benchmark **Qwen 2.5 3B** (`38.5 t/s`) and **Qwen 2.5 Coder 7B** (`18.4 t/s`, 86 GB/s bandwidth).
- [ ] **Milestone 4.1:** Inspect `mmap()` model loading mechanics in `src/llama-mmap.cpp`.
- [ ] **Milestone 4.2:** Deep-dive into **Prefill ($\text{GEMM}$)** vs. **Decode ($\text{GEMV}$)** hardware physics.
- [ ] **Milestone 4.3:** Inspect the **KV-Cache memory layout** in `src/llama-kv-cache.h`.
- [ ] **Milestone 4.4:** Understand **4-bit K-Quantization (`block_q4_K`)** variable bit precision and SIMD dequantization kernels.
- [ ] **Milestone 4.5:** Write a custom standalone C++ inference driver using the native `llama.h` C++ API.

---

# 🥧 Part 2: The Raspberry Pi 5 Edge Appliance (24/7 Autonomous Node)

### ⚙️ Hardware Profile
* **Enclosure:** Pironman 5 (Active Tower Fan, 0.96" I2C OLED display, Addressable RGB LEDs).
* **Storage:** 1 TB NVMe M.2 SSD connected via PCIe Gen 2/3 (Instant `mmap` loading, 800+ MB/s disk I/O).
* **RAM:** 8 GB LPDDR4X (~17 GB/s memory bandwidth).
* **Network:** Tailscale Private Encrypted Mesh VPN.
* **Role:** Always-on (5W power), self-hosted media/services hub, autonomous background AI sentinel, and private developer assistant.

---

### 📦 Existing Self-Hosted Services Running on Pi 5
1. **Media & Entertainment:** Plex Media Server + `*arr` stack (Sonarr, Radarr, Prowlarr, Bazarr).
2. **Photo Backup & ML Search:** Immich (Self-hosted Google Photos alternative with vector search).
3. **Custom Personal Apps:** Calorie Tracker app + Stock Fundamental Analysis app.

---

### 🚀 Planned Edge AI Deployments on Pi 5 (Stage 3 Execution)

#### 1. 🛡️ The AI "Overwatcher" (Autonomous Log Sentinel & SRE)
* **How it works:** A lightweight Python background daemon tails `docker events` and logs of Plex, Immich, `*arr`, Calorie Tracker, and system `journalctl`.
* **When an error / crash occurs:**
  * Extracts the surrounding 30 lines of stack trace.
  * Prompts the local 1B/3B model on Pi 5 to diagnose: Root Cause + Severity + Suggested Fix Command.
  * Sends an instant alert to your phone via **Telegram Bot** with interactive action buttons: `[ 🔄 Auto-Fix ]`, `[ 📜 Full Logs ]`.

#### 2. 📟 Pironman 5 Hardware Telemetry & Alert Lighting
* **OLED Display Cycling:**
  * Screen 1: `🟢 AI Sentinel: HEALTHY | Containers: 14/14`
  * Screen 2: Real-time stock prices & sentiment summary from the financial screener.
  * Screen 3: NVMe I/O read/write speeds, CPU temp, and RAM utilization.
* **RGB Alert Lighting:** Flashes case LEDs solid **RED** upon container failures until remediated.

#### 3. 💻 Private VS Code Copilot Server (via Tailscale)
* Run `llama-server` on Pi 5 with **Qwen 2.5 Coder 1.5B / 3B** (`100.x.y.z:8080`).
* Point **Continue.dev** extension in VS Code on your Mac to the Pi 5's Tailscale IP.
* Enjoy 100% private, free, local autocomplete from anywhere in the world.

#### 4. 📄 Automated "Paperless" Document Intelligence (Paperless-ngx)
* Drop PDF receipts, utility bills, and medical scans into an `/inbox` folder.
* Local AI extracts vendor, date, total amount, tax tags, and auto-archives: `2026-08-Electricity-Bescom-$45.pdf`.

#### 5. 🛡️ Network Ad-Blocker & Tailscale Subnet Router (Pi-hole)
* Blocks ads network-wide for all smart TVs, phones, and computers on home Wi-Fi.
* Subnet routing allows secure access to all home LAN devices from anywhere.

#### 6. 🌙 The "Night Shift & Day Shift" Indian Stock Fundamental Analyst
* **The Problem Solved:** Running a heavy 7B model on a Pi 5 to read 50-page financial reports takes 4–5 minutes of CPU grinding (unusable for live chat) and consumes 4.68 GB of daytime RAM.
* **🌙 The Night Shift (2:00 AM – 4:00 AM / Scheduled Cron):**
  1. Pi 5 is 100% idle (zero active Plex streams or user requests).
  2. Python script loads the **heavy Qwen 2.5 7B model** into RAM via `mmap()`.
  3. Fetches 20 Indian stock watchlist tickers (`RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`, `TATAMOTORS.NS`) via `yfinance` & BSE/NSE corporate filings.
  4. Crunches quarterly financials, cash flow ratios, and earnings concall transcripts.
  5. Stores structured markdown summaries and risk matrices in a local **SQLite database (`stocks.db`)**.
  6. Calls `munmap()` to **completely unload the 7B model from RAM** before 6:00 AM!
* **☀️ The Day Shift (Market Hours 9:00 AM – 3:30 PM IST):**
  1. Loads the **ultra-fast Llama 3.2 1B model (770 MB RAM)**.
  2. You send a query on your phone via Telegram: *"What were the key debt reduction updates for Tata Motors in the latest concall?"*
  3. The 1B model retrieves the pre-computed overnight summary from `stocks.db`.
  4. Responds to your phone in **`0.5 seconds` at `18 tokens/sec`**!
* **Architectural Advantage:** Zero perceived latency, 0 MB of wasted daytime RAM, and gives you the **analytical depth of a 7B model with the instant speed of a 1B model**!

---

### 📊 Pi 5 RAM & Resource Allocation Budget (8 GB Total)

```
┌────────────────────────────────────────────────────────────────────────┐
│  [ Linux Kernel & System OS ] ──────────> ~400 MB  (0.2% CPU)          │
│  [ Plex Media Server ] ─────────────────> ~350 MB  (0.5% CPU)          │
│  [ Immich (Photos + Vector DB) ] ───────> ~1,200 MB (1.0% CPU)         │
│  [ *arr Stack (Sonarr, Radarr, etc.) ] ─> ~600 MB  (0.5% CPU)          │
│  [ Calorie Tracker & Stock App ] ───────> ~200 MB  (0.1% CPU)          │
│  [ Pi-hole & Tailscale ] ───────────────> ~150 MB  (0.1% CPU)          │
│  [ AI Overwatcher & OLED Daemon ] ──────> ~50 MB   (0.1% CPU)          │
│  [ Local LLM (Llama 3.2 1B / Qwen 1.5B) > ~1,500 MB (Wakes on demand)   │
├────────────────────────────────────────────────────────────────────────┤
│  🟢 UNALLOCATED HEADROOM BUFFER ────────> ~3,550 MB FREE (97% CPU FREE)│
└────────────────────────────────────────────────────────────────────────┘
```

---

# 🌐 Part 3: Cross-Device Tailscale Mesh Network Architecture

```
                                    ┌────────────────────────────────────┐
                                    │    Encrypted Tailscale Mesh VPN    │
                                    └─────────────────┬──────────────────┘
                                                      │
              ┌───────────────────────────────────────┼───────────────────────────────────────┐
              │                                       │                                       │
              ▼                                       ▼                                       ▼
┌───────────────────────────┐           ┌───────────────────────────┐           ┌───────────────────────────┐
│ APPLE MAC (Workstation)   │           │ RASPBERRY PI 5 (Edge)     │           │ OFFICE RIG (RTX 3090 24GB)│
│ IP: 100.x.y.1             │           │ IP: 100.x.y.2             │           │ IP: 100.x.y.3             │
│ - Qwen 2.5 Coder 7B       │           │ - 24/7 AI Overwatcher     │           │ - Stage 5: vLLM & 70B     │
│ - Local Dev & C++ Build   │           │ - Pironman OLED Telemetry │           │ - FlashAttention-2        │
│ - Primary Coding Studio   │           │ - Media, Immich, Plex     │           │ - Enterprise Benchmarking │
└───────────────────────────┘           └───────────────────────────┘           └───────────────────────────┘
```

---
*Maintained as the authoritative operational blueprint for the Mac & Pi 5 Edge AI Lab.*
