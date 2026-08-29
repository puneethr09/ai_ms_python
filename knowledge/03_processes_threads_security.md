# Level 3: Processes, Threads & Systems Security
> **Isolation Boundaries, Real-World Systems Architectures, and Attack Vectors**

---

## 1. Processes vs. Threads: The Core Trade-Off

| Dimension | Process | Thread |
| :--- | :--- | :--- |
| **Memory Isolation** | Full Isolation (Private Page Tables & Address Space) | Shared Heap, Data Segments & File Descriptors |
| **Crash Blast Radius** | Isolated (Process crash does not affect others) | Entire Process Terminates |
| **Creation Cost** | Heavy (New Page Tables, PID, VMAs, OS structures) | Lightweight (New Stack Frame & TCB only) |
| **Data Sharing** | Requires IPC (Pipes, Sockets, Shared Memory) | Direct Memory Pointers (Zero Overhead) |

---

## 2. 💡 The Architecture "Light-Bulb" Moment: The `.so` in Memory

```
SINGLE PROCESS (FastAPI + C++ via dlopen):
┌─────────────────────────────────────────────────────────┐
│ Virtual Address Space (PID 87263)                       │
│ ├── Stack & RSP (Hot in L1 Cache)                       │
│ ├── mmap Region: engine.so (C++ Machine Code Loaded)    │
│ └── Heap: [ 0x7FFF1000 (40 MB Array) ]                  │
│               ▲                                         │
│               │ Direct 8-byte pointer (Zero copies!)    │
│  C++ Engine: ─┘ (Same Page Table! Same Process!)        │
└─────────────────────────────────────────────────────────┘

SEPARATE PROCESSES (Python multiprocessing):
┌─────────────────────────────┐       ┌─────────────────────────────┐
│ Process 1 (PID 87263)       │       │ Process 2 (PID 87264)       │
│ ├── Page Table 1            │       │ ├── Page Table 2            │
│ ├── Data: 0x7FFF1000        │       │ ├── 0x7FFF1000 is UNMAPPED! │
│ └── GIL #1 (Locked)         │       │ └── GIL #2 (Locked)         │
└─────────────────────────────┘       └─────────────────────────────┘
  * Passing pointer 0x7FFF1000 to Process 2 causes a SEGFAULT!
  * Process 2 has its own isolated Page Table and cannot touch Process 1's RAM.
```

---

## 3. Why Multi-Process on Multi-GPU vs. C++ Threads?

Why does PyTorch/vLLM use **multiple Python processes** for multi-GPU instead of just C++ threads?

1. **CUDA Driver & GIL Independence:** Each GPU (GPU 0, GPU 1) is driven by its own Python process with its own independent GIL, eliminating interpreter contention.
2. **The Data Center Rule (Threads Cannot Cross Network Cables!):**
   * If you run a 70B model across **10 servers (80 GPUs total)**:
   * **C++ Threads are trapped on 1 motherboard.** A thread on Server A cannot read RAM on Server B.
   * **Processes can scale across the entire planet!** Using **NCCL (NVIDIA Collective Communications Library)** over InfiniBand networks, Process 0 on Server A talks to Process 79 on Server J identically to how it talks on local PCIe!

---

## 4. Web Traffic Scaling: From Async to Kubernetes (K8s)

Look at the 3-Tier Scaling Pyramid for Web Servers:

```
Tier 1: Single Process Concurrency (FastAPI + asyncio)
└── 1 Process multiplexes 10,000 idle sockets using 1 CPU core (epoll/kqueue).

Tier 2: Multi-Process on 1 Server (Gunicorn / Uvicorn Workers)
└── Spawns 8 Worker Processes across an 8-core CPU to use 100% of local server hardware.

Tier 3: Distributed Multi-Server Containers (Docker + Kubernetes)
└── Kubernetes (K8s) manages 50 Docker container processes across 10 physical cloud servers.
└── Cloud Load Balancer (AWS ALB / NGINX) distributes incoming HTTP requests across all 50 containers!
```
> **Kubernetes is literally containerized multi-processing scaled across a data center!**


## 5. Real-World Architectural Case Studies

### 🌐 1. Google Chrome (Multi-Process Architecture)
* **Design:** Each Browser Tab runs in its own isolated OS Process.
* **Why:** If a rogue JavaScript infinite loop crashes Tab 1, Tab 2 continues running unharmed. Inside each tab, background threads decode video frames and pass memory buffers to the GPU without IPC latency.

### 🎵 2. Spotify Desktop (CEF + Native Real-Time Audio Thread)
* **Design:** Uses Chromium Embedded Framework (CEF) for UI rendering + a separate, real-time native C++ audio thread.
* **Why:** If heavy playlist scrolling or complex CSS animations lag the UI thread, the audio thread continues feeding DAC buffers without stuttering or audio drops.

### 💻 3. Visual Studio Code (Hybrid Sandboxing)
* **Design:** Main Electron UI Process + Extension Host Process (sandboxes community plugins) + `clangd` Language Server (spawns 4–8 background threads for C++ AST indexing).

### 🎮 4. Game Engines (Unreal Engine / Cyberpunk)
* **Design:** Single monolithic process with dedicated threads (Game Loop, PhysX, Audio, Asset Streaming) sharing a unified 16 GB Heap memory space to eliminate IPC serialization latency.

---

## 6. Systems Security & Memory Exploits

### 💉 1. DLL / Shared Object Injection
* **Mechanism:** A process abuses OS debugging syscalls (`OpenProcess`, `VirtualAllocEx`, `WriteProcessMemory`, `CreateRemoteThread`) to inject malicious machine code into the address space of a target process.

### 💣 2. Buffer Overflows
* **Mechanism:** Writing beyond allocated buffer boundaries on the Stack to overwrite the Return Address (`RIP` / Instruction Pointer), hijacking execution flow when the function returns.

### 👑 3. Kernel Privilege Escalation (Ring 3 to Ring 0)
* **Mechanism:** Exploiting vulnerable device drivers to write into kernel Page Tables, granting user-mode processes unrestricted access to physical RAM.

### 👻 4. Hardware Speculative Side-Channels (Spectre & Meltdown)
* **Mechanism:** Exploiting CPU speculative branch execution to transiently load unauthorized memory into L1/L2 caches, measuring nanosecond cache-timing differences to reconstruct passwords and encryption keys.
