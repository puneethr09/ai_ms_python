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

## 2. Real-World Systems Case Studies

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

## 3. Systems Security & Memory Exploits

### 💉 1. DLL / Shared Object Injection
* **Mechanism:** A process abuses OS debugging syscalls (`OpenProcess`, `VirtualAllocEx`, `WriteProcessMemory`, `CreateRemoteThread`) to inject malicious machine code into the address space of a target process.

### 💣 2. Buffer Overflows
* **Mechanism:** Writing beyond allocated buffer boundaries on the Stack to overwrite the Return Address (`RIP` / Instruction Pointer), hijacking execution flow when the function returns.

### 👑 3. Kernel Privilege Escalation (Ring 3 $\to$ Ring 0)
* **Mechanism:** Exploiting vulnerable device drivers to write into kernel Page Tables, granting user-mode processes unrestricted access to physical RAM.

### 👻 4. Hardware Speculative Side-Channels (Spectre & Meltdown)
* **Mechanism:** Exploiting CPU speculative branch execution to transiently load unauthorized memory into L1/L2 caches, measuring nanosecond cache-timing differences to reconstruct passwords and encryption keys.
