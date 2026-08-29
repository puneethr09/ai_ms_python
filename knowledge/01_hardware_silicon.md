# Level 1: Bare-Metal Silicon & The Latency Pyramid
> **The Hardware Foundation: From CPU Registers to DRAM Physics**

---

## 1. The Silicon Latency Pyramid

*(Normalized to Human Time: 1 CPU Clock Cycle = 1 Human Second)*

| Storage Level | Typical Size | Hardware Latency | Human-Time Analogy | Silicon Construction |
| :--- | :--- | :--- | :--- | :--- |
| **CPU Registers** | ~1 KB (per core) | 0.5 ns (1 cycle) | **1 second** (In your hands) | D-Type Flip-Flop Logic Gates in ALU |
| **L1 Cache** | ~64 KB (per core) | 1 ns (4-5 cycles) | **5 seconds** (On your desk) | 6-Transistor (6T) SRAM Cells |
| **L2 Cache** | ~1 MB (per core) | 4 ns (12-14 cycles) | **15 seconds** (In your drawer) | 6-Transistor (6T) SRAM Cells |
| **L3 Cache** | 16–128 MB (Shared)| 15 ns (~40 cycles) | **1 minute** (On bookshelf) | 6-Transistor (6T) SRAM Cells |
| **Main RAM (DRAM)**| 16–64 GB (System) | 80–100 ns (200+ cyc)| **4 minutes** (Walk to warehouse) | 1-Transistor + 1-Capacitor (1T-1C) |
| **NVMe SSD** | 1–4 TB | 10–50 microseconds | **1.5 DAYS** (Drive to factory) | NAND Flash Floating Gate Transistors |

---

## 2. CPU Registers: Hardwired ALU Logic

* **What they are:** Registers (`RAX`, `RBX`, `RSP`, `YMM0-15`) are not addressable memory banks. They are **D-Type Flip-Flop logic gates** etched directly into the ALU inputs and outputs.
* **Access Speed:** $< 0.5\text{ ns}$ (1 clock cycle). Zero address decoding, zero bus transit.

> [!NOTE]
> **Why Can't We Have 10,000 Registers?**  
> 1. **Instruction Bit-Budget:** Assembly instructions encode register IDs. With 16 registers, 4 bits are needed per operand ($2^4 = 16$). With 65,536 registers, machine instructions swell by 16 bits per operand, bloating compiled binaries and choking the L1 Instruction Cache.  
> 2. **Context-Switch Latency:** On thread switches, the OS must dump all active registers to RAM. Saving 10,000 registers on every timer interrupt would cripple OS scheduling throughput.  
> 3. **Physical Wire Congestion:** Routing physical copper multiplexer traces from thousands of registers into the ALU inputs creates an un-routable silicon bottleneck.

---

## 3. SRAM vs. DRAM: The Physics of Memory

### L1, L2, L3 (SRAM - Static RAM)
* Built using **6-Transistor (6T) flip-flop cells**.
* Stable electrical state; holds data continuously without refresh pulses as long as power is applied.
* Extremely fast, power-dense, and expensive.

### Main RAM (DRAM - Dynamic RAM)
* Built using **1-Transistor + 1-Capacitor (1T-1C) cells**.
* Capacitors leak charge continuously; memory controllers must emit thousands of electrical refresh pulses per second to prevent bit rot.
* Sits off-chip across physical motherboard copper traces, introducing a massive 80–100 ns latency penalty.

---

## 4. The $O(1)$ Circuit Reality: Why L2 and L3 are Slower than L1

While cache indexing is mathematically $O(1)$, the **hardware constant factor ($C$)** increases with cache size due to digital logic physics:

```
Virtual Address ──> [ Logic Decoder Tree ] ──> [ Wordline ] ──> [ SRAM Array ] ──> [ Bitline RC Delay ] ──> Output
```

1. **Decoder Depth:** Addressing 1 MB requires 14+ layers of NAND/Multiplexer logic gates (vs. 9 layers for 32 KB). Each gate introduces a 5–10 picosecond switching delay.
2. **Parasitic Wire Capacitance ($RC$ Delay):** In larger arrays, bitline wires are physically longer and connected to thousands of transistor drains ($T = R \times C$). Draining wire voltage from $1\text{V}$ to $0\text{V}$ takes more clock ticks.
3. **Set-Associativity Width:** L1 is typically 8-way associative (8 parallel tag comparators); L2 is 8-to-16-way (Intel/AMD) or up to 24-way (Apple Silicon) associative, requiring wider multiplexer trees and longer comparator resolution time.

---

## 5. LRU Caches: Hardware & Software Design

* **The Rule:** Discard the Least Recently Used item when capacity is full.
* **Hardware:** CPU cache controllers use Pseudo-LRU to evict 64-byte Cache Lines.
* **C++ $O(1)$ Implementation Architecture:**
  ```
  std::unordered_map<Key, list::iterator> (O(1) Search)
  └── Points directly to Node inside ──> std::list<std::pair<Key, Value>> (O(1) Node Promotion & Eviction)
                                         [Head: Most Recent] <---> [Node] <---> [Tail: Oldest (Evict)]
  ```
* **Python Standard Library:**
  ```python
  from functools import lru_cache

  @lru_cache(maxsize=128)
  def compute_heavy(n: int) -> int:
      return ... # 0 ns on repeated lookups!
  ```
