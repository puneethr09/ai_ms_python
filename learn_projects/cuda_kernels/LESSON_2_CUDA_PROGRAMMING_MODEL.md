# LESSON 2: The CUDA Programming & Coordinate Model (Complete Reference)

## 2.1 The Big Shift: For Loops Disappear

### CPU Approach (Sequential)
```cpp
// ONE core walks through N elements, one at a time
void add_arrays(float* a, float* b, float* c, int N) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}
```

### GPU Approach (Parallel)
```cpp
// No loop! Hardware spawns N threads, each does ONE element
__global__ void add_arrays(float* a, float* b, float* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}
```

### Mental Model:
- **CPU:** You are a foreman walking down 1,000,000 boxes, opening each one yourself.
- **GPU:** You hire 1,000,000 workers, each standing at their own box. Shout "GO!" — all open simultaneously.

---

## 2.2 The 3-Tier Hierarchy: Grid → Block → Thread

```
GRID  (The entire job → maps to the whole GPU)
 ├── Block 0  (256 threads → assigned to one SM)
 │    ├── Warp 0: Threads 0..31   (lockstep execution)
 │    ├── Warp 1: Threads 32..63
 │    └── ...Warp 7: Threads 224..255
 ├── Block 1  (256 threads → assigned to another SM)
 ├── Block 2
 └── Block N
```

### Why 3 levels? Direct hardware mapping:

| Software Concept | Hardware It Maps To | Why It Exists |
|---|---|---|
| **Grid** | Entire GPU (all 40-82 SMs) | Represents the full computation |
| **Block** | Runs on one SM (but multiple blocks can share one SM) | Threads share that SM's L1/Shared Memory, can synchronize |
| **Warp** (32 threads) | One Warp Scheduler + 32 ALUs | Smallest execution unit, always runs in lockstep |

### Critical Rules:
- Threads inside the **same block** CAN share data (`__shared__` memory) and sync (`__syncthreads()`)
- Threads in **different blocks** CANNOT communicate — may run on different SMs at different times

---

## 2.3 The 1D Coordinate Formula

Each thread has 3 built-in hardware variables (read-only, set automatically):

| Variable | Meaning | Example |
|---|---|---|
| `threadIdx.x` | "Who am I inside my block?" | 0, 1, 2, ... 255 |
| `blockIdx.x` | "Which block am I in?" | 0, 1, 2, 3 |
| `blockDim.x` | "How many threads in my block?" | 256 |

### The Formula:
```
Global ID (i) = blockIdx.x * blockDim.x + threadIdx.x
```

### Walkthrough (1024 elements, 256 threads/block = 4 blocks):
```
Block 0: Thread 0 → i = 0*256+0 = 0     ... Thread 255 → i = 0*256+255 = 255
Block 1: Thread 0 → i = 1*256+0 = 256    ... Thread 255 → i = 1*256+255 = 511
Block 2: Thread 0 → i = 2*256+0 = 512    ... Thread 255 → i = 2*256+255 = 767
Block 3: Thread 0 → i = 3*256+0 = 768    ... Thread 255 → i = 3*256+255 = 1023
```
Every thread gets a unique `i` from 0 to 1023. No duplicates. No gaps.

---

## 2.4 The Boundary Guard

If N = 1000 (not a clean multiple of 256), you launch 4 blocks = 1024 threads.
Threads 1000–1023 are out of bounds → would corrupt VRAM without a guard.

```cpp
__global__ void add_arrays(float* a, float* b, float* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {           // Out-of-bounds threads exit here
        c[i] = a[i] + b[i];
    }
}
```

### Ceiling division to calculate block count:
```cpp
int blockSize = 256;
int numBlocks = (N + blockSize - 1) / blockSize;  // ceil(N / blockSize)
```

---

## 2.5 Extending to 2D: Matrix Coordinates

For matrices, use `dim3` to declare 2D blocks and grids:

```cpp
dim3 blockDim(16, 16);  // 16x16 = 256 threads per block
dim3 gridDim(
    (NUM_COLS + 15) / 16,   // blocks across columns (X)
    (NUM_ROWS + 15) / 16    // blocks across rows (Y)
);
```

Inside the kernel:
```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;   // X axis
int row = blockIdx.y * blockDim.y + threadIdx.y;   // Y axis

if (row < HEIGHT && col < WIDTH) {
    int index = row * WIDTH + col;   // 2D → 1D flat index
    C[index] = ...;
}
```

### Visual: 64x64 matrix covered by 16x16 blocks (4x4 = 16 blocks):
```
          col 0        col 16       col 32       col 48
    ┌─────────────┬─────────────┬─────────────┬─────────────┐
    │ Block(0,0)  │ Block(1,0)  │ Block(2,0)  │ Block(3,0)  │ row 0
    ├─────────────┼─────────────┼─────────────┼─────────────┤
    │ Block(0,1)  │ Block(1,1)  │ Block(2,1)  │ Block(3,1)  │ row 16
    ├─────────────┼─────────────┼─────────────┼─────────────┤
    │ Block(0,2)  │ Block(1,2)  │ Block(2,2)  │ Block(3,2)  │ row 32
    ├─────────────┼─────────────┼─────────────┼─────────────┤
    │ Block(0,3)  │ Block(1,3)  │ Block(2,3)  │ Block(3,3)  │ row 48
    └─────────────┴─────────────┴─────────────┴─────────────┘
```

### Example: Thread at blockIdx=(2,1), threadIdx=(4,8):
- col = 2 * 16 + 4 = **36**
- row = 1 * 16 + 8 = **24**
- Processes element **(row 24, col 36)**
- Flat index = 24 * 64 + 36 = **1,572**

---

## 2.6 How the Hardware Scheduler Assigns Blocks to SMs

When you call `kernel<<<gridDim, blockDim>>>(...)`:

1. **GigaThread Engine** (master scheduler) receives the entire Grid
2. Distributes blocks to SMs one at a time, round-robin
3. Each SM can hold **multiple blocks simultaneously** (limited by register/shared memory usage)
4. Once a block's threads finish, SM accepts the next unprocessed block

```
Grid: 16 blocks, GPU: 4 SMs

Time 0:  SM0←Block0   SM1←Block1   SM2←Block2   SM3←Block3
Time 1:  SM0←Block4   SM1←Block5   SM2←Block6   SM3←Block7
Time 2:  SM0←Block8   SM1←Block9   SM2←Block10  SM3←Block11
Time 3:  SM0←Block12  SM1←Block13  SM2←Block14  SM3←Block15
```

**You do NOT control which SM runs which block.** Hardware handles all scheduling.

---

## 2.7 Clarifications from Discussion

### Block Size != SM Core Count
- SM has 64 CUDA cores (T4 Turing) or 128 (RTX 3090 Ampere) — fixed in silicon
- You CAN launch 256 threads on 128 cores — the SM time-slices warps across ALUs
- 256 threads = 8 warps — good for latency hiding
- Hard limits: max 1024 threads/block, max 32 warps/SM (Turing) or 48 warps/SM (Ampere)

### Warp State Lives in REGISTERS, Not L1 Cache
- Each warp's 32 threads have dedicated physical registers (never evicted)
- Scheduler swaps warps by pointing ALU inputs at different register wires → 0-cycle cost
- Shared memory: shared across ALL warps in a block (not per-warp)
- L1 cache: hardware-managed, CAN be evicted anytime

### Shared Memory and L1 Cache = Same Physical SRAM
- Each SM has on-chip SRAM: 64 KB per SM (Turing/T4) or 128 KB per SM (Ampere/3090), with configurable split:
  - Part as `__shared__` memory (YOU control, never evicted, predictable)
  - Part as L1 cache (HARDWARE controls, can be evicted, unpredictable)
- Same speed, different control model — prefer shared memory for critical data
