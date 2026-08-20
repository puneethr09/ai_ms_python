# Level 2: Operating System & Virtual Memory
> **The Software-Hardware Interface: Virtual Addressing, Segments, and RAII Mechanics**

---

## 1. Virtual Address Space vs. Physical RAM

### 🏨 The Hotel Reservation Ticket Analogy
* **Virtual Address Space (128 Terabytes in 64-bit):** The hotel prints reservation ticket numbers from #0 to #4 Billion. Printing tickets costs **0 physical rooms**.
* **Physical RAM (e.g., 4 GB / 16 GB):** The actual brick-and-mortar hotel rooms.
* **The Rule:** Giving BSS or Heap a 4 GB *virtual address range* consumes **0 bytes of physical RAM**. Physical RAM is claimed on a **first-write basis** when a thread physically writes bytes to an address (Demand Paging via Minor Page Faults).

---

## 2. The 64-Bit Process Memory Map

```
0xFFFFFFFFFFFFFFFF ┌──────────────────────────────────────────┐ (High Addresses)
                   │       KERNEL SPACE (OS Memory)           │ (Ring 0 Only - Protected)
                   ├──────────────────────────────────────────┤
                   │  STACK (Grows DOWNWARD ↓)                │ (RW-) Hot in L1 (Recycled at RSP)
                   │  - Local variables, return pointers      │
                   ├──────────────────────────────────────────┤
                   │                  ↓                       │
                   │           (Unallocated Gap)              │
                   │                  ↑                       │
                   ├──────────────────────────────────────────┤
                   │  Memory-Mapped Region (mmap / Shared)    │ (RW-) Loaded .so / .dylib libraries
                   ├──────────────────────────────────────────┤
                   │  HEAP (Grows UPWARD ↑ via brk)           │ (RW-) Dynamic memory (malloc/new)
                   ├──────────────────────────────────────────┤
                   │  BSS Segment (Uninitialized Globals)     │ (RW-) Fixed size, 0 bytes on disk!
                   ├──────────────────────────────────────────┤
                   │  DATA Segment (Initialized Globals)      │ (RW-) Fixed size, stores initial values
                   ├──────────────────────────────────────────┤
                   │  .rodata (Read-Only Data Segment)        │ (R--) String literals, const globals
                   ├──────────────────────────────────────────┤
                   │  TEXT / CODE Segment (READ-ONLY, R-X)    │ (R-X) Compiled Machine Instructions
0x0000000000000000 └──────────────────────────────────────────┘ (Low Addresses)
```

---

## 3. Data Segments: BSS, DATA, and .rodata

| Segment | What Lives Here | Writable? | Size on Disk in Executable |
| :--- | :--- | :--- | :--- |
| **`.rodata`** | String literals (`"Hello"`), `const` global arrays | ❌ No (`R--` Segfaults on write) | Takes full bytes |
| **`DATA`** | Initialized globals (`int g_val = 42;`) | ✅ Yes (`RW-`) | Takes full bytes |
| **`BSS`** | Uninitialized globals (`int g_buf[1000];`) | ✅ Yes (`RW-`) | **0 bytes** (Only stores size metadata!) |

### 🖼️ The Museum Painting Analogy (.rodata vs Stack)
* **`const char* ptr = "Hello";`** — You hold a pointer to the original painting on the museum wall (`.rodata`). If you try to draw on it (`ptr[0] = 'M'`), the MMU tackles you (**Segmentation Fault**).
* **`char arr[] = "Hello";`** — The compiler photocopies the painting onto your desk (the **Stack** via `memcpy`). You can freely edit your copy (`arr[0] = 'M'`).

---

## 4. Heap Allocation & The 4 Bottlenecks of `malloc()`

Stack allocation takes $< 0.5\text{ ns}$ (1 instruction: `sub rsp, 64`). `malloc()` takes 50–500+ CPU cycles due to:
1. **Free-List Searching:** Traversing bucket/bin structures (Best-Fit/First-Fit) to find an available block.
2. **Multi-Thread Mutex Contention:** All threads share the same Heap memory arena; lock acquisition stalls CPU execution.
3. **Chunk Headers & Metadata:** Writing 8–16 byte bookkeeping headers before every returned pointer.
4. **Kernel Syscalls (`brk` / `mmap`):** Transitioning from User Mode (Ring 3) to Kernel Mode (Ring 0) to expand virtual memory bounds.

---

## 5. The `mmap` Segment: The 4 Critical Jobs

1. **Shared Libraries:** Maps `libc.so`, Python dynamic extensions (`.so`), and system frameworks so multiple processes share the same physical code pages in RAM.
2. **Large Heap Allocations (`malloc(>128 KB)`):** Bypasses `brk` and uses `mmap()` so the OS can immediately reclaim RAM upon `free()`, preventing heap fragmentation.
3. **Memory-Mapped Files:** Maps multi-gigabyte files directly into memory, allowing array-like indexing without `read()` syscall overhead.
4. **Inter-Process Shared Memory (`shm_open` + `mmap`):** Enables ultra-fast zero-copy IPC between processes.

---

## 6. RAII Demystified (Resource Acquisition Is Initialization)

> [!IMPORTANT]
> **The Purest Definition of RAII:**  
> RAII is simply wrapping a Heap resource inside a **Stack variable**.

```cpp
template <typename T>
class UniquePtr {
    T* raw_ptr; // 8-byte pointer on the STACK
public:
    UniquePtr(T* ptr) : raw_ptr(ptr) {}
    ~UniquePtr() { if (raw_ptr) delete raw_ptr; } // Runs deterministically at '}'!
    T* operator->() { return raw_ptr; }
};
```

### What Happens at Scope Exit `}`:
1. **The Compiler Injects the Destructor:** The compiler emits an explicit call to `p.~UniquePtr()`.
2. **The Destructor Executes `delete raw_ptr`:** The heap memory is instantly released.
3. **The Stack Pops:** The CPU executes `add rsp, 8`, discarding the stack variable.
4. **Stack Unwinding:** If an exception is thrown (`throw`), the runtime walks the `.eh_frame` table, calling destructors for every stack frame on the way up, guaranteeing zero memory leaks.

> [!WARNING]
> **Why Raw Pointers Leak Memory:**
> ```cpp
> int* ptr = new int(42); // ptr is just a raw 8-byte integer on the stack
> } // Stack pops ptr, but NO DESTRUCTOR EXISTS to call 'delete'! Memory is orphaned forever.
> ```
