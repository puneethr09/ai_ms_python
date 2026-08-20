# Level 4: Concurrency & Asynchronous Engines
> **Thread Pools, Event Loops, Modern C++20 `std::jthread`, and The GIL**

---

## 1. The Chef Kitchen Analogy

* **C++ Thread Pool (8 Chefs, 8 Stoves):** 8 independent chefs handle 8 cooking orders simultaneously across 8 CPU cores. True multi-core parallelism.
* **Python Asyncio Event Loop (1 Chef, 50 Timers):** 1 chef sets timers (`await`) for boiling water/baking and switches to chopping salad.
  * **The Trap:** If an order requires pure CPU math (chopping 10,000 carrots), the single chef is blocked. The entire kitchen freezes for all other customers!

---

## 2. Modern C++ Concurrency Evolution

```cpp
// C++11: std::thread (Low-level; crashes if destroyed without join/detach)
// C++11: std::async / std::future (Task-based parallelism)
// C++20: std::jthread (RAII auto-joining thread with cooperative cancellation)

#include <thread>
#include <chrono>

void worker(std::stop_token stoken) {
    while (!stoken.stop_requested()) {
        // Do heavy background math...
    }
}

int main() {
    std::jthread t(worker); 
    // Auto-joins deterministically on destruction when exiting scope!
}
```

---

## 3. Python's 3 Concurrency Models & The GIL

| Model | Underlying OS Mechanism | Memory Sharing | Best Used For |
| :--- | :--- | :--- | :--- |
| **`threading`** | Real OS Kernel Threads | Shared Heap, but locked to **1 core** by GIL | I/O-bound tasks (file downloads, DB queries) |
| **`multiprocessing`** | Separate OS Processes | Isolated Memory (Requires IPC serialization) | CPU-bound math in pure Python |
| **`asyncio`** | Single OS Thread (Event Loop via `epoll`/`kqueue`) | Single Thread Memory | Massive concurrent I/O (10,000+ idle sockets) |

---

## 4. The Web Server Revolution: Apache vs. NGINX & FastAPI

* **Apache (Thread-Per-Connection):** 10,000 connections = 10,000 OS threads = **80 GB of idle stack RAM** + massive context-switch thrashing.
* **NGINX / FastAPI (Non-Blocking Event Loop):** 1 worker thread per CPU core multiplexes thousands of active sockets via OS `epoll`/`kqueue` with zero stack overhead.
