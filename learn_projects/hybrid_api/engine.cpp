#include <pybind11/pybind11.h>
#include <cstdint>
#include <chrono>
#include <thread>

namespace py = pybind11;

// Simulates a 3-second heavy calculation (e.g. generating LLM tokens)
uint64_t compute_heavy_cpp(int64_t n) {
    uint64_t total = 0;
    // Heavy loop that takes ~3 seconds of CPU grind
    for (int64_t j = 0; j < 300; ++j) {
        for (int64_t i = 0; i < n; ++i) {
            total += (i * i) ^ j;
        }
    }
    return total;
}

// py::buffer is pybind11's view into the C-Buffer Protocol (Py_buffer)
void double_array_zero_copy(py::buffer b) {
    // 1. Request the raw buffer metadata from Python:
    py::buffer_info info = b.request();
    
    // 2. Grab the direct memory pointer (0x7FFF1000) to the contiguous floats:
    float* raw_ptr = static_cast<float*>(info.ptr);
    size_t count = info.shape[0];

    // Step B: Drop the GIL ONLY for the bare-metal C++ loop!
    {
        py::gil_scoped_release release; // <── Unlocks GIL here
        for (size_t i = 0; i < count; ++i) {
            raw_ptr[i] *= 2.0f; // Pure pointer arithmetic in L1 cache!
        }
    } // <── Re-acquires GIL automatically here via RAII!
}


PYBIND11_MODULE(engine, m) {
    m.doc() = "C++ High Performance Math Engine";
    
    // pybind11 macro export:
    m.def("compute_heavy", &compute_heavy_cpp, "Heavy C++ computation with GIL released",py::call_guard<py::gil_scoped_release>()); // <── RELEASES THE GIL!

    m.def("double_array", &double_array_zero_copy, "Zero-copy array doubling using C-Buffer Protocol");
}
