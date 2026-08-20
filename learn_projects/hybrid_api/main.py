import time
from fastapi import FastAPI
import engine
import array

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/compute_cpp")
def compute_cpp():
    start = time.time()
    total = engine.compute_heavy(20_000_000) # Heavy 3-second C++ job
    end = time.time()
    return {"language": "C++", "total": total, "time_seconds": end - start}

@app.get("/zero_copy_test")
def zero_copy_test():
    start = time.time()
    # Allocates ONE single 40 MB contiguous block of raw 4-byte floats in RAM:
    data = array.array('f', [1.5] * 10_000_000)
    
    # Hands the raw memory pointer (0x7FFF1000) directly to C++:
    engine.double_array(data)
    end = time.time()

    return {
        "language": "C++",
        "elements_processed": len(data),
        "first_element_after_doubling": data[0], # Changed from 1.5 to 3.0 in place!
        "time_seconds": end - start
    }