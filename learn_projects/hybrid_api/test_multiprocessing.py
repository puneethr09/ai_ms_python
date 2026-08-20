import time
import array
from multiprocessing import Process, Queue

# Worker function running in a separate OS Process:
def worker(q_in, q_out):
    data = q_in.get() # <── Must deserialize (unpickle) 40 MB!
    # Double the first element
    data[0] *= 2.0
    q_out.put(data)   # <── Must serialize (pickle) 40 MB again!

def main():
    print("📦 Creating 40 MB contiguous array in RAM...")
    data = array.array('f', [1.5] * 10_000_000)

    q_in = Queue()
    q_out = Queue()

    print("🚀 Spawning a separate OS Process via multiprocessing...")
    start_total = time.time()
    
    p = Process(target=worker, args=(q_in, q_out))
    p.start()

    # The IPC Bottleneck: Sending 40 MB through a standard Python Queue
    t_send = time.time()
    q_in.put(data)
    print(f"   ⏱️ Time to send data across process boundary: {time.time() - t_send:.3f}s")

    result = q_out.get()
    p.join()
    
    print(f"✅ Total Time with Multiprocessing: {time.time() - start_total:.3f}s")
    print(f"   Result: first element is {result[0]}")

if __name__ == "__main__":
    main()
