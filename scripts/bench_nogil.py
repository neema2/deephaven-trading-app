
import sys
import os
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Force PYTHONPATH to include current dir
sys.path.append(os.getcwd())

# Configuration
NUM_SWAPS = int(os.environ.get("NUM_SWAPS", 1000))
THREADS = [1, 2, 4, 8] if sys.maxsize > 2**32 else [1, 2] # Adjusted for ARM cores

# Check for 3.14t
GIL_DISABLED = not getattr(sys, "_is_gil_enabled", lambda: True)()
print(f"\n{'='*60}")
print(f" PYTHON NO-GIL CONCURRENCY TEST")
print(f" Python: {sys.version.split()[0]} | GIL Disabled: {GIL_DISABLED}")
print(f" Swaps: {NUM_SWAPS:,}")
print(f"{'='*60}")

def heavy_math(size):
    # Simulate the work done in SkinnyEngineNumPy:
    # Multiple element-wise ops on large arrays
    a = np.random.uniform(0.01, 0.05, size)
    b = np.random.uniform(1.0, 30.0, size)
    c = np.random.uniform(0.1, 0.9, size)
    
    t0 = time.perf_counter()
    # A typical swap leg PV calculation approximation
    # NPV = sum( Weight * exp(-rate * time) )
    res = np.sum(c * np.exp(-a * b))
    duration = time.perf_counter() - t0
    return res, duration

def run_bench(num_threads):
    # Split the work into N chunks
    chunk_size = NUM_SWAPS // num_threads
    
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(heavy_math, chunk_size) for _ in range(num_threads)]
        results = [f.result() for f in futures]
    
    total_duration = (time.perf_counter() - t0) * 1000
    return total_duration

def main():
    # Warm up
    heavy_math(1000)
    
    print(f"\n{'Threads':<10} | {'Duration (ms)':>15} | {'Speedup':>10}")
    print("-" * 45)
    
    base_time = None
    for t in THREADS:
        # Run 3 times and take the best
        durations = [run_bench(t) for _ in range(3)]
        best_dur = min(durations)
        
        if base_time is None:
            base_time = best_dur
            speedup = 1.0
        else:
            speedup = base_time / best_dur
            
        print(f"{t:<10} | {best_dur:>15.2f} | {speedup:>9.2f}x")

    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()
