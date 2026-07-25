import time
import numpy as np

def time_call(fn, *args, n_calls: int = 10_000) -> float:
    """Returns median per-call time in microseconds."""
    for _ in range(10):  # warm-up: skip whatever's lazily initialized on first call
        fn(*args)

    samples = []
    for _ in range(n_calls):
        start = time.perf_counter()
        fn(*args)
        samples.append(time.perf_counter() - start)

    return float(np.median(samples)) * 1e6
