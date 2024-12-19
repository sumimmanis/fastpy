import numpy as np
import fastpy as fp
import tracemalloc


def check_memory_leak():
    tracemalloc.start()

    input_array = np.random.uniform(0, 1, 10000)

    fp_output_array = fp.cumsum(input_array)
    np_output_array = np.cumsum(input_array)

    np.testing.assert_allclose(np_output_array, fp_output_array)

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024:.2f} KB")
    print(f"Peak memory usage: {peak / 1024:.2f} KB")

    tracemalloc.stop()


for i in range(5):
    check_memory_leak()
