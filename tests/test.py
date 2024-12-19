import numpy as np
import fastpy as fp
import time

input_array = np.random.uniform(0, 1, 10000)

start_time = time.time()
fp_output_array = fp.cumsum(input_array)
end_time = time.time()

print("fastpy time (ms): ", (end_time - start_time) * 1000)

start_time = time.time()
np_output_array = np.cumsum(input_array)
end_time = time.time()

print("numpy time (ms): ", (end_time - start_time) * 1000)

np.testing.assert_allclose(np_output_array, fp_output_array)
