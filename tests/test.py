import numpy as np
import fastpy as fp
import pytest
import time


@pytest.fixture(params=["float32", "float64", "int32", "int64", "uint32", "uint64"])
def input_array(request):
    dtype = request.param
    return np.random.uniform(0, 1, 10000).astype(dtype)


def test_fastpy_vs_numpy_correctness(input_array):
    fp_output_array = fp.cumsum(input_array)
    np_output_array = np.cumsum(input_array)
    np.testing.assert_allclose(fp_output_array, np_output_array)


def test_fastpy_performance(input_array, num_runs=100):
    start_time = time.time()
    for _ in range(num_runs):
        fp.cumsum(input_array)
    fastpy_time = (time.time() - start_time) * 1000 / num_runs

    start_time = time.time()
    for _ in range(num_runs):
        np.cumsum(input_array)
    numpy_time = (time.time() - start_time) * 1000 / num_runs

    assert fastpy_time * \
        0.95 < numpy_time, "Fastpy should on average be as fast as NumPy"


@pytest.mark.parametrize("invalid_input", [
    np.array([1, 2, 3]).reshape(-1, 1),
    np.array([1, 2, 3], dtype=object),
    np.array([1, 2, 3], dtype=np.float16),
    np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3)[:, 1],
    [1, 2, 3, 4, 5, 6],
])
def test_fastpy_invalid_arguments(invalid_input):
    with pytest.raises((Exception)):
        fp.cumsum(invalid_input)
