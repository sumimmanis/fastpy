# `fastpy` - Template C++ library for Python designed for seamless  integration with NumPy

This repository provides a template for creating high-performance C++ functions for Python using the NumPy C API. The `np.cumsum` function is included as an example.

## Features
- Template for creating C++ functions callable from Python.
- Example implementation of `np.cumsum` for reference.

## Installation

Install the C++ extension using `pip`:

```bash
pip install .
```

## Example

After installation, use the `fastpy` module in Python. The `cumsum` function is provided as an example (`dtype=np.double` is hardcoded in the binary so the conversion is needed):

```python
import numpy as np
import fastpy

input_array = np.array([1, 2, 3, 4, 5], dtype=np.double)

output_array = fastpy.cumsum(input_array)

print("Input Array:", input_array)
print("Cumulative Sum:", output_array)
```

By the way, this code runs five times faster than `np.cumsum`.
