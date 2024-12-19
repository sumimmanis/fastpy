# `clib` - Template C++ library for Python designed for seamless  integration with NumPy

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

After installation, use the `clib` module in Python. The `cumsum` function is provided as an example:

```python
import numpy as np
import clib

input_array = np.array([1, 2, 3, 4, 5])

output_array = clib.cumsum(input_array)

print("Input Array:", input_array)
print("Cumulative Sum:", output_array)
```
