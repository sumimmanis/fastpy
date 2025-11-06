#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdexcept>

#include "utils.h"

template <typename T>
void cumsum_impl(PyObject *input_array, PyObject **output_array) {
    auto input_array_span = get_span_of_array<const T>(input_array); // Get input array as a span
    auto output_array_span = get_array<T>(output_array, input_array_span.size()); // Create output array and get it as a span

    T running_sum = 0;
    for (size_t i = 0; i < output_array_span.size(); ++i) {
        running_sum += input_array_span[i];
        output_array_span[i] = running_sum;
    }
}

extern "C" {
static PyObject *cumsum(PyObject *self, PyObject *args) {
    PyObject *input_array;
    PyObject *output_array = NULL;

    // Reads the input arguments: "O" means a single Python object is expected
    if (!PyArg_ParseTuple(args, "O", &input_array)) {
        return NULL;
    }

    try {
        // Determine the data type of the input array: all integer sizes and float32/float64 are supported
        int dtype = get_array_dtype(input_array);

        // Call the templated implementation based on the input array's data type
        CALL_TEMPLATED_FUNC(cumsum_impl, dtype, input_array, &output_array);

    } catch (const std::exception &e) {
        // On failure sets python's API exception variable and sets output_array = NONE
        set_exception_free_memory(&output_array, e);
    }
    return output_array;
}

// Method table for this extension
static PyMethodDef methods[] = {
    {"cumsum", cumsum, METH_VARARGS, "A faster implementation of np.cumsum"},
    {NULL, NULL, 0, NULL}  // Sentinel value indicating the end of the method table
};

// Module definition
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "clib",                                                              // Module name
    "High perfomance C++ implementation of some functions for python.",  // Module docstring
    -1,      // Size of the module state (-1 means global state)
    methods  // The method table
};

// Module initialization function
PyMODINIT_FUNC PyInit_clib(void) {
    import_array();  // Initialize numpy C API
    return PyModule_Create(&module);
}
}
