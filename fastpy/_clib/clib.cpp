#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdexcept>

#include "utils.h"

template <typename T>
void cumsum_impl(PyObject *input_array, PyObject **output_array) {
    auto input_array_span = get_span_of_array<const T>(input_array);
    auto output_array_span = get_array<T>(output_array, input_array_span.size());

    T running_sum = 0;
    for (size_t i = 0; i < output_array_span.size(); ++i) {
        running_sum += input_array_span[i];
        output_array_span[i] = running_sum;
    }
}

void cumsum_impl_wrapper(PyObject *input_array, PyObject **output_array) {
    try {

        int dtype = get_array_dtype(input_array);
        CALL_TEMPLATED_FUNC(cumsum_impl, dtype, input_array, output_array);

    } catch (const std::exception &e) {
        set_exception_free_memory(output_array, e);
    }
}

extern "C" {
static PyObject *cumsum(PyObject *self, PyObject *args) {
    PyObject *input_array, *output_array = NULL;

    if (!PyArg_ParseTuple(args, "O", &input_array)) {
        return NULL;
    }

    // On failure sets python's API exception variable and sets output_array = NONE
    cumsum_impl_wrapper(input_array, &output_array);
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
