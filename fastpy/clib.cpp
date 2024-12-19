#include <Python.h>
#include <numpy/arrayobject.h>

#include <span>
#include <stdexcept>
#include <concepts>

// >>>>>> Some usefull wrappers >>>>>>s

template <typename T>
concept ValidType =
    std::same_as<std::remove_cv_t<T>, double> || std::same_as<std::remove_cv_t<T>, int>;

template <ValidType T>
int constexpr get_numpy_type_for_c_type() {
    if constexpr (std::is_same_v<std::remove_cv_t<T>, double>) {
        return NPY_DOUBLE;
    } else if constexpr (std::is_same_v<std::remove_cv_t<T>, int>) {
        return NPY_INT;
    } else {
        static_assert(false, "Unsupported type");
    }
}

template <ValidType T>
std::span<T> get_span_of_array(PyObject *array) {
    // Check if array is a NumPy array
    if (!PyArray_Check(array)) {
        throw std::invalid_argument("Input must be a NumPy array.");
    }

    PyArrayObject *py_array = (PyArrayObject *)array;

    if (PyArray_TYPE(py_array) != get_numpy_type_for_c_type<T>()) {
        throw std::invalid_argument("NumPy array must have different type.");
    }

    // Check if the array is contiguous
    if (!PyArray_ISCONTIGUOUS(py_array)) {
        throw std::invalid_argument("NumPy array must be contiguous.");
    }

    T *data = (T *)PyArray_DATA(py_array);
    size_t n = static_cast<size_t>(PyArray_SIZE(py_array));

    return std::span<T>(data, n);
}

template <ValidType T>
std::span<T> get_array(PyObject *&array, size_t size) {

    npy_intp npy_size = static_cast<npy_intp>(size);
    array = PyArray_SimpleNew(1, &npy_size, get_numpy_type_for_c_type<T>());
    PyArrayObject *py_array = (PyArrayObject *)array;
    T *data = (T *)PyArray_DATA(py_array);

    return std::span<T>(data, size);
}

// <<<<<<< Some usefull wrappers <<<<<<<

int cumsum_impl_wrapper(PyObject *input_array, PyObject *&output_array) {
    try {

        auto input_array_span = get_span_of_array<const double>(input_array);
        auto output_array_span = get_array<double>(output_array, input_array_span.size());

        int running_sum = 0;
        for (size_t i = 0; i < output_array_span.size(); ++i) {
            running_sum += input_array_span[i];
            output_array_span[i] = running_sum;
        }

        return 0;

    } catch (const std::exception &e) {

        PyErr_SetString(PyExc_Exception, e.what());

        return 1;
    }
}

// Boilerplate code to read input data and create output array
extern "C" {
static PyObject *cumsum(PyObject *self, PyObject *args) {
    PyObject *input_array, *output_array = NULL;

    if (!PyArg_ParseTuple(args, "O", &input_array)) {
        return NULL;
    }

    // Pass arguments to C++ where the answer will be written to an allocated output_array
    if (cumsum_impl_wrapper(input_array, output_array)) {
        return NULL;
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
    "clib",                         // Module name
    "High perfomance code in C++",  // Module docstring
    -1,                             // Size of the module state (-1 means global state)
    methods                         // The method table
};

// Module initialization function
PyMODINIT_FUNC PyInit_clib(void) {
    import_array();  // Initialize numpy C API
    return PyModule_Create(&module);
}
}
