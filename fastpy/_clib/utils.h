#pragma once

#include <Python.h>
#include <numpy/arrayobject.h>

#include <span>
#include <stdexcept>

#define CALL_TEMPLATED_FUNC(func_name, dtype, ...)                       \
    do {                                                                 \
        switch (dtype) {                                                 \
            case NPY_INT64:                                              \
                func_name<int64_t>(__VA_ARGS__);                         \
                break;                                                   \
            case NPY_INT32:                                              \
                func_name<int32_t>(__VA_ARGS__);                         \
                break;                                                   \
            case NPY_INT16:                                              \
                func_name<int16_t>(__VA_ARGS__);                         \
                break;                                                   \
            case NPY_INT8:                                               \
                func_name<int8_t>(__VA_ARGS__);                          \
                break;                                                   \
            case NPY_UINT64:                                             \
                func_name<uint64_t>(__VA_ARGS__);                        \
                break;                                                   \
            case NPY_UINT32:                                             \
                func_name<uint32_t>(__VA_ARGS__);                        \
                break;                                                   \
            case NPY_UINT16:                                             \
                func_name<uint16_t>(__VA_ARGS__);                        \
                break;                                                   \
            case NPY_UINT8:                                              \
                func_name<uint8_t>(__VA_ARGS__);                         \
                break;                                                   \
            case NPY_FLOAT64:                                            \
                func_name<double>(__VA_ARGS__);                          \
                break;                                                   \
            case NPY_FLOAT32:                                            \
                func_name<float>(__VA_ARGS__);                           \
                break;                                                   \
            default:                                                     \
                throw std::invalid_argument("Unsupported NumPy dtype."); \
        }                                                                \
    } while (0)

template <typename T>
constexpr int get_numpy_type() {
    if constexpr (std::is_same_v<std::remove_cv_t<T>, double>) {
        return NPY_FLOAT64;
    } else if constexpr (std::is_same_v<std::remove_cv_t<T>, float>) {
        return NPY_FLOAT32;
    } else if constexpr (std::is_same_v<std::remove_cv_t<T>, int64_t>) {
        return NPY_INT64;
    } else if constexpr (std::is_same_v<std::remove_cv_t<T>, int32_t>) {
        return NPY_INT32;
    } else if constexpr (std::is_same_v<std::remove_cv_t<T>, int16_t>) {
        return NPY_INT16;
    } else if constexpr (std::is_same_v<std::remove_cv_t<T>, int8_t>) {
        return NPY_INT8;
    } else if constexpr (std::is_same_v<std::remove_cv_t<T>, uint64_t>) {
        return NPY_UINT64;
    } else if constexpr (std::is_same_v<std::remove_cv_t<T>, uint32_t>) {
        return NPY_UINT32;
    } else if constexpr (std::is_same_v<std::remove_cv_t<T>, uint16_t>) {
        return NPY_UINT16;
    } else if constexpr (std::is_same_v<std::remove_cv_t<T>, uint8_t>) {
        return NPY_UINT8;
    } else {
        static_assert(false, "Unsupported type.");
    }
}

template <typename T>
std::span<T> get_span_of_array(PyObject *array) {
    // Check if array is a NumPy array
    if (!PyArray_Check(array)) {
        throw std::invalid_argument("Input must be a NumPy array.");
    }

    PyArrayObject *py_array = (PyArrayObject *)array;

    if (PyArray_TYPE(py_array) != get_numpy_type<T>()) {
        throw std::invalid_argument(
            "NumPy array must have the same type as the template parameter.");
    }

    // Check if the array is 1-dimensional
    if (PyArray_NDIM(py_array) != 1) {
        throw std::invalid_argument("NumPy array must be 1-dimensional.");
    }

    // Ensure the array is C-contiguous and aligned
    if (!PyArray_CHKFLAGS(py_array, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)) {
        throw std::invalid_argument("NumPy array must be C-contiguous and properly aligned.");
    }

    T *data = (T *)PyArray_DATA(py_array);
    size_t n = static_cast<size_t>(PyArray_SIZE(py_array));

    return std::span<T>(data, n);
}

int get_array_dtype(PyObject *array) {
    // Check if array is a NumPy array
    if (!PyArray_Check(array)) {
        throw std::invalid_argument("Input must be a NumPy array.");
    }

    PyArrayObject *py_array = (PyArrayObject *)array;
    return PyArray_TYPE(py_array);
}

template <typename T>
std::span<T> get_array(PyObject **array, size_t size) {

    npy_intp npy_size = static_cast<npy_intp>(size);
    *array = PyArray_SimpleNew(1, &npy_size, get_numpy_type<T>());
    PyArrayObject *py_array = (PyArrayObject *)(*array);
    T *data = (T *)PyArray_DATA(py_array);

    return std::span<T>(data, size);
}

void set_exception_free_memory(PyObject **obj, const std::exception &e) {
    PyErr_SetString(PyExc_Exception, e.what());
    Py_CLEAR(*obj);
    *obj = NULL;
}
