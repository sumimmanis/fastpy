from setuptools import setup, Extension
import numpy

# Define the extension module
module = Extension(
    'clib',
    sources=['fastpy/clib.cpp'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-std=c++20', '-O3']
)

setup(
    name='fastpy',
    version='1.0',
    description = "Template implementation of a faster np.cumsum that works with numpy.",
    packages=['fastpy'],  # Include the package directory
    ext_modules=[module],  # Build the C extension
    install_requires=['numpy'],  # Make sure numpy is installed
)
