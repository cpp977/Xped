## Xped
Library for the manipulation of symmetric (block-sparse) tensors with arbitrary (**X**) amount of indices (**ped**).

## Status

|Builds  | Tests | 	Coverage |
|:-: | :-: | :-: |
| [![Builds](https://github.com/cpp977/Xped/workflows/Builds/badge.svg)](https://github.com/cpp977/Xped/actions)|[![Tests](https://github.com/cpp977/Xped/workflows/Tests/badge.svg)](https://github.com/cpp977/Xped/actions)|[![codecov](https://codecov.io/gh/cpp977/Xped/branch/master/graph/badge.svg?token=MRQLD834VO)](https://codecov.io/gh/cpp977/Xped)|

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Quickstart

[quickstart](https://github.com/cpp977/Xped/blob/3e65a9b532a2fa71e1c7091bb7585a10cf72b935/docs/snippets/quickstart.cpp#L1-L47 "quickstart")

## Build
1. Get the sources using:
`git clone --recurse-submodules ssh://git@github.com/cpp977/Xped`
2. Create build directory:
`mkdir /dir/for/build && cd /dir/for/build`
3. Configure with cmake:
`cmake --preset=<preset> /path/to/source/Xped`
This also installs dependencies via vcpkg so the first run takes several minutes.
4. Build tests:
`cd tests && make`
5. Run the tests:
`ctest`

To control the build, it is recomennded to choose a CMake preset so that several options are already set automatically.
The following follow the scheme `<compiler>-<backend>`. E.g. `gcc-eigen` uses the gnu c++ compiler and the Eigen backend.
Supported compilers are gcc, clang and msvc.

All build options can be seen in the following table.

### Build Options

| Option | Default | Description |
| --- | --- | --- |
| `XPED_BUILD_BENCHMARKS` | `ON` | Build the benchmarks. |
| `XPED_BUILD_CYCLOPS` | `OFF` | Build the cyclops library from source. |
| `XPED_BUILD_EXAMPLES` | `OFF` | Build the benchmarks. |
| `XPED_BUILD_TESTS` | `ON` | Build the tests. |
| `XPED_BUILD_TOOLS` | `ON` | Build the tools. |
| `XPED_COMPILED_LIB` | `OFF` | Configure the library as a compiled library. Long compile times. |
| `XPED_EFFICIENCY_MODEL` | `XPED_TIME_EFFICIENT` | Xped tries to be time efficient. |
| `XPED_ENABLE_BUILD_WITH_TIME_TRACE` | `OFF` | Enable -ftime-trace to generate time tracing .json files on clang |
| `XPED_ENABLE_CCACHE` | `OFF` | Enable a compiler cache if available |
| `XPED_ENABLE_CLANG_FORMAT` | `ON` | Enable clang-format target. |
| `XPED_ENABLE_CLANG_TIDY` | `OFF` | Enable static analysis with clang-tidy |
| `XPED_ENABLE_COVERAGE` | `OFF` | Enable coverage reporting for gcc/clang |
| `XPED_ENABLE_CPPCHECK` | `OFF` | Enable static analysis with cppcheck |
| `XPED_ENABLE_DOXYGEN` | `OFF` | Enable doxygen doc builds of source |
| `XPED_ENABLE_INCLUDE_WHAT_YOU_USE` | `OFF` | Enable static analysis with include-what-you-use |
| `XPED_ENABLE_IPO` | `OFF` | Enable Interprocedural Optimization, aka Link Time Optimization (LTO) |
| `XPED_ENABLE_LRU_CACHE` | `OFF` | Use lru cache library from github. |
| `XPED_ENABLE_SANITIZER_ADDRESS` | `OFF` | Enable address sanitizer |
| `XPED_ENABLE_SANITIZER_LEAK` | `OFF` | Enable leak sanitizer |
| `XPED_ENABLE_SANITIZER_MEMORY` | `OFF` | Enable memory sanitizer |
| `XPED_ENABLE_SANITIZER_THREAD` | `OFF` | Enable thread sanitizer |
| `XPED_ENABLE_SANITIZER_UNDEFINED_BEHAVIOR` | `OFF` | Enable undefined behavior sanitizer |
| `XPED_LAPACKE` | `/usr/lib/x86_64-linux-gnu/liblapacke.so` | Path to a library. |
| `XPED_LOG_LEVEL` | `SPDLOG_LEVEL_CRITICAL` | Compile time log level. |
| `XPED_MATRIX_LIB` | `Eigen` | Used matrix library for plain tensor operations. |
| `XPED_OPTIM_LIB` | `ceres` | Used library for nonlinear gradient-based optimization. |
| `XPED_PEDANTIC_ASSERTS` | `OFF` | Enables rigorous assertions for tensor operations. |
| `XPED_STORAGE` | `Contiguous` | Used storage for Xped::Tensor. |
| `XPED_TENSOR_LIB` | `Eigen` | Used tensor library for plain tensor operations. |
| `XPED_USE_AD` | `ON` | Use automatic differentiation (AD) with Xped Tensors. |
| `XPED_USE_BLAS` | `ON` | Enable blas linking. |
| `XPED_USE_LAPACK` | `ON` | Enable lapack linking. |
| `XPED_USE_LIBCXX` | `OFF` | Use libc++ from llvm. |
| `XPED_USE_MKL` | `OFF` | Enable use of intel math kernel library (MKL). |
| `XPED_USE_MPI` | `OFF` | Enable message parsing interface (mpi) parallelization |
| `XPED_USE_NLO` | `ON` | Use nonlinear optimization algorithms. |
| `XPED_USE_OPENMP` | `ON` | Enable openmp parallelization |
| `XPED_USE_SCALAPACK` | `OFF` | Enable scalapack linking (only useful for MPI programs). |
| `XPED_VECTOR_LIB` | `Eigen` | Used matrix library for plain tensor operations. |

