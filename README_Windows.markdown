# Prerequisites

- NVIDIA CUDA-enabled GPU
- MinGW GCC
- Visual Studio (C/C++ development)
- CUDA Toolkit, CUDA Drivers and CUDA SDK need to be installed
- freeglut

## Environment variables

### C_INCLUDE_PATH

set CUDA include path. 
(ex: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include)

### PATH

add following directory paths:

- MinGW bin directory
- MSVC bin directory (contains cl.exe)
- freeglut dll directory

# Installation

You can install cl-cuda via quicklisp.

    > (ql:quickload :cl-cuda)

## Verification environments (Windows)

#### Environment
* Windows 10 Pro (Version 1903 OS Build 18362.356)
* GeForce GTX 1080
* Visual Studio Community 2019
* CUDA 10.1
* gcc (x86_64-posix-sjlj-rev0, Built by MinGW-W64 project) 8.1.0
* freeglut 3.0.0 for MinGW
* SBCL 1.4.14 (installed via roswell 19.08.11.101(0d8e06d))
* 2 tests failed, all examples work

[testing script and result](https://gist.github.com/sgr/242b70859c9afb39ab83a1a7d5feeea6)

