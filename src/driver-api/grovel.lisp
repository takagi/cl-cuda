#+darwin (include "cuda/cuda.h")
#+linux (include "cuda.h")

(in-package :cl-cuda.driver-api)

(ctype cu-device-ptr "CUdeviceptr")
(ctype size-t "size_t")
