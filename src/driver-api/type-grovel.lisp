#|
  This file is a part of cl-cuda project.
  Copyright (c) 2014 Masayuki Takagi (kamonama@gmail.com)
|#


(in-package :cl-cuda.driver-api)


;;;
;;; Include CUDA header file
;;;

#+darwin (include "cuda/cuda.h")
#+linux (include "cuda.h")


;;;
;;; Data types
;;;

(ctype cu-result "CUresult")
(ctype cu-device "CUdevice")
(ctype cu-context "CUcontext")
(ctype cu-module "CUmodule")
(ctype cu-function "CUfunction")
(ctype cu-stream "CUstream")
(ctype cu-device-ptr "CUdeviceptr")
(ctype cu-event "CUevent")
(ctype size-t "size_t")
