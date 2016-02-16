#|
  This file is a part of cl-cuda project.
  Copyright (c) 2014 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-cuda-interop.driver-api)


;;;
;;; Include CUDA header file
;;;

#+darwin (include "cuda/cuda.h")
#-darwin (include "cuda.h")


;;;
;;; Types
;;;

(ctype cu-graphics-resource "CUgraphicsResource")
