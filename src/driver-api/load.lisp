#|
  This file is a part of cl-cuda project.
  Copyright (c) 2014 Masayuki Takagi (kamonama@gmail.com)
|#


(in-package :cl-cuda.driver-api)


;;;
;;; Load CUDA library
;;;

(cffi:define-foreign-library libcuda
  (:darwin (:framework "CUDA"))
  (:unix (:or "libcuda.so" "libcuda64.so")))

(ignore-errors
  (cffi:use-foreign-library libcuda))
