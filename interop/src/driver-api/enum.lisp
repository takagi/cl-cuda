#|
  This file is a part of cl-cuda project.
  Copyright (c) 2014 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-cuda-interop.driver-api)


;;;
;;; Enumerations
;;;

(cl-cuda.driver-api::defcuenum cu-graphics-register-flags
  (:cu-graphics-register-flags-none           #X0)
  (:cu-graphics-register-flags-read-only      #X1)
  (:cu-graphics-register-flags-write-discard  #X2)
  (:cu-graphics-register-flags-surface-ldst   #X4)
  (:cu-graphics-register-flags-texture-gather #X8))

(cl-cuda.driver-api::defcuenum cu-graphics-map-resource-flags
  (:cu-graphics-map-resource-flags-none          #X0)
  (:cu-graphics-map-resource-flags-read-only     #X1)
  (:cu-graphics-map-resource-flags-write-discard #X2))
