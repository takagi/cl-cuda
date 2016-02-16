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
;;; Enumerations
;;;

(cenum (cu-graphics-register-flags :define-constants t)
  ((:cu-graphics-register-flags-none "CU_GRAPHICS_REGISTER_FLAGS_NONE"))
  ((:cu-graphics-register-flags-read-only "CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY"))
  ((:cu-graphics-register-flags-write-discard "CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD"))
  ((:cu-graphics-register-flags-surface-ldst "CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST"))
  ((:cu-graphics-register-flags-texture-gather "CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER")))

(cenum (cu-graphics-map-resource-flags :define-constants t)
  ((:cu-graphics-map-resource-flags-none "CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE"))
  ((:cu-graphics-map-resource-flags-read-only "CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY"))
  ((:cu-graphics-map-resource-flags-write-discard "CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD")))
