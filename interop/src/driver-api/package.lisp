#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-interop.driver-api
  (:use :cl :cl-reexport)
  (:export ;; Types
           :cu-graphics-resource
           ;; Enumerations
           :cu-graphics-register-flags
           :cu-graphics-register-flags-none
           :cu-graphics-register-flags-read-only
           :cu-graphics-register-flags-write-discard
           :cu-graphics-register-flags-surface-ldst
           :cu-graphics-register-flags-texture-gather
           :cu-graphics-map-resource-flags
           :cu-graphics-map-resource-flags-none
           :cu-graphics-map-resource-flags-read-only
           :cu-graphics-map-resource-flags-write-discard
           ;; Functions
           :cu-gl-ctx-create
           :cu-graphics-gl-register-buffer
           :cu-graphics-map-resources
           :cu-graphics-resource-get-mapped-pointer
           :cu-graphics-resource-set-map-flags
           :cu-graphics-unmap-resources
           :cu-graphics-unregister-resource)
  (:import-from :cl-cuda.driver-api
                :defcufun))
(in-package :cl-cuda-interop.driver-api)

(reexport-from :cl-cuda.driver-api)
