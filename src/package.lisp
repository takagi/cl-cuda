#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda
  (:use :cl :cffi :alexandria :anaphora :cl-pattern)
  (:export :cu-result                   ; Types
           :cu-device
           :cu-context
           :cu-module
           :cu-function
           :cu-stream
           :cu-device-ptr
           :cu-init                     ; Functions
           :cu-device-get
           :cu-device-get-count
           :cu-device-compute-capability
           :cu-device-get-name
           :cu-ctx-create
           :cu-ctx-synchronize
           :cu-ctx-destroy
           :cu-mem-alloc
           :cu-mem-free
           :cu-memcpy-host-to-device
           :cu-memcpy-device-to-host
           :cu-module-load
           :cu-module-get-function
           :cu-launch-kernel
           :+cuda-success+              ; Constants
           :check-cuda-errors           ; Helpers
           :*show-messages*
           :with-cuda-context
           :with-cuda-memory-block
           :with-cuda-memory-blocks
           :defkernel                   ; defkernel
           :void :int :int* :float :float*
           :grid-dim-x :grid-dim-y :grid-dim-z
           :block-dim-x :block-dim-y :block-dim-z
           :block-idx-x :block-idx-y :block-idx-z
           :thread-idx-x :thread-idx-y :thread-idx-z
           :for :with-shared-memory :syncthreads
           ))
