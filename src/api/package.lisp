#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda.api
  (:use :cl
        :cl-cuda.driver-api
        :cl-cuda.lang)
  (:export :*nvcc-options*              ; configuration
           :*tmp-path*
           :with-cuda-context           ; CUDA context
           :init-cuda-context
           :release-cuda-context
           :synchronize-context
           :with-memory-block           ; Memory Block
           :with-memory-blocks
           :alloc-memory-block
           :free-memory-block
           :memory-block-type
           :memory-block-length
           :memory-block-bytes
           :memory-block-element-bytes
           :memory-block-vertex-buffer-object
           :mem-aref
           :memcpy-host-to-device
           :memcpy-device-to-host
           :defkernel                   ; defkernel
           :defkernelmacro
           :defkernelconst
           :create-timer                ; Timer
           :destroy-timer
           :with-timer
           :start-timer
           :stop-and-synchronize-timer
           :get-elapsed-time
           :print-kernel-manager        ; Utilities for the default kernel manager
           :clear-kernel-manager
           :expand-macro
           :expand-macro-1)
  (:shadow :expand-macro
           :expand-macro-1)
  (:import-from :alexandria
                :unwind-protect-case
                :with-gensyms
                :symbolicate
                :hash-table-alist)
  (:import-from :anaphora
                :swhen
                :it))
