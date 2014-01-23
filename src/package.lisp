#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda
  (:use :cl :alexandria :anaphora :cl-pattern)
  (:export :*nvcc-options*              ; configuration
           :*tmp-path*
           :*show-messages*
           :with-cuda-context           ; CUDA context
           :init-cuda-context
           :release-cuda-context
           :synchronize-context
           :cuda-initialized-p
           :cuda-available-p
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
           :create-timer                ; Timer
           :destroy-timer
           :with-timer
           :start-timer
           :stop-and-synchronize-timer
           :get-elapsed-time
           :float3 :make-float3         ; Built-in Vector Types
           :float3-p
           :float3-x :float3-y :float3-z :float3-=
           :float4 :make-float4
           :float4-p
           :float4-x :float4-y :float4-z :float4-w :float4-=
           :double3 :make-double3
           :double3-p
           :double3-x :double3-y :double3-z :double3-=
           :double4 :make-double4
           :double4-p
           :double4-x :double4-y :double4-z :double4-w :double4-=
           :curand-state-xorwow :curand-state-xorwow*   ; Curand
           :curand-init-xorwow :curand-uniform-float-xorwow
           :curand-uniform-double-xorwow
           :defkernel :defkernelmacro :defkernelconst   ; Kernel Description Language
           :void :bool :bool* :int :int*
           :float :float* :float3 :float3* :float4 :float4*
           :double :double* :double3 :double3* :double4 :double4*
           :grid-dim-x :grid-dim-y :grid-dim-z
           :block-dim-x :block-dim-y :block-dim-z
           :block-idx-x :block-idx-y :block-idx-z
           :thread-idx-x :thread-idx-y :thread-idx-z
           :with-shared-memory :syncthreads
           :rsqrtf                      ; Built-in functions
           :rsqrt
           :atomic-add
           :pointer
           :dot
           :print-kernel-manager        ; Utilities for the default kernel manager
           :clear-kernel-manager
           :expand-macro :expand-macro-1
           ))
