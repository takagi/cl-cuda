#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda
  (:use :cl
        :cl-cuda.driver-api
        :cl-cuda.lang
        :cl-cuda.api)
  (:export ;; Built-in Vector types
           :float3 :make-float3
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
           ;; CURAND
           :curand-state-xorwow :curand-state-xorwow*
           :curand-init-xorwow :curand-uniform-float-xorwow
           :curand-uniform-double-xorwow
           ;; Types
           :void :bool :bool* :int :int*
           :float :float* :float3 :float3* :float4 :float4*
           :double :double* :double3 :double3* :double4 :double4*
           ;; Syntax
           :grid-dim-x :grid-dim-y :grid-dim-z
           :block-dim-x :block-dim-y :block-dim-z
           :block-idx-x :block-idx-y :block-idx-z
           :thread-idx-x :thread-idx-y :thread-idx-z
           :with-shared-memory :syncthreads
           ;; Built-in functions
           :rsqrtf
           :rsqrt
           :atomic-add
           :pointer
           :dot)
  (:shadowing-import-from :cl-cuda.api
                          :expand-macro
                          :expand-macro-1))

(cl-reexport:reexport-from :cl-cuda.driver-api :cl-cuda)
(cl-reexport:reexport-from :cl-cuda.api :cl-cuda)
