#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage :cl-cuda.lang
  (:use :cl
        :cl-reexport))
(in-package :cl-cuda.lang)

;; reexport symbols of data structures cl-cuda provides
(reexport-from :cl-cuda.lang.data
               :include '(;; Float3
                          :float3
                          :make-float3
                          :float3-x
                          :float3-y
                          :float3-z
                          :float3-p
                          :float3-=
                          :with-float3
                          ;; Float4
                          :float4
                          :make-float4
                          :float4-x
                          :float4-y
                          :float4-z
                          :float4-w
                          :float4-p
                          :float4-=
                          :with-float4
                          ;; Double3
                          :double3
                          :make-double3
                          :double3-x
                          :double3-y
                          :double3-z
                          :double3-p
                          :double3-=
                          :with-double3
                          ;; Double4
                          :double4
                          :make-double4
                          :double4-x
                          :double4-y
                          :double4-z
                          :double4-w
                          :double4-p
                          :double4-=
                          :with-double4))

;; reexport symbols of cl-cuda types
(reexport-from :cl-cuda.lang.type
               :include '(:void
                          :bool
                          :int
                          :float
                          :double
                          :curand-state-xorwow
                          :float3
                          :float4
                          :double3
                          :double4
                          :bool*
                          :int*
                          :float*
                          :double*
                          :float3*
                          :float4*
                          :double3*
                          :double4*
                          :curand-state-xorwow*
                          :cffi-type))

;; reexport symbols of cl-cuda syntax except the ones exported
;; from COMMON-LISP package
(reexport-from :cl-cuda.lang.syntax
               :include '(:grid-dim-x :grid-dim-y :grid-dim-z
                          :block-dim-x :block-dim-y :block-dim-z
                          :block-idx-x :block-idx-y :block-idx-z
                          :thread-idx-x :thread-idx-y :thread-idx-z
                          :with-shared-memory
                          :set))

;; reexport symbols of cl-cuda built-in functions except the ones
;; exported from COMMON-LISP package
(reexport-from :cl-cuda.lang.built-in
               :include '(:xor
                          :shl
                          :shr
                          :rsqrt
                          :__exp
                          :__divide
                          :atomic-add
                          :pointer
                          :syncthreads
                          :double-to-int-rn
                          :dot
                          :curand-init-xorwow
                          :curand-uniform-float-xorwow
                          :curand-uniform-double-xorwow
                          :curand-normal-float-xorwow
                          :curand-normal-double-xorwow))
