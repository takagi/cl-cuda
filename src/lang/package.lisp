#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda.lang
  (:use :cl :cl-cuda.driver-api)
  (:export ;; Built-in Vector Types
           :float3 :make-float3
           :float3-p
           :float3-x :float3-y :float3-z :float3-=
           :x :y :z
           :float4 :make-float4
           :float4-p
           :float4-x :float4-y :float4-z :float4-w :float4-=
           :x :y :z :w
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
           :type-size
           :cffi-type-size
           :cffi-type
           :non-pointer-type-p
           :basic-type-p
           :vector-type-p
           :vector-type-elements
           :vector-type-selectors
           :array-type-p
           ;; Syntax
           :grid-dim-x :grid-dim-y :grid-dim-z
           :block-dim-x :block-dim-y :block-dim-z
           :block-idx-x :block-idx-y :block-idx-z
           :thread-idx-x :thread-idx-y :thread-idx-z
           :with-shared-memory :syncthreads
           ;; Built-in functions
           :rsqrtf
           :atomic-add
           :pointer
           :dot
           ;; Kernel definition
           :empty-kernel-definition
           :add-function-to-kernel-definition
           :remove-function-from-kernel-definition
           :add-macro-to-kernel-definition
           :remove-macro-from-kernel-definition
           :add-constant-to-kernel-definition
           :remove-constant-from-kernel-definition
           :add-symbol-macro-to-kernel-definition
           :remove-symbol-macro-from-kernel-definition
           :kernel-definition-function-exists-p
           :kernel-definition-macro-exists-p
           :kernel-definition-constant-exists-p
           :kernel-definition-symbol-macro-exists-p
           :kernel-definition-function-name
           :kernel-definition-function-c-name
           :kernel-definition-function-names
           :kernel-definition-function-return-type
           :kernel-definition-function-arguments
           :kernel-definition-function-argument-types
           :kernel-definition-function-body
           :kernel-definition-macro-name
           :kernel-definition-macro-names
           :kernel-definition-macro-arguments
           :kernel-definition-macro-body
           :kernel-definition-macro-expander
           :kernel-definition-constant-name
           :kernel-definition-constant-names
           :kernel-definition-constant-type
           :kernel-definition-constant-expression
           :kernel-definition-symbol-macro-name
           :kernel-definition-symbol-macro-names
           :kernel-definition-symbol-macro-expansion
           ;; Compiling
           :compile-kernel-definition
           ;; Macro expansion
           :expand-macro-1
           :expand-macro))
