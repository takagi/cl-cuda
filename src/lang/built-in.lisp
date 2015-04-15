#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda.lang.built-in
  (:use :cl
        :cl-cuda.lang.type)
  (:export ;; Built-in functions
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
           :curand-normal-double-xorwow
           ;; Interfaces
           :built-in-function-return-type
           :built-in-function-infix-p
           :built-in-function-c-name))
(in-package :cl-cuda.lang.built-in)


;;;
;;; Built-in functions
;;;

(defparameter +built-in-functions+
  '(;; arithmetic operators
    + (((int    int)    int    t   "+")
       ((float  float)  float  t   "+")
       ((float3 float3) float3 nil "float3_add")
       ((float4 float4) float4 nil "float4_add")
       ((double  double)  double  t   "+")
       ((double3 double3) double3 nil "double3_add")
       ((double4 double4) double4 nil "double4_add"))
    - (((int)           int    nil "int_negate")
       ((float)         float  nil "float_negate")
       ((float3)        float3 nil "float3_negate")
       ((float4)        float4 nil "float4_negate")
       ((double)  double  nil "double_negate")
       ((double3) double3 nil "double3_negate")
       ((double4) double4 nil "double4_negate")
       ((int    int)    int    t   "-")
       ((float  float)  float  t   "-")
       ((float3 float3) float3 nil "float3_sub")
       ((float4 float4) float4 nil "float4_sub")
       ((double  double)  double  t   "-")
       ((double3 double3) double3 nil "double3_sub")
       ((double4 double4) double4 nil "double4_sub"))
    * (((int    int)    int    t   "*")
       ((float  float)  float  t   "*")
       ((float3 float)  float3 nil "float3_scale")
       ((float  float3) float3 nil "float3_scale_flipped")
       ((float4 float)  float4 nil "float4_scale")
       ((float  float4) float4 nil "float4_scale_flipped")
       ((double  double)  double  t   "*")
       ((double3 double)  double3 nil "double3_scale")
       ((double  double3) double3 nil "double3_scale_flipped")
       ((double4 double)  double4 nil "double4_scale")
       ((double  double4) double4 nil "double4_scale_flipped"))
    / (((int)           int    nil "int_recip")
       ((float)         float  nil "float_recip")
       ((float3)        float3 nil "float3_recip")
       ((float4)        float4 nil "float4_recip")
       ((double)  double  nil "double_recip")
       ((double3) double3 nil "double3_recip")
       ((double4) double4 nil "double4_recip")
       ((int    int)    int    t   "/")
       ((float  float)  float  t   "/")
       ((float3 float)  float3 nil "float3_scale_inverted")
       ((float4 float)  float4 nil "float4_scale_inverted")
       ((double  double)  double  t   "/")
       ((double3 double)  double3 nil "double3_scale_inverted")
       ((double4 double)  double4 nil "double4_scale_inverted"))
    mod (((int    int)    int    t   "%"))
    ;; relational operators
    =    (((int   int)   bool t "==")
          ((float float) bool t "==")
          ((double double) bool t "=="))
    /=   (((int   int)   bool t "!=")
          ((float float) bool t "!=")
          ((double double) bool t "!="))
    <    (((int   int)   bool t "<")
          ((float float) bool t "<")
          ((double double) bool t "<"))
    >    (((int   int)   bool t ">")
          ((float float) bool t ">")
          ((double double) bool t ">"))
    <=   (((int   int)   bool t "<=")
          ((float float) bool t "<=")
          ((double double) bool t "<="))
    >=   (((int   int)   bool t ">=")
          ((float float) bool t ">=")
          ((double double) bool t ">="))
    ;; logical operators
    not  (((bool) bool nil "!"))
    ;; mathematical functions
    exp  (((float) float nil "expf")
          ((double) double nil "exp"))
    log  (((float) float nil "logf")
          ((double) double nil "log"))
    expt   (((float float) float nil "powf")
            ((double double) double nil "pow"))
    sin  (((float) float nil "sinf")
          ((double) double nil "sin"))
    cos  (((float) float nil "cosf")
          ((double) double nil "cos"))
    tan  (((float) float nil "tanf")
          ((double) double nil "tan"))
    sinh  (((float) float nil "sinhf")
           ((double) double nil "sinh"))
    cosh  (((float) float nil "coshf")
           ((double) double nil "cosh"))
    tanh  (((float) float nil "tanhf")
           ((double) double nil "tanh"))
    rsqrt (((float) float nil "rsqrtf")
           ((double) double nil "rsqrt"))
    sqrt   (((float) float nil "sqrtf")
            ((double) double nil "sqrt"))
    floor  (((float) int   nil "floorf")
            ((double) int   nil "floor"))
    ;; mathematical intrinsics
    ;;
    ;; If there is no double version, then fall back on a correct but
    ;; slow implementation.
    __exp    (((float) float nil "__expf")
              ((double) double nil "exp"))
    __divide (((float float) float nil "__fdividef")
              ((double double) double t "/"))
    ;; atomic functions
    atomic-add (((int* int) int nil "atomicAdd"))
    ;; address-of operator
    pointer (((int)   int*   nil "&")
             ((float) float* nil "&")
             ((double) double* nil "&")
             ((curand-state-xorwow) curand-state-xorwow* nil "&"))
    ;; built-in vector constructor
    float3 (((float float float) float3 nil "make_float3"))
    float4 (((float float float float) float4 nil "make_float4"))
    double3 (((double double double) double3 nil "make_double3"))
    double4 (((double double double double) double4 nil "make_double4"))
    ;; Synchronization functions
    syncthreads ((() void nil "__syncthreads"))
    ;; type casting intrinsics
    double-to-int-rn (((double) int nil "__double2int_rn"))
    ;; linear algebraic operators
    dot (((float3 float3) float nil "float3_dot")
         ((float4 float4) float nil "float4_dot")
         ((double3 double3) double nil "double3_dot")
         ((double4 double4) double nil "double4_dot"))
    ;; CURAND operations
    ;; It's :UNSIGNED-LONG-LONG, but this wrapper function only
    ;; supports INT.
    curand-init-xorwow (((int int int curand-state-xorwow*) void nil
                         "curand_init_xorwow"))
    curand-uniform-float-xorwow (((curand-state-xorwow*) float nil
                                  "curand_uniform_float_xorwow"))
    curand-uniform-double-xorwow (((curand-state-xorwow*) double nil
                                   "curand_uniform_double_xorwow"))
    curand-normal-float-xorwow (((curand-state-xorwow*) float nil
                                 "curand_normal_float_xorwow"))
    curand-normal-double-xorwow (((curand-state-xorwow*) double nil
                                  "curand_normal_double_xorwow"))))

(defun inferred-function-candidates (name)
  (or (getf +built-in-functions+ name)
      (error "The function ~S is undefined." name)))

(defun inferred-function (name argument-types)
  (let ((candidates (inferred-function-candidates name)))
    (or (assoc argument-types candidates :test #'equal)
        (error "The function ~S is undefined." name))))

(defun built-in-function-return-type (name argument-types)
  (cadr (inferred-function name argument-types)))

(defun built-in-function-infix-p (name argument-types)
  (caddr (inferred-function name argument-types)))

(defun built-in-function-c-name (name argument-types)
  (cadddr (inferred-function name argument-types)))
