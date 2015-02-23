#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage :cl-cuda.api.defkernel
  (:use :cl
        :cl-cuda.driver-api
        :cl-cuda.lang.syntax
        :cl-cuda.lang.type
        :cl-cuda.api.kernel-manager
        :cl-cuda.api.memory)
  (:export :defkernel
           :defkernelmacro
           :expand-macro-1
           :expand-macro
           :defkernel-symbol-macro)
  (:shadow :expand-macro-1
           :expand-macro)
  (:import-from :alexandria
                :format-symbol
                :with-gensyms))
(in-package :cl-cuda.api.defkernel)


;;;
;;; DEFKERNEL - argument helper
;;;

(defun argument-vars (arguments)
  (mapcar #'argument-var arguments))

(defun argument-var-ptr (argument)
  (let* ((var (argument-var argument))
         (package (symbol-package var)))
    (format-symbol package "~A-PTR" var)))

(defun argument-cffi-type (argument)
  (cffi-type (argument-type argument)))


;;;
;;; DEFKERNEL
;;;

(defun ptr-binding (argument)
  ;; (x int)     => (x-ptr :int)
  ;; (y float3)  => (y-ptr '(:struct float3))
  ;; (a float3*) => (a-ptr 'cu-device-ptr)
  (let ((var-ptr (argument-var-ptr argument))
        (cffi-type (argument-cffi-type argument)))
    `(,var-ptr ',cffi-type)))

(defun setf-to-foreign-object-form (argument)
  (let ((var (argument-var argument))
        (type (argument-type argument))
        (var-ptr (argument-var-ptr argument))
        (cffi-type (argument-cffi-type argument)))
    (if (array-type-p type)
        `(setf (cffi:mem-ref ,var-ptr ',cffi-type)
               (if (memory-block-p ,var)
                   (memory-block-device-ptr ,var)
                   ,var))
        `(setf (cffi:mem-ref ,var-ptr ',cffi-type) ,var))))

(defun setf-to-argument-array-form (var argument i)
  (let ((var-ptr (argument-var-ptr argument)))
    `(setf (cffi:mem-aref ,var :pointer ,i) ,var-ptr)))

(defmacro with-launching-arguments ((var arguments) &body body)
  ;; WITH-LAUNCHING-ARGUMENTS macro is used only in expansion of DEFKERNEL
  ;; macro to prepare arguments which are to be passed to a CUDA kernel
  ;; function. This macro binds the given symbol VAR to a cffi pointer which
  ;; refers an array whose elements are also cffi pointers. The contained
  ;; pointers refer arguments passed to a CUDA kernel function. In the given
  ;; body forms, a CUDA kernel function will be launched using the bound
  ;; symbol to take its arguments
  ;;
  ;; Example:
  ;;
  ;; (with-launching-arguments (kargs ((x int) (y float3) (a float3*)))
  ;;   (launch-cuda-kernel-function kargs))
  ;;
  ;; Expanded:
  ;;
  ;; (cffi:with-foreign-objects ((x-ptr :int) (y-ptr '(:struct float3)) (a-ptr 'cu-device-ptr))
  ;;   (setf (cffi:mem-ref x-ptr :int) x)
  ;;   (setf (cffi:mem-ref y-ptr '(:struct float3)) y)
  ;;   (setf (cffi:mem-ref a-ptr 'cu-device-ptr) a)
  ;;   (cffi:with-foreign-object (kargs :pointer 3)
  ;;     (setf (cffi:mem-aref kargs :pointer 0) x-ptr)
  ;;     (setf (cffi:mem-aref kargs :pointer 1) y-ptr)
  ;;     (setf (cffi:mem-aref kargs :pointer 2) a-ptr)
  ;;     (launch-cuda-kernel-function kargs)))
  ;;
  (let ((ptr-bindings (mapcar #'ptr-binding arguments)))
    `(cffi:with-foreign-objects ,ptr-bindings
       ,@(loop for argument in arguments
            collect (setf-to-foreign-object-form argument))
       (cffi:with-foreign-object (,var :pointer ,(length arguments))
         ,@(loop for argument in arguments
                 for i from 0
              collect (setf-to-argument-array-form var argument i))
         ,@body))))

(defmacro defkernel (name (return-type arguments) &body body)
  (with-gensyms (hfunc kargs)
    `(progn
       (kernel-manager-define-function *kernel-manager* ',name ',return-type ',arguments ',body)
       (defun ,name (,@(argument-vars arguments) &key (grid-dim '(1 1 1)) (block-dim '(1 1 1)))
         (let ((,hfunc (ensure-kernel-function-loaded *kernel-manager* ',name)))
           (with-launching-arguments (,kargs ,arguments)
             (destructuring-bind (grid-dim-x grid-dim-y grid-dim-z) grid-dim
             (destructuring-bind (block-dim-x block-dim-y block-dim-z) block-dim
               (cu-launch-kernel ,hfunc
                                 grid-dim-x  grid-dim-y  grid-dim-z
                                 block-dim-x block-dim-y block-dim-z
                                 0 cl-cuda.api.context:*cuda-stream*
                                 ,kargs (cffi:null-pointer))))))))))


;;;
;;; DEFKERNELMACRO
;;;

(defmacro defkernelmacro (name arguments &body body)
  `(kernel-manager-define-macro *kernel-manager* ',name ',arguments ',body))

(defun expand-macro-1 (form)
  (cl-cuda.api.kernel-manager:expand-macro-1 form *kernel-manager*))

(defun expand-macro (form)
  (cl-cuda.api.kernel-manager:expand-macro form *kernel-manager*))


;;;
;;; DEFKERNEL-SYMBOL-MACRO
;;;

(defmacro defkernel-symbol-macro (name expansion)
  `(kernel-manager-define-symbol-macro *kernel-manager* ',name ',expansion))
