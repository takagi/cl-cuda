#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-cuda.api)


;;;
;;; Definition of DEFKERNEL macro
;;;

(defun var-ptr (var)
  (symbolicate var "-PTR"))

(defun kernel-arg-names (arg-bindings)
  ;; ((a float*) (b float*) (c float*) (n int)) -> (a b c n)
  (mapcar #'car arg-bindings))

(defun kernel-arg-non-array-args (args)
  ;; ((x int) (y float3) (a float3*)) => ((x int) (y float3))
  (remove-if #'array-type-p args :key #'cadr))

(defun kernel-arg-array-args (args)
  ;; ((x int) (y float3) (a float3*)) => ((a float3*))
  (remove-if-not #'array-type-p args :key #'cadr))

(defun kernel-arg-ptr-type-binding (arg)
  ;; (x int) => (x-ptr :int), (y float3) => (y-ptr '(:struct float3))
  (destructuring-bind (var type) arg
    (if (vector-type-p type)
      `(,(var-ptr var) ',(cffi-type type))
      `(,(var-ptr var) ,(cffi-type type)))))

(defun kernel-arg-ptr-var-binding (arg)
  ;; (a float3*) => (a-ptr a)
  (let ((var (car arg)))
    (list (var-ptr var) var)))

(defun kernel-arg-pointer (arg)
  ;; (x int) => x-ptr, (y float3) => y-ptr, (a float3*) => a-ptr
  (var-ptr (car arg)))

(defun setf-basic-type-to-foreign-memory-form (var type)
  `(setf (cffi:mem-ref ,(var-ptr var) ,(cffi-type type)) ,var))

(defun setf-vector-type-to-foreign-memory-form (var type)
  `(setf ,@(loop for elm in (vector-type-elements type)
                 for selector in (vector-type-selectors type)
              append `((cffi:foreign-slot-value ,(var-ptr var) ',(cffi-type type) ',elm)
                       (,selector ,var)))))

(defun setf-to-foreign-memory-form (arg)
  (destructuring-bind (var type) arg
    (cond ((basic-type-p  type) (setf-basic-type-to-foreign-memory-form  var type))
          ((vector-type-p type) (setf-vector-type-to-foreign-memory-form var type))
          (t (error "invalid argument: ~A" arg)))))

(defun setf-to-argument-array-form (var arg n)
  `(setf (cffi:mem-aref ,var :pointer ,n) ,(kernel-arg-pointer arg)))


;; WITH-KERNEL-ARGUMENTS macro
;;  
;; Syntax:
;;
;; WITH-KERNEL-ARGUMENTS (kernel-argument ({(var type)}*) form*
;;
;; Description:
;;
;; WITH-KERNEL-ARGUMENTS macro is used only in expansion of DEFKERNEL
;; macro to prepare arguments which are to be passed to a CUDA kernel
;; function. This macro binds a given symbol to a cffi pointer which
;; refers an array whose elements are also cffi pointers. The contained
;; pointers refer arguments passed to a CUDA kernel function. In given
;; body forms, a CUDA kernel function will be launched using the binded
;; symbol to take its arguments.
;;
;; Example:
;;
;; (with-kernel-arguments (kargs ((x int) (y float3) (a float3*)))
;;   (launch-cuda-kernel-function kargs))
;;
;; Expanded:
;;
;; (cffi:with-foreign-objects ((x-ptr :int) (y-ptr '(:struct float3)))
;;   (setf (cffi:mem-ref x-ptr :int) x)
;;   (setf (cffi:slot-value y-ptr '(:struct float3) 'x) (float3-x y)
;;         (cffi:slot-value y-ptr '(:struct float3) 'y) (float3-y y)
;;         (cffi:slot-value y-ptr '(:struct float3) 'z) (float3-z y))
;;   (with-memory-block-device-ptrs ((a-ptr a))
;;     (cffi:with-foreign-object (kargs :pointer 3)
;;       (setf (cffi:mem-aref kargs :pointer 0) x-ptr)
;;       (setf (cffi:mem-aref kargs :pointer 1) y-ptr)
;;       (setf (cffi:mem-aref kargs :pointer 2) a-ptr)
;;       (launch-cuda-kernel-function kargs))))

(defmacro with-memory-block-device-ptrs (bindings &body body)
  (if bindings
      `(with-memory-block-device-ptr ,(car bindings)
         (with-memory-block-device-ptrs ,(cdr bindings)
           ,@body))
      `(progn
         ,@body)))

(defmacro with-kernel-arguments ((var args) &body body)
  (let ((non-array-args (kernel-arg-non-array-args args))
        (array-args     (kernel-arg-array-args args)))
    `(cffi:with-foreign-objects ,(mapcar #'kernel-arg-ptr-type-binding non-array-args)
       ,@(loop for arg in non-array-args
            collect (setf-to-foreign-memory-form arg))
       (with-memory-block-device-ptrs ,(mapcar #'kernel-arg-ptr-var-binding array-args)
         (cffi:with-foreign-object (,var :pointer ,(length args))
           ,@(loop for arg in   args
                   for i   from 0
                collect (setf-to-argument-array-form var arg i))
           ,@body)))))

(defmacro defkernel (name (return-type args) &body body)
  (with-gensyms (hfunc kargs)
    `(progn
       (kernel-manager-define-function *kernel-manager* ',name ',return-type ',args ',body)
       (defun ,name (,@(kernel-arg-names args) &key (grid-dim '(1 1 1)) (block-dim '(1 1 1)))
         (let ((,hfunc (ensure-kernel-function-loaded *kernel-manager* ',name)))
           (with-kernel-arguments (,kargs ,args)
             (destructuring-bind (grid-dim-x grid-dim-y grid-dim-z) grid-dim
             (destructuring-bind (block-dim-x block-dim-y block-dim-z) block-dim
               (cu-launch-kernel (cffi:mem-aref ,hfunc 'cu-function)
                                 grid-dim-x  grid-dim-y  grid-dim-z
                                 block-dim-x block-dim-y block-dim-z
                                 0 (cffi:null-pointer)
                                 ,kargs (cffi:null-pointer))))))))))


;;;
;;; Definition of DEFKERNELMACRO macro
;;;

(defmacro defkernelmacro (name args &body body)
  (with-gensyms (form-body)
    `(kernel-manager-define-macro *kernel-manager* ',name ',args ',body
       (lambda (,form-body)
         (destructuring-bind ,args ,form-body
           ,@body)))))

(defun expand-macro-1 (form)
  (let ((def (kernel-definition *kernel-manager*)))
    (cl-cuda.lang:expand-macro-1 form def)))

(defun expand-macro (form)
  (let ((def (kernel-definition *kernel-manager*)))
    (cl-cuda.lang:expand-macro form def)))


;;;
;;; Definition of DEFKERNELCONST macro
;;;

(defmacro defkernelconst (name type exp)
  (declare (ignore type))
  `(kernel-manager-define-symbol-macro *kernel-manager* ',name ',exp))
