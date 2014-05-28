#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-cuda.api)


;;;
;;; Definition of with- macro for CUDA driver API
;;;

(defmacro with-cuda-context ((dev-id &key (interop nil)) &body body)
  `(progn
     (init-cuda-context ,dev-id :interop ,interop)
     (unwind-protect (progn ,@body)
       (release-cuda-context))))

(let (device context)
  
  (defun init-cuda-context (dev-id &key (interop nil))
    (when (or device context)
      (error "CUDA context is already initialized"))
    (unwind-protect-case ()
        (let ((flags 0))
          ;; allocate memory areas for a CUDA device and a CUDA context
          (setf device  (cffi:foreign-alloc 'cu-device  :initial-element 0)
                context (cffi:foreign-alloc 'cu-context :initial-element (cffi:null-pointer)))
          ;; initialized CUDA
          (cu-init 0)
          ;; get a CUDA device
          (cu-device-get device dev-id)
          ;; create a CUDA context
          (if (not interop)
              (cu-ctx-create    context flags (cffi:mem-ref device 'cu-device))
              (cu-gl-ctx-create context flags (cffi:mem-ref device 'cu-device))))
      (:abort (release-cuda-context))))
  
  (defun release-cuda-context ()
    (symbol-macrolet ((mem-ref-context (cffi:mem-ref context 'cu-context)))
      ;; unload kernel manager
      (kernel-manager-unload *kernel-manager*)
      ;; destroy a CUDA context if created
      (when (and context (not (cffi:null-pointer-p mem-ref-context)))
        (cu-ctx-destroy mem-ref-context)
        (setf mem-ref-context (cffi:null-pointer)))
      ;; free a memory pointer to a CUDA context if allocated
      (when context
        (cffi:foreign-free context)
        (setf context nil))
      ;; free a memory area for a CUDA device if allocated
      (when device
        (cffi:foreign-free device)
        (setf device nil))))
  
  (defun synchronize-context ()
    (cu-ctx-synchronize))
  
  (defun cuda-initialized-p ()
    (not (null context)))

  (defun cuda-available-p ()
    (let ((*show-messages* nil))
      (or (cuda-initialized-p)
          (cffi:with-foreign-objects ((device 'cu-device))
            (ignore-errors
             (cu-init 0)
             (cu-device-get device 0)
             t)))))
  
  (defun device-compute-capability ()
    (cffi:with-foreign-objects ((major :int)
                                (minor :int))
      (cu-device-compute-capability major minor (cffi:mem-ref device :int))
      (values (cffi:mem-ref major :int)
              (cffi:mem-ref minor :int)))))


;;;
;;; Definition of Memory Block
;;;

(defstruct memory-block-cuda
  cffi-ptr device-ptr type length)

(defstruct memory-block-interop
  cffi-ptr vertex-buffer-object graphic-resource-ptr type length)

(defun alloc-memory-block-cuda (type n)
  (let ((blk (make-memory-block-cuda)))
    (symbol-macrolet ((cffi-ptr   (memory-block-cuda-cffi-ptr   blk))
                      (device-ptr (memory-block-cuda-device-ptr blk))
                      (blk-type   (memory-block-cuda-type       blk))
                      (length     (memory-block-cuda-length     blk)))
      (unwind-protect-case ()
          (progn
            ;; allocate a memory block
            (setf cffi-ptr   (cffi:foreign-alloc (cffi-type type) :count n)
                  device-ptr (cffi:foreign-alloc 'cu-device-ptr :initial-element 0)
                  blk-type   type
                  length     n)
            ;; allocate device memory
            (cu-mem-alloc device-ptr (* n (type-size type)))
            ;; return a memory block
            blk)
        (:abort (free-memory-block-cuda blk))))))

(defun alloc-memory-block-interop (type n)
  (let ((blk (make-memory-block-interop)))
    (symbol-macrolet ((cffi-ptr (memory-block-interop-cffi-ptr             blk))
                      (vbo      (memory-block-interop-vertex-buffer-object blk))
                      (gres-ptr (memory-block-interop-graphic-resource-ptr blk))
                      (blk-type (memory-block-interop-type                 blk))
                      (length   (memory-block-interop-length               blk)))
      (unwind-protect-case ()
          (progn
            ;; allocate memory area
            (setf cffi-ptr (cffi:foreign-alloc (cffi-type type) :count n)
                  vbo      (car (gl:gen-buffers 1))
                  gres-ptr (cffi:foreign-alloc 'cu-graphics-resource :initial-element (cffi:null-pointer))
                  blk-type type
                  length   n)
            ;; create and initialize a buffer object's data store
            (gl:bind-buffer :array-buffer vbo)
            (let ((ary (gl:alloc-gl-array (cffi-type type) n)))
              (unwind-protect (gl:buffer-data :array-buffer :dynamic-draw ary)
                (gl:free-gl-array ary)))
            (gl:bind-buffer :array-buffer 0)
            ;; register a buffer object accessed through CUDA
            (cu-graphics-gl-register-buffer gres-ptr vbo cu-graphics-register-flags-none)
            ;; return a memory block
            blk)
        (:abort (free-memory-block-interop blk))))))

(defun alloc-memory-block (type n &key (interop nil))
  (unless (non-pointer-type-p type)
    (error "invalid type: ~A" type))
  (if interop
      (alloc-memory-block-interop type n)
      (alloc-memory-block-cuda    type n)))

(defun free-memory-block-cuda (blk)
  (symbol-macrolet ((cffi-ptr           (memory-block-cuda-cffi-ptr   blk))
                    (device-ptr         (memory-block-cuda-device-ptr blk))
                    (mem-ref-device-ptr (cffi:mem-ref device-ptr 'cu-device-ptr)))
    ;; free device memory
    (when (and device-ptr (/= mem-ref-device-ptr 0))
      (cu-mem-free mem-ref-device-ptr)
      (setf mem-ref-device-ptr 0))
    ;; free a pointer to device memory
    (when device-ptr
      (cffi:foreign-free device-ptr)
      (setf device-ptr nil))
    ;; free a pointer to host memory
    (when cffi-ptr
      (cffi:foreign-free cffi-ptr)
      (setf cffi-ptr nil))))

(defun free-memory-block-interop (blk)
  (symbol-macrolet ((gres-ptr         (memory-block-graphic-resource-ptr blk))
                    (vbo              (memory-block-vertex-buffer-object blk))
                    (cffi-ptr         (memory-block-cffi-ptr             blk))
                    (mem-ref-gres-ptr (cffi:mem-ref gres-ptr 'cu-graphics-resource)))
    (when (and gres-ptr (not (cffi:null-pointer-p mem-ref-gres-ptr)))
      ;; unregister a buffer object accessed through CUDA
      (cu-graphics-unregister-resource mem-ref-gres-ptr)
      (setf mem-ref-gres-ptr (cffi:null-pointer)))
    (when gres-ptr
      ;; free a pointer to a graphics resource
      (cffi:foreign-free gres-ptr)
      (setf gres-ptr nil))
    (when vbo
      ;; delete a buffer object
      (gl:delete-buffers (list vbo))
      (setf vbo nil))
    (when cffi-ptr
      ;; free a pointer to host memory area
      (cffi:foreign-free cffi-ptr)
      (setf cffi-ptr nil))))

(defun free-memory-block (blk)
  (if (memory-block-interop-p blk)
      (free-memory-block-interop blk)
      (free-memory-block-cuda    blk)))

(defun memory-block-cffi-ptr (blk)
  (if (memory-block-interop-p blk)
      (memory-block-interop-cffi-ptr blk)
      (memory-block-cuda-cffi-ptr    blk)))

(defun (setf memory-block-cffi-ptr) (val blk)
  (if (memory-block-interop-p blk)
      (setf (memory-block-interop-cffi-ptr blk) val)
      (setf (memory-block-cuda-cffi-ptr    blk) val)))

(defun memory-block-device-ptr (blk)
  (if (memory-block-interop-p blk)
      (error "interoperable memory block can not return a device pointer through this interface")
      (memory-block-cuda-device-ptr blk)))

(defun (setf memory-block-device-ptr) (val blk)
  (if (memory-block-interop-p blk)
      (error "interoperable memory block can not be set a device pointer through this interface")
      (setf (memory-block-cuda-device-ptr blk) val)))

(defmacro with-memory-block-device-ptr ((device-ptr blk) &body body)
  (with-gensyms (do-body gres-ptr gres size-ptr)
    `(flet ((,do-body (,device-ptr)
              ,@body))
       (cond ((integerp ,blk)
              ;; we need to pass a pointer to the device pointer
              (cffi:with-foreign-object (ptr :pointer)
                (setf (cffi:mem-aref ptr :pointer) (cffi:make-pointer,blk))
                (,do-body ptr)))
             ((cffi:pointerp ,blk)
              (cffi:with-foreign-object (ptr :pointer)
                (setf (cffi:mem-aref ptr :pointer) ,blk)
                (,do-body ptr)))
             ((memory-block-interop-p ,blk)
              (let* ((,gres-ptr   (memory-block-graphic-resource-ptr ,blk))
                     (,gres       (cffi:mem-ref ,gres-ptr 'cu-graphics-resource))
                     (,device-ptr (cffi:foreign-alloc 'cu-device-ptr))
                     (,size-ptr   (cffi:foreign-alloc :unsigned-int)))
                (unwind-protect
                     (progn
                       (cu-graphics-resource-set-map-flags ,gres cu-graphics-map-resource-flags-none)
                       (cu-graphics-map-resources 1 ,gres-ptr (cffi:null-pointer))
                       (cu-graphics-resource-get-mapped-pointer ,device-ptr ,size-ptr ,gres)
                       (,do-body ,device-ptr)
                       (cu-graphics-unmap-resources 1 ,gres-ptr (cffi:null-pointer)))
                  (cffi:foreign-free ,size-ptr)
                  (cffi:foreign-free ,device-ptr))))
             (t
              (let ((,device-ptr (memory-block-device-ptr ,blk)))
                (,do-body ,device-ptr)))))))

(defun memory-block-vertex-buffer-object (blk)
  (if (memory-block-interop-p blk)
      (memory-block-interop-vertex-buffer-object blk)
      (error "not interoperable memory block")))

(defun (setf memory-block-vertex-buffer-object) (val blk)
  (if (memory-block-interop-p blk)
      (setf (memory-block-interop-vertex-buffer-object blk) val)
      (error "not interoperable memory block")))

(defun memory-block-graphic-resource-ptr (blk)
  (if (memory-block-interop-p blk)
      (memory-block-interop-graphic-resource-ptr blk)
      (error "not interoperable memory block")))

(defun (setf memory-block-graphic-resource-ptr) (val blk)
  (if (memory-block-interop-p blk)
      (setf (memory-block-interop-graphic-resource-ptr blk) val)
      (error "not interoperable memory block")))

(defun memory-block-type (blk)
  (if (memory-block-interop-p blk)
      (memory-block-interop-type blk)
      (memory-block-cuda-type    blk)))

(defun (setf memory-block-type) (val blk)
  (if (memory-block-interop-p blk)
      (setf (memory-block-interop-type blk) val)
      (setf (memory-block-cuda-type    blk) val)))

(defun memory-block-cffi-type (blk)
  (cffi-type (memory-block-type blk)))

(defun memory-block-length (blk)
  (if (memory-block-interop-p blk)
      (memory-block-interop-length blk)
      (memory-block-cuda-length    blk)))

(defun (setf memory-block-length) (val blk)
  (if (memory-block-length blk)
      (setf (memory-block-interop-length blk) val)
      (setf (memory-block-cuda-length    blk) val)))

(defun memory-block-bytes (blk)
  (* (memory-block-element-bytes blk)
     (memory-block-length blk)))

(defun memory-block-element-bytes (blk)
  (type-size (memory-block-type blk)))

(defmacro with-memory-block ((var type size &key (interop nil)) &body body)
  `(let ((,var (alloc-memory-block ,type ,size :interop ,interop)))
     (unwind-protect (progn ,@body)
       (free-memory-block ,var))))

(defmacro with-memory-blocks (bindings &body body)
  (if bindings
      `(with-memory-block ,(car bindings)
         (with-memory-blocks ,(cdr bindings)
           ,@body))
      `(progn ,@body)))

(defun basic-type-mem-aref (blk idx)
  ;; give type as constant explicitly for better performance
  (case (memory-block-type blk)
    (bool  (cffi:mem-aref (memory-block-cffi-ptr blk) '(:boolean :int8) idx))
    (int   (cffi:mem-aref (memory-block-cffi-ptr blk) :int              idx))
    (float (cffi:mem-aref (memory-block-cffi-ptr blk) :float            idx))
    (double (cffi:mem-aref (memory-block-cffi-ptr blk) :double          idx))
    (t (error "must not be reached"))))

(defun float3-mem-aref (blk idx)
  ;; give type and slot names as constant explicitly for better performance
  (let ((ptr (cffi:mem-aptr (memory-block-cffi-ptr blk) '(:struct float3) idx)))
    (make-float3 (cffi:foreign-slot-value ptr '(:struct float3) 'x)
                 (cffi:foreign-slot-value ptr '(:struct float3) 'y)
                 (cffi:foreign-slot-value ptr '(:struct float3) 'z))))

(defun float4-mem-aref (blk idx)
  ;; give type and slot names as constant explicitly for better performance
  (let ((ptr (cffi:mem-aptr (memory-block-cffi-ptr blk) '(:struct float4) idx)))
    (make-float4 (cffi:foreign-slot-value ptr '(:struct float4) 'x)
                 (cffi:foreign-slot-value ptr '(:struct float4) 'y)
                 (cffi:foreign-slot-value ptr '(:struct float4) 'z)
                 (cffi:foreign-slot-value ptr '(:struct float4) 'w))))
                 
(defun double3-mem-aref (blk idx)
  ;; give type and slot names as constant explicitly for better performance
  (let ((ptr (cffi:mem-aptr (memory-block-cffi-ptr blk) '(:struct double3) idx)))
    (make-double3 (cffi:foreign-slot-value ptr '(:struct double3) 'x)
                 (cffi:foreign-slot-value ptr '(:struct double3) 'y)
                 (cffi:foreign-slot-value ptr '(:struct double3) 'z))))

(defun double4-mem-aref (blk idx)
  ;; give type and slot names as constant explicitly for better performance
  (let ((ptr (cffi:mem-aptr (memory-block-cffi-ptr blk) '(:struct double4) idx)))
    (make-double4 (cffi:foreign-slot-value ptr '(:struct double4) 'x)
                 (cffi:foreign-slot-value ptr '(:struct double4) 'y)
                 (cffi:foreign-slot-value ptr '(:struct double4) 'z)
                 (cffi:foreign-slot-value ptr '(:struct double4) 'w))))
                 
(defun vector-type-mem-aref (blk idx)
  (case (memory-block-type blk)
    (float3 (float3-mem-aref blk idx))
    (float4 (float4-mem-aref blk idx))
    (double3 (double3-mem-aref blk idx))
    (double4 (double4-mem-aref blk idx))
    (t (error "must not be reached"))))

(defun mem-aref (blk idx)
  (unless (and (<= 0 idx) (< idx (memory-block-length blk)))
    (error (format nil "invalid index: ~A" idx)))
  (let ((type (memory-block-type blk)))
    (cond
      ((basic-type-p type)  (basic-type-mem-aref blk idx))
      ((vector-type-p type) (vector-type-mem-aref blk idx))
      (t (error "must not be reached")))))

(defun basic-type-setf-mem-aref (blk idx val)
  ;; give type as constant explicitly for better performance
  (case (memory-block-type blk)
    (bool  (setf (cffi:mem-aref (memory-block-cffi-ptr blk) '(:boolean :int8) idx) val))
    (int   (setf (cffi:mem-aref (memory-block-cffi-ptr blk) :int              idx) val))
    (float (setf (cffi:mem-aref (memory-block-cffi-ptr blk) :float            idx) val))
    (double (setf (cffi:mem-aref (memory-block-cffi-ptr blk) :double          idx) val))
    (t (error "must not be reached"))))

(defun float3-setf-mem-aref (blk idx val)
  ;; give type and slot names as constant explicitly for better performance
  (let ((ptr (cffi:mem-aptr (memory-block-cffi-ptr blk) '(:struct float3) idx)))
    (setf (cffi:foreign-slot-value ptr '(:struct float3) 'x) (float3-x val))
    (setf (cffi:foreign-slot-value ptr '(:struct float3) 'y) (float3-y val))
    (setf (cffi:foreign-slot-value ptr '(:struct float3) 'z) (float3-z val))))

(defun float4-setf-mem-aref (blk idx val)
  ;; give type and slot names as constant explicitly for better performance
  (let ((ptr (cffi:mem-aptr (memory-block-cffi-ptr blk) '(:struct float4) idx)))
    (setf (cffi:foreign-slot-value ptr '(:struct float4) 'x) (float4-x val))
    (setf (cffi:foreign-slot-value ptr '(:struct float4) 'y) (float4-y val))
    (setf (cffi:foreign-slot-value ptr '(:struct float4) 'z) (float4-z val))
    (setf (cffi:foreign-slot-value ptr '(:struct float4) 'w) (float4-w val))))

(defun double3-setf-mem-aref (blk idx val)
  ;; give type and slot names as constant explicitly for better performance
  (let ((ptr (cffi:mem-aptr (memory-block-cffi-ptr blk) '(:struct double3) idx)))
    (setf (cffi:foreign-slot-value ptr '(:struct double3) 'x) (double3-x val))
    (setf (cffi:foreign-slot-value ptr '(:struct double3) 'y) (double3-y val))
    (setf (cffi:foreign-slot-value ptr '(:struct double3) 'z) (double3-z val))))

(defun double4-setf-mem-aref (blk idx val)
  ;; give type and slot names as constant explicitly for better performance
  (let ((ptr (cffi:mem-aptr (memory-block-cffi-ptr blk) '(:struct double4) idx)))
    (setf (cffi:foreign-slot-value ptr '(:struct double4) 'x) (double4-x val))
    (setf (cffi:foreign-slot-value ptr '(:struct double4) 'y) (double4-y val))
    (setf (cffi:foreign-slot-value ptr '(:struct double4) 'z) (double4-z val))
    (setf (cffi:foreign-slot-value ptr '(:struct double4) 'w) (double4-w val))))

(defun vector-type-setf-mem-aref (blk idx val)
  (case (memory-block-type blk)
    (float3 (float3-setf-mem-aref blk idx val))
    (float4 (float4-setf-mem-aref blk idx val))
    (double3 (double3-setf-mem-aref blk idx val))
    (double4 (double4-setf-mem-aref blk idx val))
    (t (error "must not be unreached"))))

(defun (setf mem-aref) (val blk idx)
  (unless (and (<= 0 idx) (< idx (memory-block-length blk)))
    (error (format nil "invalid index: ~A" idx)))
  (let ((type (memory-block-type blk)))
    (cond
      ((basic-type-p type)  (basic-type-setf-mem-aref blk idx val))
      ((vector-type-p type) (vector-type-setf-mem-aref blk idx val))
      (t (error "must not be reached")))))

(defun memcpy-host-to-device (&rest blks)
  (dolist (blk blks)    
    (let ((cffi-ptr (memory-block-cffi-ptr blk))
          (bytes    (memory-block-bytes blk)))
      (with-memory-block-device-ptr (device-ptr blk)
        (cu-memcpy-host-to-device (cffi:mem-ref device-ptr 'cu-device-ptr)
                                  cffi-ptr
                                  bytes)))))

(defun memcpy-device-to-host (&rest blks)
  (dolist (blk blks)
    (let ((cffi-ptr (memory-block-cffi-ptr blk))
          (bytes    (memory-block-bytes blk)))
      (with-memory-block-device-ptr (device-ptr blk)
        (cu-memcpy-device-to-host cffi-ptr
                                  (cffi:mem-ref device-ptr 'cu-device-ptr)
                                  bytes)))))


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


;;;
;;; Definition of module info
;;;

(defun make-module-info ()
  (list nil nil t (make-hash-table)))

(defun module-handle (info)
  (car info))

(defun (setf module-handle) (val info)
  (setf (car info) val))

(defun module-path (info)
  (cadr info))

(defun (setf module-path) (val info)
  (setf (cadr info) val))

(defun module-compilation-needed (info)
  (caddr info))

(defun (setf module-compilation-needed) (val info)
  (setf (caddr info) val))

(defun function-handles (info)
  (cadddr info))

(defun function-handle (info name)
  (gethash name (function-handles info)))

(defun (setf function-handle) (handle info name)
  (setf (gethash name (function-handles info)) handle))


;;;
;;; Definition of kernel manager
;;;

(defun make-kernel-manager()
  (list (make-module-info) (empty-kernel-definition)))

(defun module-info (mgr)
  (car mgr))

(defun kernel-definition (mgr)
  (cadr mgr))

(defun (setf kernel-definition) (val mgr)
  (setf (cadr mgr) val))

(defun kernel-manager-module-handle (mgr)
  (module-handle (module-info mgr)))

(defun (setf kernel-manager-module-handle) (handle mgr)
  (setf (module-handle (module-info mgr)) handle))

(defun kernel-manager-module-path (mgr)
  (module-path (module-info mgr)))

(defun (setf kernel-manager-module-path) (val mgr)
  (setf (module-path (module-info mgr)) val))

(defun kernel-manager-module-compilation-needed (mgr)
  (module-compilation-needed (module-info mgr)))

(defun (setf kernel-manager-module-compilation-needed) (val mgr)
  (setf (module-compilation-needed (module-info mgr)) val))

(defun kernel-manager-function-handle (mgr name)
  (function-handle (module-info mgr) name))

(defun (setf kernel-manager-function-handle) (val mgr name)
  (setf (function-handle (module-info mgr) name) val))

(defun kernel-manager-function-exists-p (mgr name)
  (kernel-definition-function-exists-p name (kernel-definition mgr)))

(defun kernel-manager-function-names (mgr)
  (kernel-definition-function-names (kernel-definition mgr)))

(defun kernel-manager-function-name (mgr name)
  (kernel-definition-function-name name (kernel-definition mgr)))

(defun kernel-manager-function-c-name (mgr name)
  (kernel-definition-function-c-name name (kernel-definition mgr)))

(defun function-modified-p (name return-type args body def)
  (not (and (equal return-type (kernel-definition-function-return-type name def))
            (equal args (kernel-definition-function-arguments name def))
            (equal body (kernel-definition-function-body name def)))))

(defun kernel-manager-define-function (mgr name return-type args body)
  (symbol-macrolet ((def (kernel-definition mgr))
                    (info (module-info mgr)))
    (when (or (not (kernel-definition-function-exists-p name def))
              (function-modified-p name return-type args body def))
      (setf def (add-function-to-kernel-definition name return-type args body def)
            (module-compilation-needed info) t))))

(defun kernel-manager-macro-exists-p (mgr name)
  (kernel-definition-macro-exists-p name (kernel-definition mgr)))

(defun kernel-manager-macro-names (mgr)
  (kernel-definition-macro-names (kernel-definition mgr)))

(defun kernel-manager-macro-name (mgr name)
  (kernel-definition-macro-name name (kernel-definition mgr)))

(defun macro-modified-p (name args body def)
  (not (and (equal args (kernel-definition-macro-arguments name def))
            (equal body (kernel-definition-macro-body name def)))))

(defun kernel-manager-define-macro (mgr name args body expander)
  (symbol-macrolet ((def (kernel-definition mgr))
                    (info (module-info mgr)))
    (when (or (not (kernel-definition-macro-exists-p name def))
              (macro-modified-p name args body def))
      (setf def (add-macro-to-kernel-definition name args body expander def)
            (module-compilation-needed info) t))))

(defun constant-modified-p (name type exp def)
  (not (and (equal type (kernel-definition-constant-type name def))
            (equal exp (kernel-definition-constant-expression name def)))))

(defun kernel-manager-define-constant (mgr name type exp)
  (symbol-macrolet ((def (kernel-definition mgr))
                    (info (module-info mgr)))
    (when (or (not (kernel-definition-constant-exists-p name def))
              (constant-modified-p name type exp def))
      (setf def (add-constant-to-kernel-definition name type exp def)
            (module-compilation-needed info) t))))

(defun symbol-macro-modified-p (name exp def)
  (not (equal exp (kernel-definition-symbol-macro-expansion name def))))

(defun kernel-manager-define-symbol-macro (mgr name exp)
  (symbol-macrolet ((def (kernel-definition mgr))
                    (info (module-info mgr)))
    (when (or (not (kernel-definition-symbol-macro-exists-p name def))
              (symbol-macro-modified-p name exp def))
      (setf def (add-symbol-macro-to-kernel-definition name exp def)
            (module-compilation-needed info) t))))

(defun kernel-manager-load-function (mgr name)
  (unless (kernel-manager-module-handle mgr)
    (error "kernel module is not loaded yet."))
  (when (kernel-manager-function-handle mgr name)
    (error "kernel function \"~A\" is already loaded." name))
  (let ((hmodule (kernel-manager-module-handle mgr))
        (hfunc (cffi:foreign-alloc 'cu-function))
        (fname (kernel-manager-function-c-name mgr name)))
      (cu-module-get-function hfunc (cffi:mem-ref hmodule 'cu-module) fname)
      (setf (kernel-manager-function-handle mgr name) hfunc)))

(defun kernel-manager-load-module (mgr)
  (unless (not (kernel-manager-module-compilation-needed mgr))
    (error "a module needs to be compiled before loaded."))
  (when (kernel-manager-module-handle mgr)
    (error "kernel module is already loaded."))
  (unless (no-kernel-functions-loaded-p mgr)
    (error "some kernel functions are already loaded."))
  (let ((hmodule (cffi:foreign-alloc 'cu-module))
        (path (kernel-manager-module-path mgr)))
    (cu-module-load hmodule path)
    (setf (kernel-manager-module-handle mgr) hmodule)))

(defun no-kernel-functions-loaded-p (mgr)
  (notany #'(lambda (name)
              (kernel-manager-function-handle mgr name))
          (kernel-manager-function-names mgr)))

(defun kernel-manager-unload (mgr)
  (swhen (kernel-manager-module-handle mgr)
    (cu-module-unload (cffi:mem-ref it 'cu-module)))
  (free-function-handles mgr)
  (free-module-handle mgr))

(defun free-module-handle (mgr)
  (swhen (kernel-manager-module-handle mgr)
    (cffi:foreign-free it)
    (setf it nil)))

(defun free-function-handles (mgr)
  (mapcar #'(lambda (name)
              (swhen (kernel-manager-function-handle mgr name)
                (cffi:foreign-free it)
                (setf it nil)))
          (kernel-manager-function-names mgr)))

(defvar *tmp-path* "/tmp/")
(defvar *mktemp* (osicat-posix:mktemp))

(defun get-tmp-path ()
  (let ((last (subseq (reverse *tmp-path*) 0 1)))
    (if (string= last "/")
        (concatenate 'string *tmp-path*)
        (concatenate 'string *tmp-path* "/"))))

(defun get-cu-path ()
  (concatenate 'string (get-tmp-path) "cl-cuda-" *mktemp* ".cu"))

(defun get-ptx-path ()
  (concatenate 'string (get-tmp-path) "cl-cuda-" *mktemp* ".ptx"))

(defun get-include-path ()
  (namestring (asdf:system-relative-pathname :cl-cuda #P"include")))

(defparameter *nvcc-options*
  ;; compute capability 1.3 is needed for double floats, but 2.0 for
  ;; good performance
  (list "-arch=sm_11"))

(defun get-nvcc-options (include-path cu-path ptx-path)
  (append *nvcc-options*
          (list "-I" include-path "-ptx" "-o" ptx-path cu-path)))

(defun output-cu-code (mgr path)
  (with-open-file (out path :direction :output :if-exists :supersede)
    (princ (compile-kernel-definition (kernel-definition mgr)) out)))

(defvar *nvcc-binary* "nvcc"
  "Set this to an absolute path if your lisp doesn't search PATH.")

(defun output-nvcc-command (opts)
  (format t "~A~{ ~A~}~%" *nvcc-binary* opts))

(defun run-nvcc-command (opts)
  (with-output-to-string (out)
    (multiple-value-bind (status exit-code)
        (external-program:run *nvcc-binary* opts :error out)
      (unless (and (eq status :exited) (= 0 exit-code))
        (error "nvcc exits with code: ~A~%~A" exit-code
               (get-output-stream-string out))))))

(defun compile-cu-code (include-path cu-path ptx-path)
  (let ((opts (get-nvcc-options include-path cu-path ptx-path)))
    (output-nvcc-command opts)
    (run-nvcc-command opts)))

(defun kernel-manager-generate-and-compile (mgr)
  (unless (not (kernel-manager-module-handle mgr))
    (error "kernel module is already loaded."))
  (unless (no-kernel-functions-loaded-p mgr)
    (error "some kernel functions are already loaded."))
  (let ((include-path (get-include-path))
        (cu-path      (get-cu-path))
        (ptx-path     (get-ptx-path)))
    (output-cu-code mgr cu-path)
    (compile-cu-code include-path cu-path ptx-path)
    (setf (kernel-manager-module-path mgr) ptx-path
          (kernel-manager-module-compilation-needed mgr) nil)))


;;;
;;; Definition of kernel manager's ensure- functions
;;;

(defun ensure-kernel-function-loaded (mgr name)
  (ensure-kernel-module-loaded mgr)
  (or (kernel-manager-function-handle mgr name)
      (kernel-manager-load-function mgr name)))

(defun ensure-kernel-module-loaded (mgr)
  (ensure-kernel-module-compiled mgr)
  (or (kernel-manager-module-handle mgr)
      (kernel-manager-load-module mgr)))

(defun ensure-kernel-module-compiled (mgr)
  (when (kernel-manager-module-compilation-needed mgr)
    (kernel-manager-generate-and-compile mgr))
  (values))


;;;
;;; Definition of default kernel manager
;;;

(defvar *kernel-manager*
  (make-kernel-manager))

(defun clear-kernel-manager ()
  (setf *kernel-manager* (make-kernel-manager))
  (values))

(defun print-kernel-manager ()
  (let ((module-info (module-info *kernel-manager*)))
    (list (module-handle             module-info)
          (module-path               module-info)
          (module-compilation-needed module-info)
          (hash-table-alist (function-handles module-info)))))


;;;
;;; Timer
;;;

(defstruct timer-object
  start-event stop-event)

(defun create-timer ()
  (let ((timer-object (make-timer-object)))
    (symbol-macrolet ((start-event (timer-object-start-event timer-object))
                      (stop-event  (timer-object-stop-event  timer-object)))
      (unwind-protect-case ()
          (progn
            ;; allocate memory pointers to start and stop events
            (setf start-event (cffi:foreign-alloc 'cu-event :initial-element (cffi:null-pointer))
                  stop-event  (cffi:foreign-alloc 'cu-event :initial-element (cffi:null-pointer)))
            ;; create a start event
            (cu-event-create start-event cu-event-default)
            ;; create a stop event
            (cu-event-create stop-event  cu-event-default)
            ;; return a timer object
            timer-object)
        (:abort (destroy-timer timer-object))))))

(defun destroy-timer (timer-object)
  (symbol-macrolet ((start-event         (timer-object-start-event timer-object))
                    (stop-event          (timer-object-stop-event  timer-object))
                    (mem-ref-start-event (cffi:mem-ref start-event 'cu-event))
                    (mem-ref-stop-event  (cffi:mem-ref stop-event  'cu-event)))
    ;; destroy a stop event if created
    (when (and stop-event (not (cffi:null-pointer-p mem-ref-stop-event)))
      (cu-event-destroy mem-ref-stop-event)
      (setf mem-ref-stop-event (cffi:null-pointer)))
    ;; destroy a start event if created
    (when (and start-event (not (cffi:null-pointer-p mem-ref-start-event)))
      (cu-event-destroy mem-ref-start-event)
      (setf mem-ref-start-event (cffi:null-pointer)))
    ;; free a memory pointer to a stop event if allocated
    (when stop-event
      (cffi:foreign-free stop-event)
      (setf stop-event nil))
    ;; free a memory pointer to a start event if allocated
    (when start-event
      (cffi:foreign-free start-event)
      (setf start-event nil))))

(defmacro with-timer ((timer) &body body)
  `(let ((,timer (create-timer)))
     (unwind-protect (progn ,@body)
       (destroy-timer ,timer))))

(defun start-timer (timer-object)
  (let ((start-event (timer-object-start-event timer-object)))
    (cu-event-record (cffi:mem-ref start-event 'cu-event) (cffi:null-pointer))))

(defun stop-and-synchronize-timer (timer-object)
  (let ((stop-event (timer-object-stop-event timer-object)))
    (cu-event-record (cffi:mem-ref stop-event 'cu-event) (cffi:null-pointer))
    (cu-event-synchronize (cffi:mem-ref stop-event 'cu-event))))

(defun get-elapsed-time (timer-object)
  (let (milliseconds
        (start-event (timer-object-start-event timer-object))
        (stop-event  (timer-object-stop-event  timer-object)))
    (stop-and-synchronize-timer timer-object)
    (cffi:with-foreign-object (pmilliseconds :float)
      (cu-event-elapsed-time pmilliseconds
                             (cffi:mem-ref start-event 'cu-event)
                             (cffi:mem-ref stop-event  'cu-event))
      (setf milliseconds (cffi:mem-ref pmilliseconds :float)))
    (start-timer timer-object)
    milliseconds))
