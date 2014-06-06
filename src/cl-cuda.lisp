#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-cuda)


;;;
;;; Load CUDA library
;;;

(cffi:define-foreign-library libcuda
  (:darwin (:framework "CUDA"))
  (:unix (:or "libcuda.so" "libcuda64.so")))

(ignore-errors
 (cffi:use-foreign-library libcuda))


;;;
;;; Definition of defcufun macro
;;;

(defmacro defcufun (name-and-options return-type &body args)
  (let* ((name (car name-and-options))
         (name% (symbolicate name "%"))
         (name-and-options% (cons name% (cdr name-and-options)))
         (params (mapcar #'car args)))
      `(progn
         (defun ,name (,@params)
           (check-cuda-errors ',name (,name% ,@params)))
         (cffi:defcfun ,name-and-options% ,return-type ,@args))))


;;;
;;; Definition of defcuenum macro
;;;

(eval-when (:compile-toplevel :load-toplevel)
  (defun enum-keyword (enum-elem)
    (match (ensure-list enum-elem)
      ((keyword) keyword)
      ((keyword _) keyword)
      (_  (error (format nil "invalid enum element: ~A" enum-elem))))))

(eval-when (:compile-toplevel :load-toplevel)
  (defun enum-value (enum-elem)
    (match enum-elem
      ((_ value) value)
      (_ (error (format nil "invalid enum element: ~A" enum-elem))))))
  
(eval-when (:compile-toplevel :load-toplevel)
  (defun defconstant-enum-value (name enum-elem)
    (let ((keyword (enum-keyword enum-elem)))
      `(defconstant ,(symbolicate keyword)
         (cffi:foreign-enum-value ',name ,keyword)))))

(defmacro defcuenum (name-and-options &body enum-list)
  (let ((name name-and-options))
    `(progn
       (cffi:defcenum ,name
         ,@enum-list)
       ,@(mapcar (lambda (enum-elem)
                   (defconstant-enum-value name enum-elem))
                 enum-list))))


;;;
;;; Definition of CUDA driver API types
;;;

(cffi:defctype cu-result :unsigned-int)
(cffi:defctype cu-device :int)
(cffi:defctype cu-context :pointer)
(cffi:defctype cu-module :pointer)
(cffi:defctype cu-function :pointer)
(cffi:defctype cu-stream :pointer)
(cffi:defctype cu-event :pointer)
(cffi:defctype cu-graphics-resource :pointer)

(cffi:defcstruct curand-state-xorwow
  (d :unsigned-int)
  (v :unsigned-int :count 5)
  (boxmuller-flag :int)
  (boxmuller-flag-double :int)
  (boxmuller-extra :float)
  (boxmuller-extra-double :double))


;;;
;;; Definition of CUDA driver API enums
;;;

(defcuenum cu-event-flags-enum
  (:cu-event-default        #X0)
  (:cu-event-blocking-sync  #X1)
  (:cu-event-disable-timing #X2)
  (:cu-event-interprocess   #X4))

(defcuenum cu-graphics-register-flags
  (:cu-graphics-register-flags-none           #X0)
  (:cu-graphics-register-flags-read-only      #X1)
  (:cu-graphics-register-flags-write-discard  #X2)
  (:cu-graphics-register-flags-surface-ldst   #X4)
  (:cu-graphics-register-flags-texture-gather #X8))

(defcuenum cu-graphics-map-resource-flags
  (:cu-graphics-map-resource-flags-none          #X0)
  (:cu-graphics-map-resource-flags-read-only     #X1)
  (:cu-graphics-map-resource-flags-write-discard #X2))


;;;
;;; Definition of CUDA driver API functions
;;;

;; cuInit
(defcufun (cu-init "cuInit") cu-result (flags :unsigned-int))

;; cuDeviceGet
(defcufun (cu-device-get "cuDeviceGet") cu-result
  (device (:pointer cu-device))
  (ordinal :int))

;; cuDeviceGetCount
(defcufun (cu-device-get-count "cuDeviceGetCount") cu-result
  (count (:pointer :int)))

;; cuDeviceComputeCapability
(defcufun (cu-device-compute-capability "cuDeviceComputeCapability") cu-result
  (major (:pointer :int))
  (minor (:pointer :int))
  (dev cu-device))

;; cuDeviceGetName
(defcufun (cu-device-get-name "cuDeviceGetName") cu-result
  (name :string)
  (len :int)
  (dev cu-device))

;; cuCtxCreate
(defcufun (cu-ctx-create "cuCtxCreate_v2") cu-result
  (pctx (:pointer cu-context))
  (flags :unsigned-int)
  (dev cu-device))

;; cuGLCtxCreate
(defcufun (cu-gl-ctx-create "cuGLCtxCreate") cu-result
  (pctx (:pointer cu-context))
  (flags :unsigned-int)
  (dev cu-device))

;; cuCtxDestroy
(defcufun (cu-ctx-destroy "cuCtxDestroy_v2") cu-result
  (ctx cu-context))

;; cuCtxSynchronize
(defcufun (cu-ctx-synchronize "cuCtxSynchronize") cu-result)

;; cuMemAlloc
(defcufun (cu-mem-alloc "cuMemAlloc_v2") cu-result
  (dptr (:pointer cu-device-ptr))
  (bytesize size-t))

;; cuMemFree
(defcufun (cu-mem-free "cuMemFree_v2") cu-result
  (dptr cu-device-ptr))

;; cuMemcpyHtoD
(defcufun (cu-memcpy-host-to-device "cuMemcpyHtoD_v2") cu-result
  (dst-device cu-device-ptr)
  (src-host :pointer)
  (byte-count size-t))

;; cuMemcpyDtoH
(defcufun (cu-memcpy-device-to-host "cuMemcpyDtoH_v2") cu-result
  (dst-host :pointer)
  (src-device cu-device-ptr)
  (byte-count size-t))

;; cuModuleLoad
(defcufun (cu-module-load "cuModuleLoad") cu-result
  (module (:pointer cu-module))
  (fname :string))

;; cuModuleUnload
(defcufun (cu-module-unload "cuModuleUnload") cu-result
  (module cu-module))

;; cuModuleGetFunction
(defcufun (cu-module-get-function "cuModuleGetFunction") cu-result
  (hfunc (:pointer cu-function))
  (hmod cu-module)
  (name :string))

;; cuLaunchKernel
(defcufun (cu-launch-kernel "cuLaunchKernel") cu-result
  (f cu-function)
  (grid-dim-x :unsigned-int)
  (grid-dim-y :unsigned-int)
  (grid-dim-z :unsigned-int)
  (block-dim-x :unsigned-int)
  (block-dim-y :unsigned-int)
  (block-dim-z :unsigned-int)
  (shared-mem-bytes :unsigned-int)
  (hstream cu-stream)
  (kernel-params (:pointer :pointer))
  (extra (:pointer :pointer)))

;; cuEventCreate
(defcufun (cu-event-create "cuEventCreate") cu-result
  (phevent (:pointer cu-event))
  (flags :unsigned-int))

;; cuEventDestroy
(defcufun (cu-event-destroy "cuEventDestroy_v2") cu-result
  (h-event cu-event))

;; cuEventElapsedTime
(defcufun (cu-event-elapsed-time "cuEventElapsedTime") cu-result
  (pmilliseconds (:pointer :float))
  (hstart cu-event)
  (hend cu-event))

;; cuEventQuery
(defcufun (cu-event-query "cuEventQuery") cu-result
  (hevent cu-event))

;; cuEventRecord
(defcufun (cu-event-record "cuEventRecord") cu-result
  (hevent cu-event)
  (hstream cu-stream))

;; cuEventSynchronize
(defcufun (cu-event-synchronize "cuEventSynchronize") cu-result
  (hevent cu-event))

;; cuGraphicsGLRegisterBuffer
(defcufun (cu-graphics-gl-register-buffer "cuGraphicsGLRegisterBuffer") cu-result
  (p-cuda-resource (:pointer cu-graphics-resource))
  (buffer %gl:uint)
  (flags :unsigned-int))

;; cuGraphicsMapResources
(defcufun (cu-graphics-map-resources "cuGraphicsMapResources") cu-result
  (count     :unsigned-int)
  (resources (:pointer cu-graphics-resource))
  (hstream   cu-stream))

;; cuGraphicsResourceGetMappedPointer
(defcufun (cu-graphics-resource-get-mapped-pointer "cuGraphicsResourceGetMappedPointer") cu-result
  (pdevptr  (:pointer cu-device-ptr))
  (psize    (:pointer size-t))
  (resource cu-graphics-resource))

;; cuGraphicsResourceSetMapFlags
(defcufun (cu-graphics-resource-set-map-flags "cuGraphicsResourceSetMapFlags") cu-result
  (resource cu-graphics-resource)
  (flags    :unsigned-int))

;; cuGraphicsUnmapResources
(defcufun (cu-graphics-unmap-resources "cuGraphicsUnmapResources") cu-result
  (count     :unsigned-int)
  (resources (:pointer cu-graphics-resource))
  (hstream   cu-stream))

;; cuGraphicsUnregisterResource
(defcufun (cu-graphics-unregister-resource "cuGraphicsUnregisterResource") cu-result
  (resource cu-graphics-resource))


;; check-cuda-errors function
(defvar +cuda-success+ 0)
(defvar *show-messages* t)

(defun check-cuda-errors (name return-code)
  (unless (= return-code +cuda-success+)
    (error "~A failed with driver API error No. ~A.~%~A"
           name return-code (get-error-string return-code)))
  (when *show-messages*
    (format t "~A succeeded.~%" name))
  (values))

(declaim (special +built-in-functions+))
(declaim (special +built-in-macros+))
(declaim (special *kernel-manager*))

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
;;; Definition of Built-in Vector Types
;;;

(defstruct (float3 (:constructor make-float3 (x y z)))
  (x 0.0 :type single-float)
  (y 0.0 :type single-float)
  (z 0.0 :type single-float))

(defun float3-= (a b)
  (and (= (float3-x a) (float3-x b))
       (= (float3-y a) (float3-y b))
       (= (float3-z a) (float3-z b))))

(cffi:defcstruct float3
  (x :float)
  (y :float)
  (z :float))

(defstruct (float4 (:constructor make-float4 (x y z w)))
  (x 0.0 :type single-float)
  (y 0.0 :type single-float)
  (z 0.0 :type single-float)
  (w 0.0 :type single-float))

(defun float4-= (a b)
  (and (= (float4-x a) (float4-x b))
       (= (float4-y a) (float4-y b))
       (= (float4-z a) (float4-z b))
       (= (float4-w a) (float4-w b))))

(cffi:defcstruct float4
  (x :float)
  (y :float)
  (z :float)
  (w :float))

(defstruct (double3 (:constructor make-double3 (x y z)))
  (x 0.0d0 :type double-float)
  (y 0.0d0 :type double-float)
  (z 0.0d0 :type double-float))

(defun double3-= (a b)
  (and (= (double3-x a) (double3-x b))
       (= (double3-y a) (double3-y b))
       (= (double3-z a) (double3-z b))))

(cffi:defcstruct double3
  (x :double)
  (y :double)
  (z :double))

(defstruct (double4 (:constructor make-double4 (x y z w)))
  (x 0.0d0 :type double-float)
  (y 0.0d0 :type double-float)
  (z 0.0d0 :type double-float)
  (w 0.0d0 :type double-float))

(defun double4-= (a b)
  (and (= (double4-x a) (double4-x b))
       (= (double4-y a) (double4-y b))
       (= (double4-z a) (double4-z b))
       (= (double4-w a) (double4-w b))))

(cffi:defcstruct double4
  (x :double)
  (y :double)
  (z :double)
  (w :double))


;;;
;;; Definition of cl-cuda types
;;;

(defparameter +basic-types+ `((void  0 :void)
                              (bool  1 (:boolean :int8))
                              (int   4 :int)
                              (float 4 :float)
                              (double 8 :double)
                              (curand-state-xorwow
                               ,(cffi:foreign-type-size
                                 '(:struct curand-state-xorwow))
                               (:pointer :struct curand-state-xorwow))))

(defvar +vector-types+ '((float3 float 3 float3-x float3-y float3-z)
                         (float4 float 4 float4-x float4-y float4-z float4-w)
                         (double3 double 3 double3-x double3-y double3-z)
                         (double4 double 4 double4-x double4-y double4-z double4-w)))

(defvar +vector-type-elements+ '(x y z w))

(defun type-size (type)
  (cond
    ((basic-type-p type)  (basic-type-size type))
    ((vector-type-p type) (vector-type-size type))
    ((array-type-p type)  (array-type-pointer-size type))
    (t (error "invalid type:~A" type))))

(defun valid-type-p (type)
  (or (basic-type-p  type)
      (vector-type-p type)
      (array-type-p  type)))

(defun cffi-type (type)
  (cond
    ((basic-type-p  type) (basic-cffi-type  type))
    ((vector-type-p type) (vector-cffi-type type))
    ((array-type-p  type) (array-cffi-type  type))
    (t (error "invalid type: ~A" type))))

(defun non-pointer-type-p (type)
  (or (basic-type-p type)
      (vector-type-p type)))

(defun basic-type-size (type)
  (or (cadr (assoc type +basic-types+))
      (error "invalid type: ~A" type)))

(defun basic-type-p (type)
  (and (assoc type +basic-types+)
       t))

(defun basic-cffi-type (type)
  (or (caddr (assoc type +basic-types+))
      (error "invalid type: ~A" type)))

(defun vector-type-size (type)
  (* (vector-type-length type)
     (type-size (vector-type-base-type type))))

(defun vector-type-p (type)
  (and (assoc type +vector-types+)
       t))

(defun vector-cffi-type (type)
  (unless (vector-type-p type)
    (error "invalid type: ~A" type))
  (list :struct type))

(defun vector-types ()
  (mapcar #'car +vector-types+))

(defun vector-type-base-type (type)
  (or (cadr (assoc type +vector-types+))
      (error "invalid type: ~A" type)))

(defun vector-type-length (type)
  (or (caddr (assoc type +vector-types+))
      (error "invalid type: ~A" type)))

(defun vector-type-elements (type)
  (loop repeat (vector-type-length type)
     for elm in +vector-type-elements+
     collect elm))

(defun vector-type-selectors (type)
  (or (cdddr (assoc type +vector-types+))
      (error "invalid type: ~A" type)))

(defun valid-vector-type-selector-p (selector)
  (let ((selectors (apply #'append (mapcar #'vector-type-selectors (vector-types)))))
    (and (find selector selectors)
         t)))

(defun vector-type-selector-type (selector)
  (loop for type in (vector-types)
     when (member selector (vector-type-selectors type))
     return type
     finally (error "invalid selector: ~A" selector)))

(defun array-type-p (type)
  (let ((type-str (symbol-name type)))
    (let ((last (aref type-str (1- (length type-str))))
          (rest (remove-star type)))
      (and (eq last #\*)
           (or (basic-type-p rest) (vector-type-p rest))))))

(defun array-cffi-type (type)
  (unless (array-type-p type)
    (error (format nil "invalid type: ~A" type)))
  'cu-device-ptr)

(defun array-type-pointer-size (type)
  (unless (array-type-p type)
    (error (format nil "invalid type: ~A" type)))
  (cffi:foreign-type-size 'cu-device-ptr))

(defun array-type-dimension (type)
  (unless (array-type-p type)
    (error (format nil "invalid type: ~A" type)))
  (count #\* (princ-to-string type)))

(defun add-star (type n)
  (labels ((aux (str n2)
             (if (< n2 1)
                 str
                 (aux (concatenate 'string str "*") (1- n2)))))
    (cl-cuda-symbolicate (aux (princ-to-string type) n))))

(defun remove-star (type)
  (let ((rev (reverse (princ-to-string type))))
    (if (string= (subseq rev 0 1) "*")
        (remove-star (cl-cuda-symbolicate (reverse (subseq rev 1))))
        type)))


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


;;;
;;; Definition of DEFKERNELCONST macro
;;;

(defmacro defkernelconst (name type exp)
  (declare (ignore type))
  `(kernel-manager-define-symbol-macro *kernel-manager* ',name ',exp))


;;;
;;; Definition of kernel definition
;;;

(defun make-kerdef-function (name return-type args body)
  (assert (symbolp name))
  (assert (valid-type-p return-type))
  (dolist (arg args)
    (assert (= (length arg) 2))
    (assert (symbolp (car arg)))
    (assert (valid-type-p (cadr arg))))
  (assert (listp body))
  (list name :function return-type args body))

(defun kerdef-function-p (elem)
  (match elem
    ((_ :function _ _ _) t)
    (_ nil)))

(defun kerdef-function-name (elem)
  (match elem
    ((name :function _ _ _) name)
    (_ (error "invalid kernel definition function: ~A" elem))))

(defun kerdef-function-c-name (elem)
  (compile-identifier-with-package-name (kerdef-function-name elem)))

(defun kerdef-function-return-type (elem)
  (match elem
    ((_ :function return-type _ _) return-type)
    (_ (error "invalid kernel definition function: ~A" elem))))

(defun kerdef-function-arguments (elem)
  (match elem
    ((_ :function _ args _) args)
    (_ (error "invalid kernel definition function: ~A" elem))))

(defun kerdef-function-argument-types (elem)
  (mapcar #'cadr (kerdef-function-arguments elem)))

(defun kerdef-function-body (elem)
  (match elem
    ((_ :function _ _ body) body)
    (_ (error "invalid kernel definition function: ~A" elem))))

(defun make-kerdef-macro (name args body expander)
  (assert (symbolp name))
  (assert (listp args))
  (assert (listp body))
  (assert (functionp expander))
  (list name :macro args body expander))

(defun kerdef-macro-p (elem)
  (match elem
    ((_ :macro _ _ _) t)
    (_ nil)))

(defun kerdef-macro-name (elem)
  (match elem
    ((name :macro _ _ _) name)
    (_ (error "invalid kernel definition macro: ~A" elem))))

(defun kerdef-macro-arguments (elem)
  (match elem
    ((_ :macro args _ _) args)
    (_ (error "invalid kernel definition macro: ~A" elem))))

(defun kerdef-macro-body (elem)
  (match elem
    ((_ :macro _ body _) body)
    (_ (error "invalid kernel definition macro: ~A" elem))))

(defun kerdef-macro-expander (elem)
  (match elem
    ((_ :macro _ _ expander) expander)
    (_ (error "invalid kernel definition macro: ~A" elem))))

(defun make-kerdef-constant (name type expression)
  (assert (symbolp name))
  (assert (valid-type-p type))
  (list name :constant type expression))

(defun kerdef-constant-p (elem)
  (match elem
    ((_ :constant _ _) t)
    (_ nil)))

(defun kerdef-constant-name (elem)
  (match elem
    ((name :constant _ _) name)
    (_ (error "invalid kernel definition constant: ~A" elem))))

(defun kerdef-constant-type (elem)
  (match elem
    ((_ :constant type _) type)
    (_ (error "invalid kernel definition constant: ~A" elem))))

(defun kerdef-constant-expression (elem)
  (match elem
    ((_ :constant _ exp) exp)
    (_ (error "invalid kernel definition constant: ~A" elem))))

(defun make-kerdef-symbol-macro (name expansion)
  (assert (symbolp name))
  (list name :symbol-macro expansion))

(defun kerdef-symbol-macro-p (elem)
  (match elem
    ((_ :symbol-macro _) t)
    (_ nil)))

(defun kerdef-symbol-macro-name (elem)
  (match elem
    ((name :symbol-macro _) name)
    (_ nil)))

(defun kerdef-symbol-macro-expansion (elem)
  (match elem
    ((_ :symbol-macro expansion) expansion)
    (_ (error "invalid kernel definition symbol macro: ~A" elem))))

(defun kerdef-name (elem)
  (match elem
    ((name :function _ _ _) name)
    ((name :macro _ _ _) name)
    ((name :constant _ _) name)
    ((name :symbol-macro _) name)
    (_ (error "invalid kernel definition element: ~A" elem))))

(defun empty-kernel-definition ()
  (list nil nil))

(defun add-function-to-kernel-definition (name return-type arguments body def)
  (destructuring-bind (func-table var-table) def
    (let ((elem (make-kerdef-function name return-type arguments body)))
      (list (remove-duplicates (cons elem func-table) :key #'kerdef-name :from-end t)
            var-table))))

(defun remove-function-from-kernel-definition (name def)
  (unless (kernel-definition-function-exists-p name def)
    (error "undefined kernel definition function: ~A" name))
  (destructuring-bind (func-table var-table) def
    (list (remove name func-table :key #'kerdef-name) var-table)))

(defun add-macro-to-kernel-definition (name arguments body expander def)
  (destructuring-bind (func-table var-table) def
    (let ((elem (make-kerdef-macro name arguments body expander)))
      (list (remove-duplicates (cons elem func-table) :key #'kerdef-name :from-end t)
            var-table))))

(defun remove-macro-from-kernel-definition (name def)
  (unless (kernel-definition-macro-exists-p name def)
    (error "undefined kernel definition macro: ~A" name))
  (destructuring-bind (func-table var-table) def
    (list (remove name func-table :key #'kerdef-name) var-table)))

(defun add-constant-to-kernel-definition (name type expression def)
  (destructuring-bind (func-table var-table) def
    (let ((elem (make-kerdef-constant name type expression)))
      (list func-table
            (remove-duplicates (cons elem var-table) :key #'kerdef-name :from-end t)))))

(defun remove-constant-from-kernel-definition (name def)
  (unless (kernel-definition-constant-exists-p name def)
    (error "undefined kernel definition constant: ~A" name))
  (destructuring-bind (func-table var-table) def
    (list func-table (remove name var-table :key #'kerdef-name))))

(defun add-symbol-macro-to-kernel-definition (name expansion def)
  (destructuring-bind (funct-table var-table) def
    (let ((elem (make-kerdef-symbol-macro name expansion)))
      (list funct-table
            (remove-duplicates (cons elem var-table) :key #'kerdef-name :from-end t)))))

(defun remove-symbol-macro-from-kernel-definition (name def)
  (unless (kernel-definition-symbol-macro-exists-p name def)
    (error "undefined kernel definition symbol macro: ~A" name))
  (destructuring-bind (func-table var-table) def
    (list func-table (remove name var-table :key #'kerdef-name))))

(defun bulk-add-kernel-definition (bindings def)
  (reduce #'(lambda (def2 binding)
              (match binding
                ((name :function return-type args body)
                 (add-function-to-kernel-definition name return-type args body def2))
                ((name :macro args body expander)
                 (add-macro-to-kernel-definition name args body expander def2))
                ((name :constant type exp)
                 (add-constant-to-kernel-definition name type exp def2))
                ((name :symbol-macro expansion)
                 (add-symbol-macro-to-kernel-definition name expansion def2))
                (_ (error "invalid kernel definition element: ~A" binding))))
          bindings :initial-value def))

(defmacro with-kernel-definition ((def bindings) &body body)
  (labels ((aux (binding)
             (match binding
               ((name :function return-type args body) `(list ',name :function ',return-type ',args ',body))
               ((name :macro args body) (alexandria:with-gensyms (args0)
                                          `(list ',name :macro ',args ',body
                                                 (lambda (,args0) (destructuring-bind ,args ,args0 ,@body)))))
               ((name :constant type exp) `(list ',name :constant ',type ',exp))
               ((name :symbol-macro expansion) `(list ',name :symbol-macro ',expansion))
               (_ `',binding))))
    (let ((bindings2 `(list ,@(mapcar #'aux bindings))))
      `(let ((,def (bulk-add-kernel-definition ,bindings2 (empty-kernel-definition))))
         ,@body))))

(defun lookup-kernel-definition (kind name def)
  (destructuring-bind (func-table var-table) def
    (ecase kind
      (:function (let ((elem (find name func-table :key #'kerdef-name)))
                   (when (kerdef-function-p elem)
                     elem)))
      (:macro (let ((elem (find name func-table :key #'kerdef-name)))
                (when (kerdef-macro-p elem)
                  elem)))
      (:constant (let ((elem (find name var-table :key #'kerdef-name)))
                   (when (kerdef-constant-p elem)
                     elem)))
      (:symbol-macro (let ((elem (find name var-table :key #'kerdef-name)))
                       (when (kerdef-symbol-macro-p elem)
                         elem))))))

(defun kernel-definition-function-exists-p (name def)
  (and (lookup-kernel-definition :function name def)
       t))

(defun kernel-definition-macro-exists-p (name def)
  (and (lookup-kernel-definition :macro name def)
       t))

(defun kernel-definition-constant-exists-p (name def)
  (and (lookup-kernel-definition :constant name def)
       t))

(defun kernel-definition-symbol-macro-exists-p (name def)
  (and (lookup-kernel-definition :symbol-macro name def)
       t))

(defun kernel-definition-function-name (name def)
  (unless (kernel-definition-function-exists-p name def)
    (error "undefined kernel definition function: ~A" name))
  (kerdef-function-name (lookup-kernel-definition :function name def)))

(defun kernel-definition-function-c-name (name def)
  (unless (kernel-definition-function-exists-p name def)
    (error "undefined kernel definition function: ~A" name))
  (kerdef-function-c-name (lookup-kernel-definition :function name def)))

(defun kernel-definition-function-names (def)
  (destructuring-bind (func-table _) def
    (declare (ignorable _))
    (mapcar #'kerdef-name (remove-if-not #'kerdef-function-p func-table))))

(defun kernel-definition-function-return-type (name def)
  (unless (kernel-definition-function-exists-p name def)
    (error "undefined kernel definition function: ~A" name))
  (kerdef-function-return-type (lookup-kernel-definition :function name def)))

(defun kernel-definition-function-arguments (name def)
  (unless (kernel-definition-function-exists-p name def)
    (error "undefined kernel definition function: ~A" name))
  (kerdef-function-arguments (lookup-kernel-definition :function name def)))

(defun kernel-definition-function-argument-types (name def)
  (unless (kernel-definition-function-exists-p name def)
    (error "undefined kernel definition function: ~A" name))
  (kerdef-function-argument-types (lookup-kernel-definition :function name def)))

(defun kernel-definition-function-body (name def)
  (unless (kernel-definition-function-exists-p name def)
    (error "undefined kernel definition function: ~A" name))
  (kerdef-function-body (lookup-kernel-definition :function name def)))

(defun kernel-definition-macro-name (name def)
  (unless (kernel-definition-macro-exists-p name def)
    (error "undefined kernel definition macro: ~A" name))
  (kerdef-macro-name (lookup-kernel-definition :macro name def)))

(defun kernel-definition-macro-names (def)
  (destructuring-bind (func-table _) def
    (declare (ignorable _))
    (mapcar #'kerdef-name (remove-if-not #'kerdef-macro-p func-table))))

(defun kernel-definition-macro-arguments (name def)
  (unless (kernel-definition-macro-exists-p name def)
    (error "undefined kernel definition macro: ~A" name))
  (kerdef-macro-arguments (lookup-kernel-definition :macro name def)))

(defun kernel-definition-macro-body (name def)
  (unless (kernel-definition-macro-exists-p name def)
    (error "undefined kernel definition macro: ~A" name))
  (kerdef-macro-body (lookup-kernel-definition :macro name def)))

(defun kernel-definition-macro-expander (name def)
  (unless (kernel-definition-macro-exists-p name def)
    (error "undefined kernel definition macro: ~A" name))
  (kerdef-macro-expander (lookup-kernel-definition :macro name def)))

(defun kernel-definition-constant-name (name def)
  (unless (kernel-definition-constant-exists-p name def)
    (error "undefined kernel definition constant: ~A" name))
  (kerdef-constant-name (lookup-kernel-definition :constant name def)))

(defun kernel-definition-constant-names (def)
  (destructuring-bind (_ var-table) def
    (declare (ignorable _))
    (mapcar #'kerdef-name (remove-if-not #'kerdef-constant-p var-table))))

(defun kernel-definition-constant-type (name def)
  (unless (kernel-definition-constant-exists-p name def)
    (error "undefined kernel definition constant: ~A" name))
  (kerdef-constant-type (lookup-kernel-definition :constant name def)))

(defun kernel-definition-constant-expression (name def)
  (unless (kernel-definition-constant-exists-p name def)
    (error "undefined kernel definition constant: ~A" name))
  (kerdef-constant-expression (lookup-kernel-definition :constant name def)))

(defun kernel-definition-symbol-macro-name (name def)
  (unless (kernel-definition-symbol-macro-exists-p name def)
    (error "undefined kernel definition symbol macro: ~A" name))
  (kerdef-symbol-macro-name (lookup-kernel-definition :symbol-macro name def)))

(defun kernel-definition-symbol-macro-names (def)
  (destructuring-bind (_ var-table) def
    (declare (ignorable _))
    (mapcar #'kerdef-name (remove-if-not #'kerdef-symbol-macro-p var-table))))

(defun kernel-definition-symbol-macro-expansion (name def)
  (unless (kernel-definition-symbol-macro-exists-p name def)
    (error "undefined kernel definition symbol macro: ~A" name))
  (kerdef-symbol-macro-expansion (lookup-kernel-definition :symbol-macro name def)))


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
  (list "-arch=sm_20"))

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
;;; Compiling
;;;

(defun compile-function-specifier (return-type)
  (unless (valid-type-p return-type)
    (error (format nil "invalid return type: ~A" return-type)))
  (if (eq return-type 'void)
      "__global__"
      "__device__"))

(defun compile-type (type)
  (unless (valid-type-p type)
    (error (format nil "invalid type: ~A" type)))
  (cond ((eq type 'curand-state-xorwow)
         "curandStateXORWOW_t")
        ((eq type 'curand-state-xorwow*)
         "curandStateXORWOW_t *")
        (t
         (compile-identifier (princ-to-string type)))))

(defun compile-argument (arg)
  (destructuring-bind (var type) arg
    (format nil "~A ~A" (compile-type type) (compile-identifier var))))

(defun compile-arguments (args)
  (join ", " (mapcar #'compile-argument args)))

(defun compile-function-declaration (name def)
  (let ((c-name (kernel-definition-function-c-name name def))
        (arguments (kernel-definition-function-arguments name def))
        (return-type (kernel-definition-function-return-type name def)))
    (let ((specifier (compile-function-specifier return-type))
          (type (compile-type return-type))
          (args (compile-arguments arguments)))
      (format nil "~A ~A ~A (~A)" specifier type c-name args))))

(defun compile-kernel-constant (name def)
  (let ((type (kernel-definition-constant-type name def))
        (exp (kernel-definition-constant-expression name def)))
    (let ((var-env  (make-variable-environment-with-kernel-definition nil def))
          (func-env (make-function-environment-with-kernel-definition def)))
      (let ((type2 (compile-type type))
            (name2 (compile-identifier name))
            (exp2  (compile-expression exp var-env func-env)))
        (format nil "static const ~A ~A = ~A;" type2 name2 exp2)))))

(defun compile-kernel-constants (def)
  (mapcar #'(lambda (name)
              (compile-kernel-constant name def))
          (reverse (kernel-definition-constant-names def))))

(defun compile-kernel-function-prototype (name def)
  (format nil "extern \"C\" ~A;"
          (compile-function-declaration name def)))

(defun compile-kernel-function-prototypes (def)
  (mapcar #'(lambda (name)
              (compile-kernel-function-prototype name def))
          (reverse (kernel-definition-function-names def))))

(defun compile-function-statements (name def)
  (let ((var-env  (make-variable-environment-with-kernel-definition name def))
        (func-env (make-function-environment-with-kernel-definition def)))
    (mapcar #'(lambda (stmt)
                (compile-statement stmt var-env func-env))
            (kernel-definition-function-body name def))))

(defun compile-kernel-function (name def)
  (let ((declaration (compile-function-declaration name def))
        (statements  (mapcar #'(lambda (stmt)
                                 (indent 2 stmt))
                             (compile-function-statements name def))))
    (unlines `(,declaration
               "{"
               ,@statements
               "}"
               ""))))

(defun compile-kernel-functions (def)
  (mapcar #'(lambda (name)
              (compile-kernel-function name def))
          (reverse (kernel-definition-function-names def))))

(defun compile-kernel-definition (def)
  (unlines `("#include \"int.h\""
             "#include \"float.h\""
             "#include \"float3.h\""
             "#include \"float4.h\""
             "#include \"double.h\""
             "#include \"double3.h\""
             "#include \"double4.h\""
             "#include \"curand.h\""
             ""
             ,@(compile-kernel-function-prototypes def)
             ""
             ,@(compile-kernel-functions def))))
  

;;; compile statement

(defun compile-statement (stmt var-env func-env)
  (cond
    ((macro-form-p stmt func-env) (compile-macro stmt var-env func-env :statement-p t))
    ((if-p stmt) (compile-if stmt var-env func-env))
    ((let-p stmt) (compile-let stmt var-env func-env))
    ((let*-p stmt) (compile-let* stmt var-env func-env))
    ((symbol-macrolet-p stmt) (compile-symbol-macrolet stmt var-env func-env))
    ((do-p stmt) (compile-do stmt var-env func-env))
    ((with-shared-memory-p stmt) (compile-with-shared-memory stmt var-env func-env))
    ((set-p stmt) (compile-set stmt var-env func-env))
    ((progn-p stmt) (compile-progn stmt var-env func-env))
    ((return-p stmt) (compile-return stmt var-env func-env))
    ((syncthreads-p stmt) (compile-syncthreads stmt))
    ((function-p stmt) (compile-function stmt var-env func-env :statement-p t))
    (t (error "invalid statement: ~A" stmt))))


;;; if statement

(defun if-p (stmt)
  (match stmt
    (('if _ _) t)
    (('if _ _ _) t)
    (_ nil)))

(defun if-test-expression (stmt)
  (match stmt
    (('if test-exp _) test-exp)
    (('if test-exp _ _) test-exp)
    (_ (error "invalid statement: ~A" stmt))))

(defun if-then-statement (stmt)
  (match stmt
    (('if _ then-stmt) then-stmt)
    (('if _ then-stmt _) then-stmt)
    (_ (error "invalid statement: ~A" stmt))))

(defun if-else-statement (stmt)
  (match stmt
    (('if _ _) nil)
    (('if _ _ else-stmt) else-stmt)
    (_ (error "invalid statement: ~A" stmt))))

(defun compile-if (stmt var-env func-env)
  (let ((test-exp  (if-test-expression stmt))
        (then-stmt (if-then-statement stmt))
        (else-stmt (if-else-statement stmt)))
    (let ((test-type (type-of-expression test-exp var-env func-env)))
      (unless (eq test-type 'bool)
        (error "invalid type: type of test-form is ~A, not ~A" test-type 'bool)))
    (unlines (format nil "if (~A) {"
                     (compile-expression test-exp var-env func-env))
             (indent 2 (compile-statement then-stmt var-env func-env))
             (and else-stmt "} else {")
             (and else-stmt
                  (indent 2 (compile-statement else-stmt var-env func-env)))
             "}")))


;;; let statement

(defun let-p (stmt)
  (match stmt
    (('let . _) t)
    (_ nil)))

(defun let*-p (stmt)
  (match stmt
    (('let* . _) t)
    (_ nil)))

(defun let-bindings (stmt)
  (match stmt
    (('let bindings . _) bindings)
    (_ (error "invalid statement: ~A" stmt))))

(defun let*-bindings (stmt)
  (match stmt
    (('let* bindings . _) bindings)))

(defun let-statements (stmt)
  (match stmt
    (('let _ . stmts) stmts)
    (_ (error "invalid statement: ~A" stmt))))

(defun let*-statements (stmt)
  (match stmt
    (('let* _ . stmts) stmts)
    (_ (error "invalid statement: ~A" stmt))))

(defun %compile-assignment (var exp type var-env func-env)
  (let ((var2  (compile-identifier var))
        (exp2  (compile-expression exp var-env func-env))
        (type2 (compile-type type)))
    (format nil "~A ~A = ~A;" type2 var2 exp2)))

(defun compile-let-assignments (bindings var-env func-env)
  (labels ((aux (binding)
             (match binding
               ((var exp) (let ((type (type-of-expression exp var-env func-env)))
                            (%compile-assignment var exp type var-env func-env)))
               (_ (error "invalid let binding: ~A" binding)))))
    (let ((compiled-assignments (mapcar #'aux bindings)))
      (apply #'unlines compiled-assignments))))

(defun compile-let-statements (stmts var-env func-env)
  (compile-statement `(progn ,@stmts) var-env func-env))

(defun compile-let (stmt var-env func-env)
  (labels ((aux (binding)
             (match binding
               ((var exp) (let ((type (type-of-expression exp var-env func-env)))
                            (list var :variable type)))
               (_ (error "invalid let binding: ~A" binding)))))
    (let ((bindings  (let-bindings stmt))
          (let-stmts (let-statements stmt)))
      (let ((var-env2 (bulk-add-variable-environment (mapcar #'aux bindings) var-env)))
        (let ((assignments (compile-let-assignments bindings var-env func-env))
              (compiled-stmts (compile-let-statements let-stmts var-env2 func-env)))
          (unlines "{"
                   (indent 2 assignments)
                   (indent 2 compiled-stmts)
                   "}"))))))

(defun compile-let*-binding (bindings stmts var-env func-env)
  (match bindings
    (((var exp) . rest)
     (let ((type (type-of-expression exp var-env func-env)))
       (let ((assignment (%compile-assignment var exp type var-env func-env))
             (var-env2   (add-variable-to-variable-environment var type var-env)))
         (unlines assignment
                  (%compile-let* rest stmts var-env2 func-env)))))
    (_ (error "invalid bindings: ~A" bindings))))

(defun %compile-let* (bindings stmts var-env func-env)
  (if bindings
      (compile-let*-binding bindings stmts var-env func-env)
      (compile-let-statements stmts var-env func-env)))

(defun compile-let* (stmt var-env func-env)
  (let ((bindings  (let*-bindings stmt))
        (let-stmts (let*-statements stmt)))
    (unlines "{"
             (indent 2 (%compile-let* bindings let-stmts var-env func-env))
             "}")))


;;; symbol-macrolet statement

(defun symbol-macrolet-p (stmt)
  (match stmt
    (('symbol-macrolet . _) t)
    (_ nil)))

(defun symbol-macrolet-bindings (stmt)
  (match stmt
    (('symbol-macrolet bindings . _) bindings)
    (_ (error "invalid statement: ~A" stmt))))

(defun symbol-macrolet-statements (stmt)
  (match stmt
    (('symbol-macrolet _ . stmts) stmts)
    (_ (error "invalid statement: ~A" stmt))))

(defun compile-symbol-macrolet (stmt var-env func-env)
  (labels ((aux (binding)
             (match binding
               ((name expansion) (list name :symbol-macro expansion))
               (_ (error "invalid symbol-macrolet binding: ~A" binding)))))
    (let ((bindings (symbol-macrolet-bindings stmt))
          (stmts    (symbol-macrolet-statements stmt)))
      (let ((var-env2 (bulk-add-variable-environment (mapcar #'aux bindings) var-env)))
        (compile-statement `(progn ,@stmts) var-env2 func-env)))))


;;; set statement

(defun set-p (stmt)
  (match stmt
    (('set _ _) t)
    (_ nil)))

(defun set-place (stmt)
  (match stmt
    (('set place _) place)
    (_ (error "invalid statement: ~A" stmt))))

(defun set-expression (stmt)
  (match stmt
    (('set _ exp) exp)
    (_ (error "invalid statement: ~A" stmt))))

(defun compile-set (stmt var-env func-env)
  (let ((place (set-place stmt))
        (exp (set-expression stmt)))
    (let ((place-type (type-of-expression place var-env func-env))
          (exp-type   (type-of-expression exp   var-env func-env)))
      (unless (eq place-type exp-type)
        (error "invalid types: type of the place is ~A but that of the expression is ~A" place-type exp-type)))
    (format nil "~A = ~A;" (compile-place place var-env func-env)
                           (compile-expression exp var-env func-env))))

(defun compile-place (place var-env func-env)
  (cond ((symbol-place-p place) (compile-symbol-place place var-env func-env))
        ((vector-place-p place) (compile-vector-place place var-env func-env))
        ((array-place-p place)  (compile-array-place place var-env func-env))
        (t (error "invalid place: ~A" place))))

(defun symbol-place-p (place)
  (symbol-p place))

(defun vector-place-p (place)
  (vector-variable-reference-p place))

(defun array-place-p (place)
  (array-variable-reference-p place))

(defun compile-symbol-place (place var-env func-env)
  (compile-symbol place var-env func-env))

(defun compile-vector-place (place var-env func-env)
  (compile-vector-variable-reference place var-env func-env))

(defun compile-array-place (place var-env func-env)
  (compile-array-variable-reference place var-env func-env))


;;; progn statement

(defun progn-p (stmt)
  (match stmt
    (('progn . _) t)
    (_ nil)))

(defun progn-statements (stmt)
  (match stmt
    (('progn . stmts) stmts)
    (_ (error "invalid statement: ~A" stmt))))

(defun compile-progn-statements (stmts var-env func-env)
  (let ((compiled-stmts (mapcar #'(lambda (stmt)
                                    (compile-statement stmt var-env func-env))
                                stmts)))
    (unlines compiled-stmts stmts)))

(defun compile-progn (stmt var-env func-env)
  (let ((stmts (progn-statements stmt)))
    (compile-progn-statements stmts var-env func-env)))


;;; return statement

(defun return-p (stmt)
  (match stmt
    (('return) t)
    (('return _) t)
    (_ nil)))

(defun compile-return (stmt var-env func-env)
  (match stmt
    (('return) "return;")
    (('return exp) (format nil "return ~A;"
                               (compile-expression exp var-env func-env)))
    (_ (error "invalid statement: ~A" stmt))))


;;; do statement

(defun do-p (stmt)
  (match stmt
    (('do . _) t)
    (_ nil)))

(defun do-bindings (stmt)
  (match stmt
    (('do bindings . _) bindings)
    (_ (error "invalid statement: ~A" stmt))))

(defun do-var-types (stmt var-env func-env)
  (labels ((do-var-type (binding)
             (list (do-binding-var binding)
                   :variable
                   (do-binding-type binding var-env func-env))))
    (mapcar #'do-var-type (do-bindings stmt))))

(defun do-binding-var (binding)
  (match binding
    ((var _)   var)
    ((var _ _) var)
    (_ (error "invalid binding: ~A" binding))))

(defun do-binding-type (binding var-env func-env)
  (type-of-expression (do-binding-init-form binding) var-env func-env))

(defun do-binding-init-form (binding)
  (match binding
    ((_ init-form)   init-form)
    ((_ init-form _) init-form)
    (_ (error "invalid binding: ~A" binding))))

(defun do-binding-step-form (binding)
  (match binding
    ((_ _)           nil)
    ((_ _ step-form) step-form)
    (_ (error "invalid binding: ~A" binding))))

(defun do-test-form (stmt)
  (match stmt
    (('do _ (test-form) . _) test-form)
    (_ (error "invalid statement: ~A" stmt))))

(defun do-statements (stmt)
  (match stmt
    (('do _ _ . stmts) stmts)
    (_ (error "invalid statement: ~A" stmt))))

(defun compile-do (stmt var-env func-env)
  (let ((var-env2 (bulk-add-variable-environment (do-var-types stmt var-env func-env) var-env)))
    (let ((init-part (compile-do-init-part stmt var-env func-env))
          (test-part (compile-do-test-part stmt var-env2 func-env))
          (step-part (compile-do-step-part stmt var-env2 func-env)))
      (unlines (format nil "for ( ~A; ~A; ~A )" init-part test-part step-part)
               "{"
               (indent 2 (compile-do-statements stmt var-env2 func-env))
               "}"))))

(defun compile-do-init-part (stmt var-env func-env)
  (labels ((aux (binding)
             (let ((var (do-binding-var binding))
                   (type (do-binding-type binding var-env func-env))
                   (init-form (do-binding-init-form binding)))
               (format nil "~A ~A = ~A" (compile-type type)
                                        (compile-identifier var)
                                        (compile-expression init-form var-env func-env)))))
    (join ", " (mapcar #'aux (do-bindings stmt)))))

(defun compile-do-test-part (stmt var-env func-env)
  (let ((test-form (do-test-form stmt)))
    (format nil "! ~A" (compile-expression test-form var-env func-env))))

(defun compile-do-step-part (stmt var-env func-env)
  (labels ((aux (binding)
             (let ((var (do-binding-var binding))
                   (step-form (do-binding-step-form binding)))
               (format nil "~A = ~A" (compile-identifier var)
                                     (compile-expression step-form var-env func-env)))))
    (join ", " (mapcar #'aux (remove-if-not #'do-binding-step-form (do-bindings stmt))))))

(defun compile-do-statements (stmt var-env func-env)
  (compile-progn-statements (do-statements stmt) var-env func-env))


;;; with-shared-memory statement

(defun with-shared-memory-p (stmt)
  (match stmt
    (('with-shared-memory . _) t)
    (_ nil)))

(defun with-shared-memory-specs (stmt)
  (match stmt
    (('with-shared-memory specs . _) specs)
    (_ (error "invalid statement: ~A" stmt))))

(defun with-shared-memory-statements (stmt)
  (match stmt
    (('with-shared-memory _ . stmts) stmts)
    (_ (error "invalid statement: ~A" stmt))))

(defun compile-with-shared-memory-statements (stmts var-env func-env)
  (compile-let-statements stmts var-env func-env))

(defun compile-with-shared-memory-spec (specs stmts var-env func-env)
  (match specs
    (((var type . sizes) . rest)
     (let* ((type2 (add-star type (length sizes)))
            (var-env2 (add-variable-to-variable-environment var type2 var-env)))
       (unlines (format nil "__shared__ ~A ~A~{[~A]~};"
                            (compile-type type)
                            (compile-identifier var)
                            (mapcar #'(lambda (exp)
                                        (compile-expression exp var-env func-env))
                                    sizes))
                (%compile-with-shared-memory rest stmts var-env2 func-env))))
    (_ (error "invalid shared memory specs: ~A" specs))))

(defun %compile-with-shared-memory (specs stmts var-env func-env)
  (if (null specs)
      (compile-with-shared-memory-statements stmts var-env func-env)
      (compile-with-shared-memory-spec specs stmts var-env func-env)))

(defun compile-with-shared-memory (stmt var-env func-env)
  (let ((specs (with-shared-memory-specs stmt))
        (stmts (with-shared-memory-statements stmt)))
    (unlines "{"
             (indent 2 (%compile-with-shared-memory specs stmts var-env func-env))
             "}")))


;;; compile syncthreads

(defun syncthreads-p (stmt)
  (match stmt
    (('syncthreads) t)
    (_ nil)))

(defun compile-syncthreads (stmt)
  (declare (ignorable stmt))
  "__syncthreads();")


;;; compile function

(defun function-p (form)
  (and (listp form)
       (car form)
       (symbolp (car form))))

(defun defined-function-p (form func-env)
  (or (built-in-function-p form)
      (user-function-p form func-env)))

(defun built-in-function-p (form)
  (match form
    ((op . _) (and (getf +built-in-functions+ op) t))
    (_ nil)))

(defun user-function-p (form func-env)
  (match form
    ((op . _) (function-environment-function-exists-p op func-env))
    (_ nil)))

(defun function-operator (form)
  (unless (function-p form)
    (error "invalid statement or expression: ~A" form))
  (car form))

(defun function-operands (form)
  (unless (function-p form)
    (error "invalid statement or expression: ~A" form))
  (cdr form))

(defun compile-function (form var-env func-env &key (statement-p nil))
  (unless (defined-function-p form func-env)
    (error "undefined function: ~A" form))
  (let ((code (if (built-in-function-p form)
                  (compile-built-in-function form var-env func-env)
                  (compile-user-function form var-env func-env))))
    (if statement-p
        (format nil "~A;" code)
        code)))

(defun compile-built-in-function (form var-env func-env)
  (let ((op (function-operator form)))
    (cond
      ((built-in-function-infix-p form var-env func-env)
       (compile-built-in-infix-function form var-env func-env))
      ((built-in-function-prefix-p form var-env func-env)
       (compile-built-in-prefix-function form var-env func-env))
      (t (error "invalid built-in function: ~A" op)))))

(defun compile-built-in-infix-function (form var-env func-env)
  (let ((operands (function-operands form)))
    (let ((op  (built-in-function-c-string form var-env func-env))
          (lhe (compile-expression (car operands) var-env func-env))
          (rhe (compile-expression (cadr operands) var-env func-env)))
      (format nil "(~A ~A ~A)" lhe op rhe))))

(defun compile-built-in-prefix-function (form var-env func-env)
  (let ((operands (function-operands form)))
    (format nil "~A (~A)"
            (built-in-function-c-string form var-env func-env)
            (compile-operands operands var-env func-env))))

(defun type-of-operands (operands var-env func-env)
  (mapcar #'(lambda (exp)
              (type-of-expression exp var-env func-env))
          operands))

(defun compile-operands (operands var-env func-env)
  (join ", " (mapcar #'(lambda (exp)
                         (compile-expression exp var-env func-env))
                     operands)))

(defun compile-user-function (form var-env func-env)
  (let ((operator (function-operator form))
        (operands (function-operands form)))
    (let ((expected-types (function-environment-function-argument-types operator func-env))
          (actual-types (type-of-operands operands var-env func-env)))
      (unless (equal expected-types actual-types)
        (error "invalid arguments: ~A" form)))
    (let ((func (function-environment-function-c-name operator func-env))
          (compiled-operands (compile-operands operands var-env func-env)))
      (format nil "~A (~A)" func compiled-operands))))


;;; compile macro

(defun macro-form-p (form func-env)
  "Returns t if the given form is a macro form. The macro used in the
form may be an user-defined macro under the given kernel definition or
a built-in macro."
  (or (built-in-macro-p form)
      (user-macro-p form func-env)))

(defun built-in-macro-p (form)
  (match form
    ((op . _) (and (getf +built-in-macros+ op) t))
    (_ nil)))

(defun user-macro-p (form func-env)
  (match form
    ((op . _) (function-environment-macro-exists-p op func-env))
    (_ nil)))

(defun macro-operator (form func-env)
  (unless (macro-form-p form func-env)
    (error "undefined macro form: ~A" form))
  (car form))

(defun macro-operands (form func-env)
  (unless (macro-form-p form func-env)
    (error "undefined macro form: ~A" form))
  (cdr form))

(defun compile-macro (form var-env func-env &key (statement-p nil))
  (unless (macro-form-p form func-env)
    (error "undefined macro: ~A" form))
  (if statement-p
      (compile-statement  (%expand-macro-1 form func-env) var-env func-env)
      (compile-expression (%expand-macro-1 form func-env) var-env func-env)))

(defun %expand-built-in-macro-1 (form func-env)
  (let ((operator (macro-operator form func-env))
        (operands (macro-operands form func-env)))
    (let ((expander (built-in-macro-expander operator)))
      (values (funcall expander operands) t))))

(defun %expand-user-macro-1 (form func-env)
  (let ((operator (macro-operator form func-env))
        (operands (macro-operands form func-env)))
    (let ((expander (function-environment-macro-expander operator func-env)))
      (values (funcall expander operands) t))))

(defun %expand-macro-1 (form func-env)
  (if (macro-form-p form func-env)
      (if (built-in-macro-p form)
          (%expand-built-in-macro-1 form func-env)
          (%expand-user-macro-1 form func-env))
      (values form nil)))

(defun expand-macro-1 (form)
  "If a form is a macro form, then EXPAND-MACRO-1 expands the macro
form call once, and returns the macro expansion and true as values.
Otherwise, returns the given form and false as values."
  (let ((def (kernel-definition *kernel-manager*)))
    (let ((func-env (make-function-environment-with-kernel-definition def)))
      (%expand-macro-1 form func-env))))

(defun %expand-macro (form func-env)
  (if (macro-form-p form func-env)
      (values (%expand-macro (%expand-macro-1 form func-env) func-env) t)
      (values form nil)))

(defun expand-macro (form)
  "If a form is a macro form, then EXPAND-MACRO repeatedly expands
the macro form until it is no longer a macro form, and returns the
macro expansion and true as values. Otherwise, returns the given form
and false as values."
  (let ((def (kernel-definition *kernel-manager*)))
    (let ((func-env (make-function-environment-with-kernel-definition def)))
      (%expand-macro form func-env))))


;;; built-in functions
;;;   <built-in-functions>  ::= plist { <function-name> => <function-info> }
;;;   <function-info>       ::= (<infix-p> <function-candidates>)
;;;   <function-candidates> ::= (<function-candidate>*)
;;;   <function-candidate>  ::= (<arg-types> <return-type> <function-c-name>)
;;;   <arg-types>           ::= (<arg-type>*)

(defparameter +built-in-functions+
  '(%add (((int    int)    int    t   "+")
          ((float  float)  float  t   "+")
          ((float3 float3) float3 nil "float3_add")
          ((float4 float4) float4 nil "float4_add")
          ((double  double)  double  t   "+")
          ((double3 double3) double3 nil "double3_add")
          ((double4 double4) double4 nil "double4_add"))
    %sub (((int    int)    int    t   "-")
          ((float  float)  float  t   "-")
          ((float3 float3) float3 nil "float3_sub")
          ((float4 float4) float4 nil "float4_sub")
          ((double  double)  double  t   "-")
          ((double3 double3) double3 nil "double3_sub")
          ((double4 double4) double4 nil "double4_sub"))
    %mul (((int    int)    int    t   "*")
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
    %div (((int    int)    int    t   "/")
          ((float  float)  float  t   "/")
          ((float3 float)  float3 nil "float3_scale_inverted")
          ((float4 float)  float4 nil "float4_scale_inverted")
          ((double  double)  double  t   "/")
          ((double3 double)  double3 nil "double3_scale_inverted")
          ((double4 double)  double4 nil "double4_scale_inverted"))
    %negate (((int)    int    nil "int_negate")
             ((float)  float  nil "float_negate")
             ((float3) float3 nil "float3_negate")
             ((float4) float4 nil "float4_negate")
             ((double)  double  nil "double_negate")
             ((double3) double3 nil "double3_negate")
             ((double4) double4 nil "double4_negate"))
    %recip (((int)    int    nil "int_recip")
            ((float)  float  nil "float_recip")
            ((float3) float3 nil "float3_recip")
            ((float4) float4 nil "float4_recip")
            ((double)  double  nil "double_recip")
            ((double3) double3 nil "double3_recip")
            ((double4) double4 nil "double4_recip"))
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
    not  (((bool) bool nil "!"))
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
    rsqrtf (((float) float nil "rsqrtf")
            ((double) double nil "rsqrt"))
    sqrt   (((float) float nil "sqrtf")
            ((double) double nil "sqrt"))
    floor  (((float) int   nil "floorf")
            ((double) int   nil "floor"))
    atomic-add (((int* int) int nil "atomicAdd"))
    pointer (((int)   int*   nil "&")
             ((float) float* nil "&")
             ((double) double* nil "&")
             ((curand-state-xorwow) curand-state-xorwow* nil "&"))
    float3 (((float float float) float3 nil "make_float3"))
    float4 (((float float float float) float4 nil "make_float4"))
    double3 (((double double double) double3 nil "make_double3"))
    double4 (((double double double double) double4 nil "make_double4"))
    double-to-int-rn (((double) int nil "__double2int_rn"))
    dot (((float3 float3) float nil "float3_dot")
         ((float4 float4) float nil "float4_dot")
         ((double3 double3) double nil "double3_dot")
         ((double4 double4) double nil "double4_dot"))
    ;; It's :UNSIGNED-LONG-LONG, but this wrapper function only
    ;; supports INT.
    curand-init-xorwow (((int int int curand-state-xorwow*) void nil
                         "curand_init_xorwow"))
    curand-uniform-float-xorwow (((curand-state-xorwow*) float nil
                                  "curand_uniform_float_xorwow"))
    curand-uniform-double-xorwow (((curand-state-xorwow*) double nil
                                   "curand_uniform_double_xorwow"))))

(defun function-candidates (op)
  (or (getf +built-in-functions+ op)
      (error "invalid function: ~A" op)))

(defun inferred-function (form var-env func-env)
  (let ((operator (function-operator form))
        (operands (function-operands form)))
    (let ((candidates (function-candidates operator))
          (types (type-of-operands operands var-env func-env)))
      (or (find types candidates :key #'car :test #'equal)
          (error "invalid function application: ~A" form)))))

(defun inferred-function-argument-types (fun)
  (car fun))

(defun inferred-function-return-type (fun)
  (cadr fun))

(defun inferred-function-infix-p (fun)
  (caddr fun))

(defun inferred-function-prefix-p (fun)
  (not (inferred-function-infix-p fun)))

(defun inferred-function-c-string (fun)
  (cadddr fun))

(defun built-in-function-argument-types (form var-env func-env)
  (inferred-function-argument-types (inferred-function form var-env func-env)))

(defun built-in-function-return-type (form var-env func-env)
  (inferred-function-return-type (inferred-function form var-env func-env)))

(defun built-in-function-infix-p (form var-env func-env)
  (inferred-function-infix-p (inferred-function form var-env func-env)))

(defun built-in-function-prefix-p (form var-env func-env)
  (inferred-function-prefix-p (inferred-function form var-env func-env)))

(defun built-in-function-c-string (form var-env func-env)
  (inferred-function-c-string (inferred-function form var-env func-env)))


;;; built-in macros
;;;   <built-in-macros> ::= plist { <macro-name> => <macro-expander> }

(defvar +built-in-macros+
  (list '+ (lambda (args)
             (match args
               (() 0)
               ((a1) a1)
               ((a1 a2) `(%add ,a1 ,a2))
               ((a1 a2 . rest) `(+ (%add ,a1 ,a2) ,@rest))))
        '- (lambda (args)
             (match args
               (() (error "invalid number of arguments: 0"))
               ((a1) `(%negate ,a1))
               ((a1 a2) `(%sub ,a1 ,a2))
               ((a1 a2 . rest) `(- (%sub ,a1 ,a2) ,@rest))))
        '* (lambda (args)
             (match args
               (() 1)
               ((a1) a1)
               ((a1 a2) `(%mul ,a1 ,a2))
               ((a1 a2 . rest) `(* (%mul ,a1 ,a2) ,@rest))))
        '/ (lambda (args)
             (match args
               (() (error "invalid number of arguments: 0"))
               ((a1) `(%recip ,a1))
               ((a1 a2) `(%div ,a1 ,a2))
               ((a1 a2 . rest) `(/ (%div ,a1 ,a2) ,@rest))))))

(defun built-in-macro-expander (name)
  (or (getf +built-in-macros+ name)
      (error "invalid macro name: ~A" name)))


;;; compile expression

(defun compile-expression (exp var-env func-env)
  (cond
    ((macro-form-p exp func-env) (compile-macro exp var-env func-env))
    ((literal-p exp) (compile-literal exp))
    ((cuda-dimension-p exp) (compile-cuda-dimension exp))
    ((symbol-p exp) (compile-symbol exp var-env func-env))
    ((variable-reference-p exp)
     (compile-variable-reference exp var-env func-env))
    ((inline-if-p exp) (compile-inline-if exp var-env func-env))
    ((function-p exp) (compile-function exp var-env func-env))
    (t (error "invalid expression: ~A" exp))))

(defun variable-p (exp var-env)
  (variable-environment-variable-exists-p exp var-env))

(defun compile-variable (exp var-env)
  (unless (variable-environment-variable-exists-p exp var-env)
    (error "undefined variable: ~A" exp))
  (compile-identifier exp))

(defun constant-p (exp var-env)
  (variable-environment-constant-exists-p exp var-env))

(defun compile-constant (exp var-env)
  (unless (variable-environment-constant-exists-p exp var-env)
    (error "undefined constant: ~A" exp))
  (compile-identifier exp))

(defun symbol-macro-p (exp var-env)
  (variable-environment-symbol-macro-exists-p exp var-env))

(defun compile-symbol-macro (exp var-env func-env)
  (unless (variable-environment-symbol-macro-exists-p exp var-env)
    (error "undefined symbol macro: ~A" exp))
  (let ((expansion (variable-environment-symbol-macro-expansion exp var-env)))
    (compile-expression expansion var-env func-env)))

(defun symbol-p (exp)
  (symbolp exp))

(defun compile-symbol (exp var-env func-env)
  (cond
    ((variable-p exp var-env) (compile-variable exp var-env))
    ((constant-p exp var-env) (compile-constant exp var-env))
    ((symbol-macro-p exp var-env) (compile-symbol-macro exp var-env func-env))
    (t (error "undefined variable: ~A" exp))))

(defun literal-p (exp)
  (or (bool-literal-p exp)
      (int-literal-p exp)
      (float-literal-p exp)
      (double-literal-p exp)))

(defun bool-literal-p (exp)
  (typep exp 'boolean))

(defun int-literal-p (exp)
  (typep exp 'fixnum))

(defun float-literal-p (exp)
  (typep exp 'single-float))

(defun double-literal-p (exp)
  (typep exp 'double-float))

(defun compile-bool-literal (exp)
  (unless (typep exp 'boolean)
    (error "invalid literal: ~A" exp))
  (if exp "true" "false"))

(defun compile-int-literal (exp)
  (princ-to-string exp))

(defun compile-float-literal (exp)
  (princ-to-string exp))

(defun compile-double-literal (exp)
  (format nil "(double)~S" (float exp 0.0)))

(defun compile-literal (exp)
  (cond ((bool-literal-p  exp) (compile-bool-literal exp))
        ((int-literal-p   exp) (compile-int-literal exp))
        ((float-literal-p exp) (compile-float-literal exp))
        ((double-literal-p exp) (compile-double-literal exp))
        (t (error "invalid literal: ~A" exp))))

(defun cuda-dimension-p (exp)
  (or (grid-dim-p exp) (block-dim-p exp) (block-idx-p exp) (thread-idx-p exp)))

(defun grid-dim-p (exp)
  (find exp '(grid-dim-x grid-dim-y grid-dim-z)))

(defun block-dim-p (exp)
  (find exp '(block-dim-x block-dim-y block-dim-z)))

(defun block-idx-p (exp)
  (find exp '(block-idx-x block-idx-y block-idx-z)))

(defun thread-idx-p (exp)
  (find exp '(thread-idx-x thread-idx-y thread-idx-z)))

(defun compile-cuda-dimension (exp)
  (case exp
    (grid-dim-x "gridDim.x")
    (grid-dim-y "gridDim.y")
    (grid-dim-z "gridDim.z")
    (block-dim-x "blockDim.x")
    (block-dim-y "blockDim.y")
    (block-dim-z "blockDim.z")
    (block-idx-x "blockIdx.x")
    (block-idx-y "blockIdx.y")
    (block-idx-z "blockIdx.z")
    (thread-idx-x "threadIdx.x")
    (thread-idx-y "threadIdx.y")
    (thread-idx-z "threadIdx.z")
    (t (error "invalid expression: ~A" exp))))

(defun variable-reference-p (exp)
  (or (vector-variable-reference-p exp)
      (array-variable-reference-p exp)))

(defun vector-variable-reference-p (exp)
  (match exp
    ((selector _) (valid-vector-type-selector-p selector))
    (_ nil)))

(defun array-variable-reference-p (exp) 
  (match exp
    (('aref . _) t)
    (_ nil)))

(defun compile-variable-reference (exp var-env func-env)
  (cond ((vector-variable-reference-p exp)
         (compile-vector-variable-reference exp var-env func-env))
        ((array-variable-reference-p exp)
         (compile-array-variable-reference exp var-env func-env))
        (t (error "invalid expression: ~A" exp))))

(defun compile-vector-selector (selector)
  (unless (valid-vector-type-selector-p selector)
    (error "invalid selector: ~A" selector))
  (string-downcase (subseq (reverse (princ-to-string selector)) 0 1)))

(defun compile-vector-variable-reference (form var-env func-env)
  (match form
    ((selector exp)
     (let ((selector-type (vector-type-selector-type selector))
           (exp-type      (type-of-expression exp var-env func-env)))
       (unless (eq selector-type exp-type)
         (error "invalid variable reference: ~A" form))
       (format nil "~A.~A" (compile-expression exp var-env func-env)
                           (compile-vector-selector selector))))
    (_ (error "invalid variable reference: ~A" form))))

(defun compile-array-variable-reference (form var-env func-env)
  (match form
    (('aref _)
     (error "invalid variable reference: ~A" form))
    (('aref exp . idxs)
     (let ((type (type-of-expression exp var-env func-env)))
       (unless (= (array-type-dimension type) (length idxs))
         (error "invalid dimension: ~A" form))
       (format nil "~A~{[~A]~}"
                   (compile-expression exp var-env func-env)
                   (mapcar #'(lambda (idx)
                               (compile-expression idx var-env func-env)) idxs))))
    (_ (error "invalid variable reference: ~A" form))))

(defun inline-if-p (exp)
  (match exp
    (('if _ _ _) t)
    (_ nil)))

(defun inline-if-test-expression (exp)
  (match exp
    (('if test-exp _ _) test-exp)
    (_ (error "invalid expression: ~A" exp))))

(defun inline-if-then-expression (exp)
  (match exp
    (('if _ then-exp _) then-exp)
    (_ (error "invalid expression: ~A" exp))))

(defun inline-if-else-expression (exp)
  (match exp
    (('if _ _ else-exp) else-exp)
    (_ (error "invalid expression: ~A" exp))))

(defun compile-inline-if (exp var-env func-env)
  (let ((test-exp (inline-if-test-expression exp))
        (then-exp (inline-if-then-expression exp))
        (else-exp (inline-if-else-expression exp)))
    (let ((test-type (type-of-expression test-exp var-env func-env))
          (then-type (type-of-expression then-exp var-env func-env))
          (else-type (type-of-expression else-exp var-env func-env)))
      (unless (eq test-type 'bool)
        (error "invalid type: type of test-form is ~A, not ~A" test-type 'bool))
      (unless (eq then-type else-type)
        (error "invalid types: type of then-form is ~A but that of else-form is ~A" then-type else-type)))
    (format nil "(~A ? ~A : ~A)"
            (compile-expression test-exp var-env func-env)
            (compile-expression then-exp var-env func-env)
            (compile-expression else-exp var-env func-env))))


;;;
;;; Type of expression
;;;

(defun type-of-expression (exp var-env func-env)
  (cond ((macro-form-p exp func-env) (type-of-macro-form exp var-env func-env))
        ((literal-p exp) (type-of-literal exp))
        ((cuda-dimension-p exp) 'int)
        ((symbol-p exp) (type-of-symbol exp var-env func-env))
        ((variable-reference-p exp) (type-of-variable-reference exp var-env func-env))
        ((inline-if-p exp) (type-of-inline-if exp var-env func-env))
        ((function-p exp) (type-of-function exp var-env func-env))
        (t (error "invalid expression: ~A" exp))))

(defun type-of-literal (exp)
  (cond ((bool-literal-p exp) 'bool)
        ((int-literal-p exp) 'int)
        ((float-literal-p exp) 'float)
        ((double-literal-p exp) 'double)
        (t (error "invalid expression: ~A" exp))))

(defun type-of-variable (exp var-env)
  (unless (variable-environment-variable-exists-p exp var-env)
    (error "undefined variable: ~A" exp))
  (variable-environment-type-of-variable exp var-env))

(defun type-of-constant (exp var-env)
  (unless (variable-environment-constant-exists-p exp var-env)
    (error "undefined constant: ~A" exp))
  (variable-environment-type-of-constant exp var-env))

(defun type-of-symbol-macro (exp var-env func-env)
  (unless (variable-environment-symbol-macro-exists-p exp var-env)
    (error "undefined symbol macro: ~A" exp))
  (let ((expansion (variable-environment-symbol-macro-expansion exp var-env)))
    (type-of-expression expansion var-env func-env)))

(defun type-of-symbol (exp var-env func-env)
  (cond
    ((variable-p exp var-env) (type-of-variable exp var-env))
    ((constant-p exp var-env) (type-of-constant exp var-env))
    ((symbol-macro-p exp var-env) (type-of-symbol-macro exp var-env func-env))
    (t (error "undefined variable: ~A" exp))))

(defun type-of-variable-reference (exp var-env func-env)
  (cond ((vector-variable-reference-p exp)
         (type-of-vector-variable-reference exp var-env func-env))
        ((array-variable-reference-p exp)
         (type-of-array-variable-reference exp var-env func-env))
        (t (error "invalid expression: ~A" exp))))

(defun type-of-vector-variable-reference (exp var-env func-env)
  (match exp
    ((selector exp2)
     (let ((selector-type (vector-type-selector-type selector))
           (exp-type      (type-of-expression exp2 var-env func-env)))
       (unless (eq selector-type exp-type)
         (error "invalid variable reference: ~A" exp))
       (vector-type-base-type exp-type)))
    (_ (error "invalid variable reference: ~A" exp))))

(defun type-of-array-variable-reference (exp var-env func-env)
  (match exp
    (('aref _) (error "invalid variable reference: ~A" exp))
    (('aref exp2 . idxs)
     (let ((type (type-of-expression exp2 var-env func-env)))
       (unless (= (array-type-dimension type) (length idxs))
         (error "invalid dimension: ~A" exp))
       (remove-star type)))
    (_ (error "invalid variable reference: ~A" exp))))

(defun type-of-inline-if (exp var-env func-env)
  (let ((test-exp (inline-if-test-expression exp))
        (then-exp (inline-if-then-expression exp))
        (else-exp (inline-if-else-expression exp)))
    (let ((test-exp-type (type-of-expression test-exp var-env func-env))
          (then-exp-type (type-of-expression then-exp var-env func-env))
          (else-exp-type (type-of-expression else-exp var-env func-env)))
      (when (not (eq test-exp-type 'bool))
        (error "type of the test part of the inline if expression is not bool: ~A" exp))
      (when (not (eq then-exp-type else-exp-type))
        (error "types of the then part and the else part of the inline if expression are not same: ~A" exp))
      then-exp-type)))

(defun type-of-macro-form (exp var-env func-env)
  (type-of-expression (%expand-macro-1 exp func-env) var-env func-env))

(defun type-of-function (exp var-env func-env)
  (cond ((built-in-function-p exp)
         (type-of-built-in-function exp var-env func-env))
        ((user-function-p exp func-env)
         (type-of-user-function exp func-env))
        (t (error "invalid expression: ~A" exp))))

(defun type-of-built-in-function (exp var-env func-env)
  (built-in-function-return-type exp var-env func-env))

(defun type-of-user-function (exp func-env)
  (let ((operator (function-operator exp)))
    (unless (function-environment-function-exists-p operator func-env)
      (error "undefined function: ~A" operator))
    (function-environment-function-return-type operator func-env)))


;;;
;;; Variable environment
;;;

(defun make-varenv-variable (name type)
  (assert (symbolp name))
  (assert (valid-type-p type))
  (list name :variable type))

(defun varenv-variable-p (elem)
  (match elem
    ((_ :variable _) t)
    (_ nil)))

(defun varenv-variable-name (elem)
  (match elem
    ((name :variable _) name)
    (_ (error "invalid variable environment variable: ~A" elem))))

(defun varenv-variable-type (elem)
  (match elem
    ((_ :variable type) type)
    (_ (error "invalid variable environment variable: ~A" elem))))

(defun make-varenv-constant (name type)
  (assert (symbolp name))
  (assert (valid-type-p type))
  (list name :constant type))

(defun varenv-constant-p (elem)
  (match elem
    ((_ :constant _) t)
    (_ nil)))

(defun varenv-constant-name (elem)
  (match elem
    ((name :constant _) name)
    (_ (error "invalid variable environment constant: ~A" elem))))

(defun varenv-constant-type (elem)
  (match elem
    ((_ :constant type) type)
    (_ (error "invalid variable environment constant: ~A" elem))))

(defun make-varenv-symbol-macro (name expansion)
  (assert (symbolp name))
  (list name :symbol-macro expansion))

(defun varenv-symbol-macro-p (elem)
  (match elem
    ((_ :symbol-macro _) t)
    (_ nil)))

(defun varenv-symbol-macro-name (elem)
  (match elem
    ((name :symbol-macro _) name)
    (_ (error "invalid variable environment symbol macro: ~A" elem))))

(defun varenv-symbol-macro-expansion (elem)
  (match elem
    ((_ :symbol-macro expansion) expansion)
    (_ (error "invalid variable environment symbol macro: ~A" elem))))

(defun varenv-name (elem)
  (cond
    ((varenv-variable-p elem) (varenv-variable-name elem))
    ((varenv-constant-p elem) (varenv-constant-name elem))
    ((varenv-symbol-macro-p elem) (varenv-symbol-macro-name elem))
    (t (error "invalid variable environment element: ~A" elem))))

(defun empty-variable-environment ()
  '())

(defun add-variable-to-variable-environment (name type var-env)
  (let ((elem (make-varenv-variable name type)))
    (cons elem var-env)))

(defun add-constant-to-variable-environment (name type var-env)
  (let ((elem (make-varenv-constant name type)))
    (cons elem var-env)))

(defun add-symbol-macro-to-variable-environment (name expansion var-env)
  (let ((elem (make-varenv-symbol-macro name expansion)))
    (cons elem var-env)))

(defun bulk-add-variable-environment (bindings var-env)
  (reduce #'(lambda (var-env2 binding)
              (match binding
                ((name :variable type)
                 (add-variable-to-variable-environment name type var-env2))
                ((name :constant type)
                 (add-constant-to-variable-environment name type var-env2))
                ((name :symbol-macro expansion)
                 (add-symbol-macro-to-variable-environment name expansion var-env2))
                (_ (error "invalid variable environment element: ~A" binding))))
          bindings :initial-value var-env))

(defun %add-function-arguments (name def var-env)
  (if name
      (let ((arg-bindings (kernel-definition-function-arguments name def)))
        (reduce #'(lambda (var-env arg-binding)
                    (destructuring-bind (var type) arg-binding
                  (add-variable-to-variable-environment var type var-env)))
                arg-bindings :initial-value var-env))
      var-env))

(defun %add-symbol-macros (def var-env)
  (labels ((%symbol-macro-binding (name)
             (let ((name (kernel-definition-symbol-macro-name name def))
                   (expansion (kernel-definition-symbol-macro-expansion name def)))
               (list name :symbol-macro expansion))))
    (let ((symbol-macro-bindings (mapcar #'%symbol-macro-binding
                                         (kernel-definition-symbol-macro-names def))))
      (bulk-add-variable-environment symbol-macro-bindings var-env))))

(defun %add-constants (def var-env)
  (labels ((%constant-binding (name)
             (let ((name (kernel-definition-constant-name name def))
                   (type (kernel-definition-constant-type name def)))
               (list name :constant type))))
    (let ((constant-bindings (mapcar #'%constant-binding
                                     (kernel-definition-constant-names def))))
      (bulk-add-variable-environment constant-bindings var-env))))

(defun make-variable-environment-with-kernel-definition (name def)
  (%add-function-arguments name def
    (%add-symbol-macros def
      (%add-constants def
        (empty-variable-environment)))))

(defmacro with-variable-environment ((var-env bindings) &body body)
  `(let ((,var-env (bulk-add-variable-environment ',bindings (empty-variable-environment))))
     ,@body))

(defun lookup-variable-environment (name var-env)
  (find name var-env :key #'varenv-name))

(defun variable-environment-variable-exists-p (name var-env)
  (varenv-variable-p (lookup-variable-environment name var-env)))

(defun variable-environment-constant-exists-p (name var-env)
  (varenv-constant-p (lookup-variable-environment name var-env)))

(defun variable-environment-symbol-macro-exists-p (name var-env)
  (varenv-symbol-macro-p (lookup-variable-environment name var-env)))

(defun variable-environment-type-of-variable (name var-env)
  (unless (variable-environment-variable-exists-p name var-env)
    (error "undefined varialbe name: ~A" name))
  (varenv-variable-type (lookup-variable-environment name var-env)))

(defun variable-environment-type-of-constant (name var-env)
  (unless (variable-environment-constant-exists-p name var-env)
    (error "undefined constant name: ~A" name))
  (varenv-constant-type (lookup-variable-environment name var-env)))

(defun variable-environment-symbol-macro-expansion (name var-env)
  (unless (variable-environment-symbol-macro-exists-p name var-env)
    (error "undefined symbol macro name: ~A" name))
  (varenv-symbol-macro-expansion (lookup-variable-environment name var-env)))


;;;
;;; Function environment
;;;

(defun make-funcenv-function (name return-type args body)
  (assert (symbolp name))
  (assert (valid-type-p return-type))
  (assert (listp args))
  (dolist (arg args)
    (assert (= (length arg) 2))
    (assert (symbolp (car arg)))
    (assert (valid-type-p (cadr arg))))
  (assert (listp body))
  (list name :function return-type args body))

(defun funcenv-function-p (elem)
  (match elem
    ((_ :function _ _ _) t)
    (_ nil)))

(defun funcenv-function-name (elem)
  (match elem
    ((name :function _ _ _) name)
    (_ (error "invalid function environment function: ~A" elem))))

(defun funcenv-function-c-name (elem)
  (compile-identifier-with-package-name (funcenv-function-name elem)))

(defun funcenv-function-return-type (elem)
  (match elem
    ((_ :function return-type _ _) return-type)
    (_ (error "invalid function environment function: ~A" elem))))

(defun funcenv-function-arguments (elem)
  (match elem
    ((_ :function _ arguments _) arguments)
    (_ (error "invalid function environment function: ~A" elem))))

(defun funcenv-function-argument-types (elem)
  (mapcar #'cadr (funcenv-function-arguments elem)))

(defun funcenv-function-body (elem)
  (match elem
    ((_ :function _ _ body) body)
    (_ (error "invalid function environment function: ~A" elem))))

(defun make-funcenv-macro (name args body expander)
  (assert (symbolp name))
  (assert (listp args))
  (assert (listp body))
  (assert (functionp expander))
  (list name :macro args body expander))

(defun funcenv-macro-p (elem)
  (match elem
    ((_ :macro _ _ _) t)
    (_ nil)))

(defun funcenv-macro-name (elem)
  (match elem
    ((name :macro _ _ _) name)
    (_ (error "invalid function environment macro: ~A" elem))))

(defun funcenv-macro-arguments (elem)
  (match elem
    ((_ :macro arguments _ _) arguments)
    (_ (error "invalid function environment macro: ~A" elem))))

(defun funcenv-macro-body (elem)
  (match elem
    ((_ :macro _ body _) body)
    (_ (error "invalid function environment macro: ~A" elem))))

(defun funcenv-macro-expander (elem)
  (match elem
    ((_ :macro _ _ expander) expander)
    (_ (error "invalid function environment macro: ~A" elem))))

(defun funcenv-name (elem)
  (cond
    ((funcenv-function-p elem) (funcenv-function-name elem))
    ((funcenv-macro-p elem) (funcenv-macro-name elem))
    (t (error "invalid function environment element: ~A" elem))))

(defun empty-function-environment ()
  '())

(defun add-function-to-function-environment (name return-type arguments body func-env)
  (let ((elem (make-funcenv-function name return-type arguments body)))
    (cons elem func-env)))

(defun add-macro-to-function-environment (name arguments body expander func-env)
  (let ((elem (make-funcenv-macro name arguments body expander)))
    (cons elem func-env)))

(defun bulk-add-function-environment (bindings func-env)
  (reduce #'(lambda (func-env2 binding)
              (match binding
                ((name :function return-type args body)
                 (add-function-to-function-environment name return-type args body func-env2))
                ((name :macro args body expander)
                 (add-macro-to-function-environment name args body expander func-env2))
                (_ (error "invalid function environment element: ~A" binding))))
          bindings :initial-value func-env))

(defun make-function-environment-with-kernel-definition (def)
  (labels ((%function-binding (name)
             (let ((name (kernel-definition-function-name name def))
                   (return-type (kernel-definition-function-return-type name def))
                   (args (kernel-definition-function-arguments name def))
                   (body (kernel-definition-function-body name def)))
               (list name :function return-type args body)))
           (%macro-binding (name)
             (let ((name (kernel-definition-macro-name name def))
                   (args (kernel-definition-macro-arguments name def))
                   (body (kernel-definition-macro-body name def))
                   (expander (kernel-definition-macro-expander name def)))
               (list name :macro args body expander))))
    (let ((function-bindings (mapcar #'%function-binding
                                     (kernel-definition-function-names def)))
          (macro-bindings (mapcar #'%macro-binding
                                  (kernel-definition-macro-names def))))
      (bulk-add-function-environment macro-bindings
        (bulk-add-function-environment function-bindings
          (empty-function-environment))))))

(defmacro with-function-environment ((func-env bindings) &body body)
  (labels ((aux (binding)
             (match binding
               ((name :function return-type args body) `(list ',name :function ',return-type ',args ',body))
               ((name :macro args body) (alexandria:with-gensyms (args0)
                                          `(list ',name :macro ',args ',body
                                                 (lambda (,args0) (destructuring-bind ,args ,args0 ,@body)))))
               (_ `',binding))))
    (let ((bindings2 `(list ,@(mapcar #'aux bindings))))
      `(let ((,func-env (bulk-add-function-environment ,bindings2 (empty-function-environment))))
         ,@body))))

(defun lookup-function-environment (name func-env)
  (find name func-env :key #'funcenv-name))

(defun function-environment-function-exists-p (name func-env)
  (funcenv-function-p (lookup-function-environment name func-env)))

(defun function-environment-macro-exists-p (name func-env)
  (funcenv-macro-p (lookup-function-environment name func-env)))

(defun function-environment-function-c-name (name func-env)
  (unless (function-environment-function-exists-p name func-env)
    (error "undefined function name: ~A" name))
  (funcenv-function-c-name (lookup-function-environment name func-env)))

(defun function-environment-function-return-type (name func-env)
  (unless (function-environment-function-exists-p name func-env)
    (error "undefined function name: ~A" name))
  (funcenv-function-return-type (lookup-function-environment name func-env)))

(defun function-environment-function-arguments (name func-env)
  (unless (function-environment-function-exists-p name func-env)
    (error "undefined function name: ~A" name))
  (funcenv-function-arguments (lookup-function-environment name func-env)))

(defun function-environment-function-argument-types (name func-env)
  (unless (function-environment-function-exists-p name func-env)
    (error "undefined function name: ~A" name))
  (funcenv-function-argument-types (lookup-function-environment name func-env)))

(defun function-environment-function-body (name func-env)
  (unless (function-environment-function-exists-p name func-env)
    (error "undefined function name: ~A" name))
  (funcenv-function-body (lookup-function-environment name func-env)))

(defun function-environment-macro-arguments (name func-env)
  (unless (function-environment-macro-exists-p name func-env)
    (error "undefined macro name: ~A" name))
  (funcenv-macro-arguments (lookup-function-environment name func-env)))

(defun function-environment-macro-body (name func-env)
  (unless (function-environment-macro-exists-p name func-env)
    (error "undefined macro name: ~A" name))
  (funcenv-macro-body (lookup-function-environment name func-env)))

(defun function-environment-macro-expander (name func-env)
  (unless (function-environment-macro-exists-p name func-env)
    (error "undefined macro name: ~A" name))
  (funcenv-macro-expander (lookup-function-environment name func-env)))


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


;;;
;;; Utilities
;;;

(defun compile-identifier (idt)
  (substitute-if #\_ (lambda (char)
                       (and (not (alphanumericp char))
                            (not (char= #\_ char))
                            (not (char= #\* char))))
                 (string-downcase idt)))

(defun compile-identifier-with-package-name (name)
  (let ((package-name (compile-identifier (package-name (symbol-package name))))
        (function-name (compile-identifier name)))
    (concatenate 'string package-name "_" function-name)))

(defun join (str xs &key (remove-nil nil))
  (let ((xs2 (if remove-nil (remove-if #'null xs) xs)))
    (if (not (null xs2))
      (reduce #'(lambda (a b) (concatenate 'string a str b)) xs2)
      "")))

(defun indent (n str)
  (labels ((aux (x)
             (concatenate 'string (spaces n) x)))
    (unlines (mapcar #'aux (lines str)))))

(defun lines (str)
  (split-sequence:split-sequence #\LineFeed str :remove-empty-subseqs t))

(defun unlines (&rest args)
  (cond ((null args) "")
        ((listp (car args)) (join (string #\LineFeed) (car args) :remove-nil t))
        (t (join (string #\LineFeed) args :remove-nil t))))

(defun spaces (n)
  (if (< 0 n)
      (concatenate 'string " " (spaces (1- n)))
      ""))

(defun undefined ()
  (error "undefined"))

(defun cl-cuda-symbolicate (&rest args)
  (intern (apply #'concatenate 'string (mapcar #'princ-to-string args))
          :cl-cuda))
