#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-cuda)


;;;
;;; Load CUDA library
;;;

(cffi:define-foreign-library libcuda
  (t (:default "/usr/local/cuda/lib/libcuda")))
(cffi:use-foreign-library libcuda)



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
(cffi:defctype cu-device-ptr :unsigned-int)
(cffi:defctype cu-event :pointer)
(cffi:defctype cu-graphics-resource :pointer)
(cffi:defctype size-t :unsigned-int)


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
(defcufun (cu-ctx-create "cuCtxCreate") cu-result
  (pctx (:pointer cu-context))
  (flags :unsigned-int)
  (dev cu-device))

;; cuGLCtxCreate
(defcufun (cu-gl-ctx-create "cuGLCtxCreate") cu-result
  (pctx (:pointer cu-context))
  (flags :unsigned-int)
  (dev cu-device))

;; cuCtxDestroy
(defcufun (cu-ctx-destroy "cuCtxDestroy") cu-result
  (pctx cu-context))

;; cuCtxSynchronize
(defcufun (cu-ctx-synchronize "cuCtxSynchronize") cu-result)

;; cuMemAlloc
(defcufun (cu-mem-alloc "cuMemAlloc") cu-result
  (dptr (:pointer cu-device-ptr))
  (bytesize size-t))

;; cuMemFree
(defcufun (cu-mem-free "cuMemFree") cu-result
  (dptr cu-device-ptr))

;; cuMemcpyHtoD
(defcufun (cu-memcpy-host-to-device "cuMemcpyHtoD") cu-result
  (dst-device cu-device-ptr)
  (src-host :pointer)
  (byte-count size-t))

;; cuMemcpyDtoH
(defcufun (cu-memcpy-device-to-host "cuMemcpyDtoH") cu-result
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
(defcufun (cu-event-destroy "cuEventDestroy") cu-result
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
    (cu-ctx-synchronize)))


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


;;;
;;; Definition of cl-cuda types
;;;

(defvar +basic-types+ '((void  0 :void)
                        (bool  1 (:boolean :int8))
                        (int   4 :int)
                        (float 4 :float)))

(defvar +vector-types+ '((float3 float 3 float3-x float3-y float3-z)
                         (float4 float 4 float4-x float4-y float4-z float4-w)))

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
  4)

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
    `(labels ((,do-body (,device-ptr)
                ,@body))
       (if (memory-block-interop-p ,blk)
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
               (cffi:foreign-free ,device-ptr)))
           (let ((,device-ptr (memory-block-device-ptr ,blk)))
             (,do-body ,device-ptr))))))

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
    (t (error "must not be reached"))))

(defun float3-mem-aref (blk idx)
  ;; give type and slot names as constant explicitly for better performance
  (let ((ptr (cffi:mem-aref (memory-block-cffi-ptr blk) 'float3 idx)))
    (make-float3 (cffi:foreign-slot-value ptr 'float3 'x)
                 (cffi:foreign-slot-value ptr 'float3 'y)
                 (cffi:foreign-slot-value ptr 'float3 'z))))

(defun float4-mem-aref (blk idx)
  ;; give type and slot names as constant explicitly for better performance
  (let ((ptr (cffi:mem-aref (memory-block-cffi-ptr blk) 'float4 idx)))
    (make-float4 (cffi:foreign-slot-value ptr 'float4 'x)
                 (cffi:foreign-slot-value ptr 'float4 'y)
                 (cffi:foreign-slot-value ptr 'float4 'z)
                 (cffi:foreign-slot-value ptr 'float4 'w))))
                 
(defun vector-type-mem-aref (blk idx)
  (case (memory-block-type blk)
    (float3 (float3-mem-aref blk idx))
    (float4 (float4-mem-aref blk idx))
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
    (t (error "must not be reached"))))

(defun float3-setf-mem-aref (blk idx val)
  ;; give type and slot names as constant explicitly for better performance
  (let ((ptr (cffi:mem-aref (memory-block-cffi-ptr blk) 'float3 idx)))
    (setf (cffi:foreign-slot-value ptr 'float3 'x) (float3-x val))
    (setf (cffi:foreign-slot-value ptr 'float3 'y) (float3-y val))
    (setf (cffi:foreign-slot-value ptr 'float3 'z) (float3-z val))))

(defun float4-setf-mem-aref (blk idx val)
  ;; give type and slot names as constant explicitly for better performance
  (let ((ptr (cffi:mem-aref (memory-block-cffi-ptr blk) 'float4 idx)))
    (setf (cffi:foreign-slot-value ptr 'float4 'x) (float4-x val))
    (setf (cffi:foreign-slot-value ptr 'float4 'y) (float4-y val))
    (setf (cffi:foreign-slot-value ptr 'float4 'z) (float4-z val))
    (setf (cffi:foreign-slot-value ptr 'float4 'w) (float4-w val))))

(defun vector-type-setf-mem-aref (blk idx val)
  (case (memory-block-type blk)
    (float3 (float3-setf-mem-aref blk idx val))
    (float4 (float4-setf-mem-aref blk idx val))
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
;;; Definition of defkernel macro
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
  ;; (x int) => (x-ptr :int), (y float3) => (y-ptr 'float3)
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
;; (cffi:with-foreign-objects ((x-ptr :int) (y-ptr 'float3))
;;   (setf (cffi:mem-ref x-ptr :int) x)
;;   (setf (cffi:slot-value y-ptr 'float3 'x) (float3-x y)
;;         (cffi:slot-value y-ptr 'float3 'y) (float3-y y)
;;         (cffi:slot-value y-ptr 'float3 'z) (float3-z y))
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
;;; Definition of defkernelmacro macro
;;;

(defmacro defkernelmacro (name args &body body)
  (with-gensyms (form-body)
    `(kernel-manager-define-macro *kernel-manager* ',name ',args ',body
       (lambda (,form-body)
         (destructuring-bind ,args ,form-body
           ,@body)))))

       
;;;
;;; Definition of Kernel Manager
;;;

;;; function-info
;;; <function-info> ::= (<name> <return-type> <arguments> <body>)

(defun make-function-info (name return-type args body)
  (list name return-type args body))

(defun function-name (info)
  (car info))

(defun function-c-name (info)
  (let ((name (function-name info)))
    (let ((package-name (compile-identifier (package-name (symbol-package name))))
          (function-name (compile-identifier name)))
      (concatenate 'string package-name "_" function-name))))

(defun function-return-type (info)
  (cadr info))

(defun function-arguments (info)
  (caddr info))

(defun function-argument-types (info)
  (mapcar #'cadr (function-arguments info)))

(defun function-body (info)
  (cadddr info))


;;; macro-info
;;; <macro-info> ::= (<name> <arguments> <body> <expander>)

(defun make-macro-info (name args body expander)
  (list name args body expander))

(defun macro-name (info)
  (car info))

(defun macro-arguments (info)
  (cadr info))

(defun macro-body (info)
  (caddr info))

(defun macro-expander (info)
  (cadddr info))


;;; kernel-definition
;;; <kernel-definition>     ::= (<kernel-function-table> <kernel-macro-table> <kernel-constant-table>)
;;; <kernel-function-table> ::= alist { <function-name> => <function-info> }
;;; <kernel-macro-table>    ::= alist { <macro-name>    => <macro-info> }
;;; <kernel-constant-table> ::= alist { <constant-name> => <constant-info> }

(defun empty-kernel-definition ()
  (list nil nil nil))

(defun function-table (def)
  (car def))

(defun macro-table (def)
  (cadr def))

(defun constant-table (def)
  (caddr def))

(defun function-info (name def)
  (or (assoc name (function-table def))
      (error (format nil "undefined kernel function: ~A" name))))

(defun macro-info (name def)
  (or (assoc name (macro-table def))
      (error (format nil "undefined kernel macro: ~A" name))))

(defun constant-info (name def)
  (or (assoc name (constant-table def))
      (error (format nil "undefined kernel constant: ~A" name))))

(defun define-kernel-function (name return-type args body def)
  (let ((func-table  (function-table def))
        (macro-table (macro-table    def))
        (const-table (constant-table def)))
    (let ((func (make-function-info name return-type args body))
          (rest (remove name func-table :key #'car)))
      (list (cons func rest) macro-table const-table))))

(defun undefine-kernel-function (name def)
  (unless (kernel-definition-function-exists-p name def)
    (error (format nil "undefined kernel function: ~A" name)))
  (let ((func-table  (function-table def))
        (macro-table (macro-table    def))
        (const-table (constant-table def)))
    (list (remove name func-table :key #'car) macro-table const-table)))

(defun define-kernel-macro (name args body expander def)
  (let ((func-table  (function-table def))
        (macro-table (macro-table    def))
        (const-table (constant-table def)))
    (let ((macro (make-macro-info name args body expander))
          (rest  (remove name macro-table :key #'car)))
      (list func-table (cons macro rest) const-table))))

(defun undefine-kernel-macro (name def)
  (unless (kernel-definition-macro-exists-p name def)
    (error (format nil "undefined kernel macro: ~A" name)))
  (let ((func-table  (function-table def))
        (macro-table (macro-table    def))
        (const-table (constant-table def)))
    (list func-table (remove name macro-table :key #'car) const-table)))

(defun define-kernel-constant (name value def)
  (declare (ignorable name value def))
  (undefined))

(defun undefine-kernel-constant (name def)
  (declare (ignorable name def))
  (undefined))

(defun kernel-definition-function-exists-p (name def)
  (and (assoc name (function-table def))
       t))

(defun kernel-definition-function-names (def)
  (mapcar #'function-name (function-table def)))

(defun kernel-definition-function-name (name def)
  (function-name (function-info name def)))

(defun kernel-definition-function-c-name (name def)
  (function-c-name (function-info name def)))

(defun kernel-definition-function-return-type (name def)
  (function-return-type (function-info name def)))

(defun kernel-definition-function-arguments (name def)
  (function-arguments (function-info name def)))

(defun kernel-definition-function-argument-types (name def)
  (function-argument-types (function-info name def)))

(defun kernel-definition-function-body (name def)
  (function-body (function-info name def)))

(defun kernel-definition-macro-exists-p (name def)
  (and (assoc name (macro-table def))
       t))

(defun kernel-definition-macro-names (def)
  (mapcar #'macro-name (macro-table def)))

(defun kernel-definition-macro-name (name def)
  (macro-name (macro-info name def)))

(defun kernel-definition-macro-arguments (name def)
  (macro-arguments (macro-info name def)))

(defun kernel-definition-macro-body (name def)
  (macro-body (macro-info name def)))

(defun kernel-definition-macro-expander (name def)
  (macro-expander (macro-info name def)))


;;; module-info
;;; <module-info> ::= (<module-handle> <module-path> <module-compilation-needed> <function-handles>)
;;; <function-handles> ::= hashtable { <function-name> => <function-handle> }

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


;;; kernel-manager
;;; <kernel-manager> ::= (<module-info> <kernel-definition>)

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

(defun kernel-manager-function-return-type (mgr name)
  (kernel-definition-function-return-type name (kernel-definition mgr)))

(defun kernel-manager-function-arguments (mgr name)
  (kernel-definition-function-arguments name (kernel-definition mgr)))

(defun kernel-manager-function-argument-types (mgr name)
  (kernel-definition-function-argument-types name (kernel-definition mgr)))

(defun kernel-manager-function-body (mgr name)
  (kernel-definition-function-body name (kernel-definition mgr)))

(defun kernel-manager-define-function (mgr name return-type args body)
  (when (or (not (kernel-manager-function-exists-p mgr name))
            (function-modified-p mgr name return-type args body))
    (setf (kernel-definition mgr)
          (define-kernel-function name return-type args body (kernel-definition mgr)))
    (setf (kernel-manager-module-compilation-needed mgr) t)))

(defun function-modified-p (mgr name return-type args body)
  (not (and (equal return-type (kernel-manager-function-return-type mgr name))
            (equal args (kernel-manager-function-arguments mgr name))
            (equal body (kernel-manager-function-body mgr name)))))

(defun kernel-manager-macro-exists-p (mgr name)
  (kernel-definition-macro-exists-p name (kernel-definition mgr)))

(defun kernel-manager-macro-names (mgr)
  (kernel-definition-macro-names (kernel-definition mgr)))

(defun kernel-manager-macro-name (mgr name)
  (kernel-definition-macro-name name (kernel-definition mgr)))

(defun kernel-manager-macro-arguments (mgr name)
  (kernel-definition-macro-arguments name (kernel-definition mgr)))

(defun kernel-manager-macro-body (mgr name)
  (kernel-definition-macro-body name (kernel-definition mgr)))

(defun kernel-manager-macro-expander (mgr name)
  (kernel-definition-macro-expander name (kernel-definition mgr)))

(defun kernel-manager-define-macro (mgr name args body expander)
  (when (or (not (kernel-manager-macro-exists-p mgr name))
            (macro-modified-p mgr name args body))
    (setf (kernel-definition mgr)
          (define-kernel-macro name args body expander (kernel-definition mgr)))
    (setf (kernel-manager-module-compilation-needed mgr) t)))

(defun macro-modified-p (mgr name args body)
  (not (and (equal args (kernel-manager-macro-arguments mgr name))
            (equal body (kernel-manager-macro-body mgr name)))))

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

(defvar +temporary-path-template+ "/tmp/cl-cuda-")
(defvar +temp-path+ (osicat-posix:mktemp +temporary-path-template+))
(defvar +cu-path+  (concatenate 'string +temp-path+ ".cu"))
(defvar +ptx-path+ (concatenate 'string +temp-path+ ".ptx"))

(defvar +include-path+ (namestring (asdf:system-relative-pathname :cl-cuda #P"include")))

(defvar +nvcc-path+ "/usr/local/cuda/bin/nvcc")

(defun output-cu-code (mgr path)
  (with-open-file (out path :direction :output :if-exists :supersede)
    (princ (compile-kernel-definition (kernel-definition mgr)) out)))

(defun output-nvcc-command (opts)
  (format t "nvcc~{ ~A~}~%" opts))

(defun run-nvcc-command (opts)
  (with-output-to-string (out)
    (let ((p (sb-ext:run-program +nvcc-path+ opts :error out)))
      (unless (= 0 (sb-ext:process-exit-code p))
        (error "nvcc exits with code: ~A~%~A"
               (sb-ext:process-exit-code p)
               (get-output-stream-string out))))))

(defun compile-cu-code (include-path cu-path ptx-path)
  (let ((opts (list "-arch=sm_11" "-I" include-path "-ptx" "-o" ptx-path cu-path)))
    (output-nvcc-command opts)
    (run-nvcc-command opts)))

(defun kernel-manager-generate-and-compile (mgr)
  (unless (not (kernel-manager-module-handle mgr))
    (error "kernel module is already loaded."))
  (unless (no-kernel-functions-loaded-p mgr)
    (error "some kernel functions are already loaded."))
  (output-cu-code mgr +cu-path+)
  (compile-cu-code +include-path+ +cu-path+ +ptx-path+)
  (setf (kernel-manager-module-path mgr) +ptx-path+
        (kernel-manager-module-compilation-needed mgr) nil))


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

(defun compile-kernel-definition (def)
  (unlines `("#include \"float3.h\""
             ""
             ,@(mapcar #'(lambda (name)
                           (compile-kernel-function-prototype name def))
                       (kernel-definition-function-names def))
             ""
             ,@(mapcar #'(lambda (name)
                           (compile-kernel-function name def))
                       (kernel-definition-function-names def)))))


;;; compile kernel function prototype

(defun compile-kernel-function-prototype (name def)
  (let ((name (kernel-definition-function-c-name name def))
        (return-type (kernel-definition-function-return-type name def))
        (arg-bindings (kernel-definition-function-arguments name def)))
    (format nil "extern \"C\" ~A;"
            (compile-function-declaration name return-type arg-bindings))))


;;; compile kernel function

(defun compile-kernel-function (name def)
  (let ((c-name (kernel-definition-function-c-name name def))
        (return-type (kernel-definition-function-return-type name def))
        (arg-bindings (kernel-definition-function-arguments name def))
        (stmts (kernel-definition-function-body name def)))
    (let ((type-env (make-type-environment-with-kernel-definition name def)))
      (unlines `(,(compile-function-declaration c-name return-type arg-bindings)
                 "{"
                 ,@(mapcar #'(lambda (stmt)
                              (indent 2 (compile-statement stmt type-env def)))
                           stmts)
                 "}"
                  "")))))

(defun make-type-environment-with-kernel-definition (name def)
  (let ((arg-bindings (kernel-definition-function-arguments name def)))
    (reduce #'(lambda (type-env arg-binding)
                (destructuring-bind (var type) arg-binding
                  (add-type-environment var type type-env)))
            arg-bindings
            :initial-value (empty-type-environment))))

(defun compile-function-declaration (name return-type arg-bindings)
  (format nil "~A ~A ~A (~A)" (compile-function-specifier return-type)
                              (compile-type return-type)
                              name
                              (compile-arg-bindings arg-bindings)))

(defun compile-function-specifier (return-type)
  (unless (valid-type-p return-type)
    (error (format nil "invalid return type: ~A" return-type)))
  (if (eq return-type 'void)
      "__global__"
      "__device__"))

(defun compile-type (type)
  (unless (valid-type-p type)
    (error (format nil "invalid type: ~A" type)))
  (compile-identifier (princ-to-string type)))

(defun compile-arg-bindings (arg-bindings)
  (join ", " (mapcar #'compile-arg-binding arg-bindings)))

(defun compile-arg-binding (arg-binding)
  (destructuring-bind (var type) arg-binding
    (format nil "~A ~A" (compile-type type) (compile-identifier var))))

(defun compile-identifier (idt)
  (substitute #\_ #\% (substitute #\_ #\. (substitute #\_ #\- (string-downcase idt)))))
  

;;; compile statement

(defun compile-statement (stmt type-env def)
  (cond
    ((macro-form-p stmt def) (compile-macro stmt type-env def :statement-p t))
    ((if-p stmt) (compile-if stmt type-env def))
    ((let-p stmt) (compile-let stmt type-env def))
    ((do-p stmt) (compile-do stmt type-env def))
    ((with-shared-memory-p stmt) (compile-with-shared-memory stmt type-env def))
    ((set-p stmt) (compile-set stmt type-env def))
    ((progn-p stmt) (compile-progn stmt type-env def))
    ((return-p stmt) (compile-return stmt type-env def))
    ((syncthreads-p stmt) (compile-syncthreads stmt))
    ((function-p stmt) (compile-function stmt type-env def :statement-p t))
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

(defun compile-if (stmt type-env def)
  (let ((test-exp  (if-test-expression stmt))
        (then-stmt (if-then-statement stmt))
        (else-stmt (if-else-statement stmt)))
    (let ((test-type (type-of-expression test-exp type-env def)))
      (unless (eq test-type 'bool)
        (error "invalid type: type of test-form is ~A, not ~A" test-type 'bool)))
    (unlines (format nil "if (~A) {"
                     (compile-expression test-exp type-env def))
             (indent 2 (compile-statement then-stmt type-env def))
             (and else-stmt "} else {")
             (and else-stmt
                  (indent 2 (compile-statement else-stmt type-env def)))
             "}")))


;;; let statement

(defun let-p (stmt)
  (match stmt
    (('let . _) t)
    (_ nil)))

(defun let-bindings (stmt)
  (match stmt
    (('let bindings . _) bindings)
    (_ (error "invalid statement: ~A" stmt))))

(defun let-statements (stmt0)
  (match stmt0
    (('let _ . stmts) stmts)
    (_ (error "invalid statement: ~A" stmt0))))

(defun compile-let (stmt0 type-env def)
  (let ((bindings (let-bindings stmt0))
        (stmts (let-statements stmt0)))
    (unlines "{"
             (indent 2 (%compile-let bindings stmts type-env def))
             "}")))

(defun %compile-let (bindings stmts type-env def)
  (if (null bindings)
      (compile-let-statements stmts type-env def)
      (compile-let-binding bindings stmts type-env def)))

(defun compile-let-binding (bindings stmts type-env def)
  (match bindings
    (((var exp) . rest)
     (let* ((type (type-of-expression exp type-env def))
            (type-env2 (add-type-environment var type type-env)))
       (unlines (format nil "~A ~A = ~A;"
                        (compile-type type)
                        (compile-identifier var)
                        (compile-expression exp type-env def))
                (%compile-let rest stmts type-env2 def))))
    (_ (error "invalid bindings: ~A" bindings))))

(defun compile-let-statements (stmts type-env def)
  (compile-progn-statements stmts type-env def))


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

(defun compile-set (stmt type-env def)
  (let ((place (set-place stmt))
        (exp (set-expression stmt)))
    (let ((place-type (type-of-expression place type-env def))
          (exp-type   (type-of-expression exp   type-env def)))
      (unless (eq place-type exp-type)
        (error "invalid types: type of the place is ~A but that of the expression is ~A" place-type exp-type)))
    (format nil "~A = ~A;" (compile-place place type-env def)
                           (compile-expression exp type-env def))))

(defun compile-place (place type-env def)
  (cond ((scalar-place-p place) (compile-scalar-place place type-env))
        ((vector-place-p place) (compile-vector-place place type-env def))
        ((array-place-p place)  (compile-array-place place type-env def))
        (t (error "invalid place: ~A" place))))

(defun scalar-place-p (place)
  (scalar-variable-reference-p place))

(defun vector-place-p (place)
  (vector-variable-reference-p place))

(defun array-place-p (place)
  (array-variable-reference-p place))

(defun compile-scalar-place (var type-env)
  (compile-scalar-variable-reference var type-env))

(defun compile-vector-place (place type-env def)
  (compile-vector-variable-reference place type-env def))

(defun compile-array-place (place type-env def)
  (compile-array-variable-reference place type-env def))


;;; progn statement

(defun progn-p (stmt)
  (match stmt
    (('progn . _) t)
    (_ nil)))

(defun progn-statements (stmt)
  (match stmt
    (('progn . stmts) stmts)
    (_ (error "invalid statement: ~A" stmt))))

(defun compile-progn (stmt type-env def)
  (compile-progn-statements (progn-statements stmt) type-env def))

(defun compile-progn-statements (stmts type-env def)
  (unlines (mapcar #'(lambda (stmt2)
                       (compile-statement stmt2 type-env def))
                   stmts)))


;;; return statement

(defun return-p (stmt)
  (match stmt
    (('return) t)
    (('return _) t)
    (_ nil)))

(defun compile-return (stmt type-env def)
  (match stmt
    (('return) "return;")
    (('return exp) (format nil "return ~A;"
                               (compile-expression exp type-env def)))
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

(defun do-var-types (stmt type-def def)
  (labels ((do-var-type (binding)
             (list (do-binding-var binding)
                   (do-binding-type binding type-def def))))
    (mapcar #'do-var-type (do-bindings stmt))))

(defun do-binding-var (binding)
  (match binding
    ((var _)   var)
    ((var _ _) var)
    (_ (error "invalid binding: ~A" binding))))

(defun do-binding-type (binding type-env def)
  (type-of-expression (do-binding-init-form binding) type-env def))

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

(defun compile-do (stmt type-env def)
  (let ((type-env2 (bulk-add-type-environment (do-var-types stmt type-env def) type-env)))
    (let ((init-part (compile-do-init-part stmt type-env def))
          (test-part (compile-do-test-part stmt type-env2 def))
          (step-part (compile-do-step-part stmt type-env2 def)))
      (unlines (format nil "for ( ~A; ~A; ~A )" init-part test-part step-part)
               "{"
               (indent 2 (compile-do-statements stmt type-env2 def))
               "}"))))

(defun compile-do-init-part (stmt type-env def)
  (labels ((aux (binding)
             (let ((var (do-binding-var binding))
                   (type (do-binding-type binding type-env def))
                   (init-form (do-binding-init-form binding)))
               (format nil "~A ~A = ~A" (compile-type type)
                                        (compile-identifier var)
                                        (compile-expression init-form type-env def)))))
    (join ", " (mapcar #'aux (do-bindings stmt)))))

(defun compile-do-test-part (stmt type-env def)
  (let ((test-form (do-test-form stmt)))
    (format nil "! ~A" (compile-expression test-form type-env def))))

(defun compile-do-step-part (stmt type-env def)
  (labels ((aux (binding)
             (let ((var (do-binding-var binding))
                   (step-form (do-binding-step-form binding)))
               (format nil "~A = ~A" (compile-identifier var)
                                     (compile-expression step-form type-env def)))))
    (join ", " (mapcar #'aux (remove-if-not #'do-binding-step-form (do-bindings stmt))))))

(defun compile-do-statements (stmt type-env def)
  (compile-progn-statements (do-statements stmt) type-env def))


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

(defun compile-with-shared-memory (stmt type-env def)
  (let ((specs (with-shared-memory-specs stmt))
        (stmts (with-shared-memory-statements stmt)))
    (unlines "{"
             (indent 2 (%compile-with-shared-memory specs stmts type-env def))
             "}")))

(defun %compile-with-shared-memory (specs stmts type-env def)
  (if (null specs)
      (compile-with-shared-memory-statements stmts type-env def)
      (compile-with-shared-memory-spec specs stmts type-env def)))

(defun compile-with-shared-memory-spec (specs stmts type-env def)
  (match specs
    (((var type . sizes) . rest)
     (let* ((type-env2 (add-type-environment var (add-star type (length sizes))
                                             type-env)))
       (unlines (format nil "__shared__ ~A ~A~{[~A]~};"
                            (compile-type type)
                            (compile-identifier var)
                            (mapcar #'(lambda (exp)
                                        (compile-expression exp type-env def))
                                    sizes))
                (%compile-with-shared-memory rest stmts type-env2 def))))
    (_ (error "invalid shared memory specs: ~A" specs))))

(defun compile-with-shared-memory-statements (stmts type-env def)
  (compile-let-statements stmts type-env def))


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

(defun defined-function-p (form def)
  (or (built-in-function-p form)
      (user-function-p form def)))

(defun built-in-function-p (form)
  (match form
    ((op . _) (and (getf +built-in-functions+ op) t))
    (_ nil)))

(defun user-function-p (form def)
  (match form
    ((op . _) (kernel-definition-function-exists-p op def))
    (_ nil)))

(defun function-operator (form)
  (unless (function-p form)
    (error "invalid statement or expression: ~A" form))
  (car form))

(defun function-operands (form)
  (unless (function-p form)
    (error "invalid statement or expression: ~A" form))
  (cdr form))

(defun compile-function (form type-env def &key (statement-p nil))
  (unless (defined-function-p form def)
    (error "undefined function: ~A" form))
  (let ((code (if (built-in-function-p form)
                  (compile-built-in-function form type-env def)
                  (compile-user-function form type-env def))))
    (if statement-p
        (format nil "~A;" code)
        code)))

(defun compile-built-in-function (form type-env def)
  (let ((op (function-operator form)))
    (cond
      ((built-in-function-infix-p form type-env def)
       (compile-built-in-infix-function form type-env def))
      ((built-in-function-prefix-p form type-env def)
       (compile-built-in-prefix-function form type-env def))
      (t (error "invalid built-in function: ~A" op)))))

(defun compile-built-in-infix-function (form type-env def)
  (let ((operands (function-operands form)))
    (let ((op  (built-in-function-c-string form type-env def))
          (lhe (compile-expression (car operands) type-env def))
          (rhe (compile-expression (cadr operands) type-env def)))
      (format nil "(~A ~A ~A)" lhe op rhe))))

(defun compile-built-in-prefix-function (form type-env def)
  (let ((operands (function-operands form)))
    (format nil "~A (~A)"
            (built-in-function-c-string form type-env def)
            (compile-operands operands type-env def))))

(defun type-of-operands (operands type-env def)
  (mapcar #'(lambda (exp)
              (type-of-expression exp type-env def))
          operands))

(defun compile-operands (operands type-env def)
  (join ", " (mapcar #'(lambda (exp)
                         (compile-expression exp type-env def))
                     operands)))

(defun compile-user-function (form type-env def)
  (let ((operator (function-operator form))
        (operands (function-operands form)))
    (let ((expected-types (kernel-definition-function-argument-types operator def))
          (actual-types (type-of-operands operands type-env def)))
      (unless (equal expected-types actual-types)
        (error "invalid arguments: ~A" form)))
    (let ((func (kernel-definition-function-c-name operator def))
          (compiled-operands (compile-operands operands type-env def)))
      (format nil "~A (~A)" func compiled-operands))))


;;; compile macro

(defun macro-form-p (form def)
  "Returns t if the given form is a macro form. The macro used in the
form may be an user-defined macro under the given kernel definition or
a built-in macro.
TODO: consider symbol macros"
  (or (built-in-macro-p form)
      (user-macro-p form def)))

(defun built-in-macro-p (form)
  (match form
    ((op . _) (and (getf +built-in-macros+ op) t))
    (_ nil)))

(defun user-macro-p (form def)
  (match form
    ((op . _) (kernel-definition-macro-exists-p op def))
    (_ nil)))

(defun macro-operator (form def)
  (unless (macro-form-p form def)
    (error "undefined macro form: ~A" form))
  (car form))

(defun macro-operands (form def)
  (unless (macro-form-p form def)
    (error "undefined macro form: ~A" form))
  (cdr form))

(defun compile-macro (form type-env def &key (statement-p nil))
  (unless (macro-form-p form def)
    (error "undefined macro: ~A" form))
  (if statement-p
      (compile-statement  (%expand-macro-1 form def) type-env def)
      (compile-expression (%expand-macro-1 form def) type-env def)))

(defun %expand-macro-1 (form def)
  (labels ((expand (form def)
             (let ((operator (macro-operator form def))
                   (operands (macro-operands form def)))
               (let ((expander (if (built-in-macro-p form)
                                   (built-in-macro-expander operator)
                                   (kernel-definition-macro-expander operator def))))
                 (funcall expander operands)))))
    (if (macro-form-p form def)
        (values (expand form def) t)
        (values form nil))))

(defun expand-macro-1 (form)
  "If a form is a macro form, then EXPAND-MACRO-1 expands the macro
form call once, and returns the macro expansion and true as values.
Otherwise, returns the given form and false as values."
  (let ((def (kernel-definition *kernel-manager*)))
    (%expand-macro-1 form def)))

(defun %expand-macro (form def)
  (if (macro-form-p form def)
      (values (%expand-macro (%expand-macro-1 form def) def) t)
      (values form nil)))

(defun expand-macro (form)
  "If a form is a macro form, then EXPAND-MACRO repeatedly expands
the macro form until it is no longer a macro form, and returns the
macro expansion and true as values. Otherwise, returns the given form
and false as values."
  (let ((def (kernel-definition *kernel-manager*)))
    (%expand-macro form def)))


;;; built-in functions
;;;   <built-in-functions>  ::= plist { <function-name> => <function-info> }
;;;   <function-info>       ::= (<infix-p> <function-candidates>)
;;;   <function-candidates> ::= (<function-candidate>*)
;;;   <function-candidate>  ::= (<arg-types> <return-type> <function-c-name>)
;;;   <arg-types>           ::= (<arg-type>*)

(defvar +built-in-functions+
  '(%add (((int    int)    int    t   "+")
          ((float  float)  float  t   "+")
          ((float3 float3) float3 nil "float3_add"))
    %sub (((int    int)    int    t   "-")
          ((float  float)  float  t   "-")
          ((float3 float3) float3 nil "float3_sub"))
    %mul (((int    int)    int    t   "*")
          ((float  float)  float  t   "*")
          ((float3 float)  float3 nil "float3_scale")
          ((float  float3) float3 nil "float3_scale_flipped"))
    %div (((int    int)    int    t   "/")
          ((float  float)  float  t   "/")
          ((float3 float)  float3 nil "float3_scale_inverted"))
    %negate (((int)    int    nil "int_negate")
             ((float)  float  nil "float_negate")
             ((float3) float3 nil "float3_negate"))
    %recip (((int)    int    nil "int_recip")
            ((float)  float  nil "float_recip")
            ((float3) float3 nil "float3_recip"))
    =    (((int   int)   bool t "==")
          ((float float) bool t "=="))
    /=   (((int   int)   bool t "!=")
          ((float float) bool t "!="))
    <    (((int   int)   bool t "<")
          ((float float) bool t "<"))
    >    (((int   int)   bool t ">")
          ((float float) bool t ">"))
    <=   (((int   int)   bool t "<=")
          ((float float) bool t "<="))
    >=   (((int   int)   bool t ">=")
          ((float float) bool t ">="))
    not  (((bool) bool nil "!"))
    expt   (((float float) float nil "powf"))
    rsqrtf (((float) float nil "rsqrtf"))
    sqrt   (((float) float nil "sqrtf"))
    floor  (((float) int   nil "floorf"))
    atomic-add (((int* int) int nil "atomicAdd"))
    pointer (((int)   int*   nil "&")
             ((float) float* nil "&"))
    float3 (((float float float) float3 nil "make_float3"))
    float4 (((float float float float) float4 nil "make_float4"))
    dot (((float3 float3) float nil "float3_dot"))
    ))

(defun function-candidates (op)
  (or (getf +built-in-functions+ op)
      (error "invalid function: ~A" op)))

(defun inferred-function (form type-env def)
  (let ((operator (function-operator form))
        (operands (function-operands form)))
    (let ((candidates (function-candidates operator))
          (types (type-of-operands operands type-env def)))
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

(defun built-in-function-argument-types (form type-env def)
  (inferred-function-argument-types (inferred-function form type-env def)))

(defun built-in-function-return-type (form type-env def)
  (inferred-function-return-type (inferred-function form type-env def)))

(defun built-in-function-infix-p (form type-env def)
  (inferred-function-infix-p (inferred-function form type-env def)))

(defun built-in-function-prefix-p (form type-env def)
  (inferred-function-prefix-p (inferred-function form type-env def)))

(defun built-in-function-c-string (form type-env def)
  (inferred-function-c-string (inferred-function form type-env def)))


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

(defun compile-expression (exp type-env def)
  (cond
    ((macro-form-p exp def) (compile-macro exp type-env def))
    ((literal-p exp) (compile-literal exp))
    ((cuda-dimension-p exp) (compile-cuda-dimension exp))
    ((variable-reference-p exp)
     (compile-variable-reference exp type-env def))
    ((inline-if-p exp) (compile-inline-if exp type-env def))
    ((function-p exp) (compile-function exp type-env def))
    (t (error "invalid expression: ~A" exp))))

(defun literal-p (exp)
  (or (bool-literal-p exp)
      (int-literal-p exp)
      (float-literal-p exp)))

(defun bool-literal-p (exp)
  (typep exp 'boolean))

(defun int-literal-p (exp)
  (typep exp 'fixnum))

(defun float-literal-p (exp)
  (typep exp 'single-float))

(defun compile-bool-literal (exp)
  (unless (typep exp 'boolean)
    (error "invalid literal: ~A" exp))
  (if exp "true" "false"))

(defun compile-int-literal (exp)
  (princ-to-string exp))

(defun compile-float-literal (exp)
  (princ-to-string exp))

(defun compile-literal (exp)
  (cond ((bool-literal-p  exp) (compile-bool-literal exp))
        ((int-literal-p   exp) (compile-int-literal exp))
        ((float-literal-p exp) (compile-float-literal exp))
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
  (or (scalar-variable-reference-p exp)
      (vector-variable-reference-p exp)
      (array-variable-reference-p exp)))

(defun scalar-variable-reference-p (exp)
  (symbolp exp))

(defun vector-variable-reference-p (exp)
  (match exp
    ((selector _) (valid-vector-type-selector-p selector))
    (_ nil)))

(defun array-variable-reference-p (exp) 
  (match exp
    (('aref . _) t)
    (_ nil)))

(defun compile-variable-reference (exp type-env def)
  (cond ((scalar-variable-reference-p exp)
         (compile-scalar-variable-reference exp type-env))
        ((vector-variable-reference-p exp)
         (compile-vector-variable-reference exp type-env def))
        ((array-variable-reference-p exp)
         (compile-array-variable-reference exp type-env def))
        (t (error "invalid expression: ~A" exp))))

(defun compile-scalar-variable-reference (var type-env)
  (let ((type (lookup-type-environment var type-env)))
    (unless type
      (error "unbound variable: ~A" var)))
  (compile-identifier var))

(defun compile-vector-selector (selector)
  (unless (valid-vector-type-selector-p selector)
    (error "invalid selector: ~A" selector))
  (string-downcase (subseq (reverse (princ-to-string selector)) 0 1)))

(defun compile-vector-variable-reference (form type-env def)
  (match form
    ((selector exp)
     (let ((selector-type (vector-type-selector-type selector))
           (exp-type      (type-of-expression exp type-env def)))
       (unless (eq selector-type exp-type)
         (error "invalid variable reference: ~A" form))
       (format nil "~A.~A" (compile-expression exp type-env def)
                           (compile-vector-selector selector))))
    (_ (error "invalid variable reference: ~A" form))))

(defun compile-array-variable-reference (form type-env def)
  (match form
    (('aref _)
     (error "invalid variable reference: ~A" form))
    (('aref exp . idxs)
     (let ((type (type-of-expression exp type-env def)))
       (unless (= (array-type-dimension type) (length idxs))
         (error "invalid dimension: ~A" form))
       (format nil "~A~{[~A]~}"
                   (compile-expression exp type-env def)
                   (mapcar #'(lambda (idx)
                               (compile-expression idx type-env def)) idxs))))
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

(defun compile-inline-if (exp type-env def)
  (let ((test-exp (inline-if-test-expression exp))
        (then-exp (inline-if-then-expression exp))
        (else-exp (inline-if-else-expression exp)))
    (let ((test-type (type-of-expression test-exp type-env def))
          (then-type (type-of-expression then-exp type-env def))
          (else-type (type-of-expression else-exp type-env def)))
      (unless (eq test-type 'bool)
        (error "invalid type: type of test-form is ~A, not ~A" test-type 'bool))
      (unless (eq then-type else-type)
        (error "invalid types: type of then-form is ~A but that of else-form is ~A" then-type else-type)))
    (format nil "(~A ? ~A : ~A)"
            (compile-expression test-exp type-env def)
            (compile-expression then-exp type-env def)
            (compile-expression else-exp type-env def))))

;;; type of expression

(defun type-of-expression (exp type-env def)
  (cond ((macro-form-p exp def) (type-of-macro-form exp type-env def))
        ((literal-p exp) (type-of-literal exp))
        ((cuda-dimension-p exp) 'int)
        ((variable-reference-p exp) (type-of-variable-reference exp type-env def))
        ((inline-if-p exp) (type-of-inline-if exp type-env def))
        ((function-p exp) (type-of-function exp type-env def))
        (t (error "invalid expression: ~A" exp))))

(defun type-of-literal (exp)
  (cond ((bool-literal-p exp) 'bool)
        ((int-literal-p exp) 'int)
        ((float-literal-p exp) 'float)
        (t (error "invalid expression: ~A" exp))))

(defun type-of-variable-reference (exp type-env def)
  (cond ((scalar-variable-reference-p exp)
         (type-of-scalar-variable-reference exp type-env))
        ((vector-variable-reference-p exp)
         (type-of-vector-variable-reference exp type-env def))
        ((array-variable-reference-p exp)
         (type-of-array-variable-reference exp type-env def))
        (t (error "invalid expression: ~A" exp))))

(defun type-of-scalar-variable-reference (var type-env)
  (let ((type (lookup-type-environment var type-env)))
    (unless type
      (error "unbound variable: ~A" var))
    type))

(defun type-of-vector-variable-reference (exp type-env def)
  (match exp
    ((selector exp2)
     (let ((selector-type (vector-type-selector-type selector))
           (exp-type      (type-of-expression exp2 type-env def)))
       (unless (eq selector-type exp-type)
         (error "invalid variable reference: ~A" exp))
       (vector-type-base-type exp-type)))
    (_ (error "invalid variable reference: ~A" exp))))

(defun type-of-array-variable-reference (exp type-env def)
  (match exp
    (('aref _) (error "invalid variable reference: ~A" exp))
    (('aref exp2 . idxs)
     (let ((type (type-of-expression exp2 type-env def)))
       (unless (= (array-type-dimension type) (length idxs))
         (error "invalid dimension: ~A" exp))
       (remove-star type)))
    (_ (error "invalid variable reference: ~A" exp))))


(defun type-of-inline-if (exp type-env def)
  (let ((test-exp (inline-if-test-expression exp))
        (then-exp (inline-if-then-expression exp))
        (else-exp (inline-if-else-expression exp)))
    (let ((test-exp-type (type-of-expression test-exp type-env def))
          (then-exp-type (type-of-expression then-exp type-env def))
          (else-exp-type (type-of-expression else-exp type-env def)))
      (when (not (eq test-exp-type 'bool))
        (error "type of the test part of the inline if expression is not bool: ~A" exp))
      (when (not (eq then-exp-type else-exp-type))
        (error "types of the then part and the else part of the inline if expression are not same: ~A" exp))
      then-exp-type)))

(defun type-of-macro-form (exp type-env def)
  (type-of-expression (%expand-macro-1 exp def) type-env def))

(defun type-of-function (exp type-env def)
  (cond ((built-in-function-p exp)
         (type-of-built-in-function exp type-env def))
        ((user-function-p exp def)
         (type-of-user-function exp def))
        (t (error "invalid expression: ~A" exp))))

(defun type-of-built-in-function (exp type-env def)
  (built-in-function-return-type exp type-env def))

(defun type-of-user-function (exp def)
  (let ((operator (function-operator exp)))
    (kernel-definition-function-return-type operator def)))


;;; type environment
;;; type-environment ::= (<type-pair>*)
;;; type-pair        ::= (<variable> . <type>)

(defun empty-type-environment ()
  '())

(defun add-type-environment (var type type-env)
  (assert (valid-type-p type))
  (cons (cons var type) type-env))

(defun bulk-add-type-environment (bindings type-env)
  (reduce #'(lambda (type-env2 binding)
              (destructuring-bind (var type) binding
                (add-type-environment var type type-env2)))
          bindings
          :initial-value type-env))

(defun lookup-type-environment (var type-env)
  (match (assoc var type-env)
    ((_ . type) type)
    (_ (error "unbound variable: ~A" var))))

(defmacro with-type-environment ((var bindings) &body body)
  `(let ((,var (bulk-add-type-environment ',bindings (empty-type-environment))))
     ,@body))


;;; Timer

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


;;; utilities

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
