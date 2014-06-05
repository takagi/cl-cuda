#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-cuda.driver-api)


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
    (cl-pattern:match (ensure-list enum-elem)
      ((keyword) keyword)
      ((keyword _) keyword)
      (_  (error (format nil "invalid enum element: ~A" enum-elem))))))

(eval-when (:compile-toplevel :load-toplevel)
  (defun enum-value (enum-elem)
    (cl-pattern:match enum-elem
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
