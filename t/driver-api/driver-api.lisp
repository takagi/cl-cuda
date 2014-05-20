#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-cuda-test.driver-api)


(setf *test-result-output* *standard-output*)

(plan nil)


;;;
;;; test defcuenum
;;;

(diag "test defcuenum")

;; test enum-keyword
(is (enum-keyword '(:a 1)) :a)
(is (enum-keyword :a) :a)
(is-error (enum-keyword nil) simple-error)
(is-error (enum-keyword '(:a 1 2)) simple-error)

;; test enum-value
(is (enum-value '(:a 1)) 1)
(is-error (enum-value '(:a 1 2)) simple-error)
(is-error (enum-value :a) simple-error)

;; test expansion of defcuenum macro
(is-expand (defcuenum cu-event-flags-enum
             (:cu-event-default #X0)
             (:cu-event-blocking-sync #X1)
             (:cu-event-disable-timing #X2)
             (:cu-event-interprocess #X4))
           (progn
             (cffi:defcenum cu-event-flags-enum
               (:cu-event-default #X0)
               (:cu-event-blocking-sync #X1)
               (:cu-event-disable-timing #X2)
               (:cu-event-interprocess #X4))
             (defconstant cu-event-default
               (cffi:foreign-enum-value 'cu-event-flags-enum
                                        :cu-event-default))
             (defconstant cu-event-blocking-sync
               (cffi:foreign-enum-value 'cu-event-flags-enum
                                        :cu-event-blocking-sync))
             (defconstant cu-event-disable-timing
               (cffi:foreign-enum-value 'cu-event-flags-enum
                                        :cu-event-disable-timing))
             (defconstant cu-event-interprocess
               (cffi:foreign-enum-value 'cu-event-flags-enum
                                        :cu-event-interprocess))))


;;;
;;; test CUDA driver API
;;;

;; test cuInit
(diag "test cuInit")
(cu-init 0)

;; test cuDeviceGet
(diag "test cuDeviceGet")
(let ((dev-id 0))
  (cffi:with-foreign-object (device 'cu-device)
    (setf (cffi:mem-ref device :int) 42)
    (cu-device-get device dev-id)
    (format t "CUDA device handle: ~A~%" (cffi:mem-ref device 'cu-device))))

;; test cuDeviceGetCount
(diag "test cuDeviceGetCount")
(cffi:with-foreign-object (count :int)
  (cu-device-get-count count)
  (format t "CUDA device count: ~A~%" (cffi:mem-ref count :int)))

;; test cuDeviceComputeCapability
(diag "test cuDeviceComputeCapability")
(let ((dev-id 0))
  (cffi:with-foreign-objects ((major :int)
                              (minor :int)
                              (device 'cu-device))
    (cu-device-get device dev-id)
    (cu-device-compute-capability major minor (cffi:mem-ref device 'cu-device))
    (format t "CUDA device compute capability: ~A.~A~%"
              (cffi:mem-ref major :int) (cffi:mem-ref minor :int))))

;; test cuDeviceGetName
(diag "test cuDeviceGetName")
(let ((dev-id 0))
  (cffi:with-foreign-object (device 'cu-device)
  (cffi:with-foreign-pointer-as-string ((name size) 255)
    (cu-device-get device dev-id)
    (cu-device-get-name name size (cffi:mem-ref device 'cu-device))
    (format t "CUDA device name: ~A~%" (cffi:foreign-string-to-lisp name)))))

;; test cuCtxCreate/cuCtxDestroy
(diag "test cuCtxCreate/cuCtxDestroy")
(let ((flags 0)
      (dev-id 0))
  (cffi:with-foreign-objects ((pctx 'cu-context)
                              (device 'cu-device))
    (cu-device-get device dev-id)
    (cu-ctx-create pctx flags (cffi:mem-ref device 'cu-device))
    (cu-ctx-destroy (cffi:mem-ref pctx 'cu-context))))

;; test cuMemAlloc/cuMemFree
(diag "test cuMemAlloc/cuMemFree")
(let ((flags 0)
      (dev-id 0))
  (cffi:with-foreign-objects ((device 'cu-device)
                              (pctx 'cu-context)
                              (dptr 'cu-device-ptr))
    (cu-device-get device dev-id)
    (cu-ctx-create pctx flags (cffi:mem-ref device 'cu-device))
    (cu-mem-alloc dptr 1024)
    (cu-mem-free (cffi:mem-ref dptr 'cu-device-ptr))
    (cu-ctx-destroy (cffi:mem-ref pctx 'cu-context))))

;; test cuMemAlloc/cuMemFree using with-cu-context
(diag "test cuMemAlloc/cuMemFree using with-cu-context")
(let ((dev-id 0))
  (with-cu-context (dev-id)
    (cffi:with-foreign-object (dptr 'cu-device-ptr)
      (cu-mem-alloc dptr 1024)
      (cu-mem-free (cffi:mem-ref dptr 'cu-device-ptr)))))

;; test cuMemcpyHtoD/cuMemcpyDtoH
(diag "test cuMemcpyHtoD/cuMemcpyDtoH")
(let ((dev-id 0)
      (size 1024))
  (with-cu-context (dev-id)
    (cffi:with-foreign-objects ((hptr :float size)
                                (dptr 'cu-device-ptr))
      (cu-mem-alloc dptr size)
      (cu-memcpy-host-to-device (cffi:mem-ref dptr 'cu-device-ptr) hptr size)
      (cu-memcpy-device-to-host hptr (cffi:mem-ref dptr 'cu-device-ptr) size)
      (cu-mem-free (cffi:mem-ref dptr 'cu-device-ptr)))))

;; test cuModuleLoad
(diag "test cuModuleLoad")
(labels ((get-test-path ()
           (namestring (asdf:system-relative-pathname :cl-cuda #P"t"))))
  (let ((dev-id 0)
        (ptx-path (concatenate 'string (get-test-path)
                               "/vectorAdd_kernel.ptx")))
    (cffi:with-foreign-string (fname ptx-path)
      (with-cu-context (dev-id)
        (cffi:with-foreign-object (module 'cu-module)
          (cu-module-load module fname)
          (format t "CUDA module \"vectorAdd_kernel.ptx\" is loaded.~%"))))))

;; test cuModuleGetFunction
(diag "test cuModuleGetFunction")
(labels ((get-test-path ()
           (namestring (asdf:system-relative-pathname :cl-cuda #P"t"))))
  (let ((dev-id 0)
        (ptx-path (concatenate 'string (get-test-path)
                               "/vectorAdd_kernel.ptx")))
    (cffi:with-foreign-string (fname ptx-path)
      (cffi:with-foreign-string (name "VecAdd_kernel")
        (with-cu-context (dev-id)
          (cffi:with-foreign-objects ((module 'cu-module)
                                      (hfunc 'cu-function))
            (cu-module-load module fname)
            (cu-module-get-function hfunc (cffi:mem-ref module 'cu-module)
                                    name)))))))


;;;
;;; test CUDA Event Management functions
;;;

(let ((dev-id 0))
  (with-cu-context (dev-id)
    (cffi:with-foreign-objects ((start-event 'cu-event)
                                (stop-event 'cu-event)
                                (milliseconds :float))
      (cu-event-create start-event cu-event-default)
      (cu-event-create stop-event  cu-event-default)
      (cu-event-record (cffi:mem-ref start-event 'cu-event) (cffi:null-pointer))
      (cu-event-record (cffi:mem-ref stop-event 'cu-event) (cffi:null-pointer))
      (cu-event-synchronize (cffi:mem-ref stop-event 'cu-event))
      (cu-event-query (cffi:mem-ref stop-event 'cu-event))
      (cu-event-elapsed-time milliseconds
                                      (cffi:mem-ref start-event 'cu-event)
                                      (cffi:mem-ref stop-event 'cu-event))
      (format t "CUDA Event - elapsed time: ~A~%"
                (cffi:mem-ref milliseconds :float))
      (cu-event-destroy (cffi:mem-ref start-event 'cu-event))
      (cu-event-destroy (cffi:mem-ref stop-event 'cu-event)))))
