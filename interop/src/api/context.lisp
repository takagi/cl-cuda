#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-interop.api.context
  (:use :cl :cl-reexport
        :cl-cuda.api.kernel-manager
        :cl-cuda-interop.driver-api)
  (:export ;; CUDA context
           :create-cuda-context
           ;; WITH-CUDA macro
           :with-cuda))
(in-package :cl-cuda-interop.api.context)

(eval-when (:execute :load-toplevel :compile-toplevel)
  (reexport-from :cl-cuda.api.context
                 :exclude '(:create-cuda-context
                            :with-cuda)))


;;;
;;; CUDA context
;;;

(defun create-cuda-context (device)
  (cffi:with-foreign-object (context-ptr 'cu-context)
    (cu-gl-ctx-create context-ptr 0 device)
    (cffi:mem-ref context-ptr 'cu-context)))


;;;
;;; WITH-CUDA macro
;;;

(defmacro with-cuda ((dev-id &key (interop t)) &body body)
  `(progn
     ;; initialize CUDA
     (init-cuda)
     (let* (;; get CUDA device
            (*cuda-device* (get-cuda-device ,dev-id))
            ;; create CUDA context
            (*cuda-context*
              (if ,interop
                  (create-cuda-context *cuda-device*)
                  (cl-cuda:create-cuda-context *cuda-device*))))
       (unwind-protect (progn ,@body)
         ;; unload kernel manager
         (kernel-manager-unload *kernel-manager*)
         ;; destroy CUDA context
         (destroy-cuda-context *cuda-context*)))))
