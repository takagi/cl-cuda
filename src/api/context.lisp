#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda.api.context
  (:use :cl
        :cl-cuda.driver-api
        :cl-cuda.api.kernel-manager)
  (:export :*cuda-context*
           :init-cuda-context
           :release-cuda-context
           :with-cuda-context
           :synchronize-context))
(in-package :cl-cuda.api.context)


;;;
;;; CUcontext
;;;

(defun create-cu-context (dev-id)
  (let (device context)
    ;; initialize CUDA
    (cu-init 0)
    ;; get CUdevice
    (cffi:with-foreign-object (device-ptr 'cu-device)
      (cu-device-get device-ptr dev-id)
      (setf device (cffi:mem-ref device-ptr 'cu-device)))
    ;; create CUcontext
    (cffi:with-foreign-object (context-ptr 'cu-context)
      (cu-ctx-create context-ptr 0 device)
      (setf context (cffi:mem-ref context-ptr 'cu-context)))
    context))

(defun destroy-cu-context (context)
  (cu-ctx-destroy context))

(defmacro with-cu-context ((dev-id) &body body)
  (with-gensyms (context)
    `(let ((,context (create-cu-context ,dev-id)))
       (unwind-protect (progn ,@body)
         (destroy-cu-context ,context)))))


;;;
;;; CUDA context
;;;

(defvar *cuda-context*)

(defun init-cuda-context (dev-id)
  (create-cu-context dev-id))

(defun release-cuda-context (context)
  (kernel-manager-unload *kernel-manager*)
  (destroy-cu-context context))

(defmacro with-cuda-context ((dev-id) &body body)
  `(let ((*cuda-context* (init-cuda-context ,dev-id)))
     (unwind-protect (progn ,@body)
       (release-cuda-context *cuda-context*))))

(defun synchronize-context ()
  (cu-ctx-synchronize))
