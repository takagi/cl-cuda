#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda.api.context
  (:use :cl
        :cl-cuda.driver-api
        :cl-cuda.api.nvcc
        :cl-cuda.api.kernel-manager)
  (:export ;; Initialize CUDA
           :init-cuda
           ;; CUDA device
           :get-cuda-device
           :device-compute-capability
           ;; CUDA context
           :create-cuda-context
           :destroy-cuda-context
           :synchronize-context
           ;; WITH-CUDA macro
           :*cuda-device*
           :*cuda-context*
           :with-cuda
           :*cuda-stream*))
(in-package :cl-cuda.api.context)


;;;
;;; Initialize CUDA
;;;

(defun init-cuda ()
  (cu-init 0))


;;;
;;; CUDA device
;;;

(defun get-cuda-device (dev-id)
  (cffi:with-foreign-object (device-ptr 'cu-device)
    (cu-device-get device-ptr dev-id)
    (cffi:mem-ref device-ptr 'cu-device)))

(defun device-compute-capability (device)
  (cffi:with-foreign-objects ((major :int)
                              (minor :int))
    (cu-device-compute-capability major minor device)
    (values (cffi:mem-ref major :int)
            (cffi:mem-ref minor :int))))


;;;
;;; CUDA context
;;;

(defun create-cuda-context (device)
  (cffi:with-foreign-object (context-ptr 'cu-context)
    (cu-ctx-create context-ptr 0 device)
    (cffi:mem-ref context-ptr 'cu-context)))

(defun destroy-cuda-context (context)
  (cu-ctx-destroy context))

(defun synchronize-context ()
  (cu-ctx-synchronize))


;;;
;;; WITH-CUDA macro
;;;

(defvar *cuda-device*)

(defvar *cuda-context*)

(defun get-nvcc-arch (dev-id)
  (multiple-value-bind (major minor)
      (device-compute-capability dev-id)
    (format nil "-arch=sm_~D~D" major minor)))

(defun arch-exists-p (options)
  (some #'(lambda (option)
            (eql 0 (search "-arch=" option)))
        options))

(defun append-arch (options dev-id)
  (check-type options list)
  (cons (get-nvcc-arch dev-id)
        options))

(defmacro with-cuda ((dev-id) &body body)
  `(progn
     ;; Initialize CUDA.
     (init-cuda)
     (let* (;; Get CUDA device.
            (*cuda-device* (get-cuda-device ,dev-id))
            ;; Create CUDA context.
            (*cuda-context* (create-cuda-context *cuda-device*))
            ;; Append nvcc arch option if not specified.
            (*nvcc-options* (if (arch-exists-p *nvcc-options*)
                                *nvcc-options*
                                (append-arch *nvcc-options* ,dev-id))))
       (unwind-protect (progn ,@body)
         ;; Unload kernel manager.
         (kernel-manager-unload *kernel-manager*)
         ;; Destroy CUDA context.
         (destroy-cuda-context *cuda-context*)))))

(defvar *cuda-stream* (cffi:null-pointer))
