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
           :synchronize-cuda-context))
(in-package :cl-cuda.api.context)


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

(defun synchronize-cuda-context ()
  (cu-ctx-synchronize))
