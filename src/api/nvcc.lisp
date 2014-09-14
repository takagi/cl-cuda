#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda.api.nvcc
  (:use :cl)
  (:export :*tmp-path*
           :*nvcc-options*
           :*nvcc-binary*
           :nvcc-compile))
(in-package :cl-cuda.api.nvcc)


;;;
;;; Helper
;;;

(defvar *tmp-path* (make-pathname :directory "tmp"))

(defun get-tmp-path ()
  *tmp-path*)

(defun get-cu-path ()
  (let ((name (format nil "cl-cuda.~A" (osicat-posix:mktemp))))
    (make-pathname :name name :type "cu" :defaults (get-tmp-path))))

(defun get-ptx-path (cu-path)
  (make-pathname :type "ptx" :defaults cu-path))

(defun get-include-path ()
  (asdf:system-relative-pathname :cl-cuda #P"include"))

(defvar *nvcc-options*
  ;; compute capability 1.3 is needed for double floats, but 2.0 for
  ;; good performance
  (list "-arch=sm_11"))

(defun get-nvcc-options (cu-path ptx-path)
  (let ((include-path (get-include-path)))
    (append *nvcc-options*
            (list "-I" (namestring include-path)
                  "-ptx"
                  "-o" (namestring ptx-path)
                  (namestring cu-path)))))


;;;
;;; Compiling with invoking NVCC
;;;

(defun nvcc-compile (cuda-code)
  (let* ((cu-path (get-cu-path))
         (ptx-path (get-ptx-path cu-path)))
    (output-cuda-code cu-path cuda-code)
    (print-nvcc-command cu-path ptx-path)
    (run-nvcc-command cu-path ptx-path)
    (namestring ptx-path)))

(defun output-cuda-code (cu-path cuda-code)
  (with-open-file (out cu-path :direction :output :if-exists :supersede)
    (princ cuda-code out)))

(defvar *nvcc-binary* "nvcc"
  "Set this to an absolute path if your lisp doesn't search PATH.")

(defun print-nvcc-command (cu-path ptx-path)
  (let ((options (get-nvcc-options cu-path ptx-path)))
    (format t "~A~{ ~A~}~%" *nvcc-binary* options)))

(defun run-nvcc-command (cu-path ptx-path)
  (let ((options (get-nvcc-options cu-path ptx-path)))
    (with-output-to-string (out)
      (multiple-value-bind (status exit-code)
          (external-program:run *nvcc-binary* options :error out)
        (unless (and (eq status :exited) (= 0 exit-code))
          (error "nvcc exits with code: ~A~%~A" exit-code
                 (get-output-stream-string out)))))))
