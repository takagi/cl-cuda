#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(defpackage :cl-cuda-asd
  (:use :cl :asdf :uiop))
(in-package :cl-cuda-asd)

(load-system "cffi-grovel")

;;; These are the remnants of the previous solution for only groveling
;;; files when the cuda sdk is found. Unfortunately, since no output
;;; files were produced, all compononents that depended on the grovel
;;; file were always recompiled. So, this half solution is left around
;;; only to maintain backward compatibility with other systems that
;;; may use CUDA-GROVEL-FILE.
(defclass cuda-grovel-file (cffi-grovel:grovel-file) ())
(defmethod asdf:perform :around ((o operation) (c cuda-grovel-file))
  ;; Compile a grovel file only when CUDA SDK is found.
  (let ((sdk-not-found (symbol-value (intern "*SDK-NOT-FOUND*"
                                             "CL-CUDA.DRIVER-API"))))
    (if sdk-not-found
        (asdf::mark-operation-done o c)
        (call-next-method))))

;;; What we do instead is to test for the cuda sdk here, and use
;;; IF-FEATURE in the asdf system.
(cffi:define-foreign-library libcuda
  (:darwin (:framework "CUDA"))
  (:unix (:or "libcuda.so" "libcuda64.so")))
(unless (member :cuda-sdk *features*)
  (handler-case (progn
                  (cffi:use-foreign-library libcuda)
                  (pushnew :cuda-sdk *features*))
    (cffi:load-foreign-library-error (e)
      (princ e *error-output*)
      (terpri *error-output*))))

;;;
;;; Cl-cuda system definition
;;;

(defsystem "cl-cuda"
  :version "0.1"
  :author "Masayuki Takagi"
  :license "MIT"
  :depends-on ("cffi" "alexandria" "external-program" "osicat"
                      "cl-pattern" "split-sequence" "cl-reexport" "cl-ppcre")
  :components ((:module "src"
                        :serial t
                        :components
                        ((:module "driver-api"
                                  :serial t
                                  :components
                                  ((:file "package")
                                   (:file "get-error-string")
                                   (:file "cffi-grovel")
                                   (:file "sdk-not-found")
                                   (:file "library")
                                   (:file "type")
                                   (cffi-grovel:grovel-file
                                    "type-grovel"
                                    :if-feature :cuda-sdk)
                                   (:file "enum")
                                   (:file "function")))
                         (:module "lang"
                                  :serial t
                                  :components
                                  ((:file "util")
                                   (:file "data")
                                   (:file "type")
                                   (:file "syntax")
                                   (:file "environment")
                                   (:file "built-in")
                                   (:file "kernel")
                                   (:file "compiler/compile-data")
                                   (:file "compiler/compile-type")
                                   (:file "compiler/type-of-expression")
                                   (:file "compiler/compile-expression")
                                   (:file "compiler/compile-statement")
                                   (:file "compiler/compile-kernel")
                                   (:file "lang")))
                         (:module "api"
                                  :serial t
                                  :components
                                  ((:file "nvcc")
                                   (:file "kernel-manager")
                                   (:file "memory")
                                   (:file "context")
                                   (:file "defkernel")
                                   (:file "macro")
                                   (:file "timer")
                                   (:file "api")))
                         (:file "cl-cuda"))))
  :description "Cl-cuda is a library to use NVIDIA CUDA in Common Lisp programs."
  :long-description #.(read-file-string (subpathname *load-pathname* "README.markdown"))
  :in-order-to ((test-op (test-op "cl-cuda-test"))))
