#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(eval-when (:load-toplevel :execute)
  (asdf:operate 'asdf:load-op 'cffi-grovel))

(in-package :cl-user)
(defpackage cl-cuda-asd
  (:use :cl :asdf))
(in-package :cl-cuda-asd)

(defsystem cl-cuda
  :version "0.1"
  :author "Masayuki Takagi"
  :license "LLGPL"
  :depends-on (:cffi :alexandria :external-program :osicat
               :cl-pattern :split-sequence :cl-reexport :cl-ppcre)
  :components ((:module "src"
                :serial t
                :components
                ((:module "driver-api"
                  :serial t
                  :components
                  ((:file "package")
                   (:file "get-error-string")
                   (:file "cffi-grovel")
                   (:file "library")
                   (cffi-grovel:grovel-file "type-grovel")
                   (cffi-grovel:grovel-file "enum-grovel")
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
  :long-description
  #.(with-open-file (stream (merge-pathnames
                             #p"README.markdown"
                             (or *load-pathname* *compile-file-pathname*))
                            :if-does-not-exist nil
                            :direction :input)
      (when stream
        (let ((seq (make-array (file-length stream)
                               :element-type 'character
                               :fill-pointer t)))
          (setf (fill-pointer seq) (read-sequence seq stream))
          seq)))
  :in-order-to ((test-op (load-op cl-cuda-test))))
