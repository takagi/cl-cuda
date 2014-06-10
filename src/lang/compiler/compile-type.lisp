#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda.lang.compiler.compile-type
  (:use :cl
        :cl-cuda.lang.type)
  (:export :compile-type))
(in-package :cl-cuda.lang.compiler.compile-type)


;;;
;;; Type
;;;

(defun compile-type (type)
  (unless (cl-cuda-type-p type)
    (error "The value ~S is an invalid cl-cuda type." type))
  (cuda-type type))
