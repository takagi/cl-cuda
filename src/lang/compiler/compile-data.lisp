#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda.lang.compiler.compile-data
  (:use :cl
        :cl-cuda.lang.data
        :cl-cuda.lang.util)
  (:export :compile-symbol
           :compile-bool
           :compile-int
           :compile-float
           :compile-double))
(in-package :cl-cuda.lang.compiler.compile-data)


;;;
;;; Symbol
;;;

(defun compile-symbol (expr)
  (unless (cl-cuda-symbol-p expr)
    (error "The value ~S is an invalid expression." expr))
  (c-identifier expr))


;;;
;;; Bool
;;;

(defun compile-bool (expr)
  (unless (cl-cuda-bool-p expr)
    (error "The value ~S is an invalid expression." expr))
  (if expr "true" "false"))


;;;
;;; Int
;;;

(defun compile-int (expr)
  (unless (cl-cuda-int-p expr)
    (error "The value ~S is an invalid expression." expr))
  (princ-to-string expr))


;;;
;;; Float
;;;

(defun compile-float (expr)
  (unless (cl-cuda-float-p expr)
    (error "The value ~S is an invalid expression." expr))
  (princ-to-string expr))


;;;
;;; Double
;;;

(defun compile-double (expr)
  (unless (cl-cuda-double-p expr)
    (error "The value ~S is an invalid expression." expr))
  (format nil "(double)~S" (float expr 0.0)))
