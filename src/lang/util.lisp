#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda.lang.util
  (:use :cl)
  (:export :c-identifier))
(in-package cl-cuda.lang.util)


(defun c-identifier (object)
  (substitute-if #\_ (lambda (char)
                       (and (not (alphanumericp char))
                            (not (char= #\_ char))
                            (not (char= #\* char))))
                 (string-downcase object)))
