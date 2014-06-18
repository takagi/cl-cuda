#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage :cl-cuda.api.macro
  (:use :cl
        :cl-cuda.api.defkernel)
  (:export :let*
           :when
           :unless))
(in-package :cl-cuda.api.macro)


(defkernelmacro let* (bindings &body body)
  (if bindings
      `(let (,(car bindings))
         (let* (,@(cdr bindings))
           ,@body))
      `(progn ,@body)))

(defkernelmacro when (test &body body)
  `(if ,test
       (progn ,@body)))

(defkernelmacro unless (test &body body)
  `(if (not ,test)
       (progn ,@body)))
