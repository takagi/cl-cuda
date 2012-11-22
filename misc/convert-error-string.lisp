#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-cuda-misc)

(defun read-lines (path)
  (with-open-file (in path)
    (unless in
      (error "cannot open file: ~A" path))
    (loop for line = (read-line in nil)
      while line
      collect line)))

(let ((pattern "(\".+\"), (\\d+)"))
  (defun scan-error-string (s)
    (multiple-value-bind (_ xs) (cl-ppcre:scan-to-strings pattern s)
      (declare (ignorable _))
      (when xs
        (list (parse-integer (aref xs 1)) (aref xs 0))))))   

(defun read-error-strings ()
  (let ((path (asdf:system-relative-pathname :cl-cuda #P"misc/drvapi_error_string.h")))
    (remove nil (mapcar #'scan-error-string (read-lines path)))))

(defun output-comment (out)
  (let ((timestamp (format-timestring nil (now) :format '(:short-month ". " :day " " :year))))
    (format out "#|~%")
    (format out "  This file is a part of cl-cuda project.~%")
    (format out "  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)~%")
    (format out "|#~%~%")
    (format out "#|~%")
    (format out "  This file is automatically generated from drvapi_error_string.h in CUDA SDK, do not edit.~%")
    (format out "  Timestamp: ~A~%" timestamp)
    (format out "|#~%~%")))

(defun output-in-package (out)
  (format out "(in-package :cl-cuda)~%~%"))

(defun output-defparameter (out)
  (format out "(defparameter +error-strings+~%")
  (format out "  '(")
  (loop for (num str) in (read-error-strings)
     do (format out "~A ~A~%    " num str))
  (format out "))~%~%"))

(defun output-get-error-string-definition (out)
  (format out "(defun get-error-string (n)~%")
  (format out "  (or (getf +error-strings+ n)~%")
  (format out "      (error \"invalid CUDA driver API error No.: ~~A\" n)))~%"))

(defun convert-error-string ()
  (let ((path (asdf:system-relative-pathname :cl-cuda #P"src/cl-cuda-error-string.lisp")))
    (with-open-file (out path :direction :output :if-exists :supersede)
      (unless out
        (error "cannot open file: drvapi_error_string.h"))
      (output-comment out)
      (output-in-package out)
      (output-defparameter out)
      (output-get-error-string-definition out)
      )))