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

(defun list->plist (num-and-str)
  (destructuring-bind (num str) num-and-str
    `(:num ,num :str ,str)))

(defun output-template (out)
  (let ((path (asdf:system-relative-pathname :cl-cuda #P"misc/get-error-string.template"))
        (timestamp (local-time:format-timestring nil (local-time:now)
                                                 :format '(:short-month ". " :day " " :year)))
        (error-strings (mapcar #'list->plist (read-error-strings))))
    (cl-emb:register-emb "tmpl" path)
    (format out (cl-emb:execute-emb "tmpl" :env `(:timestamp     ,timestamp
                                                  :error-strings ,error-strings)))))

(defun convert-error-string ()
  (let ((path (asdf:system-relative-pathname :cl-cuda #P"src/driver-api/get-error-string.lisp")))
    (with-open-file (out path :direction :output :if-exists :supersede)
      (unless out
        (error "cannot open file: ~A" path))
      (output-template out))))

