#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-examples.defglobal
  (:use :cl :cl-cuda)
  (:export :main))
(in-package :cl-cuda-examples.defglobal)

(setf cl-cuda:*show-messages* nil)


;;
;; Globals

(defglobal foo 0)

(defglobal bar 0 :constant)


;;
;; Kernel function

(defkernel add-globals (void ((out int*)))
  (set (aref out 0) (+ foo bar)))


;;
;; Main

(defun main ()
  (let ((dev-id 0))
    (with-cuda (dev-id)
      ;; First, set values to globals.
      (setf (global-ref 'foo 'int) 1)
      (setf (global-ref 'bar 'int) 2)
      ;; Second, launch kernel function and get sum of globals.
      (with-memory-blocks ((out 'int 1))
        ;; Launch ADD-GLOBALS kernel function.
        (add-globals out :grid-dim '(1 1 1) :block-dim '(1 1 1))
        ;; Synchronize memory block.
        (sync-memory-block out :device-to-host)
        ;; Output result.
        (format t "Got ~A from adding two globals.~%"
                (memory-block-aref out 0))))))
