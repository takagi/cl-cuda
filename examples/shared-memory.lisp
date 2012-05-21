#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-examples.shared-memory
  (:use :cl
        :cl-cuda
        :alexandria)
  (:export :main))
(in-package :cl-cuda-examples.shared-memory)

(defmacro def-global-memory (n)
  (let ((name (symbolicate "GLOBAL-MEMORY-" (princ-to-string n))))
    `(defkernel ,name (void ((a float*)))
       (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
         ,@(loop repeat n
              collect '(set (aref a i) (+ (aref a i) 1.0)))))))

(def-global-memory 1000)
(def-global-memory 2000)
(def-global-memory 3000)
(def-global-memory 4000)

(defkernel global-memory (void ((a float*)))
  (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (set (aref a i) (+ (aref a i) 1.0))))

#|
(defkernel global-memory (void ((a float*) (n int)))
  (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (for (i 0 n)
      (set (aref a i) (+ (aref a i) 1.0)))))
|#

(defmacro def-shared-memory (n)
  (let ((name (symbolicate "SHARED-MEMORY-" (princ-to-string n))))
    `(defkernel ,name (void ((a float*)))
       (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
         (with-shared-memory ((s float 16))
           (set (aref s thread-idx-x) (aref a i))
           ,@(loop repeat n
                collect '(set (aref s thread-idx-x)
                              (+ (aref s thread-idx-x) 1.0)))
           (set (aref a i) (aref s thread-idx-x)))))))

(def-shared-memory 1000)
(def-shared-memory 2000)
(def-shared-memory 3000)
(def-shared-memory 4000)

(defkernel shared-memory (void ((a float*)))
  (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (with-shared-memory ((s float 16))
      (set (aref s thread-idx-x) (aref a i))
      (set (aref s thread-idx-x) (+ (aref s thread-idx-x) 1.0))
      (set (aref a i) (aref s thread-idx-x)))))

(defun init (a n)
  (dotimes (i n)
    (setf (cffi:mem-aref a :float i) 0.0)))

(defun verify (a n expected)
  (dotimes (i n)
    (unless (= (cffi:mem-aref a :float i) expected)
      (error (format nil "verification fault: ~A ~A"
                         (cffi:mem-aref a :float i) expected))))
  (format t "verification succeed.~%"))

(defun main (func expected)
  (let ((n (* 256 256)))
    (with-cuda-context (0)
      (cffi:with-foreign-object (h-a :float n)
      (with-cuda-memory-block (d-a (* n 4))
        (init h-a n)
        (cu-memcpy-host-to-device (cffi:mem-ref d-a 'cu-device-ptr) h-a (* n 4))
        (time
         (dotimes (i 100)
           (funcall func d-a
                    :grid-dim (list (/ n 16) 1 1)
                    :block-dim '(16 1 1))
           (cu-ctx-synchronize)))
        (cu-memcpy-device-to-host h-a (cffi:mem-ref d-a 'cu-device-ptr) (* n 4))
        (verify h-a n expected))))))

(defun main-shared-memory ()
  (format t "#shared-memory-1000~%")
  (main #'shared-memory-1000 100000.0)  ; took 1.421 [sec]
  (format t "#shared-memory-2000~%")
  (main #'shared-memory-2000 200000.0)  ; took 4.744 [sec]
  (format t "#shared-memory-3000~%")
  (main #'shared-memory-3000 300000.0)  ; took 7.181 [sec]
  (format t "#shared-memory-4000~%")
  (main #'shared-memory-4000 400000.0)  ; took 9,483 [sec]
  )

(defun main-global-memory ()
  (format t "#global-memory-1000~%")
  (main #'global-memory-1000 100000.0)  ; took 3.895 [sec]
  (format t "#global-memory-2000~%")
  (main #'global-memory-2000 200000.0)  ; took 7.617 [sec]
  (format t "#global-memory-3000~%")
  (main #'global-memory-3000 300000.0)  ; took 11.344 [sec]
  (format t "#global-memory-4000~%")
  (main #'global-memory-4000 400000.0)  ; took 15.077 [sec]
  )
