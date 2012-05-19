#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-examples.vector-add
  (:use :cl
        :cl-cuda)
  (:export :main))
(in-package :cl-cuda-examples.vector-add)

(defun random-init (data n)
  (dotimes (i n)
    (setf (cffi:mem-aref data :float i) (random 1.0))))

(defun verify-result (as bs cs n)
  (dotimes (i n)
    (let ((a (cffi:mem-aref as :float i))
          (b (cffi:mem-aref bs :float i))
          (c (cffi:mem-aref cs :float i)))
      (let ((sum (+ a b)))
        (when (> (abs (- c sum)) 1.0)
          (error (format nil "verification fault, i:~A a:~A b:~A c:~A"
                         i a b c))))))
  (format t "verification succeed.~%"))

(defkernel vec-add-kernel (void ((a float*) (b float*) (c float*) (n int)))
  (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (if (< i n)
        (set (aref c i)
             (+ (aref a i) (aref b i))))))

(defun main ()
  (let* ((dev-id 0)
         (n 1024)
         (size (* n 4))                   ; 4 is size of float
         (threads-per-block 256)
         (blocks-per-grid (/ n threads-per-block)))
    (with-cuda-context (dev-id)
      (cffi:with-foreign-objects ((h-a :float n)
                                  (h-b :float n)
                                  (h-c :float n))
        (with-cuda-memory-blocks ((d-a size)
                                  (d-b size)
                                  (d-c size))
          (random-init h-a n)
          (random-init h-b n)
          (cu-memcpy-host-to-device (cffi:mem-ref d-a 'cu-device-ptr) h-a size)
          (cu-memcpy-host-to-device (cffi:mem-ref d-b 'cu-device-ptr) h-b size)
          (vec-add-kernel d-a d-b d-c n
                          :grid-dim (list blocks-per-grid 1 1)
                          :block-dim (list threads-per-block 1 1))
          (cu-memcpy-device-to-host h-c (cffi:mem-ref d-c 'cu-device-ptr) size)
          (verify-result h-a h-b h-c n))))))
