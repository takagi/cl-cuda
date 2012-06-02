#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

#|
  This file is based on the CUDA SDK's "vectorAdd" sample.
|#

(in-package :cl-user)
(defpackage cl-cuda-examples.vector-add
  (:use :cl
        :cl-cuda)
  (:export :main))
(in-package :cl-cuda-examples.vector-add)

(defun random-init (data n)
  (dotimes (i n)
    (setf (mem-aref data i) (random 1.0))))

(defun verify-result (as bs cs n)
  (dotimes (i n)
    (let ((a (mem-aref as i))
          (b (mem-aref bs i))
          (c (mem-aref cs i)))
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
         (threads-per-block 256)
         (blocks-per-grid (/ n threads-per-block)))
    (with-cuda-context (dev-id)
      (with-memory-blocks ((a 'float n)
                           (b 'float n)
                           (c 'float n))
        (random-init a n)
        (random-init b n)
        (memcpy-host-to-device a b)
        (vec-add-kernel a b c n
                        :grid-dim (list blocks-per-grid 1 1)
                        :block-dim (list threads-per-block 1 1))
        (memcpy-device-to-host c)
        (verify-result a b c n)))))
