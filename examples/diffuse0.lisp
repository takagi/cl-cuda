#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-examples.diffuse0
  (:use :cl :cl-cuda)
  (:export :main))
(in-package :cl-cuda-examples.diffuse0)

(setf cl-cuda:*show-messages* nil)


;;; image output functions

(declaim (inline index))
(defun index (nx jx jy)
  (the fixnum (+ (the fixnum (* nx jy)) jx)))

(defun image-value (f i j nx fmax fmin)
  (let ((fc (memory-block-aref f (index nx i j))))
    (truncate (* 256.0
                 (/ (- fc fmin) (- fmax fmin))))))

(defun file-name (dir i nout)
  (let ((n (truncate (/ i nout))))
    (concatenate 'string dir (format nil "~4,'0D.pgm" n))))

(defun output-pnm (dir i nout nx ny f)
  (let ((image (make-instance 'imago:grayscale-image
                              :width nx :height ny)))
    (dotimes (i nx)
      (dotimes (j ny)
        (setf (imago:image-pixel image i j) (image-value f i j nx 1.0 0.0))))
    (imago:write-pnm image (file-name dir i nout) :ASCII))
  (values))


;;; print functions

(defun print-elapsed-time (elapsed-time)
  (let ((time (* elapsed-time 1.0e-3)))
    (format t "Elapsed Time = ~,3F [sec]~%" time)))

(defun print-performance (flo elapsed-time)
  (let ((time (* elapsed-time 1.0e-3)))
    (format t "Performance = ~,2F [MFlops]~%" (* (/ flo time) 1.0e-6))))

(defun print-time (cnt time)
  (format t "time(~A) = ~,5F~%" cnt time))


;;; main functions

(defkernel cuda-diffusion2d (void ((f float*) (fn float*)
                                   (nx int) (ny int)
                                   (c0 float) (c1 float) (c2 float)))
  (let* ((jy (+ (* block-dim-y block-idx-y) thread-idx-y))
         (jx (+ (* block-dim-x block-idx-x) thread-idx-x))
         (j (+ (* nx jy) jx)))
    (let ((fcc (aref f j))
          (fcw 0.0)
          (fce 0.0)
          (fcs 0.0)
          (fcn 0.0))
      (if (= jx 0)
          (set fcw fcc)
          (set fcw (aref f (- j 1))))
      (if (= jx (- nx 1))
          (set fce fcc)
          (set fce (aref f (+ j 1))))
      (if (= jy 0)
          (set fcs fcc)
          (set fcs (aref f (- j nx))))
      (if (= jy (- ny 1))
          (set fcn fcc)
          (set fcn (aref f (+ j nx))))
      (set (aref fn j) (+ (* c0 (+ fce fcw))
                          (* c1 (+ fcn fcs))
                          (* c2 fcc))))))

(defun initialize-device-memory (nx ny dx dy f)
  (let ((alpha 30.0))
    (dotimes (jy ny)
      (dotimes (jx nx)
        (let ((j (index nx jx jy))
              (x (- (* dx (+ (float jx 1.0) 0.5)) 0.5))
              (y (- (* dy (+ (float jy 1.0) 0.5)) 0.5)))
          (setf (memory-block-aref f j)
                (exp (* (- alpha)
                        (+ (* x x) (* y y)))))))))
  (sync-memory-block f :host-to-device))

(defvar +block-dim-x+ 256)
(defvar +block-dim-y+ 1)

(defun diffusion2d (nx ny f fn kappa dt dx dy)
  (let* ((c0 (* kappa (/ dt (* dx dx))))
         (c1 (* kappa (/ dt (* dy dy))))
         (c2 (- 1.0 (* 2.0 (+ c0 c1)))))
    (cuda-diffusion2d f fn nx ny c0 c1 c2
                      :grid-dim (list (/ nx +block-dim-x+)
                                      (/ ny +block-dim-y+) 1)
                      :block-dim (list +block-dim-x+ +block-dim-y+ 1))
    (synchronize-context)
    (* nx ny 7.0)))

(defmacro swap (a b)
  `(rotatef ,a ,b))

(defun main ()
  (let* ((dev-id 0)
         (nx 256) (ny 256)
         (nout 500)
         (Lx 1.0) (Ly 1.0)
         (dx (/ Lx (float nx 1.0)))
         (dy (/ Ly (float ny 1.0)))
         (kappa 0.1)
         (dt (/ (* 0.2 (min (* dx dx) (* dy dy))) kappa))
         (dir (namestring (truename "./")))
         (time 0)
         (flo 0))
    (with-cuda (dev-id)
      (with-timer (timer)
        (with-memory-blocks ((f 'float (* nx ny))
                             (fn 'float (* nx ny)))
          (initialize-device-memory nx ny dx dy f)
          (start-timer timer)
          (dotimes (i 20000)
            (when (= (mod i 100) 0)
              (print-time i time))
          ;(when (= (mod i nout) 0)
          ;  (output-pnm dir i nout nx ny f))
            (incf flo (diffusion2d nx ny f fn kappa dt dx dy))
            (incf time dt)
            (swap f fn))
          (print-time 20000 time)
          ;(output-pnm dir 20000 nout nx ny f)
          (stop-timer timer)
          (synchronize-timer timer)
          (let ((elapsed-time (elapsed-time timer)))
            (print-elapsed-time elapsed-time)
            (print-performance flo elapsed-time))))))
  (values))
