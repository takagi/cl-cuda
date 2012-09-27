#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

#|
  This file is based on the CUDA SDK's "nbody" sample.
|#

(in-package :cl-user)
(defpackage cl-cuda-examples.nbody
  (:use :cl
        :cl-cuda)
  (:export :main))
(in-package :cl-cuda-examples.nbody)


;;;
;;; Kernel functions
;;;

(defkernel body-body-interaction (float3 ((ai float3) (bi float4) (bj float4)))
  (let ((r (float3 (- (float4-x bj) (float4-x bi))
                   (- (float4-y bj) (float4-y bi))
                   (- (float4-z bj) (float4-z bi))))
        (softening-squared (* 0.1 0.1))
        (dist-sqr (+ (* (float3-x r) (float3-x r))
                     (* (float3-y r) (float3-y r))
                     (* (float3-z r) (float3-z r))
                     softening-squared))
        (inv-dist (rsqrtf dist-sqr))
        (inv-dist-cube (* inv-dist inv-dist inv-dist))
        (s (* (float4-w bj) inv-dist-cube)))
    (set (float3-x ai) (+ (float3-x ai) (* (float3-x r) s)))
    (set (float3-y ai) (+ (float3-y ai) (* (float3-y r) s)))
    (set (float3-z ai) (+ (float3-z ai) (* (float3-z r) s)))
    (return ai)))

(defkernel gravitation (float3 ((ipos float4) (accel float3) (shared-pos float4*)))
;  (with-shared-memory ((shared-pos float4 256))
    (for ((j 0 (- block-dim-x 1)))
      (set accel (body-body-interaction accel ipos (aref shared-pos j))));)
  (return accel))

(defkernel wrap (int ((x int) (m int)))
  (if (< x m)
      (return x)
      (return (- x m))))

(defkernel compute-body-accel (float3 ((body-pos float4) (positions float4*)
                                       (num-bodies int)))
  (with-shared-memory ((shared-pos float4 256))
    (let ((acc (float3 0.0 0.0 0.0))
          (p block-dim-x)
          (n num-bodies)
          (num-tiles (/ n p)))
      (for ((tile 0 (- num-tiles 1)))
        (let ((idx (+ (* (wrap (+ block-idx-x tile) grid-dim-x) p)
                      thread-idx-x)))
          (set (aref shared-pos thread-idx-x) (aref positions idx)))
        (syncthreads)
        (set acc (gravitation body-pos acc shared-pos))
        (syncthreads))
      (return acc))))

(defkernel integrate-bodies (void ((new-pos float4*) (old-pos float4*) (vel float4*)
                                   (delta-time float) (damping float)
                                   (total-num-bodies int)))
  (let ((index (+ (* block-idx-x block-dim-x) thread-idx-x)))
    (if (>= index total-num-bodies)
        (return))
    (let ((position (aref old-pos index))
          (accel (compute-body-accel position old-pos total-num-bodies))
          (velocity (aref vel index)))
      (set (float4-x velocity) (+ (float4-x velocity)
                                  (* (float3-x accel) delta-time)))
      (set (float4-y velocity) (+ (float4-y velocity)
                                  (* (float3-y accel) delta-time)))
      (set (float4-z velocity) (+ (float4-z velocity)
                                  (* (float3-z accel) delta-time)))
      
      (set (float4-x velocity) (* (float4-x velocity) damping))
      (set (float4-y velocity) (* (float4-y velocity) damping))
      (set (float4-z velocity) (* (float4-z velocity) damping))
      
      (set (float4-x position) (+ (float4-x position)
                                  (* (float4-x velocity) delta-time)))
      (set (float4-y position) (+ (float4-y position)
                                  (* (float4-y velocity) delta-time)))
      (set (float4-z position) (+ (float4-z position)
                                  (* (float4-z velocity) delta-time)))
      
      (set (aref new-pos index) position)
      (set (aref vel index) velocity))))


;;;
;;; cl-glut window subclass
;;;

(defclass nbody-window (glut:window)
  ((new-pos :initform nil)
   (old-pos :initform nil)
   (vel :initform nil)
   (delta-time :initform 0)
   (damping :initform 0)
   (num-bodies :initform 0)
   (p :initform 0))
  (:default-initargs :width 640 :height 480 :pos-x 100 :pos-y 100
                     :mode '(:double :rgb) :title "nbody"))


;;;
;;; cl-glut event handlers
;;;

(defmethod glut:display-window :before ((w nbody-window))
  (gl:clear-color 0 0 0 0))

(defmethod glut:display ((w nbody-window))
  (with-slots (new-pos old-pos vel delta-time damping num-bodies p) w
    ;; update simulation
    (nbody-demo-update-simulation new-pos old-pos vel delta-time damping num-bodies p)
    (rotatef new-pos old-pos)
    ;; clear buffers
    (gl:clear :color-buffer :depth-buffer-bit)
    ;; view transform
    (gl:matrix-mode :modelview)
    (gl:load-identity)
    (gl:translate 0.0 0.0 -100.0)
    (gl:rotate 0.0 1.0 0.0 0.0)
    (gl:rotate 0.0 0.0 1.0 0.0)
    ;; display bodies
    (nbody-demo-display new-pos num-bodies)
    ;; swap buffers
    (glut:swap-buffers)
    ;; display frame rate
    (display-frame-rate num-bodies)))

(defmethod glut:reshape ((w nbody-window) width height)
  ;; configure on projection mode
  (gl:matrix-mode :projection)
  (gl:load-identity)
  (glu:perspective 60.0 (/ width height) 0.1 1000.0)
  ;; configure on model-view mode
  (gl:matrix-mode :modelview)
  (gl:viewport 0 0 width height))

(defmethod glut:idle ((w nbody-window))
  (glut:post-redisplay))


;;;
;;; Performance measurement functions
;;;

(defvar *flops-per-interaction* 20)

(defun compute-perf-stats (num-bodies elapsed-time fps-count iterations)
  (let ((milliseconds (/ elapsed-time fps-count)))
    (let* ((ifps (/ 1.0 (/ milliseconds 1000.0)))
           (interactions-per-second (/ (* num-bodies num-bodies iterations)
                                       (/ milliseconds 1000)
                                       1.0e9))
           (gflops (* interactions-per-second *flops-per-interaction*)))
      (values ifps interactions-per-second gflops))))

(let ((fps-count 0)
      (fps-limit 5)
      (template "CUDA N-Body (~A bodies): ~,1F fps | ~,1F BIPS | ~,1F GFLOP/s | single precision~%"))
  (defun display-frame-rate (num-bodies)
    (incf fps-count)
    (when (>= fps-count fps-limit)
      (multiple-value-bind (ifps interactions-per-second gflops)
          (compute-perf-stats num-bodies (get-elapsed-time) fps-count 1)
        (glut:set-window-title
          (format nil template num-bodies ifps interactions-per-second gflops))
        (setf fps-count 0)
        (setf fps-limit (max ifps 1.0))))))


;;;
;;; Timing functions
;;;

(let (start-event stop-event)

  (defun create-timer-events ()
    (setf start-event (cffi:foreign-alloc 'cu-event)
          stop-event (cffi:foreign-alloc 'cu-event))
    (cu-event-create start-event cu-event-default)
    (cu-event-create stop-event cu-event-default))
  
  (defun destroy-timer-events ()
    (cu-event-destroy (cffi:mem-ref start-event 'cu-event))
    (cu-event-destroy (cffi:mem-ref stop-event 'cu-event))
    (cffi:foreign-free start-event)
    (cffi:foreign-free stop-event))

  (defun start-timer ()
    (cu-event-record (cffi:mem-ref start-event 'cu-event)
                     (cffi:null-pointer)))

  (defun stop-and-synchronize-timer ()
    (cu-event-record (cffi:mem-ref stop-event 'cu-event)
                     (cffi:null-pointer))
    (cu-event-synchronize (cffi:mem-ref stop-event 'cu-event)))

  (defun get-elapsed-time ()
    (let (milliseconds)
      (cffi:with-foreign-object (pmilliseconds :float)
        (stop-and-synchronize-timer)
        (cu-event-elapsed-time pmilliseconds
                               (cffi:mem-ref start-event 'cu-event)
                               (cffi:mem-ref stop-event 'cu-event))
        (start-timer)
        (setf milliseconds (cffi:mem-ref pmilliseconds :float)))
      milliseconds)))

(defmacro with-cuda-timer (&body body)
  `(unwind-protect
        (progn
          (create-timer-events)
          ,@body)
     (destroy-timer-events)))


;;;
;;; NBodyDemo
;;;

(defun nbody-demo-update-simulation (new-pos old-pos vel delta-time damping num-bodies p)
  (integrate-nbody-system new-pos old-pos vel delta-time damping num-bodies p))

(defun nbody-demo-display (pos num-bodies)
  ; (particle-renderer-set-spirte-size ...)
  (memcpy-device-to-host pos)
  (particle-renderer-set-positions pos num-bodies)
  (particle-renderer-display))


;;;
;;; Particle Renderer
;;;

(let ((m-pos           nil)
      (m-num-particles 0  )
      (m-point-size    1.0)
      ;(m-sprite-size   2.0)
      ;(m-vertex-shader nil)
      ;(m-pixel-shader  nil)
      ;(m-program       nil)
      ;(m-texture       nil)
      ;(m-pbo           nil)
      ;(m-vbo-color     nil)
      ;(m-base-color    nil))
      )
  
  (defun particle-renderer-set-positions (pos num-particles)
    (setf m-pos pos)
    (setf m-num-particles num-particles))
  
  (defun particle-renderer-set-base-color (color)
    (declare (ignore color))
    (not-implemented))
  
  (defun particle-renderer-set-colors (color num-particles)
    (declare (ignore color num-particles))
    (not-implemented))
  
  (defun particle-renderer-set-pbo (pbo num-particles)
    (declare (ignore pbo num-particles))
    (not-implemented))
  
  (defun particle-renderer-display ()
    ; mode = PARTICLE_POINTS
    (gl:color 1.0 1.0 1.0)
    (gl:point-size m-point-size)
    (draw-points))
  
  (defun particle-renderer-set-point-size (size)
    (declare (ignore size))
    (not-implemented))
  
  (defun particle-renderer-set-sprite-size (size)
    (declare (ignore size))
    (not-implemented))
  
  (defun particle-renderer-reset-pbo ()
    (not-implemented))
  
  (defun init-gl ()
    (not-implemented))
  
  (defun create-texture (resolution)
    (declare (ignore resolution))
    (not-implemented))
  
  (defun draw-points ()
    (gl:begin :points)
    (dotimes (i m-num-particles)
      (let ((p (mem-aref m-pos i)))
        (gl:vertex (float4-x p) (float4-y p) (float4-z p))))
    (gl:end)))


;;;
;;; main
;;;

(defun divided-point (inner outer k)
  (+ inner (* (- outer inner) k)))

(defun norm-float3 (x)
  (assert (float3-p x))
  (sqrt (+ (* (float3-x x) (float3-x x))
           (* (float3-y x) (float3-y x))
           (* (float3-z x) (float3-z x)))))

(defun normalize-float3 (x)
  (assert (float3-p x))
  (let ((r (norm-float3 x)))
    (if (< 1.0e-6 r)
        (make-float3 (/ (float3-x x) r)
                     (/ (float3-y x) r)
                     (/ (float3-z x) r))
        x)))

(defun cross (v0 v1)
  (assert (and (float3-p v0) (float3-p v1)))
  (make-float3 (- (* (float3-y v0) (float3-z v1))
                  (* (float3-z v0) (float3-y v1)))
               (- (* (float3-z v0) (float3-x v1))
                  (* (float3-x v0) (float3-z v1)))
               (- (* (float3-x v0) (float3-y v1))
                  (* (float3-y v0) (float3-x v1)))))

(defun randomize-bodies (pos vel cluster-scale velocity-scale num-bodies)
  (let* ((scale cluster-scale)
         (vscale (* scale velocity-scale))
         (inner (* 2.5 scale))
         (outer (* 4.0 scale)))
    (dotimes (i num-bodies)
      (let ((point (normalize-float3 (make-float3 (- (random 1.0) 0.5)
                                                  (- (random 1.0) 0.5)
                                                  (- (random 1.0) 0.5))))
            (k (divided-point inner outer (random 1.0))))
        (setf (mem-aref pos i) (make-float4 (* (float3-x point) k)
                                            (* (float3-y point) k)
                                            (* (float3-z point) k)
                                            1.0)))
      (let* ((axis (make-float3 0.0 0.0 1.0))
             (vv (cross (make-float3 (float4-x (mem-aref pos i))
                                     (float4-y (mem-aref pos i))
                                     (float4-z (mem-aref pos i)))
                        axis)))
        (setf (mem-aref vel i) (make-float4 (* (float3-x vv) vscale)
                                            (* (float3-y vv) vscale)
                                            (* (float3-z vv) vscale)
                                            1.0))))))

(defun reset (pos vel cluster-scale velocity-scale num-bodies)
  (randomize-bodies pos vel cluster-scale velocity-scale num-bodies))

(defun integrate-nbody-system (new-pos old-pos vel delta-time damping num-bodies p)
  (let ((grid-dim (list (ceiling (/ num-bodies p)) 1 1))
        (block-dim (list p 1 1)))
    (integrate-bodies new-pos old-pos vel delta-time damping num-bodies
                      :grid-dim grid-dim
                      :block-dim block-dim)))

(defun main ()
  (let ((dev-id 0)
        (num-bodies 2048)
        (cluster-scale 1.56)
        (velocity-scale 2.64)
        (delta-time 0.016)
        (damping 1.0)
        (p 256)
        (window (make-instance 'nbody-window)))
    (with-cuda-context (dev-id)
      (with-memory-blocks ((new-pos 'float4 num-bodies)
                           (old-pos 'float4 num-bodies)
                           (vel 'float4 num-bodies))
      (with-cuda-timer
        ;; set values to slots of the nbody window
        (setf (slot-value window 'new-pos) new-pos
              (slot-value window 'old-pos) old-pos
              (slot-value window 'vel) vel
              (slot-value window 'delta-time) delta-time
              (slot-value window 'damping) damping
              (slot-value window 'num-bodies) num-bodies
              (slot-value window 'p) p)
        ;; reset position and velocity of bodies
        (reset old-pos vel cluster-scale velocity-scale num-bodies)
        ;; copy position and velocity on host memory to device memory
        (memcpy-host-to-device old-pos vel)
        ;; start CUDA timer
        (start-timer)
        ;; start glut main loop
        (glut:display-window window))))))

(defun not-implemented ()
  (error "not implemented."))
