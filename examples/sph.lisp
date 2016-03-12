#|
  This file is a part of cl-cuda project.
  Copyright (c) 2013 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-examples.sph
  (:use :cl
        :cl-cuda)
  (:import-from :alexandria
                :with-gensyms
                :once-only)
  (:export :main))
(in-package :cl-cuda-examples.sph)


;;
;; Utilities

(defkernel norm (float ((x float4)))
  (return (sqrt (+ (* (float4-x x) (float4-x x))
                   (* (float4-y x) (float4-y x))
                   (* (float4-z x) (float4-z x))
                   (* (float4-w x) (float4-w x))))))

(defkernelmacro do-range ((var from to) &body body)
  `(do ((,var ,from (+ ,var 1)))
       ((> ,var ,to))
     ,@body))

(defkernelmacro and (&rest args)
  (case (length args)
    (0 t)
    (1 (car args))
    (t `(if ,(car args) (and ,@(cdr args)) nil))))

(defkernelmacro inc (place val)
  `(set ,place (+ ,place ,val)))

(defkernelmacro pow (x n)
  (check-type n fixnum)
  `(* ,@(loop repeat n collect x)))

;; (defkernel pow (float ((b float) (p float)))
;;   (return (expt b p)))

(defkernelmacro with-particle-index ((var) &body body)
  `(let ((,var (+ (* block-idx-x block-dim-x) thread-idx-x)))
     ,@body))


;;
;; Parameters

(defkernel-symbol-macro h 0.005)
(defkernel-symbol-macro dt 0.0004)
(defkernel-symbol-macro pi 3.1415927)
(defkernel-symbol-macro visc 0.2)
(defkernel-symbol-macro limit 200.0)
(defkernel-symbol-macro pmass (/ 0.00020543 8.0))
(defkernel-symbol-macro radius 0.002)
(defkernel-symbol-macro epsilon 0.00001)
(defkernel-symbol-macro extdamp 512.0)
(defkernel-symbol-macro simscale 0.004)
(defkernel-symbol-macro intstiff 3.0)
(defkernel-symbol-macro extstiff 20000.0)
(defkernel-symbol-macro restdensity 600.0)
(defkernel-symbol-macro g (float4 0.0 -9.8 0.0 0.0))

(defglobal box-min (float4 0.0 0.0 0.0 0.0) :constant)
(defglobal box-max (float4 0.0 0.0 0.0 0.0) :constant)
(defglobal origin (float4 0.0 0.0 0.0 0.0) :constant)
(defglobal delta 0.0 :constant)
(defglobal capacity 0 :constant)
(defglobal size-x 0 :constant)
(defglobal size-y 0 :constant)
(defglobal size-z 0 :constant)

(defparameter h           0.005)
(defparameter pmass       (/ 0.00020543 8.0))
(defparameter simscale    0.004)
(defparameter restdensity 600.0)
(defparameter pdist       (expt (/ pmass restdensity) (/ 1.0 3.0)))
(defparameter g           (make-float4 0.0 -9.8 0.0 0.0))
(defparameter delta       (/ h simscale))
(defparameter box-min     (make-float4 -10.0  0.0 -10.0 0.0))
(defparameter box-max     (make-float4  30.0 50.0  30.0 0.0))
(defparameter init-min    (make-float4 -10.0  0.0 -10.0 0.0))
(defparameter init-max    (make-float4   0.0 40.0  30.0 0.0))
(defparameter capacity    400)  ; # of particles contained in one cell


;;
;; Neighbor map

(defkernelmacro with-cell-index (((i j k) x) &body body)
  (once-only (x)
    `(let ((,i (floor (/ (- (float4-x ,x) (float4-x origin)) delta)))
           (,j (floor (/ (- (float4-y ,x) (float4-y origin)) delta)))
           (,k (floor (/ (- (float4-z ,x) (float4-z origin)) delta))))
       ,@body)))

(defkernel offset (int ((i int) (j int) (k int) (l int)))
  (return (+ (* capacity size-x size-y k)
             (* capacity size-x j)
             (* capacity i)
             l)))

(defkernel update-neighbor-map (void ((neighbor-map int*)
                                      (pos float4*)
                                      (n int)))
  (with-particle-index (p)
    (when (< p n)
      (with-cell-index ((i j k) (aref pos p))
        (let ((offset (offset i j k 0)))
          ;; Atomically increment the number of particles in the cell.
          (let ((l (atomic-add (pointer (aref neighbor-map offset)) 1)))
            ;; Set particle in the cell.
            (set (aref neighbor-map (offset i j k (+ l 1))) p)))))))

(defkernel clear-neighbor-map (void ((neighbor-map int*)))
  (let ((i thread-idx-x)
        (j block-idx-x)
        (k block-idx-y))
    (set (aref neighbor-map (offset i j k 0)) 0)))

(defkernelmacro do-neighbors ((var neighbor-map x) &body body)
  (with-gensyms (i0 j0 k0 i j k l)
    `(with-cell-index ((,i0 ,j0 ,k0) ,x)
       (do-range (,i (- ,i0 1) (+ ,i0 1))
         (do-range (,j (- ,j0 1) (+ ,j0 1))
           (do-range (,k (- ,k0 1) (+ ,k0 1))
             (do-range (,l 1 (aref ,neighbor-map (offset ,i ,j ,k 0)))
               (let ((,var (aref ,neighbor-map (offset ,i ,j ,k ,l))))
                 ,@body))))))))

(defun compute-origin (box-min delta)
  (let ((delta2 (* delta 2)))
    (make-float4 (- (float4-x box-min) delta2)
                 (- (float4-y box-min) delta2)
                 (- (float4-z box-min) delta2)
                 0.0)))

(defun compute-size (box-min box-max delta capacity)
  (assert (and (< (float4-x box-min) (float4-x box-max))
               (< (float4-y box-min) (float4-y box-max))
               (< (float4-z box-min) (float4-z box-max))))
  (assert (< 0.0 delta))
  (assert (< 0 capacity))
  (flet ((compute-size1 (x0 x1)
           (+ (ceiling (/ (- x1 x0) delta))
              4)))
    (let* ((size-x (compute-size1 (float4-x box-min) (float4-x box-max)))
           (size-y (compute-size1 (float4-y box-min) (float4-y box-max)))
           (size-z (compute-size1 (float4-z box-min) (float4-z box-max)))
           (size (* size-x
                    size-y
                    size-z
                    (1+ capacity))))
      (values size-x size-y size-z size))))


;;
;; Boundary condition

;; returns dummy integer to avoid __host__ qualifier
(defkernel apply-collision (int ((acc float4*)
                                 (i int)
                                 (x0 float)
                                 (x1 float)
                                 (v float4)
                                 (normal float4)))
  (let* ((distance (* (- x1 x0) simscale))
         (diff (- (* radius 2.0) distance))
         (adj (- (* extstiff diff)
                 (* extdamp (dot normal v)))))
    (when (< epsilon diff)
      (inc (aref acc i) (* adj normal))))
  (return 0))

;; returns dummy integer to avoid __host__ qualifier
(defkernel apply-accel-limit (int ((acc float4*) (i int)))
  (let ((accel (norm (aref acc i))))
    (when (< limit accel)
      (set (aref acc i) (* (aref acc i) (/ limit accel)))))
  (return 0))

(defkernel boundary-condition (void ((acc float4*)
                                     (pos float4*)
                                     (vel float4*)
                                     (n int)))
  (with-particle-index (i)
    (when (< i n)
      (let ((xi (aref pos i))
            (vi (aref vel i)))
        ;; Left boundary.
        (apply-collision acc i (float4-x box-min) (float4-x xi) vi
                         (float4 1.0 0.0 0.0 0.0))
        ;; Right boundary.
        (apply-collision acc i (float4-x xi) (float4-x box-max) vi
                         (float4 -1.0 0.0 0.0 0.0))
        ;; Bottom boundary.
        (apply-collision acc i (float4-y box-min) (float4-y xi) vi
                         (float4 0.0 1.0 0.0 0.0))
        ;; Top boundary.
        (apply-collision acc i (float4-y xi) (float4-y box-max) vi
                         (float4 0.0 -1.0 0.0 0.0))
        ;; Near side boundary.
        (apply-collision acc i (float4-z box-min) (float4-z xi) vi
                         (float4 0.0 0.0 1.0 0.0))
        ;; Far side boundary.
        (apply-collision acc i (float4-z xi) (float4-z box-max) vi
                         (float4 0.0 0.0 -1.0 0.0))
        ;; Accel limit.
        (apply-accel-limit acc i)))))


;;
;; SPH kernel functions

(defkernel poly6-kernel (float ((x float4)))
  (let ((r (norm x)))
    (return (* (/ 315.0 (* 64.0 pi (pow h 9)))
               (pow (- (* h h) (* r r)) 3)))))

(defkernel grad-spiky-kernel (float4 ((x float4)))
  (let ((r (norm x)))
    (return (* (/ -45.0 (* pi (pow h 6)))
               (pow (- h r) 2)
               (/ x r)))))

(defkernel rap-visc-kernel (float ((x float4)))
  (let ((r (norm x)))
    (return (* (/ 45.0 (* pi (pow h 6)))
               (- h r)))))


;;
;; Update density

(defkernel update-density (void ((rho float*)
                                 (pos float4*)
                                 (n int)
                                 (neighbor-map int*)))
  (with-particle-index (i)
    (when (< i n)
      (let ((xi (aref pos i))
            (tmp 0.0))
        (do-neighbors (j neighbor-map xi)
          (let* ((xj (aref pos j))
                 (dr (* (- xi xj) simscale)))
            (when (<= (norm dr) h)
              (inc tmp (* pmass (poly6-kernel dr))))))
        (set (aref rho i) tmp)))))


;;
;; Update pressure

(defkernel update-pressure (void ((prs float*)
                                  (rho float*)
                                  (n int)))
  (with-particle-index (i)
    (when (< i n)
      (set (aref prs i) (* (- (aref rho i) restdensity)
                           intstiff)))))


;;
;; Update force

(defkernel pressure-term (float4 ((rho float*)
                                  (prs float*)
                                  (i int)
                                  (j int)
                                  (dr float4)))
  (return (* (/ (* (- pmass) (+ (aref prs i) (aref prs j)))
                (* 2.0 (aref rho j)))
             (grad-spiky-kernel dr))))

(defkernel viscosity-term (float4 ((vel float4*)
                                   (rho float*)
                                   (i int)
                                   (j int)
                                   (dr float4)))
  (return (* (/ (* visc pmass (- (aref vel j) (aref vel i)))
                (aref rho j))
             (rap-visc-kernel dr))))

(defkernel update-force (void ((force float4*)
                               (pos float4*)
                               (vel float4*)
                               (rho float*)
                               (prs float*)
                               (n int)
                               (neighbor-map int*)))
  (with-particle-index (i)
    (when (< i n)
      (let ((xi (aref pos i))
            (tmp (float4 0.0 0.0 0.0 0.0)))
        (do-neighbors (j neighbor-map xi)
          (when (/= i j)
            (let* ((xj (aref pos j))
                   (dr (* (- xi xj) simscale)))
              (when (<= (norm dr) h)
                (inc tmp (pressure-term rho prs i j dr))
                (inc tmp (viscosity-term vel rho i j dr))))))
        (set (aref force i) tmp)))))


;;
;; Update acceleration

(defkernel update-acceleration (void ((acc float4*)
                                      (force float4*)
                                      (rho float*)
                                      (n int)))
  (with-particle-index (i)
    (when (< i n)
      (set (aref acc i) (+ (/ (aref force i)
                              (aref rho i))
                           g)))))


;;
;; Update velocity

(defkernel update-velocity (void ((vel float4*)
                                  (acc float4*)
                                  (n int)))

  (with-particle-index (i)
    (when (< i n)
      (inc (aref vel i) (* (aref acc i) dt)))))


;;
;; Update position

(defkernel update-position (void ((pos float4*)
                                  (vel float4*)
                                  (n int)))
  (with-particle-index (i)
    (when (< i n)
      (inc (aref pos i) (/ (* (aref vel i) dt)
                           simscale)))))


;;
;; Output functions

(defparameter +filename-template+ "result~8,'0d.pov")

(defparameter +header-template+ "#include \"colors.inc\"

camera {
  location <10, 30, -40>
  look_at <10, 10, 0>
}
light_source { <0, 30, -30> color White }

")

(defparameter +sphere-template+ "sphere {
  <~F,~F,~F>,0.25
  texture {
    pigment { color Yellow }
  }
}

")

(defun filename (i)
  (format nil +filename-template+ i))

(defun output-header (stream)
  (format stream +header-template+))

(defun output-sphere (pos stream)
  (with-float4 (x y z w) pos
    (format stream +sphere-template+ x y z)))

(defun output (step pos)
  (format t "Output step ~A...~%" step)
  (let ((n (memory-block-size pos))
        (filename (filename step)))
    (with-open-file (out filename :direction :output :if-exists :supersede)
      (output-header out)
      (loop for i from 0 below n
         do (output-sphere (memory-block-aref pos i) out)))))


;;
;; Main

(defun initial-condition (init-min init-max d)
  (with-float4 (x0 y0 z0 w0) init-min
    (with-float4 (x1 y1 z1 w1) init-max
      (let (result)
        (loop for x from (+ x0 d) below x1 by d
           do (loop for y from (+ y0 d) below y1 by d
                 do (loop for z from (+ z0 d) below z1 by d
                       do (push (make-float4 x y z 0.0) result))))
        result))))

(defun initialize (pos vel particles)
  (loop for p in particles
        for i from 0
     do (setf (memory-block-aref pos i) p)
        (setf (memory-block-aref vel i) (make-float4 0.0 0.0 0.0 0.0))))

(defun peek-memory-block (memory-block)
  (sync-memory-block memory-block :device-to-host)
  (loop repeat 10
        for i from 0
     do (print (memory-block-aref memory-block i))))

(defun main ()
  (let* (;; Grid and block dims.
         (neighbor-map-grid-dim '(45 37 1))
         (neighbor-map-block-dim '(37 1 1))
         (particle-grid-dim '(512 1 1))
         (particle-block-dim '(64 1 1))
         ;; Get initial condition.
         (particles (initial-condition init-min init-max (/ pdist simscale)))
         ;; Get number of particles.
         (n (length particles))
         ;; Compute neighbor map origin.
         (origin (compute-origin box-min delta)))
    ;; Compute neighbor map size.
    (multiple-value-bind (size-x size-y size-z size)
        (compute-size box-min box-max delta capacity)
      (with-cuda (0)
        ;; Set boundary condition globals.
        (setf (global-ref 'box-min 'float4) box-min)
        (setf (global-ref 'box-max 'float4) box-max)
        ;; Set neighbor map globals.
        (setf (global-ref 'origin 'float4) origin)
        (setf (global-ref 'delta 'float) delta)
        (setf (global-ref 'capacity 'int) capacity)
        (setf (global-ref 'size-x 'int) size-x)
        (setf (global-ref 'size-y 'int) size-y)
        (setf (global-ref 'size-z 'int) size-z)
        ;; With memory blocks.
        (with-memory-blocks ((pos 'float4 n)
                             (vel 'float4 n)
                             (acc 'float4 n)
                             (force 'float4 n)
                             (rho 'float n)
                             (prs 'float n)
                             (neighbor-map 'int size))
          ;; Print number of particles.
          (format t "~A particles~%" n)
          ;; Apply initial condition.
          (initialize pos vel particles)
          (sync-memory-block pos :host-to-device)
          (sync-memory-block vel :host-to-device)
          ;(output 0 pos)
          ;; Do simulation.
          (time
           (loop repeat 300
                 for i from 1
              do ;; Clear neighbor map.
                 (clear-neighbor-map neighbor-map
                                     :grid-dim neighbor-map-grid-dim
                                     :block-dim neighbor-map-block-dim)
                 ;; Update neighbor map.
                 (update-neighbor-map neighbor-map pos n
                                      :grid-dim particle-grid-dim
                                      :block-dim particle-block-dim)
                 ;; Update density.
                 (update-density rho pos n neighbor-map
                                 :grid-dim particle-grid-dim
                                 :block-dim particle-block-dim)
                 ;; Update pressure.
                 (update-pressure prs rho n
                                  :grid-dim particle-grid-dim
                                  :block-dim particle-block-dim)
                 ;; Update force.
                 (update-force force pos vel rho prs n neighbor-map
                               :grid-dim particle-grid-dim
                               :block-dim particle-block-dim)
                 ;; Update acceleration.
                 (update-acceleration acc force rho n
                                      :grid-dim particle-grid-dim
                                      :block-dim particle-block-dim)
                 ;; Apply boundary condition.
                 (boundary-condition acc pos vel n
                                     :grid-dim particle-grid-dim
                                     :block-dim particle-block-dim)
                 ;; Update velocity.
                 (update-velocity vel acc n
                                  :grid-dim particle-grid-dim
                                  :block-dim particle-block-dim)
                 ;; Update position.
                 (update-position pos vel n
                                  :grid-dim particle-grid-dim
                                  :block-dim particle-block-dim)
                 ;; Synchronize CUDA context.
                 (synchronize-context)
                 ;; Output POV file.
                 ;(when (= (mod i 10) 0)
                 ;  (sync-memory-block pos :device-to-host)
                 ;  (output (/ i 10) pos))
                 )))))))
