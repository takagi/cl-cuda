#|
  This file is a part of cl-cuda project.
  Copyright (c) 2013 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-examples.sph-cpu
  (:use :cl)
  (:import-from :alexandria
                :with-gensyms
                :once-only)
  (:export :main))
(in-package :cl-cuda-examples.sph-cpu)


;;
;; Vec3

(deftype vec3 ()
  '(simple-array single-float (3)))

(defmacro vec3-x (vec3)
  `(aref ,vec3 0))

(defmacro vec3-y (vec3)
  `(aref ,vec3 1))

(defmacro vec3-z (vec3)
  `(aref ,vec3 2))

(defun make-vec3 (x y z)
  (make-array 3 :element-type 'single-float
                :initial-contents (list x y z)))

(defmacro with-vec3 ((x y z) value &body body)
  (once-only (value)
    `(let ((,x (vec3-x ,value))
           (,y (vec3-y ,value))
           (,z (vec3-z ,value)))
       ,@body)))


;;
;; Vec3 array

(deftype vec3-array ()
  '(simple-array single-float (* 3)))

(defun make-vec3-array (n)
  (make-array (list n 3) :element-type 'single-float
                         :initial-element 0.0))

(defmacro with-vec3-aref ((x y z) (array index) &body body)
  (once-only (array index)
    `(let ((,x (aref ,array ,index 0))
           (,y (aref ,array ,index 1))
           (,z (aref ,array ,index 2)))
       ,@body)))

(defmacro set-vec3-aref (array i form)
  (once-only (array i)
    `(multiple-value-bind (x y z) ,form
       (setf (aref ,array ,i 0) x
             (aref ,array ,i 1) y
             (aref ,array ,i 2) z))))

(defmacro inc-vec3-aref (array i form)
  (once-only (i)
    `(with-vec3-aref (x0 y0 z0) (,array ,i)
       (multiple-value-bind (x y z) ,form
         (set-vec3-aref ,array ,i (values (the single-float (+ x0 x))
                                          (the single-float (+ y0 y))
                                          (the single-float (+ z0 z))))))))


;;
;; Scalar array

(deftype scalar-array ()
  '(simple-array single-float *))

(defun make-scalar-array (n)
  (make-array n :element-type 'single-float
              :initial-element 0.0))


;;
;; Utilities

(declaim (ftype (function (single-float
                           single-float
                           single-float)
                          single-float)
                norm))
(declaim (inline norm))
(defun norm (x y z)
  (declare (optimize (speed 3) (safety 0)))
  (sqrt
   (+ (* x x) (* y y) (* z z))))

(defmacro do-range ((var from to) &body body)
  `(do ((,var ,from (+ ,var 1)))
       ((> ,var ,to))
     ,@body))

(defmacro pow (x n)
  (check-type n fixnum)
  `(* ,@(loop repeat n collect x)))


;;
;; Parameters

(define-symbol-macro h 0.005) 
(define-symbol-macro dt 0.0004)
(define-symbol-macro visc 0.2)
(define-symbol-macro limit 200.0)
(define-symbol-macro pmass (/ 0.00020543 8.0))
(define-symbol-macro radius 0.002)
(define-symbol-macro epsilon 0.00001)
(define-symbol-macro extdamp 512.0)
(define-symbol-macro simscale 0.004)
(define-symbol-macro intstiff 3.0)
(define-symbol-macro extstiff 20000.0)
(define-symbol-macro restdensity 600.0)
(define-symbol-macro pdist (expt (/ pmass restdensity) (/ 1.0 3.0)))
(define-symbol-macro g (make-vec3 0.0 -9.8 0.0))

(defvar *box-min* (make-vec3 -10.0  0.0 -10.0))
(defvar *box-max* (make-vec3  30.0 50.0  30.0))
(defvar *init-min* (make-vec3 -10.0  0.0 -10.0))
(defvar *init-max* (make-vec3   0.0 40.0  30.0))
(defvar *delta* (/ h simscale))
(defvar *capacity* 400)        ; # of particles contained in one cell.

#+nil
(progn
  (define-symbol-macro h 0.01)
  (define-symbol-macro dt 0.004)
  (define-symbol-macro visc 0.2)
  (define-symbol-macro limit 200.0)
  (define-symbol-macro pmass 0.00020543)
  (define-symbol-macro radius 0.004)
  (define-symbol-macro epsilon 0.00001)
  (define-symbol-macro extdamp 256.0)
  (define-symbol-macro simscale 0.004)
  (define-symbol-macro intstiff 3.0)
  (define-symbol-macro extstiff 10000.0)
  (define-symbol-macro restdensity 600.0)
  (define-symbol-macro pdist (expt (/ pmass restdensity) (/ 1.0 3.0)))
  (define-symbol-macro g (make-vec3 0.0 -9.8 0.0))

  (defvar *box-min* (make-vec3 0.0 0.0 -10.0))
  (defvar *box-max* (make-vec3 20.0 50.0 10.0))
  (defvar *init-min* (make-vec3 0.0 0.0 -10.0))
  (defvar *init-max* (make-vec3 10.0 20.0 10.0))
  (defvar *delta* (/ h simscale))
  (defvar *capacity* 400))     ; # of particles contained in one cell.


;;
;; Neighbor map

(defstruct (neighbor-map (:constructor %make-neighbor-map))
  (data :data :type (simple-array fixnum *))
  (origin :origin :type vec3 :read-only t)
  (delta :delta :type single-float :read-only t)
  (capacity :capacity :type fixnum :read-only t)
  (size-x :size-x :type fixnum :read-only t)
  (size-y :size-y :type fixnum :read-only t)
  (size-z :size-z :type fixnum :read-only t))

(defun compute-origin (box-min delta)
  (let ((delta2 (* delta 2)))
    (make-vec3 (- (vec3-x box-min) delta2)
               (- (vec3-y box-min) delta2)
               (- (vec3-z box-min) delta2))))

(defun compute-size (box-min box-max delta capacity)
  (assert (and (< (vec3-x box-min) (vec3-x box-max))
               (< (vec3-y box-min) (vec3-y box-max))
               (< (vec3-z box-min) (vec3-z box-max))))
  (assert (< 0.0 delta))
  (assert (< 0 capacity))
  (flet ((compute-size1 (x0 x1)
           (+ (ceiling (/ (- x1 x0) delta))
              4)))
    (let* ((size-x (compute-size1 (vec3-x box-min) (vec3-x box-max)))
           (size-y (compute-size1 (vec3-y box-min) (vec3-y box-max)))
           (size-z (compute-size1 (vec3-z box-min) (vec3-z box-max)))
           (size (* size-x
                    size-y
                    size-z
                    (1+ capacity))))
      (values size-x size-y size-z size))))

(defun make-neighbor-map (box-min box-max delta capacity)
  ;; Compute neighbor map origin.
  (let ((origin (compute-origin box-min delta)))
    ;; Compute neighbor map size.
    (multiple-value-bind (size-x size-y size-z size)
        (compute-size box-min box-max delta capacity)
      ;; Make neighbor map.
      (let ((data (make-array size :element-type 'fixnum :initial-element 0)))
        (%make-neighbor-map :data data
                            :origin origin
                            :delta delta
                            :capacity capacity
                            :size-x size-x
                            :size-y size-y
                            :size-z size-z)))))

(defmacro with-cell-index (((i j k) pos p neighbor-map) &body body)
  (once-only (p neighbor-map)
    (with-gensyms (origin delta)
      `(let ((,origin (neighbor-map-origin ,neighbor-map))
             (,delta (neighbor-map-delta ,neighbor-map)))
         (let ((,i (floor (the (single-float -1.0s10 1.0s10)
                               (/ (- (aref ,pos ,p 0) (vec3-x ,origin))
                                  ,delta))))
               (,j (floor (the (single-float -1.0s10 1.0s10)
                               (/ (- (aref ,pos ,p 1) (vec3-y ,origin))
                                  ,delta))))
               (,k (floor (the (single-float -1.0s10 1.0s10)
                               (/ (- (aref ,pos ,p 2) (vec3-z ,origin))
                                  ,delta)))))
           ,@body)))))

(declaim (ftype (function (neighbor-map
                           fixnum
                           fixnum
                           fixnum
                           fixnum)
                          fixnum)
                offset))
(declaim (inline offset))
(defun offset (neighbor-map i j k l)
  (declare (optimize (speed 3) (safety 0)))
  (let ((capacity (neighbor-map-capacity neighbor-map))
        (size-x (neighbor-map-size-x neighbor-map))
        (size-y (neighbor-map-size-y neighbor-map)))
    (declare (type fixnum capacity size-x size-y))
    (the fixnum (+ (the fixnum (* capacity (the fixnum (* size-x (the fixnum (* size-y k))))))
                   (the fixnum (+ (the fixnum (* capacity (the fixnum (* size-x j))))
                                  (the fixnum (+ (the fixnum (* capacity i))
                                                 l))))))))

(defun update-neighbor-map (neighbor-map pos n)
  (loop for p from 0 below n
     do (with-cell-index ((i j k) pos p neighbor-map)
          (let ((data (neighbor-map-data neighbor-map))
                (offset (offset neighbor-map i j k 0)))
            ;; Atomically increment the number of particles in the cell.
            (let ((l (incf (aref data offset))))
              ;; Set particle in the cell.
              (setf (aref data (offset neighbor-map i j k l)) p))))))

(defun clear-neighbor-map (neighbor-map)
  (let ((size-x (neighbor-map-size-x neighbor-map))
        (size-y (neighbor-map-size-y neighbor-map))
        (size-z (neighbor-map-size-z neighbor-map)))
    (loop for i from 0 below size-x
       do (loop for j from 0 below size-y
             do (loop for k from 0 below size-z
                   do (let ((data (neighbor-map-data neighbor-map))
                            (offset (offset neighbor-map i j k 0)))
                        (setf (aref data offset) 0)))))))

(defmacro do-neighbors ((var neighbor-map pos p) &body body)
  (with-gensyms (i0 j0 k0 i j k l data offset)
    `(with-cell-index ((,i0 ,j0 ,k0) ,pos ,p ,neighbor-map)
       (do-range (,i (- ,i0 1) (+ ,i0 1))
         (do-range (,j (- ,j0 1) (+ ,j0 1))
           (do-range (,k (- ,k0 1) (+ ,k0 1))
             (let ((,data (neighbor-map-data ,neighbor-map))
                   (,offset (offset ,neighbor-map ,i ,j ,k 0)))
               (do-range (,l 1 (aref ,data ,offset))
                 (let ((,var (aref ,data (+ ,offset ,l))))
                   ,@body)))))))))


;;
;; Boundary condition

(defun dot (x0 y0 z0 x1 y1 z1)
  (+ (* x0 x1) (* y0 y1) (* z0 z1)))

(defun apply-collision (acc i x0 x1 u v w nx ny nz)
  (let* ((distance (* (- x1 x0) simscale))
         (diff (- (* radius 2.0) distance))
         (adj (- (* extstiff diff)
                 (* extdamp (dot nx ny nz u v w)))))
    (when (< epsilon diff)
      (inc-vec3-aref acc i (values (* adj nx)
                                   (* adj ny)
                                   (* adj nz))))))

(defun apply-accel-limit (acc i)
  (with-vec3-aref (ax ay az) (acc i)
    (let ((accel (norm ax ay az)))
      (when (< limit accel)
        (set-vec3-aref acc i (values (* ax (/ limit accel))
                                     (* ay (/ limit accel))
                                     (* az (/ limit accel))))))))

(defun boundary-condition (acc pos vel n box-min box-max)
  (loop for i from 0 below n
     do (with-vec3-aref (x y z) (pos i)
          (with-vec3-aref (u v w) (vel i)
            ;; Left boundary.
            (apply-collision acc i (vec3-x box-min) x u v w 1.0 0.0 0.0)
            ;; Right boundary.
            (apply-collision acc i x (vec3-x box-max) u v w -1.0 0.0 0.0)
            ;; Bottom boundary.
            (apply-collision acc i (vec3-y box-min) y u v w 0.0 1.0 0.0)
            ;; Top boundary.
            (apply-collision acc i y (vec3-y box-max) u v w 0.0 -1.0 0.0)
            ;; Near side boundary.
            (apply-collision acc i (vec3-z box-min) z u v w 0.0 0.0 1.0)
            ;; Far side boundary.
            (apply-collision acc i z (vec3-z box-max) u v w 0.0 0.0 -1.0)
            ;; Accel limit.
            (apply-accel-limit acc i)))))


;;
;; SPH kernel functions

(declaim (ftype (function (single-float
                           single-float
                           single-float)
                          single-float)
                poly6-kernel))
(defun poly6-kernel (dx dy dz)
  (declare (optimize (speed 3) (safety 0)))
  (let ((r (norm dx dy dz)))
    (* (/ 315.0 (* 64.0 (float pi 0.0) (pow h 9)))
       (pow (- (* h h) (* r r)) 3))))

(declaim (ftype (function (single-float
                           single-float
                           single-float)
                          (values single-float single-float single-float))
                grad-spiky-kernel))
(defun grad-spiky-kernel (dx dy dz)
  (declare (optimize (speed 3) (safety 0)))
  (let* ((r (norm dx dy dz))
         (coeff (* (/ -45.0 (* (float pi 0.0) (pow h 6)))
                   (pow (- h r) 2)
                   (/ 1.0 r))))
    (values (* coeff dx) (* coeff dy) (* coeff dz))))

(declaim (ftype (function (single-float
                           single-float
                           single-float)
                          single-float)
                rap-visc-kernel))
(defun rap-visc-kernel (dx dy dz)
  (declare (optimize (speed 3) (safety 0)))
  (let ((r (norm dx dy dz)))
    (* (/ 45.0 (* (float pi 0.0) (pow h 6)))
       (- h r))))


;;
;; Update density

(declaim (ftype (function (scalar-array
                           vec3-array
                           fixnum
                           neighbor-map))
                update-density))
(defun update-density (rho pos n neighbor-map)
  (declare (optimize (speed 3) (safety 0)))
  (loop for i from 0 below n
     do (setf (aref rho i) 0.0)
       (do-neighbors (j neighbor-map pos i)
         (with-vec3-aref (xi yi zi) (pos i)
           (with-vec3-aref (xj yj zj) (pos j)
             (let* ((dx (* (- xi xj) simscale))
                    (dy (* (- yi yj) simscale))
                    (dz (* (- zi zj) simscale))
                    (dr (norm dx dy dz)))
               (when (<= dr h)
                 (incf (aref rho i) (* pmass (poly6-kernel dx dy dz))))))))))


;;
;; Update pressure

(defun update-pressure (prs rho n)
  (loop for i from 0 below n
     do (setf (aref prs i) (* (- (aref rho i) restdensity)
                              intstiff))))


;;
;; Update force

(declaim (ftype (function (scalar-array
                           scalar-array
                           fixnum
                           fixnum
                           single-float
                           single-float
                           single-float)
                          (values single-float single-float single-float))
                pressure-term))
(defun pressure-term (rho prs i j dx dy dz)
  (declare (optimize (speed 3) (safety 0)))
  (multiple-value-bind (x y z) (grad-spiky-kernel dx dy dz)
    (let ((coeff (/ (* (- pmass) (+ (aref prs i) (aref prs j)))
                    (* 2.0 (aref rho j)))))
      (values (* coeff x) (* coeff y) (* coeff z)))))

(declaim (ftype (function (vec3-array
                           scalar-array
                           fixnum
                           fixnum
                           single-float
                           single-float
                           single-float)
                          (values single-float single-float single-float))
                viscosity-term))
(defun viscosity-term (vel rho i j dx dy dz)
  (declare (optimize (speed 3) (safety 0)))
  (with-vec3-aref (ui vi wi) (vel i)
    (with-vec3-aref (uj vj wj) (vel j)
      (let ((coeff (* (/ (* visc pmass)
                         (aref rho j))
                      (rap-visc-kernel dx dy dz))))
        (values (* coeff (- uj ui))
                (* coeff (- vj vi))
                (* coeff (- wj wi)))))))

(declaim (ftype (function (vec3-array
                           vec3-array
                           vec3-array
                           scalar-array
                           scalar-array
                           fixnum
                           neighbor-map))
                update-force))
(defun update-force (force pos vel rho prs n neighbor-map)
  (declare (optimize (speed 3) (safety 0)))
  (loop for i from 0 below n
     do (set-vec3-aref force i (values 0.0 0.0 0.0))
       (do-neighbors (j neighbor-map pos i)
         (when (/= i j)
           (with-vec3-aref (xi yi zi) (pos i)
             (with-vec3-aref (xj yj zj) (pos j)
               (let* ((dx (* (- xi xj) simscale))
                      (dy (* (- yi yj) simscale))
                      (dz (* (- zi zj) simscale))
                      (dr (norm dx dy dz)))
                 (when (<= dr h)
                   (inc-vec3-aref force i
                                  (pressure-term rho prs i j dx dy dz))
                   (inc-vec3-aref force i
                                  (viscosity-term vel rho i j dx dy dz))))))))))


;;
;; Update acceleration

(defun update-acceleration (acc force rho n)
  (loop for i from 0 below n
     do (with-vec3-aref (fx fy fz) (force i)
          (with-vec3 (gx gy gz) g
            (set-vec3-aref acc i (values (+ (/ fx (aref rho i)) gx)
                                         (+ (/ fy (aref rho i)) gy)
                                         (+ (/ fz (aref rho i)) gz)))))))


;;
;; Update velocity

(defun update-velocity (vel acc n)
  (loop for i from 0 below n
     do (with-vec3-aref (ax ay az) (acc i)
          (inc-vec3-aref vel i (values (* ax dt)
                                       (* ay dt)
                                       (* az dt))))))


;;
;; Update position

(defun update-position (pos vel n)
  (loop for i from 0 below n
     do (with-vec3-aref (u v w) (vel i)
          (inc-vec3-aref pos i (values (/ (* u dt) simscale)
                                       (/ (* v dt) simscale)
                                       (/ (* w dt) simscale))))))


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

(defun output-sphere (pos i stream)
  (format stream +sphere-template+ (aref pos i 0)
                                   (aref pos i 1)
                                   (aref pos i 2)))

(defun output (step pos n)
  (format t "Output step ~A...~%" step)
  (let ((filename (filename step)))
    (with-open-file (out filename :direction :output :if-exists :supersede)
      (output-header out)
      (loop for i from 0 below n
         do (output-sphere pos i out)))))


;;
;; Main

(defun initial-condition (init-min init-max d)
  (with-vec3 (x0 y0 z0) init-min
    (with-vec3 (x1 y1 z1) init-max
      (let (result)
        (loop for x from (+ x0 d) below x1 by d
           do (loop for y from (+ y0 d) below y1 by d
                 do (loop for z from (+ z0 d) below z1 by d
                       do (push (make-vec3 x y z) result))))
        result))))

(defun initialize (pos vel particles)
  (loop for p in particles
        for i from 0
     do (set-vec3-aref pos i (values (vec3-x p) (vec3-y p) (vec3-z p)))
        (set-vec3-aref vel i (values 0.0 0.0 0.0))))

(defun main ()
  (let* (;; Get initial condition.
         (particles (initial-condition *init-min* *init-max*
                                       (/ pdist simscale)))
         ;; Get number of particles.
         (n (length particles))
         ;; Make neighbor map.
         (neighbor-map (make-neighbor-map *box-min* *box-max*
                                          *delta* *capacity*)))
    ;; With arrays.
    (let ((pos (make-vec3-array n))
          (vel (make-vec3-array n))
          (acc (make-vec3-array n))
          (force (make-vec3-array n))
          (rho (make-scalar-array n))
          (prs (make-scalar-array n)))
      ;; Print number of particles.
      (format t "~A particles~%" n)
      ;; Apply initial condition.
      (initialize pos vel particles)
      ;(output 0 pos n)
      ;; Do simulation.
      (time
       (loop repeat 300
             for i from 1
          do ;; Clear neighbor map.
             (clear-neighbor-map neighbor-map)
             ;; Update neighbor map.
             (update-neighbor-map neighbor-map pos n)
             ;; Update density.
             (update-density rho pos n neighbor-map)
             ;; Update pressure.
             (update-pressure prs rho n)
             ;; Update force.
             (update-force force pos vel rho prs n neighbor-map)
             ;; Update acceleration.
             (update-acceleration acc force rho n)
             ;; Apply boundary condition.
             (boundary-condition acc pos vel n *box-min* *box-max*)
             ;; Update velocity.
             (update-velocity vel acc n)
             ;; Update position.
             (update-position pos vel n)
             ;; Output POV file.
             ;(when (= (mod i 10) 0)
             ;  (output (/ i 10) pos n))
             )))))
