#|
  This file is a part of cl-cuda project.
  Copyright (c) 2013 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-examples.sph2
  (:use :cl
        :cl-cuda)
  (:export :main))
(in-package :cl-cuda-examples.sph2)


;;;
;;; Utilities
;;;

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

(defkernel norm (float ((x float4)))
  (return (sqrt (+ (* (float4-x x) (float4-x x))
                   (* (float4-y x) (float4-y x))
                   (* (float4-z x) (float4-z x))
                   (* (float4-w x) (float4-w x))))))

(defkernelmacro pow (x n)
  (check-type n fixnum)
  `(* ,@(loop repeat n collect x)))

(defun float4-zero ()
  (make-float4 0.0 0.0 0.0 0.0))

(defmacro with-float4-cpu ((x y z w) val &body body)
  `(let ((,x (float4-x ,val))
         (,y (float4-y ,val))
         (,z (float4-z ,val))
         (,w (float4-w ,val)))
     ,@(loop for var in (list x y z w)
          when (string-equal var "_")
          collect `(declare (ignorable ,var)))
     ,@body))


;;;
;;; Neighbor map cell
;;;

(defkernel cell-number-of-particles (int ((offset int) (nbr int*)))
  (return (aref nbr offset)))

(defkernel cell-nth-particle (int ((n int) (offset int) (nbr int*)))
  (return (aref nbr (+ offset n 1))))   ; increment need because n begins with 0

;; returns dummy integer to avoid __host__ qualifier
(defkernel insert-cell (int ((p int) (offset int) (nbr int*)))
  (let ((n (atomic-add (pointer (aref nbr offset)) 1)))
    (set (aref nbr (+ offset n 1)) p))
  (return 0))

;; returns dummy integer to avoid __host__ qualifier
(defkernel clear-cell (int ((offset int) (nbr int*)))
  (set (aref nbr offset) 0)   ; particles in cell are not cleared for performance reason
  (return 0))


;;;
;;; Neibhro map - in cell
;;;

(defkernel cell-offset (int ((nbr int*) (info float*) (i int) (j int) (k int)))
  (let ((size-x (info-size-x info))
        (size-y (info-size-y info))
        (capacity (info-capacity info)))
    (let ((offset-base (+ (* size-x size-y k)
                          (* size-x j)
                          i))
          (capacity1 (+ capacity 1)))
      (return (* offset-base capacity1)))))

(defkernel valid-cell-index (bool ((info float*) (i int) (j int) (k int)))
  (return (and (<= 0 i) (< i (info-size-x info))
               (<= 0 j) (< j (info-size-y info))
               (<= 0 k) (< k (info-size-z info))
               t)))

(defkernel number-of-particles-in-cell (int ((nbr int*) (info float*) (i int) (j int) (k int)))
  (unless (valid-cell-index info i j k)
    (return 0))
  (let ((offset (cell-offset nbr info i j k)))
    (return (cell-number-of-particles offset nbr))))

(defkernel nth-particle-in-cell (int ((n int) (nbr int*) (info float*) (i int) (j int) (k int)))
  (unless (valid-cell-index info i j k)
    (return 0))
  (let ((offset (cell-offset nbr info i j k)))
    (return (cell-nth-particle n offset nbr))))

(defkernelmacro do-particles-in-cell ((p nbr info i j k) &body body)
  (alexandria:with-gensyms (n index)
    `(when (valid-cell-index ,info ,i ,j ,k)
       (let ((,n (number-of-particles-in-cell ,nbr ,info ,i ,j ,k)))
         (do-range (,index 0 (- ,n 1))
           (let ((,p (nth-particle-in-cell ,index ,nbr ,info ,i ,j ,k)))
             ,@body))))))

;; returns dummy integer to avoid __host__ qualifier
(defkernel insert-particle-in-cell (int ((p int) (nbr int*) (info float*) (i int) (j int) (k int)))
  (unless (valid-cell-index info i j k)
    (return 0))
  (let ((offset (cell-offset nbr info i j k)))
    (insert-cell p offset nbr))
  (return 0))


;; returns dummy integer to avoid __host__ qualifier
(defkernel clear-particles-in-cell (int ((nbr int*) (info float*) (i int) (j int) (k int)))
  (unless (valid-cell-index info i j k)
    (return 0))
  (let ((offset (cell-offset nbr info i j k)))
    (clear-cell offset nbr))
  (return 0))


;;;
;;; Neighbor map
;;;

(defkernel %pos-to-cell (int ((x float) (x0 float) (delta float)))
  (return (floor (/ (- x x0) delta))))

(defkernel pos-to-cell-x (int ((pos float4) (info float*)))
  (let ((x (float4-x pos))
        (x0 (info-origin-x info))
        (delta (info-delta info)))
    (return (%pos-to-cell x x0 delta))))

(defkernel pos-to-cell-y (int ((pos float4) (info float*)))
  (let ((y (float4-y pos))
        (y0 (info-origin-y info))
        (delta (info-delta info)))
    (return (%pos-to-cell y y0 delta))))

(defkernel pos-to-cell-z (int ((pos float4) (info float*)))
  (let ((z (float4-z pos))
        (z0 (info-origin-z info))
        (delta (info-delta info)))
    (return (%pos-to-cell z z0 delta))))

(defkernelmacro with-cell ((i j k info x) &body body)
  `(let ((,i (pos-to-cell-x ,x ,info))
         (,j (pos-to-cell-y ,x ,info))
         (,k (pos-to-cell-z ,x ,info)))
     ,@body))

(defkernelmacro do-neighbor-particles ((p nbr info x) &body body)
  (alexandria:with-gensyms (i0 j0 k0 i j k shared-info)
    `(with-shared-memory ((,shared-info float 15))
       ;; set neighbor map info into shared memory
       (do-range (,i 0 14)
         (set (aref ,shared-info ,i) (aref ,info ,i)))
       ;; do body forms using neighbor map info contained in shared memory
       (with-cell (,i0 ,j0 ,k0 ,shared-info ,x)
         (do-range (,i (- ,i0 1) (+ ,i0 1))
           (do-range (,j (- ,j0 1) (+ ,j0 1))
             (do-range (,k (- ,k0 1) (+ ,k0 1))
               (do-particles-in-cell (,p ,nbr ,shared-info ,i ,j ,k)
                 ,@body))))))))

;; returns dummy integer to avoid __host__ qualifier
(defkernel insert-particle-in-neighbor-map (int ((p int) (x float4) (nbr int*) (info float*)))
  (with-cell (i j k info x)
    (insert-particle-in-cell p nbr info i j k))
  (return 0))

(defkernel clear-neighbor-map (void ((nbr int*) (info float*)))
  (let ((i thread-idx-x)
        (j block-idx-x)
        (k block-idx-y))
    (clear-particles-in-cell nbr info i j k)))

(defun alloc-neighbor-map (box-min box-max delta capacity)
  (assert (and (< (float4-x box-min) (float4-x box-max))
               (< (float4-y box-min) (float4-y box-max))
               (< (float4-z box-min) (float4-z box-max))))
  (assert (< 0.0 delta))
  (assert (< 0 capacity))
  (labels ((%size (x0 x1)
             (float (ceiling (/ (- x1 x0) delta)))))
    (with-float4-cpu (box-min-x box-min-y box-min-z _) box-min
    (with-float4-cpu (box-max-x box-max-y box-max-z _) box-max
      (let ((size-x (%size box-min-x box-max-x))
            (size-y (%size box-min-y box-max-y))
            (size-z (%size box-min-z box-max-z))
            (origin-x (- box-min-x delta))
            (origin-y (- box-min-y delta))
            (origin-z (- box-min-z delta)))
        (let* ((size-x-2 (+ size-x 2))
               (size-y-2 (+ size-y 2))
               (size-z-2 (+ size-z 2))
               (size-2 (* size-x-2 size-y-2 size-z-2)))
          (let (nbr info)
            ;; alloc and initialize neighbor map info
            (setf info (alloc-memory-block 'float 15))
            (setf (memory-block-aref info 0)  box-min-x
                  (memory-block-aref info 1)  box-min-y
                  (memory-block-aref info 2)  box-min-z
                  (memory-block-aref info 3)  (+ box-min-x (* delta size-x))
                  (memory-block-aref info 4)  (+ box-min-y (* delta size-y))
                  (memory-block-aref info 5)  (+ box-min-z (* delta size-z))
                  (memory-block-aref info 6)  origin-x
                  (memory-block-aref info 7)  origin-y
                  (memory-block-aref info 8)  origin-z
                  (memory-block-aref info 9)  delta
                  (memory-block-aref info 10) (float capacity)
                  (memory-block-aref info 11) size-x-2
                  (memory-block-aref info 12) size-y-2
                  (memory-block-aref info 13) size-z-2
                  (memory-block-aref info 14) size-2)
            ;; alloc neighbor map
            (setf nbr (alloc-memory-block 'int (* (floor size-2)
                                                  (1+ capacity))))
            ;; return them
            (values nbr info))))))))

(defun free-neighbor-map (nbr info)
  (free-memory-block nbr)
  (free-memory-block info))

(defkernel info-min-x    (float ((info float*))) (return (aref info 0)))
(defkernel info-min-y    (float ((info float*))) (return (aref info 1)))
(defkernel info-min-z    (float ((info float*))) (return (aref info 2)))
(defkernel info-max-x    (float ((info float*))) (return (aref info 3)))
(defkernel info-max-y    (float ((info float*))) (return (aref info 4)))
(defkernel info-max-z    (float ((info float*))) (return (aref info 5)))
(defkernel info-origin-x (float ((info float*))) (return (aref info 6)))
(defkernel info-origin-y (float ((info float*))) (return (aref info 7)))
(defkernel info-origin-z (float ((info float*))) (return (aref info 8)))
(defkernel info-delta    (float ((info float*))) (return (aref info 9)))
(defkernel info-capacity (int ((info float*)))   (return (floor (aref info 10))))
(defkernel info-size-x   (int ((info float*)))   (return (floor (aref info 11))))
(defkernel info-size-y   (int ((info float*)))   (return (floor (aref info 12))))
(defkernel info-size-z   (int ((info float*)))   (return (floor (aref info 13))))
(defkernel info-size     (int ((info float*)))   (return (floor (aref info 14))))

(defun sync-neighbor-map (nbr info direction)
  (sync-memory-block nbr direction)
  (sync-memory-block info direction))

(defmacro with-neighbor-map ((nbr info box-min box-max delta capacity) &body body)
  `(multiple-value-bind (,nbr ,info)
       (alloc-neighbor-map ,box-min ,box-max ,delta ,capacity)
     (unwind-protect (progn ,@body)
       (free-neighbor-map ,nbr ,info))))


;;;
;;; Constants
;;;

(defkernel-symbol-macro h 0.01)
(defkernel-symbol-macro dt 0.004)
(defkernel-symbol-macro pi 3.1415927)
(defkernel-symbol-macro visc 0.2)
(defkernel-symbol-macro limit 200.0)
(defkernel-symbol-macro pmass 0.00020543)
(defkernel-symbol-macro radius 0.004)
(defkernel-symbol-macro epsilon 0.00001)
(defkernel-symbol-macro extdamp 256.0)
(defkernel-symbol-macro simscale 0.004)
(defkernel-symbol-macro intstiff 3.0)
(defkernel-symbol-macro extstiff 10000.0)
(defkernel-symbol-macro restdensity 600.0)
(defkernel-symbol-macro pdist (expt (/ pmass restdensity) (/ 1.0 3.0)))
(defkernel-symbol-macro g (float4 0.0 -9.8 0.0 0.0))

(defparameter h           0.01)
(defparameter simscale    0.004)
(defparameter pmass       0.00020543)
(defparameter restdensity 600.0)
(defparameter pdist       (expt (/ pmass restdensity) (/ 1.0 3.0)))

(defparameter delta       (/ h simscale))
(defparameter box-min     (make-float4  0.0  0.0 -10.0 0.0))
(defparameter box-max     (make-float4 20.0 50.0  10.0 0.0))
(defparameter init-min    (make-float4  0.0  0.0 -10.0 0.0))
(defparameter init-max    (make-float4 10.0 20.0  10.0 0.0))
(defparameter capacity    20)          ; # of particles contained in one


;;;
;;; Update neighbor map
;;;

(defkernelmacro with-valid-index ((i n) &body body)
  `(let ((,i (+ (* block-idx-x block-dim-x) thread-idx-x)))
     (when (< ,i ,n)
       ,@body)))

;; returns dummy integer to avoid __host__ qualifier
(defkernel %update-neighbor-map (int ((i int) (pos float4*) (nbr int*) (info float*)))
  (let ((x (aref pos i)))
    (insert-particle-in-neighbor-map i x nbr info))
  (return 0))

(defkernel update-neighbor-map (void ((pos float4*) (nbr int*) (info float*) (n int)))
  (with-valid-index (i n)
    (%update-neighbor-map i pos nbr info)))


;;;
;;; Kernel functions
;;;

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


;;;
;;; Update density
;;;

;; returns dummy integer to avoid __host__ qualifier
(defkernel %update-density (int ((i int) (rho float*) (x float4*) (nbr int*) (info float*)))
  (let ((xi (aref x i)))
  (symbol-macrolet (;(xi (aref x i))
                    (xj (aref x j))
                    (rhoi (aref rho i)))
    (let ((tmp 0.0))
      (do-neighbor-particles (j nbr info xi)
        (let ((dr (* (- xi xj) simscale)))
          (when (<= (norm dr) h)
            (inc tmp (* pmass (poly6-kernel dr))))))
      (set rhoi tmp))))
  (return 0))

(defkernel update-density (void ((rho float*) (x float4*) (nbr int*) (info float*) (n int)))
  (with-valid-index (i n)
    (%update-density i rho x nbr info)))


;;;
;;; Update pressure
;;;

;; returns dummy integer to avoid __host__ qualifier
(defkernel %update-pressure (int ((i int) (prs float*) (rho float*)))
  (symbol-macrolet ((rhoi (aref rho i))
                    (prsi (aref prs i)))
    (set prsi (* (- rhoi restdensity)
                 intstiff)))
  (return 0))

(defkernel update-pressure (void ((prs float*) (rho float*) (n int)))
  (with-valid-index (i n)
    (%update-pressure i prs rho)))


;;;
;;; Update force
;;;

(defkernel pressure-term (float4 ((i int) (j int) (dr float4) (rho float*) (prs float*)))
  (symbol-macrolet ((rhoj (aref rho j))
                    (prsi (aref prs i))
                    (prsj (aref prs j)))
    (return (* (/ (* (- pmass) (+ prsi prsj))
                  (* 2.0 rhoj))
               (grad-spiky-kernel dr)))))

(defkernel viscosity-term (float4 ((i int) (j int) (dr float4) (v float4*) (rho float*)))
  (symbol-macrolet ((vi (aref v i))
                    (vj (aref v j))
                    (rhoj (aref rho j)))
    (return (* (/ (* visc pmass (- vj vi))
                  rhoj)
               (rap-visc-kernel dr)))))

;; returns dummy integer to avoid __host__ qualifier
(defkernel %update-force (int ((i int) (f float4*) (x float4*) (v float4*)
                               (rho float*) (prs float*) (nbr int*) (info float*)))
  (let ((xi (aref x i)))
  (symbol-macrolet (;(xi (aref x i))
                    (xj (aref x j))
                    (fi (aref f i)))
    (let ((tmp (float4 0.0 0.0 0.0 0.0)))
      (do-neighbor-particles (j nbr info xi)
        (when (/= i j)
          (let ((dr (* (- xi xj) simscale)))
            (when (<= (norm dr) h)
              (inc tmp (pressure-term  i j dr rho prs))
              (inc tmp (viscosity-term i j dr v rho))))))
      (set fi tmp))))
  (return 0))

(defkernel update-force (void ((f float4*) (x float4*) (v float4*)
                               (rho float*) (prs float*) (nbr int*) (info float*) (n int)))
  (with-valid-index (i n)
    (%update-force i f x v rho prs nbr info)))


;;;
;;; Boundary condition
;;;

(defkernel collision-diff (float ((x0 float) (x1 float)))
  (let ((distance (* (- x1 x0) simscale)))
    (return (- (* radius 2.0) distance))))

(defkernel collision-adj (float4 ((diff float) (v float4) (normal float4)))
  (let ((adj (- (* extstiff diff)
                (* extdamp (dot normal v)))))
    (return (* adj normal))))

;; returns dummy integer to avoid __host__ qualifier
(defkernel apply-collision (int ((a float4*) (i int) (x0 float) (x1 float) (v float4) (normal float4)))
  (symbol-macrolet ((ai (aref a i)))
    (let ((diff (collision-diff x0 x1)))
      (when (< epsilon diff)
        (inc ai (collision-adj diff v normal)))))
  (return 0))

(defkernel accel-limit-adj (float ((accel float)))
  (return (/ limit accel)))

;; returns dummy integer to avoid __host__ qualifier
(defkernel apply-accel-limit (int ((a float4*) (i int)))
  (symbol-macrolet ((ai (aref a i)))
    (let ((accel (norm ai)))
      (when (< limit accel)
        (set ai (* ai (accel-limit-adj accel))))))
  (return 0))

;; returns dummy integer to avoid __host__ qualifier
(defkernel %boundary-condition (int ((i int) (x float4*) (v float4*) (a float4*) (info float*)))
  (symbol-macrolet ((xi (aref x i))
                    (vi (aref v i)))
    ;; left boundary
    (apply-collision a i (info-min-x info) (float4-x xi) vi (float4 1.0 0.0 0.0 0.0))
    ;; right boundary
    (apply-collision a i (float4-x xi) (info-max-x info) vi (float4 -1.0 0.0 0.0 0.0))
    ;; bottom boundary
    (apply-collision a i (info-min-y info) (float4-y xi) vi (float4 0.0 1.0 0.0 0.0))
    ;; top boundary
    (apply-collision a i (float4-y xi) (info-max-y info) vi (float4 0.0 -1.0 0.0 0.0))
    ;; near-side boundary
    (apply-collision a i (info-min-z info) (float4-z xi) vi (float4 0.0 0.0 1.0 0.0))
    ;; far-side boundary
    (apply-collision a i (float4-z xi) (info-max-z info) vi (float4 0.0 0.0 -1.0 0.0))
    ;; accel limit
    (apply-accel-limit a i))
  (return 0))

(defkernel boundary-condition (void ((x float4*) (v float4*) (a float4*) (info float*) (n int)))
  (with-valid-index (i n)
    (%boundary-condition i x v a info)))


;;;
;;; Update acceleration
;;;

;; returns dummy integer to avoid __host__ qualifier
(defkernel %update-acceleration (int ((i int) (a float4*) (f float4*) (rho float*)))
  (symbol-macrolet ((ai   (aref a i))
                    (fi   (aref f i))
                    (rhoi (aref rho i)))
    (set ai (+ (/ fi rhoi) g)))
  (return 0))

(defkernel update-acceleration (void ((a float4*) (f float4*) (rho float*) (n int)))
  (with-valid-index (i n)
    (%update-acceleration i a f rho)))


;;;
;;; Update velocity
;;;

;; returns dummy integer to avoid __host__ qualifier
(defkernel %update-velocity (int ((i int) (v float4*) (a float4*)))
  (symbol-macrolet ((vi (aref v i))
                    (ai (aref a i)))
    (inc vi (* ai dt)))
  (return 0))

(defkernel update-velocity (void ((v float4*) (a float4*) (n int)))
  (with-valid-index (i n)
    (%update-velocity i v a)))


;;;
;;; Update position
;;;

;; returns dummy integer to avoid __host__ qualifier
(defkernel %update-position (int ((i int) (x float4*) (v float4*)))
  (symbol-macrolet ((xi (aref x i))
                    (vi (aref v i)))
    (inc xi (/ (* vi dt)
              simscale)))
  (return 0))

(defkernel update-position (void ((x float4*) (v float4*) (n int)))
  (with-valid-index (i n)
    (%update-position i x v)))


;;;
;;; Initial condition
;;;

(defun initial-condition ()
  (with-float4-cpu (init-min-x init-min-y init-min-z _) init-min
  (with-float4-cpu (init-max-x init-max-y init-max-z _) init-max
  (let ((d (* (/ pdist simscale) 0.95)))
    (let (result)
      (loop for x from (+ init-min-x d) to (- init-max-x d) by d do
        (loop for y from (+ init-min-y d) to (- init-max-y d) by d do
          (loop for z from (+ init-min-z d) to (- init-max-z d) by d do
            (push (make-float4 x y z 0.0) result))))
      result)))))


;;;
;;; Output functions
;;;

(defparameter *file-name-template* "result~8,'0d.pov")

(defparameter *header* (concatenate 'string
                                    "#include \"colors.inc\"~%"
                                    "camera {~%"
                                    "  location <10, 30, -40>~%"
                                    "  look_at <10, 10, 0>~%"
                                    "}~%"
                                    "light_source { <0, 30, -30> color White }~%"))

(defparameter *sphere-template* (concatenate 'string
                                             "sphere {~%"
                                             "  <~F,~F,~F>,0.5~%"
                                             "  texture {~%"
                                             "    pigment { color Yellow }~%"
                                             "  }~%"
                                             "}~%"))

(defun file-name (i)
  (format nil *file-name-template* i))

(defun sphere (pos i)
  (with-float4-cpu (x y z _) (memory-block-aref pos i)
    (format nil *sphere-template* x y z)))

(defun output (pos i)
  (let ((n (memory-block-size pos))
        (fname (file-name i)))
    (with-open-file (out fname :direction :output :if-exists :supersede)
      (princ *header* out)
      (dotimes (i n)
        (princ (sphere pos i) out)))))


;;;
;;; Main
;;;

(defun initialize (x v particles)
  (let ((zero (float4-zero)))
    (loop for p in particles
          for i from 0
       do (setf (memory-block-aref x i) p
                (memory-block-aref v i) zero))))

(defun run-sph (particles)
  (let ((dev-id 0)
        (n (length particles))
        (grid-dim '(16 1 1))
        (block-dim '(64 1 1)))
    ;; with CUDA context
    (with-cuda (dev-id)
      ;; with memory blocks
      (with-memory-blocks ((x   'float4 n)
                           (v   'float4 n)
                           (a   'float4 n)
                           (f   'float4 n)
                           (rho 'float  n)
                           (prs 'float  n))
      ;; with neighbor map
      (with-neighbor-map (nbr info box-min box-max delta capacity)
        ;; print # of particles
        (format t "~A particles~%" n)
        ;; apply given initial condition
        (initialize x v particles)
        ;; copy initial position and velocity to device memory
        (sync-memory-block x :host-to-device)
        (sync-memory-block v :host-to-device)
        ;; copy neighbor map to device memory
        (sync-neighbor-map nbr info :host-to-device)
        ;; loop
        (time
         (dotimes (_ 300)
           ;; clear neighbor map
           (clear-neighbor-map nbr info :grid-dim '(23 11 1) :block-dim '(11 1 1))
           ;; update neighbor map with position
           (update-neighbor-map x nbr info n :grid-dim grid-dim :block-dim '(128 1 1))
           ;; update density
           (update-density rho x nbr info n :grid-dim grid-dim :block-dim '(128 1 1))
           ;; update pressure
           (update-pressure prs rho n :grid-dim grid-dim :block-dim block-dim)
           ;; update force
           (update-force f x v rho prs nbr info n :grid-dim grid-dim :block-dim block-dim)
           ;; update acceleration
           (update-acceleration a f rho n :grid-dim grid-dim :block-dim block-dim)
           ;; apply boundary condition
           (boundary-condition x v a info n :grid-dim grid-dim :block-dim block-dim)
           ;; update velocity
           (update-velocity v a n :grid-dim grid-dim :block-dim block-dim)
           ;; update position
           (update-position x v n :grid-dim grid-dim :block-dim block-dim)
           ;; synchronize CUDA context
           (synchronize-context))))))))

(defun main ()
  (run-sph (initial-condition)))
