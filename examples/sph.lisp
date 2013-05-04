#|
  This file is a part of cl-cuda project.
  Copyright (c) 2013 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-examples.sph
  (:use :cl
        :cl-cuda)
  (:export :main))
(in-package :cl-cuda-examples.sph)


;;
;; kernel macro utilities
;;

(defkernelmacro get-index ()
  `(+ (* block-idx-x block-dim-x) thread-idx-x))

(defkernelmacro inc (place val)
  `(set ,place (+ ,place ,val)))

(defkernelmacro do-range ((var from to) &body body)
  `(do ((,var ,from (+ ,var 1)))
       ((> ,var ,to))
     ,@body))

(defkernelmacro when (test &body forms)
  `(if ,test
       (progn ,@forms)))

(defkernelmacro unless (test &body forms)
  `(when (not ,test)
     ,@forms))


;;
;; NEIGHBOR-MAP-INFO
;;   Metadata about neighbor map
;;
;; info ::= [ box-min.x, box-min.y, box-min.z
;;          , box-max.x, box-max.y, box-max.z
;;          , delta    , cell-size, size-x
;;          , size-y   , size-z   , raw-size ]
;;

(defun %size (selector box-min box-max delta)
  (let ((max (funcall selector box-max))
        (min (funcall selector box-min)))
    (1+ (float (floor (/ (- max min) delta))))))
(defun size-x (box-min box-max delta) (%size #'float3-x box-min box-max delta))
(defun size-y (box-min box-max delta) (%size #'float3-y box-min box-max delta))
(defun size-z (box-min box-max delta) (%size #'float3-z box-min box-max delta))

(defun raw-size (box-min box-max delta cell-size)
  (* (size-x box-min box-max delta)
     (size-y box-min box-max delta)
     (size-z box-min box-max delta)
     cell-size))

(defun alloc-neighbor-map-info (box-min box-max delta cell-size)
  (let ((info (alloc-memory-block 'float 12)))
    (setf (mem-aref info  0) (float3-x box-min)
          (mem-aref info  1) (float3-y box-min)
          (mem-aref info  2) (float3-z box-min)
          (mem-aref info  3) (float3-x box-max)
          (mem-aref info  4) (float3-y box-max)
          (mem-aref info  5) (float3-z box-max)
          (mem-aref info  6) delta
          (mem-aref info  7) cell-size
          (mem-aref info  8) (size-x box-min box-max delta)
          (mem-aref info  9) (size-y box-min box-max delta)
          (mem-aref info 10) (size-z box-min box-max delta)
          (mem-aref info 11) (raw-size box-min box-max delta cell-size))
    info))

(defun free-neighbor-map-info (info)
  (free-memory-block info))

(defmacro with-neighbor-map-info ((var box-min box-max delta cell-size) &body body)
  `(let ((,var (alloc-neighbor-map-info ,box-min ,box-max ,delta ,cell-size)))
     (unwind-protect
          (progn ,@body)
       (free-neighbor-map-info ,var))))

;;
;; DEF-INFO-SELECTOR
;;
;; Expanded to the definitions of NEIGHBOR-MAP-INFO selectors on cpu and gpu.
;;
;; (def-info-selector info-box-min-x info-box-min-x-cpu 0 float)
;; =>
;; (progn
;;   (defkernel info-box-min-x (float ((info float*)))
;;     (return (aref info 0)))
;;   (defun info-box-min-x-cpu (info)
;;     (mem-aref info 0)))
;;

(defmacro def-info-selector-int (selector selector-cpu n)
  (alexandria:with-gensyms (info)
    `(progn
       (defkernel ,selector (int ((,info float*)))
         (return (floor (aref ,info ,n))))
       (defun ,selector-cpu (,info)
         (floor (mem-aref ,info ,n))))))

(defmacro def-info-selector-float (selector selector-cpu n)
  (alexandria:with-gensyms (info)
    `(progn
       (defkernel ,selector (float ((,info float*)))
         (return (aref ,info ,n)))
       (defun ,selector-cpu (,info)
         (mem-aref ,info ,n)))))

(defmacro def-info-selector (selector selector-cpu n return-type)
  (case return-type
    (:int   `(def-info-selector-int   ,selector ,selector-cpu ,n))
    (:float `(def-info-selector-float ,selector ,selector-cpu ,n))
    (t (error "invalid return type: ~A" return-type))))

(def-info-selector info-box-min-x info-box-min-x-cpu 0  :float)
(def-info-selector info-box-min-y info-box-min-y-cpu 1  :float)
(def-info-selector info-box-min-z info-box-min-z-cpu 2  :float)
(def-info-selector info-box-max-x info-box-max-x-cpu 3  :float)
(def-info-selector info-box-max-y info-box-max-y-cpu 4  :float)
(def-info-selector info-box-max-z info-box-max-z-cpu 5  :float)
(def-info-selector info-delta     info-delta-cpu     6  :float)
(def-info-selector info-cell-size info-cell-size-cpu 7  :int)
(def-info-selector info-size-x    info-size-x-cpu    8  :int)
(def-info-selector info-size-y    info-size-y-cpu    9  :int)
(def-info-selector info-size-z    info-size-z-cpu    10 :int)
(def-info-selector info-raw-size  info-raw-size-cpu  11 :int)

(defun output-neighbor-map-info (info)
  (format t "box-min.x: ~A~%" (mem-aref info 0))
  (format t "box-min.y: ~A~%" (mem-aref info 1))
  (format t "box-min.z: ~A~%" (mem-aref info 2))
  (format t "box-max.x: ~A~%" (mem-aref info 3))
  (format t "box-max.y: ~A~%" (mem-aref info 4))
  (format t "box-max.z: ~A~%" (mem-aref info 5))
  (format t "delta    : ~A~%" (mem-aref info 6))
  (format t "cell-size: ~A~%" (mem-aref info 7))
  (format t "size-x   : ~A~%" (mem-aref info 8))
  (format t "size-y   : ~A~%" (mem-aref info 9))
  (format t "size-z   : ~A~%" (mem-aref info 10))
  (format t "raw-size : ~A~%" (mem-aref info 11)))


;;
;; cell
;;

;; Returns an array index of the lth element of the cell of given indices
(defkernel raw-index (int ((info float*) (i int) (j int) (k int) (l int)))
  (let ((nx (info-size-x info))
        (ny (info-size-y info)))
    (let ((cell-index (+ i (* j nx) (* k nx ny)) )
          (cell-size (info-cell-size info)))
      (return (+ (* cell-index cell-size) l)))))

(defun raw-index-cpu (info i j k l)
  (let ((nx (info-size-x-cpu info))
        (ny (info-size-y-cpu info)))
    (let ((cell-index (+ i (* j nx) (* k nx ny)))
          (cell-size (info-cell-size-cpu info)))
      (+ (* cell-index cell-size) l))))

;; Returns number of particles contained in the cell of given indices.
(defkernel cell-number-of-particles (int ((nbr int*) (info float*) (i int) (j int) (k int)))
  (let ((index (raw-index info i j k 0)))
    (return (aref nbr index))))

(defun cell-number-of-particles-cpu (nbr info i j k)
  (let ((index (raw-index-cpu info i j k 0)))
    (mem-aref nbr index)))

;; Returns index of the nth particle in the cell of given indices.
;; An index of particle begins from zero.
(defkernel cell-nth-particle (int ((nbr int*) (info float*) (i int) (j int) (k int) (n int)))
  (let ((index (raw-index info i j k (+ n 1))))
    (return (- (aref nbr index) 1))))

;; Inserts a new particle into the cell of given indices.
;; TODO: function type specifier
(defkernel insert-particle-to-cell (int ((nbr int*) (info float*) (i int) (j int) (k int) (n int)))
  (let ((index (raw-index info i j k 0)))
    (let ((l (atomic-add (pointer (aref nbr index)) 1)))
      (set (aref nbr (+ index l 1)) (+ n 1))))
  (return 0))

;; Clears the cell of given indices.
;; TODO: function type specifier
(defkernel clear-cell (int ((nbr int*) (info float*) (i int) (j int) (k int)))
  (do-range (l 0 (- (info-cell-size info) 1))
    (let ((index (raw-index info i j k l)))
      (set (aref nbr index) 0)))
  (return 0))


;;
;; neighbor-map
;;

(defun alloc-neighbor-map (info)
  (alloc-memory-block 'int (info-raw-size-cpu info)))

(defun free-neighbor-map (nbr)
  (free-memory-block nbr))

(defmacro with-neighbor-map ((var info) &body body)
  `(let ((,var (alloc-neighbor-map ,info)))
     (unwind-protect
          (progn ,@body)
       (free-neighbor-map ,var))))

(defkernelmacro and (&rest args)
  (case (length args)
    (0 1)
    (1 (car args))
    (t `(if ,(car args) (and ,@(cdr args)) 0))))

(defkernelmacro do-particles-in-cell ((var nbr info i j k) &body body)
  (alexandria:with-gensyms (n l)
    `(when (and (<= 0 ,i) (< ,i (info-size-x ,info))
                (<= 0 ,j) (< ,j (info-size-y ,info))
                (<= 0 ,k) (< ,k (info-size-z ,info)))
       (let ((,n (cell-number-of-particles ,nbr ,info ,i ,j ,k)))
         (do-range (,l 0 (- ,n 1))
           (let ((,var (cell-nth-particle ,nbr ,info ,i ,j ,k ,l)))
             ,@body))))))

;; TODO: should assert bounding
(defkernel pos-to-index-x (int ((info float*) (pos float3)))
  (return (floor (/ (- (float3-x pos) (info-box-min-x info))
                    (info-delta info)))))
(defkernel pos-to-index-x2 (int ((info float*) (pos float4)))
  (return (floor (/ (- (float4-x pos) (info-box-min-x info))
                    (info-delta info)))))

;; TODO: should assert bounding
(defkernel pos-to-index-y (int ((info float*) (pos float3)))
  (return (floor (/ (- (float3-y pos) (info-box-min-y info))
                    (info-delta info)))))
(defkernel pos-to-index-y2 (int ((info float*) (pos float4)))
  (return (floor (/ (- (float4-y pos) (info-box-min-y info))
                    (info-delta info)))))

;; TODO: should assert bounding
(defkernel pos-to-index-z (int ((info float*) (pos float3)))
  (return (floor (/ (- (float3-z pos) (info-box-min-z info))
                    (info-delta info)))))
(defkernel pos-to-index-z2 (int ((info float*) (pos float4)))
  (return (floor (/ (- (float4-z pos) (info-box-min-z info))
                    (info-delta info)))))

;; OK
;; Convert given position to corresponding indices of a cell and bind
;; it to i, j and k symbols, then evaluate body forms
(defkernelmacro with-cell-index ((i j k info pos) &body body)
  `(let ((,i (pos-to-index-x ,info ,pos))
         (,j (pos-to-index-y ,info ,pos))
         (,k (pos-to-index-z ,info ,pos)))
     ,@body))
(defkernelmacro with-cell-index2 ((i j k info pos) &body body)
  `(let ((,i (pos-to-index-x2 ,info ,pos))
         (,j (pos-to-index-y2 ,info ,pos))
         (,k (pos-to-index-z2 ,info ,pos)))
     ,@body))

;; OK
#|
(defkernelmacro do-neighbors ((var nbr info pos) &body body)
  (alexandria:with-gensyms (x y z i j k)
    `(with-cell-index (,x ,y ,z ,info ,pos)
       (do-range (,i (- ,x 1) (+ ,x 1))
         (do-range (,j (- ,y 1) (+ ,y 1))
           (do-range (,k (- ,z 1) (+ ,z 1))
             (do-particles-in-cell (,var ,nbr ,info ,i ,j ,k)
               ,@body)))))))
|#

(defkernelmacro do-neighbors ((var nbr info pos) &body body)
  (alexandria:with-gensyms (x y z i info2)
    `(with-shared-memory ((,info2 float 12))
       (do-range (,i 0 11)
         (set (aref ,info2 ,i) (aref ,info ,i)))
       (with-cell-index (,x ,y ,z ,info2 ,pos)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x -1) (+ ,y -1)  (+ ,z -1))
           ,@body)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x -1) (+ ,y -1)  (+ ,z  0))
           ,@body)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x -1) (+ ,y -1)  (+ ,z  1))
           ,@body)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x -1) (+ ,y  0)  (+ ,z -1))
           ,@body)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x -1) (+ ,y  0)  (+ ,z  0))
           ,@body)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x -1) (+ ,y  0)  (+ ,z  1))
           ,@body)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x -1) (+ ,y  1)  (+ ,z -1))
           ,@body)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x -1) (+ ,y  1)  (+ ,z  0))
           ,@body)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x -1) (+ ,y  1)  (+ ,z  1))
           ,@body)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x  0) (+ ,y -1)  (+ ,z -1))
           ,@body)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x  0) (+ ,y -1)  (+ ,z  0))
           ,@body)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x  0) (+ ,y -1)  (+ ,z  1))
           ,@body)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x  0) (+ ,y  0)  (+ ,z -1))
           ,@body)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x  0) (+ ,y  0)  (+ ,z  0))
           ,@body)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x  0) (+ ,y  0)  (+ ,z  1))
           ,@body)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x  0) (+ ,y  1)  (+ ,z -1))
           ,@body)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x  0) (+ ,y  1)  (+ ,z  0))
           ,@body)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x  0) (+ ,y  1)  (+ ,z  1))
           ,@body)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x  1) (+ ,y -1)  (+ ,z -1))
           ,@body)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x  1) (+ ,y -1)  (+ ,z  0))
           ,@body)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x  1) (+ ,y -1)  (+ ,z  1))
           ,@body)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x  1) (+ ,y  0)  (+ ,z -1))
           ,@body)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x  1) (+ ,y  0)  (+ ,z  0))
           ,@body)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x  1) (+ ,y  0)  (+ ,z  1))
           ,@body)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x  1) (+ ,y  1)  (+ ,z -1))
           ,@body)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x  1) (+ ,y  1)  (+ ,z  0))
           ,@body)
         (do-particles-in-cell (,var ,nbr ,info (+ ,x  1) (+ ,y  1)  (+ ,z  1))
           ,@body)))))

(defkernelmacro do-neighbors2 ((var nbr info pos) &body body)
  (alexandria:with-gensyms (x y z i j k info2)
    `(with-shared-memory ((,info2 float 12))
       (do-range (,i 0 11)
         (set (aref ,info2 ,i) (aref ,info ,i)))
       (with-cell-index2 (,x ,y ,z ,info2 ,pos)
         (do-range (,i (- ,x 1) (+ ,x 1))
           (do-range (,j (- ,y 1) (+ ,y 1))
             (do-range (,k (- ,z 1) (+ ,z 1))
               (do-particles-in-cell (,var ,nbr ,info2 ,i ,j ,k)
                 ,@body))))))))

(defkernel clear-neighbor-map (void ((nbr int*) (info float*)))
  (let ((i thread-idx-x)
        (j block-idx-x)
        (k block-idx-y))
    (clear-cell nbr info i j k)))

(defkernel clear-neighbor-map-serial (void ((nbr int*) (info float*)))
  (do-range (i 0 (- (info-size-x info) 1))
    (do-range (j 0 (- (info-size-y info) 1))
      (do-range (k 0 (- (info-size-z info) 1))
        (clear-cell nbr info i j k)))))

;; TODO: should cause error if cell is more than full
(defkernel update-neighbor-map (void ((nbr int*) (info float*) (x float3*) (n int)))
  (let ((index (get-index)))
    (unless (< index n) (return))
    (let ((xi (aref x index)))
      (with-cell-index (i j k info xi)
        (insert-particle-to-cell nbr info i j k index)))))

(defun dump-number-of-particles (nbr info)
  (dotimes (i (info-size-x-cpu info))
    (dotimes (j (info-size-y-cpu info))
      (dotimes (k (info-size-z-cpu info))
        (format t "(~A, ~A, ~A): ~A~%" i j k (cell-number-of-particles-cpu nbr info i j k))))))


;;;
;;;
;;;

(defkernel norm (float ((x float3)))
  (return (sqrt (+ (* (float3-x x) (float3-x x))
                   (* (float3-y x) (float3-y x))
                   (* (float3-z x) (float3-z x))))))

(defkernelmacro pi ()
  3.14159)

(defkernelmacro pmass ()
  0.00020543)

(defkernelmacro restdensity ()
  600.0)

(defkernelmacro simscale ()
  0.004)

(defkernelmacro intstiff ()
  3.0)

(defkernelmacro pdist ()
  '(expt (/ (pmass) (restdensity)) (/ 1.0 3.0)))

(defvar pmass 0.00020543)
(defvar restdensity 600.0)
(defvar pdist (expt (/ pmass restdensity) (/ 1.0 3.0)))
(defvar init-min (make-float3 0.0 0.0 -10.0))

(defkernelmacro h ()
  0.01)

(defkernelmacro visc ()
  0.2)

(defkernelmacro radius ()
  0.004)

(defkernelmacro extstiff ()
  10000.0)

(defkernelmacro extdamp ()
  256.0)

(defkernelmacro epsilon ()
  0.00001)

(defkernelmacro limit ()
  200.0)

(defkernelmacro box-min ()
  '(float3 0.0 0.0 -10.0))

(defkernelmacro box-max ()
  '(float3 20.0 50.0 10.0))

(defkernelmacro pow (x n)
  (check-type n fixnum)
  `(* ,@(loop repeat n collect x)))

(defkernel poly6-kernel (float ((x float3) (h float)))
  (let ((r (norm x)))
    (if (<= r (h))
        (return (* (/ 315.0
                      (* 64.0 (pi) (pow (h) 9)))
                   (pow (- (pow (h) 2) (pow r 2)) 3)))
        (return 0.0))))

(defkernel grad-spiky-kernel (float3 ((x float3) (h float)))
  (let ((r (norm x)))
;    (if (<= r (h))
        (return (* (/ -45.0 (* (pi) (pow (h) 6)))
                   (pow (- (h) r) 2)
                   (/ x r)))
;        (return (float3 0.0 0.0 0.0)))))
        ))

(defkernel rap-visc-kernel (float ((x float3) (h float)))
  (let ((r (norm x)))
;    (if (<= r (h))
        (return (* (/ 45.0 (* (pi) (pow (h) 6)))
                   (- (h) r)))
;        (return 0.0))))
        ))

(defkernel update-density (void ((rho float*) (x float3*) (nbr int*) (info float*) (h float) (simscale float) (n int)))
  (let ((i (get-index)))
    (unless (< i n) (return))
    (set (aref rho i) 0.0)
    (do-neighbors (j nbr info (aref x i))
      (let ((dr (* (- (aref x i) (aref x j))
                   simscale)))
        (inc (aref rho i) (* (pmass) (poly6-kernel dr h)))))))

(defkernel update-density2 (void ((rho float*) (x float4*) (nbr int*) (info float*) (h float) (simscale float) (n int)))
  (let ((i (get-index)))
    (unless (< i n) (return))
    (let ((xi (float4-to-float3 (aref x i))))
      (let ((tmp 0.0))
        (do-neighbors (j nbr info xi)
          (let ((xj (float4-to-float3 (aref x j)))
                (dr (* (- xi xj) simscale)))
            (when (<= (norm dr) (h))
              (inc tmp (* (pmass) (poly6-kernel dr h))))))
        (set (aref rho i) tmp)))))

(defkernel update-pressure (void ((prs float*) (rho float*) (n int)))
  (let ((i (get-index)))
    (unless (< i n) (return))
    (set (aref prs i) (* (- (aref rho i) (restdensity))
                         (intstiff)))))

(defkernel float4-to-float3 (float3 ((x float4)))
  (return (float3 (float4-x x) (float4-y x) (float4-z x))))

(defkernel float3-to-float4 (float4 ((x float3)))
  (return (float4 (float3-x x) (float3-y x) (float3-z x) 0.0)))

(defkernel pressure-term (float3 ((i int) (j int) (x float3*) (rho float*) (prs float*) (simscale float) (h float)))
  (let ((dr (* (- (aref x i) (aref x j))
               simscale)))
    (return (* (* (- (pmass)) (/ (+ (aref prs i) (aref prs j))
                                 (* 2.0 (aref rho j))))
               (grad-spiky-kernel dr h)))))

(defkernel pressure-term2 (float3 ((i int) (j int) (dr float3) (rho float*) (prs float*)
                                   (simscale float) (h float)))
;  (let ((dr (* (- xi xj) simscale)))
    (return (* (* (- (pmass)) (/ (+ (aref prs i) (aref prs j))
                                 (* 2.0 (aref rho j))))
               (grad-spiky-kernel dr h))));)

(defkernel viscosity-term (float3 ((i int) (j int) (x float3*) (v float3*) (rho float*) (simscale float) (h float)))
  (let ((dr (* (- (aref x i) (aref x j))
               simscale)))
    (return (* (* (visc) (/ (* (pmass) (- (aref v j) (aref v i)))
                            (aref rho j)))
               (rap-visc-kernel dr h)))))

(defkernel viscosity-term2 (float3 ((i int) (j int) (dr float3) (vi float3) (vj float3) (rho float*)
                                    (simscale float) (h float)))
;  (let ((dr (* (- xi xj) simscale)))
    (return (* (* (visc) (/ (* (pmass) (- vj vi))
                            (aref rho j)))
               (rap-visc-kernel dr h))));)

(defkernel update-force (void ((f float3*) (x float3*) (v float3*)
                               (rho float*) (prs float*)
                               (nbr int*) (info float*) (simscale float) (h float) (n int)))
  (let ((i (get-index)))
    (unless (< i n) (return))
    (set (aref f i) (float3 0.0 0.0 0.0))
    (do-neighbors (j nbr info (aref x i))
      (when (/= i j)
        (inc (aref f i) (pressure-term i j x rho prs simscale h))
        (inc (aref f i) (viscosity-term i j x v rho simscale h))))))

(defkernel update-force2 (void ((f float4*) (x float4*) (v float4*)
                                (rho float*) (prs float*)
                                (nbr int*) (info float*) (simscale float) (h float) (n int)))
  (let ((i (get-index)))
    (unless (< i n) (return))
    (let ((xi (float4-to-float3 (aref x i)))
          (vi (float4-to-float3 (aref v i))))
      (let ((force (float3 0.0 0.0 0.0)))
        (do-neighbors (j nbr info xi)
          (when (/= i j)
            (let ((xj (float4-to-float3 (aref x j)))
                  (vj (float4-to-float3 (aref v j))))
              (let ((dr (* (- xi xj) simscale)))
                (when (<= (norm dr) (h))
                  (inc force (pressure-term2 i j dr rho prs simscale h))
                  (inc force (viscosity-term2 i j dr vi vj rho simscale h)))))))
        (set (aref f i) (float3-to-float4 force))))))

(defkernel wall (float3 ((i int) (v float3*) (d float) (norm float3) (a float3) (simscale float)))
  (let ((diff (- (* 2.0 (radius))
                 (* d simscale)))
        (adj  (- (* (extstiff) diff)
                 (* (extdamp) (dot norm (aref v i))))))
    (if (> diff (epsilon))
        (return (+ a (* adj norm)))
        (return a))))

(defkernel x-wall-min (float3 ((i int) (x float3*) (v float3*) (a float3) (simscale float)))
  (return (wall i v
                (- (float3-x (aref x i)) (float3-x (box-min)))
                (float3 1.0 0.0 0.0)
                a
                simscale)))

(defkernel x-wall-max (float3 ((i int) (x float3*) (v float3*) (a float3) (simscale float)))
  (return (wall i v
                (- (float3-x (box-max)) (float3-x (aref x i)))
                (float3 -1.0 0.0 0.0)
                a
                simscale)))

(defkernel y-wall-min (float3 ((i int) (x float3*) (v float3*) (a float3) (simscale float)))
  (return (wall i v
                (- (float3-y (aref x i)) (float3-y (box-min)))
                (float3 0.0 1.0 0.0)
                a
                simscale)))

(defkernel y-wall-max (float3 ((i int) (x float3*) (v float3*) (a float3) (simscale float)))
  (return (wall i v
                (- (float3-y (box-max)) (float3-y (aref x i)))
                (float3 0.0 -1.0 0.0)
                a
                simscale)))

(defkernel z-wall-min (float3 ((i int) (x float3*) (v float3*) (a float3) (simscale float)))
  (return (wall i v
                (- (float3-z (aref x i)) (float3-z (box-min)))
                (float3 0.0 0.0 1.0)
                a
                simscale)))

(defkernel z-wall-max (float3 ((i int) (x float3*) (v float3*) (a float3) (simscale float)))
  (return (wall i v
                (- (float3-z (box-max)) (float3-z (aref x i)))
                (float3 0.0 0.0 -1.0)
                a
                simscale)))

(defkernel accel-limit (float3 ((a float3)))
  (let ((speed (norm a)))
    (if (> speed (limit))
        (return (* (/ (limit) speed) a))
        (return a))))

(defkernel boundary-condition (void ((x float3*) (v float3*) (a float3*) (simscale float) (n int)))
  (let ((i (get-index)))
    (unless (< i n) (return))
    (set (aref a i) (x-wall-min i x v (aref a i) simscale))
    (set (aref a i) (x-wall-max i x v (aref a i) simscale))
    (set (aref a i) (y-wall-min i x v (aref a i) simscale))
    (set (aref a i) (y-wall-max i x v (aref a i) simscale))
    (set (aref a i) (z-wall-min i x v (aref a i) simscale))
    (set (aref a i) (z-wall-max i x v (aref a i) simscale))
    (set (aref a i) (accel-limit (aref a i)))))

(defkernel update-acceleration (void ((a float3*) (f float3*) (rho float*) (g float3) (n int)))
  (let ((i (get-index)))
    (unless (< i n) (return))
    (set (aref a i) (+ (/ (aref f i) (aref rho i))
                       g))))

(defkernel update-velocity (void ((v float3*) (a float3*) (dt float) (n int)))
  (let ((i (get-index)))
    (unless (< i n) (return))
    (inc (aref v i) (* (aref a i) dt))))

(defkernel update-position (void ((x float3*) (v float3*) (dt float) (n int)))
  (let ((i (get-index)))
    (unless (< i n) (return))
    (inc (aref x i) (/ (* (aref v i) dt)
                       (simscale)))))

(defun initialize (pos vel simscale nx ny nz n)
  (dotimes (x nx)
    (dotimes (y ny)
      (dotimes (z nz)
        (let* ((d (* (/ pdist simscale) 0.95))
               (i (+ x (* y nx) (* z nx ny))))
          (setf (mem-aref pos i)
                (make-float3 (+ (float3-x init-min) d (* x d))
                             (+ (float3-y init-min) d (* y d))
                             (+ (float3-z init-min) d (* z d))))))))
  (dotimes (i n)
    (setf (mem-aref vel i) (make-float3 0.0 0.0 0.0))))


;;; Output functions

(defun file-name (i)
  (format nil "result~8,'0d.pov" i))

(defun head ()
  (format nil (concatenate 'string
                           "#include \"colors.inc\"~%"
                           "camera {~%"
                           "  location <10, 30, -40>~%"
                           "  look_at <10, 10, 0>~%"
                           "}~%"
                           "light_source { <0, 30, -30> color White }~%")))

(defun sphere (pos i)
  (let ((x (float3-x (mem-aref pos i)))
        (y (float3-y (mem-aref pos i)))
        (z (float3-z (mem-aref pos i))))
    (format nil
            (concatenate 'string
                         "sphere {~%"
                         "  <~F,~F,~F>,0.5~%"
                         "  texture {~%"
                         "    pigment { color Yellow }~%"
                         "  }~%"
                         "}~%")
            x y z)))

(defun output (pos frame n)
  (with-open-file (out (file-name frame)
                       :direction :output
                       :if-exists :supersede)
    (princ (head) out)
    (dotimes (i n)
      (princ (sphere pos i) out))))

(defun test-output ()
  (let ((dev-id 0)
        (nx 6) (ny 10) (nz 10)
        (n 600)
        (simscale 0.004))
    (with-cuda-context (dev-id)
      (with-memory-blocks ((x 'float3 n)
                           (v 'float3 n))
        (initialize x v simscale nx ny nz n)
        (output x 0 n)))))

(defkernel copy-float3-to-float4 (void ((x float3*) (y float4*) (n int)))
  (let ((i (get-index)))
    (unless (< i n) (return))
    (set (float4-x (aref y i)) (float3-x (aref x i)))
    (set (float4-y (aref y i)) (float3-y (aref x i)))
    (set (float4-z (aref y i)) (float3-z (aref x i)))
    (set (float4-w (aref y i)) 0.0)))

(defkernel copy-float4-to-float3 (void ((x float4*) (y float3*) (n int)))
  (let ((i (get-index)))
    (unless (< i n) (return))
    (set (float3-x (aref y i)) (float4-x (aref x i)))
    (set (float3-y (aref y i)) (float4-y (aref x i)))
    (set (float3-z (aref y i)) (float4-z (aref x i)))))

(defun main ()
  (let ((dev-id 0)
        (nx 6) (ny 10) (nz 10)
        (n 600)
        (dt 0.004)
        (h 0.01)
        (simscale 0.004)
        (g (make-float3 0.0 -9.8 0.0))
        (box-min (make-float3 0.0 0.0 -10.0))
        (box-max (make-float3 20.0 50.0 10.0))
        (cell-size 21.0)                 ; TODO int
        (grid-dim (list 16 1 1))
        (block-dim (list 64 1 1)))
    (with-cuda-context (dev-id)
      (with-memory-blocks ((x 'float3 n)
                           (x2 'float4 n)
                           (v 'float3 n)
                           (v2 'float4 n)
                           (a 'float3 n)
                           (f 'float3 n)
                           (f2 'float4 n)
                           (rho 'float n)
                           (prs 'float n))
        (with-neighbor-map-info (info box-min box-max (/ h simscale) cell-size)
        (with-neighbor-map (nbr info)
          (initialize x v simscale nx ny nz n)
          (memcpy-host-to-device x v info)
          (time
          (dotimes (i 30)
            ;; clear neighbor map
            (clear-neighbor-map  nbr info                     :grid-dim (list 21 9 1) :block-dim (list 9 1 1))
;            (clear-neighbor-map-serial nbr info)
            ;; update neighbor map with position
            (update-neighbor-map nbr info x n                 :grid-dim grid-dim :block-dim '(128 1 1))
            ;; update density, pressure and force
            (copy-float3-to-float4 x x2 n                     :grid-dim grid-dim :block-dim block-dim)
            (copy-float3-to-float4 v v2 n                     :grid-dim grid-dim :block-dim block-dim)
              ;; update density
              (update-density2 rho x2 nbr info h simscale n       :grid-dim grid-dim :block-dim '(384 1 1))
              ;; update pressure
              (update-pressure prs rho n                        :grid-dim grid-dim :block-dim block-dim)
              ;; update force
              (update-force2 f2 x2 v2 rho prs nbr info simscale h n :grid-dim grid-dim :block-dim '(64 1 1))
            (copy-float4-to-float3 f2 f n                     :grid-dim grid-dim :block-dim block-dim)
;            ;; update density
;            (update-density rho x nbr info h simscale n        :grid-dim grid-dim :block-dim '(384 1 1))
;            ;; update pressure
;            (update-pressure prs rho n                        :grid-dim grid-dim :block-dim block-dim)
;            ;; update force
;            (update-force f x v rho prs nbr info simscale h n  :grid-dim grid-dim :block-dim '(64 1 1))
            ;; update acceleration
            (update-acceleration a f rho g n                  :grid-dim grid-dim :block-dim block-dim)
            ;; apply boundary condition
            (boundary-condition x v a simscale n              :grid-dim grid-dim :block-dim block-dim)
            ;; update velocity
            (update-velocity v a dt n                         :grid-dim grid-dim :block-dim block-dim)
            ;; update position
            (update-position x v dt n                         :grid-dim grid-dim :block-dim block-dim)
;            (memcpy-device-to-host x)
;            (output x i n)
            (synchronize-context)
            ))))))))


;;
;; test neighbor-map
;;

(defun test-clear-neighbor-map ()
  (labels ((initialize (nbr info)
             (dotimes (i (info-raw-size-cpu info))
               (setf (mem-aref nbr i) 1)))
           (verify (nbr info)
             (cl-test-more:plan nil)
             (dotimes (i (info-size-x-cpu info))
               (dotimes (j (info-size-y-cpu info))
                 (dotimes (k (info-size-z-cpu info))
                   (cl-test-more:is (cell-number-of-particles-cpu nbr info i j k) 0))))
             (cl-test-more:finalize)
             (values)))
    (let ((dev-id 0)
          (cell-size 2.0)  ; TODO int
          (box-min (make-float3 0.0 0.0 0.0))
          (box-max (make-float3 3.0 3.0 3.0))
          (h 1.0))
      (with-cuda-context (dev-id)
        (with-neighbor-map-info (info box-min box-max h cell-size)
        (with-neighbor-map (nbr info)
          (initialize nbr info)
          (memcpy-host-to-device nbr info)
          (clear-neighbor-map nbr info :grid-dim (list 4 4 1) :block-dim (list 4 1 1))
          (memcpy-device-to-host nbr)
          (verify nbr info)))))))

(defun test-update-neighbor-map ()
  (labels ((init-position (x)
             (setf (mem-aref x 0) (make-float3 1.5 0.5 0.5)
                   (mem-aref x 1) (make-float3 2.5 0.5 0.5)
                   (mem-aref x 2) (make-float3 2.5 0.5 0.5)
                   (mem-aref x 3) (make-float3 3.5 0.5 0.5)
                   (mem-aref x 4) (make-float3 3.5 0.5 0.5)
                   (mem-aref x 5) (make-float3 3.5 0.5 0.5)
                   (mem-aref x 6) (make-float3 4.5 0.5 0.5)
                   (mem-aref x 7) (make-float3 4.5 0.5 0.5)
                   (mem-aref x 8) (make-float3 4.5 0.5 0.5)
                   (mem-aref x 9) (make-float3 4.5 0.5 0.5)))
           (verify (nbr)
             (cl-test-more:plan nil)
             (mapcar #'(lambda (arg)
                         (destructuring-bind (i expected) arg
                             (cl-test-more:is (mem-aref nbr i) expected)))
                     '(( 0 0) ( 1 0) ( 2 0) ( 3 0) ( 4  0)
                       ( 5 1) ( 6 1) ( 7 0) ( 8 0) ( 9  0)
                       (10 2) (11 2) (12 3) (13 0) (14  0)
                       (15 3) (16 4) (17 5) (18 6) (19  0)
                       (20 4) (21 7) (22 8) (23 9) (24 10)
                       (25 0) (26 0) (27 0) (28 0) (29  0)))
             (cl-test-more:finalize)
             (values)))
    (let ((dev-id 0)
          (cell-size 5.0)  ; TODO int
          (box-min (make-float3  0.0  0.0  0.0))
          (box-max (make-float3 10.0 10.0 10.0))
          (h 1.0))
      (with-cuda-context (dev-id)
        (with-neighbor-map-info (info box-min box-max h cell-size)
        (with-neighbor-map (nbr info)
          (with-memory-blocks ((x 'float3 10))
            (init-position x)
            (memcpy-host-to-device nbr info x)
            (clear-neighbor-map  nbr info      :grid-dim (list 11 11 1) :block-dim (list 11 1 1))
            (update-neighbor-map nbr info x 10 :grid-dim (list 1 1 1) :block-dim (list 10 1 1))
            (memcpy-device-to-host nbr)
            (verify nbr))))))))


;;
;; test cell
;;

(defkernel kernel-cell-number-of-particles (void ((nbr int*) (info float*) (y int*)))
  (set (aref y 0) (cell-number-of-particles nbr info 0 0 0))
  (set (aref y 1) (cell-number-of-particles nbr info 1 1 1)))

(defun test-cell-number-of-particles ()
  (labels ((init-position (x)
             (setf (mem-aref x 0) (make-float3 0.5 0.5 0.5)
                   (mem-aref x 1) (make-float3 0.5 0.5 0.5)
                   (mem-aref x 2) (make-float3 0.5 0.5 0.5)))
           (verify (y)
             (cl-test-more:plan nil)
             (cl-test-more:is (mem-aref y 0) 3)
             (cl-test-more:is (mem-aref y 1) 0)
             (cl-test-more:finalize)))
    (let ((dev-id 0)
          (cell-size 5.0)  ; TODO int
          (box-min (make-float3  0.0  0.0  0.0))
          (box-max (make-float3 10.0 10.0 10.0))
          (h 1.0))
      (with-cuda-context (dev-id)
        (with-neighbor-map-info (info box-min box-max h cell-size)
        (with-neighbor-map (nbr info)
          (with-memory-blocks ((x 'float3 3)
                               (y 'int 2))
            (init-position x)
            (memcpy-host-to-device nbr info x)
            (clear-neighbor-map  nbr info     :grid-dim (list 11 11 1) :block-dim (list 11 1 1))
            (update-neighbor-map nbr info x 3 :grid-dim (list 1 1 1) :block-dim (list 3 1 1))
            (kernel-cell-number-of-particles nbr info y :grid-dim (list 1 1 1) :block-dim (list 1 1 1))
            (memcpy-device-to-host y)
            (verify y))))))))

(defkernel kernel-do-particles-in-cell (void ((nbr int*) (info float*) (y int*)))
  (let ((i 0))
    (do-particles-in-cell (p nbr info 0 0 0)
      (set (aref y i) p)
      (set i (+ i 1)))
    (do-particles-in-cell (p nbr info 1 1 1)
      (set (aref y i) p)
      (set i (+ i 1)))))

(defun test-do-particles-in-cell ()
  (labels ((initialize (x y)
             (setf (mem-aref x 0) (make-float3 0.5 0.5 0.5)
                   (mem-aref x 1) (make-float3 0.5 0.5 0.5)
                   (mem-aref x 2) (make-float3 0.5 0.5 0.5))
             (dotimes (i 6)
               (setf (mem-aref y i) 0)))
           (verify (y)
             (cl-test-more:plan nil)
             (cl-test-more:is (mem-aref y 0) 0)
             (cl-test-more:is (mem-aref y 1) 1)
             (cl-test-more:is (mem-aref y 2) 2)
             (cl-test-more:is (mem-aref y 3) 0)
             (cl-test-more:is (mem-aref y 4) 0)
             (cl-test-more:is (mem-aref y 5) 0)
             (cl-test-more:finalize)))
    (let ((dev-id 0)
          (cell-size 5.0)  ; TODO int
          (box-min (make-float3  0.0  0.0  0.0))
          (box-max (make-float3 10.0 10.0 10.0))
          (h 1.0))
      (with-cuda-context (dev-id)
        (with-neighbor-map-info (info box-min box-max h cell-size)
        (with-neighbor-map (nbr info)
          (with-memory-blocks ((x 'float3 3)
                               (y 'int 6))
            (initialize x y)
            (memcpy-host-to-device nbr info x y)
            (clear-neighbor-map  nbr info     :grid-dim (list 11 11 1) :block-dim (list 11 1 1))
            (update-neighbor-map nbr info x 3 :grid-dim (list 1 1 1) :block-dim (list 3 1 1))
            (kernel-do-particles-in-cell nbr info y :grid-dim (list 1 1 1) :block-dim (list 1 1 1))
            (memcpy-device-to-host y)
            (verify y))))))))

(defun test-dump-number-of-particles ()
  (let ((dev-id 0)
        (nx 6) (ny 10) (nz 10)
        (n 600)
        (cell-size 11.0)
        (h 0.01)
        (simscale 0.004)
        (box-min (make-float3 0.0 0.0 -10.0))
        (box-max (make-float3 20.0 50.0 10.0))
        (grid-dim (list 3 1 1))
        (block-dim (list 256 1 1)))
    (labels ((initialize (pos)
               (dotimes (x nx)
                 (dotimes (y ny)
                   (dotimes (z nz)
                     (let* ((d (* (/ pdist simscale) 0.95))
                            (i (+ x (* y nx) (* z nx ny))))
                       (setf (mem-aref pos i)
                             (make-float3 (+ (float3-x init-min) d (* x d))
                                          (+ (float3-y init-min) d (* y d))
                                          (+ (float3-z init-min) d (* z d))))))))))
      (with-cuda-context (dev-id)
        (with-neighbor-map-info (info box-min box-max (/ h simscale) cell-size)
        (with-neighbor-map (nbr info)
        (with-memory-blocks ((x 'float3 n))
          (initialize x)
          (memcpy-host-to-device x info)
          (clear-neighbor-map  nbr info     :grid-dim (list 21 9 1) :block-dim (list 9 1 1))
          (update-neighbor-map nbr info x n :grid-dim grid-dim :block-dim block-dim)
          (memcpy-device-to-host nbr)
          (dump-number-of-particles nbr info))))))))


(defkernel kernel-do-neighbors (void ((nbr int*) (info float*) (y int*)))
  (let ((i 0))
    (do-neighbors (p nbr info (float3 0.5 0.5 0.5))
      (set (aref y i) p)
      (set i (+ i 1)))))

(defun test-do-neighbors ()
  (labels ((init-position (x)
             (setf (mem-aref x 0) (make-float3 0.5 0.5 0.5)
                   (mem-aref x 1) (make-float3 0.5 0.5 1.5)
                   (mem-aref x 2) (make-float3 0.5 1.5 0.5)
                   (mem-aref x 3) (make-float3 0.5 1.5 1.5)
                   (mem-aref x 4) (make-float3 1.5 0.5 0.5)
                   (mem-aref x 5) (make-float3 1.5 0.5 1.5)
                   (mem-aref x 6) (make-float3 1.5 1.5 0.5)
                   (mem-aref x 7) (make-float3 1.5 1.5 1.5)))
           (verify (y)
             (cl-test-more:plan nil)
             (cl-test-more:is (mem-aref y 0) 0)
             (cl-test-more:is (mem-aref y 1) 1)
             (cl-test-more:is (mem-aref y 2) 2)
             (cl-test-more:is (mem-aref y 3) 3)
             (cl-test-more:is (mem-aref y 4) 4)
             (cl-test-more:is (mem-aref y 5) 5)
             (cl-test-more:is (mem-aref y 6) 6)
             (cl-test-more:is (mem-aref y 7) 7)
             (cl-test-more:finalize)))
    (let ((dev-id 0)
          (cell-size 5.0)  ; TODO int
          (box-min (make-float3  0.0  0.0  0.0))
          (box-max (make-float3 10.0 10.0 10.0))
          (h 1.0))
      (with-cuda-context (dev-id)
        (with-neighbor-map-info (info box-min box-max h cell-size)
          (with-neighbor-map (nbr info)
            (with-memory-blocks ((x 'float3 8)
                                 (y 'int 8))
              (init-position x)
              (memcpy-host-to-device nbr info x)
              (clear-neighbor-map  nbr info     :grid-dim (list 11 11 1) :block-dim (list 11 1 1))
              (update-neighbor-map nbr info x 8 :grid-dim (list 1 1 1) :block-dim (list 8 1 1))
              (kernel-do-neighbors nbr info y   :grid-dim (list 1 1 1) :block-dim (list 1 1 1))
              (memcpy-device-to-host y)
              (verify y))))))))

(defkernel kernel-and (void ((x int*)))
  (if (and 1 2 3)
      (set (aref x 0) 1))
  (if (and 0 0 0)
      (set (aref x 1) 1)))

(defun test-and ()
  (let ((dev-id 0))
    (with-cuda-context (dev-id)
      (with-memory-blocks ((x 'int 2))
        (setf (mem-aref x 0) 0
              (mem-aref x 1) 0)
        (memcpy-host-to-device x)
        (kernel-and x)
        (memcpy-device-to-host x)
        (cl-test-more:is (mem-aref x 0) 1)))))
