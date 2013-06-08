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

(defkernelmacro with-valid-index ((i n) &body body)
  `(let ((,i (+ (* block-idx-x block-dim-x) thread-idx-x)))
     (when (< ,i ,n)
       ,@body)))

(defmacro with-valid-index-cpu ((i n) &body body)
  `(let ((,i (+ (* block-idx-x block-dim-x) thread-idx-x)))
     (when (< ,i ,n)
       ,@body)))

(defmacro with-thread-block ((grid-dim block-dim) &body body)
  `(destructuring-bind (grid-dim-x grid-dim-y grid-dim-z) ,grid-dim
     (destructuring-bind (block-dim-x block-dim-y block-dim-z) ,block-dim
       (dotimes (block-idx-x grid-dim-x)
         (dotimes (block-idx-y grid-dim-y)
           (dotimes (block-idx-z grid-dim-z)
             (dotimes (thread-idx-x block-dim-x)
               (dotimes (thread-idx-y block-dim-y)
                 (dotimes (thread-idx-z block-dim-z)
                   ,@body)))))))))

(defkernelmacro when (test &body forms)
  `(if ,test
       (progn ,@forms)))

(defkernelmacro unless (test &body forms)
  `(when (not ,test)
     ,@forms))

(defkernelmacro and (&rest args)
  (case (length args)
    (0 t)
    (1 (car args))
    (t `(if ,(car args) (and ,@(cdr args)) nil))))

(defkernelmacro inc (place val)
  `(set ,place (+ ,place ,val)))

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

(defun float4-add-cpu (a b)
  (make-float4 (+ (float4-x a) (float4-x b))
               (+ (float4-y a) (float4-y b))
               (+ (float4-z a) (float4-z b))
               (+ (float4-w a) (float4-w b))))

(defun float4-sub-cpu (a b)
  (make-float4 (- (float4-x a) (float4-x b))
               (- (float4-y a) (float4-y b))
               (- (float4-z a) (float4-z b))
               (- (float4-w a) (float4-w b))))

(defun float4-scale-cpu (a k)
  (make-float4 (* (float4-x a) k)
               (* (float4-y a) k)
               (* (float4-z a) k)
               (* (float4-w a) k)))

(defun float4-scale-flipped-cpu (k a)
  (float4-scale-cpu a k))

(defun float4-scale-inverted-cpu (a k)
  (float4-scale-cpu a (/ k)))

(defmacro float4-incf-cpu (var val)
  `(setf ,var (float4-add-cpu ,var ,val)))

(defkernel norm (float ((x float4)))
  (return (sqrt (+ (* (float4-x x) (float4-x x))
                   (* (float4-y x) (float4-y x))
                   (* (float4-z x) (float4-z x))
                   (* (float4-w x) (float4-w x))))))

(defun norm-cpu (val)
  (with-float4-cpu (x y z w) val
    (sqrt (+ (* x x) (* y y) (* z z) (* w w)))))

(defun dot-cpu (a b)
  (+ (* (float4-x a) (float4-x b))
     (* (float4-y a) (float4-y b))
     (* (float4-z a) (float4-z b))
     (* (float4-w a) (float4-w b))))

(defkernelmacro pow (x n)
  (check-type n fixnum)
  `(* ,@(loop repeat n collect x)))

;; (defkernel pow (float ((b float) (p float)))
;;   (return (expt b p)))

(defun pow-cpu (b p)
  (float (expt b p) 0.0))


;;;
;;; Neighbor map
;;;

(defun alloc-neighbor-map-info (box-min box-max delta capacity)
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
          (let ((info (alloc-memory-block 'float 15)))
            (setf (mem-aref info 0)  box-min-x
                  (mem-aref info 1)  box-min-y
                  (mem-aref info 2)  box-min-z
                  (mem-aref info 3)  (+ box-min-x (* delta size-x))
                  (mem-aref info 4)  (+ box-min-y (* delta size-y))
                  (mem-aref info 5)  (+ box-min-z (* delta size-z))
                  (mem-aref info 6)  origin-x
                  (mem-aref info 7)  origin-y
                  (mem-aref info 8)  origin-z
                  (mem-aref info 9)  delta
                  (mem-aref info 10) (float capacity)
                  (mem-aref info 11) (+ size-x 2)
                  (mem-aref info 12) (+ size-y 2)
                  (mem-aref info 13) (+ size-z 2)
                  (mem-aref info 14) (* (mem-aref info 11)   ; size-x
                                        (mem-aref info 12)   ; size-y
                                        (mem-aref info 13))) ; size-z
            info))))))

(defun free-neighbor-map-info (info)
  (free-memory-block info))

(defun info-min-x-cpu    (info) (mem-aref info 0))
(defun info-min-y-cpu    (info) (mem-aref info 1))
(defun info-min-z-cpu    (info) (mem-aref info 2))
(defun info-max-x-cpu    (info) (mem-aref info 3))
(defun info-max-y-cpu    (info) (mem-aref info 4))
(defun info-max-z-cpu    (info) (mem-aref info 5))
(defun info-origin-x-cpu (info) (mem-aref info 6))
(defun info-origin-y-cpu (info) (mem-aref info 7))
(defun info-origin-z-cpu (info) (mem-aref info 8))
(defun info-delta-cpu    (info) (mem-aref info 9))
(defun info-capacity-cpu (info) (floor (mem-aref info 10)))
(defun info-size-x-cpu   (info) (floor (mem-aref info 11)))
(defun info-size-y-cpu   (info) (floor (mem-aref info 12)))
(defun info-size-z-cpu   (info) (floor (mem-aref info 13)))
(defun info-size-cpu     (info) (floor (mem-aref info 14)))

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

(defun neighbor-map-info-plist (info)
  (list :box-min-x (info-min-x-cpu info)
        :box-min-y (info-min-y-cpu info)
        :box-min-z (info-min-z-cpu info)
        :box-max-x (info-max-x-cpu info)
        :box-max-y (info-max-y-cpu info)
        :box-max-z (info-max-z-cpu info)
        :origin-x  (info-origin-x-cpu info)
        :origin-y  (info-origin-y-cpu info)
        :origin-z  (info-origin-z-cpu info)
        :delta     (info-delta-cpu info)
        :capacity  (info-capacity-cpu info)
        :size-x    (info-size-x-cpu info)
        :size-y    (info-size-y-cpu info)
        :size-z    (info-size-z-cpu info)
        :size      (info-size-cpu info)))

(defkernel %pos-to-cell (int ((x float) (x0 float) (delta float)))
  (return (floor (/ (- x x0) delta))))

(defun %pos-to-cell-cpu (x x0 delta)
  (floor (/ (- x x0) delta)))

(defkernel pos-to-cell-x (int ((pos float4) (info float*)))
  (let ((x (float4-x pos))
        (x0 (info-origin-x info))
        (delta (info-delta info)))
    (return (%pos-to-cell x x0 delta))))

(defun pos-to-cell-x-cpu (pos info)
  (let ((x (float4-x pos))
        (x0 (info-origin-x-cpu info))
        (delta (info-delta-cpu info)))
    (%pos-to-cell-cpu x x0 delta)))

(defkernel pos-to-cell-y (int ((pos float4) (info float*)))
  (let ((y (float4-y pos))
        (y0 (info-origin-y info))
        (delta (info-delta info)))
    (return (%pos-to-cell y y0 delta))))

(defun pos-to-cell-y-cpu (pos info)
  (let ((y (float4-y pos))
        (y0 (info-origin-y-cpu info))
        (delta (info-delta-cpu info)))
    (%pos-to-cell-cpu y y0 delta)))

(defkernel pos-to-cell-z (int ((pos float4) (info float*)))
  (let ((z (float4-z pos))
        (z0 (info-origin-z info))
        (delta (info-delta info)))
    (return (%pos-to-cell z z0 delta))))

(defun pos-to-cell-z-cpu (pos info)
  (let ((z (float4-z pos))
        (z0 (info-origin-z-cpu info))
        (delta (info-delta-cpu info)))
    (%pos-to-cell-cpu z z0 delta)))

(defkernelmacro with-cell ((i j k info x) &body body)
  `(let ((,i (pos-to-cell-x ,x ,info))
         (,j (pos-to-cell-y ,x ,info))
         (,k (pos-to-cell-z ,x ,info)))
     ,@body))

(defmacro with-cell-cpu ((i j k info x) &body body)
  `(let ((,i (pos-to-cell-x-cpu ,x ,info))
         (,j (pos-to-cell-y-cpu ,x ,info))
         (,k (pos-to-cell-z-cpu ,x ,info)))
     ,@body))

(defun alloc-neighbor-map (info)
  (let ((size (info-size-cpu info))
        (capacity (info-capacity-cpu info)))
    (alloc-memory-block 'int (* size (1+ capacity)))))

(defun free-neighbor-map (nbr)
  (free-memory-block nbr))

(defkernel cell-number-of-particles (int ((offset int) (nbr int*)))
  (return (aref nbr offset)))

(defun cell-number-of-particles-cpu (offset nbr)
  (mem-aref nbr offset))

(defkernel cell-nth-particle (int ((n int) (offset int) (nbr int*)))
  (return (aref nbr (+ offset n 1))))   ; increment need because n begins with 0

(defun cell-nth-particle-cpu (n offset nbr)
  (mem-aref nbr (+ offset n 1)))

;; returns dummy integer to avoid __host__ qualifier
(defkernel insert-cell (int ((p int) (offset int) (nbr int*)))
  (let ((n (atomic-add (pointer (aref nbr offset)) 1)))
    (set (aref nbr (+ offset n 1)) p))
  (return 0))

(defun insert-cell-cpu (p offset nbr)
  (let ((n (mem-aref nbr offset)))
    (setf (mem-aref nbr offset) (+ (mem-aref nbr offset) 1))
    (setf (mem-aref nbr (+ offset n 1)) p)))

;; returns dummy integer to avoid __host__ qualifier
(defkernel clear-cell (int ((offset int) (nbr int*)))
  (set (aref nbr offset) 0)   ; particles in cell are not cleared for performance reason
  (return 0))

(defun clear-cell-cpu (offset nbr)
  (setf (mem-aref nbr offset) 0))

(defkernel cell-offset (int ((nbr int*) (info float*) (i int) (j int) (k int)))
  (let ((size-x (info-size-x info))
        (size-y (info-size-y info))
        (capacity (info-capacity info)))
    (let ((offset-base (+ (* size-x size-y k)
                          (* size-x j)
                          i))
          (capacity1 (+ capacity 1)))
      (return (* offset-base capacity1)))))

(defun cell-offset-cpu (nbr info i j k)
  (declare (ignore nbr))
  (let ((size-x (info-size-x-cpu info))
        (size-y (info-size-y-cpu info))
        (capacity (info-capacity-cpu info)))
    (let ((offset-base (+ (* size-x size-y k)
                          (* size-x j)
                          i))
          (capacity1 (+ capacity 1)))
      (* offset-base capacity1))))

(defkernel valid-cell-index (bool ((info float*) (i int) (j int) (k int)))
  (return (and (<= 0 i) (< i (info-size-x info))
               (<= 0 j) (< j (info-size-y info))
               (<= 0 k) (< k (info-size-z info))
               t)))

(defun valid-cell-index-cpu (info i j k)
  (and (<= 0 i) (< i (info-size-x-cpu info))
       (<= 0 j) (< j (info-size-y-cpu info))
       (<= 0 k) (< k (info-size-z-cpu info))
       t))

(defkernel number-of-particles-in-cell (int ((nbr int*) (info float*) (i int) (j int) (k int)))
  (unless (valid-cell-index info i j k)
    (return 0))
  (let ((offset (cell-offset nbr info i j k)))
    (return (cell-number-of-particles offset nbr))))

(defun number-of-particles-in-cell-cpu (nbr info i j k)
  (unless (valid-cell-index-cpu info i j k)
    (return-from number-of-particles-in-cell-cpu 0))
  (let ((offset (cell-offset-cpu nbr info i j k)))
    (cell-number-of-particles-cpu offset nbr)))

(defkernel nth-particle-in-cell (int ((n int) (nbr int*) (info float*) (i int) (j int) (k int)))
  (let ((offset (cell-offset nbr info i j k)))
    (return (cell-nth-particle n offset nbr))))

(defun nth-particle-in-cell-cpu (n nbr info i j k)
  (let ((offset (cell-offset-cpu nbr info i j k)))
    (cell-nth-particle-cpu n offset nbr)))

(defkernelmacro do-particles-in-cell ((p nbr info i j k) &body body)
  (alexandria:with-gensyms (n index)
    `(when (valid-cell-index ,info ,i ,j ,k)
       (let ((,n (number-of-particles-in-cell ,nbr ,info ,i ,j ,k)))
         (do-range (,index 0 (- ,n 1))
           (let ((,p (nth-particle-in-cell ,index ,nbr ,info ,i ,j ,k)))
             ,@body))))))

(defmacro do-particles-in-cell-cpu ((p nbr info i j k) &body body)
  (alexandria:with-gensyms (n index)
    `(when (valid-cell-index-cpu ,info ,i ,j ,k)
       (let ((,n (number-of-particles-in-cell-cpu ,nbr ,info ,i ,j ,k)))
         (do-range-cpu (,index 0 (- ,n 1))
           (let ((,p (nth-particle-in-cell-cpu ,index ,nbr ,info ,i ,j ,k)))
             ,@body))))))

;; returns dummy integer to avoid __host__ qualifier
(defkernel insert-particle-in-cell (int ((p int) (nbr int*) (info float*) (i int) (j int) (k int)))
  (unless (valid-cell-index info i j k)
    (return 0))
  (let ((offset (cell-offset nbr info i j k)))
    (insert-cell p offset nbr))
  (return 0))

(defun insert-particle-in-cell-cpu (p nbr info i j k)
  (unless (valid-cell-index-cpu info i j k)
    (return-from insert-particle-in-cell-cpu))
  (let ((offset (cell-offset-cpu nbr info i j k)))
    (insert-cell-cpu p offset nbr)))

;; returns dummy integer to avoid __host__ qualifier
(defkernel clear-particles-in-cell (int ((nbr int*) (info float*) (i int) (j int) (k int)))
  (unless (valid-cell-index info i j k)
    (return 0))
  (let ((offset (cell-offset nbr info i j k)))
    (clear-cell offset nbr))
  (return 0))

(defun clear-particles-in-cell-cpu (nbr info i j k)
  (unless (valid-cell-index-cpu info i j k)
    (return-from clear-particles-in-cell-cpu))
  (let ((offset (cell-offset-cpu nbr info i j k)))
    (clear-cell-cpu offset nbr)))

(defkernelmacro do-range ((var from to) &body body)
  `(do ((,var ,from (+ ,var 1)))
       ((> ,var ,to))
     ,@body))

(defmacro do-range-cpu ((var from to) &body body)
  `(do ((,var ,from (+ ,var 1)))
       ((> ,var ,to))
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

(defmacro do-neighbor-particles-cpu ((p nbr info x) &body body)
  (alexandria:with-gensyms (i0 j0 k0 i j k)
    `(with-cell-cpu (,i0 ,j0 ,k0 ,info ,x)
       (do-range-cpu (,i (- ,i0 1) (+ ,i0 1))
         (do-range-cpu (,j (- ,j0 1) (+ ,j0 1))
           (do-range-cpu (,k (- ,k0 1) (+ ,k0 1))
             (do-particles-in-cell-cpu (,p ,nbr ,info ,i ,j ,k)
               ,@body)))))))

;; returns dummy integer to avoid __host__ qualifier
(defkernel insert-particle-in-neighbor-map (int ((p int) (x float4) (nbr int*) (info float*)))
  (with-cell (i j k info x)
    (insert-particle-in-cell p nbr info i j k))
  (return 0))

(defun insert-particle-in-neighbor-map-cpu (p x nbr info)
  (with-cell-cpu (i j k info x)
    (insert-particle-in-cell-cpu p nbr info i j k)))

;; returns dummy integer to avoid __host__ qualifier
(defkernel %update-neighbor-map (int ((i int) (pos float4*) (nbr int*) (info float*)))
  (let ((x (aref pos i)))
    (insert-particle-in-neighbor-map i x nbr info))
  (return 0))

(defkernel update-neighbor-map (void ((pos float4*) (nbr int*) (info float*) (n int)))
  (with-valid-index (i n)
    (%update-neighbor-map i pos nbr info)))

(defun %update-neighbor-map-cpu (i pos nbr info)
  (let ((x (mem-aref pos i)))
    (insert-particle-in-neighbor-map-cpu i x nbr info)))

(defun update-neighbor-map-cpu (pos nbr info n &key grid-dim block-dim)
  (with-thread-block (grid-dim block-dim)
    (with-valid-index-cpu (i n)
      (%update-neighbor-map-cpu i pos nbr info))))

(defkernel clear-neighbor-map (void ((nbr int*) (info float*)))
  (let ((i thread-idx-x)
        (j block-idx-x)
        (k block-idx-y))
    (clear-particles-in-cell nbr info i j k)))

(defun clear-neighbor-map-cpu (nbr info &key grid-dim block-dim)
  (with-thread-block (grid-dim block-dim)
    (let ((i thread-idx-x)
          (j block-idx-x)
          (k block-idx-y))
      (clear-particles-in-cell-cpu nbr info i j k))))

(defun memcpy-neighbor-map-host-to-device (nbr info)
  (memcpy-host-to-device nbr info))

(defun memcpy-neighbor-map-device-to-host (nbr info)
  (memcpy-device-to-host nbr info))

(defmacro with-neighbor-map ((nbr info box-min box-max delta capacity) &body body)
  `(let* ((,info (alloc-neighbor-map-info ,box-min ,box-max ,delta ,capacity))
          (,nbr  (alloc-neighbor-map ,info)))
     (unwind-protect
          (progn ,@body)
       (free-neighbor-map ,nbr)
       (free-neighbor-map-info ,info))))


;;;
;;; Constants
;;;

;; (defmacro defkernelconst (var val)
;;   (when (and (listp val)
;;              (eq (car val) 'float4))
;;     (setf (car val) 'make-float4))
;;   `(defparameter ,var ,val))

(defkernelconst h           float  0.01)
(defkernelconst dt          float  0.004)
(defkernelconst pi          float  3.1415927)
(defkernelconst visc        float  0.2)
(defkernelconst limit       float  200.0)
(defkernelconst pmass       float  0.00020543)
(defkernelconst radius      float  0.004)
(defkernelconst epsilon     float  0.00001)
(defkernelconst extdamp     float  256.0)
(defkernelconst simscale    float  0.004)
(defkernelconst intstiff    float  3.0)
(defkernelconst extstiff    float  10000.0)
(defkernelconst restdensity float  600.0)
(defkernelconst pdist       float  (expt (/ pmass restdensity) (/ 1.0 3.0)))
(defkernelconst g           float4 (float4  0.0 -9.8   0.0 0.0))

(defparameter h           0.01)
(defparameter dt          0.004)
(defparameter pi-cpu      3.1415927)
(defparameter visc        0.2)
(defparameter limit       200.0)
(defparameter pmass       0.00020543)
(defparameter radius      0.004)
(defparameter epsilon     0.00001)
(defparameter extdamp     256.0)
(defparameter simscale    0.004)
(defparameter intstiff    3.0)
(defparameter extstiff    10000.0)
(defparameter restdensity 600.0)
(defparameter pdist       (expt (/ pmass restdensity) (/ 1.0 3.0)))
(defparameter g           (make-float4  0.0 -9.8   0.0 0.0))

(defparameter delta       (/ h simscale))
(defparameter box-min     (make-float4  0.0  0.0 -10.0 0.0))
(defparameter box-max     (make-float4 20.0 50.0  10.0 0.0))
(defparameter init-min    (make-float4  0.0  0.0 -10.0 0.0))
(defparameter init-max    (make-float4 10.0 20.0  10.0 0.0))
(defparameter capacity    20.0)          ; # of particles contained in one cell


;;;
;;; Kernel functions
;;;

(defkernel poly6-kernel (float ((x float4)))
  (let ((r (norm x)))
    (return (* (/ 315.0 (* 64.0 pi (pow h 9)))
               (pow (- (* h h) (* r r)) 3)))))

(defun poly6-kernel-cpu (x)
  (let ((r (norm-cpu x)))
    (* (/ 315.0 (* 64.0 pi-cpu (pow-cpu h 9)))
       (pow-cpu (- (* h h) (* r r)) 3))))


(defkernel grad-spiky-kernel (float4 ((x float4)))
  (let ((r (norm x)))
    (return (* (/ -45.0 (* pi (pow h 6)))
               (pow (- h r) 2)
               (/ x r)))))

(defun grad-spiky-kernel-cpu (x)
  (let ((r (norm-cpu x)))
    (float4-scale-flipped-cpu (* (/ -45.0 (* pi-cpu (pow-cpu h 6)))
                                 (pow-cpu (- h r) 2))
                              (float4-scale-inverted-cpu x r))))

(defkernel rap-visc-kernel (float ((x float4)))
  (let ((r (norm x)))
    (return (* (/ 45.0 (* pi (pow h 6)))
               (- h r)))))

(defun rap-visc-kernel-cpu (x)
  (let ((r (norm-cpu x)))
    (* (/ 45.0 (* pi-cpu (pow-cpu h 6)))
       (- h r))))

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

(defun %update-density-cpu (i rho x nbr info)
  (symbol-macrolet ((xi (mem-aref x i))
                    (xj (mem-aref x j))
                    (rhoi (mem-aref rho i)))
    (let ((tmp 0.0))
      (do-neighbor-particles-cpu (j nbr info xi)
        (let ((dr (float4-scale-cpu (float4-sub-cpu xi xj) simscale)))
          (when (<= (norm-cpu dr) h)
            (incf tmp (* pmass (poly6-kernel-cpu dr))))))
      (setf rhoi tmp))))

(defun update-density-cpu (rho x nbr info n &key grid-dim block-dim)
  (with-thread-block (grid-dim block-dim)
    (with-valid-index-cpu (i n)
      (%update-density-cpu i rho x nbr info))))

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

(defun %update-pressure-cpu (i prs rho)
  (symbol-macrolet ((rhoi (mem-aref rho i))
                    (prsi (mem-aref prs i)))
    (setf prsi (* (- rhoi restdensity)
                  intstiff))))

(defun update-pressure-cpu (prs rho n &key grid-dim block-dim)
  (with-thread-block (grid-dim block-dim)
    (with-valid-index-cpu (i n)
      (%update-pressure-cpu i prs rho))))

(defkernel pressure-term (float4 ((i int) (j int) (dr float4) (rho float*) (prs float*)))
  (symbol-macrolet ((rhoj (aref rho j))
                    (prsi (aref prs i))
                    (prsj (aref prs j)))
    (return (* (/ (* (- pmass) (+ prsi prsj))
                  (* 2.0 rhoj))
               (grad-spiky-kernel dr)))))

(defun pressure-term-cpu (i j dr rho prs)
  (symbol-macrolet ((rhoj (mem-aref rho j))
                    (prsi (mem-aref prs i))
                    (prsj (mem-aref prs j)))
    (float4-scale-flipped-cpu (/ (* (- pmass) (+ prsi prsj))
                                 (* 2.0 rhoj))
                              (grad-spiky-kernel-cpu dr))))

(defkernel viscosity-term (float4 ((i int) (j int) (dr float4) (v float4*) (rho float*)))
  (symbol-macrolet ((vi (aref v i))
                    (vj (aref v j))
                    (rhoj (aref rho j)))
    (return (* (/ (* visc pmass (- vj vi))
                  rhoj)
               (rap-visc-kernel dr)))))

(defun viscosity-term-cpu (i j dr v rho)
  (symbol-macrolet ((vi (mem-aref v i))
                    (vj (mem-aref v j))
                    (rhoj (mem-aref rho j)))
    (float4-scale-cpu (float4-scale-inverted-cpu (float4-scale-flipped-cpu (* visc pmass) (float4-sub-cpu vj vi))
                                                 rhoj)
                      (rap-visc-kernel-cpu dr))))

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

(defun %update-force-cpu (i f x v rho prs nbr info)
  (symbol-macrolet ((xi (mem-aref x i))
                    (xj (mem-aref x j))
                    (fi (mem-aref f i)))
    (let ((tmp (make-float4 0.0 0.0 0.0 0.0)))
      (do-neighbor-particles-cpu (j nbr info xi)
        (when (/= i j)
          (let ((dr (float4-scale-cpu (float4-sub-cpu xi xj) simscale)))
            (when (<= (norm-cpu dr) h)
              (float4-incf-cpu tmp (pressure-term-cpu  i j dr rho prs))
              (float4-incf-cpu tmp (viscosity-term-cpu i j dr v rho))))))
      (setf fi tmp))))

(defun update-force-cpu (f x v rho prs nbr info n &key grid-dim block-dim)
  (with-thread-block (grid-dim block-dim)
    (with-valid-index-cpu (i n)
      (%update-force-cpu i f x v rho prs nbr info))))

(defkernel collision-diff (float ((x0 float) (x1 float)))
  (let ((distance (* (- x1 x0) simscale)))
    (return (- (* radius 2.0) distance))))

(defun collision-diff-cpu (x0 x1)
  (let ((distance (* (- x1 x0) simscale)))
    (- (* 2.0 radius) distance)))

(defkernel collision-adj (float4 ((diff float) (v float4) (normal float4)))
  (let ((adj (- (* extstiff diff)
                (* extdamp (dot normal v)))))
    (return (* adj normal))))

(defun collision-adj-cpu (diff v normal)
  (let ((adj (- (* extstiff diff)
                (* extdamp (dot-cpu normal v)))))
    (float4-scale-flipped-cpu adj normal)))

;; returns dummy integer to avoid __host__ qualifier
(defkernel apply-collision (int ((a float4*) (i int) (x0 float) (x1 float) (v float4) (normal float4)))
  (symbol-macrolet ((ai (aref a i)))
    (let ((diff (collision-diff x0 x1)))
      (when (< epsilon diff)
        (inc ai (collision-adj diff v normal)))))
  (return 0))

(defun apply-collision-cpu (a i x0 x1 v normal)
  (symbol-macrolet ((ai (mem-aref a i)))
    (let ((diff (collision-diff-cpu x0 x1)))
      (when (< epsilon diff)
        (float4-incf-cpu ai (collision-adj-cpu diff v normal))))))
                    
(defkernel accel-limit-adj (float ((accel float)))
  (return (/ limit accel)))

(defun accel-limit-adj-cpu (accel)
  (/ limit accel))

;; returns dummy integer to avoid __host__ qualifier
(defkernel apply-accel-limit (int ((a float4*) (i int)))
  (symbol-macrolet ((ai (aref a i)))
    (let ((accel (norm ai)))
      (when (< limit accel)
        (set ai (* ai (accel-limit-adj accel))))))
  (return 0))

(defun apply-accel-limit-cpu (a i)
  (symbol-macrolet ((ai (mem-aref a i)))
    (let ((accel (norm-cpu ai)))
      (when (< limit accel)
        (setf ai (float4-scale-cpu ai (accel-limit-adj-cpu accel)))))))

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

(defun %boundary-condition-cpu (i x v a info)
  (symbol-macrolet ((xi (mem-aref x i))
                    (vi (mem-aref v i)))
    ;; left boundary
    (apply-collision-cpu a i (info-min-x-cpu info) (float4-x xi) vi (make-float4 1.0 0.0 0.0 0.0))
    ;; right boundary
    (apply-collision-cpu a i (float4-x xi) (info-max-x-cpu info) vi (make-float4 -1.0 0.0 0.0 0.0))
    ;; bottom boundary
    (apply-collision-cpu a i (info-min-y-cpu info) (float4-y xi) vi (make-float4 0.0 1.0 0.0 0.0))
    ;; top boundary
    (apply-collision-cpu a i (float4-y xi) (info-max-y-cpu info) vi (make-float4 0.0 -1.0 0.0 0.0))
    ;; near-side boundary
    (apply-collision-cpu a i (info-min-z-cpu info) (float4-z xi) vi (make-float4 0.0 0.0 1.0 0.0))
    ;; far-side boundary
    (apply-collision-cpu a i (float4-z xi) (info-max-z-cpu info) vi (make-float4 0.0 0.0 -1.0 0.0))
    ;; accel limit
    (apply-accel-limit-cpu a i)))

(defun boundary-condition-cpu (x v a info n &key grid-dim block-dim)
  (with-thread-block (grid-dim block-dim)
    (with-valid-index-cpu (i n)
      (%boundary-condition-cpu i x v a info))))

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

(defun %update-acceleration-cpu (i a f rho)
  (symbol-macrolet ((ai   (mem-aref a i))
                    (fi   (mem-aref f i))
                    (rhoi (mem-aref rho i)))
    (setf ai (float4-add-cpu (float4-scale-inverted-cpu fi rhoi)
                             g))))

(defun update-acceleration-cpu (a f rho n &key grid-dim block-dim)
  (with-thread-block (grid-dim block-dim)
    (with-valid-index-cpu (i n)
      (%update-acceleration-cpu i a f rho))))

;; returns dummy integer to avoid __host__ qualifier
(defkernel %update-velocity (int ((i int) (v float4*) (a float4*)))
  (symbol-macrolet ((vi (aref v i))
                    (ai (aref a i)))
    (inc vi (* ai dt)))
  (return 0))

(defkernel update-velocity (void ((v float4*) (a float4*) (n int)))
  (with-valid-index (i n)
    (%update-velocity i v a)))

(defun %update-velocity-cpu (i v a)
  (symbol-macrolet ((vi (mem-aref v i))
                    (ai (mem-aref a i)))
    (float4-incf-cpu vi (float4-scale-cpu ai dt))))

(defun update-velocity-cpu (v a n &key grid-dim block-dim)
  (with-thread-block (grid-dim block-dim)
    (with-valid-index-cpu (i n)
      (%update-velocity-cpu i v a))))

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

(defun %update-position-cpu (i x v)
  (symbol-macrolet ((xi (mem-aref x i))
                    (vi (mem-aref v i)))
    (float4-incf-cpu xi (float4-scale-inverted-cpu (float4-scale-cpu vi dt)
                                                   simscale))))

(defun update-position-cpu (x v n &key grid-dim block-dim)
  (with-thread-block (grid-dim block-dim)
    (with-valid-index-cpu (i n)
      (%update-position-cpu i x v))))


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
  (with-float4-cpu (x y z _) (mem-aref pos i)
    (format nil *sphere-template* x y z)))

(defun output (pos i)
  (let ((n (memory-block-length pos))
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
       do (setf (mem-aref x i) p
                (mem-aref v i) zero))))

(defun run-sph (particles)
  (let ((dev-id 0)
        (n (length particles))
        (grid-dim '(16 1 1))
        (block-dim '(64 1 1)))
    ;; with CUDA context
    (with-cuda-context (dev-id)
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
        (memcpy-host-to-device x v)
        ;; copy neighbor map to device memory
        (memcpy-neighbor-map-host-to-device nbr info)
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

(defun run-sph-cpu (particles)
  (let ((dev-id 0)
        (n (length particles))
        (grid-dim '(16 1 1))
        (block-dim '(64 1 1)))
    ;; with CUDA context
    (with-cuda-context (dev-id)
      ;; with memory blocks
      (with-memory-blocks ((x   'float4 n)
                           (v   'float4 n)
                           (a   'float4 n)
                           (f   'float4 n)
                           (rho 'float  n)
                           (prs 'float  n))
      ;; with neighbor map
      (with-neighbor-map (nbr info box-min box-max delta capacity) 
        ;; apply given initial condition
        (initialize x v particles)
        ;; loop
        (time
         (dotimes (i 100)
           ;; clear neighbor map
           (clear-neighbor-map-cpu nbr info :grid-dim '(23 11 1) :block-dim '(11 1 1))
           ;; update neighbor map with position
           (update-neighbor-map-cpu x nbr info n :grid-dim grid-dim :block-dim '(128 1 1))
           ;; update density
           (update-density-cpu rho x nbr info n :grid-dim grid-dim :block-dim '(384 1 1))
           ;; update pressure
           (update-pressure-cpu prs rho n :grid-dim grid-dim :block-dim block-dim)
           ;; update force
           (update-force-cpu f x v rho prs nbr info n :grid-dim grid-dim :block-dim block-dim)
           ;; update acceleration
           (update-acceleration-cpu a f rho n :grid-dim grid-dim :block-dim block-dim)
           ;; apply boundary condition
           (boundary-condition-cpu x v a info n :grid-dim grid-dim :block-dim block-dim)
           ;; update velocity
           (update-velocity-cpu v a n :grid-dim grid-dim :block-dim block-dim)
           ;; update position
           (update-position-cpu x v n :grid-dim grid-dim :block-dim block-dim)
           ;; synchronize all threads
           (synchronize-context)
           ;; output
           (output x i)
           )))))))

(defun main ()
  (run-sph (initial-condition)))


;;;
;;; Test
;;;

(defparameter test-box-min  (make-float4  0.0  0.0  0.0 0.0))
(defparameter test-box-max  (make-float4 10.0 10.0 10.0 0.0))
(defparameter test-delta    2.0)
(defparameter test-capacity 20)

(defun test-neighbor-map-info1 ()
  (cl-test-more:diag "test-neighbor-map-info1")
  (with-cuda-context (0)
    (let ((info (alloc-neighbor-map-info test-box-min test-box-max test-delta test-capacity)))
      (unwind-protect
           (progn
             (cl-test-more:plan nil)
             (cl-test-more:is (info-min-x-cpu info) 0.0)
             (cl-test-more:is (info-min-y-cpu info) 0.0)
             (cl-test-more:is (info-min-z-cpu info) 0.0)
             (cl-test-more:is (info-max-x-cpu info) 10.0)
             (cl-test-more:is (info-max-y-cpu info) 10.0)
             (cl-test-more:is (info-max-z-cpu info) 10.0)
             (cl-test-more:is (info-delta-cpu info) 2.0)
             (cl-test-more:is (info-capacity-cpu info) 20)
             (cl-test-more:is (info-size-x-cpu info) 7)
             (cl-test-more:is (info-size-y-cpu info) 7)
             (cl-test-more:is (info-size-z-cpu info) 7)
             (cl-test-more:is (info-size-cpu info) 343)
             (cl-test-more:finalize))
        (free-neighbor-map-info info)))))

(defun test-neighbor-map-info2 ()
  (cl-test-more:diag "test-neighbor-map-info2")
  (let ((test-box-max (make-float4 9.0 9.0 9.0 0.0)))
    (with-cuda-context (0)
      (let ((info (alloc-neighbor-map-info test-box-min test-box-max test-delta test-capacity)))
        (unwind-protect
             (progn
               (cl-test-more:plan nil)
               (cl-test-more:is (info-max-x-cpu info) 10.0)
               (cl-test-more:is (info-max-y-cpu info) 10.0)
               (cl-test-more:is (info-max-z-cpu info) 10.0)
               (cl-test-more:finalize))
          (free-neighbor-map-info info))))))

(defun test-neighbor-map-info3 ()
  (cl-test-more:diag "test-neighbor-map-info3")
  (let ((test-box-min  (make-float4   0.0   0.0   0.0 0.0))
        (test-box-max1 (make-float4 -10.0  10.0  10.0 0.0))
        (test-box-max2 (make-float4  10.0 -10.0  10.0 0.0))
        (test-box-max3 (make-float4  10.0  10.0 -10.0 0.0)))
    (with-cuda-context (0)
      (cl-test-more:plan nil)
      (cl-test-more:is-error (alloc-neighbor-map-info test-box-min test-box-max1 test-delta test-capacity)
                             simple-error)
      (cl-test-more:is-error (alloc-neighbor-map-info test-box-min test-box-max2 test-delta test-capacity)
                             simple-error)
      (cl-test-more:is-error (alloc-neighbor-map-info test-box-min test-box-max3 test-delta test-capacity)
                             simple-error)
      (cl-test-more:finalize))))

(defun test-neighbor-map-info4 ()
  (cl-test-more:diag "test-neighbor-map-info4")
  (let ((test-delta1 -1.0)
        (test-delta2  0.0))
    (with-cuda-context (0)
      (cl-test-more:plan nil)
      (cl-test-more:is-error (alloc-neighbor-map-info test-box-min test-box-max test-delta1 test-capacity)
                             simple-error)
      (cl-test-more:is-error (alloc-neighbor-map-info test-box-min test-box-max test-delta2 test-capacity)
                             simple-error)
      (cl-test-more:finalize))))

(defun test-neighbor-map-info5 ()
  (cl-test-more:diag "test-neighbor-map-info5")
  (let ((test-capacity1 -1)
        (test-capacity2 0))
    (with-cuda-context (0)
      (cl-test-more:plan nil)
      (cl-test-more:is-error (alloc-neighbor-map-info test-box-min test-box-max test-delta test-capacity1)
                             simple-error)
      (cl-test-more:is-error (alloc-neighbor-map-info test-box-min test-box-max test-delta test-capacity2)
                             simple-error)
      (cl-test-more:finalize))))

;; kernel function for test-neighbor-map1
(defkernel kernel-test-neighbor-map1 (void ((result int*) (nbr int*) (info float*) (i int) (j int) (k int)))
  (let ((index 0))
    (do-particles-in-cell (p nbr info i j k)
      (set (aref result index) p)
      (set index (+ index 1)))))

(defun kernel-test-neighbor-map1-cpu (result nbr info i j k)
  (let ((index 0))
    (do-particles-in-cell-cpu (p nbr info i j k)
      (setf (mem-aref result index) p)
      (setf index (+ index 1)))))

;; test functions of neighbor map for one cell
(defun test-neighbor-map1 ()
  (cl-test-more:diag "test-neighbor-map1")
  (with-cuda-context (0)
    (with-neighbor-map (nbr info test-box-min test-box-max test-delta test-capacity)
      (cl-test-more:plan nil)
      ;; clear cells and test them
      (clear-particles-in-cell-cpu nbr info 0 0 0)
      (clear-particles-in-cell-cpu nbr info 1 0 0)
      (cl-test-more:is (number-of-particles-in-cell-cpu nbr info 0 0 0) 0)
      (cl-test-more:is (number-of-particles-in-cell-cpu nbr info 1 0 0) 0)
      ;; insert particles into cells and test them
      (insert-particle-in-cell-cpu 0 nbr info 0 0 0)
      (insert-particle-in-cell-cpu 1 nbr info 0 0 0)
      (insert-particle-in-cell-cpu 2 nbr info 1 0 0)
      (cl-test-more:is (number-of-particles-in-cell-cpu nbr info 0 0 0) 2)
      (cl-test-more:is (number-of-particles-in-cell-cpu nbr info 1 0 0) 1)
      (cl-test-more:is (nth-particle-in-cell-cpu 0 nbr info 0 0 0) 0)
      (cl-test-more:is (nth-particle-in-cell-cpu 1 nbr info 0 0 0) 1)
      (cl-test-more:is (nth-particle-in-cell-cpu 0 nbr info 1 0 0) 2)
      (with-memory-blocks ((x 'int 2))
        (kernel-test-neighbor-map1-cpu x nbr info 0 0 0)
        (cl-test-more:is (mem-aref x 0) 0)
        (cl-test-more:is (mem-aref x 1) 1))
      (with-memory-blocks ((x 'int 1))
        (kernel-test-neighbor-map1-cpu x nbr info 1 0 0)
        (cl-test-more:is (mem-aref x 0) 2))
      ;; test to insert particles out of neighbor map
      (clear-particles-in-cell-cpu nbr info -1 0 0)
      (insert-particle-in-cell-cpu 0 nbr info -1 0 0)
      (cl-test-more:is (number-of-particles-in-cell-cpu nbr info -1 0 0) 0)
      (cl-test-more:finalize))))

;; kernel function for test-neighbor-map2
(defkernel kernel-test-neighbor-map2 (void ((result int*) (nbr int*) (info float*) (x float) (y float) (z float)))
  (let ((index 0))
    (do-neighbor-particles (p nbr info (float4 x y z 0.0))
      (set (aref result index) p)
      (set index (+ index 1)))))

(defun kernel-test-neighbor-map2-cpu (result nbr info x y z)
  (let ((index 0))
    (do-neighbor-particles-cpu (p nbr info (make-float4 x y z 0.0))
      (setf (mem-aref result index) p)
      (setf index (+ index 1)))))

;; test functions of neighbor map for cells
(defun test-neighbor-map2 ()
  (cl-test-more:diag "test-neighbor-map2")
  (with-cuda-context (0)
    (with-memory-blocks ((xs 'float4 3))
    (with-neighbor-map (nbr info test-box-min test-box-max test-delta test-capacity)
      (cl-test-more:plan nil)
      ;; initialize
      (setf (mem-aref xs 0) (make-float4 1.0 1.0 1.0 0.0)
            (mem-aref xs 1) (make-float4 1.0 1.0 1.0 0.0)
            (mem-aref xs 2) (make-float4 3.0 1.0 1.0 0.0))
      ;; clear neighbor map and test it
      (clear-neighbor-map-cpu nbr info :grid-dim '(7 7 1) :block-dim '(7 1 1))
      (cl-test-more:is (number-of-particles-in-cell-cpu nbr info 0 0 0) 0)
      (cl-test-more:is (number-of-particles-in-cell-cpu nbr info 1 0 0) 0)
      ;; update neighbor map and test it
      (let ((n (memory-block-length xs)))
        (update-neighbor-map-cpu xs nbr info n :grid-dim '(1 1 1) :block-dim '(32 1 1))
        (cl-test-more:is (number-of-particles-in-cell-cpu nbr info 1 1 1) 2)
        (cl-test-more:is (number-of-particles-in-cell-cpu nbr info 2 1 1) 1))
      (with-memory-blocks ((x 'int 3))
        (kernel-test-neighbor-map2-cpu x nbr info 1.0 1.0 1.0)
        (cl-test-more:is (mem-aref x 0) 0)
        (cl-test-more:is (mem-aref x 1) 1)
        (cl-test-more:is (mem-aref x 2) 2))
      (with-memory-blocks ((x 'int 3))
        (kernel-test-neighbor-map2-cpu x nbr info 3.0 1.0 1.0)
        (cl-test-more:is (mem-aref x 0) 0)
        (cl-test-more:is (mem-aref x 1) 1)
        (cl-test-more:is (mem-aref x 2) 2))
      (with-memory-blocks ((x 'int 1))
        (kernel-test-neighbor-map2-cpu x nbr info 5.0 1.0 1.0)
        (cl-test-more:is (mem-aref x 0) 2))
      (cl-test-more:finalize)))))

(defun test-neighbor-map2-gpu ()
  (cl-test-more:diag "test-neighbor-map2-gpu")
  (with-cuda-context (0)
    (with-memory-blocks ((xs 'float4 3))
    (with-neighbor-map (nbr info test-box-min test-box-max test-delta test-capacity)
      (cl-test-more:plan nil)
      ;; initialize
      (setf (mem-aref xs 0) (make-float4 1.0 1.0 1.0 0.0)
            (mem-aref xs 1) (make-float4 1.0 1.0 1.0 0.0)
            (mem-aref xs 2) (make-float4 3.0 1.0 1.0 0.0))
      ;; copy required data from host to device
      (memcpy-host-to-device xs)
      (memcpy-neighbor-map-host-to-device nbr info)
      ;; clear neighbor map
      (clear-neighbor-map nbr info :grid-dim '(7 7 1) :block-dim '(7 1 1))
      ;; update neighbor map
      (let ((n (memory-block-length xs)))
        (update-neighbor-map xs nbr info n :grid-dim '(1 1 1) :block-dim '(32 1 1)))
      ;; test contained particles in neighbor map
      (with-memory-blocks ((x 'int 3))
        (kernel-test-neighbor-map2 x nbr info 1.0 1.0 1.0)
        (memcpy-device-to-host x)
        (cl-test-more:is (mem-aref x 0) 0)
        (cl-test-more:is (mem-aref x 1) 1)
        (cl-test-more:is (mem-aref x 2) 2))
      (with-memory-blocks ((x 'int 3))
        (kernel-test-neighbor-map2 x nbr info 3.0 1.0 1.0)
        (memcpy-device-to-host x)
        (cl-test-more:is (mem-aref x 0) 0)
        (cl-test-more:is (mem-aref x 1) 1)
        (cl-test-more:is (mem-aref x 2) 2))
      (with-memory-blocks ((x 'int 1))
        (kernel-test-neighbor-map2 x nbr info 5.0 1.0 1.0)
        (memcpy-device-to-host x)
        (cl-test-more:is (mem-aref x 0) 2))
      (cl-test-more:finalize)))))

(defun test ()
  (test-neighbor-map-info1)
  (test-neighbor-map-info2)
  (test-neighbor-map-info3)
  (test-neighbor-map-info4)
  (test-neighbor-map-info5)
  (test-neighbor-map1)
  (test-neighbor-map2)
  (test-neighbor-map2-gpu))
