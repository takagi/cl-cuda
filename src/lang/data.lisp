#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda.lang.data
  (:use :cl)
  (:export ;; Symbol
           :cl-cuda-symbol
           :cl-cuda-symbol-p
           ;; Bool
           :cl-cuda-bool-p
           ;; Int
           :cl-cuda-int-p
           ;; Float
           :cl-cuda-float-p
           ;; Double
           :cl-cuda-double-p
           ;; Float3
           :float3
           :make-float3
           :float3-x
           :float3-y
           :float3-z
           :float3-p
           :float3-=
           :with-float3
           ;; Float4
           :float4
           :make-float4
           :float4-x
           :float4-y
           :float4-z
           :float4-w
           :float4-p
           :float4-=
           :with-float4
           ;; Double3
           :double3
           :make-double3
           :double3-x
           :double3-y
           :double3-z
           :double3-p
           :double3-=
           :with-double3
           ;; Double4
           :double4
           :make-double4
           :double4-x
           :double4-y
           :double4-z
           :double4-w
           :double4-p
           :double4-=
           :with-double4)
  (:import-from :alexandria
                :once-only))
(in-package :cl-cuda.lang.data)


;;;
;;; Symbol
;;;

(deftype cl-cuda-symbol ()
  `(satisfies cl-cuda-symbol-p))

(defun cl-cuda-symbol-p (object)
  (symbolp object))


;;;
;;; Bool
;;;

(defun cl-cuda-bool-p (object)
  (typep object 'boolean))


;;;
;;; Int
;;;

(defun cl-cuda-int-p (object)
  (integerp object))


;;;
;;; Float
;;;

(defun cl-cuda-float-p (object)
  (typep object 'single-float))


;;;
;;; Double
;;;

(defun cl-cuda-double-p (object)
  (typep object 'double-float))


;;;
;;; Float3
;;;

(defstruct (float3 (:constructor make-float3 (x y z)))
  (x 0.0 :type single-float)
  (y 0.0 :type single-float)
  (z 0.0 :type single-float))

(defun float3-= (a b)
  (and (= (float3-x a) (float3-x b))
       (= (float3-y a) (float3-y b))
       (= (float3-z a) (float3-z b))))

(cffi:defcstruct (float3 :class float3-c)
  (x :float)
  (y :float)
  (z :float))

(defmacro with-float3 ((x y z) value &body body)
  (once-only (value)
    `(let ((,x (float3-x ,value))
           (,y (float3-y ,value))
           (,z (float3-z ,value)))
       (declare (ignorable ,x ,y ,z))
       ,@body)))

(defmethod cffi:translate-into-foreign-memory ((value float3)
                                               (type float3-c)
                                               ptr)
  (cffi:with-foreign-slots ((x y z) ptr (:struct float3))
    (setf x (float3-x value)
          y (float3-y value)
          z (float3-z value))))

(defmethod cffi:translate-from-foreign (value (type float3-c))
  (cffi:with-foreign-slots ((x y z) value (:struct float3))
    (make-float3 x y z)))


;;;
;;; Float4
;;;

(defstruct (float4 (:constructor make-float4 (x y z w)))
  (x 0.0 :type single-float)
  (y 0.0 :type single-float)
  (z 0.0 :type single-float)
  (w 0.0 :type single-float))

(defun float4-= (a b)
  (and (= (float4-x a) (float4-x b))
       (= (float4-y a) (float4-y b))
       (= (float4-z a) (float4-z b))
       (= (float4-w a) (float4-w b))))

(cffi:defcstruct (float4 :class float4-c)
  (x :float)
  (y :float)
  (z :float)
  (w :float))

(defmacro with-float4 ((x y z w) value &body body)
  (once-only (value)
    `(let ((,x (float4-x ,value))
           (,y (float4-y ,value))
           (,z (float4-z ,value))
           (,w (float4-w ,value)))
       (declare (ignorable ,x ,y ,z ,w))
       ,@body)))

(defmethod cffi:translate-into-foreign-memory ((value float4)
                                               (type float4-c)
                                               ptr)
  (cffi:with-foreign-slots ((x y z w) ptr (:struct float4))
    (setf x (float4-x value)
          y (float4-y value)
          z (float4-z value)
          w (float4-w value))))

(defmethod cffi:translate-from-foreign (value (type float4-c))
  (cffi:with-foreign-slots ((x y z w) value (:struct float4))
    (make-float4 x y z w)))


;;;
;;; Double3
;;;

(defstruct (double3 (:constructor make-double3 (x y z)))
  (x 0.0d0 :type double-float)
  (y 0.0d0 :type double-float)
  (z 0.0d0 :type double-float))

(defun double3-= (a b)
  (and (= (double3-x a) (double3-x b))
       (= (double3-y a) (double3-y b))
       (= (double3-z a) (double3-z b))))

(cffi:defcstruct (double3 :class double3-c)
  (x :double)
  (y :double)
  (z :double))

(defmacro with-double3 ((x y z) value &body body)
  (once-only (value)
    `(let ((,x (double3-x ,value))
           (,y (double3-y ,value))
           (,z (double3-z ,value)))
       (declare (ignorable ,x ,y ,z))
       ,@body)))

(defmethod cffi:translate-into-foreign-memory ((value double3)
                                               (type double3-c)
                                               ptr)
  (cffi:with-foreign-slots ((x y z) ptr (:struct double3))
    (setf x (double3-x value)
          y (double3-y value)
          z (double3-z value))))

(defmethod cffi:translate-from-foreign (value (type double3-c))
  (cffi:with-foreign-slots ((x y z) value (:struct double3))
    (make-double3 x y z)))


;;;
;;; Double4
;;;

(defstruct (double4 (:constructor make-double4 (x y z w)))
  (x 0.0d0 :type double-float)
  (y 0.0d0 :type double-float)
  (z 0.0d0 :type double-float)
  (w 0.0d0 :type double-float))

(defun double4-= (a b)
  (and (= (double4-x a) (double4-x b))
       (= (double4-y a) (double4-y b))
       (= (double4-z a) (double4-z b))
       (= (double4-w a) (double4-w b))))

(cffi:defcstruct (double4 :class double4-c)
  (x :double)
  (y :double)
  (z :double)
  (w :double))

(defmacro with-double4 ((x y z w) value &body body)
  (once-only (value)
    `(let ((,x (double4-x ,value))
           (,y (double4-y ,value))
           (,z (double4-z ,value))
           (,w (double4-w ,value)))
       (declare (ignorable ,x ,y ,z ,w))
       ,@body)))

(defmethod cffi:translate-into-foreign-memory ((value double4)
                                               (type double4-c)
                                               ptr)
  (cffi:with-foreign-slots ((x y z w) ptr (:struct double4))
    (setf x (double4-x value)
          y (double4-y value)
          z (double4-z value)
          w (double4-w value))))

(defmethod cffi:translate-from-foreign (value (type double3-c))
  (cffi:with-foreign-slots ((x y z w) value (:struct double4))
    (make-double4 x y z w)))


;;;
;;; CURAND State XORWOW
;;;

(cffi:defcstruct curand-state-xorwow
  (d :unsigned-int)
  (v :unsigned-int :count 5)
  (boxmuller-flag :int)
  (boxmuller-flag-double :int)
  (boxmuller-extra :float)
  (boxmuller-extra-double :double))
