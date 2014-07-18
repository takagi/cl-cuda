#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda.lang.type
  (:use :cl
        :cl-cuda.driver-api
        :cl-cuda.lang.data)
  (:export ;; Cl-cuda types
           :void
           :bool
           :int
           :float
           :double
           :curand-state-xorwow
           :float3
           :float4
           :double3
           :double4
           :bool*
           :int*
           :float*
           :double*
           :curand-state-xorwow*
           :float3*
           :float4*
           :double3*
           :double4*
           ;; Type
           :cl-cuda-type
           :cl-cuda-type-p
           :cffi-type
           :cffi-type-size
           :cuda-type
           ;; Scalar type
           :scalar-type-p
           ;; Structure type
           :structure-type-p
           ;; Structure accessor
           :structure-accessor-p
           :structure-from-accessor
           :structure-accessor-cuda-accessor
           :structure-accessor-return-type
           ;; Array type
           :array-type-p
           :array-type-base
           :array-type-dimension
           :array-type)
  (:import-from :alexandria
                :format-symbol))
(in-package :cl-cuda.lang.type)


;;;
;;; Type
;;;

(deftype cl-cuda-type ()
  `(satisfies cl-cuda-type-p))

(defun cl-cuda-type-p (object)
  (or (scalar-type-p object)
      (structure-type-p object)
      (array-type-p object)))

(defun cffi-type (type)
  (cond
    ((scalar-type-p type) (scalar-cffi-type type))
    ((structure-type-p type) (structure-cffi-type type))
    ((array-type-p type) (array-cffi-type type))
    (t (error "The value ~S is an invalid type." type))))

(defun cffi-type-size (type)
  (cond
    ((scalar-type-p type) (scalar-cffi-type-size type))
    ((structure-type-p type) (structure-cffi-type-size type))
    ((array-type-p type) (array-cffi-type-size type))
    (t (error "The value ~S is an invalid type." type))))

(defun cuda-type (type)
  (cond
    ((scalar-type-p type) (scalar-cuda-type type))
    ((structure-type-p type) (structure-cuda-type type))
    ((array-type-p type) (array-cuda-type type))
    (t (error "The value ~S is an invalid type." type))))


;;;
;;; Scalar type
;;;

(defparameter +scalar-types+
  '((void :void "void")
    (bool (:boolean :int8) "bool")
    (int :int "int")
    (float :float "float")
    (double :double "double")
    (curand-state-xorwow (:struct curand-state-xorwow)
                         "curandStateXORWOW_t")))

(defun scalar-type-p (object)
  (and (assoc object +scalar-types+)
       t))

(defun scalar-cffi-type (type)
  (unless (scalar-type-p type)
    (error "The vaue ~S is an invalid type." type))
  (cadr (assoc type +scalar-types+)))

(defun scalar-cffi-type-size (type)
  (cffi:foreign-type-size (scalar-cffi-type type)))

(defun scalar-cuda-type (type)
  (unless (scalar-type-p type)
    (error "The vaue ~S is an invalid type." type))
  (caddr (assoc type +scalar-types+)))


;;;
;;; Structure type
;;;

(defparameter +structure-table+
  '((float3 "float3" ((float3-x "x" float)
                      (float3-y "y" float)
                      (float3-z "z" float)))
    (float4 "float4" ((float4-x "x" float)
                      (float4-y "y" float)
                      (float4-z "z" float)
                      (float4-w "w" float)))
    (double3 "double3" ((double3-x "x" double)
                        (double3-y "y" double)
                        (double3-z "z" double)))
    (double4 "double4" ((double4-x "x" double)
                        (double4-y "y" double)
                        (double4-z "z" double)
                        (double4-w "w" double)))))

(defparameter +structure-types+
  (mapcar #'car +structure-table+))

(defun structure-type-p (object)
  (and (member object +structure-types+)
       t))

(defun structure-cffi-type (type)
  (unless (structure-type-p type)
    (error "The vaue ~S is an invalid type." type))
  `(:struct ,type))

(defun structure-cffi-type-size (type)
  (cffi:foreign-type-size (structure-cffi-type type)))

(defun structure-cuda-type (type)
  (unless (structure-type-p type)
    (error "The vaue ~S is an invalid type." type))
  (cadr (assoc type +structure-table+)))

(defun structure-accessors (type)
  (unless (structure-type-p type)
    (error "The vaue ~S is an invalid type." type))
  (caddr (assoc type +structure-table+)))


;;;
;;; Structure type - accessor
;;;

(defparameter +accessor->structure+
  (loop for structure in +structure-types+
     append (loop for (accessor nil nil) in (structure-accessors structure)
               collect (list accessor structure))))

(defun %structure-from-accessor (accessor)
  (cadr (assoc accessor +accessor->structure+)))

(defun structure-accessor-p (accessor)
  (and (%structure-from-accessor accessor)
       t))

(defun structure-from-accessor (accessor)
  (or (%structure-from-accessor accessor)
      (error "The value ~S is not a structure accessor." accessor)))

(defun structure-accessor-cuda-accessor (accessor)
  (let ((structure (structure-from-accessor accessor)))
    (second (assoc accessor (structure-accessors structure)))))

(defun structure-accessor-return-type (accessor)
  (let ((structure (structure-from-accessor accessor)))
    (third (assoc accessor (structure-accessors structure)))))


;;;
;;; Array type
;;;

(defparameter +array-type-regex+
  "^([^\\*]+)(\\*+)$")

(defun array-type-p (object)
  (when (symbolp object)
    (let ((package (symbol-package object))
          (object-string (princ-to-string object)))
      (cl-ppcre:register-groups-bind (base-string nil)
          (+array-type-regex+ object-string)
        (let ((base (intern (string base-string) package)))
          (cl-cuda-type-p base))))))

(defun array-type-base (type)
  (unless (array-type-p type)
    (error "The value ~S is an invalid type." type))
  (let ((type-string (princ-to-string type)))
    (cl-ppcre:register-groups-bind (base-string nil)
        (+array-type-regex+ type-string)
      (intern (string base-string) 'cl-cuda.lang.type))))

(defun array-type-stars (type)
  (unless (array-type-p type)
    (error "The value ~S is an invalid type." type))
  (let ((type-string (princ-to-string type)))
    (cl-ppcre:register-groups-bind (_ stars-string)
        (+array-type-regex+ type-string)
      (declare (ignore _))
      (intern (string stars-string) 'cl-cuda.lang.type))))

(defun array-type-dimension (type)
  (length (princ-to-string (array-type-stars type))))

(defun array-cffi-type (type)
  (unless (array-type-p type)
    (error "The value ~S is an invalid type." type))
  'cu-device-ptr)

(defun array-cffi-type-size (type)
  (cffi:foreign-type-size (array-cffi-type type)))

(defun array-cuda-type (type)
  (let ((base (array-type-base type))
        (stars (array-type-stars type)))
    (format nil "~A~A" (cuda-type base) stars)))

(defun array-type (type dimension)
  (unless (and (cl-cuda-type-p type)
               (not (array-type-p type)))
    (error "The value ~S is an invalid type." type))
  (let ((stars (loop repeat dimension collect #\*)))
    (format-symbol 'cl-cuda.lang.type "~A~{~A~}" type stars)))
