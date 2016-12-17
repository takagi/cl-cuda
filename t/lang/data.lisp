#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-test.lang.data
  (:use :cl :prove
        :cl-cuda.lang.data
        :cl-cuda.lang.type))
(in-package :cl-cuda-test.lang.data)

(plan nil)


;;;
;;; test Float3
;;;

(diag "Float3")

(subtest "float3 foreign translation"

  (let ((cffi-type (cffi-type 'float3)))
    (cffi:with-foreign-object (x cffi-type)
      (setf (cffi:mem-ref x cffi-type) (make-float3 0.0 1.0 2.0))
      (with-float3 (x y z) (cffi:mem-ref x cffi-type)
        (is x 0.0)
        (is y 1.0)
        (is z 2.0)))))

(subtest "with-float3"

  (with-float3 (x y z) (make-float3 0.0 1.0 2.0)
    (is x 0.0)
    (is y 1.0)
    (is z 2.0)))


;;
;; Float4

(subtest "with-float4"

  (with-float4 (x y z w) (make-float4 0.0 1.0 2.0 3.0)
    (is x 0.0)
    (is y 1.0)
    (is z 2.0)
    (is w 3.0)))


;;
;; Double3

(subtest "with-double3"

  (with-double3 (x y z) (make-double3 0d0 1d0 2d0)
    (is x 0d0)
    (is y 1d0)
    (is z 2d0)))


;;
;; Double4

(subtest "with-double4"

  (with-double4 (x y z w) (make-double4 0d0 1d0 2d0 3d0)
    (is x 0d0)
    (is y 1d0)
    (is z 2d0)
    (is w 3d0)))


(finalize)
