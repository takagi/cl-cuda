#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-test.api.memory
  (:use :cl :prove
        :cl-cuda.api.memory
        :cl-cuda.api.context
        :cl-cuda.lang))
(in-package :cl-cuda-test.api.memory)

(plan nil)


;;;
;;; test ALLOC-MEMORY-BLOCK / FREE-MEMORY-BLOCK function
;;;

(diag "ALLOC-MEMORY-BLOCK / FREE-MEMORY-BLOCK")

(with-cuda (0)
  (let (memory-block)
    (ok (setf memory-block (alloc-memory-block 'int 1024))
        "basic case 1")
    (free-memory-block memory-block)))

(with-cuda (0)
  (is-error (alloc-memory-block 'void 1024) simple-error
            "TYPE which is a void type"))

(with-cuda (0)
  (is-error (alloc-memory-block 'int  (* 1024 1024 1024 1024))
            simple-error
            "SIZE which specifies memory larger than available on the gpu"))

(with-cuda (0)
  (is-error (alloc-memory-block 'int 0)
            simple-error
            "SIZE which is zero"))

(with-cuda (0)
  (is-error (alloc-memory-block 'int -1)
            type-error
            "SIZE which is negative"))


;;;
;;; test MEMORY-BLOCK-DEVICE-PTR function
;;;

(diag "MEMORY-BLOCK-DEVICE-PTR")

(with-cuda (0)
  (with-memory-block (a 'int 1)
    (ok (memory-block-device-ptr a)
        "basic case 1")))


;;;
;;; test MEMORY-BLOCK-HOST-PTR function
;;;

(diag "MEMORY-BLOCK-HOST-PTR")

(with-cuda (0)
  (with-memory-block (a 'int 1)
    (ok (memory-block-host-ptr a)
        "basic case 1")))
  

;;;
;;; test MEMORY-BLOCK-TYPE function
;;;

(diag "MEMORY-BLOCK-TYPE")

(with-cuda (0)
  (with-memory-block (a 'int 1)
    (is (memory-block-type a) 'int
        "basic case 1")))


;;;
;;; test MEMORY-BLOCK-SIZE function
;;;

(diag "MEMORY-BLOCK-SIZE")

(with-cuda (0)
  (with-memory-block (a 'int 1)
    (is (memory-block-size a) 1
        "basic case 1")))


;;;
;;; test MEMORY-BLOCK-AREF function
;;;

(diag "MEMORY-BLOCK-AREF")

(with-cuda (0)
  (with-memory-block (a 'int 1)
    (setf (memory-block-aref a 0) 1)
    (is (memory-block-aref a 0) 1
        "basic case 1")))

(with-cuda (0)
  (with-memory-block (a 'float3 1)
    (setf (memory-block-aref a 0) (make-float3 1.0 1.0 1.0))
    (is (memory-block-aref a 0) (make-float3 1.0 1.0 1.0)
        :test #'float3-=
        "basic case 2")))


;;;
;;; test SYNC-MEMORY-BLOCK function
;;;

(diag "SYNC-MEMORY-BLOCK")

(with-cuda (0)
  (with-memory-block (a 'int 1)
    (setf (memory-block-aref a 0) 1)
    (sync-memory-block a :host-to-device)
    (setf (memory-block-aref a 0) 2)
    (sync-memory-block a :device-to-host)
    (is (memory-block-aref a 0) 1
        "basic case 1")))


(finalize)
