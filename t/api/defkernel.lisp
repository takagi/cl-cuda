#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-test.api.defkernel
  (:use :cl :cl-test-more
        :cl-cuda.api.defkernel
        :cl-cuda.api.context
        :cl-cuda.api.memory
        :cl-cuda.lang
        :cl-cuda.driver-api)
  (:import-from :cl-cuda.api.defkernel
                :with-launching-arguments))
(in-package :cl-cuda-test.api.defkernel)

(plan nil)


;;;
;;; test WITH-LAUNCHING-ARGUMENTS macro
;;;

(diag "WITH-LAUNCHING-ARGUMENTS")

(is-expand
  (with-launching-arguments (kargs ((x int) (y float3) (a float3*)))
    (do-something))
  (cffi:with-foreign-objects ((x-ptr ':int)
                              (y-ptr '(:struct float3))
                              (a-ptr 'cu-device-ptr))
    (setf (cffi:mem-ref x-ptr ':int) x)
    (setf (cffi:mem-ref y-ptr '(:struct float3)) y)
    (setf (cffi:mem-ref a-ptr 'cu-device-ptr)
          (if (memory-block-p a)
              (memory-block-device-ptr a)
              a))
    (cffi:with-foreign-object (kargs :pointer 3)
      (setf (cffi:mem-aref kargs :pointer 0) x-ptr)
      (setf (cffi:mem-aref kargs :pointer 1) y-ptr)
      (setf (cffi:mem-aref kargs :pointer 2) a-ptr)
      (do-something)))
  "basic case 1")


;;;
;;; test DEFKERNEL macro
;;;

(diag "DEFKERNEL")

;; test "let1" kernel
(defkernel let1 (void ())
  (let ((i 0))
    (return))
  (let ((i 0))))

(with-cuda-context (0)
  (ok (let1 :grid-dim (list 1 1 1)
            :block-dim (list 1 1 1))
      "basic case 1"))

;; test "use-one" kernel
(defkernel use-one (void ())
  (let ((i (one)))
    (return)))

(defkernel one (int ())
  (return 1))

(with-cuda-context (0)
  (ok (use-one :grid-dim (list 1 1 1)
               :block-dim (list 1 1 1))
      "basic case 2"))

;; test "argument" kernel
(defkernel argument (void ((i int) (j float3)))
  (return))

(with-cuda-context (0)
  (ok (argument 1 (make-float3 0.0 0.0 0.0) :grid-dim (list 1 1 1)
                                            :block-dim (list 1 1 1))
      "basic case 3"))

;; test "kernel-bool" kernel
(defkernel kernel-bool (void ((a bool*)))
  (set (aref a 0) t)
  (set (aref a 1) nil)
  (return))

(with-cuda-context (0)
  (with-memory-blocks ((a 'bool 2))
    (setf (memory-block-aref a 0) nil
          (memory-block-aref a 1) t)
    (sync-memory-block a :host-to-device)
    (ok (kernel-bool a :grid-dim '(1 1 1) :block-dim '(1 1 1))
        "basic case 4")
    (sync-memory-block a :device-to-host)
    (is (memory-block-aref a 0) t
        "basic case 5")
    (is (memory-block-aref a 1) nil
        "basic case 6")))

;; test "kernel-float3" kernel
(defkernel kernel-float3 (void ((a float*) (x float3)))
  (set (aref a 0) (+ (float3-x x) (float3-y x) (float3-z x))))

(let ((x (make-float3 1.0 2.0 3.0)))
  (with-cuda-context (0)
    (with-memory-blocks ((a 'float 1))
      (setf (memory-block-aref a 0) 1.0)
      (sync-memory-block a :host-to-device)
      (ok (kernel-float3 a x :grid-dim '(1 1 1) :block-dim '(1 1 1))
          "basic case 7")
      (sync-memory-block a :device-to-host)
      (is (memory-block-aref a 0) 6.0
          "basic case 8"))))

;; test DO statement
(defkernel test-do-kernel (void ((x int*)))
  (do ((i 0 (+ i 1)))
      ((> i 15))
    (set (aref x 0) (+ (aref x 0) 1))))

(with-cuda-context (0)
  (with-memory-blocks ((x 'int 1))
    (setf (memory-block-aref x 0) 0)
    (sync-memory-block x :host-to-device)
    (ok (test-do-kernel x :grid-dim '(1 1 1) :block-dim '(1 1 1))
        "basic case 9")
    (sync-memory-block x :device-to-host)
    (is (memory-block-aref x 0) 16
        "basic case 10")))

;; test multi-argument arithmetic
(defkernel test-add (void ((x int*)))
  (set (aref x 0) (+ 1 1 1)))

(with-cuda-context (0)
  (with-memory-blocks ((x 'int 1))
    (setf (memory-block-aref x 0) 0)
      (sync-memory-block x :host-to-device)
      (ok (test-add x :grid-dim '(1 1 1) :block-dim '(1 1 1))
          "basic case 11")
      (sync-memory-block x :device-to-host)
      (is (memory-block-aref x 0) 3
          "basic case 12")))

;; test atomic function
(defkernel test-atomic-add (void ((x int*)))
  (atomic-add (pointer (aref x 0)) 1))

(defkernel test-no-atomic-add (void ((x int*)))
  (set (aref x 0) (+ (aref x 0) 1)))

(with-cuda-context (0)
  (with-memory-blocks ((x 'int 1)
                       (y 'int 1))
    (setf (memory-block-aref x 0) 0
          (memory-block-aref y 0) 0)
    (sync-memory-block x :host-to-device)
    (sync-memory-block y :host-to-device)
    (ok (test-atomic-add x :grid-dim '(1 1 1) :block-dim '(256 1 1))
        "basic case 13")
    (ok (test-no-atomic-add y :grid-dim '(1 1 1) :block-dim '(256 1 1))
        "basic case 14")
    (sync-memory-block x :device-to-host)
    (sync-memory-block y :host-to-device)
    (is (memory-block-aref x 0) 256
        "basic case 15")
    (isnt (memory-block-aref y 0) 256
          "basic case 16")))

;; test built-in vector type
(defkernel test-float3-add (void ((x float3*)))
  (set (aref x 0) (+ (aref x 0) (float3 1.0 1.0 1.0))))

(with-cuda-context (0)
  (with-memory-blocks ((x 'float3 1))
    (setf (memory-block-aref x 0) (make-float3 0.0 0.0 0.0))
    (sync-memory-block x :host-to-device)
    (ok (test-float3-add x :grid-dim '(1 1 1) :block-dim '(1 1 1))
        "basic case 17")
    (sync-memory-block x :device-to-host)
    (is (memory-block-aref x 0) (make-float3 1.0 1.0 1.0) :test #'float3-=
        "basic case 18")))

;; test kernel macro
(defkernelmacro when (test &body forms)
  `(if ,test
       (progn ,@forms)))

(defkernel test-when (void ())
  (when t (return))
  (return))

(with-cuda-context (0)
  (ok (test-when :grid-dim '(1 1 1) :block-dim '(1 1 1))
      "basic case 19"))

;; test kernel symbol macro
(defkernel-symbol-macro x 1)

(defkernel test-symbol-macro (void ((ret int*)))
  (set (aref ret 0) x)
  (return))

(with-cuda-context (0)
  (with-memory-blocks ((x 'int 1))
    (setf (memory-block-aref x 0) 0)
    (sync-memory-block x :host-to-device)
    (ok (test-symbol-macro x :grid-dim '(1 1 1) :block-dim '(1 1 1))
        "basic case 20")
    (sync-memory-block x :device-to-host)
    (is (memory-block-aref x 0) 1
        "basic case 21")))


;;;
;;; test EXPAND-MACRO function
;;;

(diag "EXPAND-MACRO")

(defkernelmacro foo (x)
  `(return ,x))

(defkernelmacro bar (x)
  `(foo ,x))

(defmacro is-values (got expected &rest args)
  `(is (multiple-value-list ,got) ,expected ,@args))

(is-values (expand-macro-1 '(foo 1)) '((return 1) t))
(is-values (expand-macro-1 '(bar 1)) '((foo 1) t))
(is-values (expand-macro-1 '(baz 1)) '((baz 1) nil))
(is-error (expand-macro-1 '(foo)) error)

(is-values (expand-macro '(foo 1)) '((return 1) t))
(is-values (expand-macro '(bar 1)) '((return 1) t))
(is-values (expand-macro '(baz 1)) '((baz 1) nil))
(is-error (expand-macro '(foo)) error)


(finalize)
