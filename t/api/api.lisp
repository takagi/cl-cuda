#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-cuda-test.api)

(plan nil)


;;;
;;; test timer
;;;

(diag "test timer")

(let (timer)
  (with-cuda-context (0)
    (setf timer (create-timer))
    (start-timer timer)
    (format t "elapsed time: ~A~%" (get-elapsed-time timer))
    (stop-and-synchronize-timer timer)
    (destroy-timer timer)
    (ok (null (cl-cuda.api::timer-object-start-event timer)))
    (ok (null (cl-cuda.api::timer-object-stop-event  timer)))))

(with-cuda-context (0)
  (with-timer (timer)
    (start-timer timer)
    (format t "elapsed time: ~A~%" (get-elapsed-time timer))
    (stop-and-synchronize-timer timer))
  (ok t))


;;;
;;; test memory blocks
;;;

(diag "test memory blocks")

;; test alloc-memory-block/free-memory-block
(let ((dev-id 0))
  (with-cuda-context (dev-id)
    (let (blk)
      (ok (setf blk (cl-cuda.api::alloc-memory-block 'int 1024)))
      (cl-cuda.api::free-memory-block blk))
    (is-error (cl-cuda.api::alloc-memory-block 'void 1024             ) simple-error)
    (is-error (cl-cuda.api::alloc-memory-block 'int* 1024             ) simple-error)
    (is-error (cl-cuda.api::alloc-memory-block 'int  (* 1024 1024 256)) simple-error)
    (is-error (cl-cuda.api::alloc-memory-block 'int  0                ) simple-error)
    (is-error (cl-cuda.api::alloc-memory-block 'int  -1               ) type-error)))

;; test selectors of memory-block
(let ((dev-id 0))
  (with-cuda-context (dev-id)
    (with-memory-blocks ((blk 'int 1024))
      (ok       (cl-cuda.api::memory-block-cffi-ptr blk))
      (ok       (cl-cuda.api::memory-block-device-ptr blk))
      (cl-cuda.api::with-memory-block-device-ptr (device-ptr blk)
        (ok device-ptr))
      (is       (cl-cuda.api::memory-block-interop-p blk) nil)
      (is-error (cl-cuda.api::memory-block-vertex-buffer-object blk) simple-error)
      (is-error (cl-cuda.api::memory-block-graphic-resource-ptr blk) simple-error)
      (is       (cl-cuda.api::memory-block-type blk) 'int)
      (is       (cl-cuda.api::memory-block-cffi-type blk) :int)
      (is       (cl-cuda.api::memory-block-length blk) 1024)
      (is       (cl-cuda.api::memory-block-bytes blk) (* 1024 4))
      (is       (cl-cuda.api::memory-block-element-bytes blk) 4))))

;; test setf functions of memory-block
(let ((dev-id 0))
  (with-cuda-context (dev-id)
    ;; bool array
    (with-memory-blocks ((x 'bool 2))
      (setf (mem-aref x 0) t
            (mem-aref x 1) nil)
      (is (mem-aref x 0) t)
      (is (mem-aref x 1) nil))
    ;; int array
    (with-memory-blocks ((x 'int 1))
      (setf (mem-aref x 0) 1)
      (is (mem-aref x 0) 1))
    ;; float array
    (with-memory-blocks ((x 'float 1))
      (setf (mem-aref x 0) 1.0)
      (is (mem-aref x 0) 1.0))
    ;; float3 array
    (with-memory-blocks ((x 'float3 1))
      (setf (mem-aref x 0) (make-float3 1.0 1.0 1.0))
      (is (mem-aref x 0) (make-float3 1.0 1.0 1.0) :test #'float3-=))
    ;; float4 array
    (with-memory-blocks ((x 'float4 1))
      (setf (mem-aref x 0) (make-float4 1.0 1.0 1.0 1.0))
      (is (mem-aref x 0) (make-float4 1.0 1.0 1.0 1.0) :test #'float4-=))
    ;; error cases
    (with-memory-blocks ((x 'int 1))
      (is-error (mem-aref x -1) simple-error)
      (is-error (setf (mem-aref x -1) 0) simple-error)
      (is-error (mem-aref x 1) simple-error)
      (is-error (setf (mem-aref x 1) 0) simple-error))))

;; test set statement on memory-block
(defkernel test-memcpy (void ((x int*) (y float*) (z float3*)))
  (set (aref x 0) (+ (aref x 0) 1))
  (set (aref y 0) (+ (aref y 0) 1.0))
  (set (float3-x (aref z 0)) (+ (float3-x (aref z 0)) 1.0)) ; vector math wanted
  (set (float3-y (aref z 0)) (+ (float3-y (aref z 0)) 1.0))
  (set (float3-z (aref z 0)) (+ (float3-z (aref z 0)) 1.0)))

(let ((dev-id 0))
  (with-cuda-context (dev-id)
    (with-memory-blocks ((x 'int    1)
                         (y 'float  1)
                         (z 'float3 1))
      (setf (mem-aref x 0) 1)
      (setf (mem-aref y 0) 1.0)
      (setf (mem-aref z 0) (make-float3 1.0 1.0 1.0))
      (memcpy-host-to-device x y z)
      (test-memcpy x y z :grid-dim '(1 1 1)
                         :block-dim '(1 1 1))
      (memcpy-device-to-host x y z)
      (is (mem-aref x 0) 2)
      (is (mem-aref y 0) 2.0)
      (is (mem-aref z 0) (make-float3 2.0 2.0 2.0) :test #'float3-=))))


;;;
;;; test kernel-defun
;;;

(diag "test kernel-defun")

;; test var-ptr
(is (cl-cuda.api::var-ptr 'x) 'x-ptr)

;; test kernel-arg-names
(is (cl-cuda.api::kernel-arg-names '((x cl-cuda.lang:int) (y cl-cuda.lang:float3) (a cl-cuda.lang:float3*))) '(x y a))

;; test kernel-arg-non-array-args
(is (cl-cuda.api::kernel-arg-non-array-args '((x cl-cuda.lang:int) (y cl-cuda.lang:float3) (a cl-cuda.lang:float3*)))
                                        '((x cl-cuda.lang:int) (y cl-cuda.lang:float3)))

;; test kernel-arg-array-args
(is (cl-cuda.api::kernel-arg-array-args '((x cl-cuda.lang:int) (y cl-cuda.lang:float3) (a cl-cuda.lang:float3*)))
                                    '((a cl-cuda.lang:float3*)))

;; test kernel-arg-ptr-type-binding
(is (cl-cuda.api::kernel-arg-ptr-type-binding '(x cl-cuda.lang:int))    '(x-ptr :int))
(is (cl-cuda.api::kernel-arg-ptr-type-binding '(x cl-cuda.lang:float3)) '(x-ptr '(:struct float3)))

;; test kernel-arg-ptr-type-binding
(is (cl-cuda.api::kernel-arg-ptr-var-binding '(a cl-cuda.lang:float3*)) '(a-ptr a))

;; test kernel-arg-pointer
(is (cl-cuda.api::kernel-arg-pointer '(x int))     'x-ptr)
(is (cl-cuda.api::kernel-arg-pointer '(x float3))  'x-ptr)
(is (cl-cuda.api::kernel-arg-pointer '(x float3*)) 'x-ptr)

;; test setf-to-foreign-memory-form
(is (cl-cuda.api::setf-to-foreign-memory-form '(x int)) '(setf (cffi:mem-ref x-ptr :int) x))
(is (cl-cuda.api::setf-to-foreign-memory-form '(x float3))
    '(setf (cffi:foreign-slot-value x-ptr '(:struct float3) 'cl-cuda.api::x) (float3-x x)
           (cffi:foreign-slot-value x-ptr '(:struct float3) 'cl-cuda.api::y) (float3-y x)
           (cffi:foreign-slot-value x-ptr '(:struct float3) 'cl-cuda.api::z) (float3-z x)))

;; test setf-to-argument-array-form
(is (cl-cuda.api::setf-to-argument-array-form 'kargs '(x int) 0)
    '(setf (cffi:mem-aref kargs :pointer 0) x-ptr))
(is (cl-cuda.api::setf-to-argument-array-form 'kargs '(y float3) 1)
    '(setf (cffi:mem-aref kargs :pointer 1) y-ptr))
(is (cl-cuda.api::setf-to-argument-array-form 'kargs '(a float3*) 2)
    '(setf (cffi:mem-aref kargs :pointer 2) a-ptr))

;; test with-kernel-arguments
(is-expand
  (cl-cuda.api::with-kernel-arguments (kargs ((x int) (y float3) (a float3*)))
    nil)
  (cffi:with-foreign-objects ((x-ptr :int)
                              (y-ptr '(:struct float3)))
    (setf (cffi:mem-ref x-ptr :int) x)
    (setf (cffi:foreign-slot-value y-ptr '(:struct float3) 'cl-cuda.api::x) (float3-x y)
          (cffi:foreign-slot-value y-ptr '(:struct float3) 'cl-cuda.api::y) (float3-y y)
          (cffi:foreign-slot-value y-ptr '(:struct float3) 'cl-cuda.api::z) (float3-z y))
    (cl-cuda.api::with-memory-block-device-ptrs ((a-ptr a))
      (cffi:with-foreign-object (kargs :pointer 3)
        (setf (cffi:mem-aref kargs :pointer 0) x-ptr)
        (setf (cffi:mem-aref kargs :pointer 1) y-ptr)
        (setf (cffi:mem-aref kargs :pointer 2) a-ptr)
        nil))))


;;;
;;; test variable interfaces
;;;

;; test *tmp-path*
(let ((*tmp-path* "/tmp/"))
  (is (cl-cuda.api::get-tmp-path) "/tmp/"))
(let ((*tmp-path* "/tmp"))
  (is (cl-cuda.api::get-tmp-path) "/tmp/"))

(let ((*tmp-path* "/tmp/")
      (cl-cuda.api::*mktemp* "xxxxxx"))
  (is (cl-cuda.api::get-cu-path) "/tmp/cl-cuda-xxxxxx.cu"))

(let ((*tmp-path* "/tmp/")
      (cl-cuda.api::*mktemp* "xxxxxx"))
  (is (cl-cuda.api::get-ptx-path) "/tmp/cl-cuda-xxxxxx.ptx"))

;; test *nvcc-options*
(let ((*nvcc-options* (list "--verbose")))
  (is (cl-cuda.api::get-nvcc-options "include-path" "cu-path" "ptx-path")
      (list "--verbose" "-I" "include-path" "-ptx" "-o" "ptx-path" "cu-path")))


;;;
;;; test kernel functions
;;;

;; test "let1" kernel
(defkernel let1 (void ())
  (let ((i 0))
    (return))
  (let ((i 0))))

(let ((dev-id 0))
  (with-cuda-context (dev-id)
    (let1 :grid-dim (list 1 1 1)
          :block-dim (list 1 1 1))))

;; test "use-one" kernel
(defkernel use-one (void ())
  (let ((i (one)))
    (return)))

(defkernel one (int ())
  (return 1))

(let ((dev-id 0))
  (with-cuda-context (dev-id)
    (use-one :grid-dim (list 1 1 1)
             :block-dim (list 1 1 1))))

;; test "argument" kernel
(defkernel argument (void ((i int) (j float3)))
  (return))

(let ((dev-id 0))
  (with-cuda-context (dev-id)
    (argument 1 (make-float3 0.0 0.0 0.0) :grid-dim (list 1 1 1)
                                          :block-dim (list 1 1 1))))

;; test "kernel-bool" kernel
(defkernel kernel-bool (void ((ary bool*)))
  (set (aref ary 0) t)
  (set (aref ary 1) nil)
  (return))

(let ((dev-id 0))
  (with-cuda-context (dev-id)
    (with-memory-blocks ((a 'bool 2))
      (setf (mem-aref a 0) nil
            (mem-aref a 1) t)
      (memcpy-host-to-device a)
      (kernel-bool a :grid-dim '(1 1 1) :block-dim '(1 1 1))
      (memcpy-device-to-host a)
      (is (mem-aref a 0) t)
      (is (mem-aref a 1) nil))))

;; test "kernel-float3" kernel
(defkernel kernel-float3 (void ((ary float*) (x float3)))
  (set (aref ary 0) (+ (float3-x x) (float3-y x) (float3-z x))))

(let ((dev-id 0)
      (x (make-float3 1.0 2.0 3.0)))
  (with-cuda-context (dev-id)
    (with-memory-blocks ((a 'float 1))
      (setf (mem-aref a 0) 1.0)
      (memcpy-host-to-device a)
      (kernel-float3 a x :grid-dim '(1 1 1)
                         :block-dim '(1 1 1))
      (memcpy-device-to-host a)
      (is (mem-aref a 0) 6.0))))

;; test DO statement
(defkernel test-do-kernel (void ((x int*)))
  (do ((i 0 (+ i 1)))
      ((> i 15))
    (set (aref x 0) (+ (aref x 0) 1))))

(let ((dev-id 0))
  (with-cuda-context (dev-id)
    (with-memory-blocks ((x 'int 1))
      (setf (mem-aref x 0) 0)
      (memcpy-host-to-device x)
      (test-do-kernel x :grid-dim '(1 1 1) :block-dim '(1 1 1))
      (memcpy-device-to-host x)
      (is (mem-aref x 0) 16))))

;; test multi-argument arithmetic
(defkernel test-add (void ((x int*)))
  (set (aref x 0) (+ 1 1 1)))

(let ((dev-id 0))
  (with-cuda-context (dev-id)
    (with-memory-blocks ((x 'int 1))
      (setf (mem-aref x 0) 0)
      (memcpy-host-to-device x)
      (test-add x :grid-dim '(1 1 1) :block-dim '(1 1 1))
      (memcpy-device-to-host x)
      (is (mem-aref x 0) 3))))

;; test atomic function
(defkernel test-atomic-add (void ((x int*)))
  (atomic-add (pointer (aref x 0)) 1))

(defkernel test-no-atomic-add (void ((x int*)))
  (set (aref x 0) (+ (aref x 0) 1)))

(let ((dev-id 0))
  (with-cuda-context (dev-id)
    (with-memory-blocks ((x 'int 1)
                         (y 'int 1))
      (setf (mem-aref x 0) 0
            (mem-aref y 0) 0)
      (memcpy-host-to-device x y)
      (test-atomic-add    x :grid-dim '(1 1 1) :block-dim '(256 1 1))
      (test-no-atomic-add y :grid-dim '(1 1 1) :block-dim '(256 1 1))
      (memcpy-device-to-host x y)
      (is   (mem-aref x 0) 256)
      (isnt (mem-aref y 0) 256))))

;; test built-in vector type
(defkernel test-float3-add (void ((x float3*)))
  (set (aref x 0) (+ (aref x 0) (float3 1.0 1.0 1.0))))

(let ((dev-id 0))
  (with-cuda-context (dev-id)
    (with-memory-blocks ((x 'float3 1))
      (setf (mem-aref x 0) (make-float3 0.0 0.0 0.0))
      (memcpy-host-to-device x)
      (test-float3-add x :grid-dim '(1 1 1) :block-dim '(1 1 1))
      (memcpy-device-to-host x)
      (is (mem-aref x 0) (make-float3 1.0 1.0 1.0) :test #'float3-=))))

;; test kernel macro
(defkernelmacro when (test &body forms)
  `(if ,test
       (progn ,@forms)))

(defkernel test-when (void ())
  (when t (return))
  (return))

(let ((dev-id 0))
  (with-cuda-context (dev-id)
    (test-when :grid-dim '(1 1 1) :block-dim '(1 1 1))))

;; test kernel constant
(defkernelconst x int 1)

(defkernel test-const (void ((ret int*)))
  (set (aref ret 0) x)
  (return))

(let ((dev-id 0))
  (with-cuda-context (dev-id)
    (with-memory-blocks ((x 'int 1))
      (test-const x :grid-dim '(1 1 1) :block-dim '(1 1 1)))))



;; test expand-macro-1 and expand-macro
(defkernelmacro foo (x)
  `(return ,x))

(defkernelmacro bar (x)
  `(foo ,x))

(defmacro is-values (got expected &rest args)
  `(is (multiple-value-list ,got) ,expected ,@args))

(is-values (expand-macro-1 '(foo 1))       '((return 1) t))
(is-error  (expand-macro-1 '(foo))         error)
(is-error  (expand-macro-1 '(foo 1 2))     error)
(is-values (expand-macro-1 '(1))           '((1) nil))
(is-values (expand-macro-1 1)              '(1 nil))
(is-values (expand-macro-1 '(bar 1))       '((foo 1) t))
(is-values (expand-macro-1 '(baz 1))       '((baz 1) nil))
(is-values (expand-macro   '(foo 1))       '((return 1) t))
(is-values (expand-macro   '(bar 1))       '((return 1) t))
(is-values (expand-macro   '(baz 1))       '((baz 1) nil))
(is-values (expand-macro   1)              '(1 nil))
(is-values (expand-macro   '())            '(() nil))
(is-values (expand-macro   '(foo (foo 1))) '((return (foo 1)) t))

;; test built-in macros
(is       (expand-macro '(+))       0)
(is       (expand-macro '(+ 1))     1)
(is       (expand-macro '(+ 1 2))   '(cl-cuda.lang::%add 1 2))
(is       (expand-macro '(+ 1 2 3)) '(cl-cuda.lang::%add (cl-cuda.lang::%add 1 2) 3))
(is-error (expand-macro '(-))       simple-error)
(is       (expand-macro '(- 1))     '(cl-cuda.lang::%negate 1))
(is       (expand-macro '(- 1 2))   '(cl-cuda.lang::%sub 1 2))
(is       (expand-macro '(- 1 2 3)) '(cl-cuda.lang::%sub (cl-cuda.lang::%sub 1 2) 3))
(is       (expand-macro '(*))       1)
(is       (expand-macro '(* 1))     1)
(is       (expand-macro '(* 1 2))   '(cl-cuda.lang::%mul 1 2))
(is       (expand-macro '(* 1 2 3)) '(cl-cuda.lang::%mul (cl-cuda.lang::%mul 1 2) 3))
(is-error (expand-macro '(/))       simple-error)
(is       (expand-macro '(/ 1))     '(cl-cuda.lang::%recip 1))
(is       (expand-macro '(/ 1 2))   '(cl-cuda.lang::%div 1 2))
(is       (expand-macro '(/ 1 2 3)) '(cl-cuda.lang::%div (cl-cuda.lang::%div 1 2) 3))


(finalize)
