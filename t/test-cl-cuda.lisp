#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-cuda-test)

(setf *test-result-output* *standard-output*)

(plan nil)


;;; test cuInit
(cu-init 0)


;;; test cuDeviceGet
(let ((dev-id 0))
  (cffi:with-foreign-object (device 'cu-device)
    (setf (cffi:mem-ref device :int) 42)
    (check-cuda-errors (cu-device-get device dev-id))
    (format t "CUDA device handle: ~A~%" (cffi:mem-ref device 'cu-device))))


;;; test cuDeviceGetCount
(cffi:with-foreign-object (count :int)
  (check-cuda-errors (cu-device-get-count count))
  (format t "CUDA device count: ~A~%" (cffi:mem-ref count :int)))


;;; test cuDeviceComputeCapability
(let ((dev-id 0))
  (cffi:with-foreign-objects ((major :int)
                              (minor :int)
                              (device 'cu-device))
    (check-cuda-errors (cu-device-get device dev-id))
    (check-cuda-errors
     (cu-device-compute-capability major minor
                                   (cffi:mem-ref device 'cu-device)))
    (format t "CUDA device compute capability: ~A.~A~%"
            (cffi:mem-ref major :int) (cffi:mem-ref minor :int))))


;;; test cuDeviceGetName
(let ((dev-id 0))
  (cffi:with-foreign-object (device 'cu-device)
  (cffi:with-foreign-pointer-as-string ((name size) 255)
    (check-cuda-errors (cu-device-get device dev-id))
    (check-cuda-errors (cu-device-get-name name size
                                           (cffi:mem-ref device 'cu-device)))
    (format t "CUDA device name: ~A~%" (cffi:foreign-string-to-lisp name)))))


;;; test cuCtxCreate/cuCtxDestroy
(let ((flags 0)
      (dev-id 0))
  (cffi:with-foreign-objects ((pctx 'cu-context)
                              (device 'cu-device))
    (check-cuda-errors (cu-device-get device dev-id))
    (check-cuda-errors (cu-ctx-create pctx flags
                                      (cffi:mem-ref device 'cu-device)))
    (format t "a CUDA context is created.~%")
    (check-cuda-errors (cu-ctx-destroy (cffi:mem-ref pctx 'cu-context)))
    (format t "a CUDA context is destroyed.~%")))


;;; test cuMemAlloc/cuMemFree
(let ((flags 0)
      (dev-id 0))
  (cffi:with-foreign-objects ((device 'cu-device)
                              (pctx 'cu-context)
                              (dptr 'cu-device-ptr))
    (check-cuda-errors (cu-device-get device dev-id))
    (check-cuda-errors (cu-ctx-create pctx flags
                                      (cffi:mem-ref device 'cu-device)))
    (check-cuda-errors (cu-mem-alloc dptr 1024))
    (format t "a CUDA memory block is allocated.~%")
    (check-cuda-errors (cu-mem-free (cffi:mem-ref dptr 'cu-device-ptr)))
    (format t "a CUDA memory block is freed.~%")
    (check-cuda-errors (cu-ctx-destroy (cffi:mem-ref pctx 'cu-context)))))


;;; test cuMemAlloc/cuMemFree using with-cuda-context
(let ((dev-id 0))
  (with-cuda-context (dev-id)
    (cffi:with-foreign-object (dptr 'cu-device-ptr)
      (check-cuda-errors (cu-mem-alloc dptr 1024))
      (format t "a CUDA memory block is allocated.~%")
      (check-cuda-errors (cu-mem-free (cffi:mem-ref dptr 'cu-device-ptr)))
      (format t "a CUDA memory block is freed.~%"))))


;;; test cuMemAlloc/cuMemFree using with-cuda-context and with-cuda-mem-block
(let ((dev-id 0))
  (with-cuda-context (dev-id)
    (with-cuda-memory-block (dptr 1024)
      (format t "a CUDA memory block is allocated.~%"))))


;;; test cuMemAlloc/cuMemFree using with-cuda-context and with-cuda-mem-blocks
(let ((dev-id 0))
  (with-cuda-context (dev-id)
    (with-cuda-memory-blocks ((dptr1 1024)
                              (dptr2 1024))
      (format t "two CUDA memory blocks are allocated.~%"))))


;;; test cuMemcpyHtoD/cuMemcpyDtoH
(let ((dev-id 0)
      (size 1024))
  (with-cuda-context (dev-id)
    (cffi:with-foreign-object (hptr :float size)
      (with-cuda-memory-block (dptr size)
        (check-cuda-errors
         (cu-memcpy-host-to-device (cffi:mem-ref dptr 'cu-device-ptr)
                                   hptr size))
        (format t "a CUDA memory block is copied from host to device.~%")
        (check-cuda-errors
         (cu-memcpy-device-to-host hptr
                                   (cffi:mem-ref dptr 'cu-device-ptr) size))
        (format t "a CUDA memory block is copied from device to host.~%")))))


;;; test cuModuleLoad
(let ((dev-id 0))
  (cffi:with-foreign-string (fname "/Developer/GPU Computing/C/src/vectorAddDrv/data/vectorAdd_kernel.ptx")
    (with-cuda-context (dev-id)
      (cffi:with-foreign-object (module 'cu-module)
        (check-cuda-errors (cu-module-load module fname))
        (format t "CUDA module \"vectorAdd_kernel.ptx\" is loaded.~%")))))


;;; test cuModuleGetFunction
(let ((dev-id 0))
  (cffi:with-foreign-string (fname "/Developer/GPU Computing/C/src/vectorAddDrv/data/vectorAdd_kernel.ptx")
    (cffi:with-foreign-string (name "VecAdd_kernel")
      (with-cuda-context (dev-id)
        (cffi:with-foreign-objects ((module 'cu-module)
                                    (hfunc 'cu-function))
          (check-cuda-errors (cu-module-load module fname))
          (check-cuda-errors
           (cu-module-get-function hfunc
                                   (cffi:mem-ref module 'cu-module)
                                   name))
          (format t "CUDA function \"VecAdd_kernel\" is loaded.~%"))))))


;;; test cuLaunchKernel

(defun random-init (data n)
  (dotimes (i n)
    (setf (cffi:mem-aref data :float i) (random 1.0))))

(defun verify-result (as bs cs n)
  (dotimes (i n)
    (let ((a (cffi:mem-aref as :float i))
          (b (cffi:mem-aref bs :float i))
          (c (cffi:mem-aref cs :float i)))
      (let ((sum (+ a b)))
        (when (> (abs (- c sum)) 1.0)
          (error (format nil "verification fault, i:~A a:~A b:~A c:~A"
                         i a b c))))))
  (format t "verification succeed.~%"))

(defkernel vec-add-kernel (void ((a float*) (b float*) (c float*) (n int)))
  (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (if (< i n)
        (set (aref c i)
             (+ (aref a i) (aref b i))))))

(let* ((dev-id 0)
       (n 1024)
       (size (* n 4))                   ; 4 is size of float
       (threads-per-block 256)
       (blocks-per-grid (/ n threads-per-block)))
  (with-cuda-context (dev-id)
    (cffi:with-foreign-objects ((h-a :float n)
                                (h-b :float n)
                                (h-c :float n))
    (with-cuda-memory-blocks ((d-a size)
                              (d-b size)
                              (d-c size))
      (random-init h-a n)
      (random-init h-b n)
      (check-cuda-errors
       (cu-memcpy-host-to-device (cffi:mem-ref d-a 'cu-device-ptr)
                                 h-a size))
      (check-cuda-errors
       (cu-memcpy-host-to-device (cffi:mem-ref d-b 'cu-device-ptr)
                                 h-b size))
      (vec-add-kernel d-a d-b d-c n
                      :grid-dim (list blocks-per-grid 1 1)
                      :block-dim (list threads-per-block 1 1))
      (format t "CUDA function \"vec_add_kernel\" is launched.~%")
      (check-cuda-errors
       (cu-memcpy-device-to-host h-c
                                 (cffi:mem-ref d-c 'cu-device-ptr)
                                 size))
      (verify-result h-a h-b h-c n)))))

(defkernel test-let1 (void ())
  (let ((i 0))
    (return))
  (let ((i 0))))

(defun test-test-let1 ()
  (let ((dev-id 0))
    (with-cuda-context (dev-id)
      (test-let1 :grid-dim (list 1 1 1)
                 :block-dim (list 1 1 1)))))

(defkernel test-one (void ())
  (let ((i (one)))
    (return)))

(defkernel one (int ())
  (return 1))

(defun test-test-one ()
  (let ((dev-id 0))
    (with-cuda-context (dev-id)
      (test-one :grid-dim (list 1 1 1)
                :block-dim (list 1 1 1)))))


;;; test compile-kernel-function

(diag "test compile-kernel-function")

(let ((lisp-code '((return)))
      (c-code (cl-cuda::unlines "extern \"C\" __global__ void foo () {"
                                "  return;"
                                "}")))
  (is (cl-cuda::compile-kernel-function "foo" 'void '() lisp-code nil) c-code))


;;; test compile-if

(diag "test compile-if")

(let ((lisp-code '(if 1
                      (return)
                      (return)))
      (c-code (cl-cuda::unlines "if (1) {"
                                "  return;"
                                "} else {"
                                "  return;"
                                "}")))
  (is (cl-cuda::compile-if lisp-code nil nil) c-code))

(let ((lisp-code '(if 1
                      (progn
                        (return 0)
                        (return 0))))
      (c-code (cl-cuda::unlines "if (1) {"
                                "  return 0;"
                                "  return 0;"
                                "}")))
  (is (cl-cuda::compile-if lisp-code nil nil) c-code))


;;; test compile-let  

(diag "test compile-let")

(let ((lisp-code '(let ((i 0))
                    (return)
                    (return)))
      (c-code (cl-cuda::unlines "{"
                                "  int i = 0;"
                                "  return;"
                                "  return;"
                                "}")))
  (is (cl-cuda::compile-let lisp-code nil nil) c-code))


;;; test compile-set

(diag "test compile-set")

(is (cl-cuda::set-p '(set x 1)) t)
(is (cl-cuda::set-p '(set (aref x i) 1)) t)

(let ((type-env (cl-cuda::add-type-environment
                  'x 'int (cl-cuda::empty-type-environment))))
  (is (cl-cuda::compile-set '(set x 1) type-env nil) "x = 1;"))
(let ((type-env (cl-cuda::add-type-environment
                  'x 'int* (cl-cuda::empty-type-environment))))
  (is (cl-cuda::compile-set '(set (aref x 0) 1) type-env nil) "x[0] = 1;"))


;;; test compile-function

(diag "test compile-function")

(is (cl-cuda::built-in-function-p '(+ 1 1)) t "built-in-function-p 1")
(is (cl-cuda::built-in-function-p '(- 1 1)) t "built-in-function-p 2")
(is (cl-cuda::built-in-function-p '(foo 1 1)) nil "built-in-function-p 3")

(is (cl-cuda::function-candidates '+)
    '(((int int) int "+")
      ((float float) float "+"))
    "built-in-function-candidates 1")
(is-error (cl-cuda::function-candidates 'foo)
          simple-error "built-in-function-candidates 2")

(is (cl-cuda::built-in-function-infix-p '+)
    t "built-in-function-infix-p 1")
(is-error (cl-cuda::built-in-function-infix-p 'foo)
          simple-error "built-in-function-infix-p 2")

(is (cl-cuda::function-p 'a) nil)
(is (cl-cuda::function-p '()) nil)
(is (cl-cuda::function-p '1) nil)
(is (cl-cuda::function-p '(foo)) t)
(is (cl-cuda::function-p '(+ 1 1)) t)
(is (cl-cuda::function-p '(foo 1 1)) t)

(is-error (cl-cuda::function-operator 'a) simple-error)
(is (cl-cuda::function-operator '(foo)) 'foo)
(is (cl-cuda::function-operator '(+ 1 1)) '+)
(is (cl-cuda::function-operator '(foo 1 1)) 'foo)

(is-error (cl-cuda::function-operands 'a) simple-error)
(is (cl-cuda::function-operands '(foo)) '())
(is (cl-cuda::function-operands '(+ 1 1)) '(1 1))
(is (cl-cuda::function-operands '(foo 1 1)) '(1 1))

(is-error (cl-cuda::compile-function 'a nil nil) simple-error)
(let ((funcs '(foo (() void))))
  (is (cl-cuda::compile-function '(foo) nil funcs :statement-p t) "foo ();"))
(is (cl-cuda::compile-function '(+ 1 1) nil nil) "(1 + 1)")
(is-error (cl-cuda::compile-function '(+ 1 1 1) nil nil) simple-error)
(is-error (cl-cuda::compile-function '(foo 1 1) nil nil) simple-error)
(let ((funcs '(foo ((int int) int))))
  (is (cl-cuda::compile-function '(foo 1 1) nil funcs :statement-p t)
      "foo (1, 1);"))
(let ((funcs '(foo ((int int) int))))
  (is-error (cl-cuda::compile-function '(foo 1 1 1) nil funcs :statement-p t)
            simple-error))


;;; test type-of-expression

(diag "test type-of-expression")

(is (cl-cuda::type-of-expression '1 nil nil) 'int)
(is (cl-cuda::type-of-expression '1.0 nil nil) 'float)

(is (cl-cuda::type-of-literal '1) 'int)
(is (cl-cuda::type-of-literal '1.0) 'float)
(is-error (cl-cuda::type-of-literal '1.0d0) simple-error)

(is-error (cl-cuda::type-of-variable-reference 'x nil) simple-error)
(let ((type-env (cl-cuda::add-type-environment
                  'x 'int (cl-cuda::empty-type-environment))))
  (is (cl-cuda::type-of-variable-reference 'x type-env) 'int))
(let ((type-env (cl-cuda::add-type-environment
                  'x 'float* (cl-cuda::empty-type-environment))))
  (is (cl-cuda::type-of-variable-reference '(aref x 0) type-env) 'float))

(is (cl-cuda::type-of-function '(+ 1 1) nil nil) 'int)
(let ((funcs '(foo ((int int) int))))
  (is (cl-cuda::type-of-function '(foo 1 1) nil funcs) 'int))

(is (cl-cuda::type-of-expression 'cl-cuda::grid-dim-x nil nil) 'int)
(is (cl-cuda::type-of-expression 'cl-cuda::grid-dim-y nil nil) 'int)
(is (cl-cuda::type-of-expression 'cl-cuda::grid-dim-z nil nil) 'int)
(is (cl-cuda::type-of-expression 'cl-cuda::block-idx-x nil nil) 'int)
(is (cl-cuda::type-of-expression 'cl-cuda::block-idx-y nil nil) 'int)
(is (cl-cuda::type-of-expression 'cl-cuda::block-idx-z nil nil) 'int)
(is (cl-cuda::type-of-expression 'cl-cuda::block-dim-x nil nil) 'int)
(is (cl-cuda::type-of-expression 'cl-cuda::block-dim-y nil nil) 'int)
(is (cl-cuda::type-of-expression 'cl-cuda::block-dim-z nil nil) 'int)
(is (cl-cuda::type-of-expression 'cl-cuda::thread-idx-x nil nil) 'int)
(is (cl-cuda::type-of-expression 'cl-cuda::thread-idx-y nil nil) 'int)
(is (cl-cuda::type-of-expression 'cl-cuda::thread-idx-z nil nil) 'int)


;;; test compile-identifier

(diag "test compile-identifier")

(is (cl-cuda::compile-identifier 'x) "x")
(is (cl-cuda::compile-identifier 'vec-add-kernel) "vec_add_kernel")
(is (cl-cuda::compile-identifier 'VecAdd_kernel) "vecadd_kernel")


;;; test compile-variable-reference

(diag "test compile-variable-reference")

(is (cl-cuda::variable-reference-p 'x) t)
(is (cl-cuda::variable-reference-p 1) nil)
(is (cl-cuda::variable-reference-p '(aref x i)) t)
(is (cl-cuda::variable-reference-p '(aref x i i)) nil)

(is-error (cl-cuda::compile-variable-reference 'x nil nil) simple-error)
(let ((type-env (cl-cuda::add-type-environment
                  'x 'int (cl-cuda::empty-type-environment))))
  (is (cl-cuda::compile-variable-reference 'x type-env nil) "x"))
(let ((type-env (cl-cuda::add-type-environment
                  'x 'int* (cl-cuda::empty-type-environment))))
  (is (cl-cuda::compile-variable-reference '(aref x 0) type-env nil) "x[0]"))

(let ((type-env (cl-cuda::add-type-environment
                  'x 'int (cl-cuda::empty-type-environment))))
  (is (cl-cuda::type-of-variable-reference 'x type-env) 'int))
(let ((type-env (cl-cuda::add-type-environment
                  'x 'int* (cl-cuda::empty-type-environment))))
  (is (cl-cuda::type-of-variable-reference '(aref x 0) type-env) 'int))


;;; test user-functions

(diag "test user-functions")

(is-error (cl-cuda::user-function-c-name 'foo '() nil nil) simple-error)
(let ((funcs (cl-cuda::make-user-functions '((foo void ())))))
  (is (cl-cuda::user-function-c-name 'foo '() nil funcs) "foo"))
(let ((funcs (cl-cuda::make-user-functions '((foo void ())))))
  (is-error (cl-cuda::user-function-c-name 'foo '(1) nil funcs) simple-error))

(let ((funcs (cl-cuda::make-user-functions '((foo void ())))))
  (is (cl-cuda::user-function-type 'foo '() nil funcs) '()))

(let ((funcs (cl-cuda::make-user-functions '((foo void ())))))
  (is (cl-cuda::user-function-return-type 'foo '() nil funcs) 'void))

(is (cl-cuda::user-function-exists-p 'foo nil) nil)
(let ((funcs (cl-cuda::make-user-functions '((foo void ())))))
  (is (cl-cuda::user-function-exists-p 'foo funcs) t))


(finalize)
