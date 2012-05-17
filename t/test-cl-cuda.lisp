#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-cuda-test)

(setf *test-result-output* *standard-output*)

(plan nil)


;;; test cuInit
(diag "test cuInit")
(cu-init 0)


;;; test cuDeviceGet
(diag "test cuDeviceGet")
(let ((dev-id 0))
  (cffi:with-foreign-object (device 'cu-device)
    (setf (cffi:mem-ref device :int) 42)
    (cu-device-get device dev-id)
    (format t "CUDA device handle: ~A~%" (cffi:mem-ref device 'cu-device))))


;;; test cuDeviceGetCount
(diag "test cuDeviceGetCount")
(cffi:with-foreign-object (count :int)
  (cu-device-get-count count)
  (format t "CUDA device count: ~A~%" (cffi:mem-ref count :int)))


;;; test cuDeviceComputeCapability
(diag "test cuDeviceComputeCapability")
(let ((dev-id 0))
  (cffi:with-foreign-objects ((major :int)
                              (minor :int)
                              (device 'cu-device))
    (cu-device-get device dev-id)
    (cu-device-compute-capability major minor (cffi:mem-ref device 'cu-device))
    (format t "CUDA device compute capability: ~A.~A~%"
              (cffi:mem-ref major :int) (cffi:mem-ref minor :int))))


;;; test cuDeviceGetName
(diag "test cuDeviceGetName")
(let ((dev-id 0))
  (cffi:with-foreign-object (device 'cu-device)
  (cffi:with-foreign-pointer-as-string ((name size) 255)
    (cu-device-get device dev-id)
    (cu-device-get-name name size (cffi:mem-ref device 'cu-device))
    (format t "CUDA device name: ~A~%" (cffi:foreign-string-to-lisp name)))))


;;; test cuCtxCreate/cuCtxDestroy
(diag "test cuCtxCreate/cuCtxDestroy")
(let ((flags 0)
      (dev-id 0))
  (cffi:with-foreign-objects ((pctx 'cu-context)
                              (device 'cu-device))
    (cu-device-get device dev-id)
    (cu-ctx-create pctx flags (cffi:mem-ref device 'cu-device))
    (cu-ctx-destroy (cffi:mem-ref pctx 'cu-context))))


;;; test cuMemAlloc/cuMemFree
(diag "test cuMemAlloc/cuMemFree")
(let ((flags 0)
      (dev-id 0))
  (cffi:with-foreign-objects ((device 'cu-device)
                              (pctx 'cu-context)
                              (dptr 'cu-device-ptr))
    (cu-device-get device dev-id)
    (cu-ctx-create pctx flags (cffi:mem-ref device 'cu-device))
    (cu-mem-alloc dptr 1024)
    (cu-mem-free (cffi:mem-ref dptr 'cu-device-ptr))
    (cu-ctx-destroy (cffi:mem-ref pctx 'cu-context))))


;;; test cuMemAlloc/cuMemFree using with-cuda-context
(diag "test cuMemAlloc/cuMemFree using with-cuda-context")
(let ((dev-id 0))
  (with-cuda-context (dev-id)
    (cffi:with-foreign-object (dptr 'cu-device-ptr)
      (cu-mem-alloc dptr 1024)
      (cu-mem-free (cffi:mem-ref dptr 'cu-device-ptr)))))


;;; test cuMemAlloc/cuMemFree using with-cuda-context and with-cuda-mem-block
(diag "test cuMemAlloc/cuMemFree using with-cuda-context and with-cuda-mem-block")
(let ((dev-id 0))
  (with-cuda-context (dev-id)
    (with-cuda-memory-block (dptr 1024))))


;;; test cuMemAlloc/cuMemFree using with-cuda-context and with-cuda-mem-blocks
(diag "test cuMemAlloc/cuMemFree using with-cuda-context and with-cuda-mem-blocks")
(let ((dev-id 0))
  (with-cuda-context (dev-id)
    (with-cuda-memory-blocks ((dptr1 1024)
                              (dptr2 1024)))))


;;; test cuMemcpyHtoD/cuMemcpyDtoH
(diag "test cuMemcpyHtoD/cuMemcpyDtoH")
(let ((dev-id 0)
      (size 1024))
  (with-cuda-context (dev-id)
    (cffi:with-foreign-object (hptr :float size)
      (with-cuda-memory-block (dptr size)
        (cu-memcpy-host-to-device (cffi:mem-ref dptr 'cu-device-ptr) hptr size)
        (cu-memcpy-device-to-host hptr (cffi:mem-ref dptr 'cu-device-ptr) size)))))


;;; test cuModuleLoad
(diag "test cuModuleLoad")
(let ((dev-id 0))
  (cffi:with-foreign-string (fname "/Developer/GPU Computing/C/src/vectorAddDrv/data/vectorAdd_kernel.ptx")
    (with-cuda-context (dev-id)
      (cffi:with-foreign-object (module 'cu-module)
        (cu-module-load module fname)
        (format t "CUDA module \"vectorAdd_kernel.ptx\" is loaded.~%")))))


;;; test cuModuleGetFunction
(diag "test cuModuleGetFunction")
(let ((dev-id 0))
  (cffi:with-foreign-string (fname "/Developer/GPU Computing/C/src/vectorAddDrv/data/vectorAdd_kernel.ptx")
    (cffi:with-foreign-string (name "VecAdd_kernel")
      (with-cuda-context (dev-id)
        (cffi:with-foreign-objects ((module 'cu-module)
                                    (hfunc 'cu-function))
          (cu-module-load module fname)
          (cu-module-get-function hfunc (cffi:mem-ref module 'cu-module) name))))))


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
      (cu-memcpy-host-to-device (cffi:mem-ref d-a 'cu-device-ptr) h-a size)
      (cu-memcpy-host-to-device (cffi:mem-ref d-b 'cu-device-ptr) h-b size)
      (vec-add-kernel d-a d-b d-c n
                      :grid-dim (list blocks-per-grid 1 1)
                      :block-dim (list threads-per-block 1 1))
      (cu-memcpy-device-to-host h-c (cffi:mem-ref d-c 'cu-device-ptr) size)
      (verify-result h-a h-b h-c n)))))

(defkernel let1 (void ())
  (let ((i 0))
    (return))
  (let ((i 0))))

(defun test-let1 ()
  (let ((dev-id 0))
    (with-cuda-context (dev-id)
      (let1 :grid-dim (list 1 1 1)
                 :block-dim (list 1 1 1)))))

(defkernel use-one (void ())
  (let ((i (one)))
    (return)))

(defkernel one (int ())
  (return 1))

(defun test-one ()
  (let ((dev-id 0))
    (with-cuda-context (dev-id)
      (use-one :grid-dim (list 1 1 1)
               :block-dim (list 1 1 1)))))

(defkernel argument (void ((i int)))
  (let ((j i))
    (return)))

(defun test-argument ()
  (let ((dev-id 0))
    (with-cuda-context (dev-id)
      (argument 1 :grid-dim (list 1 1 1)
                  :block-dim (list 1 1 1)))))


;;; test kernel definition

(is (cl-cuda::empty-kernel-definition) '(nil nil))

(is (cl-cuda::define-kernel-function 'foo 'void '() '((return))
      (cl-cuda::empty-kernel-definition))
    '(((foo void () ((return)))) ()))

(is-error (cl-cuda::define-kernel-constant 'foo 1
            (cl-cuda::empty-kernel-definition))
          simple-error)

(is (cl-cuda::undefine-kernel-function 'foo
      (cl-cuda::define-kernel-function 'foo 'void '() '((return))
        (cl-cuda::empty-kernel-definition)))
    (cl-cuda::empty-kernel-definition))

(is-error (cl-cuda::undefine-kernel-function 'foo
            (cl-cuda::empty-kernel-definition))
          simple-error)

(is-error (cl-cuda::undefine-kernel-constant 'foo
            (cl-cuda::define-kernel-constant 'foo 1
              (cl-cuda::empty-kernel-definition)))
          simple-error)

(is-error (cl-cuda::undefine-kernel-constant 'foo
            (cl-cuda::empty-kernel-definition))
          simple-error)

(let ((def (cl-cuda::empty-kernel-definition)))
  (is (cl-cuda::kernel-function-exists-p 'foo def) nil))

(let ((def (cl-cuda::define-kernel-function 'foo 'void '() '()
             (cl-cuda::empty-kernel-definition))))
  (is (cl-cuda::kernel-function-exists-p 'foo def) t))

(let ((def (cl-cuda::define-kernel-function 'foo 'void '() '((return))
             (cl-cuda::empty-kernel-definition))))
  (is (cl-cuda::kernel-function-name 'foo def) 'foo)
  (is (cl-cuda::kernel-function-c-name 'foo def) "foo")
  (is (cl-cuda::kernel-function-return-type 'foo def) 'void)
  (is (cl-cuda::kernel-function-arg-bindings 'foo def) '())
  (is (cl-cuda::kernel-function-body 'foo def) '((return))))

(let ((def (cl-cuda::empty-kernel-definition)))
  (is-error (cl-cuda::kernel-function-name 'foo def) simple-error))

(let ((def (cl-cuda::empty-kernel-definition)))
  (is (cl-cuda::kernel-function-names def) nil))

(let ((def (cl-cuda::define-kernel-function 'foo 'void '() '((return))
             (cl-cuda::define-kernel-function 'bar 'int '() '((return 1))
               (cl-cuda::empty-kernel-definition)))))
  (is (cl-cuda::kernel-function-names def) '(foo bar)))


;;; test compile-kernel-definition

(let ((def (cl-cuda::define-kernel-function 'foo 'void '() '((return))
             (cl-cuda::empty-kernel-definition)))
      (c-code (cl-cuda::unlines "extern \"C\" __global__ void foo ();"
                                ""
                                "__global__ void foo ()"
                                "{"
                                "  return;"
                                "}"
                                "")))
  (is (cl-cuda::compile-kernel-definition def) c-code))


;;; test compile-kernel-function-prototype

(let ((def (cl-cuda::define-kernel-function 'foo 'void '() '((return))
             (cl-cuda::empty-kernel-definition)))
      (c-code (cl-cuda::unlines "extern \"C\" __global__ void foo ();")))
  (is (cl-cuda::compile-kernel-function-prototype 'foo def) c-code))


;;; test compile-kernel-function

(diag "test compile-kernel-function")

(let ((def (cl-cuda::define-kernel-function 'foo 'void '() '((return))
             (cl-cuda::empty-kernel-definition)))
      (c-code (cl-cuda::unlines "__global__ void foo ()"
                                "{"
                                "  return;"
                                "}"
                                "")))
  (is (cl-cuda::compile-kernel-function 'foo def) c-code))


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

(is (cl-cuda::built-in-function-p '(+ 1 1)) t)
(is (cl-cuda::built-in-function-p '(- 1 1)) t)
(is (cl-cuda::built-in-function-p '(foo 1 1)) nil)

(is (cl-cuda::function-candidates '+)
    '(((int int) int "+")
      ((float float) float "+")))
(is-error (cl-cuda::function-candidates 'foo)
          simple-error)

(is (cl-cuda::built-in-function-infix-p '+) t)
(is-error (cl-cuda::built-in-function-infix-p 'foo) simple-error)

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
(let ((def (cl-cuda::define-kernel-function 'foo 'void '() '()
             (cl-cuda::empty-kernel-definition))))
  (is (cl-cuda::compile-function '(foo) nil def :statement-p t) "foo ();"))
(is (cl-cuda::compile-function '(+ 1 1) nil nil) "(1 + 1)")
(is-error (cl-cuda::compile-function '(+ 1 1 1) nil nil) simple-error)
(is-error (cl-cuda::compile-function '(foo 1 1) nil nil) simple-error)
(let ((def (cl-cuda::define-kernel-function 'foo 'void '((x int) (y int)) '()
             (cl-cuda::empty-kernel-definition))))
  (is (cl-cuda::compile-function '(foo 1 1) nil def :statement-p t)
      "foo (1, 1);")
  (is-error (cl-cuda::compile-function '(foo 1 1 1) nil def :statement-p t)
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
(let ((def (cl-cuda::define-kernel-function 'foo 'int '((x int) (y int)) '()
             (cl-cuda::empty-kernel-definition))))
  (is (cl-cuda::type-of-function '(foo 1 1) nil def) 'int))

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


(finalize)
