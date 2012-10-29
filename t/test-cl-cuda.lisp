#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-cuda-test)

(setf *test-result-output* *standard-output*)

(plan nil)


;;;
;;; test defcuenum
;;;

(diag "test defcuenum")

;; test enum-keyword
(is       (cl-cuda::enum-keyword '(:a 1)  ) :a          )
(is       (cl-cuda::enum-keyword :a       ) :a          )
(is-error (cl-cuda::enum-keyword nil      ) simple-error)
(is-error (cl-cuda::enum-keyword '(:a 1 2)) simple-error)

;; test enum-value
(is       (cl-cuda::enum-value '(:a 1)  ) 1           )
(is-error (cl-cuda::enum-value '(:a 1 2)) simple-error)
(is-error (cl-cuda::enum-value :a       ) simple-error)

;; test expansion of defcuenum macro
(is-expand (cl-cuda::defcuenum cu-event-flags-enum
             (:cu-event-default #X0)
             (:cu-event-blocking-sync #X1)
             (:cu-event-disable-timing #X2)
             (:cu-event-interprocess #X4))
           (progn
             (cffi:defcenum cu-event-flags-enum
               (:cu-event-default #X0)
               (:cu-event-blocking-sync #X1)
               (:cu-event-disable-timing #X2)
               (:cu-event-interprocess #X4))
             (defconstant cu-event-default
               (cffi:foreign-enum-value 'cu-event-flags-enum
                                        :cu-event-default))
             (defconstant cu-event-blocking-sync
               (cffi:foreign-enum-value 'cu-event-flags-enum
                                        :cu-event-blocking-sync))
             (defconstant cu-event-disable-timing
               (cffi:foreign-enum-value 'cu-event-flags-enum
                                        :cu-event-disable-timing))
             (defconstant cu-event-interprocess
               (cffi:foreign-enum-value 'cu-event-flags-enum
                                        :cu-event-interprocess))))


;;;
;;; test CUDA driver API
;;;

;; test cuInit
(diag "test cuInit")
(cl-cuda::cu-init 0)

;; test cuDeviceGet
(diag "test cuDeviceGet")
(let ((dev-id 0))
  (cffi:with-foreign-object (device 'cl-cuda::cu-device)
    (setf (cffi:mem-ref device :int) 42)
    (cl-cuda::cu-device-get device dev-id)
    (format t "CUDA device handle: ~A~%" (cffi:mem-ref device 'cl-cuda::cu-device))))

;; test cuDeviceGetCount
(diag "test cuDeviceGetCount")
(cffi:with-foreign-object (count :int)
  (cl-cuda::cu-device-get-count count)
  (format t "CUDA device count: ~A~%" (cffi:mem-ref count :int)))

;; test cuDeviceComputeCapability
(diag "test cuDeviceComputeCapability")
(let ((dev-id 0))
  (cffi:with-foreign-objects ((major :int)
                              (minor :int)
                              (device 'cl-cuda::cu-device))
    (cl-cuda::cu-device-get device dev-id)
    (cl-cuda::cu-device-compute-capability major minor (cffi:mem-ref device 'cl-cuda::cu-device))
    (format t "CUDA device compute capability: ~A.~A~%"
              (cffi:mem-ref major :int) (cffi:mem-ref minor :int))))

;; test cuDeviceGetName
(diag "test cuDeviceGetName")
(let ((dev-id 0))
  (cffi:with-foreign-object (device 'cl-cuda::cu-device)
  (cffi:with-foreign-pointer-as-string ((name size) 255)
    (cl-cuda::cu-device-get device dev-id)
    (cl-cuda::cu-device-get-name name size (cffi:mem-ref device 'cl-cuda::cu-device))
    (format t "CUDA device name: ~A~%" (cffi:foreign-string-to-lisp name)))))

;; test cuCtxCreate/cuCtxDestroy
(diag "test cuCtxCreate/cuCtxDestroy")
(let ((flags 0)
      (dev-id 0))
  (cffi:with-foreign-objects ((pctx   'cl-cuda::cu-context)
                              (device 'cl-cuda::cu-device))
    (cl-cuda::cu-device-get device dev-id)
    (cl-cuda::cu-ctx-create pctx flags (cffi:mem-ref device 'cl-cuda::cu-device))
    (cl-cuda::cu-ctx-destroy (cffi:mem-ref pctx 'cl-cuda::cu-context))))

;; test cuMemAlloc/cuMemFree
(diag "test cuMemAlloc/cuMemFree")
(let ((flags 0)
      (dev-id 0))
  (cffi:with-foreign-objects ((device 'cl-cuda::cu-device)
                              (pctx   'cl-cuda::cu-context)
                              (dptr   'cl-cuda::cu-device-ptr))
    (cl-cuda::cu-device-get device dev-id)
    (cl-cuda::cu-ctx-create pctx flags (cffi:mem-ref device 'cl-cuda::cu-device))
    (cl-cuda::cu-mem-alloc dptr 1024)
    (cl-cuda::cu-mem-free (cffi:mem-ref dptr 'cl-cuda::cu-device-ptr))
    (cl-cuda::cu-ctx-destroy (cffi:mem-ref pctx 'cl-cuda::cu-context))))

;; test cuMemAlloc/cuMemFree using with-cuda-context
(diag "test cuMemAlloc/cuMemFree using with-cuda-context")
(let ((dev-id 0))
  (with-cuda-context (dev-id)
    (cffi:with-foreign-object (dptr 'cl-cuda::cu-device-ptr)
      (cl-cuda::cu-mem-alloc dptr 1024)
      (cl-cuda::cu-mem-free (cffi:mem-ref dptr 'cl-cuda::cu-device-ptr)))))

;; test cuMemAlloc/cuMemFree using with-cuda-context and with-cuda-mem-block
(diag "test cuMemAlloc/cuMemFree using with-cuda-context and with-cuda-mem-block")
(let ((dev-id 0))
  (with-cuda-context (dev-id)
    (cl-cuda::with-cuda-memory-block (dptr 1024))))

;; test cuMemAlloc/cuMemFree using with-cuda-context and with-cuda-mem-blocks
(diag "test cuMemAlloc/cuMemFree using with-cuda-context and with-cuda-mem-blocks")
(let ((dev-id 0))
  (with-cuda-context (dev-id)
    (cl-cuda::with-cuda-memory-blocks ((dptr1 1024)
                                       (dptr2 1024)))))

;; test cuMemcpyHtoD/cuMemcpyDtoH
(diag "test cuMemcpyHtoD/cuMemcpyDtoH")
(let ((dev-id 0)
      (size 1024))
  (with-cuda-context (dev-id)
    (cffi:with-foreign-object (hptr :float size)
      (cl-cuda::with-cuda-memory-block (dptr size)
        (cl-cuda::cu-memcpy-host-to-device (cffi:mem-ref dptr 'cl-cuda::cu-device-ptr) hptr size)
        (cl-cuda::cu-memcpy-device-to-host hptr (cffi:mem-ref dptr 'cl-cuda::cu-device-ptr) size)))))

;; test cuModuleLoad
(diag "test cuModuleLoad")
(let ((dev-id 0))
  (cffi:with-foreign-string (fname "/Developer/GPU Computing/C/src/vectorAddDrv/data/vectorAdd_kernel.ptx")
    (with-cuda-context (dev-id)
      (cffi:with-foreign-object (module 'cl-cuda::cu-module)
        (cl-cuda::cu-module-load module fname)
        (format t "CUDA module \"vectorAdd_kernel.ptx\" is loaded.~%")))))

;; test cuModuleGetFunction
(diag "test cuModuleGetFunction")
(let ((dev-id 0))
  (cffi:with-foreign-string (fname "/Developer/GPU Computing/C/src/vectorAddDrv/data/vectorAdd_kernel.ptx")
    (cffi:with-foreign-string (name "VecAdd_kernel")
      (with-cuda-context (dev-id)
        (cffi:with-foreign-objects ((module 'cl-cuda::cu-module)
                                    (hfunc  'cl-cuda::cu-function))
          (cl-cuda::cu-module-load module fname)
          (cl-cuda::cu-module-get-function hfunc (cffi:mem-ref module 'cl-cuda::cu-module) name))))))


;;;
;;; test CUDA Event Management functions
;;;

(let ((dev-id 0))
  (with-cuda-context (dev-id)
    (cffi:with-foreign-objects ((start-event 'cl-cuda::cu-event)
                                (stop-event  'cl-cuda::cu-event)
                                (milliseconds :float))
      (cl-cuda::cu-event-create start-event cl-cuda::cu-event-default)
      (cl-cuda::cu-event-create stop-event  cl-cuda::cu-event-default)
      (cl-cuda::cu-event-record (cffi:mem-ref start-event 'cl-cuda::cu-event) (cffi:null-pointer))
      (cl-cuda::cu-event-record (cffi:mem-ref stop-event  'cl-cuda::cu-event) (cffi:null-pointer))
      (cl-cuda::cu-event-synchronize (cffi:mem-ref stop-event 'cl-cuda::cu-event))
      (cl-cuda::cu-event-query       (cffi:mem-ref stop-event 'cl-cuda::cu-event))
      (cl-cuda::cu-event-elapsed-time milliseconds
                                      (cffi:mem-ref start-event 'cl-cuda::cu-event)
                                      (cffi:mem-ref stop-event  'cl-cuda::cu-event))
      (format t "CUDA Event - elapsed time: ~A~%" (cffi:mem-ref milliseconds :float))
      (cl-cuda::cu-event-destroy (cffi:mem-ref start-event 'cl-cuda::cu-event))
      (cl-cuda::cu-event-destroy (cffi:mem-ref stop-event  'cl-cuda::cu-event)))))


;;; test timer

(diag "test timer")

(let (timer)
  (with-cuda-context (0)
    (setf timer (create-timer))
    (start-timer timer)
    (format t "elapsed time: ~A~%" (get-elapsed-time timer))
    (stop-timer timer)
    (destroy-timer timer)
    (ok t)))

(with-cuda-context (0)
  (with-timer (timer)
    (start-timer timer)
    (format t "elapsed time: ~A~%" (get-elapsed-time timer))
    (stop-timer timer))
  (ok t))


;;;
;;; test memory blocks
;;;

(diag "test memory blocks")

;; test alloc-memory-block
(let ((dev-id 0))
  (with-cuda-context (dev-id)
    (let (blk)
      (ok (setf blk (cl-cuda::alloc-memory-block 'int 1024)))
      (cl-cuda::free-memory-block blk))
    (is-error (cl-cuda::alloc-memory-block 'void 1024             ) simple-error)
    (is-error (cl-cuda::alloc-memory-block 'int* 1024             ) simple-error)
    (is-error (cl-cuda::alloc-memory-block 'int  (* 1024 1024 256)) simple-error)
    (is-error (cl-cuda::alloc-memory-block 'int  0                ) simple-error)
    (is-error (cl-cuda::alloc-memory-block 'int  -1               ) type-error)))

;; test selectors of memory-block
(let ((dev-id 0))
  (with-cuda-context (dev-id)
    (with-memory-blocks ((blk 'int 1024))
      (ok (cl-cuda::memory-block-cffi-ptr blk))
      (ok (cl-cuda::memory-block-device-ptr blk))
      (is (cl-cuda::memory-block-type blk) 'int)
      (is (cl-cuda::memory-block-cffi-type blk) :int)
      (is (cl-cuda::memory-block-length blk) 1024)
      (is (cl-cuda::memory-block-bytes blk) (* 1024 4))
      (is (cl-cuda::memory-block-element-bytes blk) 4))))

;; test setf functions of memory-block
(let ((dev-id 0))
  (with-cuda-context (dev-id)
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
(defkernel test-memcpy (void ((x int*) (y float*)))
  (set (aref x 0) (+ (aref x 0) 1))
  (set (aref y 0) (+ (aref y 0) 1.0)))

(let ((dev-id 0))
  (with-cuda-context (dev-id)
    (with-memory-blocks ((x 'int 1)
                         (y 'float 1))
      (setf (mem-aref x 0) 1)
      (setf (mem-aref y 0) 1.0)
      (memcpy-host-to-device x y)
      (test-memcpy x y :grid-dim '(1 1 1)
                       :block-dim '(1 1 1))
      (memcpy-device-to-host x y)
      (is (mem-aref x 0) 2)
      (is (mem-aref y 0) 2.0))))


;;;
;;; test kernel-defun
;;;

(diag "test kernel-defun")

;; test vector-type-length
(is       (cl-cuda::vector-type-length 'float3 ) 3           )
(is-error (cl-cuda::vector-type-length 'float  ) simple-error)
(is-error (cl-cuda::vector-type-length 'float3*) simple-error)

;; test vector-type-base-type
(is       (cl-cuda::vector-type-base-type 'float3 ) 'float      )
(is-error (cl-cuda::vector-type-base-type 'float  ) simple-error)
(is-error (cl-cuda::vector-type-base-type 'float3*) simple-error)

;; test vector-type-selector-symbol
(is       (cl-cuda::vector-type-selector-symbol 'float3 'cl-cuda::x) 'float3-x   )
(is-error (cl-cuda::vector-type-selector-symbol 'float  'cl-cuda::x) simple-error)
(is-error (cl-cuda::vector-type-selector-symbol 'float3 'cl-cuda::a) simple-error)

;; test vector-type-selector-symbols
(is (cl-cuda::vector-type-selector-symbols)
    '(float3-x float3-y float3-z
      float4-x float4-y float4-z float4-w))

;; test foreign-pointer-setf-vector-type
(is (cl-cuda::foreign-pointer-setf-vector-type 'x 'x-ptr 'float3)
    '(progn
      (setf (cffi:foreign-slot-value x-ptr 'float3 'cl-cuda::x) (float3-x x))
      (setf (cffi:foreign-slot-value x-ptr 'float3 'cl-cuda::y) (float3-y x))
      (setf (cffi:foreign-slot-value x-ptr 'float3 'cl-cuda::z) (float3-z x))))

;; test expansion of with-non-pointer-arguments macro
(is-expand
  (cl-cuda::with-non-pointer-arguments ((n n-ptr :int)
                                        (x x-ptr :float)
                                        (a a-ptr float3))
    nil)
  (cffi:with-foreign-objects ((n-ptr :int)
                              (x-ptr :float)
                              (a-ptr 'float3))
    (setf (cffi:mem-ref n-ptr :int) n)
    (setf (cffi:mem-ref x-ptr :float) x)
    (progn
      (setf (cffi:foreign-slot-value a-ptr 'float3 'cl-cuda::x) (float3-x a))
      (setf (cffi:foreign-slot-value a-ptr 'float3 'cl-cuda::y) (float3-y a))
      (setf (cffi:foreign-slot-value a-ptr 'float3 'cl-cuda::z) (float3-z a)))
    nil))

;; test expansion of with-kernel-arguments macro
(is-expand
 (cl-cuda::with-kernel-arguments (args
                                  (cl-cuda::memory-block-device-ptr a)
                                  (cl-cuda::memory-block-device-ptr b)
                                  (cl-cuda::memory-block-device-ptr c)
                                  n-ptr)
   nil)
 (cffi:with-foreign-object (args :pointer 4)
   (setf (cffi:mem-aref args :pointer 0) (cl-cuda::memory-block-device-ptr a))
   (setf (cffi:mem-aref args :pointer 1) (cl-cuda::memory-block-device-ptr b))
   (setf (cffi:mem-aref args :pointer 2) (cl-cuda::memory-block-device-ptr c))
   (setf (cffi:mem-aref args :pointer 3) n-ptr)
   nil))

;; test kernel-arg-names
(is (cl-cuda::kernel-arg-names
      '((a float*) (b float*) (c float*) (n int) (x float3)))
    '(a b c n x))

;; test kernel-arg-names-as-pointer
(is (cl-cuda::kernel-arg-names-as-pointer
      '((a float*) (b float*) (c float*) (n int) (x float3)))
    '((cl-cuda::memory-block-device-ptr a)
      (cl-cuda::memory-block-device-ptr b)
      (cl-cuda::memory-block-device-ptr c)
      n-ptr x-ptr))

;; test kernel-arg-foreign-pointer-bindings
(is (cl-cuda::kernel-arg-foreign-pointer-bindings
      '((a float*) (b float*) (c float*) (n int) (x float3)))
    '((n n-ptr :int) (x x-ptr float3)))


;;;
;;; test kernel functions
;;;

;; test "let1" kernel
(defkernel let1 (void ())
  (let ((i 0))
    (return))
  (let ((i 0))))

(defun test-let1 ()
  (let ((dev-id 0))
    (with-cuda-context (dev-id)
      (let1 :grid-dim (list 1 1 1)
            :block-dim (list 1 1 1)))))

;; test "use-one" kernel
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

;; test "argument" kernel
(defkernel argument (void ((i int)))
  (let ((j i))
    (return)))

(defun test-argument ()
  (let ((dev-id 0))
    (with-cuda-context (dev-id)
      (argument 1 :grid-dim (list 1 1 1)
                  :block-dim (list 1 1 1)))))

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


;;;
;;; test valid-type-p
;;;

(diag "test valid-type-p")

;; test basic-type-p
(is (cl-cuda::basic-type-p 'void ) t)
(is (cl-cuda::basic-type-p 'int  ) t)
(is (cl-cuda::basic-type-p 'float) t)

;; test vector-type-p
(is (cl-cuda::vector-type-p 'float3) t  )
(is (cl-cuda::vector-type-p 'float4) t  )
(is (cl-cuda::vector-type-p 'float5) nil)

;; test valid-type-p
(is (cl-cuda::valid-type-p 'void    ) t  )
(is (cl-cuda::valid-type-p 'int     ) t  )
(is (cl-cuda::valid-type-p 'float   ) t  )
(is (cl-cuda::valid-type-p 'double  ) nil)
(is (cl-cuda::valid-type-p 'float3  ) t  )
(is (cl-cuda::valid-type-p 'float4  ) t  )
(is (cl-cuda::valid-type-p 'float*  ) t  )
(is (cl-cuda::valid-type-p 'float** ) t  )
(is (cl-cuda::valid-type-p '*float**) nil)

;; test pointer-type-p
(is (cl-cuda::pointer-type-p 'int    ) nil)
(is (cl-cuda::pointer-type-p 'float* ) t  )
(is (cl-cuda::pointer-type-p 'float3*) t  )
(is (cl-cuda::pointer-type-p 'float4*) t  )
(is (cl-cuda::pointer-type-p '*float*) nil)

;; test non-pointer-type-p
(is (cl-cuda::non-pointer-type-p 'int     ) t  )
(is (cl-cuda::non-pointer-type-p 'float*  ) nil)
(is (cl-cuda::non-pointer-type-p 'float3* ) nil)
(is (cl-cuda::non-pointer-type-p 'float4* ) nil)
(is (cl-cuda::non-pointer-type-p '*float3*) nil)

;; test add-star
(is (cl-cuda::add-star 'int -1) 'int           )
(is (cl-cuda::add-star 'int 0 ) 'int           )
(is (cl-cuda::add-star 'int 1 ) 'int*          )
(is (cl-cuda::add-star 'int 2 ) 'cl-cuda::int**)

;; test remove-star
(is (cl-cuda::remove-star 'int  ) 'int)
(is (cl-cuda::remove-star 'int* ) 'int)
(is (cl-cuda::remove-star 'int**) 'int)

;; test type-dimension
(is (cl-cuda::type-dimension 'int   ) 0)
(is (cl-cuda::type-dimension 'int*  ) 1)
(is (cl-cuda::type-dimension 'int** ) 2)
(is (cl-cuda::type-dimension 'int***) 3)

;; test cffi-type
(is-error (cl-cuda::cffi-type 'void   ) simple-error  )
(is       (cl-cuda::cffi-type 'int    ) :int          )
(is       (cl-cuda::cffi-type 'float  ) :float        )
(is       (cl-cuda::cffi-type 'float3 ) 'float3       )
(is       (cl-cuda::cffi-type 'float4 ) 'float4       )
(is       (cl-cuda::cffi-type 'float* ) 'cl-cuda::cu-device-ptr)
(is       (cl-cuda::cffi-type 'float3*) 'cl-cuda::cu-device-ptr)
(is       (cl-cuda::cffi-type 'float4*) 'cl-cuda::cu-device-ptr)

(is (cl-cuda::size-of 'void  ) 0 )
(is (cl-cuda::size-of 'int   ) 4 )
(is (cl-cuda::size-of 'float ) 4 )
(is (cl-cuda::size-of 'float3) 12)
(is (cl-cuda::size-of 'float4) 16)
(is (cl-cuda::size-of 'int*  ) 4 )
(is (cl-cuda::size-of 'int** ) 4 )
(is (cl-cuda::size-of 'int***) 4 )


;;;
;;; test kernel definition
;;;

(diag "test kernel definition")

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
  (is (cl-cuda::kernel-definition-function-exists-p 'foo def) nil))

(let ((def (cl-cuda::define-kernel-function 'foo 'void '() '()
             (cl-cuda::empty-kernel-definition))))
  (is (cl-cuda::kernel-definition-function-exists-p 'foo def) t))

(let ((def (cl-cuda::define-kernel-function 'foo 'void '() '((return))
             (cl-cuda::empty-kernel-definition))))
  (is (cl-cuda::kernel-definition-function-name        'foo def) 'foo              )
  (is (cl-cuda::kernel-definition-function-c-name      'foo def) "cl_cuda_test_foo")
  (is (cl-cuda::kernel-definition-function-return-type 'foo def) 'void             )
  (is (cl-cuda::kernel-definition-function-arguments   'foo def) '()               )
  (is (cl-cuda::kernel-definition-function-body        'foo def) '((return)))      )

(let ((def (cl-cuda::empty-kernel-definition)))
  (is-error (cl-cuda::kernel-definition-function-name 'foo def) simple-error))

(let ((def (cl-cuda::empty-kernel-definition)))
  (is (cl-cuda::kernel-definition-function-names def) nil))

(let ((def (cl-cuda::define-kernel-function 'foo 'void '() '((return))
             (cl-cuda::define-kernel-function 'bar 'int '() '((return 1))
               (cl-cuda::empty-kernel-definition)))))
  (is (cl-cuda::kernel-definition-function-names def) '(foo bar)))


;;;
;;; test compile-kernel-definition
;;;

(diag "test compile-kernel-definition")

(let ((def (cl-cuda::define-kernel-function 'foo 'void '() '((return))
             (cl-cuda::empty-kernel-definition)))
      (c-code (cl-cuda::unlines "extern \"C\" __global__ void cl_cuda_test_foo ();"
                                ""
                                "__global__ void cl_cuda_test_foo ()"
                                "{"
                                "  return;"
                                "}"
                                "")))
  (is (cl-cuda::compile-kernel-definition def) c-code))


;;;
;;; test compile-kernel-function-prototype
;;;

(diag "test compile-kernel-function-prototype")

(let ((def (cl-cuda::define-kernel-function 'foo 'void '() '((return))
             (cl-cuda::empty-kernel-definition)))
      (c-code (cl-cuda::unlines "extern \"C\" __global__ void cl_cuda_test_foo ();")))
  (is (cl-cuda::compile-kernel-function-prototype 'foo def) c-code))


;;;
;;; test compile-kernel-function
;;;

(diag "test compile-kernel-function")

(let ((def (cl-cuda::define-kernel-function 'foo 'void '() '((return))
             (cl-cuda::empty-kernel-definition)))
      (c-code (cl-cuda::unlines "__global__ void cl_cuda_test_foo ()"
                                "{"
                                "  return;"
                                "}"
                                "")))
  (is (cl-cuda::compile-kernel-function 'foo def) c-code))


;;;
;;; test compile-function-specifier (not implemented)
;;;



;;;
;;; test compile-type (not implemented)
;;;



;;;
;;; test compile-identifier
;;;

(diag "test compile-identifier")

;; test compile-identifier
(is (cl-cuda::compile-identifier 'x             ) "x"             )
(is (cl-cuda::compile-identifier 'vec-add-kernel) "vec_add_kernel")
(is (cl-cuda::compile-identifier 'VecAdd_kernel ) "vecadd_kernel" )


;;;
;;; test compile-if
;;;

(diag "test compile-if")

(let ((lisp-code '(if 1
                      (return)
                      (return)))
      (c-code (cl-cuda::unlines "if 1 {"
                                "  return;"
                                "} else {"
                                "  return;"
                                "}")))
  (is (cl-cuda::compile-if lisp-code nil nil) c-code))

(let ((lisp-code '(if 1
                      (progn
                        (return 0)
                        (return 0))))
      (c-code (cl-cuda::unlines "if 1 {"
                                "  return 0;"
                                "  return 0;"
                                "}")))
  (is (cl-cuda::compile-if lisp-code nil nil) c-code))


;;;
;;; test compile-let  
;;;

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


;;;
;;; test compile-do
;;;

(diag "test compile-do")

;; test do selectors
(let* ((code '(do ((a 0 (+ a 1))
                   (b 0 (+ b 1)))
                  ((> a 15))
                (return)))
       (binding (first (cl-cuda::do-bindings code))))
  (cl-cuda::with-type-environment (type-env ((a int) (b int)))
    (is (cl-cuda::do-p code)                    t)
    (is (cl-cuda::do-bindings code)             '((a 0 (+ a 1))
                                                  (b 0 (+ b 1))))
    (is (cl-cuda::do-var-types code nil nil)    '((a int) (b int)))
    (is (cl-cuda::do-binding-var binding)       'a)
    (is (cl-cuda::do-binding-type binding type-env nil) 'int)
    (is (cl-cuda::do-binding-init-form binding) 0)
    (is (cl-cuda::do-binding-step-form binding) '(+ a 1))
    (is (cl-cuda::do-test-form code)            '(> a 15))
    (is (cl-cuda::do-statements code)           '((return)))))


;; test compile-do
(let ((lisp-code '(do ((a 0 (+ a 1))
                       (b 0 (+ b 1)))
                      ((> a 15))
                    (return)))
      (c-code (cl-cuda::unlines "for ( int a = 0, int b = 0; ! (a > 15); a = (a + 1), b = (b + 1) )"
                                "{"
                                "  return;"
                                "}")))
  (cl-cuda::with-type-environment (type-env ((a int) (b int)))
    (is (cl-cuda::compile-do-init-part lisp-code nil      nil) "int a = 0, int b = 0")
    (is (cl-cuda::compile-do-test-part lisp-code type-env nil) "! (a > 15)")
    (is (cl-cuda::compile-do-step-part lisp-code type-env nil) "a = (a + 1), b = (b + 1)")
    (is (cl-cuda::compile-do lisp-code nil nil) c-code)))

(let ((lisp-code '(do ((a 0.0 (+ a 1.0)))
                      ((> a 15.0))
                    (return)))
      (c-code (cl-cuda::unlines "for ( float a = 0.0; ! (a > 15.0); a = (a + 1.0) )"
                                "{"
                                "  return;"
                                "}")))
  (is (cl-cuda::compile-do lisp-code nil nil) c-code))


;;;
;;; test compile-with-shared-memory
;;;

(diag "test compile-with-shared-memory")

;; test with-shared-memory-p
(is (cl-cuda::with-shared-memory-p '(with-shared-memory ((a float 16))
                                      (return)))
    t)
(is (cl-cuda::with-shared-memory-p '(with-shared-memory () (return))) t)
(is (cl-cuda::with-shared-memory-p '(with-shared-memory ())         ) t)
(is (cl-cuda::with-shared-memory-p '(with-shared-memory)            ) t)

;; test compile-with-shared-memory
(let ((lisp-code '(with-shared-memory ((a int 16)
                                       (b float 16 16))
                   (return)))
      (c-code (cl-cuda::unlines "{"
                                "  __shared__ int a[16];"
                                "  __shared__ float b[16][16];"
                                "  return;"
                                "}")))
  (is (cl-cuda::compile-with-shared-memory lisp-code nil nil) c-code))

(let ((lisp-code '(with-shared-memory () (return)))
      (c-code (cl-cuda::unlines "{"
                                "  return;"
                                "}")))
  (is (cl-cuda::compile-with-shared-memory lisp-code nil nil) c-code))

(let ((lisp-code '(with-shared-memory ()))
      (c-code (cl-cuda::unlines "{"
                                ""
                                "}")))
  (is (cl-cuda::compile-with-shared-memory lisp-code nil nil) c-code))

(is-error (cl-cuda::compile-with-shared-memory '(with-shared-memory) nil nil) simple-error)

(let ((lisp-code '(with-shared-memory ((a float))
                    (return)))
      (c-code (cl-cuda::unlines "{"
                                "  __shared__ float a;"
                                "  return;"
                                "}")))
  (is (cl-cuda::compile-with-shared-memory lisp-code nil nil) c-code))

(let ((lisp-code '(with-shared-memory (a float)
                    (return))))
  (is-error (cl-cuda::compile-with-shared-memory lisp-code nil nil) simple-error))

(let ((lisp-code '(with-shared-memory ((a float 16 16))
                    (set (aref a 0 0) 1.0)))
      (c-code (cl-cuda::unlines "{"
                                "  __shared__ float a[16][16];"
                                "  a[0][0] = 1.0;"
                                "}")))
  (is (cl-cuda::compile-with-shared-memory lisp-code nil nil) c-code))

(let ((lisp-code '(with-shared-memory ((a float 16 16))
                    (set (aref a 0) 1.0))))
  (is-error (cl-cuda::compile-with-shared-memory lisp-code nil nil) simple-error))


;;;
;;; test compile-set
;;;

(diag "test compile-set")

;; test set-p
(is (cl-cuda::set-p '(set x          1)) t)
(is (cl-cuda::set-p '(set (aref x i) 1)) t)

;; compile-set
(cl-cuda::with-type-environment (type-env ((x int)))
  (is (cl-cuda::compile-set '(set x 1) type-env nil) "x = 1;"))

(cl-cuda::with-type-environment (type-env ((x int*)))
  (is (cl-cuda::compile-set '(set (aref x 0) 1) type-env nil) "x[0] = 1;"))

(cl-cuda::with-type-environment (type-env ((x float3)))
  (is (cl-cuda::compile-set '(set (float3-x x) 1.0) type-env nil) "x.x = 1.0;"))


;;;
;;; test compile-place (not implemented)
;;;

 

;;;
;;; test compile-progn (not implemented)
;;;



;;;
;;; test compile-return (not implemented)
;;;



;;;
;;; test compile-syncthreads
;;;

(diag "test compile-syncthreads")

;; test syncthreads-p
(is (cl-cuda::syncthreads-p '(syncthreads)) t)

;; test compile-syncthreads
(is (cl-cuda::compile-syncthreads '(syncthreads)) "__syncthreads();")


;;;
;;; test compile-function
;;;

(diag "test compile-function")

;; test built-in-function-p
(is (cl-cuda::built-in-function-p '(+ 1 1)  ) t  )
(is (cl-cuda::built-in-function-p '(- 1 1)  ) t  )
(is (cl-cuda::built-in-function-p '(foo 1 1)) nil)

;; test function-candidates
(is       (cl-cuda::function-candidates '+  ) '(((int int) int "+")
                                                ((float float) float "+")))
(is-error (cl-cuda::function-candidates 'foo) simple-error)

;; test built-in-function-infix-p
(is       (cl-cuda::built-in-function-infix-p '+   ) t           )
(is       (cl-cuda::built-in-function-infix-p 'expt) nil         )
(is-error (cl-cuda::built-in-function-infix-p 'foo ) simple-error)

;; test built-in-function-prefix-p
(is       (cl-cuda::built-in-function-prefix-p '+   ) nil         )
(is       (cl-cuda::built-in-function-prefix-p 'expt) t           )
(is-error (cl-cuda::built-in-function-prefix-p 'foo ) simple-error)

;; test function-p
(is (cl-cuda::function-p 'a        ) nil)
(is (cl-cuda::function-p '()       ) nil)
(is (cl-cuda::function-p '1        ) nil)
(is (cl-cuda::function-p '(foo)    ) t  )
(is (cl-cuda::function-p '(+ 1 1)  ) t  )
(is (cl-cuda::function-p '(foo 1 1)) t  )

;; test function-operator
(is-error (cl-cuda::function-operator 'a        ) simple-error)
(is       (cl-cuda::function-operator '(foo)    ) 'foo        )
(is       (cl-cuda::function-operator '(+ 1 1)  ) '+          )
(is       (cl-cuda::function-operator '(foo 1 1)) 'foo        )

;; test function-operands
(is-error (cl-cuda::function-operands 'a        ) simple-error)
(is       (cl-cuda::function-operands '(foo)    ) '()         )
(is       (cl-cuda::function-operands '(+ 1 1)  ) '(1 1)      )
(is       (cl-cuda::function-operands '(foo 1 1)) '(1 1)      )

;; test compile-function
(is-error (cl-cuda::compile-function 'a         nil nil) simple-error   )
(is       (cl-cuda::compile-function '(+ 1 1)   nil nil) "(1 + 1)"      )
(is       (cl-cuda::compile-function '(+ 1 1 1) nil nil) "((1 + 1) + 1)")

(let ((def (cl-cuda::define-kernel-function 'foo 'void '() '()
             (cl-cuda::empty-kernel-definition))))
  (is (cl-cuda::compile-function '(foo) nil def :statement-p t) "cl_cuda_test_foo ();"))

(let ((def (cl-cuda::define-kernel-function 'foo 'void '((x int) (y int)) '()
             (cl-cuda::empty-kernel-definition))))
  (is-error (cl-cuda::compile-function '(foo 1 1)   nil nil               ) simple-error              )
  (is       (cl-cuda::compile-function '(foo 1 1)   nil def :statement-p t) "cl_cuda_test_foo (1, 1);")
  (is-error (cl-cuda::compile-function '(foo 1 1 1) nil def :statement-p t) simple-error              ))

(is (cl-cuda::compile-function '(float3 1.0 1.0 1.0)     nil nil) "make_float3 (1.0, 1.0, 1.0)"     )
(is (cl-cuda::compile-function '(float4 1.0 1.0 1.0 1.0) nil nil) "make_float4 (1.0, 1.0, 1.0, 1.0)")


;;;
;;; test built-in arithmetic functions
;;;

(diag "test built-in arithmetic functions")

;; test compile-function
(is       (cl-cuda::compile-function '(+ 1 1)         nil nil) "(1 + 1)"            )
(is       (cl-cuda::compile-function '(+ 1 1 1)       nil nil) "((1 + 1) + 1)"      )
(is       (cl-cuda::compile-function '(+ 1.0 1.0 1.0) nil nil) "((1.0 + 1.0) + 1.0)")
(is-error (cl-cuda::compile-function '(+ 1 1 1.0)     nil nil) simple-error         )
(is-error (cl-cuda::compile-function '(+)             nil nil) simple-error         )
(is-error (cl-cuda::compile-function '(+ 1)           nil nil) simple-error         )

;; test built-in-arithmetic-function-return-type
(is-error (cl-cuda::built-in-arithmetic-function-return-type '(+)         nil nil) simple-error)
(is       (cl-cuda::built-in-arithmetic-function-return-type '(+ 1 1)     nil nil) 'int        )
(is       (cl-cuda::built-in-arithmetic-function-return-type '(+ 1.0 1.0) nil nil) 'float      )
(is-error (cl-cuda::built-in-arithmetic-function-return-type '(+ 1 1.0)   nil nil) simple-error)
(is-error (cl-cuda::built-in-arithmetic-function-return-type '(foo)       nil nil) simple-error)


;;;
;;; test compile-literal (not implemented)
;;;



;;;
;;; test compile-cuda-dimension (not implemented)
;;;



;;;
;;; test compile-variable-reference
;;;

(diag "test compile-variable-reference")

;; test variable-reference-p
(is (cl-cuda::variable-reference-p 'x)              t  )
(is (cl-cuda::variable-reference-p 1)               nil)
(is (cl-cuda::variable-reference-p '(aref x))       t  )
(is (cl-cuda::variable-reference-p '(aref x i))     t  )
(is (cl-cuda::variable-reference-p '(aref x i i))   t  )
(is (cl-cuda::variable-reference-p '(aref x i i i)) t  )
(is (cl-cuda::variable-reference-p '(float3-x x))   t  )
(is (cl-cuda::variable-reference-p '(float3-y x))   t  )
(is (cl-cuda::variable-reference-p '(float3-z x))   t  )
(is (cl-cuda::variable-reference-p '(float4-x x))   t  )
(is (cl-cuda::variable-reference-p '(float4-y x))   t  )
(is (cl-cuda::variable-reference-p '(float4-z x))   t  )
(is (cl-cuda::variable-reference-p '(float4-w x))   t  )

;; test compile-variable-reference
(is-error (cl-cuda::compile-variable-reference 'x nil nil) simple-error)

(cl-cuda::with-type-environment (type-env ((x int)))
  (is       (cl-cuda::compile-variable-reference 'x          type-env nil) "x"         )
  (is-error (cl-cuda::compile-variable-reference '(aref x)   type-env nil) simple-error)
  (is-error (cl-cuda::compile-variable-reference '(aref x 0) type-env nil) simple-error))

(cl-cuda::with-type-environment (type-env ((x int*)))
  (is       (cl-cuda::compile-variable-reference 'x            type-env nil) "x"         )
  (is       (cl-cuda::compile-variable-reference '(aref x 0)   type-env nil) "x[0]"      )
  (is-error (cl-cuda::compile-variable-reference '(aref x 0 0) type-env nil) simple-error))

(cl-cuda::with-type-environment (type-env ((x int**)))
  (is       (cl-cuda::compile-variable-reference 'x            type-env nil) "x"         )
  (is-error (cl-cuda::compile-variable-reference '(aref x 0)   type-env nil) simple-error)
  (is       (cl-cuda::compile-variable-reference '(aref x 0 0) type-env nil) "x[0][0]"   ))

(cl-cuda::with-type-environment (type-env ((x float3)))
  (is (cl-cuda::compile-variable-reference '(float3-x x) type-env nil) "x.x")
  (is (cl-cuda::compile-variable-reference '(float3-y x) type-env nil) "x.y")
  (is (cl-cuda::compile-variable-reference '(float3-z x) type-env nil) "x.z"))

(cl-cuda::with-type-environment (type-env ((x float4)))
  (is (cl-cuda::compile-variable-reference '(float4-x x) type-env nil) "x.x")
  (is (cl-cuda::compile-variable-reference '(float4-y x) type-env nil) "x.y")
  (is (cl-cuda::compile-variable-reference '(float4-z x) type-env nil) "x.z")
  (is (cl-cuda::compile-variable-reference '(float4-w x) type-env nil) "x.w"))


;;;
;;; test type-of-expression
;;;

(diag "test type-of-expression")

;; test type-of-expression
(is (cl-cuda::type-of-expression '1   nil nil) 'int  )
(is (cl-cuda::type-of-expression '1.0 nil nil) 'float)

;; test type-of-literal
(is       (cl-cuda::type-of-literal '1    ) 'int        )
(is       (cl-cuda::type-of-literal '1.0  ) 'float      )
(is-error (cl-cuda::type-of-literal '1.0d0) simple-error)

;; test type-of-function
(let ((def (cl-cuda::define-kernel-function 'foo 'int '((x int) (y int)) '()
             (cl-cuda::empty-kernel-definition))))
  (is (cl-cuda::type-of-function '(+ 1 1)   nil nil) 'int)
  (is (cl-cuda::type-of-function '(foo 1 1) nil def) 'int))

;; test type-of-function
(is       (cl-cuda::type-of-function '(+ 1 1 1)       nil nil) 'int        )
(is       (cl-cuda::type-of-function '(+ 1.0 1.0 1.0) nil nil) 'float      )
(is-error (cl-cuda::type-of-function '(+ 1 1 1.0)     nil nil) simple-error)
(is       (cl-cuda::type-of-function '(expt 1.0 1.0)  nil nil) 'float      )

;; test type-of-expression
(is (cl-cuda::type-of-expression 'cl-cuda::grid-dim-x   nil nil) 'int)
(is (cl-cuda::type-of-expression 'cl-cuda::grid-dim-y   nil nil) 'int)
(is (cl-cuda::type-of-expression 'cl-cuda::grid-dim-z   nil nil) 'int)
(is (cl-cuda::type-of-expression 'cl-cuda::block-idx-x  nil nil) 'int)
(is (cl-cuda::type-of-expression 'cl-cuda::block-idx-y  nil nil) 'int)
(is (cl-cuda::type-of-expression 'cl-cuda::block-idx-z  nil nil) 'int)
(is (cl-cuda::type-of-expression 'cl-cuda::block-dim-x  nil nil) 'int)
(is (cl-cuda::type-of-expression 'cl-cuda::block-dim-y  nil nil) 'int)
(is (cl-cuda::type-of-expression 'cl-cuda::block-dim-z  nil nil) 'int)
(is (cl-cuda::type-of-expression 'cl-cuda::thread-idx-x nil nil) 'int)
(is (cl-cuda::type-of-expression 'cl-cuda::thread-idx-y nil nil) 'int)
(is (cl-cuda::type-of-expression 'cl-cuda::thread-idx-z nil nil) 'int)

;; test type-of-variable-reference
(cl-cuda::with-type-environment (type-env ((x int)))
  (is-error (cl-cuda::type-of-variable-reference 'x        nil     ) simple-error)
  (is       (cl-cuda::type-of-variable-reference 'x        type-env) 'int        )
  (is-error (cl-cuda::type-of-variable-reference '(aref x) type-env) simple-error))

(cl-cuda::with-type-environment (type-env ((x int*)))
  (is       (cl-cuda::type-of-variable-reference 'x            type-env) 'int*       )
  (is       (cl-cuda::type-of-variable-reference '(aref x 0)   type-env) 'int        )
  (is-error (cl-cuda::type-of-variable-reference '(aref x 0 0) type-env) simple-error))

(cl-cuda::with-type-environment (type-env ((x int**)))
  (is       (cl-cuda::type-of-variable-reference 'x            type-env) 'int**      )
  (is-error (cl-cuda::type-of-variable-reference '(aref x 0)   type-env) simple-error)
  (is       (cl-cuda::type-of-variable-reference '(aref x 0 0) type-env) 'int        ))

(cl-cuda::with-type-environment (type-env ((x float3)))
  (is (cl-cuda::type-of-variable-reference '(float3-x x) type-env) 'float)
  (is (cl-cuda::type-of-variable-reference '(float3-y x) type-env) 'float)
  (is (cl-cuda::type-of-variable-reference '(float3-z x) type-env) 'float))

(cl-cuda::with-type-environment (type-env ((x float4)))
  (is (cl-cuda::type-of-variable-reference '(float4-x x) type-env) 'float)
  (is (cl-cuda::type-of-variable-reference '(float4-y x) type-env) 'float)
  (is (cl-cuda::type-of-variable-reference '(float4-z x) type-env) 'float)
  (is (cl-cuda::type-of-variable-reference '(float4-w x) type-env) 'float))


;;;
;;; test utilities
;;;

;; test cl-cuda-symbolicate
(is (cl-cuda::cl-cuda-symbolicate 'a   ) 'cl-cuda::a )
(is (cl-cuda::cl-cuda-symbolicate 'a 'b) 'cl-cuda::ab)


(finalize)
