#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-cuda-test)

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

(defkernel vec-add-kernel (cu-device-ptr cu-device-ptr cu-device-ptr :int)
  "VecAdd_kernel")

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
      (format t "CUDA function \"VecAdd_kernel\" is launched.~%")
      (check-cuda-errors
       (cu-memcpy-device-to-host h-c
                                 (cffi:mem-ref d-c 'cu-device-ptr)
                                 size))
      (verify-result h-a h-b h-c n)))))


(finalize)
