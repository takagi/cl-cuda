#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-cuda-test-interop)

(setf *test-result-output* *standard-output*)

(plan nil)

(defclass test-window (glut:window) ()
  (:default-initargs :width 100 :height 100 :pos-x 100 :pos-y 100
                     :mode '(:double :rgb) :title "cl-cuda test window"))

(defmethod glut:display ((w test-window))
  
  ;; test alloc-memory-block/free-memory-block
  (let ((dev-id 0))
    (with-cuda-context (dev-id)
      (is-error (cl-cuda::alloc-memory-block 'int 1024 :interop t) simple-error)))
  
  ;; test alloc-memory-block/free-memory-block with OpenGL interoperability
  (let ((dev-id 0))
    (with-cuda-context (dev-id :interop t)
      (let (blk)
        (ok (setf blk (cl-cuda::alloc-memory-block 'int 1024)))
        (cl-cuda::free-memory-block blk))
      (let (blk)
        (ok (setf blk (cl-cuda::alloc-memory-block 'int 1024 :interop t)))
        (cl-cuda::free-memory-block blk))))
  
  ;; test selectors of memory-block with OpenGL interoperability
  (let ((dev-id 0))
    (with-cuda-context (dev-id :interop t)
      (with-memory-blocks ((blk 'int 1024 :interop t))
        (ok       (cl-cuda::memory-block-cffi-ptr blk))
        (is-error (cl-cuda::memory-block-device-ptr blk) simple-error)
        (cl-cuda::with-memory-block-device-ptr (device-ptr blk)
          (ok device-ptr))
        (ok       (cl-cuda::memory-block-interop-p blk))
        (ok       (cl-cuda::memory-block-vertex-buffer-object blk))
        (ok       (cl-cuda::memory-block-graphic-resource-ptr blk))
        (is       (cl-cuda::memory-block-type blk) 'int)
        (is       (cl-cuda::memory-block-cffi-type blk) :int)
        (is       (cl-cuda::memory-block-length blk) 1024)
        (is       (cl-cuda::memory-block-bytes blk) (* 1024 4))
        (is       (cl-cuda::memory-block-element-bytes blk) 4))))
  
  ;; test setf functions of memory-block with OpenGL interoperability
  (let ((dev-id 0))
    (with-cuda-context (dev-id :interop t)
      ;; int array
      (with-memory-blocks ((x 'int 1 :interop t))
        (setf (mem-aref x 0) 1)
        (is   (mem-aref x 0) 1))
      ;; float array
      (with-memory-blocks ((x 'float 1 :interop t))
        (setf (mem-aref x 0) 1.0)
        (is   (mem-aref x 0) 1.0))
      ;; float3 array
      (with-memory-blocks ((x 'float3 1 :interop t))
        (setf (mem-aref x 0) (make-float3 1.0 1.0 1.0))
        (is   (mem-aref x 0) (make-float3 1.0 1.0 1.0) :test #'float3-=))
      ;; float4 array
      (with-memory-blocks ((x 'float4 1 :interop t))
        (setf (mem-aref x 0) (make-float4 1.0 1.0 1.0 1.0))
        (is   (mem-aref x 0) (make-float4 1.0 1.0 1.0 1.0) :test #'float4-=))
      ;; error cases
      (with-memory-blocks ((x 'int 1 :interop t))
        (is-error (mem-aref x -1) simple-error)
        (is-error (setf (mem-aref x -1) 0) simple-error)
        (is-error (mem-aref x 1) simple-error)
        (is-error (setf (mem-aref x 1) 0) simple-error))))
  
  ;; test set statement on memory-block with OpenGL interoperability
  (defkernel test-memcpy (void ((x int*) (y float*) (z float3*)))
    (set (aref x 0) (+ (aref x 0) 1))
    (set (aref y 0) (+ (aref y 0) 1.0))
    (set (float3-x (aref z 0)) (+ (float3-x (aref z 0)) 1.0)) ; vector math wanted
    (set (float3-y (aref z 0)) (+ (float3-y (aref z 0)) 1.0))
    (set (float3-z (aref z 0)) (+ (float3-z (aref z 0)) 1.0)))
  
  (let ((dev-id 0))
    (with-cuda-context (dev-id :interop t)
      (with-memory-blocks ((x 'int   1 :interop t)
                           (y 'float 1 :interop t)
                           (z 'float3 1 :interop t))
        (setf (mem-aref x 0) 1)
        (setf (mem-aref y 0) 1.0)
        (setf (mem-aref z 0) (make-float3 1.0 1.0 1.0))
        (memcpy-host-to-device x y z)
        (test-memcpy x y z :grid-dim  '(1 1 1)
                     :block-dim '(1 1 1))
        (memcpy-device-to-host x y z)
        (is (mem-aref x 0) 2)
        (is (mem-aref y 0) 2.0)
        (is (mem-aref z 0) (make-float3 2.0 2.0 2.0) :test #'float3-=))))
  (format t "press 'Q' on the test window to quit~%"))

(defmethod glut:keyboard ((w test-window) key x y)
  (case key
    (#\q (glut:destroy-window (glut:id w))
         (glut:leave-main-loop))))

(let ((glut:*run-main-loop-after-display* t))
  (glut:display-window (make-instance 'test-window)))

(finalize)

