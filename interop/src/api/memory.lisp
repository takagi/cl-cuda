#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-interop.api.memory
  (:use :cl :cl-reexport
        :cl-cuda.lang.type
        :cl-cuda-interop.driver-api)
  (:export ;; Memory block
           :alloc-memory-block
           :free-memory-block
           :memory-block-p
           :with-memory-block-device-ptr
           :memory-block-init-device-ptr
           :memory-block-release-device-ptr
           :memory-block-host-ptr
           :memory-block-type
           :memory-block-size
           :memory-block-vertex-buffer-object
           :memory-block-graphic-resource-ptr
           :with-memory-block
           :with-memory-blocks
           :sync-memory-block
           :memory-block-aref)
  (:import-from :cl-cuda.api.memory
                :alloc-host-memory
                :free-host-memory
                :host-memory-aref
                :memcpy-host-to-device
                :memcpy-device-to-host))
(in-package :cl-cuda-interop.api.memory)

(eval-when (:execute :load-toplevel :compile-toplevel)
  (reexport-from :cl-cuda.api.memory
                 :include '(:alloc-host-memory
                            :free-host-memory
                            :with-host-memory
                            :host-memory-aref
                            :memcpy-host-to-device
                            :memcpy-device-to-host)))


;;;
;;; Memory block
;;;

(defstruct (memory-block (:constructor %make-memory-block))
  (host-ptr :host-ptr :read-only t)
  (vertex-buffer-object :vertex-buffer-object :read-only t)
  (graphic-resource-ptr :graphic-resource-ptr :read-only t)
  (type :type :read-only t)
  (size :type :read-only t))

(defun bare-cffi-type (type)
  ;; BARE-CFFI-TYPE is a work around for cl-opengl's issue #41,
  ;; where GL-ARRAY accepts only bare references for structure types.
  (cl-pattern:match (cffi-type type)
    ((:struct type) type)
    (type type)))

(defun alloc-memory-block (type size)
  (let ((host-ptr (alloc-host-memory type size))
        (vbo (first (gl:gen-buffers 1)))
        (gres-ptr (cffi:foreign-alloc 'cu-graphics-resource
                                      :initial-element (cffi:null-pointer))))
    ;; create and initialize a buffer object's data store
    (gl:bind-buffer :array-buffer vbo)
    (let ((array (gl:alloc-gl-array (bare-cffi-type type) size)))
      (unwind-protect (gl:buffer-data :array-buffer :dynamic-draw array)
        (gl:free-gl-array array)))
    (gl:bind-buffer :array-buffer 0)
    ;; register a buffer object accessed through CUDA
    (cu-graphics-gl-register-buffer gres-ptr vbo
                                    cu-graphics-register-flags-none)
    ;; return a memory block
    (%make-memory-block :host-ptr host-ptr
                        :vertex-buffer-object vbo
                        :graphic-resource-ptr gres-ptr
                        :type type
                        :size size)))

(defun free-memory-block (memory-block)
  ;; unregister a buffer object
  (let ((gres (memory-block-graphic-resource memory-block)))
    (cu-graphics-unregister-resource gres))
  ;; free a pointer to a graphics resource
  (let ((gres-ptr (memory-block-graphic-resource-ptr memory-block)))
    (cffi:foreign-free gres-ptr))
  ;; delete a buffer object
  (let ((vbo (memory-block-vertex-buffer-object memory-block)))
    (gl:delete-buffers (list vbo)))
  ;; free host memory
  (let ((host-ptr (memory-block-host-ptr memory-block)))
    (free-host-memory host-ptr)))

(defun memory-block-graphic-resource (memory-block)
  (let ((gres-ptr (memory-block-graphic-resource-ptr memory-block)))
    (cffi:mem-ref gres-ptr 'cu-graphics-resource)))

(defun memory-block-init-device-ptr (memory-block)
  (let ((gres-ptr (memory-block-graphic-resource-ptr memory-block))
        (gres (memory-block-graphic-resource memory-block)))
    (cffi:with-foreign-objects ((device-ptr 'cu-device-ptr)
                                (size-ptr :unsigned-int))
      (cu-graphics-resource-set-map-flags gres
                                          cu-graphics-map-resource-flags-none)
      (cu-graphics-map-resources 1 gres-ptr (cffi:null-pointer))
      (cu-graphics-resource-get-mapped-pointer device-ptr size-ptr gres)
      (cffi:mem-ref device-ptr 'cu-device-ptr))))

(defun memory-block-release-device-ptr (memory-block)
  (let ((gres-ptr (memory-block-graphic-resource-ptr memory-block)))
    (cu-graphics-unmap-resources 1 gres-ptr (cffi:null-pointer))))

(defmacro with-memory-block-device-ptr ((device-ptr memory-block) &body body)
  `(let ((,device-ptr (memory-block-init-device-ptr ,memory-block)))
     (unwind-protect (progn ,@body)
       (memory-block-release-device-ptr ,memory-block))))

(defmacro with-memory-block ((var type size) &body body)
  `(let ((,var (alloc-memory-block ,type ,size)))
     (unwind-protect (progn ,@body)
       (free-memory-block ,var))))

(defmacro with-memory-blocks (bindings &body body)
  (if bindings
      `(with-memory-block ,(car bindings)
         (with-memory-blocks ,(cdr bindings)
           ,@body))
      `(progn ,@body)))

(defun sync-memory-block (memory-block direction)
  (declare ((member :host-to-device :device-to-host) direction))
  (let ((host-ptr (memory-block-host-ptr memory-block))
        (type (memory-block-type memory-block))
        (size (memory-block-size memory-block)))
    (with-memory-block-device-ptr (device-ptr memory-block)
      (ecase direction
        (:host-to-device
         (memcpy-host-to-device device-ptr host-ptr type size))
        (:device-to-host
         (memcpy-device-to-host host-ptr device-ptr type size))))))

(defun memory-block-aref (memory-block index)
  (let ((host-ptr (memory-block-host-ptr memory-block))
        (type (memory-block-type memory-block)))
    (host-memory-aref host-ptr type index)))

(defun (setf memory-block-aref) (new-value memory-block index)
  (let ((host-ptr (memory-block-host-ptr memory-block))
        (type (memory-block-type memory-block)))
    (setf (host-memory-aref host-ptr type index) new-value)))
