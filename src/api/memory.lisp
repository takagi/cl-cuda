#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda.api.memory
  (:use :cl
        :cl-cuda.driver-api
        :cl-cuda.lang.type)
  (:export ;; Device memory
           :device-total-bytes
           :device-total-kbytes
           :device-total-mbytes
           :device-total-gbytes
           :alloc-device-memory
           :free-device-memory
           :with-device-memory
           ;; Host memory
           :alloc-host-memory
           :free-host-memory
           :with-host-memory
           :host-memory-aref
           ;; Memcpy
           :memcpy-host-to-device
           :memcpy-device-to-host
           ;; Memory block
           :alloc-memory-block
           :free-memory-block
           :memory-block-p
           :memory-block-device-ptr
           :memory-block-host-ptr
           :memory-block-type
           :memory-block-size
           :with-memory-block
           :with-memory-blocks
           :sync-memory-block
           :memory-block-aref))
(in-package :cl-cuda.api.memory)


;;;
;;; Device memory
;;;

(defun alloc-device-memory (type n)
  (cffi:with-foreign-object (device-ptr 'cu-device-ptr)
    (cu-mem-alloc device-ptr (* n (cffi-type-size type)))
    (cffi:mem-ref device-ptr 'cu-device-ptr)))

(defun free-device-memory (device-ptr)
  (cu-mem-free device-ptr))

(defmacro with-device-memory ((var type n) &body body)
  `(let ((,var (alloc-device-memory ,type ,n)))
     (unwind-protect (progn ,@body)
       (free-device-memory ,var))))

(defun device-total-bytes (device)
  (cffi:with-foreign-objects ((bytes :unsigned-long)) ; size-t doesn't work for some reason...
    (cu-device-total-mem bytes device)
    (cffi:mem-ref bytes :unsigned-long)))

(defun device-total-kbytes (device)
  (/ (device-total-bytes device) 1024))

(defun device-total-mbytes (device)
  (/ (device-total-kbytes device) 1024))

(defun device-total-gbytes (device)
  (/ (device-total-mbytes device) 1024))

;;;
;;; Host memory
;;;

(defun alloc-host-memory (type n)
  (cffi:foreign-alloc (cffi-type type) :count n))

(defun free-host-memory (host-ptr)
  (cffi:foreign-free host-ptr))

(defmacro with-host-memory ((var type n) &body body)
  `(let ((,var (alloc-host-memory ,type ,n)))
     (unwind-protect (progn ,@body)
       (free-host-memory ,var))))

(defun host-memory-aref (host-ptr type index)
  ;; give type as constant explicitly for performance reason
  (let ((cffi-type (cffi-type type)))
    (cl-pattern:match cffi-type
      (:int (cffi:mem-aref host-ptr :int index))
      (:float (cffi:mem-aref host-ptr :float index))
      (:double (cffi:mem-aref host-ptr :double index))
      ((:boolean :int8) (cffi:mem-aref host-ptr '(:boolean :int8) index))
      ((:struct 'float3) (cffi:mem-aref host-ptr '(:struct float3) index))
      ((:struct 'float4) (cffi:mem-aref host-ptr '(:struct float4) index))
      ((:struct 'double3) (cffi:mem-aref host-ptr '(:struct double3) index))
      ((:struct 'double4) (cffi:mem-aref host-ptr '(:struct double4) index))
      (_ (error "The value ~S is an invalid CFFI type to access host memory." cffi-type)))))

(defun (setf host-memory-aref) (new-value host-ptr type index)
  ;; give type as constant explicitly for performance reason
  (let ((cffi-type (cffi-type type)))
    (cl-pattern:match cffi-type
      (:int (setf (cffi:mem-aref host-ptr :int index) new-value))
      (:float (setf (cffi:mem-aref host-ptr :float index) new-value))
      (:double (setf (cffi:mem-aref host-ptr :double index) new-value))
      ((:boolean :int8)
       (setf (cffi:mem-aref host-ptr '(:boolean :int8) index) new-value))
      ((:struct 'float3)
       (setf (cffi:mem-aref host-ptr '(:struct float3) index) new-value))
      ((:struct 'float4)
       (setf (cffi:mem-aref host-ptr '(:struct float4) index) new-value))
      ((:struct 'double3)
       (setf (cffi:mem-aref host-ptr '(:struct double3) index) new-value))
      ((:struct 'double4)
       (setf (cffi:mem-aref host-ptr '(:struct double4) index) new-value))
      (_ (error "The value ~S is an invalid CFFI type to access host memory."
                cffi-type)))))


;;;
;;; Memcpy
;;;

(defun memcpy-host-to-device (device-ptr host-ptr type n)
  (let ((size (cffi-type-size type)))
    (cu-memcpy-host-to-device device-ptr host-ptr (* n size))))

(defun memcpy-host-to-device-async (device-ptr host-ptr type n stream)
  (let ((size (cffi-type-size type)))
    (cu-memcpy-host-to-device-async device-ptr host-ptr (* n size) stream)))

(defun memcpy-device-to-host (host-ptr device-ptr type n)
  (let ((size (cffi-type-size type)))
    (cu-memcpy-device-to-host host-ptr device-ptr (* n size))))

(defun memcpy-device-to-host-async (host-ptr device-ptr type n stream)
  (let ((size (cffi-type-size type)))
    (cu-memcpy-device-to-host-async host-ptr device-ptr (* n size) stream)))


;;;
;;; Memory block
;;;

(defstruct (memory-block (:constructor %make-memory-block))
  (device-ptr :device-ptr :read-only t)
  (host-ptr :host-ptr :read-only t)
  (type :type :read-only t)
  (size :size :read-only t))

(defun alloc-memory-block (type size)
  (let ((device-ptr (alloc-device-memory type size))
        (host-ptr (alloc-host-memory type size)))
    (%make-memory-block :device-ptr device-ptr
                        :host-ptr host-ptr
                        :type type
                        :size size)))

(defun free-memory-block (memory-block)
  (let ((device-ptr (memory-block-device-ptr memory-block))
        (host-ptr (memory-block-host-ptr memory-block)))
    (free-device-memory device-ptr)
    (free-host-memory host-ptr)))

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
  (let ((device-ptr (memory-block-device-ptr memory-block))
        (host-ptr (memory-block-host-ptr memory-block))
        (type (memory-block-type memory-block))
        (size (memory-block-size memory-block)))
    (ecase direction
      (:host-to-device
       (memcpy-host-to-device device-ptr host-ptr type size))
      (:device-to-host
       (memcpy-device-to-host host-ptr device-ptr type size)))))

(defun memory-block-aref (memory-block index)
  (let ((host-ptr (memory-block-host-ptr memory-block))
        (type (memory-block-type memory-block)))
    (host-memory-aref host-ptr type index)))

(defun (setf memory-block-aref) (new-value memory-block index)
  (let ((host-ptr (memory-block-host-ptr memory-block))
        (type (memory-block-type memory-block)))
    (setf (host-memory-aref host-ptr type index) new-value)))
