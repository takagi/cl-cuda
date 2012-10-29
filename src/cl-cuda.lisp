#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-cuda)


;;;
;;; Load CUDA library
;;;

(cffi:define-foreign-library libcuda
  (t (:default "/usr/local/cuda/lib/libcuda")))
(cffi:use-foreign-library libcuda)



;;;
;;; Definition of defcufun macro
;;;

(defmacro defcufun (name-and-options return-type &body args)
  (let* ((name (car name-and-options))
         (name% (symbolicate name "%"))
         (name-and-options% (cons name% (cdr name-and-options)))
         (params (mapcar #'car args)))
      `(progn
         (defun ,name (,@params)
           (check-cuda-errors ',name (,name% ,@params)))
         (cffi:defcfun ,name-and-options% ,return-type ,@args))))


;;;
;;; Definition of defcuenum macro
;;;

(eval-when (:compile-toplevel :load-toplevel)
  (defun enum-keyword (enum-elem)
    (match (ensure-list enum-elem)
      ((keyword) keyword)
      ((keyword _) keyword)
      (_  (error (format nil "invalid enum element: ~A" enum-elem))))))

(eval-when (:compile-toplevel :load-toplevel)
  (defun enum-value (enum-elem)
    (match enum-elem
      ((_ value) value)
      (_ (error (format nil "invalid enum element: ~A" enum-elem))))))
  
(eval-when (:compile-toplevel :load-toplevel)
  (defun defconstant-enum-value (name enum-elem)
    (let ((keyword (enum-keyword enum-elem)))
      `(defconstant ,(symbolicate keyword)
         (cffi:foreign-enum-value ',name ,keyword)))))

(defmacro defcuenum (name-and-options &body enum-list)
  (let ((name name-and-options))
    `(progn
       (cffi:defcenum ,name
         ,@enum-list)
       ,@(mapcar (lambda (enum-elem)
                   (defconstant-enum-value name enum-elem))
                 enum-list))))


;;;
;;; Definition of CUDA driver API types
;;;

(cffi:defctype cu-result :unsigned-int)
(cffi:defctype cu-device :int)
(cffi:defctype cu-context :pointer)
(cffi:defctype cu-module :pointer)
(cffi:defctype cu-function :pointer)
(cffi:defctype cu-stream :pointer)
(cffi:defctype cu-device-ptr :unsigned-int)
(cffi:defctype cu-event :pointer)
(cffi:defctype size-t :unsigned-int)


;;;
;;; Definition of CUDA driver API enums
;;;

(defcuenum cu-event-flags-enum
  (:cu-event-default #X0)
  (:cu-event-blocking-sync #X1)
  (:cu-event-disable-timing #X2)
  (:cu-event-interprocess #X4))


;;;
;;; Definition of CUDA driver API functions
;;;

;; cuInit
(defcufun (cu-init "cuInit") cu-result (flags :unsigned-int))

;; cuDeviceGet
(defcufun (cu-device-get "cuDeviceGet") cu-result
  (device (:pointer cu-device))
  (ordinal :int))

;; cuDeviceGetCount
(defcufun (cu-device-get-count "cuDeviceGetCount") cu-result
  (count (:pointer :int)))

;; cuDeviceComputeCapability
(defcufun (cu-device-compute-capability "cuDeviceComputeCapability") cu-result
  (major (:pointer :int))
  (minor (:pointer :int))
  (dev cu-device))

;; cuDeviceGetName
(defcufun (cu-device-get-name "cuDeviceGetName") cu-result
  (name :string)
  (len :int)
  (dev cu-device))

;; cuCtxCreate
(defcufun (cu-ctx-create "cuCtxCreate") cu-result
  (pctx (:pointer cu-context))
  (flags :unsigned-int)
  (dev cu-device))

;; cuCtxDestroy
(defcufun (cu-ctx-destroy "cuCtxDestroy") cu-result
  (pctx cu-context))

;; cuCtxSynchronize
(defcufun (cu-ctx-synchronize "cuCtxSynchronize") cu-result)

;; cuMemAlloc
(defcufun (cu-mem-alloc "cuMemAlloc") cu-result
  (dptr (:pointer cu-device-ptr))
  (bytesize size-t))

;; cuMemFree
(defcufun (cu-mem-free "cuMemFree") cu-result
  (dptr cu-device-ptr))

;; cuMemcpyHtoD
(defcufun (cu-memcpy-host-to-device "cuMemcpyHtoD") cu-result
  (dst-device cu-device-ptr)
  (src-host :pointer)
  (byte-count size-t))

;; cuMemcpyDtoH
(defcufun (cu-memcpy-device-to-host "cuMemcpyDtoH") cu-result
  (dst-host :pointer)
  (src-device cu-device-ptr)
  (byte-count size-t))

;; cuModuleLoad
(defcufun (cu-module-load "cuModuleLoad") cu-result
  (module (:pointer cu-module))
  (fname :string))

;; cuModuleUnload
(defcufun (cu-module-unload "cuModuleUnload") cu-result
  (module cu-module))

;; cuModuleGetFunction
(defcufun (cu-module-get-function "cuModuleGetFunction") cu-result
  (hfunc (:pointer cu-function))
  (hmod cu-module)
  (name :string))

;; cuLaunchKernel
(defcufun (cu-launch-kernel "cuLaunchKernel") cu-result
  (f cu-function)
  (grid-dim-x :unsigned-int)
  (grid-dim-y :unsigned-int)
  (grid-dim-z :unsigned-int)
  (block-dim-x :unsigned-int)
  (block-dim-y :unsigned-int)
  (block-dim-z :unsigned-int)
  (shared-mem-bytes :unsigned-int)
  (hstream cu-stream)
  (kernel-params (:pointer :pointer))
  (extra (:pointer :pointer)))

;; cuEventCreate
(defcufun (cu-event-create "cuEventCreate") cu-result
  (phevent (:pointer cu-event))
  (flags :unsigned-int))

;; cuEventDestroy
(defcufun (cu-event-destroy "cuEventDestroy") cu-result
  (h-event cu-event))

;; cuEventElapsedTime
(defcufun (cu-event-elapsed-time "cuEventElapsedTime") cu-result
  (pmilliseconds (:pointer :float))
  (hstart cu-event)
  (hend cu-event))

;; cuEventQuery
(defcufun (cu-event-query "cuEventQuery") cu-result
  (hevent cu-event))

;; cuEventRecord
(defcufun (cu-event-record "cuEventRecord") cu-result
  (hevent cu-event)
  (hstream cu-stream))

;; cuEventSynchronize
(defcufun (cu-event-synchronize "cuEventSynchronize") cu-result
  (hevent cu-event))

;; check-cuda-errors function
(defvar +cuda-success+ 0)
(defvar *show-messages* t)

(defun check-cuda-errors (name return-code)
  (unless (= return-code +cuda-success+)
    (error (format nil "~A failed with driver API error No. ~A.~%"
                       name return-code)))
  (when *show-messages*
    (format t "~A succeeded.~%" name))
  (values))


;;;
;;; Definition of with- macro for CUDA driver API
;;;

(defmacro with-cuda-context ((dev-id) &body body)
  `(progn
     (init-cuda-context ,dev-id)
     (unwind-protect (progn ,@body)
       (release-cuda-context))))

(let (device context)
  
  (defun init-cuda-context (dev-id)
    (let ((flags 0))
      (setf device  (cffi:foreign-alloc 'cu-device)
            context (cffi:foreign-alloc 'cu-context))
      (cu-init 0)
      (cu-device-get device dev-id)
      (cu-ctx-create context flags (cffi:mem-ref device 'cu-device))))
  
  (defun release-cuda-context ()
    (kernel-manager-unload *kernel-manager*)
    (cu-ctx-destroy (cffi:mem-ref context 'cu-context))
    (cffi:foreign-free context)
    (cffi:foreign-free device))
  
  (defun synchronize-context ()
    (cu-ctx-synchronize)))

(defmacro with-cuda-memory-block (args &body body)
  (destructuring-bind (dptr size) args
    `(cffi:with-foreign-object (,dptr 'cu-device-ptr)
       (cu-mem-alloc ,dptr ,size)
       (unwind-protect
            (progn ,@body)
         (cu-mem-free (cffi:mem-ref ,dptr 'cu-device-ptr))))))

(defmacro with-cuda-memory-blocks (bindings &body body)
  (if bindings
      `(with-cuda-memory-block ,(car bindings)
         (with-cuda-memory-blocks ,(cdr bindings)
           ,@body))
      `(progn ,@body)))


;;;
;;; Definition of Built-in Vector Types
;;;

(defstruct (float3 (:constructor make-float3 (x y z)))
  (x 0.0 :type single-float)
  (y 0.0 :type single-float)
  (z 0.0 :type single-float))

(defun float3-= (a b)
  (and (= (float3-x a) (float3-x b))
       (= (float3-y a) (float3-y b))
       (= (float3-z a) (float3-z b))))

(cffi:defcstruct float3
  (x :float)
  (y :float)
  (z :float))

(defstruct (float4 (:constructor make-float4 (x y z w)))
  (x 0.0 :type single-float)
  (y 0.0 :type single-float)
  (z 0.0 :type single-float)
  (w 0.0 :type single-float))

(defun float4-= (a b)
  (and (= (float4-x a) (float4-x b))
       (= (float4-y a) (float4-y b))
       (= (float4-z a) (float4-z b))
       (= (float4-w a) (float4-w b))))

(cffi:defcstruct float4
  (x :float)
  (y :float)
  (z :float)
  (w :float))


;;;
;;; Definition of cl-cuda types
;;;

(defvar +basic-types+ '(void 0
                        int 4
                        float 4))

(defvar +vector-type-table+ '(float3 float4))

(defvar +vector-type-elements+ '(x y z w))

(defun basic-type-p (type)
  (and (getf +basic-types+ type)
       t))

(defun vector-type-p (type)
  (and (find type +vector-type-table+)
       t))

(defun vector-type-length (type)
  ;; e.g. float3 => 3
  (unless (vector-type-p type)
    (error (format nil "invalid type: ~A" type)))
  (let ((str (symbol-name type)))
    (parse-integer (subseq str (1- (length str))))))

(defun vector-type-base-type (type)
  ;; e.g. float3 => float
  (unless (vector-type-p type)
    (error (format nil "invalid type: ~A" type)))
  (cl-cuda-symbolicate (reverse (subseq (reverse (princ-to-string type)) 1))))

(defun vector-type-selector-symbol (type elm)
  ;; e.g. float3 x => float3-x
  (unless (vector-type-p type)
    (error (format nil "invalid type: ~A" type)))
  (unless (find elm +vector-type-elements+)
    (error (format nil "invalid element: ~A" elm)))
  (cl-cuda-symbolicate (princ-to-string type) "-" (princ-to-string elm)))

(defun vector-type-selector-return-type (selector)
  ;; e.g. float3-x => float
  (unless (find selector (vector-type-selector-symbols))
    (error (format nil "invalid selector: ~A" selector)))
  (cl-cuda-symbolicate
    (reverse (subseq (reverse (princ-to-string selector)) 3))))

(defun vector-type-selector-symbols ()
  (labels ((aux (type)
             (loop repeat (vector-type-length type)
                   for elm in +vector-type-elements+
                collect (vector-type-selector-symbol type elm))))
    (flatten
      (mapcar #'aux +vector-type-table+))))

(defun non-pointer-type-p (type)
  (or (basic-type-p type)
      (vector-type-p type)))

(defun pointer-type-p (type)
  (let ((type-str (symbol-name type)))
    (let ((last (aref type-str (1- (length type-str)))))
      (and (eq last #\*)
           (non-pointer-type-p (remove-star type))
           t))))

(defun valid-type-p (type)
  (or (pointer-type-p type)
      (non-pointer-type-p type)))

(defun add-star (type n)
  (labels ((aux (str n2)
             (if (< n2 1)
                 str
                 (aux (concatenate 'string str "*") (1- n2)))))
    (cl-cuda-symbolicate (aux (princ-to-string type) n))))

(defun remove-star (type)
  (let ((rev (reverse (princ-to-string type))))
    (if (string= (subseq rev 0 1) "*")
        (remove-star (cl-cuda-symbolicate (reverse (subseq rev 1))))
        type)))

(defun type-dimension (type)
  (unless (valid-type-p type)
    (error (format nil "invalid type: ~A" type)))
  (count #\* (princ-to-string type)))

(defvar +cffi-type-table+ '(int :int
                            float :float))

(defun cffi-type (type)
  (acond
    ((pointer-type-p type) 'cu-device-ptr)
    ((getf +cffi-type-table+ type) it)
    ((vector-type-p type) type)
    (t (error (format nil "invalid type: ~A" type)))))

(defun basic-type-size (type)
  (getf +basic-types+ type))

(defun vector-type-size (type)
  (* (vector-type-length type)
     (size-of (vector-type-base-type type))))

(defun size-of (type)
  (cond
    ((pointer-type-p type) 4)
    ((basic-type-p type) (basic-type-size type))
    ((vector-type-p type) (vector-type-size type))
    (t (error (format nil "invalid type: ~A" type)))))


;;;
;;; Definition of Memory Block
;;;

(defun alloc-memory-block (type n)
  (unless (non-pointer-type-p type)
    (error (format nil "invalid type: ~A" type)))
  (let ((cffi-ptr (cffi:foreign-alloc (cffi-type type) :count n))
        (device-ptr (cffi:foreign-alloc 'cu-device-ptr)))
    (cu-mem-alloc device-ptr (* n (size-of type)))
    (list cffi-ptr device-ptr type n)))

(defun free-memory-block (blk)
  (if blk
      (let ((device-ptr (memory-block-device-ptr blk))
            (cffi-ptr (memory-block-cffi-ptr blk)))
        (cu-mem-free (cffi:mem-ref device-ptr 'cu-device-ptr))
        (cffi:foreign-free device-ptr)
        (cffi:foreign-free cffi-ptr)
        (setf device-ptr nil)
        (setf cffi-ptr nil))))

(defun memory-block-cffi-ptr (blk)
  (car blk))

(defun memory-block-device-ptr (blk)
  (cadr blk))

(defun memory-block-type (blk)
  (caddr blk))

(defun memory-block-cffi-type (blk)
  (cffi-type (memory-block-type blk)))

(defun memory-block-length (blk)
  (cadddr blk))

(defun memory-block-bytes (blk)
  (* (memory-block-element-bytes blk)
     (memory-block-length blk)))

(defun memory-block-element-bytes (blk)
  (size-of (memory-block-type blk)))

(defun memory-block-binding-var (binding)
  (match binding
    ((var _ _) var)
    (_ (error (format nil "invalid memory block binding: ~A" binding)))))

(defun memory-block-binding-type (binding)
  (match binding
    ((_ type _) type)
    (_ (error (format nil "invalid memory block binding: ~A" binding)))))

(defun memory-block-binding-size (binding)
  (match binding
    ((_ _ size) size)
    (_ (error (format nil "invalid memory block binding: ~A" binding)))))

(defun alloc-memory-block-form (binding)
  (let ((var (memory-block-binding-var binding))
        (type (memory-block-binding-type binding))
        (size (memory-block-binding-size binding)))
    `(setf ,var (alloc-memory-block ,type ,size))))

(defun free-memory-block-form (binding)
  `(free-memory-block ,(memory-block-binding-var binding)))

(defmacro with-memory-blocks (bindings &body body)
  `(let ,(mapcar #'memory-block-binding-var bindings)
     (unwind-protect
          (progn
            ,@(mapcar #'alloc-memory-block-form bindings)
            ,@body)
       (progn
         ,@(mapcar #'free-memory-block-form bindings)))))

(defun basic-type-mem-aref (blk idx)
  ;; give type as constant explicitly for better performance
  (case (memory-block-type blk)
    (int (cffi:mem-aref (memory-block-cffi-ptr blk) :int idx))
    (float (cffi:mem-aref (memory-block-cffi-ptr blk) :float idx))
    (t (error "must not be reached"))))

(defun float3-mem-aref (blk idx)
  ;; give type and slot names as constant explicitly for better performance
  (let ((ptr (cffi:mem-aref (memory-block-cffi-ptr blk) 'float3 idx)))
    (make-float3 (cffi:foreign-slot-value ptr 'float3 'x)
                 (cffi:foreign-slot-value ptr 'float3 'y)
                 (cffi:foreign-slot-value ptr 'float3 'z))))

(defun float4-mem-aref (blk idx)
  ;; give type and slot names as constant explicitly for better performance
  (let ((ptr (cffi:mem-aref (memory-block-cffi-ptr blk) 'float4 idx)))
    (make-float4 (cffi:foreign-slot-value ptr 'float4 'x)
                 (cffi:foreign-slot-value ptr 'float4 'y)
                 (cffi:foreign-slot-value ptr 'float4 'z)
                 (cffi:foreign-slot-value ptr 'float4 'w))))
                 
(defun vector-type-mem-aref (blk idx)
  (case (memory-block-type blk)
    (float3 (float3-mem-aref blk idx))
    (float4 (float4-mem-aref blk idx))
    (t (error "must not be reached"))))

(defun mem-aref (blk idx)
  (unless (and (<= 0 idx) (< idx (memory-block-length blk)))
    (error (format nil "invalid index: ~A" idx)))
  (let ((type (memory-block-type blk)))
    (cond
      ((basic-type-p type) (basic-type-mem-aref blk idx))
      ((vector-type-p type) (vector-type-mem-aref blk idx))
      (t (error "must not be reached")))))

(defun basic-type-setf-mem-aref (blk idx val)
  ;; give type as constant explicitly for better performance
  (case (memory-block-type blk)
    (int (setf (cffi:mem-aref (memory-block-cffi-ptr blk) :int idx) val))
    (float (setf (cffi:mem-aref (memory-block-cffi-ptr blk) :float idx) val))
    (t (error "must not be reached"))))

(defun float3-setf-mem-aref (blk idx val)
  ;; give type and slot names as constant explicitly for better performance
  (let ((ptr (cffi:mem-aref (memory-block-cffi-ptr blk) 'float3 idx)))
    (setf (cffi:foreign-slot-value ptr 'float3 'x) (float3-x val))
    (setf (cffi:foreign-slot-value ptr 'float3 'y) (float3-y val))
    (setf (cffi:foreign-slot-value ptr 'float3 'z) (float3-z val))))

(defun float4-setf-mem-aref (blk idx val)
  ;; give type and slot names as constant explicitly for better performance
  (let ((ptr (cffi:mem-aref (memory-block-cffi-ptr blk) 'float4 idx)))
    (setf (cffi:foreign-slot-value ptr 'float4 'x) (float4-x val))
    (setf (cffi:foreign-slot-value ptr 'float4 'y) (float4-y val))
    (setf (cffi:foreign-slot-value ptr 'float4 'z) (float4-z val))
    (setf (cffi:foreign-slot-value ptr 'float4 'w) (float4-w val))))

(defun vector-type-setf-mem-aref (blk idx val)
  (case (memory-block-type blk)
    (float3 (float3-setf-mem-aref blk idx val))
    (float4 (float4-setf-mem-aref blk idx val))
    (t (error "must not be unreached"))))

(defun (setf mem-aref) (val blk idx)
  (unless (and (<= 0 idx) (< idx (memory-block-length blk)))
    (error (format nil "invalid index: ~A" idx)))
  (let ((type (memory-block-type blk)))
    (cond
      ((basic-type-p type) (basic-type-setf-mem-aref blk idx val))
      ((vector-type-p type) (vector-type-setf-mem-aref blk idx val))
      (t (error "must not be reached")))))

(defun memcpy-host-to-device (&rest blks)
  (dolist (blk blks)
    (let ((device-ptr (memory-block-device-ptr blk))
          (cffi-ptr (memory-block-cffi-ptr blk))
          (bytes (memory-block-bytes blk)))
      (cu-memcpy-host-to-device (cffi:mem-ref device-ptr 'cu-device-ptr)
                                cffi-ptr
                                bytes))))

(defun memcpy-device-to-host (&rest blks)
  (dolist (blk blks)
    (let ((device-ptr (memory-block-device-ptr blk))
          (cffi-ptr (memory-block-cffi-ptr blk))
          (bytes (memory-block-bytes blk)))
      (cu-memcpy-device-to-host cffi-ptr
                                (cffi:mem-ref device-ptr 'cu-device-ptr)
                                bytes))))


;;;
;;; Definition of defkernel macro
;;;

(defmacro with-module-and-function (args &body body)
  (destructuring-bind (hfunc module function) args
    (with-gensyms (module-name func-name hmodule)
      `(cffi:with-foreign-string (,module-name ,module)
         (cffi:with-foreign-string (,func-name ,function)
           (cffi:with-foreign-objects ((,hmodule 'cu-module)
                                       (,hfunc 'cu-function))
             (cu-module-load ,hmodule ,module-name)
             (cu-module-get-function ,hfunc (cffi:mem-ref ,hmodule 'cu-module)
                                     ,func-name)
             ,@body))))))

(defun foreign-pointer-setf-vector-type (var var-ptr type)
  (let ((n (vector-type-length type)))
    `(progn
       ,@(loop repeat n
               for elm in +vector-type-elements+
            collect `(setf (cffi:foreign-slot-value ,var-ptr ',type ',elm)
                           (,(vector-type-selector-symbol type elm) ,var))))))

(defun foreign-pointer-setf-else (var var-ptr type)
  `(setf (cffi:mem-ref ,var-ptr ,type) ,var))

(defun foreign-pointer-setf (binding)
  (destructuring-bind (var var-ptr type) binding
    (if (vector-type-p type)
        (foreign-pointer-setf-vector-type var var-ptr type)
        (foreign-pointer-setf-else var var-ptr type))))

(defmacro with-non-pointer-arguments (bindings &body body)
  (if bindings
      (labels ((ptr-type-pair (binding)
                 (destructuring-bind (_ var-ptr type) binding
                   (declare (ignorable _))
                   (if (vector-type-p type)
                       `(,var-ptr ',type)
                       `(,var-ptr ,type)))))
        `(cffi:with-foreign-objects (,@(mapcar #'ptr-type-pair bindings))
           ,@(mapcar #'foreign-pointer-setf bindings)
           ,@body))
      `(progn ,@body)))

(defmacro with-kernel-arguments (args &body body)
  (let ((var (car args))
        (ptrs (cdr args)))
    `(cffi:with-foreign-object (,var :pointer ,(length ptrs))
       ,@(loop for ptr in ptrs
            for i from 0
            collect `(setf (cffi:mem-aref ,var :pointer ,i) ,ptr))
       ,@body)))

(defmacro defkernel (name args &body body)
  (destructuring-bind (return-type arg-bindings) args
    (with-gensyms (hfunc kargs)
      `(progn
         (kernel-manager-define-function *kernel-manager* ',name ',return-type ',arg-bindings ',body)
         (defun ,name (,@(kernel-arg-names arg-bindings) &key grid-dim block-dim)
           (let ((,hfunc (ensure-kernel-function-loaded *kernel-manager* ',name)))
             (with-non-pointer-arguments
                 ,(kernel-arg-foreign-pointer-bindings arg-bindings)
               (with-kernel-arguments
                   (,kargs ,@(kernel-arg-names-as-pointer arg-bindings))
                 (destructuring-bind
                       (grid-dim-x grid-dim-y grid-dim-z) grid-dim
                 (destructuring-bind
                       (block-dim-x block-dim-y block-dim-z) block-dim
                   (cu-launch-kernel (cffi:mem-ref ,hfunc 'cu-function)
                                     grid-dim-x grid-dim-y grid-dim-z
                                     block-dim-x block-dim-y block-dim-z
                                     0 (cffi:null-pointer)
                                     ,kargs (cffi:null-pointer))))))))))))


;;; kernel-arg

(defun kernel-arg-names (arg-bindings)
  ;; ((a float*) (b float*) (c float*) (n int)) -> (a b c n)
  (mapcar #'car arg-bindings))

(defun kernel-arg-names-as-pointer (arg-bindings)
  ;; ((a float*) (b float*) (c float*) (n int)) -> (a b c n-ptr)
  (mapcar #'arg-name-as-pointer arg-bindings))

(defun arg-name-as-pointer (arg-binding)
  ; (a float*) -> a, (n int) -> n-ptr
  (destructuring-bind (var type) arg-binding
    (unless (valid-type-p type)
      (error (format nil "invalid type: ~A" type)))
    (if (non-pointer-type-p type)
        (var-ptr var)
        `(memory-block-device-ptr ,var))))

(defun kernel-arg-foreign-pointer-bindings (arg-bindings)
  ; ((a float*) (b float*) (c float*) (n int)) -> ((n n-ptr :int))
  (mapcar #'foreign-pointer-binding
    (remove-if-not #'arg-binding-with-non-pointer-type-p arg-bindings)))

(defun foreign-pointer-binding (arg-binding)
  (destructuring-bind (var type) arg-binding
    (list var (var-ptr var) (cffi-type type))))

(defun arg-binding-with-non-pointer-type-p (arg-binding)
  (let ((type (cadr arg-binding)))
    (unless (valid-type-p type)
      (error (format nil "invalid type: ~A" type)))
    (non-pointer-type-p type)))

(defun var-ptr (var)
  (symbolicate var "-PTR"))


;;;
;;; Definition of Kernel Manager
;;;

;;; function-info
;;; <function-info> ::= (<name> <return-type> <arguments> <body>)

(defun make-function-info (name return-type args body)
  (list name return-type args body))

(defun function-name (info)
  (car info))

(defun function-c-name (info)
  (let ((name (function-name info)))
    (let ((package-name (compile-identifier (package-name (symbol-package name))))
          (function-name (compile-identifier name)))
      (concatenate 'string package-name "_" function-name))))

(defun function-return-type (info)
  (cadr info))

(defun function-arguments (info)
  (caddr info))

(defun function-argument-types (info)
  (mapcar #'cadr (function-arguments info)))

(defun function-body (info)
  (cadddr info))


;;; kernel-definition
;;; <kernel-definition>     ::= (<kernel-function-table> <kernel-constant-table>)
;;; <kernel-function-table> ::= alist { <function-name> => <function-info> }
;;; <kernel-constant-table> ::= alist { <constant-name> => <constant-info> }

(defun empty-kernel-definition ()
  (list nil nil))

(defun function-table (def)
  (car def))

(defun constant-table (def)
  (cadr def))

(defun function-info (name def)
  (or (assoc name (function-table def))
      (error (format nil "undefined kernel function: ~A" name))))

(defun constant-info (name def)
  (or (assoc name (constant-table def))
      (error (format nil "undefined kernel constant: ~A" name))))

(defun define-kernel-function (name return-type args body def)
  (let ((func-table (function-table def))
        (const-table (constant-table def)))
    (let ((func (make-function-info name return-type args body))
          (rest (remove name func-table :key #'car)))
      (list (cons func rest) const-table))))

(defun undefine-kernel-function (name def)
  (unless (kernel-definition-function-exists-p name def)
    (error (format nil "undefined kernel function: ~A" name)))
  (let ((func-table (function-table def))
        (const-table (constant-table def)))
    (list (remove name func-table :key #'car)
          const-table)))

(defun define-kernel-constant (name value def)
  (declare (ignorable name value def))
  (undefined))

(defun undefine-kernel-constant (name def)
  (declare (ignorable name def))
  (undefined))

(defun kernel-definition-function-exists-p (name def)
  (and (assoc name (function-table def))
       t))

(defun kernel-definition-function-names (def)
  (mapcar #'car (function-table def)))

(defun kernel-definition-function-name (name def)
  (function-name (function-info name def)))

(defun kernel-definition-function-c-name (name def)
  (function-c-name (function-info name def)))

(defun kernel-definition-function-return-type (name def)
  (function-return-type (function-info name def)))

(defun kernel-definition-function-arguments (name def)
  (function-arguments (function-info name def)))

(defun kernel-definition-function-argument-types (name def)
  (function-argument-types (function-info name def)))

(defun kernel-definition-function-body (name def)
  (function-body (function-info name def)))


;;; module-info
;;; <module-info> ::= (<module-handle> <module-path> <module-compilation-needed> <function-handles>)
;;; <function-handles> ::= hashtable { <function-name> => <function-handle> }

(defun make-module-info ()
  (list nil nil t (make-hash-table)))

(defun module-handle (info)
  (car info))

(defun (setf module-handle) (val info)
  (setf (car info) val))

(defun module-path (info)
  (cadr info))

(defun (setf module-path) (val info)
  (setf (cadr info) val))

(defun module-compilation-needed (info)
  (caddr info))

(defun (setf module-compilation-needed) (val info)
  (setf (caddr info) val))

(defun function-handles (info)
  (cadddr info))

(defun function-handle (info name)
  (gethash name (function-handles info)))

(defun (setf function-handle) (handle info name)
  (setf (gethash name (function-handles info)) handle))


;;; kernel-manager
;;; <kernel-manager> ::= (<module-info> <kernel-definition>)

(defun make-kernel-manager()
  (list (make-module-info) (empty-kernel-definition)))

(defun module-info (mgr)
  (car mgr))

(defun kernel-definition (mgr)
  (cadr mgr))

(defun (setf kernel-definition) (val mgr)
  (setf (cadr mgr) val))

(defun kernel-manager-module-handle (mgr)
  (module-handle (module-info mgr)))

(defun (setf kernel-manager-module-handle) (handle mgr)
  (setf (module-handle (module-info mgr)) handle))

(defun kernel-manager-module-path (mgr)
  (module-path (module-info mgr)))

(defun (setf kernel-manager-module-path) (val mgr)
  (setf (module-path (module-info mgr)) val))

(defun kernel-manager-module-compilation-needed (mgr)
  (module-compilation-needed (module-info mgr)))

(defun (setf kernel-manager-module-compilation-needed) (val mgr)
  (setf (module-compilation-needed (module-info mgr)) val))

(defun kernel-manager-function-handle (mgr name)
  (function-handle (module-info mgr) name))

(defun (setf kernel-manager-function-handle) (val mgr name)
  (setf (function-handle (module-info mgr) name) val))

(defun kernel-manager-function-exists-p (mgr name)
  (kernel-definition-function-exists-p name (kernel-definition mgr)))

(defun kernel-manager-function-names (mgr)
  (kernel-definition-function-names (kernel-definition mgr)))

(defun kernel-manager-function-name (mgr name)
  (kernel-definition-function-name name (kernel-definition mgr)))

(defun kernel-manager-function-c-name (mgr name)
  (kernel-definition-function-c-name name (kernel-definition mgr)))

(defun kernel-manager-function-return-type (mgr name)
  (kernel-definition-function-return-type name (kernel-definition mgr)))

(defun kernel-manager-function-arguments (mgr name)
  (kernel-definition-function-arguments name (kernel-definition mgr)))

(defun kernel-manager-function-argument-types (mgr name)
  (kernel-definition-function-argument-types name (kernel-definition mgr)))

(defun kernel-manager-function-body (mgr name)
  (kernel-definition-function-body name (kernel-definition mgr)))

(defun kernel-manager-define-function (mgr name return-type args body)
  (when (or (not (kernel-manager-function-exists-p mgr name))
            (function-modified-p mgr name return-type args body))
    (setf (kernel-definition mgr)
          (define-kernel-function name return-type args body (kernel-definition mgr)))
    (setf (kernel-manager-module-compilation-needed mgr) t)))

(defun function-modified-p (mgr name return-type args body)
  (not (and (equal return-type (kernel-manager-function-return-type mgr name))
            (equal args (kernel-manager-function-arguments mgr name))
            (equal body (kernel-manager-function-body mgr name)))))

(defun kernel-manager-load-function (mgr name)
  (unless (kernel-manager-module-handle mgr)
    (error "kernel module is not loaded yet."))
  (when (kernel-manager-function-handle mgr name)
    (error "kernel function \"~A\" is already loaded." name))
  (let ((hmodule (kernel-manager-module-handle mgr))
        (hfunc (cffi:foreign-alloc 'cu-function))
        (fname (kernel-manager-function-c-name mgr name)))
      (cu-module-get-function hfunc (cffi:mem-ref hmodule 'cu-module) fname)
      (setf (kernel-manager-function-handle mgr name) hfunc)))

(defun kernel-manager-load-module (mgr)
  (when (kernel-manager-module-handle mgr)
    (error "kernel module is already loaded."))
  (unless (no-kernel-functions-loaded-p mgr)
    (error "some kernel functions are already loaded."))
  (let ((hmodule (cffi:foreign-alloc 'cu-module))
        (path (kernel-manager-module-path mgr)))
    (cu-module-load hmodule path)
    (setf (kernel-manager-module-handle mgr) hmodule)))

(defun no-kernel-functions-loaded-p (mgr)
  (notany #'(lambda (name)
              (kernel-manager-function-handle mgr name))
          (kernel-manager-function-names mgr)))

(defun kernel-manager-unload (mgr)
  (swhen (kernel-manager-module-handle mgr)
    (cu-module-unload (cffi:mem-ref it 'cu-module)))
  (free-function-handles mgr)
  (free-module-handle mgr))

(defun free-module-handle (mgr)
  (swhen (kernel-manager-module-handle mgr)
    (cffi:foreign-free it)
    (setf it nil)))

(defun free-function-handles (mgr)
  (mapcar #'(lambda (name)
              (swhen (kernel-manager-function-handle mgr name)
                (cffi:foreign-free it)
                (setf it nil)))
          (kernel-manager-function-names mgr)))

(defvar +temporary-path-template+ "/tmp/cl-cuda-")

(defvar +nvcc-path+ "/usr/local/cuda/bin/nvcc")

(defun kernel-manager-generate-and-compile (mgr)
  (when (kernel-manager-module-handle mgr)
    (error "kernel module is already loaded."))
  (unless (no-kernel-functions-loaded-p mgr)
    (error "some kernel functions are already loaded."))
  (let* ((temp-path (osicat-posix:mktemp +temporary-path-template+))
         (cu-path (concatenate 'string temp-path ".cu"))
         (ptx-path (concatenate 'string temp-path ".ptx")))
    (output-cu-code mgr cu-path)
    (compile-cu-code cu-path ptx-path)
    (setf (kernel-manager-module-path mgr) ptx-path)
    (setf (kernel-manager-module-compilation-needed mgr) nil)
    (values)))

(defun output-cu-code (mgr path)
  (with-open-file (out path :direction :output :if-exists :supersede)
    (princ (compile-kernel-definition (kernel-definition mgr)) out))
  (values))

(defun output-nvcc-command (cu-path ptx-path)
  (format t "nvcc -ptx -o ~A ~A~%" cu-path ptx-path))

(defun compile-cu-code (cu-path ptx-path)
  (output-nvcc-command cu-path ptx-path)
  (with-output-to-string (out)
    (let ((p (sb-ext:run-program +nvcc-path+ `("-ptx" "-o" ,ptx-path ,cu-path)
                                 :error out)))
      (unless (= 0 (sb-ext:process-exit-code p))
        (error (format nil "nvcc exits with code: ~A~%~A"
                       (sb-ext:process-exit-code p)
                       (get-output-stream-string out))))))
  (values))


;;;
;;; Definition of kernel manager's ensure- functions
;;;

(defun ensure-kernel-function-loaded (mgr name)
  (ensure-kernel-module-loaded mgr)
  (or (kernel-manager-function-handle mgr name)
      (kernel-manager-load-function mgr name)))

(defun ensure-kernel-module-loaded (mgr)
  (ensure-kernel-module-compiled mgr)
  (or (kernel-manager-module-handle mgr)
      (kernel-manager-load-module mgr)))

(defun ensure-kernel-module-compiled (mgr)
  (when (kernel-manager-module-compilation-needed mgr)
    (kernel-manager-generate-and-compile mgr))
  (values))


;;;
;;; Definition of default kernel manager
;;;

(defvar *kernel-manager*
  (make-kernel-manager))

(defun clear-kernel-manager ()
  (setf *kernel-manager* (make-kernel-manager))
  (values))

(defun print-kernel-manager ()
  (list (module-info *kernel-manager*)
        (hash-table-alist (function-handles *kernel-manager*))
        (kernel-definition *kernel-manager*)))


;;;
;;; Compiling
;;;

(defun compile-kernel-definition (def)
  (unlines `(,@(mapcar #'(lambda (name)
                           (compile-kernel-function-prototype name def))
                       (kernel-definition-function-names def))
             ""
             ,@(mapcar #'(lambda (name)
                           (compile-kernel-function name def))
                       (kernel-definition-function-names def)))))


;;; compile kernel function prototype

(defun compile-kernel-function-prototype (name def)
  (let ((name (kernel-definition-function-c-name name def))
        (return-type (kernel-definition-function-return-type name def))
        (arg-bindings (kernel-definition-function-arguments name def)))
    (format nil "extern \"C\" ~A;"
            (compile-function-declaration name return-type arg-bindings))))


;;; compile kernel function

(defun compile-kernel-function (name def)
  (let ((c-name (kernel-definition-function-c-name name def))
        (return-type (kernel-definition-function-return-type name def))
        (arg-bindings (kernel-definition-function-arguments name def))
        (stmts (kernel-definition-function-body name def)))
    (let ((type-env (make-type-environment-with-kernel-definition name def)))
      (unlines `(,(compile-function-declaration c-name return-type arg-bindings)
                 "{"
                 ,@(mapcar #'(lambda (stmt)
                              (indent 2 (compile-statement stmt type-env def)))
                           stmts)
                 "}"
                  "")))))

(defun make-type-environment-with-kernel-definition (name def)
  (let ((arg-bindings (kernel-definition-function-arguments name def)))
    (reduce #'(lambda (type-env arg-binding)
                (destructuring-bind (var type) arg-binding
                  (add-type-environment var type type-env)))
            arg-bindings
            :initial-value (empty-type-environment))))

(defun compile-function-declaration (name return-type arg-bindings)
  (format nil "~A ~A ~A (~A)" (compile-function-specifier return-type)
                              (compile-type return-type)
                              name
                              (compile-arg-bindings arg-bindings)))

(defun compile-function-specifier (return-type)
  (unless (valid-type-p return-type)
    (error (format nil "invalid return type: ~A" return-type)))
  (if (eq return-type 'void)
      "__global__"
      "__device__"))

(defun compile-type (type)
  (unless (valid-type-p type)
    (error (format nil "invalid type: ~A" type)))
  (compile-identifier (princ-to-string type)))

(defun compile-arg-bindings (arg-bindings)
  (join ", " (mapcar #'compile-arg-binding arg-bindings)))

(defun compile-arg-binding (arg-binding)
  (destructuring-bind (var type) arg-binding
    (format nil "~A ~A" (compile-type type) (compile-identifier var))))

(defun compile-identifier (idt)
  (substitute #\_ #\. (substitute #\_ #\- (string-downcase idt))))
  

;;; compile statement

(defun compile-statement (stmt type-env def)
  (cond
    ((if-p stmt) (compile-if stmt type-env def))
    ((let-p stmt) (compile-let stmt type-env def))
    ((do-p stmt) (compile-do stmt type-env def))
    ((with-shared-memory-p stmt) (compile-with-shared-memory stmt type-env def))
    ((set-p stmt) (compile-set stmt type-env def))
    ((progn-p stmt) (compile-progn stmt type-env def))
    ((return-p stmt) (compile-return stmt type-env def))
    ((syncthreads-p stmt) (compile-syncthreads stmt))
    ((function-p stmt) (compile-function stmt type-env def :statement-p t))
    (t (error "invalid statement: ~A" stmt))))

;;; if statement

(defun if-p (stmt)
  (match stmt
    (('if _ _) t)
    (('if _ _ _) t)
    (_ nil)))

(defun if-test-expression (stmt)
  (match stmt
    (('if test-exp _) test-exp)
    (('if test-exp _ _) test-exp)
    (_ (error (format nil "invalid statement: ~A" stmt)))))

(defun if-then-statement (stmt)
  (match stmt
    (('if _ then-stmt) then-stmt)
    (('if _ then-stmt _) then-stmt)
    (_ (error (format nil "invalid statement: ~A" stmt)))))

(defun if-else-statement (stmt)
  (match stmt
    (('if _ _) nil)
    (('if _ _ else-stmt) else-stmt)
    (_ (error (format nil "invalid statement: ~A" stmt)))))

(defun compile-if (stmt type-env def)
  (let ((test-exp (if-test-expression stmt))
        (then-stmt (if-then-statement stmt))
        (else-stmt (if-else-statement stmt)))
    (unlines (format nil "if ~A {"
                     (compile-expression test-exp type-env def))
             (indent 2 (compile-statement then-stmt type-env def))
             (and else-stmt "} else {")
             (and else-stmt
                  (indent 2 (compile-statement else-stmt type-env def)))
             "}")))


;;; let statement

(defun let-p (stmt)
  (match stmt
    (('let . _) t)
    (_ nil)))

(defun let-bindings (stmt)
  (match stmt
    (('let bindings . _) bindings)
    (_ (error (format nil "invalid statement: ~A" stmt)))))

(defun let-statements (stmt0)
  (match stmt0
    (('let _ . stmts) stmts)
    (_ (error (format nil "invalid statement: ~A" stmt0)))))

(defun compile-let (stmt0 type-env def)
  (let ((bindings (let-bindings stmt0))
        (stmts (let-statements stmt0)))
    (unlines "{"
             (indent 2 (%compile-let bindings stmts type-env def))
             "}")))

(defun %compile-let (bindings stmts type-env def)
  (if (null bindings)
      (compile-let-statements stmts type-env def)
      (compile-let-binding bindings stmts type-env def)))

(defun compile-let-binding (bindings stmts type-env def)
  (match bindings
    (((var exp) . rest)
     (let* ((type (type-of-expression exp type-env def))
            (type-env2 (add-type-environment var type type-env)))
       (unlines (format nil "~A ~A = ~A;"
                        (compile-type type)
                        (compile-identifier var)
                        (compile-expression exp type-env def))
                (%compile-let rest stmts type-env2 def))))
    (_ (error (format nil "invalid bindings: ~A" bindings)))))

(defun compile-let-statements (stmts type-env def)
  (compile-progn-statements stmts type-env def))


;;; set statement

(defun set-p (stmt)
  (match stmt
    (('set _ _) t)
    (_ nil)))

(defun set-place (stmt)
  (match stmt
    (('set place _) place)
    (_ (error (format nil "invalid statement: ~A" stmt)))))

(defun set-expression (stmt)
  (match stmt
    (('set _ exp) exp)
    (_ (error (format nil "invalid statement: ~A" stmt)))))

(defun compile-set (stmt type-env def)
  (let ((place (set-place stmt))
        (exp (set-expression stmt)))
    (format nil "~A = ~A;" (compile-place place type-env def)
                           (compile-expression exp type-env def))))

(defun compile-place (place type-env def)
  (cond ((scalar-place-p place) (compile-scalar-place place type-env))
        ((vector-place-p place) (compile-vector-place place type-env))
        ((array-place-p place) (compile-array-place place type-env def))
        (t (error (format nil "invalid place: ~A" place)))))

(defun scalar-place-p (place)
  (scalar-variable-reference-p place))

(defun vector-place-p (place)
  (vector-variable-reference-p place))

(defun array-place-p (place)
  (array-variable-reference-p place))

(defun compile-scalar-place (var type-env)
  (compile-scalar-variable-reference var type-env))

(defun compile-vector-place (place type-env)
  (compile-vector-variable-reference place type-env))

(defun compile-array-place (place type-env def)
  (compile-array-variable-reference place type-env def))


;;; progn statement

(defun progn-p (stmt)
  (match stmt
    (('progn . _) t)
    (_ nil)))

(defun progn-statements (stmt)
  (match stmt
    (('progn . stmts) stmts)
    (_ (error (format nil "invalid statement: ~A" stmt)))))

(defun compile-progn (stmt type-env def)
  (compile-progn-statements (progn-statements stmt) type-env def))

(defun compile-progn-statements (stmts type-env def)
  (unlines (mapcar #'(lambda (stmt2)
                       (compile-statement stmt2 type-env def))
                   stmts)))


;;; return statement

(defun return-p (stmt)
  (match stmt
    (('return) t)
    (('return _) t)
    (_ nil)))

(defun compile-return (stmt type-env def)
  (match stmt
    (('return) "return;")
    (('return exp) (format nil "return ~A;"
                               (compile-expression exp type-env def)))
    (_ (error (format nil "invalid statement: ~A" stmt)))))


;;; do statement

(defun do-p (stmt)
  (match stmt
    (('do . _) t)
    (_ nil)))

(defun do-bindings (stmt)
  (match stmt
    (('do bindings . _) bindings)
    (_ (error "invalid statement: ~A" stmt))))

(defun do-var-types (stmt type-def def)
  (labels ((do-var-type (binding)
             (list (do-binding-var binding)
                   (do-binding-type binding type-def def))))
    (mapcar #'do-var-type (do-bindings stmt))))

(defun do-binding-var (binding)
  (match binding
    ((var _)   var)
    ((var _ _) var)
    (_ (error "invalid binding: ~A" binding))))

(defun do-binding-type (binding type-env def)
  (type-of-expression (do-binding-init-form binding) type-env def))

(defun do-binding-init-form (binding)
  (match binding
    ((_ init-form)   init-form)
    ((_ init-form _) init-form)
    (_ (error "invalid binding: ~A" binding))))

(defun do-binding-step-form (binding)
  (match binding
    ((_ _)           nil)
    ((_ _ step-form) step-form)
    (_ (error "invalid binding: ~A" binding))))

(defun do-test-form (stmt)
  (match stmt
    (('do _ (test-form) . _) test-form)
    (_ (error "invalid statement: ~A" stmt))))

(defun do-statements (stmt)
  (match stmt
    (('do _ _ . stmts) stmts)
    (_ (error "invalid statement: ~A" stmt))))

(defun compile-do (stmt type-env def)
  (let ((type-env2 (bulk-add-type-environment (do-var-types stmt type-env def) type-env)))
    (let ((init-part (compile-do-init-part stmt type-env def))
          (test-part (compile-do-test-part stmt type-env2 def))
          (step-part (compile-do-step-part stmt type-env2 def)))
      (unlines (format nil "for ( ~A; ~A; ~A )" init-part test-part step-part)
               "{"
               (indent 2 (compile-do-statements stmt type-env2 def))
               "}"))))

(defun compile-do-init-part (stmt type-env def)
  (labels ((aux (binding)
             (let ((var (do-binding-var binding))
                   (type (do-binding-type binding type-env def))
                   (init-form (do-binding-init-form binding)))
               (format nil "~A ~A = ~A" (compile-type type)
                                        (compile-identifier var)
                                        (compile-expression init-form type-env def)))))
    (join ", " (mapcar #'aux (do-bindings stmt)))))

(defun compile-do-test-part (stmt type-env def)
  (let ((test-form (do-test-form stmt)))
    (format nil "! ~A" (compile-expression test-form type-env def))))

(defun compile-do-step-part (stmt type-env def)
  (labels ((aux (binding)
             (let ((var (do-binding-var binding))
                   (step-form (do-binding-step-form binding)))
               (format nil "~A = ~A" (compile-identifier var)
                                     (compile-expression step-form type-env def)))))
    (join ", " (mapcar #'aux (remove-if-not #'do-binding-step-form (do-bindings stmt))))))

(defun compile-do-statements (stmt type-env def)
  (compile-progn-statements (do-statements stmt) type-env def))


;;; with-shared-memory statement

(defun with-shared-memory-p (stmt)
  (match stmt
    (('with-shared-memory . _) t)
    (_ nil)))

(defun with-shared-memory-specs (stmt)
  (match stmt
    (('with-shared-memory specs . _) specs)
    (_ (error (format nil "invalid statement: ~A" stmt)))))

(defun with-shared-memory-statements (stmt)
  (match stmt
    (('with-shared-memory _ . stmts) stmts)
    (_ (error (format nil "invalid statement: ~A" stmt)))))

(defun compile-with-shared-memory (stmt type-env def)
  (let ((specs (with-shared-memory-specs stmt))
        (stmts (with-shared-memory-statements stmt)))
    (unlines "{"
             (indent 2 (%compile-with-shared-memory specs stmts type-env def))
             "}")))

(defun %compile-with-shared-memory (specs stmts type-env def)
  (if (null specs)
      (compile-with-shared-memory-statements stmts type-env def)
      (compile-with-shared-memory-spec specs stmts type-env def)))

(defun compile-with-shared-memory-spec (specs stmts type-env def)
  (match specs
    (((var type . sizes) . rest)
     (let* ((type-env2 (add-type-environment var (add-star type (length sizes))
                                             type-env)))
       (unlines (format nil "__shared__ ~A ~A~{[~A]~};"
                            (compile-type type)
                            (compile-identifier var)
                            (mapcar #'(lambda (exp)
                                        (compile-expression exp type-env def))
                                    sizes))
                (%compile-with-shared-memory rest stmts type-env2 def))))
    (_ (error (format nil "invalid shared memory specs: ~A" specs)))))

(defun compile-with-shared-memory-statements (stmts type-env def)
  (compile-let-statements stmts type-env def))


;;; compile syncthreads

(defun syncthreads-p (stmt)
  (match stmt
    (('syncthreads) t)
    (_ nil)))

(defun compile-syncthreads (stmt)
  (declare (ignorable stmt))
  "__syncthreads();")


;;; compile function

(defun function-p (form)
  (and (listp form)
       (car form)
       (symbolp (car form))))

(defun defined-function-p (form def)
  (or (built-in-function-p form)
      (user-function-p form def)))

(defun built-in-function-p (form)
  (match form
    ((op . _) (and (getf +built-in-functions+ op) t))
    (_ nil)))

(defun user-function-p (form def)
  (match form
    ((op . _) (kernel-definition-function-exists-p op def))
    (_ nil)))

(defun function-operator (form)
  (if (function-p form)
      (car form)
      (error (format nil "invalid statement or expression: ~A" form))))

(defun function-operands (form)
  (if (function-p form)
      (cdr form)
      (error (format nil "invalid statement or expression: ~A" form))))

(defun compile-function (form type-env def &key (statement-p nil))
  (unless (defined-function-p form def)
    (error (format nil "undefined function: ~A" form)))
  (let ((code (if (built-in-function-p form)
                  (compile-built-in-function form type-env def)
                  (compile-user-function form type-env def))))
    (if statement-p
        (format nil "~A;" code)
        code)))

(defun compile-built-in-function (form type-env def)
  (let ((op (function-operator form)))
    (cond
      ((built-in-function-arithmetic-p op)
       (compile-built-in-arithmetic-function form type-env def))
      ((built-in-function-infix-p op)
       (compile-built-in-infix-function form type-env def))
      ((built-in-function-prefix-p op)
       (compile-built-in-prefix-function form type-env def))
      (t (error (format nil "invalid built-in function: ~A" op))))))

(defun compile-built-in-arithmetic-function (form type-env def)
  (compile-built-in-infix-function (binarize-1 form) type-env def))

(defun binarize-1 (form)
  (if (atom form)
      form
      (if (and (nthcdr 3 form)
               (member (car form) +built-in-arithmetic-functions+))
          (destructuring-bind (op a1 a2 . rest) form
            (binarize-1 `(,op (,op ,a1 ,a2) ,@rest)))
          form)))

(defun compile-built-in-infix-function (form type-env def)
  (let ((operands (function-operands form)))
    (let ((op (built-in-function-inferred-operator form type-env def))
          (lhe (compile-expression (car operands) type-env def))
          (rhe (compile-expression (cadr operands) type-env def)))
      (format nil "(~A ~A ~A)" lhe op rhe))))

(defun compile-built-in-prefix-function (form type-env def)
  (let ((operands (function-operands form)))
    (format nil "~A (~A)"
            (built-in-function-inferred-operator form type-env def)
            (compile-operands operands type-env def))))

(defun compile-user-function (form type-env def)
  (let ((operator (function-operator form))
        (operands (function-operands form)))
    (let ((func (kernel-definition-function-c-name operator def)))
      (unless (equal (kernel-definition-function-argument-types operator def)
                     (type-of-operands operands type-env def))
        (error (format nil "invalid arguments: ~A" (cons operator operands))))
      (format nil "~A (~A)" func (compile-operands operands type-env def)))))

(defun type-of-operands (operands type-env def)
  (mapcar #'(lambda (exp)
              (type-of-expression exp type-env def))
          operands))

(defun compile-operands (operands type-env def)
  (join ", " (mapcar #'(lambda (exp)
                         (compile-expression exp type-env def))
                     operands)))


;;; built-in functions
;;;   <built-in-functions>  ::= plist { <function-name> => <function-info> }
;;;   <function-info>       ::= (<infix-p> <function-candidates>)
;;;   <function-candidates> ::= (<function-candidate>*)
;;;   <function-candidate>  ::= (<arg-types> <return-type> <function-c-name>)
;;;   <arg-types>           ::= (<arg-type>*)

(defvar +built-in-arithmetic-functions+
  '(+ - * /))

(defvar +built-in-functions+
  '(+ (t (((int int) int "+")
          ((float float) float "+")))
    - (t (((int int) int "-")
          ((float float) float "-")))
    * (t (((int int) int "*")
          ((float float) float "*")))
    / (t (((int int) int "/")
          ((float float) float "/")))
    = (t (((int int) bool "==")
          ((float float) bool "==")))
    < (t (((int int) bool "<")
          ((float float) bool "<")))
    > (t (((int int) bool ">")
          ((float float) bool ">")))
    <= (t (((int int) bool "<=")
           ((float float) bool "<=")))
    >= (t (((int int) bool ">=")
           ((float float) bool ">=")))
    expt (nil (((float float) float "powf")))
    rsqrtf (nil (((float) float "rsqrtf")))
    float3 (nil (((float float float) float3 "make_float3")))
    float4 (nil (((float float float float) float4 "make_float4")))
    ))

(defun built-in-function-arithmetic-p (op)
  (and (getf +built-in-arithmetic-functions+ op)
       t))

(defun built-in-function-infix-p (op)
  (aif (getf +built-in-functions+ op)
       (and (car it)
            t)
       (error (format nil "invalid built-in function: ~A" op))))

(defun built-in-function-prefix-p (op)
  (aif (getf +built-in-functions+ op)
       (not (car it))
       (error (format nil "invalid built-in function: ~A" op))))

(defun built-in-function-inferred-operator (form type-env def)
  (caddr (inferred-function form type-env def)))

(defun built-in-function-inferred-return-type (form type-env def)
  (cadr (inferred-function form type-env def)))

(defun inferred-function (form type-env def)
  (let ((operator (function-operator form))
        (operands (function-operands form)))
    (let ((candidates (function-candidates operator))
          (types (type-of-operands operands type-env def)))
      (or (find types candidates :key #'car :test #'equal)
          (error (format nil "invalid function application: ~A" form))))))

(defun function-candidates (op)
  (or (cadr (getf +built-in-functions+ op))
      (error (format nil "invalid operator: ~A" op))))

(defun built-in-arithmetic-function-return-type (form type-env def)
  (built-in-function-inferred-return-type (binarize-1 form) type-env def))


;;; compile expression

(defun compile-expression (exp type-env def)
  (cond ((literal-p exp) (compile-literal exp))
        ((cuda-dimension-p exp) (compile-cuda-dimension exp))
        ((variable-reference-p exp)
         (compile-variable-reference exp type-env def))
        ((function-p exp) (compile-function exp type-env def))
        (t (error (format nil "invalid expression: ~A" exp)))))

(defun literal-p (exp)
  (or (int-literal-p exp)
      (float-literal-p exp)))

(defun int-literal-p (exp)
  (typep exp 'fixnum))

(defun float-literal-p (exp)
  (typep exp 'single-float))

(defun compile-literal (exp)
  (princ-to-string exp))

(defun cuda-dimension-p (exp)
  (or (grid-dim-p exp) (block-dim-p exp) (block-idx-p exp) (thread-idx-p exp)))

(defun grid-dim-p (exp)
  (find exp '(grid-dim-x grid-dim-y grid-dim-z)))

(defun block-dim-p (exp)
  (find exp '(block-dim-x block-dim-y block-dim-z)))

(defun block-idx-p (exp)
  (find exp '(block-idx-x block-idx-y block-idx-z)))

(defun thread-idx-p (exp)
  (find exp '(thread-idx-x thread-idx-y thread-idx-z)))

(defun compile-cuda-dimension (exp)
  (case exp
    (grid-dim-x "gridDim.x")
    (grid-dim-y "gridDim.y")
    (grid-dim-z "gridDim.z")
    (block-dim-x "blockDim.x")
    (block-dim-y "blockDim.y")
    (block-dim-z "blockDim.z")
    (block-idx-x "blockIdx.x")
    (block-idx-y "blockIdx.y")
    (block-idx-z "blockIdx.z")
    (thread-idx-x "threadIdx.x")
    (thread-idx-y "threadIdx.y")
    (thread-idx-z "threadIdx.z")
    (t (error (format nil "invalid expression: ~A" exp)))))

(defun variable-reference-p (exp)
  (or (scalar-variable-reference-p exp)
      (vector-variable-reference-p exp)
      (array-variable-reference-p exp)))

(defun scalar-variable-reference-p (exp)
  (symbolp exp))

(defun vector-variable-reference-p (exp)
  (match exp
    ((selector _) (and (find selector (vector-type-selector-symbols))
                       t))
    (_ nil)))

(defun array-variable-reference-p (exp) 
  (match exp
    (('aref . _) t)
    (_ nil)))

(defun compile-variable-reference (exp type-env def)
  (cond ((scalar-variable-reference-p exp)
         (compile-scalar-variable-reference exp type-env))
        ((vector-variable-reference-p exp)
         (compile-vector-variable-reference exp type-env))
        ((array-variable-reference-p exp)
         (compile-array-variable-reference exp type-env def))
        (t (error (format nil "invalid expression: ~A" exp)))))

(defun compile-scalar-variable-reference (var type-env)
  (let ((type (lookup-type-environment var type-env)))
    (unless type
      (error (format nil "unbound variable: ~A" var))))
  (compile-identifier var))

(defun compile-vector-selector (selector)
  (unless (find selector (vector-type-selector-symbols))
    (error (format nil "invalid selector: ~A" selector)))
  (string-downcase (subseq (reverse (princ-to-string selector)) 0 1)))

(defun compile-vector-variable-reference (form type-env)
  (match form
    ((selector var) (let ((type (lookup-type-environment var type-env)))
                      (unless type
                        (error (format nil "unbound variable: ~A" form)))
                      (format nil "~A.~A"
                              (compile-identifier var)
                              (compile-vector-selector selector))))
    (_ (error (format nil "invalid variable reference: ~A" form)))))

(defun compile-array-variable-reference (form type-env def)
  (match form
    (('aref _)
     (error (format nil "invalid variable reference: ~A" form)))
    (('aref var . idxs)
     (let ((type (lookup-type-environment var type-env)))
       (unless type
         (error (format nil "unbound variable: ~A" form)))
       (unless (= (type-dimension type) (length idxs))
         (error (format nil "invalid dimension: ~A" form)))
       (format nil "~A~{[~A]~}"
                   (compile-identifier var)
                   (mapcar #'(lambda (idx)
                               (compile-expression idx type-env def)) idxs))))
    (_ (error (format nil "invalid variable reference: ~A" form)))))


;;; type of expression

(defun type-of-expression (exp type-env def)
  (cond ((literal-p exp) (type-of-literal exp))
        ((cuda-dimension-p exp) 'int)
        ((variable-reference-p exp) (type-of-variable-reference exp type-env))
        ((function-p exp) (type-of-function exp type-env def))
        (t (error (format nil "invalid expression: ~A" exp)))))

(defun type-of-literal (exp)
  (cond ((int-literal-p exp) 'int)
        ((float-literal-p exp) 'float)
        (t (error (format nil "invalid expression: ~A" exp)))))

(defun type-of-variable-reference (exp type-env)
  (cond ((scalar-variable-reference-p exp)
         (type-of-scalar-variable-reference exp type-env))
        ((vector-variable-reference-p exp)
         (type-of-vector-variable-reference exp type-env))
        ((array-variable-reference-p exp)
         (type-of-array-variable-reference exp type-env))
        (t (error (format nil "invalid expression: ~A" exp)))))

(defun type-of-scalar-variable-reference (var type-env)
  (let ((type (lookup-type-environment var type-env)))
    (unless type
      (error (format nil "unbound variable: ~A" var)))
    type))

(defun type-of-vector-variable-reference (exp type-env)
  (match exp
    ((selector var) (let ((type (lookup-type-environment var type-env)))
                      (unless type
                        (error (format nil "unbound variable: ~A" exp)))
                      (vector-type-selector-return-type selector)))
    (_ (error (format nil "invalid variable reference: ~A" exp)))))

(defun type-of-array-variable-reference (exp type-env)
  (match exp
    (('aref _) (error (format nil "invalid variable reference: ~A" exp)))
    (('aref var . idxs)
     (let ((type (lookup-type-environment var type-env)))
       (unless type
         (error (format nil "unbound variable: ~A" exp)))
       (unless (= (type-dimension type) (length idxs))
         (error (format nil "invalid dimension: ~A" exp)))
       (remove-star type)))
    (_ (error (format nil "invalid variable reference: ~A" exp)))))


(defun type-of-function (exp type-env def)
  (cond ((built-in-function-p exp)
         (type-of-built-in-function exp type-env def))
        ((user-function-p exp def)
         (type-of-user-function exp def))
        (t (error (format nil "invalid expression: ~A" exp)))))

(defun type-of-built-in-function (exp type-env def)
  (let ((op (function-operator exp)))
    (if (built-in-function-arithmetic-p op)
        (built-in-arithmetic-function-return-type exp type-env def)
        (built-in-function-inferred-return-type exp type-env def))))

(defun type-of-user-function (exp def)
  (let ((operator (function-operator exp)))
    (kernel-definition-function-return-type operator def)))


;;; type environment
;;; type-environment ::= (<type-pair>*)
;;; type-pair        ::= (<variable> . <type>)

(defun empty-type-environment ()
  '())

(defun add-type-environment (var type type-env)
  (assert (valid-type-p type))
  (cons (cons var type) type-env))

(defun bulk-add-type-environment (bindings type-env)
  (reduce #'(lambda (type-env2 binding)
              (destructuring-bind (var type) binding
                (add-type-environment var type type-env2)))
          bindings
          :initial-value type-env))

(defun lookup-type-environment (var type-env)
  (match (assoc var type-env)
    ((_ . type) type)
    (_ (error (format nil "unbound variable: ~A" var)))))

(defmacro with-type-environment ((var bindings) &body body)
  `(let ((,var (bulk-add-type-environment ',bindings (empty-type-environment))))
     ,@body))


;;; Timer

(defun make-timer-object (start-event stop-event)
  (cons start-event stop-event))

(defun timer-object-start-event (timer-object)
  (car timer-object))

(defun (setf timer-object-start-event) (val timer-object)
  (setf (car timer-object) val))

(defun timer-object-stop-event (timer-object)
  (cdr timer-object))

(defun (setf timer-object-stop-event) (val timer-object)
  (setf (cdr timer-object) val))

(defun create-timer ()
  (let ((start-event (cffi:foreign-alloc 'cu-event))
        (stop-event  (cffi:foreign-alloc 'cu-event)))
    (cu-event-create start-event cu-event-default)
    (cu-event-create stop-event  cu-event-default)
    (make-timer-object start-event stop-event)))

(defun destroy-timer (timer-object)
  (let ((start-event (timer-object-start-event timer-object))
        (stop-event  (timer-object-stop-event  timer-object)))
    (cu-event-destroy (cffi:mem-ref start-event 'cu-event))
    (cu-event-destroy (cffi:mem-ref stop-event 'cu-event))
    (cffi:foreign-free start-event)
    (cffi:foreign-free stop-event)
    (setf (timer-object-start-event timer-object) (cffi:null-pointer)
          (timer-object-stop-event  timer-object) (cffi:null-pointer))))

(defmacro with-timer ((timer)&body body)
  `(progn
     (let (,timer)
       (setf ,timer (create-timer))
       (unwind-protect (progn ,@body)
         (destroy-timer ,timer)))))

(defun start-timer (timer-object)
  (let ((start-event (timer-object-start-event timer-object)))
    (cu-event-record (cffi:mem-ref start-event 'cu-event) (cffi:null-pointer))))

(defun stop-and-synchronize-timer (timer-object)
  (let ((stop-event (timer-object-stop-event timer-object)))
    (cu-event-record (cffi:mem-ref stop-event 'cu-event) (cffi:null-pointer))
    (cu-event-synchronize (cffi:mem-ref stop-event 'cu-event))))

(defun get-elapsed-time (timer-object)
  (let (milliseconds
        (start-event (timer-object-start-event timer-object))
        (stop-event  (timer-object-stop-event  timer-object)))
    (stop-and-synchronize-timer timer-object)
    (cffi:with-foreign-object (pmilliseconds :float)
      (cu-event-elapsed-time pmilliseconds
                             (cffi:mem-ref start-event 'cu-event)
                             (cffi:mem-ref stop-event  'cu-event))
      (setf milliseconds (cffi:mem-ref pmilliseconds :float)))
    (start-timer timer-object)
    milliseconds))


;;; utilities

(defun join (str xs &key (remove-nil nil))
  (let ((xs2 (if remove-nil (remove-if #'null xs) xs)))
    (if (not (null xs2))
      (reduce #'(lambda (a b) (concatenate 'string a str b)) xs2)
      "")))

(defun indent (n str)
  (labels ((aux (x)
             (concatenate 'string (spaces n) x)))
    (unlines (mapcar #'aux (lines str)))))

(defun lines (str)
  (split-sequence:split-sequence #\LineFeed str :remove-empty-subseqs t))

(defun unlines (&rest args)
  (cond ((null args) "")
        ((listp (car args)) (join (string #\LineFeed) (car args) :remove-nil t))
        (t (join (string #\LineFeed) args :remove-nil t))))

(defun spaces (n)
  (if (< 0 n)
      (concatenate 'string " " (spaces (1- n)))
      ""))

(defun undefined ()
  (error "undefined"))

(defun cl-cuda-symbolicate (&rest args)
  (intern (apply #'concatenate 'string (mapcar #'princ-to-string args))
          :cl-cuda))
