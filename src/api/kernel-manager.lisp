#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda.api.kernel-manager
  (:use :cl
        :cl-cuda.driver-api
        :cl-cuda.lang.kernel
        :cl-cuda.lang.compiler.compile-kernel
        :cl-cuda.api.nvcc)
  (:export :kernel-manager
           :make-kernel-manager
           :kernel-manager-compiled-p
           :kernel-manager-module-handle
           :kernel-manager-function-handles-empty-p
           :kernel-manager-function-handle
           :kernel-manager-global-device-ptrs-empty-p
           :kernel-manager-global-device-ptr
           :kernel-manager-global-qualifiers
           :kernel-manager-define-function
           :kernel-manager-define-macro
           :kernel-manager-define-symbol-macro
           :kernel-manager-define-global
           :kernel-manager-compile-module
           :kernel-manager-load-module
           :kernel-manager-load-function
           :kernel-manager-load-global
           :kernel-manager-unload
           :ensure-kernel-module-compiled
           :ensure-kernel-module-loaded
           :ensure-kernel-function-loaded
           :ensure-kernel-global-loaded
           :expand-macro-1
           :expand-macro
           :*kernel-manager*)
  (:shadow :expand-macro-1
           :expand-macro)
  (:import-from :alexandria
                :ensure-list))
(in-package :cl-cuda.api.kernel-manager)


;;;
;;; Kernel manager
;;;

(defstruct (kernel-manager (:constructor %make-kernel-manager))
  module-path
  module-handle
  %function-handles
  %global-device-ptrs
  kernel)

(defun make-kernel-manager ()
  (%make-kernel-manager :module-path nil
                        :module-handle nil
                        :%function-handles (make-hash-table)
                        :%global-device-ptrs (make-hash-table)
                        :kernel (make-kernel)))

(defun kernel-manager-%function-handle (manager name)
  (let ((function-handles (kernel-manager-%function-handles manager)))
    (gethash name function-handles)))

(defun (setf kernel-manager-%function-handle) (value manager name)
  (let ((function-handles (kernel-manager-%function-handles manager)))
    (setf (gethash name function-handles) value)))

(defun kernel-manager-%global-device-ptr (manager name)
  (let ((global-device-ptrs (kernel-manager-%global-device-ptrs manager)))
    (gethash name global-device-ptrs)))

(defun (setf kernel-manager-%global-device-ptr) (value manager name)
  (let ((global-device-ptrs (kernel-manager-%global-device-ptrs manager)))
    (setf (gethash name global-device-ptrs) value)))

(defun kernel-manager-compiled-p (manager)
  (and (kernel-manager-module-path manager)
       t))

(defun kernel-manager-function-handles-empty-p (manager)
  (let ((function-handles (kernel-manager-%function-handles manager)))
    (zerop (hash-table-count function-handles))))

(defun kernel-manager-function-handle (manager name)
  (kernel-manager-%function-handle manager name))

(defun kernel-manager-global-device-ptrs-empty-p (manager)
  (let ((global-device-ptrs (kernel-manager-%global-device-ptrs manager)))
    (zerop (hash-table-count global-device-ptrs))))

(defun kernel-manager-global-device-ptr (manager name)
  (kernel-manager-%global-device-ptr manager name))

(defun kernel-manager-global-qualifiers (manager name)
  (let ((kernel (kernel-manager-kernel manager)))
    (kernel-global-qualifiers kernel name)))

(defun kernel-manager-define-function (manager name return-type arguments body)
  (unless (not (kernel-manager-module-handle manager))
    (error "The kernel manager has already loaded the kernel module."))
  (symbol-macrolet ((module-path (kernel-manager-module-path manager))
                    (kernel (kernel-manager-kernel manager)))
    (when (function-modified-p kernel name return-type arguments body)
      (kernel-define-function kernel name return-type arguments body)
      (setf module-path nil)))
  name)

(defun function-modified-p (kernel name return-type arguments body)
  (not (and (kernel-function-exists-p kernel name)
            (equal return-type (kernel-function-return-type kernel name))
            (equal arguments (kernel-function-arguments kernel name))
            (equal body (kernel-function-body kernel name)))))

(defun kernel-manager-define-macro (manager name arguments body)
  (unless (not (kernel-manager-module-handle manager))
    (error "The kernel manager has already loaded the kernel module."))
  (symbol-macrolet ((module-path (kernel-manager-module-path manager))
                    (kernel (kernel-manager-kernel manager)))
    (when (macro-modified-p kernel name arguments body)
      (kernel-define-macro kernel name arguments body)
      (setf module-path nil)))
  name)

(defun macro-modified-p (kernel name arguments body)
  (not (and (kernel-macro-exists-p kernel name)
            (equal arguments (kernel-macro-arguments kernel name))
            (equal body (kernel-macro-body kernel name)))))

(defun kernel-manager-define-symbol-macro (manager name expansion)
  (unless (not (kernel-manager-module-handle manager))
    (error "The kernel manager has already loaded the kernel module."))
  (symbol-macrolet ((module-path (kernel-manager-module-path manager))
                    (kernel (kernel-manager-kernel manager)))
    (when (symbol-macro-modified-p kernel name expansion)
      (kernel-define-symbol-macro kernel name expansion)
      (setf module-path nil)))
  name)

(defun symbol-macro-modified-p (kernel name expansion)
  (not (and (kernel-symbol-macro-exists-p kernel name)
            (equal expansion (kernel-symbol-macro-expansion kernel name)))))

(defun kernel-manager-define-global (manager name qualifiers
                                     &optional expression)
  (unless (not (kernel-manager-module-handle manager))
    (error "The kernel manager has already loaded the kernel module."))
  (symbol-macrolet ((module-path (kernel-manager-module-path manager))
                    (kernel (kernel-manager-kernel manager)))
    (when (global-modified-p kernel name qualifiers expression)
      (kernel-define-global kernel name qualifiers expression)
      (setf module-path nil)))
  name)

(defun global-modified-p (kernel name qualifiers expression)
  (not (and (kernel-global-exists-p kernel name)
            (equal (ensure-list qualifiers)
                   (kernel-global-qualifiers kernel name))
            (equal expression (kernel-global-initializer kernel name)))))

(defun kernel-manager-compile-module (manager)
  (unless (not (kernel-manager-compiled-p manager))
    (error "The kernel manager has already been compiled."))
  (let ((kernel (kernel-manager-kernel manager)))
    (let ((module-path (nvcc-compile (compile-kernel kernel))))
      (setf (kernel-manager-module-path manager) module-path)))
  t)

(defun kernel-manager-load-module (manager)
  (unless (kernel-manager-compiled-p manager)
    (error "The kernel manager has not been compiled yet."))
  (unless (not (kernel-manager-module-handle manager))
    (error "The kernel manager has already loaded the kernel module."))
  (cffi:with-foreign-object (hmodule 'cu-module)
    (let ((module-path (kernel-manager-module-path manager)))
      (cu-module-load hmodule module-path)
      (setf (kernel-manager-module-handle manager)
            (cffi:mem-ref hmodule 'cu-module)))))

(defun kernel-manager-load-function (manager name)
  (unless (kernel-manager-compiled-p manager)
    (error "The kernel manager has not been compiled yet."))
  (unless (kernel-manager-module-handle manager)
    (error "The kernel manager has not loaded the kernel module yet."))
  (unless (not (kernel-manager-function-handle manager name))
    (error "The kernel function ~S has already been loaded." name))
  (symbol-macrolet
      ((module-handle (kernel-manager-module-handle manager))
       (kernel (kernel-manager-kernel manager)))
    (let ((c-name (kernel-function-c-name kernel name)))
      (cffi:with-foreign-object (hfunc 'cu-function)
        (cu-module-get-function hfunc module-handle c-name)
        (setf (kernel-manager-%function-handle manager name)
              (cffi:mem-ref hfunc 'cu-function))))))

(defun kernel-manager-load-global (manager name)
  (unless (kernel-manager-compiled-p manager)
    (error "The kernel manager has not been compiled yet."))
  (unless (kernel-manager-module-handle manager)
    (error "The kernel manager has not loaded the kernel module yet."))
  (unless (not (kernel-manager-global-device-ptr manager name))
    (error "The kernel global ~S has already been loaded." name))
  (symbol-macrolet
      ((module-handle (kernel-manager-module-handle manager))
       (kernel (kernel-manager-kernel manager)))
    (let ((c-name (kernel-global-c-name kernel name)))
      (cffi:with-foreign-object (device-ptr 'cu-device-ptr)
        (cu-module-get-global device-ptr
                              (cffi:null-pointer)
                              module-handle
                              c-name)
        (setf (kernel-manager-%global-device-ptr manager name)
              (cffi:mem-ref device-ptr 'cu-device-ptr))))))

(defun kernel-manager-unload (manager)
  ;; Clear function handles.
  (let ((function-handles (kernel-manager-%function-handles manager)))
    (clrhash function-handles))
  ;; Clear global device pointers.
  (let ((global-device-ptrs (kernel-manager-%global-device-ptrs manager)))
    (clrhash global-device-ptrs))
  ;; Unload module.
  (symbol-macrolet ((module-handle (kernel-manager-module-handle manager)))
    (when module-handle
      (cu-module-unload module-handle)
      (setf module-handle nil)))
  t)

(defun ensure-kernel-module-compiled (manager)
  (or (kernel-manager-compiled-p manager)
      (kernel-manager-compile-module manager)))

(defun ensure-kernel-module-loaded (manager)
  (ensure-kernel-module-compiled manager)
  (or (kernel-manager-module-handle manager)
      (kernel-manager-load-module manager)))

(defun ensure-kernel-function-loaded (manager name)
  (ensure-kernel-module-loaded manager)
  (or (kernel-manager-function-handle manager name)
      (kernel-manager-load-function manager name)))

(defun ensure-kernel-global-loaded (manager name)
  (ensure-kernel-module-loaded manager)
  (or (kernel-manager-global-device-ptr manager name)
      (kernel-manager-load-global manager name)))

(defun expand-macro-1 (form manager)
  (let ((kernel (kernel-manager-kernel manager)))
    (cl-cuda.lang.kernel:expand-macro-1 form kernel)))

(defun expand-macro (form manager)
  (let ((kernel (kernel-manager-kernel manager)))
    (cl-cuda.lang.kernel:expand-macro form kernel)))

(defvar *kernel-manager* (make-kernel-manager))
