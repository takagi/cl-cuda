#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-cuda)

;;; defcufun

(defmacro defcufun (name-and-options return-type &body args)
  (let* ((name (car name-and-options))
         (name% (symbolicate name "%"))
         (name-and-options% (cons name% (cdr name-and-options)))
         (params (mapcar #'car args)))
      `(progn
         (defun ,name (,@params)
           (check-cuda-errors ',name (,name% ,@params)))
         (defcfun ,name-and-options% ,return-type ,@args))))


;;; load CUDA driver API

(define-foreign-library libcuda
  (t (:default "/usr/local/cuda/lib/libcuda")))
(use-foreign-library libcuda)


;;; Types

(defctype cu-result :unsigned-int)
(defctype cu-device :int)
(defctype cu-context :pointer)
(defctype cu-module :pointer)
(defctype cu-function :pointer)
(defctype cu-stream :pointer)
(defctype cu-device-ptr :unsigned-int)
(defctype size-t :unsigned-int)


;;; Functions

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


;;; Built-in Vector Types

(defstruct (float3 (:constructor make-float3 (x y z)))
  (x 0.0 :type single-float)
  (y 0.0 :type single-float)
  (z 0.0 :type single-float))

(defcstruct float3
  (x :float)
  (y :float)
  (z :float))


;;; Constants

(defvar +cuda-success+ 0)


;;; Helpers

(defvar *show-messages* t)

(defun check-cuda-errors (name return-code)
  (unless (= return-code +cuda-success+)
    (error (format nil "~A failed with driver API error No. ~A.~%"
                       name return-code)))
  (when *show-messages*
    (format t "~A succeeded.~%" name))
  (values))

(defmacro with-cuda-context (args &body body)
  (destructuring-bind (dev-id) args
    (let ((flags 0))
      (with-gensyms (device ctx)
        `(with-foreign-objects ((,device 'cu-device)
                                (,ctx 'cu-context))
           (cu-init 0)
           (cu-device-get ,device ,dev-id)
           (cu-ctx-create ,ctx ,flags (mem-ref ,device 'cu-device))
           (unwind-protect
             (progn
               (ensure-kernel-module-loaded *kernel-manager*)
               ,@body)
             (progn
               (kernel-manager-unload *kernel-manager*)
               (cu-ctx-destroy (mem-ref ,ctx 'cu-context)))))))))

(defmacro with-cuda-memory-block (args &body body)
  (destructuring-bind (dptr size) args
    `(with-foreign-object (,dptr 'cu-device-ptr)
       (cu-mem-alloc ,dptr ,size)
       (unwind-protect
            (progn ,@body)
         (cu-mem-free (mem-ref ,dptr 'cu-device-ptr))))))

(defmacro with-cuda-memory-blocks (bindings &body body)
  (if bindings
      `(with-cuda-memory-block ,(car bindings)
         (with-cuda-memory-blocks ,(cdr bindings)
           ,@body))
      `(progn ,@body)))


;;; defkernel

(defmacro with-module-and-function (args &body body)
  (destructuring-bind (hfunc module function) args
    (with-gensyms (module-name func-name hmodule)
      `(with-foreign-string (,module-name ,module)
         (with-foreign-string (,func-name ,function)
           (with-foreign-objects ((,hmodule 'cu-module)
                                  (,hfunc 'cu-function))
             (cu-module-load ,hmodule ,module-name)
             (cu-module-get-function ,hfunc (mem-ref ,hmodule 'cu-module)
                                     ,func-name)
             ,@body))))))

(defun foreign-pointer-setf-vector-type (var var-ptr type)
  (let ((n (vector-type-length type)))
    `(progn
       ,@(loop repeat n
               for elm in +vector-type-elements+
            collect `(setf (cffi:foreign-slot-value ,var-ptr ',type ',elm)
                           (,(vector-type-selector type elm) ,var))))))

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
        `(with-foreign-objects (,@(mapcar #'ptr-type-pair bindings))
           ,@(mapcar #'foreign-pointer-setf bindings)
           ,@body))
      `(progn ,@body)))

(defmacro with-kernel-arguments (args &body body)
  (let ((var (car args))
        (ptrs (cdr args)))
    `(with-foreign-object (,var :pointer ,(length ptrs))
       ,@(loop for ptr in ptrs
            for i from 0
            collect `(setf (mem-aref ,var :pointer ,i) ,ptr))
       ,@body)))

(defun kernel-defun (mgr mgr-symbol name)
  (let ((kargs (kernel-manager-function-arg-bindings mgr name)))
    (with-gensyms (hfunc args)
      `(defun ,name (,@(kernel-arg-names kargs) &key grid-dim block-dim)
         (let ((,hfunc (ensure-kernel-function-loaded ,mgr-symbol ',name)))
           (with-non-pointer-arguments
               ,(kernel-arg-foreign-pointer-bindings kargs)
             (with-kernel-arguments
                 (,args ,@(kernel-arg-names-as-pointer kargs))
               (destructuring-bind
                     (grid-dim-x grid-dim-y grid-dim-z) grid-dim
               (destructuring-bind
                     (block-dim-x block-dim-y block-dim-z) block-dim
                 (cu-launch-kernel (mem-ref ,hfunc 'cu-function)
                                   grid-dim-x grid-dim-y grid-dim-z
                                   block-dim-x block-dim-y block-dim-z
                                   0 (null-pointer)
                                   ,args (null-pointer)))))))))))

(defmacro defkernel (name arg-bindings &rest body)
  (kernel-manager-define-function *kernel-manager* name arg-bindings body)
  (kernel-defun *kernel-manager* '*kernel-manager* name))


;;; kernel-arg

(defvar +basic-types+ '(void int float))

(defvar +vector-type-table+ '(float3 (float 3)))

(defvar +vector-type-elements+ '(x y z w))

(defun basic-type-p (type)
  (and (find type +basic-types+)
       t))

(defun vector-type-p (type)
  (and (getf +vector-type-table+ type)
       t))

(defun vector-type-length (type)
  ;; e.g. float3 => 3
  (cadr (getf +vector-type-table+ type)))

(defun vector-type-selector (type elm)
  ;; e.g. float3 x => float3-x
  (symbolicate (princ-to-string type) "-" (princ-to-string elm)))

(defun vector-type-selector-return-type (selector)
  ;; e.g. float3-x => float
  (unless (find selector (vector-type-selectors))
    (error (format nil "invalid selector: ~A" selector)))
  (symbolicate (reverse (subseq (reverse (princ-to-string selector)) 3))))

(defun vector-type-selectors ()
  (labels ((aux (type)
             (loop repeat (vector-type-length type)
                   for elm in +vector-type-elements+
                collect (vector-type-selector type elm))))
    (flatten
      (mapcar #'aux
        (mapcar #'car (group +vector-type-table+ 2))))))

(defun non-pointer-type-p (type)
  (or (basic-type-p type)
      (vector-type-p type)))

(defun pointer-type-p (type)
  (let ((last (subseq (reverse (princ-to-string type)) 0 1)))
    (and (string= last "*")
         (non-pointer-type-p (remove-star type))
         t)))

(defun valid-type-p (type)
  (or (pointer-type-p type)
      (non-pointer-type-p type)))

(defun add-star (type n)
  (labels ((aux (str n2)
             (if (< n2 1)
                 str
                 (aux (concatenate 'string str "*") (1- n2)))))
;    (intern (aux (princ-to-string type) n) :cl-cuda))
    (symbolicate (aux (princ-to-string type) n))))

(defun remove-star (type)
  (let ((rev (reverse (princ-to-string type))))
    (if (string= (subseq rev 0 1) "*")
        (remove-star (symbolicate (reverse (subseq rev 1))))
        type)))

(defun type-dimension (type)
  (unless (valid-type-p type)
    (error (format nil "invalid type: ~A" type)))
  (count #\* (princ-to-string type)))

(defvar +cffi-type-table+ '(int :int
                            float :float))

(defun cffi-type (type)
  (unless (valid-type-p type)
    (error (format nil "invalid type: ~A" type)))
  (acond
    ((pointer-type-p type) 'cu-device-ptr)
    ((getf +cffi-type-table+ type) it)
    ((vector-type-p type) type)
    (t (error (format nil "invalid type: ~A" type)))))

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
        var)))

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


;;; module-info

(defun make-module-info ()
  (list nil nil t))

(defmacro module-handle (info)
  `(car ,info))

(defmacro module-path (info)
  `(cadr ,info))

(defmacro module-compilation-needed (info)
  `(caddr ,info))


;;; kernel definition
;;; <kernel-definition> ::= (<kernel-function-table> <kernel-constant-table>)
;;; <kernel-function-table> ::= alist { <function-name> => <function-info> }
;;; <kernel-constant-table> ::= alist { <constant-name> => <constant-info> }

(defun empty-kernel-definition ()
  '(nil nil))

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

(defun define-kernel-function (name return-type arg-bindings body def)
  (let ((func-table (function-table def))
        (const-table (constant-table def)))
    (let ((func (make-function-info name return-type arg-bindings body)))
      (list (cons func (remove name func-table :key #'car))
            const-table))))

(defun define-kernel-constant (name value def)
  (declare (ignorable name value def))
  (undefined))

(defun undefine-kernel-function (name def)
  (unless (kernel-function-exists-p name def)
    (error (format nil "undefined kernel function: ~A" name)))
  (let ((func-table (function-table def))
        (const-table (constant-table def)))
    (list (remove name func-table :key #'car)
          const-table)))

(defun undefine-kernel-constant (name def)
  (declare (ignorable name def))
  (undefined))

(defun kernel-function-exists-p (name def)
  (and (assoc name (function-table def))
       t))

(defun kernel-function-names (def)
  (mapcar #'car (function-table def)))

(defun kernel-function-name (name def)
  (function-name (function-info name def)))

(defun kernel-function-c-name (name def)
  (compile-identifier (kernel-function-name name def)))

(defun kernel-function-return-type (name def)
  (function-return-type (function-info name def)))

(defun kernel-function-arg-bindings (name def)
  (function-arg-bindings (function-info name def)))

(defun kernel-function-arg-types (name def)
  (mapcar #'cadr (kernel-function-arg-bindings name def)))

(defun kernel-function-body (name def)
  (function-body (function-info name def)))


;;; <function-info> ::= (<name> <return-type> <arg-bindings> <body>)

(defun make-function-info (name return-type arg-bindings body)
  (list name return-type arg-bindings body))

(defun function-name (info)
  (car info))

(defun function-return-type (info)
  (cadr info))

(defun function-arg-bindings (info)
  (caddr info))

(defun function-body (info)
  (cadddr info))


;;; kernel-manager

(defun make-kernel-manager ()
  (list (make-module-info) (make-function-handles) (empty-kernel-definition)))

(defun make-function-handles ()
  (make-hash-table))

(defmacro module-info (mgr)
  `(car ,mgr))

(defmacro function-handles (mgr)
  `(cadr ,mgr))

(defmacro kernel-definition (mgr)
  `(caddr ,mgr))

(defmacro kernel-manager-module-handle (mgr)
  `(module-handle (module-info ,mgr)))

(defmacro kernel-manager-module-path (mgr)
  `(module-path (module-info ,mgr)))

(defmacro kernel-manager-module-compilation-needed (mgr)
  `(module-compilation-needed (module-info ,mgr)))

(defmacro kernel-manager-function-handle (mgr name)
  `(gethash ,name (function-handles ,mgr)))

(defun kernel-manager-function-exists-p (mgr name)
  (kernel-function-exists-p name (kernel-definition mgr)))

(defun kernel-manager-function-names (mgr)
  (kernel-function-names (kernel-definition mgr)))

(defun kernel-manager-function-c-name (mgr name)
  (kernel-function-c-name name (kernel-definition mgr)))

(defun kernel-manager-function-arg-bindings (mgr name)
  (kernel-function-arg-bindings name (kernel-definition mgr)))

(defun kernel-manager-kernel-code (mgr)
  (compile-kernel-definition (kernel-definition mgr)))

(defun kernel-manager-define-function (mgr name args body)
  (destructuring-bind (return-type arg-bindings) args
    (let ((def (kernel-definition mgr)))
      (when (or (not (kernel-manager-function-exists-p mgr name))
                (function-modified-p name def return-type arg-bindings body))
        (setf (kernel-definition mgr)
              (define-kernel-function name return-type arg-bindings body def))
        (setf (kernel-manager-module-compilation-needed mgr) t)))))

(defun function-modified-p (name def return-type arg-bindings body)
  (or (nequal return-type (kernel-function-return-type name def))
      (nequal arg-bindings (kernel-function-arg-bindings name def))
      (nequal body (kernel-function-body name def))))

(defun kernel-manager-load-function (mgr name)
  (unless (kernel-manager-module-handle mgr)
    (error "kernel module is not loaded yet."))
  (when (kernel-manager-function-handle mgr name)
    (error "kernel function \"~A\" is already loaded." name))
  (let ((hmodule (kernel-manager-module-handle mgr))
        (hfunc (foreign-alloc 'cu-function))
        (fname (kernel-manager-function-c-name mgr name)))
    (cu-module-get-function hfunc (mem-ref hmodule 'cu-module) fname)
    (setf (kernel-manager-function-handle mgr name) hfunc)))

(defun kernel-manager-load-module (mgr)
  (when (kernel-manager-module-handle mgr)
    (error "kernel module is already loaded."))
  (unless (no-kernel-functions-loaded-p mgr)
    (error "some kernel functions are already loaded."))
  (let ((hmodule (foreign-alloc 'cu-module))
        (path (kernel-manager-module-path mgr)))
    (cu-module-load hmodule path)
    (setf (kernel-manager-module-handle mgr) hmodule)))

(defun no-kernel-functions-loaded-p (mgr)
  "return t if no kernel functions are loaded."
  (notany #'(lambda (key)
              (kernel-manager-function-handle mgr key))
          (kernel-manager-function-names mgr)))

(defun kernel-manager-unload (mgr)
  (swhen (kernel-manager-module-handle mgr)
    (cu-module-unload (mem-ref it 'cu-module)))
  (free-function-handles mgr)
  (free-module-handle mgr))

(defun free-module-handle (mgr)
  (swhen (kernel-manager-module-handle mgr)
    (foreign-free it)
    (setf it nil)))

(defun free-function-handles (mgr)
  (let ((handles (function-handles mgr)))
    (maphash-keys #'(lambda (key)
                      (swhen (gethash key handles)
                        (foreign-free it)
                        (setf it nil))) handles)))

(defvar +temporary-path-template+ "/tmp/cl-cuda-")
(defvar +nvcc-path+ "/usr/local/cuda/bin/nvcc")

(defun kernel-manager-compile (mgr)
  (when (kernel-manager-module-handle mgr)
    (error "kernel module is already loaded."))
  (unless (no-kernel-functions-loaded-p mgr)
    (error "some kernel functions are already loaded."))
  (let* ((temp-path (osicat-posix:mktemp +temporary-path-template+))
         (cu-path (concatenate 'string temp-path ".cu"))
         (ptx-path (concatenate 'string temp-path ".ptx")))
    (kernel-manager-output-kernel-code mgr cu-path)
    (compile-kernel-module cu-path ptx-path)
    (setf (kernel-manager-module-path mgr) ptx-path)
    (setf (kernel-manager-module-compilation-needed mgr) nil)
    (values)))

(defun compile-kernel-module (cu-path ptx-path)
  (output-nvcc-command cu-path ptx-path)
  (with-output-to-string (out)
    (let ((p (sb-ext:run-program +nvcc-path+ `("-ptx" "-o" ,ptx-path ,cu-path)
                                 :error out)))
      (unless (= 0 (sb-ext:process-exit-code p))
        (error (format nil "nvcc exits with code: ~A~%~A"
                       (sb-ext:process-exit-code p)
                       (get-output-stream-string out))))))
  (values))

(defun output-nvcc-command (cu-path ptx-path)
  (format t "nvcc -ptx -o ~A ~A~%" cu-path ptx-path))

(defun kernel-manager-output-kernel-code (mgr path)
  (with-open-file (out path :direction :output :if-exists :supersede)
    (princ (kernel-manager-kernel-code mgr) out))
  (values))


;;; ensuring kernel manager

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
    (kernel-manager-compile mgr))
  (values))


;;; *kernel-manager*

(defvar *kernel-manager*
  (make-kernel-manager))

(defun clear-kernel-manager ()
  (setf *kernel-manager* (make-kernel-manager))
  (values))

(defun print-kernel-manager ()
  (list (module-info *kernel-manager*)
        (hash-table-alist (function-handles *kernel-manager*))
        (kernel-definition *kernel-manager*)))


;;; compile kernel definition

(defun compile-kernel-definition (def)
  (unlines `(,@(mapcar #'(lambda (name)
                           (compile-kernel-function-prototype name def))
                       (kernel-function-names def))
             ""
             ,@(mapcar #'(lambda (name)
                           (compile-kernel-function name def))
                       (kernel-function-names def)))))


;;; compile kernel function prototype

(defun compile-kernel-function-prototype (name def)
  (let ((name (kernel-function-c-name name def))
        (return-type (kernel-function-return-type name def))
        (arg-bindings (kernel-function-arg-bindings name def)))
    (format nil "extern \"C\" ~A;"
            (compile-function-declaration name return-type arg-bindings))))


;;; compile kernel function

(defun compile-kernel-function (name def)
  (let ((c-name (kernel-function-c-name name def))
        (return-type (kernel-function-return-type name def))
        (arg-bindings (kernel-function-arg-bindings name def))
        (stmts (kernel-function-body name def)))
    (let ((type-env (make-type-environment-with-kernel-definition name def)))
      (unlines `(,(compile-function-declaration c-name return-type arg-bindings)
                 "{"
                 ,@(mapcar #'(lambda (stmt)
                              (indent 2 (compile-statement stmt type-env def)))
                           stmts)
                 "}"
                  "")))))

(defun make-type-environment-with-kernel-definition (name def)
  (let ((arg-bindings (kernel-function-arg-bindings name def)))
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
  (substitute #\_ #\- (string-downcase idt)))
  

;;; compile statement

(defun compile-statement (stmt type-env def)
  (cond
    ((if-p stmt) (compile-if stmt type-env def))
    ((let-p stmt) (compile-let stmt type-env def))
    ((for-p stmt) (compile-for stmt type-env def))
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
    (unlines (format nil "if (~A) {"
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
  (unlines (mapcar #'(lambda (stmt)
                       (compile-statement stmt type-env def)) stmts)))


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
        ((array-place-p place) (compile-array-place place type-env def))
        (t (error (format nil "invalid place: ~A" place)))))

(defun scalar-place-p (place)
  (scalar-variable-reference-p place))

(defun array-place-p (place)
  (array-variable-reference-p place))

(defun compile-scalar-place (var type-env)
  (compile-scalar-variable-reference var type-env))

(defun compile-array-place (place type-env def)
  (compile-array-variable-reference place type-env def))


;;; progn statement

(defun progn-p (stmt)
  (match stmt
    (('progn . _) t)
    (_ nil)))

(defun progn-statements (stmt0)
  (match stmt0
    (('progn . stmts) stmts)
    (_ (error (format nil "invalid statement: ~A" stmt0)))))

(defun compile-progn (stmt0 type-env def)
  (unlines (mapcar #'(lambda (stmt)
                       (compile-statement stmt type-env def))
                   (progn-statements stmt0))))
    

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


;;; for statement

(defun for-p (stmt)
  (match stmt
    (('for . _) t)
    (_ nil)))

(defun for-bindings (stmt)
  (match stmt
    (('for bindings . _) bindings)
    (_ (error (format nil "invalid statement: ~A" stmt)))))

(defun for-types (stmt type-env def)
  (mapcar #'(lambda (begin)
              (type-of-expression begin type-env def))
          (for-begins stmt)))

(defun for-vars (stmt)
  (mapcar #'car (for-bindings stmt)))

(defun for-begins (stmt)
  (mapcar #'cadr (for-bindings stmt)))

(defun for-ends (stmt)
  (mapcar #'caddr (for-bindings stmt)))

(defun for-steps (stmt type-env def)
  (labels ((aux (type val)
             (or val
                 (case type
                   (int 1)
                   (float 1.0)
                   (t (error (format nil "invalid statement: ~A" stmt)))))))
    (let ((types (for-types stmt type-env def))
          (steps (mapcar #'cadddr (for-bindings stmt))))
      (mapcar #'aux types steps))))

(defun for-statements (stmt)
  (match stmt
    (('for _ . stmts) stmts)
    (_ (error (format nil "invalid statement: ~A" stmt)))))

(defun compile-for (stmt type-env def)
  (let ((begin-part (compile-for-begin-part stmt type-env def))
        (end-part (compile-for-end-part stmt type-env def))
        (step-part (compile-for-step-part stmt type-env def)))
    (unlines (format nil "for ( ~A; ~A; ~A )" begin-part end-part step-part)
             "{"
             (indent 2 (compile-for-statements stmt type-env def))
             "}")))

(defun compile-for-begin-part (stmt type-env def)
  (labels ((part (type var begin)
             (format nil "~A ~A = ~A" (compile-type type)
                                      (compile-identifier var)
                                      (compile-expression begin type-env def))))
    (join ", " (mapcar #'part (for-types stmt type-env def)
                              (for-vars stmt)
                              (for-begins stmt)))))

(defun compile-for-end-part (stmt type-env def)
  (labels ((part (var end)
             (format nil "~A <= ~A" (compile-identifier var)
                                    (compile-expression end type-env def))))
    (join ", " (mapcar #'part (for-vars stmt) (for-ends stmt)))))

(defun compile-for-step-part (stmt type-env def)
  (labels ((part (var step)
             (format nil "~A += ~A" (compile-identifier var)
                                    (compile-expression step type-env def))))
    (join ", " (loop for var in (for-vars stmt)
                     for step in (for-steps stmt type-env def)
                  when step
                  collect (part var step)))))

(defun compile-for-statements (stmt type-env def)
  (compile-let-statements (for-statements stmt) type-env def))


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
    (_ (error (format nil "invalid specs: ~A" specs)))))

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
    ((op . _) (kernel-function-exists-p op def))
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
  (let ((operator (function-operator form))
        (operands (function-operands form)))
    (unless (arithmetic-function-valid-type-p operator operands type-env def)
      (error (format nil "invalid arguments: ~A" (cons operator operands))))
    (compile-built-in-infix-function (binarize-1 form) type-env def)))

(defun binarize-1 (form)
  (if (atom form)
      form
      (if (and (nthcdr 3 form)
               (member (car form) '(+ - * /)))
          (destructuring-bind (op a1 a2 . rest) form
            (binarize-1 `(,op (,op ,a1 ,a2) ,@rest)))
          form)))

(defun compile-built-in-infix-function (form type-env def)
  (let ((operator (function-operator form))
        (operands (function-operands form)))
    (let ((lhe (car operands))
          (op (built-in-function-inferred-operator operator operands
                                                   type-env def))
          (rhe (cadr operands)))
      (format nil "(~A ~A ~A)" (compile-expression lhe type-env def)
                               op
                               (compile-expression rhe type-env def)))))

(defun compile-built-in-prefix-function (form type-env def)
  (let ((operator (function-operator form))
        (operands (function-operands form)))
    (format nil "~A (~A)" (built-in-function-inferred-operator operator operands
                                                               type-env def)
                          (compile-operands operands type-env def))))

(defun compile-user-function (form type-env def)
  (let ((operator (function-operator form))
        (operands (function-operands form)))
    (let ((func (kernel-function-c-name operator def)))
      (unless (equal (kernel-function-arg-types operator def)
                     (type-of-operands operands type-env def))
        (error (format nil "invalid arguments: ~A" (cons operands operands))))
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
    <= (t (((int int) bool "<=")
           ((float float) bool "<=")))
    expt (nil (((float float) float "powf")))
    ))

(defun built-in-function-infix-p (op)
  (aif (getf +built-in-functions+ op)
       (and (car it)
            t)
       (error (format nil "invalid built-in function: ~A" op))))

(defun built-in-function-prefix-p (op)
  (aif (getf +built-in-functions+ op)
       (not (car it))
       (error (format nil "invalid built-in function: ~A" op))))

(defun built-in-function-inferred-operator (operator operands type-env def)
  (caddr (inferred-function operator operands type-env def)))

(defun built-in-function-inferred-return-type (operator operands type-env def)
  (cadr (inferred-function operator operands type-env def)))

(defun inferred-function (operator operands type-env def)
  (let ((candidates (function-candidates operator))
        (types (type-of-operands operands type-env def)))
    (or (find types candidates :key #'car :test #'equal)
        (error (format nil "invalid function application: ~A"
                       (cons operator operands))))))

(defun function-candidates (op)
  (or (cadr (getf +built-in-functions+ op))
      (error (format nil "invalid operator: ~A" op))))

(defvar +built-in-arithmetic-functions+
  '(+ (int float)
    - (int float)
    * (int float)
    / (int float)))

(defun built-in-function-arithmetic-p (op)
  (and (getf +built-in-arithmetic-functions+ op)
       t))

(defun arithmetic-function-valid-type-p (operator operands type-env def)
  (and (%arithmetic-function-return-type operator operands type-env def)
       t))

(defun arithmetic-function-return-type (operator operands type-env def)
  (or (%arithmetic-function-return-type operator operands type-env def)
      (error (format nil "invalid arguments: ~A" (cons operator operands)))))

(defun %arithmetic-function-return-type (operator operands type-env def)
  (let ((candidates (arithmetic-function-type-candidates operator))
        (arg-type (remove-duplicates (type-of-operands operands type-env def))))
    (and (= (length arg-type) 1)
         (find (car arg-type) candidates))))

(defun arithmetic-function-type-candidates (op)
  (or (getf +built-in-arithmetic-functions+ op)
      (error (format nil "invalid operator: ~A" op))))


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
    ((selector _) (and (find selector (vector-type-selectors))
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
      (error (format nil "unbound variable: ~A" var)))
    (unless (non-pointer-type-p type)
      (error (format nil "invalid variable: ~A" var))))
  (compile-identifier var))

(defun compile-vector-selector (selector)
  (unless (find selector (vector-type-selectors))
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
    (unless (non-pointer-type-p type)
      (error (format nil "invalid variable: ~A" var)))
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
  (let ((operator (function-operator exp))
        (operands (function-operands exp)))
    (if (built-in-function-arithmetic-p operator)
        (type-of-built-in-arithmetic-function
          operator operands type-env def)
        (built-in-function-inferred-return-type
          operator operands type-env def))))

(defun type-of-built-in-arithmetic-function (operator operands type-env def)
  (arithmetic-function-return-type operator operands type-env def))

(defun type-of-user-function (exp def)
  (let ((operator (function-operator exp)))
    (kernel-function-return-type operator def)))


;;; type environment
;;; type-environment ::= (<type-pair>*)
;;; type-pair        ::= (<variable> . <type>)

(defun empty-type-environment ()
  '())

(defun add-type-environment (var type type-env)
  (assert (valid-type-p type))
  (cons (cons var type) type-env))

(defun lookup-type-environment (var type-env)
  (match (assoc var type-env)
    ((_ . type) type)
    (_ (error (format nil "unbound variable: ~A" var)))))


;;; utilities

(defun nequal (&rest args)
  (not (apply #'equal args)))

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

(defun group (source n)
  (if (zerop n) (error "zero length"))
  (labels ((rec (source acc)
             (let ((rest (nthcdr n source)))
               (if (consp rest)
                   (rec rest (cons (subseq source 0 n) acc))
                   (nreverse (cons source acc))))))
    (if source (rec source nil) nil)))

(defun undefined ()
  (error "undefined"))
