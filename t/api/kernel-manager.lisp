#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-test.api.kernel-manager
  (:use :cl :prove
        :cl-cuda.lang
        :cl-cuda.api.context
        :cl-cuda.api.kernel-manager))
(in-package :cl-cuda-test.api.kernel-manager)

(plan nil)


;;;
;;; test KERNEL-MANAGER's state transfer
;;;

(diag "KERNEL-MANAGER")

(let* ((mgr (make-kernel-manager))
       (*kernel-manager* mgr))
  (with-cuda (0)
    ;; I - initial state
    (kernel-manager-define-global mgr 'a :device 1)
    (kernel-manager-define-function mgr 'foo 'void '() '())
    (is (kernel-manager-compiled-p mgr) nil
        "basic case 1")
    (is (kernel-manager-module-handle mgr) nil
        "basic case 2")
    (is (kernel-manager-function-handles-empty-p mgr) t
        "basic case 3")
    (is (kernel-manager-global-device-ptrs-empty-p mgr) t
        "basic case 4")
    ;; II - compiled state
    (kernel-manager-compile-module mgr)
    (is (kernel-manager-compiled-p mgr) t
        "basic case 5")
    (is (kernel-manager-module-handle mgr) nil
        "basic case 6")
    (is (kernel-manager-function-handles-empty-p mgr) t
        "basic case 7")
    (is (kernel-manager-global-device-ptrs-empty-p mgr) t
        "basic case 8")
    ;; III - module-loaded state
    (kernel-manager-load-module mgr)
    (is (kernel-manager-compiled-p mgr) t
        "basic case 9")
    (is (not (null (kernel-manager-module-handle mgr))) t
        "basic case 10")
    (is (kernel-manager-function-handles-empty-p mgr) t
        "basic case 11")
    (is (kernel-manager-global-device-ptrs-empty-p mgr) t
        "basic case 12")
    ;; IV - funciton-loaded state
    (kernel-manager-load-function mgr 'foo)
    (is (kernel-manager-compiled-p mgr) t
        "basic case 11")
    (is (not (null (kernel-manager-module-handle mgr))) t
        "basic case 12")
    (is (kernel-manager-function-handles-empty-p mgr) nil
        "basic case 13")
    (is (kernel-manager-global-device-ptrs-empty-p mgr) t
        "basic case 14")
    (kernel-manager-load-global mgr 'a)
    (is (kernel-manager-global-device-ptrs-empty-p mgr) nil
        "basic case 15")
    ;; II - compiled state
    (kernel-manager-unload mgr)
    (is (kernel-manager-compiled-p mgr) t
        "basic case 16")
    (is (kernel-manager-module-handle mgr) nil
        "basic case 17")
    (is (kernel-manager-function-handles-empty-p mgr) t
        "basic case 18")
    (is (kernel-manager-global-device-ptrs-empty-p mgr) t
        "basic case 19")
    ;; I - initial state
    (kernel-manager-define-function mgr 'bar 'void '() '())
    (is (kernel-manager-compiled-p mgr) nil
        "basic case 20")
    (is (kernel-manager-module-handle mgr) nil
        "basic case 21")
    (is (kernel-manager-function-handles-empty-p mgr) t
        "basic case 22")
    (is (kernel-manager-global-device-ptrs-empty-p mgr) t
        "basic case 23")))


;;;
;;; test KERNEL-MANAGER-COMPILE-MODULE function
;;;

(diag "KERNEL-MANAGER-COMPILE-MODULE")

(let* ((mgr (make-kernel-manager))
       (*kernel-manager* mgr))
  (with-cuda (0)
    ;; I - initial state
    (kernel-manager-define-function mgr 'foo 'void '() '())
    ;; II - compiled state
    (kernel-manager-compile-module mgr)
    (is-error (kernel-manager-compile-module mgr) simple-error
              "KERNEL-MANAGER whose state is compiled state.")
    ;; III - module-loaded state
    (kernel-manager-load-module mgr)
    (is-error (kernel-manager-compile-module mgr) simple-error
              "KERNEL-MANAGER whose state is module-loaded state.")
    ;; IV - funciton-loaded state
    (kernel-manager-load-function mgr 'foo)
    (is-error (kernel-manager-compile-module mgr) simple-error
              "KERNEL-MANAGER whose state is function-loaded state.")))


;;;
;;; test KERNEL-MANAGER-LOAD-MODULE function
;;;

(diag "KERNEL-MANAGER-LOAD-MODULE")

(let* ((mgr (make-kernel-manager))
       (*kernel-manager* mgr))
  (with-cuda (0)
    ;; I - initial state
    (kernel-manager-define-function mgr 'foo 'void '() '())
    (is-error (kernel-manager-load-module mgr) simple-error
              "KERNEL-MANAGER whose state is initial state.")
    ;; II - compiled state
    (kernel-manager-compile-module mgr)
    nil
    ;; III - module-loaded state
    (kernel-manager-load-module mgr)
    (is-error (kernel-manager-load-module mgr) simple-error
              "KERNEL-MANAGER whose state is module-loaded state.")
    ;; IV - funciton-loaded state
    (kernel-manager-load-function mgr 'foo)
    (is-error (kernel-manager-load-module mgr) simple-error
              "KERNEL-MANAGER whose state is function-loaded state.")))


;;;
;;; test KERNEL-MANAGER-LOAD-FUNCTION function
;;;

(diag "KERNEL-MANAGER-LOAD-FUNCTION")

(let* ((mgr (make-kernel-manager))
       (*kernel-manager* mgr))
  (with-cuda (0)
    ;; I - initial state
    (kernel-manager-define-function mgr 'foo 'void '() '())
    (is-error (kernel-manager-load-function mgr 'foo) simple-error
              "KERNEL-MANAGER whose state is initial state.")
    ;; II - compiled state
    (kernel-manager-compile-module mgr)
    (is-error (kernel-manager-load-function mgr 'foo) simple-error
              "KERNEL-MANAGER whose state is compiled state.")
    ;; III - module-loaded state
    (kernel-manager-load-module mgr)
    nil
    ;; IV - function-loaded state
    (kernel-manager-load-function mgr 'foo)
    (is-error (kernel-manager-load-function mgr 'foo) simple-error
              "The kernel function FOO has been already loaded.")
    (is-error (kernel-manager-load-function mgr 'bar) simple-error
              "The kernel function BAR is not defined.")))

(let* ((mgr (make-kernel-manager))
       (*kernel-manager* mgr))
  (with-cuda (0)
    ;; transfer state from I to II
    (kernel-manager-compile-module mgr)
    ;; delete kernel module
    (let ((module-path
           (cl-cuda.api.kernel-manager::kernel-manager-module-path mgr)))
      (delete-file module-path))
    ;; try to load module which does not exist
    (is-error (kernel-manager-load-module mgr) simple-error
           "The kernel module which KERNEL-MANAGER specifies does not exist.")))


;;;
;;; test KERNEL-MANAGER-LOAD-GLOBAL function
;;;

(diag "KERNEL-MANAGER-LOAD-GLOBAL")

(let ((mgr (make-kernel-manager)))
  (with-cuda (0)
    ;; I - initial state
    (kernel-manager-define-global mgr 'a :device 42)
    (is-error (kernel-manager-load-global mgr 'a)
              simple-error
              "Invalid kernel manager state.")
    ;; II - compiled state
    (kernel-manager-compile-module mgr)
    (is-error (kernel-manager-load-global mgr 'a)
              simple-error
              "Invalid kernel manager state.")
    ;; III - module-loaded state
    (kernel-manager-load-module mgr)
    nil
    ;; IV - function-loaded state
    (kernel-manager-load-global mgr 'a)
    (is-error (kernel-manager-load-global mgr 'a)
              simple-error
              "The kernel global A has been already loaded.")
    (is-error (kernel-manager-load-global mgr 'b)
              simple-error
              "The kernel global B is not defined.")))


;;;
;;; test KERNEL-MANAGER-UNLOAD function
;;;

(diag "KERNEL-MANAGER-UNLOAD")

(let* ((mgr (make-kernel-manager))
       (*kernel-manager* mgr))
  (with-cuda (0)
    ;; I - initial state
    (ok (kernel-manager-unload mgr)
        "basic case 1")
    ;; II - compiled state
    (kernel-manager-compile-module mgr)
    (ok (kernel-manager-unload mgr)
        "basic case 2")
    ;; III - module-loaded state
    nil
    ;; IV - function-loaded state
    nil))


;;;
;;; test KERNEL-MANAGER-DEFINE-FUNCTION function
;;;

(diag "KERNEL-MANAGER-DEFINE-FUNCTION")

(let* ((mgr (make-kernel-manager))
       (*kernel-manager* mgr))
  (with-cuda (0)
    ;; transfer state from I to II
    (kernel-manager-define-function mgr 'foo 'void '() '())
    (kernel-manager-compile-module mgr)
    (is (kernel-manager-compiled-p mgr) t
        "basic case 1")
    ;; defining function without change makes no state transfer
    (kernel-manager-define-function mgr 'foo 'void '() '())
    (is (kernel-manager-compiled-p mgr) t
        "basic case 2")
    ;; defining function with change makes state transfer
    (kernel-manager-define-function mgr 'foo 'int '((i int)) '(return i))
    (is (kernel-manager-compiled-p mgr) nil
        "basic case 3")))

(let* ((mgr (make-kernel-manager))
       (*kernel-manager* mgr))
  (with-cuda (0)
    ;; I - initial state
    (kernel-manager-define-function mgr 'foo 'void '() '())
    nil
    ;; II - compiled state
    (kernel-manager-compile-module mgr)
    nil
    ;; III - module-loaded state
    (kernel-manager-load-module mgr)
    (is-error (kernel-manager-define-function mgr 'bar 'void '() '())
              simple-error
              "KERNEL-MANAGER whose state is module-loaded state.")
    ;; IV - function-loaded state
    (kernel-manager-load-function mgr 'foo)
    (is-error (kernel-manager-define-function mgr 'bar 'void '() '())
              simple-error
              "KERNEL-MANAGER whose state is function-loaded state.")))


;;;
;;; test KERNEL-MANAGER-DEFINE-MACRO function
;;;

(diag "KERNEL-MANAGER-DEFINE-MACRO")

(let* ((mgr (make-kernel-manager))
       (*kernel-manager* mgr))
  (with-cuda (0)
    ;; transfer state from I to II
    (kernel-manager-define-macro mgr 'foo '() '())
    (kernel-manager-compile-module mgr)
    (is (kernel-manager-compiled-p mgr) t
        "basic case 1")
    ;; defining macro without change makes no state transfer
    (kernel-manager-define-macro mgr 'foo '() '())
    (is (kernel-manager-compiled-p mgr) t
        "basic case 2")
    ;; defining macro with change makes state transfer
    (kernel-manager-define-macro mgr 'foo '(a) '(a))
    (is (kernel-manager-compiled-p mgr) nil
        "basic case 3")))

(let* ((mgr (make-kernel-manager))
       (*kernel-manager* mgr))
  (with-cuda (0)
    ;; I - initial state
    (kernel-manager-define-function mgr 'foo 'void '() '())
    nil
    ;; II - compiled state
    (kernel-manager-compile-module mgr)
    nil
    ;; III - module-loaded state
    (kernel-manager-load-module mgr)
    (is-error (kernel-manager-define-macro mgr 'bar '() '())
              simple-error
              "KERNEL-MANAGER whose state is module-loaded state.")
    ;; IV - function-loaded state
    (kernel-manager-load-function mgr 'foo)
    (is-error (kernel-manager-define-macro mgr 'bar '() '())
              simple-error
              "KERNEL-MANAGER whose state is function-loaded state.")))


;;;
;;; test KERNEL-MANAGER-DEFINE-SYMBOL-MACRO function
;;;

(diag "KERNEL-MANAGER-DEFINE-SYMBOL-MACRO")

(let* ((mgr (make-kernel-manager))
       (*kernel-manager* mgr))
  (with-cuda (0)
    ;; transfer state from I to II
    (kernel-manager-define-symbol-macro mgr 'foo 1)
    (kernel-manager-compile-module mgr)
    (is (kernel-manager-compiled-p mgr) t
        "basic case 1")
    ;; defining macro without change makes no state transfer
    (kernel-manager-define-symbol-macro mgr 'foo 1)
    (is (kernel-manager-compiled-p mgr) t
        "basic case 2")
    ;; defining macro with change makes state transfer
    (kernel-manager-define-symbol-macro mgr 'foo 2)
    (is (kernel-manager-compiled-p mgr) nil
        "basic case 3")))

(let* ((mgr (make-kernel-manager))
       (*kernel-manager* mgr))
  (with-cuda (0)
    ;; I - initial state
    (kernel-manager-define-function mgr 'foo 'void '() '())
    nil
    ;; II - compiled state
    (kernel-manager-compile-module mgr)
    nil
    ;; III - module-loaded state
    (kernel-manager-load-module mgr)
    (is-error (kernel-manager-define-symbol-macro mgr 'foo 1) simple-error
              "KERNEL-MANAGER whose state is module-loaded state.")
    ;; IV - function-loaded state
    (kernel-manager-load-function mgr 'foo)
    (is-error (kernel-manager-define-symbol-macro mgr 'foo 2) simple-error
              "KERNEL-MANAGER whose state is function-loaded state.")))


;;;
;;; test KERNEL-MANAGER-DEFINE-GLOBAL function
;;;

(diag "KERNEL-MANAGER-DEFINE-GLOBAL")

(let ((mgr (make-kernel-manager)))
  (with-cuda (0)
    ;; Transfer state from I to II.
    (kernel-manager-define-global mgr 'a :device 42)
    (kernel-manager-compile-module mgr)
    (is (kernel-manager-compiled-p mgr)
        t
        "basic case 1")
    ;; Defining global without change makes no state transfer.
    (kernel-manager-define-global mgr 'a :device 42)
    (is (kernel-manager-compiled-p mgr)
        t
        "basic case 2")
    ;; Defining global with change makes state transfer.
    (kernel-manager-define-global mgr 'a :device 43)
    (is (kernel-manager-compiled-p mgr)
        nil
        "basic case 3")))

(let ((mgr (make-kernel-manager)))
  (with-cuda (0)
    ;; I - initial state
    (kernel-manager-define-global mgr 'a :device 42)
    nil
    ;; II - compiled state
    (kernel-manager-compile-module mgr)
    nil
    ;; III - module-loaded state
    (kernel-manager-load-module mgr)
    (is-error (kernel-manager-define-global mgr 'a :device 42)
              simple-error
              "Invalid kernel manager state.")
    ;; IV - function-loaded state
    (kernel-manager-load-global mgr 'a)
    (is-error (kernel-manager-define-global mgr 'a :device 43)
              simple-error
              "Invalid kernel manager state.")))


;;;
;;; test EXPAND-MACRO-1 function
;;;

(diag "EXPAND-MACRO-1")

(let ((mgr (make-kernel-manager)))
  (kernel-manager-define-macro mgr 'foo '(x) '(`(return ,x)))
  (kernel-manager-define-macro mgr 'bar '(x) '(`(foo ,x)))
  (kernel-manager-define-symbol-macro mgr 'a 1.0)
  (kernel-manager-define-symbol-macro mgr 'b 'a)
  (is-values (expand-macro-1 '(foo 1) mgr) '((return 1) t))
  (is-values (expand-macro-1 '(bar 1) mgr) '((foo 1) t))
  (is-values (expand-macro-1 '(baz 1) mgr) '((baz 1) nil))
  (is-values (expand-macro-1 'a mgr) '(1.0 t))
  (is-values (expand-macro-1 'b mgr) '(a t))
  (is-values (expand-macro-1 'c mgr) '(c nil))
  (is-error (expand-macro-1 '(foo) mgr) error))


;;;
;;; test EXPAND-MACRO function
;;;

(diag "EXPAND-MACRO")

(let ((mgr (make-kernel-manager)))
  (kernel-manager-define-macro mgr 'foo '(x) '(`(return ,x)))
  (kernel-manager-define-macro mgr 'bar '(x) '(`(foo ,x)))
  (kernel-manager-define-symbol-macro mgr 'a 1.0)
  (kernel-manager-define-symbol-macro mgr 'b 'a)
  (is-values (expand-macro '(foo 1) mgr) '((return 1) t))
  (is-values (expand-macro '(bar 1) mgr) '((return 1) t))
  (is-values (expand-macro '(baz 1) mgr) '((baz 1) nil))
  (is-values (expand-macro 'a mgr) '(1.0 t))
  (is-values (expand-macro 'b mgr) '(1.0 t))
  (is-values (expand-macro 'c mgr) '(c nil))
  (is-error (expand-macro '(foo) mgr) error))


(finalize)
