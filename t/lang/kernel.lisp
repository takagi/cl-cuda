#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-test.lang.kernel
  (:use :cl :cl-test-more
        :cl-cuda.lang.kernel
        :cl-cuda.lang.type))
(in-package :cl-cuda-test.lang.kernel)

(plan nil)


;;;
;;; test MAKE-KERNEL function
;;;

(diag "MAKE-KERNEL")

(let ((kernel (make-kernel)))
  (is (kernel-function-names kernel) nil
      "basic case 1")
  (is (kernel-symbol-macro-names kernel) nil
      "basic case 2"))


;;;
;;; test KERNEL-FUNCTION-NAMES function
;;;

(diag "KERNEL-FUNCTION-NAMES")

(let ((kernel (make-kernel)))
  (kernel-define-function kernel 'foo 'int '((x int)) '((return x)))
  (kernel-define-macro kernel 'bar '(x) '(`(return ,x)))
  (is (kernel-function-names kernel) '(foo)
      "basic case 1"))


;;;
;;; test KERNEL-MACRO-NAMES function
;;;

(diag "KERNEL-MACRO-NAMES")

(let ((kernel (make-kernel)))
  (kernel-define-function kernel 'foo 'int '((x int)) '((return x)))
  (kernel-define-macro kernel 'bar '(x) '(`(return ,x)))
  (is (kernel-macro-names kernel) '(bar)
      "basic case 1"))


;;;
;;; test KERNEL-SYMBOL-MACRO-NAMES function
;;;

(diag "KERNEL-SYMBOL-MACRO-NAMES")

(let ((kernel (make-kernel)))
  (kernel-define-symbol-macro kernel 'x 1.0)
  (is (kernel-symbol-macro-names kernel) '(x)
      "kernel basic 1"))


;;;
;;; test KERNEL-DEFINE-FUNCTION function
;;;

(diag "KERNEL-DEFINE-FUNCTION")

(let ((kernel (make-kernel)))
  (is (kernel-define-function kernel 'foo 'int '((x int)) '((return x)))
      'foo "basic case 1"))

(let ((kernel (make-kernel)))
  (is-error (kernel-define-function kernel
                                    1 'int '((x int)) '((return x)))
            type-error
            "NAME which is not a cl-cuda symbol."))

(let ((kernel (make-kernel)))
  (is-error (kernel-define-function kernel 'foo 1 '((x int)) '((return x)))
            type-error
            "RETURN-TYPE which is not a cl-cuda type."))

(let ((kernel (make-kernel)))
  (is-error (kernel-define-function kernel 'foo 1 'bar '((return x)))
            type-error
            "ARGUMENTS which are invlalid arguments."))


;;;
;;; test KERNEL-FUNCTION-EXISTS-P function
;;;

(diag "KERNEL-FUNCTION-EXISTS-P")

(let ((kernel (make-kernel)))
  (kernel-define-function kernel 'foo 'int '((x int)) '((return x)))
  (kernel-define-macro kernel 'bar '(x) '(`(return ,x)))
  (is (kernel-function-exists-p kernel 'foo) t
      "basic case 1")
  (is (kernel-function-exists-p kernel 'bar) nil
      "basic case 2")
  (is (kernel-function-exists-p kernel 'baz) nil
      "basic case 3"))


;;;
;;; test KERNEL-FUNCTION-NAME function
;;;


;;;
;;; test KERNEL-FUNCTION-C-NAME function
;;;


;;;
;;; test KERNEL-FUNCTION-RETURN-TYPE function
;;;


;;;
;;; test KERNEL-FUNCTION-ARGUMENTS function
;;;




;;;
;;; test KERNEL-FUNCTION-ARGUMENT-VARS function
;;;




;;;
;;; test KERNEL-FUNCTION-ARGUMENT-TYPES function
;;;




;;;
;;; test KERNEL-FUNCTION-BODY function
;;;




;;;
;;; test KERNEL-DEFINE-MACRO function
;;;

(diag "KERNEL-DEFINE-MACRO")

(let ((kernel (make-kernel)))
  (is (kernel-define-macro kernel 'foo '(x) '(`(return ,x)))
      'foo "basic case 1"))

(let ((kernel (make-kernel)))
  (is-error (kernel-define-macro kernel 1 '(x) '(`(return ,x)))
            type-error
            "NAME which is not a cl-cuda symbol."))


;;;
;;; test KERNEL-MACRO-EXISTS-P function
;;;

(diag "KERNEL-MACRO-EXISTS-P")

(let ((kernel (make-kernel)))
  (kernel-define-function kernel 'foo 'int '((x int)) '((return x)))
  (kernel-define-macro kernel 'bar '(x) '(`(return ,x)))
  (is (kernel-macro-exists-p kernel 'foo) nil
      "basic case 1")
  (is (kernel-macro-exists-p kernel 'bar) t
      "basic case 2")
  (is (kernel-macro-exists-p kernel 'baz) nil
      "basic case 3"))


;;;
;;; test KERNEL-MACRO-NAME function
;;;




;;;
;;; test KERNEL-MACRO-ARGUMENTS function
;;;




;;;
;;; test KERNEL-MACRO-BODY function
;;;




;;;
;;; test KERNEL-MACRO-EXPANDER function
;;;




;;;
;;; test EXPAND-MACRO-1 function
;;;

(diag "EXPAND-MACRO-1")

(let ((kernel (make-kernel)))
  (kernel-define-macro kernel 'foo '(x) '(`(return ,x)))
  (kernel-define-macro kernel 'bar '(x) '(`(foo ,x)))
  (kernel-define-symbol-macro kernel 'a 1.0)
  (kernel-define-symbol-macro kernel 'b 'a)
  (is-values (expand-macro-1 '(foo 1) kernel) '((return 1) t))
  (is-values (expand-macro-1 '(bar 1) kernel) '((foo 1) t))
  (is-values (expand-macro-1 '(baz 1) kernel) '((baz 1) nil))
  (is-values (expand-macro-1 'a kernel) '(1.0 t))
  (is-values (expand-macro-1 'b kernel) '(a t))
  (is-values (expand-macro-1 'c kernel) '(c nil))
  (is-error (expand-macro-1 '(foo)) error))


;;;
;;; test EXPAND-MACRO function
;;;

(diag "EXPAND-MACRO")

(let ((kernel (make-kernel)))
  (kernel-define-macro kernel 'foo '(x) '(`(return ,x)))
  (kernel-define-macro kernel 'bar '(x) '(`(foo ,x)))
  (kernel-define-symbol-macro kernel 'a 1.0)
  (kernel-define-symbol-macro kernel 'b 'a)
  (is-values (expand-macro '(foo 1) kernel) '((return 1) t))
  (is-values (expand-macro '(bar 1) kernel) '((return 1) t))
  (is-values (expand-macro '(baz 1) kernel) '((baz 1) nil))
  (is-values (expand-macro 'a kernel) '(1.0 t))
  (is-values (expand-macro 'b kernel) '(1.0 t))
  (is-values (expand-macro 'c kernel) '(c nil))
  (is-error (expand-macro '(foo)) error))


;;;
;;; test KERNEL-DEFINE-SYMBOL-MACRO function
;;;

(diag "KERNEL-DEFINE-SYMBOL-MACRO")

(let ((kernel (make-kernel)))
  (is (kernel-define-symbol-macro kernel 'x 1.0)
      'x "basic case 1"))

(let ((kernel (make-kernel)))
  (is-error (kernel-define-symbol-macro kernel 1 1.0) type-error
            "NAME which is not a cl-cuda symbol."))


;;;
;;; test KERNEL-SYMBOL-MACRO-EXISTS-P function
;;;

(diag "KERNEL-SYMBOL-MACRO-EXISTS-P")

(let ((kernel (make-kernel)))
  (kernel-define-symbol-macro kernel 'x 1.0)
  (is (kernel-symbol-macro-exists-p kernel 'x) t
      "basic case 1")
  (is (kernel-symbol-macro-exists-p kernel 'y) nil
      "basic case 2"))


;;;
;;; test KERNEL-SYMBOL-MACRO-NAME function
;;;




;;;
;;; test KERNEL-SYMBOL-MACRO-EXPANSION function
;;;





(finalize)
