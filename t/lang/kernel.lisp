#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-test.lang.kernel
  (:use :cl :prove
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
;;; test KERNEL-GLOBAL-NAMES function
;;;

(diag "KERNEL-GLOBAL-NAMES")

(let ((kernel (make-kernel)))
  (kernel-define-global kernel 'x :device 42)
  (kernel-define-symbol-macro kernel 'y 42)
  (is (kernel-global-names kernel) '(x)
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
  (is-error (expand-macro-1 '(foo) kernel) error))


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
  (is-error (expand-macro '(foo) kernel) error))


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




;;
;; Global

(subtest "kernel-define-global"

  (let ((kernel (make-kernel)))
    (kernel-define-global kernel 'foo :device 42)
    (is (kernel-global-exists-p kernel 'foo)
        t)
    (is (kernel-global-name kernel 'foo)
        'foo)
    (is (kernel-global-c-name kernel 'foo)
        "cl_cuda_test_lang_kernel_foo")
    (is (kernel-global-qualifiers kernel 'foo)
        '(:device))
    (is (kernel-global-initializer kernel 'foo)
        42))

  (let ((kernel (make-kernel)))
    ;; Name in variable namespace should be overwrited.
    (kernel-define-symbol-macro kernel 'foo 42)
    (kernel-define-global kernel 'foo :device 42)
    (is (kernel-global-exists-p kernel 'foo)
        t)
    (is (kernel-symbol-macro-exists-p kernel 'foo)
        nil))

  (let ((kernel (make-kernel)))
    ;; Give multiple qualifiers.
    (kernel-define-global kernel 'foo '(:device :constant) 42)
    (is (kernel-global-qualifiers kernel 'foo)
        '(:device :constant)))

  (is-error (kernel-define-global :foo 'foo :device 42)
            type-error
            "Invalid kernel.")

  (let ((kernel (make-kernel)))
    (is-error (kernel-define-global kernel "foo" :device 42)
              type-error
              "Invalid name."))

  (let ((kernel (make-kernel)))
    (is-error (kernel-define-global kernel 'foo :foo 42)
              type-error
              "Invalid qualifier.")))

(subtest "kernel-global-exists-p"

  (let ((kernel (make-kernel)))
    (kernel-define-global kernel 'foo :device 42)
    (kernel-define-symbol-macro kernel 'bar 42)
    (is (kernel-global-exists-p kernel 'foo)
        t)
    (is (kernel-global-exists-p kernel 'bar)
        nil)
    (is (kernel-global-exists-p kernel 'baz)
        nil))

  (is-error (kernel-global-exists-p :foo 'foo)
            type-error
            "Invalid kernel.")

  (let ((kernel (make-kernel)))
    (is-error (kernel-global-exists-p kernel "foo")
              type-error
              "Invalid name.")))

(subtest "kernel-global-name"

  (let ((kernel (make-kernel)))
    (kernel-define-global kernel 'foo :device 42)
    (is (kernel-global-name kernel 'foo)
        'foo))

  (let ((kernel (make-kernel)))
    (is-error (kernel-global-name kernel 'foo)
              simple-error
              "Global not found."))

  (is-error (kernel-global-name :foo 'foo)
            type-error
            "Invalid kernel.")

  (let ((kernel (make-kernel)))
    (is-error (kernel-global-name kernel "foo")
              type-error
              "Invalid name.")))

(subtest "kernel-global-c-name"

  (let ((kernel (make-kernel)))
    (kernel-define-global kernel 'foo :device 42)
    (is (kernel-global-c-name kernel 'foo)
        "cl_cuda_test_lang_kernel_foo"))

  (let ((kernel (make-kernel)))
    (is-error (kernel-global-c-name kernel 'foo)
              simple-error
              "Global not found."))

  (is-error (kernel-global-c-name :foo 'foo)
            type-error
            "Invalid kernel.")

  (let ((kernel (make-kernel)))
    (is-error (kernel-global-c-name kernel "foo")
              type-error
              "Invalid name.")))

(subtest "kernel-global-qualifiers"

  (let ((kernel (make-kernel)))
    (kernel-define-global kernel 'foo :device 42)
    (is (kernel-global-qualifiers kernel 'foo)
        '(:device)))

  (let ((kernel (make-kernel)))
    (is-error (kernel-global-qualifiers kernel 'foo)
              simple-error
              "Global not found."))

  (is-error (kernel-global-qualifiers :foo 'foo)
            type-error
            "Invalid kernel.")

  (let ((kernel (make-kernel)))
    (is-error (kernel-global-qualifiers kernel "foo")
              type-error
              "Invalid name.")))

(subtest "kernel-global-initializer"

  (let ((kernel (make-kernel)))
    (kernel-define-global kernel 'foo :device 42)
    (is (kernel-global-initializer kernel 'foo)
        42))

  (let ((kernel (make-kernel)))
    (is-error (kernel-global-initializer kernel 'foo)
              simple-error
              "Global not found."))

  (is-error (kernel-global-initializer :foo 'foo)
            type-error
            "Invalid kernel.")

  (let ((kernel (make-kernel)))
    (is-error (kernel-global-initializer kernel "foo")
              type-error
              "Invalid name.")))


(finalize)
