#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-test.lang.environment
  (:use :cl :prove
        :cl-cuda.lang.type
        :cl-cuda.lang.environment))
(in-package :cl-cuda-test.lang.environment)

(plan nil)


;;;
;;; test Variable environment - Variable
;;;

(diag "Variable environment - Variable")

(let ((var-env (variable-environment-add-symbol-macro 'y 1.0
                 (variable-environment-add-variable 'x 'int
                   (empty-variable-environment)))))
  (is (variable-environment-variable-exists-p var-env 'x) t
      "basic case 1")
  (is (variable-environment-variable-exists-p var-env 'y) nil
      "basic case 2")
  (is (variable-environment-variable-exists-p var-env 'z) nil
      "basic case 3")
  (is (variable-environment-variable-name var-env 'x) 'x
      "basic case 4")
  (is (variable-environment-variable-type var-env 'x) 'int
      "basic case 5"))

(let ((var-env (variable-environment-add-variable 'x 'float
                 (variable-environment-add-variable 'x 'int
                   (empty-variable-environment)))))
  (is (variable-environment-variable-type var-env 'x) 'float
      "basic case 6"))

(let ((var-env (variable-environment-add-symbol-macro 'x '1.0
                 (variable-environment-add-variable 'x 'int
                   (empty-variable-environment)))))
  (is (variable-environment-variable-exists-p var-env 'x) nil
      "basic case 7")
  (is (variable-environment-symbol-macro-exists-p var-env 'x) t
      "basic case 8"))


;;;
;;; test Variable environment - Symbol macro
;;;

(diag "Variable environment - Symbol macro")

(let ((var-env (variable-environment-add-symbol-macro 'y 1.0
                 (variable-environment-add-variable 'x 'int
                   (empty-variable-environment)))))
  (is (variable-environment-symbol-macro-exists-p var-env 'x) nil
      "basic case 1")
  (is (variable-environment-symbol-macro-exists-p var-env 'y) t
      "basic case 2")
  (is (variable-environment-symbol-macro-exists-p var-env 'z) nil
      "basic case 3")
  (is (variable-environment-symbol-macro-name var-env 'y) 'y
      "basic case 4")
  (is (variable-environment-symbol-macro-expansion var-env 'y) 1.0
      "basic case 5"))


;;;
;;; test Variable environment - Global
;;;

(diag "Variable environment - Global")

(let ((var-env (variable-environment-add-variable 'z 'int
                (variable-environment-add-global 'y 'int nil
                 (variable-environment-add-global 'x 'int 1
                  (empty-variable-environment))))))
  (is (variable-environment-global-exists-p var-env 'x) t
      "basic case 1")
  (is (variable-environment-global-exists-p var-env 'y) t
      "basic case 2")
  (is (variable-environment-global-exists-p var-env 'z) nil
      "basic case 3")
  (is (variable-environment-global-name var-env 'x) 'x
      "basic case 4")
  (is (variable-environment-global-c-name var-env 'x)
      "cl_cuda_test_lang_environment_x"
      "basic case 5")
  (is (variable-environment-global-type var-env 'x) 'int
      "basic case 6")
  (is (variable-environment-global-initializer var-env 'x) 1
      "basic case 7")
  (is (variable-environment-global-initializer var-env 'y) nil
      "basic case 8"))

(is-error (variable-environment-add-global 1 'int 1
           (empty-variable-environment))
          type-error
          "Invalid name.")

(is-error (variable-environment-add-global 'x :foo 1
           (empty-variable-environment))
          type-error
          "Invalid cl-cuda type.")

(is-error (variable-environment-add-global 'x 'int 1
           :foo)
          type-error
          "Invalid variable environment.")

(is-error (variable-environment-global-exists-p :foo 'x)
          type-error
          "Invalid variable environment.")

(is-error (variable-environment-global-exists-p (empty-variable-environment) 1)
          type-error
          "Invalid name.")

(is-error (variable-environment-global-name :foo 'x)
          type-error
          "Invalid variable environment.")

(is-error (variable-environment-global-name (empty-variable-environment) 1)
          type-error
          "Invalid name.")

(is-error (variable-environment-global-c-name :foo 'x)
          type-error
          "Invalid variable environment.")

(is-error (variable-environment-global-c-name (empty-variable-environment) 1)
          type-error
          "Invalid name.")

(is-error (variable-environment-global-type :foo 'x)
          type-error
          "Invalid variable environment.")

(is-error (variable-environment-global-type (empty-variable-environment) 1)
          type-error
          "Invalid name.")

(is-error (variable-environment-global-initializer :foo 'x)
          type-error
          "Invalid variable environment.")

(is-error (variable-environment-global-initializer (empty-variable-environment)
                                                   1)
          type-error
          "Invalid name.")


;;;
;;; test Function environment - Function
;;;

(diag "Function environment - Function")

(let ((func-env (function-environment-add-macro 'bar '(x) '(`(return ,x))
                  (function-environment-add-function 'foo 'int '(int)
                    (empty-function-environment)))))
  (is (function-environment-function-exists-p func-env 'foo) t
      "basic case 1")
  (is (function-environment-function-exists-p func-env 'bar) nil
      "basic case 2")
  (is (function-environment-function-exists-p func-env 'baz) nil
      "basic case 3")
  (is (function-environment-function-name func-env 'foo) 'foo
      "basic case 4")
  (is (function-environment-function-c-name func-env 'foo) "cl_cuda_test_lang_environment_foo"
      "basic case 5")
  (is (function-environment-function-return-type func-env 'foo) 'int
      "basic case 6")
  (is (function-environment-function-argument-types func-env 'foo) '(int)
      "basic case 7"))


;;;
;;; test Function environment - Macro
;;;

(diag "Function environment - Macro")

(let ((func-env (function-environment-add-macro 'bar '(x) '(`(return ,x))
                  (function-environment-add-function 'foo 'int '(int)
                    (empty-function-environment)))))
  (is (function-environment-macro-exists-p func-env 'foo) nil
      "basic case 1")
  (is (function-environment-macro-exists-p func-env 'bar) t
      "basic case 2")
  (is (function-environment-macro-exists-p func-env 'baz) nil
      "basic case 3")
  (is (function-environment-macro-name func-env 'bar) 'bar
      "basic case 4")
  (is (funcall (function-environment-macro-expander func-env 'bar) '(1))
      '(return 1)
      "basic case 5"))


(finalize)
