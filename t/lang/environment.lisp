#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-test.lang.environment
  (:use :cl :cl-test-more
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
;;; test Function environment - Function
;;;

(diag "Function environment - Function")

(let ((func-env (function-environment-add-macro 'bar #'(lambda (x)
                                                         `(return ,x))
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
  (is (function-environment-function-return-type func-env 'foo) 'int
      "basic case 5")
  (is (function-environment-function-argument-types func-env 'foo) '(int)
      "basic case 6"))


;;;
;;; test Function environment - Macro
;;;

(diag "Function environment - Macro")

(let ((func-env (function-environment-add-macro 'bar #'(lambda (x)
                                                         `(return ,x))
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
  (is (funcall (function-environment-macro-expander func-env 'bar) 1)
      '(return 1)
      "basic case 5"))


(finalize)
