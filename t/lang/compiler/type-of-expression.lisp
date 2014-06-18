#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-test.lang.compiler.type-of-expression
  (:use :cl :cl-test-more
        :cl-cuda.lang.compiler.type-of-expression
        :cl-cuda.lang.data
        :cl-cuda.lang.type
        :cl-cuda.lang.syntax
        :cl-cuda.lang.environment)
  (:import-from :cl-cuda.lang.compiler.type-of-expression
                :type-of-macro
                :type-of-symbol-macro
                :type-of-literal
                :type-of-cuda-dimension
                :type-of-reference
                :type-of-inline-if
                :type-of-arithmetic
                :type-of-function))
(in-package :cl-cuda-test.lang.compiler.type-of-expression)

(plan nil)


;;;
;;; test TYPE-OF-EXPRESSION function
;;;

(diag "TYPE-OF-EXPRESSION")

(let ((var-env (empty-variable-environment))
      (func-env (empty-function-environment)))
  (is (type-of-expression 1 nil nil) 'int))


;;;
;;; test TYPE-OF-MACRO function (not implemented)
;;;


;;;
;;; test TYPE-OF-SYMBOL-MACRO function (not implemented)
;;;


;;;
;;; test TYPE-OF-LITERAL function
;;;

(diag "TYPE-OF-LITERAL")

(is (type-of-literal t) 'bool
    "basic case 1")

(is (type-of-literal nil) 'bool
    "basic case 2")

(is (type-of-literal 1) 'int
    "basic case 3")

(is (type-of-literal 1.0) 'float
    "basic case 4")

(is (type-of-literal 1.0d0) 'double
    "basic case 5")


;;;
;;; test TYPE-OF-CUDA-DIMENSION function
;;;

(diag "TYPE-OF-CUDA-DIMENSION")

(is (type-of-cuda-dimension 'grid-dim) 'int
    "basic case 1")


;;;
;;; test TYPE-OF-REFERENCE function
;;;

(diag "TYPE-OF-REFERENCE - VARIABLE")

(let ((var-env (variable-environment-add-variable 'y-expansion 'float
                 (variable-environment-add-symbol-macro 'y 'y-expansion
                   (variable-environment-add-variable 'x 'int
                     (empty-variable-environment)))))
      (func-env (empty-function-environment)))
  (is (type-of-reference 'x var-env func-env) 'int
      "basic case 1")
  (is-error (type-of-reference 'y var-env func-env) simple-error
            "FORM which is a variable not found.")
  (is-error (type-of-reference 'a var-env func-env) simple-error
            "FORM which is a variable not found."))


(diag "TYPE-OF-REFERENCE - STRUCTURE")

(let ((var-env (variable-environment-add-variable 'x 'float3
                 (empty-variable-environment)))
      (func-env (empty-function-environment)))
  (is (type-of-reference '(float3-x x) var-env func-env) 'float)
  (is (type-of-reference '(float3-y x) var-env func-env) 'float)
  (is-error (type-of-reference '(float4-x x) var-env func-env)
            simple-error))


(diag "TYPE-OF-REFERENCE - ARRAY")

(let ((var-env (variable-environment-add-variable 'x 'int
                 (empty-variable-environment)))
      (func-env (empty-function-environment)))
  (is-error (type-of-reference '(aref x) var-env func-env) simple-error))

(let ((var-env (variable-environment-add-variable 'x 'int*
                 (empty-variable-environment)))
      (func-env (empty-function-environment)))
  (is (type-of-reference '(aref x 0) var-env func-env) 'int)
  (is-error (type-of-reference '(aref x 0 0) var-env func-env)
            simple-error))

(let ((var-env (variable-environment-add-variable 'x 'int**
                 (empty-variable-environment)))
      (func-env (empty-function-environment)))
  (is-error (type-of-reference '(aref x 0) var-env func-env) simple-error)
  (is (type-of-reference '(aref x 0 0) var-env func-env) 'int))


;;;
;;; test TYPE-OF-INLINE-IF function
;;;

(diag "TYPE-OF-INLINE-IF")

(let ((var-env (empty-variable-environment))
      (func-env (empty-function-environment)))
  (is-error (type-of-inline-if '(if) var-env func-env)
            simple-error)
  (is-error (type-of-inline-if '(if (= 1 1)) var-env func-env)
            simple-error)
  (is-error (type-of-inline-if '(if (= 1 1) 1) var-env func-env)
            simple-error)
  (is (type-of-inline-if '(if (= 1 1) 1 2) var-env func-env)
      'int)
  (is-error (type-of-inline-if '(if 1 2 3) var-env func-env)
            simple-error)
  (is-error (type-of-inline-if '(if (= 1 1) 1 2.0) var-env func-env)
            simple-error))


;;;
;;; test TYPE-OF-ARITHMETIC function (not implemented)
;;;


;;;
;;; test TYPE-OF-FUNCTION function
;;;

(diag "TYPE-OF-FUNCTION")

(let ((var-env (empty-variable-environment))
      (func-env (function-environment-add-function 'foo 'int '(int int)
                  (empty-function-environment))))
  (is (type-of-function '(+ 1 1) var-env func-env) 'int)
  (is (type-of-function '(foo 1 1) var-env func-env) 'int)
  (is (type-of-function '(+ 1.0 1.0) var-env func-env) 'float)
  (is-error (type-of-function '(+ 1 1.0) var-env func-env) simple-error)
  (is (type-of-function '(expt 1.0 1.0) var-env func-env) 'float))


(finalize)
