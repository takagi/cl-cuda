#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-test.lang.compiler.compile-expression
  (:use :cl :cl-test-more
        :cl-cuda.lang.syntax
        :cl-cuda.lang.data
        :cl-cuda.lang.type
        :cl-cuda.lang.built-in
        :cl-cuda.lang.environment
        :cl-cuda.lang.compiler.compile-expression)
  (:import-from :cl-cuda.lang.compiler.compile-expression
                :compile-macro
                :compile-symbol-macro
                :compile-literal
                :compile-cuda-dimension
                :compile-reference
                :compile-inline-if
                :compile-arithmetic
                :compile-function))
(in-package :cl-cuda-test.lang.compiler.compile-expression)

(plan nil)


;;;
;;; test COMPILE-EXPRESSION function
;;;

(diag "COMPILE-EXPRESSION")

(let ((var-env (empty-variable-environment))
      (func-env (empty-function-environment)))
  (is (compile-expression 1 var-env func-env) "1"))


;;;
;;; test COMPILE-MACRO function
;;;

(diag "COMPILE-MACRO")

(let ((var-env (empty-variable-environment))
      (func-env (function-environment-add-macro 'foo '(x) '(`(+ ,x ,x))
                  (empty-function-environment))))
  (is (compile-macro '(foo 1) var-env func-env) "(1 + 1)"
      "basic case 1"))


;;;
;;; test COMPILE-SYMBOL-MACRO function
;;;

(diag "COMPILE-SYMBOL-MACRO")

(let ((var-env (variable-environment-add-symbol-macro 'x 1
                 (empty-variable-environment)))
      (func-env (empty-function-environment)))
  (is (compile-symbol-macro 'x var-env func-env) "1"
      "basic case 1"))


;;;
;;; test COMPILE-LITERAL function
;;;

(diag "COMPILE-LITERAL")

(is (compile-literal t) "true"
    "basic case 1")

(is (compile-literal nil) "false"
    "basic case 2")

(is (compile-literal 1) "1"
    "basic case 3")

(is (compile-literal 1.0) "1.0"
    "basic case 4")

(is (compile-literal 1.0d0) "(double)1.0"
    "basic case 5")


;;;
;;; test COMPILE-CUDA-DIMENSION function
;;;

(diag "COMPILE-CUDA-DIMENSION")

(is (compile-cuda-dimension 'grid-dim-x) "gridDim.x"
    "basic case 1")


;;;
;;; test COMPILE-REFERENCE funcion
;;;

(diag "COMPILE-REFERENCE - VARIABLE")

(let ((var-env (variable-environment-add-variable 'y-expansion 'float
                 (variable-environment-add-symbol-macro 'y 'y-expansion
                   (variable-environment-add-variable 'x 'int
                     (empty-variable-environment)))))
      (func-env (empty-function-environment)))
  (is (compile-reference 'x var-env func-env) "x"
      "basic case 1")
  (is-error (compile-reference 'y var-env func-env) simple-error
            "FORM which is a variable not found.")
  (is-error (compile-reference 'a var-env func-env) simple-error
            "FORM which is a variable not found."))

(diag "COMPILE-REFERENCE - STRUCTURE")

(let ((var-env (variable-environment-add-variable 'x 'float3
                 (empty-variable-environment)))
      (func-env (empty-function-environment)))
  (is (compile-reference '(float3-x x) var-env func-env) "x.x"
      "basic case 1")
  (is (compile-reference '(float3-y x) var-env func-env) "x.y"
      "basic case 2")
  (is-error (compile-reference '(float4-x x) var-env func-env)
            simple-error))


(diag "COMPILE-REFERENCE - ARRAY")

(let ((var-env (variable-environment-add-variable 'i 'int
                 (variable-environment-add-variable 'x 'int*
                   (empty-variable-environment))))
      (func-env (empty-function-environment)))
  (is (compile-reference '(aref x i) var-env func-env) "x[i]"
      "basic case 1"))


;;;
;;; test COMPILE-INLINE-IF function
;;;

(diag "COMPILE-INLINE-IF")

(let ((var-env (empty-variable-environment))
      (func-env (empty-function-environment)))
  (is (compile-inline-if '(if (= 1 1) 1 2) var-env func-env)
      "((1 == 1) ? 1 : 2)"
      "basic case 1"))


;;;
;;; test COMPILE-ARITHMETIC function
;;;

(diag "COMPILE-ARITHMETIC")

(let ((var-env (empty-variable-environment))
      (func-env (empty-function-environment)))
  (is (compile-arithmetic '(+ 1 1 1) var-env func-env) "(1 + (1 + 1))"
      "basic case 1"))


;;;
;;; test COMPILE-FUNCTION function
;;;

(diag "COMPILE-FUNCTION")

(let ((var-env (empty-variable-environment))
      (func-env (function-environment-add-function 'foo 'int '(int int)
                  (empty-function-environment))))
  (is (compile-function '(foo 1 1) var-env func-env)
      "cl_cuda_test_lang_compiler_compile_expression_foo( 1, 1 )"
      "basic case 1")
  (is-error (compile-function '(foo 1 1 1) var-env func-env) simple-error))

(let ((var-env (empty-variable-environment))
      (func-env (empty-function-environment)))
  (is (compile-function '(+ 1 1) var-env func-env) "(1 + 1)"
      "basic case 2"))

(let ((var-env (empty-variable-environment))
      (func-env (empty-function-environment)))
  (is (compile-function '(- 1) var-env func-env) "int_negate( 1 )"
      "basic case 3")
  (is (compile-function '(+ (float3 1.0 1.0 1.0) (float3 2.0 2.0 2.0))
                        var-env func-env)
      "float3_add( make_float3( 1.0, 1.0, 1.0 ), make_float3( 2.0, 2.0, 2.0 ) )"
      "basic case 4"))

(let ((var-env (empty-variable-environment))
      (func-env (empty-function-environment)))
  (is (compile-function '(syncthreads) var-env func-env) "__syncthreads()"
      "basic case 1"))


(finalize)
