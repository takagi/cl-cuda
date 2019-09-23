#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-test.lang.compiler.compile-statement
  (:use :cl :prove
        :cl-cuda.lang.util
        :cl-cuda.lang.data
        :cl-cuda.lang.type
        :cl-cuda.lang.syntax
        :cl-cuda.lang.environment
        :cl-cuda.lang.compiler.compile-statement)
  (:import-from :cl-cuda.lang.compiler.compile-statement
                :compile-macro
                :compile-if
                :compile-let
                :compile-symbol-macrolet
                :compile-macrolet
                :compile-do
                :compile-with-shared-memory
                :compile-set
                :compile-progn
                :compile-return
                :compile-function))
(in-package :cl-cuda-test.lang.compiler.compile-statement)

(plan nil)


;;;
;;; test COMPILE-STATEMENT function (not implemented)
;;;


;;;
;;; test COMPILE-MACRO function (not implemented)
;;;


;;;
;;; test COMPILE-IF funciton
;;;

(diag "COMPILE-IF")

(let ((var-env (empty-variable-environment))
      (func-env (empty-function-environment)))
  (let ((lisp-code '(if t (return) (return)))
        (c-code (unlines "if (true) {"
                         "  return;"
                         "} else {"
                         "  return;"
                         "}")))
    (is (compile-if lisp-code var-env func-env) c-code
        "basic case 1")))

(let ((var-env (empty-variable-environment))
      (func-env (empty-function-environment)))
  (let ((lisp-code '(if t (return 0)))
        (c-code (unlines "if (true) {"
                         "  return 0;"
                         "}")))
    (is (compile-if lisp-code var-env func-env) c-code
        "basic case 2")))


(let ((var-env (empty-variable-environment))
      (func-env (empty-function-environment)))
  (let ((lisp-code '(if 1 (return))))
    (is-error (compile-if lisp-code var-env func-env) simple-error)))


;;;
;;; test COMPILE-LET function
;;;

(diag "COMPILE-LET")

(let ((var-env (empty-variable-environment))
      (func-env (empty-function-environment)))
  (let ((lisp-code '(let ((i 0))
                      (return)))
        (c-code (unlines "{"
                         "  int i = 0;"
                         "  return;"
                         "}")))
    (is (compile-let lisp-code var-env func-env) c-code
        "basic case 1")))

(let ((var-env (empty-variable-environment))
      (func-env (empty-function-environment)))
  (is-error (compile-let '(let (i) (return)) var-env func-env)
            simple-error)
  (is-error (compile-let '(let ((i)) (return)) var-env func-env)
            simple-error)
  (is-error (compile-let '(let ((x 1) (y x)) (return y)) var-env func-env)
            simple-error))


;;;
;;; test COMPILE-SYMBOL-MACROLET function
;;;

(diag "COMPILE-SYMBOL-MACROLET")

(let ((var-env (empty-variable-environment))
      (func-env (empty-function-environment)))
  (let ((lisp-code '(symbol-macrolet ((x 1))
                      (return x)))
        (c-code (unlines "{"
                         "  return 1;"
                         "}")))
    (is (compile-symbol-macrolet lisp-code var-env func-env) c-code
        "basic case 1")))


;;;
;;; test COMPILE-MACROLET function
;;;

(diag "COMPILE-MACROLET")

(let ((var-env (empty-variable-environment))
      (func-env (empty-function-environment)))
  (let ((lisp-code '(macrolet ((square (x) `(* ,x ,x)))
                     (return (square 2))))
        (c-code (unlines "{"
                         "  return (2 * 2);"
                         "}")))
    (is (compile-macrolet lisp-code var-env func-env) c-code
        "basic case 1"))
  (let ((lisp-code '(macrolet ((optimized-square (x) (* x x)))
                     (return (optimized-square 2))))
        (c-code (unlines "{"
                         "  return 4;"
                         "}")))
    (is (compile-macrolet lisp-code var-env func-env) c-code
        "basic case 2")))


;;;
;;; test COMPILE-DO function
;;;

(diag "COMPILE-DO")

(let ((var-env (empty-variable-environment))
      (func-env (empty-function-environment)))
  (let ((lisp-code '(do ((a 0 (+ a 1))
                         (b 0 (+ b 1)))
                        ((> a 15))
                      (return)))
        (c-code (unlines "for ( int a = 0, int b = 0; ! (a > 15); a = (a + 1), b = (b + 1) )"
                         "{"
                         "  return;"
                         "}")))
    (is (compile-do lisp-code var-env func-env) c-code
        "basic case 1")))


;;;
;;; test COMPILE-WITH-SHARED-MEMORY function
;;;

(diag "COMPILE-WITH-SHARED-MEMORY")

(let ((var-env (empty-variable-environment))
      (func-env (empty-function-environment)))
  (let ((lisp-code '(with-shared-memory ((a int 16)
                                         (b float 16 16))
                      (return)))
        (c-code (unlines "{"
                         "  __shared__ int a[16];"
                         "  __shared__ float b[16][16];"
                         "  return;"
                         "}")))
    (is (compile-with-shared-memory lisp-code var-env func-env) c-code
        "basic case 1")))

(let ((var-env (empty-variable-environment))
      (func-env (empty-function-environment)))
  (let ((lisp-code '(with-shared-memory () (return)))
        (c-code (unlines "{"
                         "  return;"
                         "}")))
    (is (compile-with-shared-memory lisp-code var-env func-env) c-code
        "basic case 2")))

(let ((var-env (empty-variable-environment))
      (func-env (empty-function-environment)))
  (let ((lisp-code '(with-shared-memory ()))
        (c-code (unlines "{"
                         "}")))
    (is (compile-with-shared-memory lisp-code var-env func-env) c-code
        "basic case 3")))

(let ((var-env (empty-variable-environment))
      (func-env (empty-function-environment)))
  (let ((lisp-code '(with-shared-memory ((a float))
                      (return a)))
        (c-code (unlines "{"
                         "  __shared__ float a;"
                         "  return a;"
                         "}")))
    (is (compile-with-shared-memory lisp-code var-env func-env) c-code
        "basic case 4")))

(let ((var-env (empty-variable-environment))
      (func-env (empty-function-environment)))
  (let ((lisp-code '(with-shared-memory ((a float 16 16))
                      (set (aref a 0 0) 1.0)))
        (c-code (unlines "{"
                         "  __shared__ float a[16][16];"
                         "  a[0][0] = 1.0f;"
                         "}")))
    (is (compile-with-shared-memory lisp-code var-env func-env) c-code
        "basic case 5")))

(let ((var-env (empty-variable-environment))
      (func-env (empty-function-environment)))
  (let ((lisp-code '(with-shared-memory ((a float (+ 16 2)))
                      (set (aref a 0) 1.0)))
        (c-code (unlines "{"
                         "  __shared__ float a[(16 + 2)];"
                         "  a[0] = 1.0f;"
                         "}")))
    (is (compile-with-shared-memory lisp-code var-env func-env) c-code
        "basic case 6")))

(let ((var-env (empty-variable-environment))
      (func-env (empty-function-environment)))
  (let ((lisp-code '(with-shared-memory (a float)
                      (return))))
    (is-error (compile-with-shared-memory lisp-code var-env func-env)
              simple-error)))

(let ((var-env (empty-variable-environment))
      (func-env (empty-function-environment)))
  (let ((lisp-code '(with-shared-memory ((a float 16 16))
                      (set (aref a 0) 1.0))))
    (is-error (compile-with-shared-memory lisp-code var-env func-env)
              simple-error)))


;;;
;;; test COMPILE-SET function
;;;

(diag "COMPILE-SET")

(let ((var-env (variable-environment-add-variable 'x 'int
                 (empty-variable-environment)))
      (func-env (empty-function-environment)))
  (is (compile-set '(set x 1) var-env func-env)
      (unlines "x = 1;")
      "basic case 1")
  (is-error (compile-set '(set x 1.0) var-env func-env) simple-error))

(let ((var-env (variable-environment-add-variable 'x 'int*
                 (empty-variable-environment)))
      (func-env (empty-function-environment)))
  (is (compile-set '(set (aref x 0) 1) var-env func-env)
      (unlines "x[0] = 1;")
      "basic case 2")
  (is-error (compile-set '(set (aref x 0) 1.0) var-env func-env)
            simple-error))

(let ((var-env (variable-environment-add-variable 'x 'float3
                 (empty-variable-environment)))
      (func-env (empty-function-environment)))
  (is (compile-set '(set (float3-x x) 1.0) var-env func-env)
      (unlines "x.x = 1.0f;")
      "basic case 3")
  (is-error (compile-set '(set (float3-x x) 1) var-env func-env)
            simple-error))


;;;
;;; test COMPILE-PROGN function (not implemented)
;;;


;;;
;;; test COMPILE-RETURN function (not implemented)
;;;


;;;
;;; test COMPILE-FUNCTION function (not implemented)
;;;




(finalize)
