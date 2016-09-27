#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-test.lang.built-in
  (:use :cl :prove
        :cl-cuda.lang.type
        :cl-cuda.lang.built-in))
(in-package :cl-cuda-test.lang.built-in)

(plan nil)


;;;
;;; test BUILT-IN-FUNCTION-RETURN-TYPE function
;;;

(diag "BUILT-IN-FUNCTION-RETURN-TYPE")

(is (built-in-function-return-type '+ '(int int)) 'int
    "basic case 1")

(is (built-in-function-return-type '+ '(float3 float3)) 'float3
    "basic case 2")

(is (built-in-function-return-type '- '(int int)) 'int
    "basic case 3")

(is (built-in-function-return-type 'mod '(int int)) 'int
    "basic case 4")

;;;
;;; test BUILT-IN-FUNCTION-INFIX-P function
;;;

(diag "BUILT-IN-FUNCTION-INFIX-P")

(is (built-in-function-infix-p '+ '(int int)) t
    "basic case 1")

(is (built-in-function-infix-p '+ '(float3 float3)) nil
    "basic case 2")

(is (built-in-function-infix-p '- '(int int)) t
    "basic case 3")

(is (built-in-function-infix-p 'mod '(int int)) t
    "basic case 4")

;;;
;;; test BUILT-IN-FUNCTION-C-NAME function
;;;

(diag "BUILT-IN-FUNCTION-C-NAME")

(is (built-in-function-c-name '+ '(int int)) "+"
    "basic case 1")

(is (built-in-function-c-name '+ '(float3 float3)) "float3_add"
    "basic case 2")

(is (built-in-function-c-name '- '(int int)) "-"
    "basic case 3")

(is (built-in-function-c-name 'mod '(int int)) "%"
    "basic case 4")


(finalize)
