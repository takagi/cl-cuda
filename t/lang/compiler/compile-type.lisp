#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-test.lang.compiler.compile-type
  (:use :cl :cl-test-more
        :cl-cuda.lang.type
        :cl-cuda.lang.compiler.compile-type))
(in-package :cl-cuda-test.lang.compiler.compile-type)

(plan nil)


;;;
;;; test COMPILE-TYPE function
;;;

(diag "COMPILE-TYPE")

(is (compile-type 'int) "int"
    "basic case 1")



(finalize)
