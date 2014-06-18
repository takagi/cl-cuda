#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-test.lang.util
  (:use :cl :cl-test-more
        :cl-cuda.lang.util))
(in-package :cl-cuda-test.lang.util)

(plan nil)


;;;
;;; test C-IDENTIFIER function
;;;

(diag "C-IDENTIFIER")

(is (c-identifier 'x) "x"
    "basic case 1")
(is (c-identifier 'vec-add-kernel) "vec_add_kernel"
    "basic case 2")
(is (c-identifier 'vec.add.kernel) "vec_add_kernel"
    "basic case 3")
(is (c-identifier '%vec-add-kernel) "_vec_add_kernel"
    "basic case 4")
(is (c-identifier 'VecAdd_kernel) "vecadd_kernel"
    "basic case 5")
(is (c-identifier 'foo t) "cl_cuda_test_lang_util_foo"
    "basic case 6")


;;;
;;; test LINES function
;;;

(diag "LINES")

(is (lines (format nil "1~%2~%3~%")) '("1" "2" "3")
    "basic case 1")

(is (lines (format nil "1~%2~%3")) '("1" "2" "3")
    "basic case 2")


;;;
;;; test UNLINES function
;;;

(diag "UNLINES")

(is (unlines "1" "2" "3") "1
2
3
" "basic case 1")


;;;
;;; test INDENT function
;;;

(diag "INDENT")

(is (indent 2 (format nil "1~%2~%3~%")) "  1
  2
  3
" "basic case 1")

(is (indent 2 (format nil "1~%2~%3")) "  1
  2
  3
" "basic case 2")


(finalize)
