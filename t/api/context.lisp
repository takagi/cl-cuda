#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-test.api.context
  (:use :cl :cl-test-more
        :cl-cuda.api.context))
(in-package :cl-cuda-test.api.context)

(plan nil)


;;
;; WITH-CUDA macro

(subtest "arch-exists-p"

  (is (cl-cuda.api.context::arch-exists-p '("-arch=sm_11"))
      t)

  (is (cl-cuda.api.context::arch-exists-p '())
      nil)

  (is-error (cl-cuda.api.context::arch-exists-p :foo)
            type-error
            "Invalid options."))
