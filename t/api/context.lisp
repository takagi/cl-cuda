#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-test.api.context
  (:use :cl :prove
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

(subtest "append-arch"

  (let ((dev-id 0))
    (cl-cuda.driver-api:cu-init 0)
    (is (cl-cuda.api.context::append-arch '("foo") dev-id)
        (list (cl-cuda.api.context::get-nvcc-arch dev-id)
              "foo")))

  (let ((dev-id 0))
    (is-error (cl-cuda.api.context::append-arch :foo dev-id)
              type-error
              "Invalid options."))

  (is-error (cl-cuda.api.context::append-arch nil :foo)
            type-error
            "Invalid device ID."))
